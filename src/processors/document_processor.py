import os
from typing import Dict, List, Optional
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, Text, Title, CompositeElement
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_parse import LlamaParse
import logging
import uuid
import os
from PIL import Image as PILImage
import io
import markdownify
import pytesseract
from pydantic import BaseModel, Field, ValidationError
from typing import Any


class ChunkModel(BaseModel):
    chunk_id: str
    doc_id: str
    modality: str
    content: str
    type: str
    page_number: int
    section_path: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    ocr_text: str = ""
    summary: str = ""
    prev_chunk: Any = None
    next_chunk: Any = None


def make_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable types.
    Converts frozenset/set to list, objects with __dict__ to dict, and all non-serializable types to string.
    """
    import json
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, frozenset) or isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConsultingReportProcessor:
    def __init__(
        self,
        model: AzureChatOpenAI = None,
        extract_images: bool = False,
        batch_size: int = 5,
    ):
        self.model = model
        self.extract_images = extract_images
        self.batch_size = batch_size
        self.llama_parser = LlamaParse(
            api_key=os.getenv("LLAMA_API_KEY"), result_type="json"
        )

    def _process_text_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Process a batch of texts to generate summaries"""
        if not self.model or not texts:
            return [None] * len(texts)

        try:
            prompt = ChatPromptTemplate.from_template(
                "Summarize the key points from this text in 2-3 sentences. Focus on the main ideas and insights:\n\n{text}"
            )
            chain = prompt | self.model | StrOutputParser()

            # Process texts in batches
            summaries = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_summaries = []
                for text in batch:
                    try:
                        summary = chain.invoke({"text": text}) if text.strip() else None
                        batch_summaries.append(summary)
                    except Exception as e:
                        logger.error(f"Error generating summary: {e}")
                        batch_summaries.append(None)
                summaries.extend(batch_summaries)
            return summaries
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [None] * len(texts)

    def _generate_summary(self, text: str) -> Optional[str]:
        """Generate a summary for a single text"""
        return self._process_text_batch([text])[0] if text else None

    def _safe_summary(self, text: str) -> str:
        """Guarantee that summary is always a string, never None."""
        summary = self._generate_summary(text)
        return summary if isinstance(summary, str) and summary else ""

    def process_report(self, pdf_path: str) -> Dict:
        """Process a single PDF report"""
        try:
            logger.info(f"Processing {pdf_path}")

            # ---
            # Multimodal Extraction Approach:
            # 1. Use unstructured.partition.pdf (Unstructured) to extract all elements (text, images, tables, etc.) from the PDF.
            # 2. Use llama_parse to analyze the document and build a hierarchy mapping (section paths, page structure).
            # 3. For each element from Unstructured, assign hierarchy/context using the llama_parse mapping.
            # This enables robust, context-aware chunking for RAG, with images/tables linked to their section.
            # ---
            logger.info("Extracting content with unstructured...")
            elements = partition_pdf(
                pdf_path,
                strategy="hi_res",
                extract_images_in_pdf=self.extract_images,
                extract_image_block_types=["Image", "Table"],
                pdf_image_dpi=150,
            )

            logger.info("Getting document structure with llama_parse...")
            parsed_doc = self.llama_parser.load_data(pdf_path)
            logger.info(f"DEBUG: llama_parse output type={type(parsed_doc)}, repr={repr(parsed_doc)[:500]}")
            # Build a mapping from page number to hierarchical path using llama_parse
            page_hierarchy = {}
            if isinstance(parsed_doc, list):
                for doc_obj in parsed_doc:
                    # Defensive: Some Document-like objects may not have metadata as a dict
                    metadata = getattr(doc_obj, "metadata", {})
                    # If metadata is not a dict, try to convert
                    if not isinstance(metadata, dict) and hasattr(metadata, '__dict__'):
                        metadata = dict(metadata.__dict__)
                    page = metadata.get("page_number") or metadata.get("page")
                    section_path = metadata.get("section_path") or metadata.get("path") or []
                    if page is not None:
                        page_hierarchy[page] = section_path
            else:
                logger.error(f"Unexpected type from llama_parse: {type(parsed_doc)}")

            # Initialize container for all output chunks
            chunks = []

            # Process all elements
            logger.info("Processing elements...")
            doc_id = str(uuid.uuid4())  # One doc_id per processed document
            prev_chunk_id = None
            multimodal_chunks = []
            image_counter = 0
            output_dir = os.getenv('OUTPUT_DIR', 'processed_outputs')
            # Remove creation of images_dir in processed_outputs unless actually needed
            images_dir = os.path.join(output_dir, 'images')
            # Only create images_dir if an image is actually saved there (see below)

            for element in elements:
                if element is None:
                    continue
                try:
                    element_type = type(element).__name__
                    logger.debug(f"Element type: {element_type}, metadata: {getattr(element, 'metadata', None)}")
                    page_number = getattr(element.metadata, "page_number", 1)
                    section_path = page_hierarchy.get(page_number, [])
                    chunk_id = str(uuid.uuid4())

                    # MULTIMODAL CHUNK LOGIC
                    # ---
                    # 1. If Image, always add as image chunk (if image_path or image_base64 exists)
                    # 2. If Table or CompositeElement with HTML, always add as table chunk
                    # 3. If Text, add as text chunk if it has meaningful content
                    # ---
                    # IMAGE CHUNKS
                    if (hasattr(element, 'metadata') and (
                        (isinstance(element, Image) and (getattr(element.metadata, 'image_path', None) or 'image_base64' in getattr(element.metadata, '__dict__', {})))
                    )):
                        try:
                            os.makedirs(images_dir, exist_ok=True)
                            image_counter += 1
                            # Prefer image_path if present, else decode image_base64
                            rel_image_path = None
                            if getattr(element.metadata, 'image_path', None):
                                abs_image_path = element.metadata.image_path
                                rel_image_path = os.path.relpath(abs_image_path, start=os.getcwd())
                            elif 'image_base64' in getattr(element.metadata, '__dict__', {}):
                                image_b64 = element.metadata.image_base64
                                img_data = base64.b64decode(image_b64)
                                img_pil = PILImage.open(io.BytesIO(img_data))
                                image_filename = f"{doc_id}_page_{page_number}_img_{image_counter}.png"
                                abs_image_path = os.path.join(images_dir, image_filename)
                                img_pil.save(abs_image_path)
                                rel_image_path = os.path.relpath(abs_image_path, start=os.getcwd())
                            # OCR extraction using pytesseract
                            ocr_text = ""
                            try:
                                img_path = os.path.join(os.getcwd(), rel_image_path)
                                img = PILImage.open(img_path)
                                ocr_text = pytesseract.image_to_string(img)
                            except Exception as e:
                                logger.warning(f"OCR failed for image {rel_image_path}: {e}")
                                ocr_text = ""
                            # Clean metadata, remove _known_field_names if present
                            metadata = make_json_serializable(element.metadata.__dict__) if hasattr(element.metadata, '__dict__') else {}
                            if '_known_field_names' in metadata:
                                metadata.pop('_known_field_names', None)
                            # Build chunk dict and validate with Pydantic
                            chunk_dict = {
                                "chunk_id": str(uuid.uuid4()),
                                "doc_id": doc_id,
                                "modality": "image",
                                "content": rel_image_path or "",
                                "type": "image",
                                "page_number": page_number,
                                "section_path": section_path,
                                "metadata": metadata,
                                "ocr_text": ocr_text or "",
                                "summary": self._safe_summary(ocr_text) if ocr_text else "",
                                "prev_chunk": prev_chunk_id,
                            }
                            try:
                                chunk = ChunkModel(**chunk_dict).model_dump()
                            except ValidationError as ve:
                                logger.error(f"Chunk validation error: {ve}")
                                continue
                            logger.info(f"Extracted image chunk: {chunk['chunk_id']} saved at {rel_image_path} (page {page_number}, section {section_path})")
                            multimodal_chunks.append(chunk)
                            prev_chunk_id = chunk["chunk_id"]
                        except Exception as e:
                            logger.error(f"Error saving image chunk: {e}")
                            continue

                    # TABLE CHUNKS
                    elif (isinstance(element, CompositeElement) and hasattr(element, 'metadata') and getattr(element.metadata, 'text_as_html', None)):
                        html_table = getattr(element.metadata, 'text_as_html', None)
                        if html_table:
                            table_md = markdownify.markdownify(html_table, heading_style="ATX")
                            table_chunk_id = str(uuid.uuid4())
                            metadata = make_json_serializable(element.metadata.__dict__) if hasattr(element.metadata, '__dict__') else {}
                            if '_known_field_names' in metadata:
                                metadata.pop('_known_field_names', None)
                            chunk_dict = {
                                "chunk_id": table_chunk_id,
                                "doc_id": doc_id,
                                "modality": "table",
                                "content": table_md.strip(),
                                "type": "table",
                                "page_number": page_number,
                                "section_path": section_path,
                                "metadata": metadata,
                                "ocr_text": "",
                                "summary": self._safe_summary(table_md),
                                "prev_chunk": prev_chunk_id,
                            }
                            try:
                                chunk = ChunkModel(**chunk_dict).model_dump()
                            except ValidationError as ve:
                                logger.error(f"Chunk validation error: {ve}")
                                continue
                            logger.info(f"Extracted table chunk: {chunk['chunk_id']} on page {page_number} in section {section_path}")
                            multimodal_chunks.append(chunk)
                            prev_chunk_id = table_chunk_id

                    # TEXT CHUNKS
                    elif isinstance(element, Text):
                        try:
                            element_text = str(element) if element else ""
                        except Exception as e:
                            logger.error(f"Error converting element to string: {e}")
                            element_text = ""
                        if not element_text or len(element_text.strip()) < 15 or not any(c.isalnum() for c in element_text):
                            continue
                        metadata = make_json_serializable(element.metadata.__dict__) if hasattr(element.metadata, '__dict__') else {}
                        if '_known_field_names' in metadata:
                            metadata.pop('_known_field_names', None)
                        chunk_dict = {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "modality": "text",
                            "content": element_text.strip(),
                            "type": "text",
                            "page_number": page_number,
                            "section_path": section_path,
                            "metadata": metadata,
                            "ocr_text": "",
                            "summary": self._safe_summary(element_text),
                            "prev_chunk": prev_chunk_id,
                        }
                        try:
                            chunk = ChunkModel(**chunk_dict).model_dump()
                        except ValidationError as ve:
                            logger.error(f"Chunk validation error: {ve}")
                            continue
                        multimodal_chunks.append(chunk)
                        prev_chunk_id = chunk_id

                    # FALLBACK: If element is not recognized, log and skip
                    else:
                        logger.warning(f"Skipping unrecognized element type: {element_type} on page {page_number}")
                        continue

                except Exception as e:
                    logger.error(f"Error processing multimodal element: {e}")
                    continue

            # Set prev_chunk and next_chunk using chunk IDs only (robust for navigation)
            for idx, chunk in enumerate(multimodal_chunks):
                chunk["prev_chunk"] = multimodal_chunks[idx-1]["chunk_id"] if idx > 0 else None
                chunk["next_chunk"] = multimodal_chunks[idx+1]["chunk_id"] if idx < len(multimodal_chunks) - 1 else None

            chunks.extend(multimodal_chunks)

            logger.info(
                f"Processed document: {len(chunks)} multimodal chunks."
            )
            return {
                "chunks": chunks,
                "document_metadata": {
                    "filename": os.path.basename(pdf_path),
                    "path": pdf_path,
                    "total_chunks": len(chunks),
                },
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
