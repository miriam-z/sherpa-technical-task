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
                filename=pdf_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                include_metadata=True,
                languages=["eng"],
                pdf_image_dpi=150,
            )

            logger.info("Getting document structure with llama_parse...")
            parsed_doc = self.llama_parser.load_data(pdf_path)
            logger.info(f"DEBUG: llama_parse output type={type(parsed_doc)}, repr={repr(parsed_doc)[:500]}")
            # Build a mapping from page number to hierarchical path using llama_parse
            page_hierarchy = {}
            if isinstance(parsed_doc, list):
                for doc_obj in parsed_doc:
                    metadata = getattr(doc_obj, "metadata", {})
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
            images_dir = os.path.join(output_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)

            for element in elements:
                if element is None:
                    continue
                try:
                    element_type = type(element).__name__
                    logger.debug(f"Element type: {element_type}, metadata: {getattr(element, 'metadata', None)}")
                    page_number = getattr(element.metadata, "page_number", 1)
                    section_path = page_hierarchy.get(page_number, [])
                    chunk_id = str(uuid.uuid4())

                    # TEXT CHUNKS
                    if isinstance(element, Text):
                        try:
                            # Robust element-to-string conversion
                            try:
                                element_text = str(element)
                                if not isinstance(element_text, str):
                                    element_text = ""
                            except Exception as e:
                                logger.error(f"Error converting element to string: {e}")
                                element_text = ""
                            element_text = str(element) if element else ""
                        except Exception as e:
                            logger.error(f"Error converting element to string: {e}")
                            element_text = ""
                        if not element_text or len(element_text.strip()) < 15 or not any(c.isalnum() for c in element_text):
                            continue
                        chunk = {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "modality": "text",
                            "content": element_text.strip(),
                            "type": "text",
                            "page_number": page_number,
                            "section_path": section_path,
                            "metadata": {
                                "element_type": element_type,
                            },
                            "summary": self._generate_summary(element_text) or "",
                            "prev_chunk": prev_chunk_id,
                        }
                        multimodal_chunks.append(chunk)
                        prev_chunk_id = chunk_id

                    # TABLE CHUNKS
                    elif isinstance(element, CompositeElement) and hasattr(element, 'metadata') and getattr(element.metadata, 'text_as_html', None):
                        html_table = getattr(element.metadata, 'text_as_html', None)
                        if html_table:
                            table_md = markdownify.markdownify(html_table, heading_style="ATX")
                            if len(table_md.strip()) < 10:
                                continue
                            table_chunk_id = str(uuid.uuid4())
                            chunk = {
                                "chunk_id": table_chunk_id,
                                "doc_id": doc_id,
                                "modality": "table",
                                "content": table_md.strip(),
                                "type": "table",
                                "page_number": page_number,
                                "section_path": section_path,
                                "metadata": {
                                    "element_type": element_type,
                                },
                                "summary": self._generate_summary(table_md) or "",
                                "prev_chunk": prev_chunk_id,
                            }
                            logger.info(f"Extracted table chunk: {chunk['chunk_id']} on page {page_number} in section {section_path}")
                            multimodal_chunks.append(chunk)
                            prev_chunk_id = table_chunk_id

                    # IMAGE CHUNKS
                    elif isinstance(element, Image) and getattr(element, 'image', None) is not None:
                        try:
                            image_counter += 1
                            img_data = element.image
                            img_pil = PILImage.open(io.BytesIO(img_data)) if isinstance(img_data, (bytes, bytearray)) else img_data
                            image_filename = f"image_{doc_id}_{image_counter}.png"
                            image_path = os.path.join(images_dir, image_filename)
                            img_pil.save(image_path)
                            # Optionally run OCR or captioning here
                            ocr_text = getattr(element.metadata, 'ocr_text', None)
                            chunk = {
                                "chunk_id": str(uuid.uuid4()),
                                "doc_id": doc_id,
                                "modality": "image",
                                "content": image_path,
                                "type": "image",
                                "page_number": page_number,
                                "section_path": section_path,
                                "metadata": {
                                    "element_type": element_type,
                                },
                                "ocr_text": ocr_text or "",
                                "summary": self._generate_summary(ocr_text) if ocr_text else "",
                                "prev_chunk": prev_chunk_id,
                            }
                            logger.info(f"Extracted image chunk: {chunk['chunk_id']} saved at {image_path} (page {page_number}, section {section_path})")
                            multimodal_chunks.append(chunk)
                            prev_chunk_id = chunk["chunk_id"]
                        except Exception as e:
                            logger.error(f"Error saving image chunk: {e}")
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
