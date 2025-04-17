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

            # Extract content with unstructured first
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

            # Get document structure from llama_parse
            logger.info("Getting document structure with llama_parse...")
            parsed_doc = self.llama_parser.load_data(pdf_path)
            logger.info(f"DEBUG: llama_parse output type={type(parsed_doc)}, repr={repr(parsed_doc)[:500]}")
            # Build a mapping from page number to hierarchical path using llama_parse
            page_hierarchy = {}
            if isinstance(parsed_doc, list):
                for doc_obj in parsed_doc:
                    logger.info(f"DEBUG: doc_obj attributes: {dir(doc_obj)}")
                    logger.info(f"DEBUG: doc_obj.metadata: {getattr(doc_obj, 'metadata', None)}")
                    logger.info(f"DEBUG: doc_obj.text: {getattr(doc_obj, 'text', None)[:200] if getattr(doc_obj, 'text', None) else None}")
                    # TODO: Extract page and hierarchy info from doc_obj.metadata or doc_obj.text
                    # Example: if doc_obj.metadata and 'pages' in doc_obj.metadata:
                    #     for page in doc_obj.metadata['pages']:
                    #         page_hierarchy[page] = doc_obj.metadata.get('path', [])
            elif isinstance(parsed_doc, dict):
                for section in parsed_doc.get("sections", []):
                    pages = section.get("pages", [])
                    for page in pages:
                        page_hierarchy[page] = section.get("path", [])
            else:
                logger.error(f"Unexpected type from llama_parse: {type(parsed_doc)}")

            # Initialize containers
            chunks = []
            images = []
            tables = []
            sections = []

            # Process all elements
            logger.info("Processing elements...")
            for element in elements:
                if element is None:
                    continue

                try:
                    element_type = type(element).__name__
                    page_number = getattr(element.metadata, "page_number", 1)

                    try:
                        element_text = str(element) if element else ""
                    except Exception as e:
                        logger.error(f"Error converting element to string: {e}")
                        element_text = ""

                    if isinstance(element, Text) and element_text:
                        # Use llama_parse hierarchy if available
                        section_path = page_hierarchy.get(page_number, [])
                        chunk = {
                            "content": element_text,
                            "type": "text",
                            "page_number": page_number,
                            "section_path": section_path,
                            "metadata": {
                                "element_type": element_type,
                                "coordinates": getattr(
                                    element.metadata, "coordinates", None
                                ),
                            },
                        }
                        if len(element_text.strip()) > 0:
                            chunk["summary"] = self._generate_summary(element_text)
                        else:
                            chunk["summary"] = ""
                        chunks.append(chunk)

                    elif isinstance(element, Image) and self.extract_images:
                        # Handle images with built-in OCR/hierarchy
                        section_path = page_hierarchy.get(page_number, [])
                        image_data = {
                            "type": "image",
                            "page_number": page_number,
                            "section_path": section_path,
                            "metadata": {
                                "element_type": element_type,
                                "coordinates": getattr(
                                    element.metadata, "coordinates", None
                                ),
                            },
                        }
                        images.append(image_data)

                    elif (
                        isinstance(element, CompositeElement)
                        and element_text
                        and "table" in element_text.lower()
                    ):
                        # Handle tables with built-in OCR/hierarchy
                        section_path = page_hierarchy.get(page_number, [])
                        table_data = {
                            "content": element_text,
                            "type": "table",
                            "page_number": page_number,
                            "section_path": section_path,
                            "metadata": {
                                "element_type": element_type,
                                "coordinates": getattr(
                                    element.metadata, "coordinates", None
                                ),
                            },
                        }
                        tables.append(table_data)

                    elif isinstance(element, Title):
                        # Track sections
                        sections.append(
                            {
                                "title": element_text,
                                "page_number": page_number,
                                "level": getattr(element.metadata, "level", 1),
                            }
                        )

                except Exception as e:
                    logger.error(f"Error processing element: {e}")
                    continue

            # Add relationships between chunks
            for i, chunk in enumerate(chunks):
                if i > 0:
                    chunk["prev_chunk"] = (
                        chunks[i - 1]["content"][:100]
                        if chunks[i - 1]["content"]
                        else ""
                    )
                if i < len(chunks) - 1:
                    chunk["next_chunk"] = (
                        chunks[i + 1]["content"][:100]
                        if chunks[i + 1]["content"]
                        else ""
                    )

            logger.info(
                f"Processed document: {len(chunks)} chunks, {len(images)} images, {len(tables)} tables, {len(sections)} sections"
            )
            return {
                "chunks": chunks,
                "images": images,
                "tables": tables,
                "sections": sections,
                "document_metadata": {
                    "filename": os.path.basename(pdf_path),
                    "path": pdf_path,
                    "total_chunks": len(chunks),
                    "total_images": len(images),
                    "total_tables": len(tables),
                    "total_sections": len(sections),
                },
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
