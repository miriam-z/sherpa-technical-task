import os
from typing import Dict, List, Optional
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image, Text, Title, CompositeElement
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_parse import LlamaParse
import base64
import re


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
        self.current_section = None
        self.section_counters = {}  # Track figure numbers per section
        self.llama_parser = LlamaParse(
            api_key=os.getenv("LLAMA_API_KEY"), result_type="markdown"
        )

    def _get_section_number(self, title_text: str) -> str:
        """Extract section number from title text"""
        # Try to find a section number pattern (e.g., "1.", "1.2.", "A.", etc.)
        match = re.match(r"^(?:(?:\d+\.)+|\w+\.)\s*", title_text)
        if match:
            return match.group(0).strip(".")
        return None

    def _get_figure_number(self, section_number: str) -> str:
        """Get next figure number for a section"""
        if section_number not in self.section_counters:
            self.section_counters[section_number] = 1
        else:
            self.section_counters[section_number] += 1
        return f"{section_number}-{self.section_counters[section_number]}"

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
                        print(f"Error generating summary: {e}")
                        batch_summaries.append(None)
                summaries.extend(batch_summaries)
            return summaries
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return [None] * len(texts)

    def _generate_summary(self, text: str) -> Optional[str]:
        """Generate a summary for a single text"""
        return self._process_text_batch([text])[0] if text else None

    def _save_image(
        self, image_content: bytes, pdf_path: str, section_number: str
    ) -> str:
        """Save image with proper figure numbering"""
        company = Path(pdf_path).parent.name
        doc_name = Path(pdf_path).stem
        figures_dir = Path("processed_outputs") / company / doc_name / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        figure_number = self._get_figure_number(section_number)
        image_filename = f"figure-{figure_number}.jpg"
        image_path = figures_dir / image_filename

        try:
            with open(image_path, "wb") as f:
                f.write(image_content)
            print(f"✓ Saved image: {image_path}")
            return str(image_path.relative_to("processed_outputs"))
        except Exception as e:
            print(f"Error saving image {image_path}: {e}")
            return ""

    def _build_section_hierarchy(self, parsed_doc) -> List[Dict]:
        """Build section hierarchy from LlamaParse output"""
        sections = []
        section_map = {}  # Map to track parent-child relationships

        def process_section(section_data, parent_id=None, level=0):
            section_number = section_data.get("section_number", str(len(sections) + 1))
            section = {
                "id": f"section_{section_number}",
                "number": section_number,
                "title": section_data.get("heading", "Untitled Section"),
                "content": section_data.get("content", ""),
                "level": level,
                "parent_id": parent_id,
                "children": [],
                "page_number": section_data.get("page_number", 1),
            }
            sections.append(section)
            section_map[section["id"]] = section

            # Process subsections
            for subsection in section_data.get("subsections", []):
                child_id = process_section(subsection, section["id"], level + 1)
                section["children"].append(child_id)

            return section["id"]

        # Process root sections
        for section in parsed_doc.get("sections", []):
            process_section(section)

        return sections

    def process_report(self, pdf_path: str) -> Dict:
        """Process a single PDF report"""
        try:
            print(f"\nProcessing {pdf_path}")

            # Extract content with unstructured first
            print("1. Extracting content...")
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

            # Initialize containers
            chunks = []
            images = []
            tables = []
            sections = []
            current_section = None

            # First pass: identify sections from titles
            for element in elements:
                if element is None:
                    continue

                if isinstance(element, Title):
                    try:
                        title_text = str(element).strip() if element else ""
                        if (
                            title_text and len(title_text) < 200
                        ):  # Reasonable title length
                            current_section = {
                                "id": f"section_{len(sections)}",
                                "title": title_text,
                                "content": "",
                                "page_number": getattr(
                                    element.metadata, "page_number", 1
                                ),
                            }
                            sections.append(current_section)
                    except Exception as e:
                        print(f"Error processing title: {e}")
                        continue

            # Second pass: process all elements
            for element in elements:
                if element is None:
                    continue

                try:
                    element_type = type(element).__name__
                    page_number = getattr(element.metadata, "page_number", 1)

                    try:
                        element_text = str(element) if element else ""
                    except Exception as e:
                        print(f"Error converting element to string: {e}")
                        element_text = ""

                    if isinstance(element, Text) and element_text:
                        # Create a chunk for each text element
                        chunk = {
                            "content": element_text,
                            "type": "text",
                            "page_number": page_number,
                            "section_path": [
                                s["title"]
                                for s in sections
                                if s["page_number"] <= page_number
                            ],
                            "metadata": {
                                "element_type": element_type,
                                "coordinates": getattr(
                                    element.metadata, "coordinates", None
                                ),
                            },
                        }

                        # Only generate summary for non-empty content
                        if len(element_text.strip()) > 0:
                            chunk["summary"] = self._generate_summary(element_text)
                        else:
                            chunk["summary"] = ""

                        chunks.append(chunk)

                    elif isinstance(element, Image) and self.extract_images:
                        # Handle images
                        image_data = {
                            "type": "image",
                            "page_number": page_number,
                            "section_path": [
                                s["title"]
                                for s in sections
                                if s["page_number"] <= page_number
                            ],
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
                        # Handle tables
                        table_data = {
                            "content": element_text,
                            "type": "table",
                            "page_number": page_number,
                            "section_path": [
                                s["title"]
                                for s in sections
                                if s["page_number"] <= page_number
                            ],
                            "metadata": {
                                "element_type": element_type,
                                "coordinates": getattr(
                                    element.metadata, "coordinates", None
                                ),
                            },
                        }
                        tables.append(table_data)
                except Exception as e:
                    print(f"Error processing element: {e}")
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
            print(f"Error processing document: {e}")
            raise
