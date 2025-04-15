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
                strategy="fast", # testing 
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                include_metadata=True,
                ocr_languages=["eng"],
                pdf_image_dpi=150, # testing 
            )

            # Debug: Print what we got
            print("\nFound elements:")
            element_types = {}
            for el in elements:
                el_type = type(el).__name__
                if el_type not in element_types:
                    element_types[el_type] = 0
                element_types[el_type] += 1

                # Print details for images
                if el_type == "Image":
                    print(f"\nImage found:")
                    print(f"  Page: {getattr(el.metadata, 'page_number', 'unknown')}")
                    print(
                        f"  Coordinates: {getattr(el.metadata, 'coordinates', 'unknown')}"
                    )
                    for attr in ["image", "image_base64", "image_path"]:
                        if hasattr(el, attr) or (
                            hasattr(el, "metadata") and hasattr(el.metadata, attr)
                        ):
                            print(f"  Has {attr}")

            print("\nElement type counts:", element_types)

            # Initialize containers
            text_chunks = []
            images = []
            tables = []
            sections = []
            current_section = None

            # First pass: identify sections from titles
            for element in elements:
                if isinstance(element, Title):
                    title_text = str(element).strip()
                    if title_text and len(title_text) < 200:  # Reasonable title length
                        current_section = {
                            "id": f"section_{len(sections)}",
                            "title": title_text,
                            "content": "",
                            "page_number": getattr(element.metadata, "page_number", 1),
                        }
                        sections.append(current_section)

            # Second pass: process all elements
            for element in elements:
                element_type = type(element).__name__

                # Handle text content
                if element_type in [
                    "NarrativeText",
                    "Text",
                    "ListItem",
                    "Header",
                    "Footer",
                    "Title",
                    "FigureCaption",
                ]:
                    text = str(element).strip()
                    if text:
                        chunk = {
                            "id": f"chunk_{len(text_chunks)}",
                            "content": text,
                            "type": element_type,
                            "page_number": getattr(element.metadata, "page_number", 1),
                            "section_id": (
                                current_section["id"] if current_section else None
                            ),
                        }
                        text_chunks.append(chunk)

                        if current_section and element_type != "FigureCaption":
                            current_section["content"] += text + "\n"

                # Handle images
                elif element_type == "Image":
                    try:
                        # Try multiple ways to get image data
                        image_data = None

                        # Method 1: Direct image attribute
                        if hasattr(element, "image"):
                            image_data = element.image

                        # Method 2: Metadata image
                        elif hasattr(element.metadata, "image"):
                            image_data = element.metadata.image

                        # Method 3: Base64 encoded image
                        elif hasattr(element.metadata, "image_base64"):
                            try:
                                image_data = base64.b64decode(
                                    element.metadata.image_base64
                                )
                            except Exception as e:
                                print(f"Error decoding base64 image: {e}")

                        # Method 4: Image path
                        elif hasattr(element.metadata, "image_path"):
                            try:
                                with open(element.metadata.image_path, "rb") as f:
                                    image_data = f.read()
                            except Exception as e:
                                print(f"Error reading image from path: {e}")

                        if image_data:
                            # Convert string data to bytes if needed
                            if isinstance(image_data, str):
                                try:
                                    image_data = base64.b64decode(image_data)
                                except Exception as e:
                                    print(f"Error decoding image string: {e}")

                            if isinstance(image_data, bytes):
                                # Save image with section context
                                section_prefix = (
                                    f"section_{len(sections)}"
                                    if current_section
                                    else "misc"
                                )
                                image_path = self._save_image(
                                    image_data,
                                    pdf_path,
                                    f"{section_prefix}_img_{len(images)}",
                                )

                                if image_path:
                                    image_info = {
                                        "id": f"image_{len(images)}",
                                        "path": image_path,
                                        "page_number": getattr(
                                            element.metadata, "page_number", 1
                                        ),
                                        "section_id": (
                                            current_section["id"]
                                            if current_section
                                            else None
                                        ),
                                        "coordinates": getattr(
                                            element.metadata, "coordinates", None
                                        ),
                                    }

                                    # Try to find associated caption
                                    nearby_captions = [
                                        t
                                        for t in text_chunks[-5:]
                                        if t["type"] == "FigureCaption"
                                    ]
                                    if nearby_captions:
                                        image_info["caption"] = nearby_captions[-1][
                                            "content"
                                        ]

                                    images.append(image_info)
                                    print(f"✓ Saved image: {image_path}")
                            else:
                                print(f"Invalid image data type: {type(image_data)}")
                        else:
                            print("No image data found in element")
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        import traceback

                        print(traceback.format_exc())

                # Handle tables
                elif element_type == "Table":
                    table_content = str(element)
                    if table_content.strip():
                        tables.append(
                            {
                                "id": f"table_{len(tables)}",
                                "content": table_content,
                                "page_number": getattr(
                                    element.metadata, "page_number", 1
                                ),
                                "section_id": (
                                    current_section["id"] if current_section else None
                                ),
                            }
                        )

            # Clean up sections
            for section in sections:
                section["content"] = section["content"].strip()

            # Generate summaries for longer text chunks
            if self.model:
                chunks_to_summarize = [
                    chunk
                    for chunk in text_chunks
                    if len(chunk["content"]) > 200 and chunk["type"] == "NarrativeText"
                ]
                if chunks_to_summarize:
                    summaries = self._process_text_batch(
                        [c["content"] for c in chunks_to_summarize]
                    )
                    for chunk, summary in zip(chunks_to_summarize, summaries):
                        chunk["summary"] = summary

            # Organize results
            result = {
                "document": {
                    "id": f"{Path(pdf_path).parent.name}_{Path(pdf_path).stem}",
                    "company": Path(pdf_path).parent.name,
                    "filename": Path(pdf_path).name,
                },
                "content": {"text": text_chunks, "images": images, "tables": tables},
                "structure": {"sections": sections},
            }

            print(f"\nExtracted from {Path(pdf_path).name}:")
            print(f"  Sections: {len(sections)}")
            print(
                f"  Text chunks: {len(text_chunks)} ({', '.join(f'{v} {k}s' for k, v in element_types.items() if 'Text' in k or k in ['Header', 'Footer', 'ListItem'])})"
            )
            print(f"  Images: {len(images)} of {element_types.get('Image', 0)} found")
            print(f"  Tables: {len(tables)}")

            return result

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            import traceback

            print(traceback.format_exc())
            return {
                "error": str(e),
                "document": {
                    "id": f"{Path(pdf_path).parent.name}_{Path(pdf_path).stem}",
                    "company": Path(pdf_path).parent.name,
                    "filename": Path(pdf_path).name,
                },
                "content": {"text": [], "tables": [], "images": []},
                "structure": {"sections": []},
            }
