import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import weaviate
from dotenv import load_dotenv
from langchain_openai import AzureOpenAI
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.readers.file.docs import PDFReader
from pydantic import BaseModel, Field
import ray
import uuid
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

from utils.weaviate_setup import setup_weaviate_client, batch_store_chunks
from processors.document_processor import ConsultingReportProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Pydantic Models
class DocumentInfo(BaseModel):
    """Information about the source document"""

    id: str = Field(..., description="Unique identifier for the document")
    company: str = Field(..., description="Company name (e.g., Bain, BCG)")
    filename: str = Field(..., description="Original filename")
    path: str = Field(..., description="Full path to the document")


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk"""

    document: DocumentInfo
    section_title: str = Field(default="", description="Title of the section")
    section_path: List[str] = Field(
        default_factory=list, description="Hierarchical path"
    )
    page_number: int = Field(default=0, description="Page number in document")
    chunk_type: str = Field(default="text", description="Type of content")
    parent_chunk_id: str = Field(default="", description="ID of parent chunk")
    child_chunk_ids: List[str] = Field(
        default_factory=list, description="IDs of child chunks"
    )


class DocumentChunk(BaseModel):
    """Model for document chunks to be stored in Weaviate"""

    chunk_id: str
    document_id: str
    tenant_id: str
    content: str
    summary: str = ""
    page_number: int
    chunk_type: str
    metadata: Dict[str, Any]
    relationships: Dict[str, Any]


def process_and_store_document(
    pdf_path: str,
    tenant_id: str,
    output_dir: str = "processed_outputs",
    username: str = None,
) -> tuple:
    """Process a document and store its chunks in Weaviate"""
    try:
        # Initialize processor and Weaviate client with RBAC
        processor = ConsultingReportProcessor()
        client = setup_weaviate_client(username=username, tenant=tenant_id)

        # Process the document
        logging.info(f"Processing document: {pdf_path}")
        result = processor.process_report(pdf_path)

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Prepare chunks for storage
        chunks = []
        for chunk in result.get("chunks", []):
            chunk_data = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                tenant_id=tenant_id,
                content=chunk["content"],
                summary=chunk.get("summary", ""),
                page_number=chunk.get("page_number", 0),
                chunk_type=chunk.get("type", "text"),
                metadata={
                    "source": os.path.basename(pdf_path),
                    "document_metadata": result.get("document_metadata", {}),
                    "section_path": chunk.get("section_path", []),
                    **chunk.get("metadata", {}),
                },
                relationships={
                    "next_chunk": chunk.get("next_chunk"),
                    "prev_chunk": chunk.get("prev_chunk"),
                    "related_images": chunk.get("related_images", []),
                    "related_tables": chunk.get("related_tables", []),
                },
            ).dict()
            chunks.append(chunk_data)

        # Store chunks in Weaviate
        if chunks:
            logging.info(f"Storing {len(chunks)} chunks in Weaviate")
            successful, failed = batch_store_chunks(client, chunks)
            logging.info(f"Successfully stored {successful} chunks, {failed} failed")
            return successful, failed
        else:
            logging.warning("No chunks found to store")
            return 0, 0

    except Exception as e:
        logging.error(f"Error processing document {pdf_path}: {str(e)}")
        raise


def process_directory(
    input_dir: str,
    tenant_id: str,
    output_dir: str = "processed_outputs",
    username: str = None,
) -> Dict[str, int]:
    """Process all PDF files in a directory"""
    stats = {"processed": 0, "failed": 0, "chunks_stored": 0, "chunks_failed": 0}

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    for pdf_file in input_path.glob("**/*.pdf"):
        try:
            successful, failed = process_and_store_document(
                str(pdf_file), tenant_id, output_dir, username
            )
            stats["processed"] += 1
            stats["chunks_stored"] += successful
            stats["chunks_failed"] += failed
            logging.info(f"Successfully processed {pdf_file}")
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {str(e)}")
            stats["failed"] += 1

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDFs and store vectors in Weaviate"
    )
    parser.add_argument(
        "--input-dir",
        default="test_data",
        help="Directory containing PDF files to process (default: test_data)",
    )
    parser.add_argument(
        "--tenant-id",
        default="sherpa",
        help="Tenant ID for the documents (default: sherpa)",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_outputs",
        help="Directory for processed outputs (default: processed_outputs)",
    )
    parser.add_argument(
        "--username", default=None, help="Username for RBAC authentication"
    )

    args = parser.parse_args()

    try:
        logging.info(
            f"Starting processing with input_dir={args.input_dir}, tenant_id={args.tenant_id}"
        )
        stats = process_directory(
            args.input_dir, args.tenant_id, args.output_dir, args.username
        )
        logging.info(f"Processing complete. Stats: {stats}")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        exit(1)
