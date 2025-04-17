import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import weaviate
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
from langchain_openai import AzureOpenAI
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.readers.file.docs import PDFReader
from pydantic import BaseModel, Field
import ray
import uuid
from datetime import datetime
import logging
from pydantic import field_validator

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
    summary: str = Field(default="", description="Summary of the chunk content")
    page_number: int
    chunk_type: str
    metadata: Dict[str, Any]
    relationships: Dict[str, Any]

    @field_validator("summary", mode="before")
    @classmethod
    def convert_none_to_empty_string(cls, v):
        return "" if v is None else str(v)


def process_and_store_document(
    pdf_path: str,
    tenant_id: str,
    output_dir: str = "processed_outputs",
    username: str = None,
    parse_only: bool = False,
) -> tuple:
    """Process a document and store its chunks in Weaviate"""
    try:
        # Initialize processor (and Weaviate client only if not parse_only)
        processor = ConsultingReportProcessor(extract_images=True)
        logging.info(f"Processing document: {pdf_path}")
        result = processor.process_report(pdf_path)

        # If parse_only, write result to results.json and return
        if parse_only:
            os.makedirs(output_dir, exist_ok=True)
            company = result.get("document", {}).get("company", tenant_id)
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            out_dir = os.path.join(output_dir, company, filename)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "result.json")
            # Sanitize all chunk metadata for JSON serialization
            def make_json_serializable(obj):
                if isinstance(obj, (tuple, set)):
                    return list(obj)
                if hasattr(obj, "__dict__"):
                    return str(obj)
                return obj

            if "chunks" in result:
                for chunk in result["chunks"]:
                    if "metadata" in chunk:
                        chunk["metadata"] = {k: make_json_serializable(v) for k, v in chunk["metadata"].items()}

            try:
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                logging.info(f"Wrote parsed output to {out_path}")
            except Exception as e:
                logging.error(f"Failed to write results.json: {e}")
                raise
            return 1, 0

        # Otherwise, proceed with Weaviate logic
        client = setup_weaviate_client(username=username, tenant=tenant_id)
        document_id = str(uuid.uuid4())
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
                    "next_chunk": chunk.get("next_chunk", ""),
                    "prev_chunk": chunk.get("prev_chunk", ""),
                    "related_images": chunk.get("related_images", []),
                    "related_tables": chunk.get("related_tables", []),
                },
            ).dict()
            chunks.append(chunk_data)

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
    parse_only: bool = False,
) -> Dict[str, int]:
    """Process all PDF files in a directory"""
    stats = {"processed": 0, "failed": 0, "chunks_stored": 0, "chunks_failed": 0}

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    # Only process PDFs in the tenant-specific subdirectory
    tenant_dir = input_path / tenant_id
    if not tenant_dir.exists():
        raise ValueError(f"Tenant directory {tenant_dir} does not exist")

    for pdf_file in tenant_dir.glob("*.pdf"):
        try:
            successful, failed = process_and_store_document(
                str(pdf_file), tenant_id, output_dir, username, parse_only=parse_only
            )
            stats["processed"] += 1
            stats["chunks_stored"] += successful
            stats["chunks_failed"] += failed
            logging.info(f"Successfully processed {pdf_file}")
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {str(e)}")
            stats["failed"] += 1

    return stats


def batch_store_chunks(client: weaviate.Client, chunks: list) -> tuple:
    """Store multiple chunks in Weaviate and validate the batch"""
    successful = 0
    failed = 0

    try:
        # Get the collection
        collection = client.collections.get("DocumentChunk")

        # Get the tenant ID from the first chunk
        tenant_id = chunks[0].get("tenant_id") if chunks else None
        if not tenant_id:
            raise ValueError("No tenant_id found in chunks")

        logging.info(f"Starting batch operation for tenant {tenant_id}")

        # Get tenant-specific collection
        tenant_collection = collection.with_tenant(tenant_id)

        # Store chunks one by one (we can optimize this later if needed)
        for chunk in chunks:
            try:
                # Prepare chunk data
                chunk_data = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "tenant_id": chunk["tenant_id"],
                    "content": chunk["content"],
                    "summary": chunk.get("summary", ""),
                    "page_number": chunk.get("page_number", 0),
                    "section_title": chunk.get("section_title", ""),
                    "section_path": chunk.get("section_path", []),
                    "metadata": {
                        "source": chunk.get("metadata", {}).get("source", ""),
                        "extraction_date": chunk.get("metadata", {}).get(
                            "extraction_date", datetime.now().isoformat()
                        ),
                        "document_type": chunk.get("metadata", {}).get(
                            "document_type", "pdf"
                        ),
                    },
                }

                # Remove any potential non-serializable data
                if isinstance(chunk_data["metadata"], dict):
                    chunk_data["metadata"] = {
                        k: (
                            str(v)
                            if not isinstance(v, (str, int, float, bool, list, dict))
                            else v
                        )
                        for k, v in chunk_data["metadata"].items()
                    }

                # Store the chunk
                tenant_collection.data.insert(chunk_data)
                successful += 1

                # Log progress
                if successful % 50 == 0:
                    logging.info(f"Stored {successful} chunks...")

            except Exception as e:
                logging.error(f"Failed to store chunk: {str(e)}")
                failed += 1

        logging.info(
            f"Completed storing chunks. Success: {successful}, Failed: {failed}"
        )

        # Verify a few chunks with similarity search
        if successful > 0:
            logging.info("Verifying chunk storage with similarity search...")
            query_result = tenant_collection.query.near_text(
                query=chunks[0]["content"][
                    :100
                ],  # Use first chunk's content as test query
                limit=1,
            )
            if query_result and len(query_result.objects) > 0:
                logging.info("Verification successful - chunks are searchable")
            else:
                logging.warning(
                    "Verification failed - chunks may not be properly vectorized"
                )

        return successful, failed

    except Exception as e:
        logging.error(f"Error in batch operation: {str(e)}")
        return successful, failed


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
    parser.add_argument(
        "--parse-only", action="store_true", help="Only parse and write results.json, do not vectorize or upload to Weaviate"
    )

    args = parser.parse_args()

    try:
        logging.info(
            f"Starting processing with input_dir={args.input_dir}, tenant_id={args.tenant_id}, parse_only={args.parse_only}"
        )
        stats = process_directory(
            args.input_dir, args.tenant_id, args.output_dir, args.username, parse_only=args.parse_only
        )
        logging.info(f"Processing complete. Stats: {stats}")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        exit(1)
