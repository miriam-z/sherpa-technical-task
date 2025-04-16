import os
import logging
import pprint
import weaviate
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.collections.classes.config import (
    DataType,
    Configure,
)
from dotenv import load_dotenv
from typing import List, Dict, Any
from .rbac_config import setup_rbac, validate_user_access

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_weaviate_client(username: str = None, tenant: str = None) -> WeaviateClient:
    """Initialize Weaviate client with schema and RBAC"""
    # Load environment variables
    load_dotenv()

    # Validate environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([weaviate_url, weaviate_api_key, openai_api_key]):
        raise ValueError("Missing required environment variables")

    try:
        # Initialize client with v4 syntax for cloud instance
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(api_key=weaviate_api_key),
            headers={"X-OpenAI-Api-Key": openai_api_key},
        )

        logging.info(f"Connected to Weaviate at {weaviate_url}")

        # Delete existing collection if it exists
        if client.collections.exists("DocumentChunk"):
            logging.info("Deleting existing DocumentChunk collection")
            client.collections.delete("DocumentChunk")
            logging.info("Deleted existing DocumentChunk collection")

        # Define properties for the collection
        properties = [
            # Core identifiers
            {"name": "chunk_id", "data_type": DataType.TEXT},
            {"name": "document_id", "data_type": DataType.TEXT},
            {"name": "tenant_id", "data_type": DataType.TEXT},
            # Content
            {"name": "content", "data_type": DataType.TEXT},
            {"name": "summary", "data_type": DataType.TEXT},
            {"name": "page_number", "data_type": DataType.NUMBER},
            # Section information
            {"name": "section_title", "data_type": DataType.TEXT},
            {"name": "section_path", "data_type": DataType.TEXT_ARRAY},
            # Metadata
            {
                "name": "metadata",
                "data_type": DataType.OBJECT,
                "nested_properties": [
                    {"name": "source", "data_type": DataType.TEXT},
                    {"name": "extraction_date", "data_type": DataType.TEXT},
                ],
            },
        ]

        # Create collection with configuration
        collection = client.collections.create(
            name="DocumentChunk",
            description="Document chunks with metadata and hierarchy",
            properties=properties,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            multi_tenancy_config=Configure.multi_tenancy(
                enabled=True, auto_tenant_creation=False, auto_tenant_activation=True
            ),
        )
        logging.info("Created DocumentChunk collection with complete configuration")

        # Debug: Print the actual collection configuration
        actual_config = collection.config.get()
        logging.info("Actual collection configuration:")
        pprint.pprint(actual_config)

        return client

    except Exception as e:
        logging.error(f"Error creating schema: {str(e)}")
        if "client" in locals():
            client.close()
        raise


def validate_embedding(client: WeaviateClient, chunk_id: str) -> bool:
    """Validate that a chunk has been properly embedded"""
    try:
        # Get the object with vector
        result = client.collections.get("DocumentChunk").get(
            uuid=chunk_id, include_vector=True
        )

        if not result or "vector" not in result:
            logging.error(f"No vector found for chunk {chunk_id}")
            return False

        vector = result["vector"]
        if not vector or len(vector) == 0:
            logging.error(f"Empty vector for chunk {chunk_id}")
            return False

        logging.info(f"Valid vector found for chunk {chunk_id} (length: {len(vector)})")
        return True

    except Exception as e:
        logging.error(f"Error validating embedding for chunk {chunk_id}: {str(e)}")
        return False


def validate_chunk_relationships(
    client: WeaviateClient, chunk_id: str
) -> Dict[str, Any]:
    """Validate relationships for a chunk"""
    try:
        result = client.collections.get("DocumentChunk").get(uuid=chunk_id)

        if not result:
            return {"valid": False, "error": "Chunk not found"}

        # Check required relationship fields
        relationships = {
            "parent_chunk_id": result.get("parent_chunk_id"),
            "child_chunk_ids": result.get("child_chunk_ids", []),
            "related_image_ids": result.get("related_image_ids", []),
            "related_table_ids": result.get("related_table_ids", []),
        }

        # Log relationship status
        logging.info(f"Relationships for chunk {chunk_id}:")
        for key, value in relationships.items():
            if isinstance(value, list):
                logging.info(f"  {key}: {len(value)} items")
            else:
                logging.info(f"  {key}: {value}")

        return {"valid": True, "relationships": relationships}

    except Exception as e:
        logging.error(f"Error validating relationships for chunk {chunk_id}: {str(e)}")
        return {"valid": False, "error": str(e)}


def validate_batch_embeddings(
    client: WeaviateClient, chunk_ids: List[str]
) -> Dict[str, int]:
    """Validate embeddings for a batch of chunks"""
    results = {
        "total": len(chunk_ids),
        "valid_embeddings": 0,
        "invalid_embeddings": 0,
        "valid_relationships": 0,
        "invalid_relationships": 0,
    }

    for chunk_id in chunk_ids:
        # Validate embedding
        if validate_embedding(client, chunk_id):
            results["valid_embeddings"] += 1
        else:
            results["invalid_embeddings"] += 1

        # Validate relationships
        rel_result = validate_chunk_relationships(client, chunk_id)
        if rel_result["valid"]:
            results["valid_relationships"] += 1
        else:
            results["invalid_relationships"] += 1

    logging.info(f"Batch validation results: {results}")
    return results


def store_chunk(client: WeaviateClient, chunk: dict) -> str:
    """Store a single chunk in Weaviate and validate its embedding"""
    try:
        # Ensure required fields are present
        required_fields = ["chunk_id", "document_id", "tenant_id", "content"]
        for field in required_fields:
            if field not in chunk:
                raise ValueError(f"Missing required field: {field}")

        # Store the chunk using v4 syntax
        result = client.collections.get("DocumentChunk").create(
            properties=chunk, uuid=chunk["chunk_id"]
        )

        # Validate the embedding
        if not validate_embedding(client, chunk["chunk_id"]):
            logging.warning(
                f"Embedding validation failed for chunk {chunk['chunk_id']}"
            )

        return result
    except Exception as e:
        logging.error(f"Error storing chunk {chunk.get('chunk_id')}: {str(e)}")
        raise


def batch_store_chunks(client: WeaviateClient, chunks: list) -> tuple:
    """Store multiple chunks in Weaviate and validate the batch"""
    successful = 0
    failed = 0
    chunk_ids = []

    for chunk in chunks:
        try:
            store_chunk(client, chunk)
            successful += 1
            chunk_ids.append(chunk["chunk_id"])
        except Exception as e:
            logging.error(f"Failed to store chunk: {str(e)}")
            failed += 1

    # Validate the batch if we have successful chunks
    if chunk_ids:
        validation_results = validate_batch_embeddings(client, chunk_ids)
        logging.info(f"Batch validation results: {validation_results}")

    return successful, failed
