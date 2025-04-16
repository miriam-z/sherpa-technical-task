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


def setup_weaviate_client(username: str = None, tenant: str = None) -> weaviate.Client:
    """Set up Weaviate client and schema"""
    try:
        # Initialize client
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
            headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY"),
                "Content-Type": "application/json",
            },
        )

        # Only create collection if it doesn't exist
        if not client.collections.exists("DocumentChunk"):
            collection = client.collections.create(
                name="DocumentChunk",
                description="Document chunks with metadata and hierarchy",
                properties=properties,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small",
                    dimensions=1536,
                    vectorize_collection_name=False,
                ),
                multi_tenancy_config=Configure.multi_tenancy(
                    enabled=True, auto_tenant_creation=True, auto_tenant_activation=True
                ),
                generative_config=Configure.Generative.openai(),
            )
            logging.info("Created new DocumentChunk collection")
        else:
            collection = client.collections.get("DocumentChunk")
            logging.info("Using existing DocumentChunk collection")

        return client

    except Exception as e:
        logging.error(f"Error setting up Weaviate: {str(e)}")
        raise


def validate_embedding(
    client: WeaviateClient, chunk_id: str, tenant_id: str = None
) -> bool:
    """Validate that a chunk has been properly embedded"""
    try:
        # Get the collection and set tenant context if provided
        collection = client.collections.get("DocumentChunk")
        if tenant_id:
            collection = collection.with_tenant(tenant_id)

        # Get the object with vector
        result = collection.query.fetch_object_by_id(uuid=chunk_id, include_vector=True)

        if not result or not hasattr(result, "vector"):
            logging.error(f"No vector found for chunk {chunk_id}")
            return False

        vector = result.vector
        if not vector or len(vector) == 0:
            logging.error(f"Empty vector for chunk {chunk_id}")
            return False

        logging.info(f"Valid vector found for chunk {chunk_id} (length: {len(vector)})")
        return True

    except Exception as e:
        logging.error(f"Error validating embedding for chunk {chunk_id}: {str(e)}")
        return False


def validate_chunk_relationships(
    client: WeaviateClient, chunk_id: str, tenant_id: str = None
) -> Dict[str, Any]:
    """Validate relationships for a chunk"""
    try:
        # Get the collection and set tenant context if provided
        collection = client.collections.get("DocumentChunk")
        if tenant_id:
            collection = collection.with_tenant(tenant_id)

        # Get the object
        result = collection.query.fetch_object_by_id(uuid=chunk_id)

        if not result:
            return {"valid": False, "error": "Chunk not found"}

        # Check required relationship fields
        relationships = {
            "parent_chunk_id": getattr(result, "parent_chunk_id", None),
            "child_chunk_ids": getattr(result, "child_chunk_ids", []),
            "related_image_ids": getattr(result, "related_image_ids", []),
            "related_table_ids": getattr(result, "related_table_ids", []),
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
    client: WeaviateClient, chunk_ids: List[str], tenant_id: str = None
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
        if validate_embedding(client, chunk_id, tenant_id):
            results["valid_embeddings"] += 1
        else:
            results["invalid_embeddings"] += 1

        # Validate relationships
        rel_result = validate_chunk_relationships(client, chunk_id, tenant_id)
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

        # Start a batch
        with tenant_collection.batch.dynamic() as batch:
            for chunk in chunks:
                try:
                    # Verify tenant ID matches
                    if chunk.get("tenant_id") != tenant_id:
                        raise ValueError(
                            f"Chunk tenant_id {chunk.get('tenant_id')} does not match batch tenant_id {tenant_id}"
                        )

                    # Add the chunk to the batch
                    batch.add_object(properties=chunk, uuid=chunk["chunk_id"])
                    successful += 1
                    chunk_ids.append(chunk["chunk_id"])

                    # Log every 50 chunks
                    if successful % 50 == 0:
                        logging.info(f"Added {successful} chunks to batch")

                except Exception as e:
                    logging.error(f"Failed to add chunk to batch: {str(e)}")
                    failed += 1

        logging.info(
            f"Batch operation completed. Successfully added {successful} chunks to batch"
        )

        # Verify chunks were stored
        if chunk_ids:
            # Try to retrieve a few chunks to verify storage
            test_chunks = chunk_ids[:5]  # Test first 5 chunks
            for chunk_id in test_chunks:
                try:
                    result = tenant_collection.query.fetch_object_by_id(uuid=chunk_id)
                    if result:
                        logging.info(f"Verified chunk {chunk_id} exists in Weaviate")
                    else:
                        logging.error(f"Chunk {chunk_id} not found in Weaviate")
                except Exception as e:
                    logging.error(f"Error verifying chunk {chunk_id}: {str(e)}")

            # Run validation
            validation_results = validate_batch_embeddings(client, chunk_ids, tenant_id)
            logging.info(f"Batch validation results: {validation_results}")

        return successful, failed

    except Exception as e:
        logging.error(f"Error in batch operation: {str(e)}")
        return successful, failed
