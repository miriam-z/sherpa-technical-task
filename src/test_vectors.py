import os
import logging
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_vector_generation():
    """Test vector generation with a small chunk of text"""
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

        # Generate UUID
        chunk_id = str(uuid.uuid4())

        # Simple test data
        test_data = {
            "content": "Artificial Intelligence is transforming the luxury goods industry.",
            "tenant_id": "bain",
            "chunk_id": chunk_id,
            "document_id": str(uuid.uuid4()),
        }

        # Get collection with tenant context
        collection = client.collections.get("DocumentChunk")
        tenant_collection = collection.with_tenant("bain")

        # Store the chunk
        logger.info("Storing test chunk...")
        tenant_collection.data.insert(test_data)
        logger.info(f"Stored chunk with ID: {chunk_id}")

        # Wait a moment for vector generation
        import time

        time.sleep(5)  # Give it 5 seconds to generate the vector

        # Try a similarity search
        logger.info("\nTesting similarity search...")
        query_result = tenant_collection.query.near_text(
            query="AI in luxury retail", limit=1
        )

        if query_result and len(query_result.objects) > 0:
            logger.info("✅ Similarity search successful!")
            logger.info(
                f"Found matching content: {query_result.objects[0].properties['content']}"
            )
            return True
        else:
            logger.error("❌ No results from similarity search")
            return False

    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False

    finally:
        if "client" in locals():
            client.close()


if __name__ == "__main__":
    test_vector_generation()
