import os
import logging
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.exceptions import UnexpectedStatusCodeException
from contextlib import closing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_tenant_access(tenant_id: str):
    """Test access for a specific tenant."""
    try:
        # Initialize client with tenant-specific credentials
        with closing(
            weaviate.connect_to_weaviate_cloud(
                cluster_url=os.getenv("WEAVIATE_URL"),
                auth_credentials=AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
                headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY"),
                    "X-Tenant-ID": tenant_id,
                },
            )
        ) as client:
            # Try to query the DocumentChunk class
            response = client.collections.get("DocumentChunk").query.fetch_objects(
                limit=1, return_properties=["chunk_id", "document_id", "tenant_id"]
            )

            logger.info(f"Successfully connected to tenant {tenant_id}")
            logger.info(f"Query response: {response}")
            return True

    except UnexpectedStatusCodeException as e:
        logger.error(f"Failed to access tenant {tenant_id}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error for tenant {tenant_id}: {str(e)}")
        return False


def main():
    # Test each tenant
    tenants = ["bain", "bcg", "mck"]

    results = {}
    for tenant_id in tenants:
        results[tenant_id] = test_tenant_access(tenant_id)

    # Print summary
    logger.info("\nTest Results:")
    for tenant_id, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"{tenant_id.upper()}: {status}")


if __name__ == "__main__":
    main()
