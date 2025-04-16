import os
import logging
from dotenv import load_dotenv
from utils.weaviate_setup import setup_weaviate_client
from utils.rbac_config import setup_rbac
from store_vectors import process_directory
from contextlib import closing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Main function to orchestrate the entire setup and processing flow"""
    try:
        # Step 1: Set up Weaviate schema and client
        logger.info("Setting up Weaviate client and schema...")
        with closing(setup_weaviate_client()) as client:
            # Step 2: Set up RBAC and tenants
            logger.info("Setting up RBAC and tenants...")
            setup_rbac(client)

            # Step 3: Process and store documents
            logger.info("Processing and storing documents...")
            stats = process_directory(
                input_dir="test_data",
                tenant_id="sherpa",  # Default tenant
                output_dir="processed_outputs",
            )
            logger.info(f"Processing stats: {stats}")

            logger.info("Setup completed successfully!")
            return True

    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        return False


if __name__ == "__main__":
    main()
