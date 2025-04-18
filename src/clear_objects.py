import weaviate
from weaviate.auth import AuthApiKey
import logging
from dotenv import load_dotenv
import os
from utils.rbac_config import TENANTS
from weaviate.classes.query import Filter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clear_tenant_objects(weaviate_url, api_key, class_name, tenant_id, dry_run=False):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(api_key)
    )
    try:
        collection = client.collections.get(class_name).with_tenant(tenant_id)
        result = collection.data.delete_many(
            where=Filter.by_property("tenant_id").equal(tenant_id),
            dry_run=dry_run,
            verbose=True
        )
        if dry_run:
            logging.info(f"Dry run for tenant {tenant_id}: {result}")
        else:
            logging.info(f"Successfully deleted objects for tenant: {tenant_id}")
    except Exception as e:
        logging.error(f"Error deleting objects: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    API_KEY = os.getenv("WEAVIATE_API_KEY")
    CLASS_NAME = "DocumentChunk"
    print("WARNING: This will delete ALL objects for all tenants in the collection 'DocumentChunk'.")
    dry_run_input = input("Dry run only? Type 'yes' for dry run, anything else to proceed: ").strip().lower()
    dry_run = dry_run_input == 'yes'
    if not dry_run:
        confirm = input("Type 'yes' to confirm actual deletion: ").strip().lower()
        if confirm != 'yes':
            print("Aborted. No objects were deleted.")
            exit(0)
    for tenant_id in TENANTS.keys():
        clear_tenant_objects(WEAVIATE_URL, API_KEY, CLASS_NAME, tenant_id, dry_run=dry_run)
    if dry_run:
        print("Dry run complete. No objects were deleted.")
    else:
        print("Deletion complete.")
