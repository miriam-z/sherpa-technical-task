import os
import logging
from pathlib import Path
from typing import Dict, List
from store_vectors import process_directory
from utils.weaviate_setup import setup_weaviate_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_all_tenants(
    input_dir: str = "test_data", output_dir: str = "processed_outputs", parse_only: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Process all PDF files for all tenants in the input directory.
    Each tenant should have their own subdirectory.
    """
    stats = {}
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    # Get all tenant directories
    tenant_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not tenant_dirs:
        logging.warning(f"No tenant directories found in {input_dir}")
        return stats

    for tenant_dir in tenant_dirs:
        tenant_id = tenant_dir.name.lower()
        logging.info(f"Processing tenant: {tenant_id} (directory: {tenant_dir.name})")
        try:
            tenant_stats = process_directory(input_dir, tenant_id, output_dir, parse_only=parse_only)
            stats[tenant_id] = tenant_stats
            logging.info(
                f"Completed processing for tenant {tenant_id}. Stats: {tenant_stats}"
            )
        except Exception as e:
            logging.error(f"Failed to process tenant {tenant_id}: {str(e)}")
            stats[tenant_id] = {
                "processed": 0,
                "failed": 1,
                "chunks_stored": 0,
                "chunks_failed": 0,
            }

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDFs and store vectors in Weaviate for all tenants"
    )
    parser.add_argument(
        "--input-dir",
        default="test_data",
        help="Directory containing tenant subdirectories with PDFs (default: test_data)",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_outputs",
        help="Directory for processed outputs (default: processed_outputs)",
    )

    parser.add_argument(
        "--parse-only", action="store_true", help="Only parse and write results.json, do not vectorize or upload to Weaviate"
    )
    args = parser.parse_args()

    try:
        logging.info(f"Starting processing with input_dir={args.input_dir}, parse_only={args.parse_only}")
        all_stats = process_all_tenants(args.input_dir, args.output_dir, parse_only=args.parse_only)
        logging.info("Processing complete. Stats by tenant:")
        for tenant_id, stats in all_stats.items():
            logging.info(f"{tenant_id}: {stats}")
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        exit(1)
