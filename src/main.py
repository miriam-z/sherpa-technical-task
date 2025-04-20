import asyncio
import sys
import os
import logging
import logging
from pathlib import Path
from typing import Dict, List
from store_vectors import process_directory, process_all_tenants
from utils.weaviate_setup import setup_weaviate_client

# Configure logging
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# The process_all_tenants function now lives in store_vectors.py to centralize processing logic for CLI usage.
# from store_vectors import process_all_tenants

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDFs and store vectors in Weaviate for all tenants"
    )
    parser.add_argument(
        "--input-dir",
        default="mbb_ai_reports",
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
