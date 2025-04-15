import os
from pathlib import Path
import ray
import json
from datetime import datetime
from processors.document_processor import ConsultingReportProcessor
from langchain_openai import AzureChatOpenAI
from typing import Dict, List


def validate_result(result: Dict) -> bool:
    """Validate the processing result has the required structure and content"""
    required_keys = ["document", "content", "structure"]
    if not all(key in result for key in required_keys):
        return False

    # Check content sections
    content = result["content"]
    if not isinstance(content, dict):
        return False

    return True


def check_env_variables() -> bool:
    """Quick check for required environment variables"""
    required_vars = [
        "DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_KEY",
        "ENDPOINT_URL",
        "LLAMA_API_KEY",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"\nMissing environment variables: {', '.join(missing)}")
        return False
    return True


@ray.remote(num_cpus=2, memory=4 * 1024 * 1024 * 1024)  # 4GB per worker
def process_pdf(pdf_path: str) -> dict:
    """Process a single PDF file with optimized settings"""
    try:
        print(f"\nProcessing {pdf_path}")

        # Initialize model and processor only once per worker
        model = AzureChatOpenAI(
            deployment_name=os.getenv("DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("ENDPOINT_URL"),
            api_version="2024-05-01-preview",
            temperature=0.3,
            request_timeout=30,  # Add timeout to prevent hanging
            max_retries=2,  # Limit retries
        )

        processor = ConsultingReportProcessor(
            model=model,
            extract_images=True,
            batch_size=3,  # Process text in smaller batches for stability
        )

        # Process the PDF
        result = processor.process_report(pdf_path)

        if not validate_result(result):
            raise ValueError("Invalid result structure")

        # Create output directory and save results
        output_dir = (
            Path("processed_outputs") / Path(pdf_path).parent.name / Path(pdf_path).stem
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = output_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        # Count elements for summary
        counts = {
            "text_chunks": len(result["content"].get("text", [])),
            "tables": len(result["content"].get("tables", [])),
            "images": len(result["content"].get("images", [])),
            "sections": len(result["structure"].get("sections", [])),
        }

        print(
            f"✓ {Path(pdf_path).name} - {counts['text_chunks']} texts, {counts['tables']} tables, {counts['images']} images"
        )

        return {
            "status": "success",
            "pdf_path": pdf_path,
            "output_path": str(result_path),
            **counts,
        }

    except Exception as e:
        print(f"✗ Error processing {pdf_path}: {str(e)}")
        return {"status": "error", "pdf_path": pdf_path, "error": str(e)}


def main():
    if not check_env_variables():
        return

    # Initialize Ray with increased resources
    ray.init(
        runtime_env={
            "env_vars": {
                "DEPLOYMENT_NAME": os.getenv("DEPLOYMENT_NAME"),
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
                "ENDPOINT_URL": os.getenv("ENDPOINT_URL"),
                "LLAMA_API_KEY": os.getenv("LLAMA_API_KEY"),
            }
        }
    )

    # Create output directory
    output_dir = Path("processed_outputs")
    output_dir.mkdir(exist_ok=True)

    # Collect PDF paths
    pdf_paths = []
    for company in ["Bain", "BCG", "McK"]:
        company_dir = Path("test_data") / company
        if company_dir.exists():
            pdf_paths.extend(list(company_dir.glob("*.pdf")))

    if not pdf_paths:
        print("No PDFs found in test_data directory")
        ray.shutdown()
        return

    print(f"\nProcessing {len(pdf_paths)} PDFs...")

    # Process PDFs in parallel with better error handling
    try:
        futures = [process_pdf.remote(str(path)) for path in pdf_paths]
        results = ray.get(futures)
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        ray.shutdown()
        return

    # Save processing summary
    summary = {
        "start_time": datetime.now().isoformat(),
        "total_files": len(pdf_paths),
        "successful_files": len([r for r in results if r["status"] == "success"]),
        "failed_files": len([r for r in results if r["status"] == "error"]),
        "results": results,
    }

    with open(output_dir / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nProcessed {summary['successful_files']}/{summary['total_files']} files successfully"
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
