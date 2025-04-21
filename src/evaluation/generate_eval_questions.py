import os
import sys
import argparse
import openai
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import re
import glob


# Load environment variables
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def extract_text_from_parsed(parsed_path):
    """Read all text from a parsed output (txt/json) file."""
    ext = Path(parsed_path).suffix
    if ext == ".txt":
        with open(parsed_path, "r") as f:
            return f.read()
    elif ext == ".json":
        import json
        with open(parsed_path, "r") as f:
            data = json.load(f)
            # If data is a dict with 'chunks', aggregate 'content' from each chunk
            if isinstance(data, dict) and "chunks" in data:
                return "\n".join(chunk.get("content", "") for chunk in data["chunks"] if isinstance(chunk, dict))
            # Fallback: Try to extract all 'text' fields if present
            if isinstance(data, list):
                return "\n".join([d.get("text", "") for d in data if isinstance(d, dict)])
            elif isinstance(data, dict):
                return data.get("text", "")
    raise ValueError(f"Unsupported file type for {parsed_path}")


def generate_questions(text, n=10, model="gpt-4-turbo-preview"):
    prompt = f"""
You are an expert consultant. Given the following report content, generate {n} diverse, high-quality evaluation questions that could be answered by this content. Focus on key facts, trends, insights, and recommendations. Output only the questions as a numbered list.

Report Content:
{text[:4000]}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    # Extract questions from numbered list
    questions = []
    for line in answer.split("\n"):
        if line.strip() and any(c.isdigit() for c in line[:3]):
            q = line.split(".", 1)[-1].strip()
            if q:
                questions.append(q)
    return questions



PROCESSED_OUTPUTS_DIR = "processed_outputs"

def safe_filename(name):
    # Replace spaces and special chars with underscores
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def run_for_tenant(parsed_path, tenant, output=None, n=10):
    text = extract_text_from_parsed(parsed_path)
    questions = generate_questions(text, n=n)
    output_path = output or f"eval_questions_{tenant}.txt"
    with open(output_path, "w") as f:
        for q in questions:
            f.write(q + "\n")
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Saved {len(questions)} questions to {output_path}")

def run_for_all_tenants(n=10):
    import logging
    logging.basicConfig(level=logging.INFO)
    output_dir = os.path.join(os.path.dirname(__file__), "eval_questions")
    os.makedirs(output_dir, exist_ok=True)
    tenants = [d for d in os.listdir(PROCESSED_OUTPUTS_DIR) if os.path.isdir(os.path.join(PROCESSED_OUTPUTS_DIR, d))]
    for tenant in tenants:
        tenant_dir = os.path.join(PROCESSED_OUTPUTS_DIR, tenant)
        all_texts = []
        for root, dirs, files in os.walk(tenant_dir):
            for file in files:
                if file == "result.json":
                    parsed_path = os.path.join(root, file)
                    text = extract_text_from_parsed(parsed_path)
                    if text:
                        all_texts.append(text)
        aggregated_text = "\n".join(all_texts)
        output_path = os.path.join(output_dir, f"eval_questions_{tenant}.txt")
        if aggregated_text.strip():
            logging.info(f"Generating questions for tenant {tenant} (all PDFs)...")
            questions = generate_questions(aggregated_text, n=n)
            with open(output_path, "w") as f:
                for q in questions:
                    f.write(q + "\n")
            logging.info(f"Saved {len(questions)} questions to {output_path}")
        else:
            logging.warning(f"No content found for tenant {tenant}, skipping question generation.")

def generate_groundtruth_csvs():
    import glob
    import pandas as pd
    eval_dir = os.path.join(os.path.dirname(__file__), "eval_questions")
    txt_files = glob.glob(os.path.join(eval_dir, "eval_questions_*.txt"))
    if not txt_files:
        print(f"No eval_questions_*.txt files found in {eval_dir}.")
        return
    for txt_path in txt_files:
        tenant = os.path.basename(txt_path).replace("eval_questions_", "").replace(".txt", "")
        csv_path = os.path.join(eval_dir, f"groundtruth_{tenant}.csv")
        with open(txt_path, "r") as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]
        df = pd.DataFrame({"query": questions, "expected_response": ["" for _ in questions]})
        df.to_csv(csv_path, index=False)
        print(f"Generated {csv_path} with {len(df)} questions.")

def main():
    import sys
    parser = argparse.ArgumentParser(description="Generate evaluation questions from parsed PDF text.")
    parser.add_argument("--parsed_path", type=str, help="Path to parsed txt or json file.")
    parser.add_argument("--tenant", type=str, help="Tenant name (e.g., bain, mck, bcg).")
    parser.add_argument("--output", type=str, help="Where to save the questions. If omitted, defaults to eval_questions_{tenant}.txt")
    parser.add_argument("--n", type=int, default=10, help="Number of questions to generate.")
    parser.add_argument("--all-tenants", action="store_true", help="Run for all tenants defined in TENANT_PARSED_PATHS.")
    parser.add_argument("--generate-groundtruth-csvs", action="store_true", help="Generate groundtruth CSVs from eval_questions_*.txt files.")
    args = parser.parse_args()

    if args.generate_groundtruth_csvs:
        generate_groundtruth_csvs()
        return

    if args.all_tenants:
        run_for_all_tenants(args.n)
    else:
        if not args.tenant or not args.parsed_path:
            parser.error("--tenant and --parsed_path are required unless --all-tenants is set.")
        run_for_tenant(args.parsed_path, args.tenant, args.output, args.n)

if __name__ == "__main__":
    main()
