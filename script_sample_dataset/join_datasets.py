import json
import re
from pathlib import Path


def join_datasets():
    input_dir = Path(
        "/root/vast-ai-setup/MMD-Flagger/script_sample_dataset/dataset/example_summarization"
    )
    ragtruth_path = Path(
        "/root/vast-ai-setup/MMD-Flagger/script_sample_dataset/dataset/RAGTruth/summary_mini_models.jsonl"
    )
    output_dir = Path(
        "/root/vast-ai-setup/MMD-Flagger/script_sample_dataset/dataset/example_summarization_plus"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load RAGTruth mini models into a lookup dict keyed by prompt text
    # Map prompt -> full record for mistral model
    rag_lookup = {}
    with open(ragtruth_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_text = record.get("source", {}).get("prompt", "")
            model_name = record.get("response", {}).get("model", "")
            if model_name.lower() in ["mistral-7b-instruct", "mistralai/mistral-7b-instruct-v0.2"]:
                rag_lookup[prompt_text.strip()] = record

    print(f"Loaded {len(rag_lookup)} Mistral response lookup entries from RAGTruth.")

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    print(f"Processing {len(jsonl_files)} files from {input_dir}...")

    for file_path in jsonl_files:
        # Extract count_hallucination from filename (e.g. ..._h4_...)
        h_match = re.search(r"_h(\d+)_", file_path.name)
        count_hallucination = int(h_match.group(1)) if h_match else None

        # 1. Load jsonl file
        stochastic_records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    stochastic_records.append(json.loads(line))

        if not stochastic_records:
            print(f"Warning: {file_path.name} is empty.")
            continue

        # 2. Extract prompt to top level, set response_stochastic, delete prompt key from dict objects
        prompt = stochastic_records[0].get("prompt", "")
        for record in stochastic_records:
            record.pop("prompt", None)

        # 3. Search RAGTruth record matching prompt text for mistralai/Mistral-7B-Instruct-v0.2
        matched_record = rag_lookup.get(prompt.strip())
        if matched_record is None:
            print(f"Error: No matching Mistral response found for prompt in {file_path.name}")
            continue

        matched_response = matched_record.get("response", {}).get("response", "")
        response_id = int(matched_record["response"]["id"])
        source_id = int(matched_record["source"]["source_id"])

        if count_hallucination is None:
            count_hallucination = int(matched_record.get("count_hallucination", 0))

        # 4. Construct complete model with updated fields
        complete_model = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt": prompt,
            "count_hallucination": count_hallucination,
            "response_id": response_id,
            "source_id": source_id,
            "response_hyp": {
                "response": matched_response
            },
            "response_stochastic": stochastic_records
        }

        # 5. Save complete model to output directory with same filename
        output_file_path = output_dir / file_path.name
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(complete_model, ensure_ascii=False) + "\n")

        print(f"Saved complete model to {output_file_path.name}")

    print("Processing complete!")


if __name__ == "__main__":
    join_datasets()
