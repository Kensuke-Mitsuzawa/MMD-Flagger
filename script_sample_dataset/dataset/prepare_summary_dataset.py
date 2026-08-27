import json
from pathlib import Path


def main():
    dataset_dir = Path("/root/vast-ai-setup/MMD-Flagger/dataset/RAGTruth")
    source_path = dataset_dir / "source_info.jsonl"
    response_path = dataset_dir / "response.jsonl"
    summary_path = dataset_dir / "summary.jsonl"
    mini_models_path = dataset_dir / "summary_mini_models.jsonl"

    # 1. Extract records from source_info.jsonl where task_type is Summary
    sources = {}
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("task_type") == "Summary":
                sources[data["source_id"]] = data

    print(f"Loaded {len(sources)} summarization source records.")

    # 2 & 3. Search matching response records and format
    output_records = []
    with open(response_path, "r", encoding="utf-8") as rf:
        for line in rf:
            if not line.strip():
                continue
            resp_data = json.loads(line)
            sid = resp_data.get("source_id")
            if sid in sources:
                src_data = sources[sid]
                out_obj = {
                    "count_hallucination": len(resp_data.get("labels", [])),
                    "source": src_data,
                    "response": resp_data,
                }
                output_records.append(out_obj)

    # 4. Sort records by count_hallucination in descending order
    output_records.sort(key=lambda x: x["count_hallucination"], reverse=True)

    # 5. Write full summary output (summary.jsonl)
    with open(summary_path, "w", encoding="utf-8") as wf:
        for out_obj in output_records:
            wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(
        f"Successfully created {summary_path} with {len(output_records)} records sorted by count_hallucination (descending)."
    )

    # 6. Filter for mini models (Llama-2-7B-chat and Mistral-7B-Instruct)
    target_models = {"llama-2-7b-chat", "mistral-7b-instruct"}
    mini_records = [
        obj for obj in output_records
        if obj["response"].get("model", "").lower() in target_models
    ]

    with open(mini_models_path, "w", encoding="utf-8") as wf:
        for out_obj in mini_records:
            wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(
        f"Successfully created {mini_models_path} with {len(mini_records)} records."
    )


if __name__ == "__main__":
    main()
