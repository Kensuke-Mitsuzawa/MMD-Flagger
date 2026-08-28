import hashlib
import json
from pathlib import Path
from typing import List

from llm_generation import MistralLLM, TemperatureResponses
from tqdm import tqdm


def main():
    dataset_dir = Path("/root/vast-ai-setup/MMD-Flagger/dataset/RAGTruth")
    mini_models_path = dataset_dir / "summary_mini_models.jsonl"
    already_used_source_id = "11318"

    # 1. Select 10 prompt-response pairs for mistral-7b-instruct (skipping source_id 11318)
    selected_records = []
    with open(mini_models_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            model_name = obj["response"].get("model", "")
            sid = obj["source"]["source_id"]
            if model_name.lower() == "mistral-7b-instruct" and sid != already_used_source_id:
                selected_records.append(obj)
                if len(selected_records) == 10:
                    break

    print(f"Selected {len(selected_records)} prompt-response pairs to process.")

    # 2. Load model once
    model_llm = MistralLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit=True,
    )

    set_temp = [round(0.1 * i, 1) for i in range(1, 11)]
    n_sampling = 5
    max_new_tokens = 256

    # 3. Process each selected record
    for idx, record in enumerate(selected_records, 1):
        prompt_text = record["source"]["prompt"]
        source_id = record["source"]["source_id"]
        response_id = record["response"]["id"]
        count_hallucination = record.get("count_hallucination", 0)

        prompt_hash = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()
        output_file = (
            dataset_dir
            / f"mistral_stochastic_sampling_h{count_hallucination}_{prompt_hash}.jsonl"
        )

        print(
            f"\n--- [{idx}/{len(selected_records)}] Processing Source ID {source_id} (h={count_hallucination}, hash={prompt_hash}) ---"
        )

        if output_file.exists():
            print(f"Output file {output_file.name} already exists. Skipping.")
            continue

        tensor_token_ids = model_llm.encode_token_ids(prompt_text)
        output_lines: List[TemperatureResponses] = []

        for _temp in tqdm(
            set_temp, desc=f"Prompt {idx}/{len(selected_records)} Temps"
        ):
            _seq_response = model_llm.generate(
                input_ids=tensor_token_ids,
                temperature=_temp,
                n_sampling=n_sampling,
                max_new_tokens=max_new_tokens,
            )

            temp_responses = TemperatureResponses(
                temperature=_temp,
                responses=[res.response for res in _seq_response],
                prompt=prompt_text,
            )
            output_lines.append(temp_responses)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in output_lines:
                f.write(item.model_dump_json() + "\n")

        print(f"Successfully saved {len(output_lines)} lines to {output_file.name}")


if __name__ == "__main__":
    main()
