import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import torch
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Response(BaseModel):
    temperature: float
    response: str


class TemperatureResponses(BaseModel):
    temperature: float
    responses: List[str]
    prompt: str


class MistralLLM:
    """Wrapper for loading Mistral-7B-Instruct with 4-bit quantization and performing temperature-only stochastic sampling."""

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map

        print(f"Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model {self.model_id} (4-bit quantization={load_in_4bit})...")
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device_map,
                trust_remote_code=True,
            )
        self.model.eval()

    def encode_token_ids(
        self, prompt: str, return_tensors: str = "pt"
    ) -> torch.Tensor:
        """Tokenize and encode prompt string into tensor token IDs."""
        inputs = self.tokenizer(
            prompt, return_tensors=return_tensors, padding=True
        )
        return inputs.input_ids.to(self.model.device)

    def generate(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        n_sampling: int = 5,
        max_new_tokens: int = 256,
    ) -> List[Response]:
        """Perform stochastic sampling using ONLY temperature hyperparameter (no top-k, no top-p)."""
        # Ensure temperature-only sampling
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=temperature,
                top_k=0,  # disable top-k filtering
                top_p=1.0,  # disable top-p (nucleus) filtering
                num_return_sequences=n_sampling,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Slice generated tokens past input prompt length
        prompt_len = input_ids.shape[-1]
        generated_tokens = output_ids[:, prompt_len:]

        decoded_texts = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        seq_responses = [
            Response(temperature=temperature, response=text)
            for text in decoded_texts
        ]
        return seq_responses


def run_stochastic_sampling(
    prompt: str,
    output_file: Optional[Union[str, Path]] = None,
    set_temp: Optional[List[float]] = None,
    n_sampling: int = 5,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens: int = 256,
    load_in_4bit: bool = True,
) -> List[TemperatureResponses]:
    """Execute stochastic sampling over temperature range [0.1, 1.0] repeated n_sampling times per temperature."""
    if set_temp is None:
        set_temp = [round(0.1 * i, 1) for i in range(1, 11)]

    # Load LLM model
    model_llm = MistralLLM(model_id=model_id, load_in_4bit=load_in_4bit)

    # Encode token ids
    tensor_token_ids = model_llm.encode_token_ids(prompt)

    output_lines: List[TemperatureResponses] = []

    for _temp in tqdm(set_temp, desc="Stochastic Sampling Temperatures"):
        _seq_response: List[Response] = model_llm.generate(
            input_ids=tensor_token_ids,
            temperature=_temp,
            n_sampling=n_sampling,
            max_new_tokens=max_new_tokens,
        )

        temp_responses = TemperatureResponses(
            temperature=_temp,
            responses=[res.response for res in _seq_response],
            prompt=prompt,
        )
        output_lines.append(temp_responses)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in output_lines:
                f.write(item.model_dump_json() + "\n")
        print(f"Saved results to {output_path}")

    return output_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run temperature-only stochastic sampling using Mistral-7B-Instruct with 4-bit quantization."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Summarize the key benefits of solar energy in 3 bullet points.",
        help="Input prompt for generation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="stochastic_sampling_output.jsonl",
        help="Output jsonl file path.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--n_sampling",
        type=int,
        default=5,
        help="Number of samples per temperature.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization.",
    )

    args = parser.parse_args()

    run_stochastic_sampling(
        prompt=args.prompt,
        output_file=args.output_file,
        model_id=args.model_id,
        n_sampling=args.n_sampling,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=not args.no_4bit,
    )
