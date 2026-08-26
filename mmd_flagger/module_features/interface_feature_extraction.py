from typing import Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .models import GenerationInfoDict


class InterfaceFeatureExtraction(object):
    def __init__(
        self,
        tokenizer_target: Optional[AutoTokenizer] = None,
        model_target: Optional[AutoModelForCausalLM] = None,
        default_model_name_or_path: str = "sshleifer/tiny-gpt2"
    ):
        self.tokenizer_target = tokenizer_target
        self.model_target = model_target
        self.default_model_name_or_path = default_model_name_or_path

    def extract(
        self, 
        prompt_text: str,
        response_text: str,
        tokenizer_target: Optional[AutoTokenizer] = None,
        model_target: Optional[AutoModelForCausalLM] = None
    ) -> GenerationInfoDict:
        """Extracting features from the target model using the teacher-forcing mode.
        The premise is that a pair of (prompt, generation-response) is given.
        The `generation-response` is the generated text by the target model.

        Args:
            prompt_text (str): The prompt text.
            response_text (str): The generated text.
            tokenizer_target (Optional[AutoTokenizer]): The tokenizer of the target model.
            model_target (Optional[AutoModelForCausalLM]): The model of the target model.

        Returns:
            GenerationInfoDict: Container with extracted internal states.
        """
        tokenizer = tokenizer_target or self.tokenizer_target
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.default_model_name_or_path)

        model = model_target or self.model_target
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(self.default_model_name_or_path, attn_implementation="eager")

        # Encode prompt
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_input_ids = prompt_inputs["input_ids"]

        # Encode response
        response_inputs = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)
        generated_token_ids = response_inputs["input_ids"]

        # Concatenate for teacher-forcing forward pass
        full_input_ids = torch.cat([prompt_input_ids, generated_token_ids], dim=1)

        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=full_input_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )

        layer_hidden_states = {
            i: h.cpu().squeeze(0) for i, h in enumerate(outputs.hidden_states)
        }


        attention_matrix = torch.stack([att.squeeze(0) for att in outputs.attentions], dim=0)
        attention_matrix = attention_matrix.cpu()

        prompt_ids_1d = prompt_input_ids.squeeze(0)
        generated_ids_1d = generated_token_ids.squeeze(0)

        input_embeddings = model.get_input_embeddings()
        word_embeddings_generated_tokens = input_embeddings(generated_ids_1d)
        # removing the gradient computation
        word_embeddings_generated_tokens = word_embeddings_generated_tokens.detach().cpu()

        generation_info = GenerationInfoDict(
            prompt_input_ids=prompt_ids_1d,
            generated_token_ids=generated_ids_1d,
            layer_hidden_states=layer_hidden_states,
            attention_matrix=attention_matrix,
            word_embeddings_generated_tokens=word_embeddings_generated_tokens,
            random_seed=None,
            batch_size=1,
            repetition_penalty=None
        )

        return generation_info