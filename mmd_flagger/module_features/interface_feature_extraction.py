import typing as ty
import torch
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM

from pydantic import BaseModel

from .models import GenerationInfoDict
from .module_feature_extraction import *

from ..model_objects import SampleSetContainer
from ..module_mmd_flagger.module_models import (
    SampleSet,
    SingleSample
)


class ResponseStochastic(BaseModel):
    temperature: float
    responses: ty.List[str]


class InterfaceFeatureExtraction(object):
    def __init__(
        self,
        tokenizer_target: ty.Optional[AutoTokenizer] = None,
        model_target: ty.Optional[AutoModelForCausalLM] = None,
        default_model_name_or_path: str = "sshleifer/tiny-gpt2"
    ):
        self.tokenizer_target = tokenizer_target
        self.model_target = model_target
        self.default_model_name_or_path = default_model_name_or_path

    def transform(
        self,
        prompt: str,
        response_y_hyp: str,
        responses_y_stoch_obj: ty.List[ResponseStochastic],
        extractors: ty.List[ty.Union[HiddenStatesExtractor, WordEmbeddingExtractor, LapEigvalsExtractor, AttentionEigenValsExtractor]]
    ) -> ty.List[SampleSetContainer]:
        """Extracting the feature based on the `run_llm_teacher_forcing`.

        Return: [SampleSetContainer]
        """
        stack_y_hyp: ty.List[ty.Tuple[str, str, torch.Tensor]] = []  # [(feature-name, response-text, torch.Tensor)]
        stack_y_sto: ty.List[ty.Tuple[float, str, torch.Tensor]] = []  # [(temp-param, feature-name, torch.Tensor)]

        for _ext in extractors:
            # extracting the feature from response_y_hyp.
            _generation_dict_y_hyp = self._run_llm_teacher_forcing(prompt, response_y_hyp)
            _feat_obj_h_hyp = _ext.extract(_generation_dict_y_hyp)
            
            for _f_h_hyp in _feat_obj_h_hyp:
                stack_y_hyp.append([_f_h_hyp.get_feature_name(), response_y_hyp, _f_h_hyp.get_feature_vector()])
            # end for

            # TODO: [(feature-name, response-text, torch.Tensor)] -> Aggregate by feature-name 

            # _sample_h_hyp = SampleSet(
            #     label='Y_hyp', decoding_config=None, temperature_parameter=None,
            #     samples=[SingleSample(sample_unique_id=0, text=response_y_hyp, feature_vector=_tensor_h_hyp)]
            #     )

            # [0.1, 0.2, ..., 1.0]
            # [feat_vec_0.1, feat_vec_0.2, ... feat_vec_1.0]

            # extracting the features from responses_y_stoch_obj.
            # _stack_local_text = []
            _stack_local_tensor = []
            for _res_sto in responses_y_stoch_obj:
                _res_sto.temperature
                
                for _res_text in _res_sto.responses:
                    _generation_dict_h_sto = self._run_llm_teacher_forcing(prompt, _res_text)
                    _feat_obj_h_sto = _ext.extract(_generation_dict_h_sto)
                    
                    for _f_h_sto in _feat_obj_h_sto:
                        _stack_local_tensor.append([_f_h_sto.get_feature_name(), _f_h_sto.get_feature_vector()])
                    # end for
                # end for

                # [(feat-name, Tensor)] -> Agg-by-feat-name -> [(feat-name, Tensor)]. Tensor-length is the length-of-response.
                # TODO: make the aggregation code.
                _seq_feature_name2tensor = [[]]  # TODO: have to implement. [(feat-name, tensor)]. The Tensor-size should be (N-response, N-dim).

                for (_f_name, _tensor) in _stack_local_tensor:
                    stack_y_sto.append([_res_sto.temperature, _f_name, _tensor])                
                # end for
            # end for
        # end for

        # --------
        # Aggregation at the end.

        seq_sample_set_container = []

        # TODO: we have to form the following data structure.
        # {feat-name: (tensor-y-hyp, [(temp, tensor_y_stoch)])}
        
        # TODO: we have to set the following data object.
        SampleSetContainer(feature_name=)

        return seq_sample_set_container

    def _run_llm_teacher_forcing(
        self, 
        prompt_text: str,
        response_text: str,
        tokenizer_target: ty.Optional[AutoTokenizer] = None,
        model_target: ty.Optional[AutoModelForCausalLM] = None
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
