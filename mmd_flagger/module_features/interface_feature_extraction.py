import typing as ty
import torch
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM

from pydantic import BaseModel

from .models import GenerationInfoDict
from .module_feature_extraction import *

from ..model_objects import (
    SampleSetContainer,
    LLMResponseTextStochastic
)
from ..module_mmd_flagger.module_models.model_sample_set_container import (
    SampleSet,
    SingleSample,
    TemperatureParameter
)
from ..utils.llm_decoding_conf_models import DecodingConfig, DecodingStrategyName

DEFAULT_EXTRACTORS = [
    HiddenStatesExtractor(resolved_layer_ids=['middle']),
    WordEmbeddingExtractor()
]

class InterfaceFeatureExtraction(object):
    def __init__(
        self,
        tokenizer_target: AutoTokenizer,
        model_target: AutoModelForCausalLM,
        extractors: ty.List[ty.Union[HiddenStatesExtractor, WordEmbeddingExtractor, LapEigvalsExtractor, AttentionEigenValsExtractor]] = DEFAULT_EXTRACTORS,
    ):
        self.tokenizer_target = tokenizer_target
        self.model_target = model_target
        self.extractors = extractors

    def extract(
        self,
        prompt_text: str,
        response_text: str,
        tokenizer_target: ty.Optional[AutoTokenizer] = None,
        model_target: ty.Optional[AutoModelForCausalLM] = None
    ) -> GenerationInfoDict:
        return self._run_llm_teacher_forcing(
            prompt_text=prompt_text,
            response_text=response_text,
            tokenizer_target=tokenizer_target,
            model_target=model_target
        )

    def transform(
        self,
        prompt: str,
        response_y_hyp: str,
        responses_y_stoch_obj: ty.List[LLMResponseTextStochastic],
        extractors: ty.Optional[ty.List[ty.Union[HiddenStatesExtractor, WordEmbeddingExtractor, LapEigvalsExtractor, AttentionEigenValsExtractor]]] = None,
    ) -> ty.List[SampleSetContainer]:
        """Extracting the feature based on the `run_llm_teacher_forcing`.

        Return: [SampleSetContainer]
        """
        active_extractors = extractors if extractors is not None else self.extractors
        needs_attentions = any(
            isinstance(ext, (LapEigvalsExtractor, AttentionEigenValsExtractor))
            for ext in active_extractors
        )

        # 1. Extract feature objects for hypothesis response
        gen_dict_hyp = self._run_llm_teacher_forcing(prompt, response_y_hyp, output_attentions=needs_attentions)
        hyp_samples_by_feat: ty.Dict[str, SingleSample] = {}

        for ext in active_extractors:
            feat_objs_hyp = ext.extract(gen_dict_hyp)
            for f_hyp in feat_objs_hyp:
                f_name = f_hyp.get_feature_name()
                f_vec = f_hyp.get_feature_vector()
                sample_id = hashlib.sha256(f"hyp_{f_name}_{response_y_hyp}".encode("utf-8")).hexdigest()
                hyp_samples_by_feat[f_name] = SingleSample(
                    sample_unique_id=sample_id,
                    text=response_y_hyp,
                    feature_vector=f_vec
                )

        # 2. Extract feature objects for stochastic responses
        # Map feature_name -> list of SampleSet (one per temperature)
        stoch_sample_sets_by_feat: ty.Dict[str, ty.List[SampleSet]] = {
            f_name: [] for f_name in hyp_samples_by_feat.keys()
        }

        for res_sto in responses_y_stoch_obj:
            temp = res_sto.temperature
            samples_this_temp_by_feat: ty.Dict[str, ty.List[SingleSample]] = {
                f_name: [] for f_name in hyp_samples_by_feat.keys()
            }

            for idx, res_text in enumerate(res_sto.responses):
                gen_dict_sto = self._run_llm_teacher_forcing(prompt, res_text, output_attentions=needs_attentions)

                for ext in active_extractors:
                    feat_objs_sto = ext.extract(gen_dict_sto)
                    for f_sto in feat_objs_sto:
                        f_name = f_sto.get_feature_name()
                        f_vec = f_sto.get_feature_vector()
                        sample_id = hashlib.sha256(f"sto_{f_name}_{temp}_{idx}_{res_text}".encode("utf-8")).hexdigest()
                        single_sample = SingleSample(
                            sample_unique_id=sample_id,
                            text=res_text,
                            feature_vector=f_vec
                        )
                        if f_name not in samples_this_temp_by_feat:
                            samples_this_temp_by_feat[f_name] = []
                        samples_this_temp_by_feat[f_name].append(single_sample)

            dec_config = DecodingConfig(
                method=DecodingStrategyName.Stochastic,
                temperature=temp
            )
            temp_param = TemperatureParameter(temperature_parameter=temp)

            for f_name, sample_list in samples_this_temp_by_feat.items():
                if sample_list:
                    sto_sample_set = SampleSet(
                        label="Y_sto",
                        decoding_config=dec_config,
                        temperature_parameter=temp_param,
                        samples=sample_list
                    )
                    if f_name not in stoch_sample_sets_by_feat:
                        stoch_sample_sets_by_feat[f_name] = []
                    stoch_sample_sets_by_feat[f_name].append(sto_sample_set)

        # 3. Construct SampleSetContainer for each feature
        seq_sample_set_container: ty.List[SampleSetContainer] = []
        for f_name, hyp_sample in hyp_samples_by_feat.items():
            #
            sample_y_hyp = SampleSet(
                label="Y_hyp",
                decoding_config=None,
                temperature_parameter=None,
                samples=[hyp_sample, hyp_sample]
            )
            sample_set_y_stoch = stoch_sample_sets_by_feat.get(f_name, [])
            container = SampleSetContainer(
                feature_name=f_name,
                sample_y_hyp=sample_y_hyp,
                sample_set_y_stoch=sample_set_y_stoch
            )
            seq_sample_set_container.append(container)

        return seq_sample_set_container

    def _run_llm_teacher_forcing(
        self, 
        prompt_text: str,
        response_text: str,
        tokenizer_target: ty.Optional[AutoTokenizer] = None,
        model_target: ty.Optional[AutoModelForCausalLM] = None,
        output_attentions: ty.Optional[bool] = None,
    ) -> GenerationInfoDict:
        """Extracting features from the target model using the teacher-forcing mode."""
        tokenizer = tokenizer_target or self.tokenizer_target
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.default_model_name_or_path)

        model = model_target or self.model_target
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(self.default_model_name_or_path, attn_implementation="eager")
        # end if

        device_model = model.device

        # Encode prompt
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_input_ids= prompt_input_ids.to(device_model)

        # Encode response
        response_inputs = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)
        generated_token_ids = response_inputs["input_ids"]
        generated_token_ids = generated_token_ids.to(device_model)

        # Concatenate for teacher-forcing forward pass
        full_input_ids = torch.cat([prompt_input_ids, generated_token_ids], dim=1)

        if output_attentions is None:
            needs_attentions = any(
                isinstance(ext, (LapEigvalsExtractor, AttentionEigenValsExtractor))
                for ext in self.extractors
            ) or True
        else:
            needs_attentions = output_attentions

        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = model(
                input_ids=full_input_ids,
                output_hidden_states=True,
                output_attentions=needs_attentions,
                return_dict=True
            )

        layer_hidden_states = {
            i: h.cpu().squeeze(0) for i, h in enumerate(outputs.hidden_states)
        }

        if outputs.attentions is not None:
            attention_matrix = torch.stack([att.squeeze(0).cpu() for att in outputs.attentions], dim=0)
        else:
            attention_matrix = torch.empty((0, 0, 0, 0))

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
