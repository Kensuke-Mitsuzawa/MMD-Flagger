import typing as ty
import torch
import logging
import tqdm
import dataclasses
import json
import sys
import random
from pathlib import Path

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BatchEncoding
from transformers.generation.utils import GenerateEncoderDecoderOutput
from torch.nn.functional import log_softmax
import torch

import GPUtil

from .module_base import BaseTranslationModelHandler

from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...exceptions import ParameterSettingException


module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())


class GeneratedTranslationObject(ty.NamedTuple):
    translation_text: str
    tensor_translation_tokens: ty.Optional[torch.Tensor]
    log_score: ty.Optional[float]


class TransformersTranslationModelHandler(BaseTranslationModelHandler):
    def __init__(self,
                 src_lang: str,
                 target_lang: str,
                 model_name: str = "facebook/nllb-200-distilled-600M",
                 sampling_topk: int = -1,
                 sampling_topp: float = -1.0,     
                 max_len_a: float = 0.0,
                 max_len_b: int = 200,
                 random_seed: int = -1,
                 batch_size: int = 5,
                 is_select_gpu_flexible: bool = True, 
                 ):
        """A class for handling the Fairseq translation model.
        
        Args:
            random_seed: A random seed for the FairSeq Call. 
                If -1, the random seed is set randomly.
        """
        # loading the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.target_lang = target_lang

        # assert batch_size < n_sampling, f"batch_size={batch_size} should be less than n_sampling={n_sampling}"
        # self.n_sampling = n_sampling
        self.batch_size = batch_size

        # self.is_sampling = is_sampling

        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

        # parameters for the generation. See the following link.
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

        self.random_seed = random_seed

        # Check the device
        self.is_select_gpu_flexible = is_select_gpu_flexible
        if torch.cuda.is_available():
            # multiple GPUs
            if self.is_select_gpu_flexible:
                # select the less busy GPU
                device = torch.device(f"cuda:{self._get_less_busy_cuda_device()}")
            else:
                device = torch.device("cuda:0")
            # end if
        else:
            device = torch.device("cpu")
        # end if
        module_logger.debug(f"Device: {device}")
        self.device = device
        self.model.to(device)


    def _load_tokenizer(self, model_name: str, translation_source: str) -> AutoModelForSeq2SeqLM:
        """loading the tokenizer. The tokenizer requires setting the source language."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=translation_source)
        assert isinstance(tokenizer, AutoModelForSeq2SeqLM), f"Unexpected type of the tokenizer: {type(tokenizer)}"
        
        return tokenizer

    def _calculate_max_length(self, tensor_source_tokens: BatchEncoding) -> int:
        """Calculating the max token size that I make a request to the model."""
        input_token_length = len(tensor_source_tokens["input_ids"][0])
        max_tokens_output = int((self.max_len_a * input_token_length) + self.max_len_b)
        
        return max_tokens_output

    @staticmethod
    def _calculate_log_probability(logits_scores: ty.Tuple, generated_tokens: torch.Tensor) -> ty.List[float]:
        log_probabilities = []
        for i in range(generated_tokens.shape[0]):  # Iterate through each sequence in the batch
            sequence_log_prob = 0.0
            for step, logits in enumerate(logits_scores):
                # Get the logits for the current sequence in the batch
                current_logits = logits[i, :]  # Shape: (vocab_size,)

                # Get the token ID generated at this step (offset by 1 as scores predict the next token)
                if step < generated_tokens.shape[1] - 1:
                    predicted_token_id = generated_tokens[i, step + 1]

                    # Calculate log probability of the predicted token
                    log_prob = log_softmax(current_logits, dim=-1)[predicted_token_id].item()
                    sequence_log_prob += log_prob
                else:
                    break  # Stop if we've gone through all predicted tokens

            log_probabilities.append(sequence_log_prob)
        # end for
        return log_probabilities

    # ------------------------------
    # Sampling

    def _sampling_multi_input(self,
                              tensor_source_tokens: BatchEncoding,
                              n_sampling: int,
                              temperature: float,
                              target_lang_id: int,
                              random_seed: int = -1) -> ty.List[GeneratedTranslationObject]:
        # List to store all generated sequences
        seq_generated_tokens: ty.List[torch.Tensor] = []
        seq_log_prob: ty.List[ty.Optional[float]] = []

        # Generation parameters
        max_token_length = self._calculate_max_length(tensor_source_tokens)

        # Generation parameters
        generation_kwargs = {
            "forced_bos_token_id": target_lang_id,
            "do_sample": True,
            "temperature": temperature,
            "top_k": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            # "length_penalty": 1.0,
            # "no_repeat_ngram_size": 4,
            "min_length": 1,
            "max_length": max_token_length,
            "output_scores": True,  # if True, then the method returns a probability tensor.
            "output_logits": False,  # if True, then the method returns a logits tensor.
            "return_dict_in_generate": True,
            "num_return_sequences": self.batch_size,
        }

        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(random_seed)
        assert n_sampling < 10000, f"n_sampling={n_sampling} should be less than 10000."        
        seq_random_seed_values = _gen_random.sample(list(range(0, 9999)), k=n_sampling)

        for i in range(0, n_sampling, self.batch_size):
            module_logger.debug(f"Sampling {i} to {i + self.batch_size} / {n_sampling}")

            _current_batch_size = min(self.batch_size, n_sampling - i)
            
            _random_seed = seq_random_seed_values[i]

            with torch.random.fork_rng():
                if self.random_seed != -1:
                    torch.manual_seed(_random_seed)
                    torch.cuda.manual_seed_all(_random_seed)  # if you are using multi-GPU.
                else:
                    seed = random.randint(0, 2**32 - 1)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
                # end if

                tensor_source_tokens = tensor_source_tokens.to(self.model.device)

                # Create a batch of input token tensors by repeating the input
                assert isinstance(tensor_source_tokens["input_ids"], torch.Tensor), "Input IDs should be a torch.Tensor."
                assert isinstance(tensor_source_tokens["attention_mask"], torch.Tensor), "Attention mask should be a torch.Tensor."
                # _batch_input_ids = tensor_source_tokens["input_ids"].repeat(_current_batch_size, 1)
                # _batch_attention_mask = tensor_source_tokens["attention_mask"].repeat(_current_batch_size, 1)
                _batch_inputs = {
                    "input_ids": tensor_source_tokens["input_ids"],
                    "attention_mask": tensor_source_tokens["attention_mask"],
                }
                
                outputs: GenerateEncoderDecoderOutput = self.model.generate(**_batch_inputs, **generation_kwargs)
                
                # Extend the list of all generated sequences
                assert isinstance(outputs.sequences, torch.Tensor), "Generated sequences should be a torch.Tensor."
                # assert outputs.sequences.shape == (_current_batch_size, None), \
                #     f"Unexpected shape of generated sequences: {outputs.sequences.shape}"
                # moving into CPU.
                _tensor_sequence_cpu = outputs.sequences.cpu()
                seq_generated_tokens.extend(_tensor_sequence_cpu)
                
                if outputs.scores is not None:
                    _seq_log_prob = self._calculate_log_probability(outputs.scores, _tensor_sequence_cpu)
                else:
                    _seq_log_prob = [None] * _current_batch_size
                # end if
                seq_log_prob.extend(_seq_log_prob)
        # end with
        # end for

        # Decode all generated sequences
        seq_translations = self.tokenizer.batch_decode(seq_generated_tokens, skip_special_tokens=True)

        assert len(seq_translations) == len(seq_generated_tokens), \
            f"Length mismatch: {len(seq_translations)} != {len(seq_generated_tokens)}"
        assert len(seq_translations) == len(seq_log_prob), \
            f"Length mismatch: {len(seq_translations)} != {len(seq_log_prob)}"
        
        seq_translation_obj = []
        for _translation, _tensor_output, _log_prob in zip(seq_translations, seq_generated_tokens, seq_log_prob):
            module_logger.debug(f"Translation: {_translation}, Log Probability: {_log_prob}")
            _translation_obj = GeneratedTranslationObject(
                translation_text=_translation,
                tensor_translation_tokens=_tensor_output,
                log_score=_log_prob
            )
            seq_translation_obj.append(_translation_obj)
        # end for

        return seq_translation_obj

    def _sampling_single_input(self,
                               tensor_source_tokens: BatchEncoding,
                               temperature: float,
                               target_lang_id: int,
                               n_sampling: int,
                               n_max_attempts: int = 10,
                               random_seed: int = -1) -> ty.List[GeneratedTranslationObject]:
        # Generation parameters
        max_token_length = self._calculate_max_length(tensor_source_tokens)

        generation_kwargs = {
            "forced_bos_token_id": target_lang_id,
            "do_sample": True,
            "temperature": temperature,
            "top_k": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            # "length_penalty": 1.0,
            # "no_repeat_ngram_size": 4,
            "min_length": 1,
            "max_length": max_token_length,
            "output_scores": True,
            "output_logits": False,
            "return_dict_in_generate": True,
        }

        output_stack = []
        i_error_attempt = 0

        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(random_seed)
        assert n_sampling < 10000, f"n_sampling={n_sampling} should be less than 10000."
        seq_random_seed_values = _gen_random.sample(list(range(0, 9999)), k=n_sampling)

        __index_sampling = 0

        while len(output_stack) < n_sampling:
            with torch.random.fork_rng():
                _random_seed = seq_random_seed_values[__index_sampling]
                if random_seed != -1:
                    torch.manual_seed(_random_seed)
                    torch.cuda.manual_seed_all(_random_seed)  # if you are using multi-GPU.
                else:
                    seed = random.randint(0, 2**32 - 1)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
                # end if
                
                tensor_source_tokens = tensor_source_tokens.to(self.model.device)

                try:
                    # outputs = self.model.generate(**tensor_source_tokens, **generation_kwargs)
                    outputs = self.model.generate(**tensor_source_tokens, **generation_kwargs)
                except (AssertionError, RuntimeError) as e:
                    if i_error_attempt >= n_max_attempts:
                        error_message = (
                                f"Exceeded the maximum number of attempts: {n_max_attempts}",
                                f"Exception: {e}",
                                f"With the temperature paramater = {temperature}",
                                f"Source Text: {tensor_source_tokens}"
                        )
                        module_logger.error(error_message)
                        raise ParameterSettingException(error_message)
                    else:
                        i_error_attempt += 1
                        continue
                    # end if
                else:
                    # Access the generated token IDs (sequences)
                    generated_token_ids = outputs.sequences.cpu()
                    translated_text = self.tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)

                    if outputs.scores is not None:
                        log_prob = self._calculate_log_probability(outputs.scores, generated_token_ids)[0]
                    else:
                        log_prob = None
                    # end if

                    output_stack.append(GeneratedTranslationObject(
                        translation_text=translated_text,
                        tensor_translation_tokens=generated_token_ids,
                        log_score=log_prob))
                # end try
            # end with
            __index_sampling += 1
        # end while

        # TODO: check the output object. Can I obtain the log-sequence??
        return output_stack
    # end def
    
    def _call_interface(self, 
                        tensor_source_tokens: BatchEncoding,
                        temperature: float,
                        target_lang_id: int,
                        n_sampling: int,
                        is_sampling_in_iteration: bool = False,
                        is_auto_recovery_sampling: bool = True,
                        n_max_attempts: int = 100) -> ty.List[GeneratedTranslationObject]:
        """Simply, I call the fairseq interface to generate translations.
        This interface has dedicated procedures for calling the fairseq translation model since the interface often causes assertion errors when the temperature is a small value.
        See the description of `is_auto_recovery_sampling` for the details.

        Args:
            is_auto_recovery_sampling: If True, the function tries to recover the sampling process when the assertion error occurs.
                It switches the sampling method to the iteration-based sampling automatically when this method encounters the assertion error.
            n_max_attempts: The maximum number of attempts to recover the sampling process.
                When the attemtps exceed this value, the function raises an exception.
        """
        with torch.no_grad():
            if is_sampling_in_iteration:
                output_stack = self._sampling_single_input(
                    tensor_source_tokens=tensor_source_tokens,
                    temperature=temperature,
                    target_lang_id=target_lang_id,
                    n_sampling=n_sampling,
                    n_max_attempts=n_max_attempts,
                    random_seed=self.random_seed)
            else:
                try:
                    output_stack = self._sampling_multi_input(
                        tensor_source_tokens=tensor_source_tokens,
                        n_sampling=n_sampling,
                        temperature=temperature,
                        target_lang_id=target_lang_id,
                        random_seed=self.random_seed)
                except (AssertionError, RuntimeError) as e:
                    if is_auto_recovery_sampling:
                        module_logger.warning(f"Assertion error occurred: {e}")
                        output_stack = self._sampling_single_input(
                            tensor_source_tokens=tensor_source_tokens,
                            temperature=temperature,
                            target_lang_id=target_lang_id,
                            n_sampling=n_sampling,
                            n_max_attempts=n_max_attempts,
                            random_seed=self.random_seed)
                    else:
                        raise e
                    # end if
                # end try-except
            # end if
        # end with

        return output_stack

    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]
    
    # ------------------------------
    # Public Interface

    def translate_beam_search(self,
                              input_text: str,
                              temperature: float
                              ) -> GeneratedTranslationObject:
        inputs = self.tokenizer(input_text, return_tensors="pt")

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        assert isinstance(tgt_lang_id, int), f"Unexpected type of the target language ID: {type(tgt_lang_id)}"

        max_length = self._calculate_max_length(inputs)
        # Generation parameters
        generation_kwargs = {
            "forced_bos_token_id": tgt_lang_id,
            "num_beams": 5,
            # "length_penalty": 1.0,
            # "no_repeat_ngram_size": 4,
            "temperature": temperature,
            "min_length": 1,
            "max_length": max_length,
            "output_scores": True,
            "output_logits": False,
            "return_dict_in_generate": True,
        }
        # Generate translation
        with torch.no_grad():
            if torch.cuda.is_available():
                # multiple GPUs
                if self.is_select_gpu_flexible:
                    # select the less busy GPU
                    device = torch.device(f"cuda:{self._get_less_busy_cuda_device()}")
                else:
                    device = torch.device("cuda:0")
                # end if
            else:
                device = torch.device("cpu")
            # end if
            model = self.model.to(device)
            inputs = inputs.to(device)
            
            outputs = model.generate(**inputs, **generation_kwargs)

        if isinstance(outputs, torch.Tensor):
            translated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            _log_score = None
            tensor_token_ids = outputs.cpu()[0]
        else:
            # Access the generated token IDs (sequences)
            generated_token_ids = outputs.sequences.cpu()
            translated_text = self.tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
            if outputs.scores is not None:
                _log_score = self._calculate_log_probability(outputs.scores, generated_token_ids)[0]
            else:
                _log_score = None
            # end if
            tensor_token_ids = generated_token_ids[0]
        # end if

        module_logger.debug(f"Translated: {translated_text}, Log Score: {_log_score}")
        score = GeneratedTranslationObject(
            translation_text=translated_text,
            tensor_translation_tokens=tensor_token_ids,
            log_score=_log_score
        )
        return score


    def sample_multiple_times(
            self,
            input_text: str,
            n_sampling: int,
            temperature: float,
            is_sampling_in_iteration: bool = False,
            n_max_attempts: int = 100) -> ty.List[GeneratedTranslationObject]:
        """A custom function to sample multiple times with the same input text.
        This function conducts tokenization just one time.
        
        Args:
            is_sampling_in_iteration: If True, the sampling is executed in the iteration.
                This flag is for saving the RAM or GPU memory.
                However, the execution speed will be slower.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt")

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        assert isinstance(tgt_lang_id, int), f"Unexpected type of the target language ID: {type(tgt_lang_id)}"

        output_stack = self._call_interface(
            tensor_source_tokens=inputs,
            temperature=temperature,
            n_sampling=n_sampling,
            target_lang_id=tgt_lang_id,
            is_sampling_in_iteration=is_sampling_in_iteration,
            n_max_attempts=n_max_attempts)
        
        return output_stack
