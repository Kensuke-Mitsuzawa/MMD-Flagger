import typing as ty
import logging
import re
import random
import copy
from pathlib import Path

import torch
import numpy as np
from torch.nn.functional import log_softmax

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BatchEncoding
from transformers.models.m2m_100.modeling_m2m_100 import M2M100ScaledWordEmbedding

from .module_base import (
    BaseTranslationModelHandlerVer2,
    TranslationResultContainer,
    EvaluationTargetTranslationPair)

from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...exceptions import ParameterSettingException
from ...logger_module import formatter

module_logger = logging.getLogger(__name__)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
module_logger.addHandler(handler)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())



class TransformersTranslationModelHandlerVer2(BaseTranslationModelHandlerVer2):
    def __init__(self,
                 model_name: str,
                 src_lang: str,
                 target_lang: str,
                 random_seed: int = 42,
                 path_cache_dir: ty.Optional[Path] = None,
                 is_use_cache: bool = True,
                 is_select_gpu_flexible: bool = True,
                 is_save_convert_float16: bool = False):
        super().__init__(path_cache_dir=path_cache_dir, is_use_cache=is_use_cache, is_save_convert_float16=is_save_convert_float16)

        # loading the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.tokenizer_target_language = AutoTokenizer.from_pretrained(model_name, src_lang=target_lang)        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.target_lang = target_lang
        self.src_lang = src_lang

        # ------------------------------------------------------
        # allocating the model to a device
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
        # ------------------------------------------------------        
        # attributes
        self.random_seed: int = random_seed


    # -------------------------------------------------------------
    # Private Methods

    def _get_word_embedding_decoder(self,
                                    tensor_decoder_token_id: torch.Tensor) -> torch.Tensor:
        decoder_embedding_layer: M2M100ScaledWordEmbedding = self.model.model.decoder.embed_tokens
        with torch.no_grad():
            decoder_embeddings = decoder_embedding_layer(tensor_decoder_token_id.to(self.model.device))
        # end with
        return decoder_embeddings.cpu()

    def __calculate_max_length(self, 
                              tensor_source_tokens: BatchEncoding,
                              max_len_a: float,
                              max_len_b: int) -> int:
        """Calculating the max token size that I make a request to the model."""
        if len(tensor_source_tokens["input_ids"]) == 1:
            input_token_length = len(tensor_source_tokens["input_ids"][0])
        else:
            input_token_length = len(tensor_source_tokens["input_ids"])
        # end if
        
        max_tokens_output = int((max_len_a * input_token_length) + max_len_b)
        
        return max_tokens_output

    # -------------------------------------------------------------
    # Semi Private Methods

    # Sampling

    def _sampling_multi_input(self,
                              source_text: str,
                              tensor_source_tokens: BatchEncoding,
                              n_sampling: int,
                              temperature: float,
                              max_len_a: float,
                              max_len_b: int,
                              batch_size: int = 5,
                              target_layers_extraction: ty.Optional[ty.List[str]] = None
                              ) -> ty.List[TranslationResultContainer]:
        # Generation parameters
        max_token_length = self.__calculate_max_length(tensor_source_tokens,
                                                       max_len_a=max_len_a,
                                                       max_len_b=max_len_b)

        target_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)

        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(self.random_seed)
        assert n_sampling < 10000, f"n_sampling={n_sampling} should be less than 10000."        
        seq_random_seed_values = _gen_random.sample(list(range(0, 9999)), k=n_sampling)
        
        name_layer_word_embedding = self._get_decoder_word_embedding_layer_name()        
        
        output_stack = []
        for i in range(0, n_sampling, batch_size):
            module_logger.debug(f"Sampling {i} to {i + batch_size} / {n_sampling}")

            _current_batch_size = min(batch_size, n_sampling - i)

            # Generation parameters
            generation_kwargs = {
                # "length_penalty": 1.0,
                # "no_repeat_ngram_size": 4,                
                "forced_bos_token_id": target_lang_id,
                "do_sample": True,
                "temperature": temperature,
                "top_k": 0.0,
                "top_p": 1.0,
                "min_length": 1,
                "max_length": max_token_length,
                "output_scores": False,  # if True, then the method returns a probability tensor.
                "output_logits": False,  # if True, then the method returns a logits tensor.
                "return_dict_in_generate": True,
                "num_return_sequences": _current_batch_size,
                "output_hidden_states": True
            }

            _random_seed = seq_random_seed_values[i]

            with (torch.random.fork_rng(), torch.no_grad()):
                torch.manual_seed(_random_seed)
                torch.cuda.manual_seed_all(_random_seed)  # if you are using multi-GPU.

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
                
                _outputs: transformers.generation.utils.GenerateEncoderDecoderOutput = self.model.generate(**_batch_inputs, **generation_kwargs)
            # end with
            # Extend the list of all generated sequences
            assert isinstance(_outputs.sequences, torch.Tensor), "Generated sequences should be a torch.Tensor."
            assert _outputs.sequences.shape[0] == _current_batch_size, \
                f"Unexpected shape of generated sequences: {_outputs.sequences.shape}"
            # moving into CPU.
            _tensor_sequence_cpu = _outputs.sequences.cpu()
            
            # Decode all generated sequences
            seq_translations: ty.List[str] = self.tokenizer.batch_decode(_tensor_sequence_cpu, skip_special_tokens=True)

            # computing log probability score.
            if _outputs.scores is not None:
                _seq_log_prob = self._calculate_log_probability(_outputs.scores, _tensor_sequence_cpu)
            else:
                _seq_log_prob = [None] * _current_batch_size
            # end if

            # extracting hidden layers.
            if target_layers_extraction is None:
                seq_hidden_layer_obj = [None] * _current_batch_size
            else:
                seq_hidden_layer_obj = self._extract_hidden_vector(
                    tensor_translated_sequence=_tensor_sequence_cpu,
                    outputs=_outputs,
                    target_layers_extraction=target_layers_extraction,
                    is_batch_mode=True,
                    n_batch_size=_current_batch_size)
            # end if
            
            _tensor_translation: torch.Tensor
            _text_translation: str
            _score_log_prop: ty.Optional[float]
            _d_hidden_layer: ty.Optional[ty.Dict[str, torch.Tensor]]
            for _tensor_translation, _text_translation, _score_log_prop, _d_hidden_layer in zip(_outputs.sequences, seq_translations, _seq_log_prob, seq_hidden_layer_obj):
                _d_obj_kwargs= copy.deepcopy(generation_kwargs)
                _d_obj_kwargs['random_seed'] = _random_seed

                # ------------------------------------------------------------
                # getting the word embedding TODO
                # assert isinstance(_d_hidden_layer, dict)
                # _tensor_word_embedding = self._get_word_embedding_decoder(_tensor_translation)
                # assert len(_tensor_word_embedding.shape) == 2
                # assert _tensor_word_embedding.shape[0] == len(_tensor_translation)

                # _d_hidden_layer[name_layer_word_embedding] = _tensor_word_embedding
                # ------------------------------------------------------------

                _output_obj = TranslationResultContainer(
                    source_text=source_text,
                    translation_text=_text_translation,
                    source_language=self.src_lang,
                    target_language=self.target_lang,
                    source_tensor_tokens=tensor_source_tokens['input_ids'][0],
                    target_tensor_tokens=_tensor_translation,
                    log_probability_score=_score_log_prop,
                    dict_layer_embeddings=_d_hidden_layer,
                    argument_translation_conditions=generation_kwargs
                )
                output_stack.append(_output_obj)
            # end for
        # end with
        return output_stack

    def _sampling_single_input(self,
                               source_text: str,
                               tensor_source_tokens: BatchEncoding,
                               temperature: float,
                               n_sampling: int,
                               max_len_a: float,
                               max_len_b: int,                               
                               n_max_attempts: int = 10,
                               target_layers_extraction: ty.Optional[ty.List[str]] = None
                               ) -> ty.List[TranslationResultContainer]:
        # Generation parameters
        max_token_length = self.__calculate_max_length(tensor_source_tokens,
                                                       max_len_a=max_len_a,
                                                       max_len_b=max_len_b)
        
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)

        generation_kwargs = {
            "forced_bos_token_id": tgt_lang_id,
            "do_sample": True,
            "temperature": temperature,
            "top_k": 0.0,
            "top_p": 1.0,
            # "length_penalty": 1.0,
            # "no_repeat_ngram_size": 4,
            "min_length": 1,
            "max_length": max_token_length,
            "output_scores": True,
            "output_logits": False,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }

        output_stack = []
        i_error_attempt = 0

        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(self.random_seed)
        assert n_sampling < 10000, f"n_sampling={n_sampling} should be less than 10000."
        seq_random_seed_values = _gen_random.sample(list(range(0, 9999)), k=n_sampling)

        __index_sampling = 0

        name_layer_word_embedding = self._get_decoder_word_embedding_layer_name()

        while len(output_stack) < n_sampling:
            tensor_source_tokens = tensor_source_tokens.to(self.model.device)

            try:
                # I use the random seed manager and torch inference mode.
                with (torch.random.fork_rng(), torch.no_grad()):
                    _random_seed = seq_random_seed_values[__index_sampling]
                    torch.manual_seed(_random_seed)
                    torch.cuda.manual_seed_all(_random_seed)  # if you are using multi-GPU.

                    outputs = self.model.generate(**tensor_source_tokens, **generation_kwargs)
                # end with
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

                # ------------------------------------------------------------
                # hidden vector extraction
                if target_layers_extraction is not None:
                    d_hidden_layers = self._extract_hidden_vector(
                        tensor_translated_sequence=generated_token_ids,
                        outputs=outputs, 
                        target_layers_extraction=target_layers_extraction)
                else:
                    d_hidden_layers = {}
                # end if

                # ------------------------------------------------------------
                # getting the word embedding TODO
                # assert isinstance(d_hidden_layers, dict)
                # tensor_word_embedding = self._get_word_embedding_decoder(generated_token_ids[0])
                # assert len(tensor_word_embedding.shape) == 2
                # assert tensor_word_embedding.shape[0] == len(generated_token_ids[0])

                # d_hidden_layers[name_layer_word_embedding] = tensor_word_embedding
                # ------------------------------------------------------------

                _d_obj_kwargs= copy.deepcopy(generation_kwargs)
                _d_obj_kwargs['random_seed'] = _random_seed

                _output_obj = TranslationResultContainer(
                    source_text=source_text,
                    translation_text=translated_text,
                    source_language=self.src_lang,
                    target_language=self.target_lang,
                    source_tensor_tokens=tensor_source_tokens['input_ids'][0],
                    target_tensor_tokens=generated_token_ids[0],
                    log_probability_score=log_prob,
                    dict_layer_embeddings=d_hidden_layers,
                    argument_translation_conditions=generation_kwargs
                )
                output_stack.append(_output_obj)
            # end try
            __index_sampling += 1
        # end while

        # TODO: check the output object. Can I obtain the log-sequence??
        return output_stack
    # end def
    
    def _call_interface_sampling(self, 
                                 source_text: str,
                                    tensor_source_tokens: BatchEncoding,
                                    temperature: float,
                                    n_sampling: int,
                                    max_len_a: float,
                                    max_len_b: int,
                                    is_sampling_in_iteration: bool = False,
                                    is_auto_recovery_sampling: bool = True,
                                    n_max_attempts: int = 100,
                                    batch_size: int = 5,
                                    target_layers_extraction: ty.Optional[ty.List[str]] = None
                                    ) -> ty.List[TranslationResultContainer]:
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
                    source_text=source_text,
                    tensor_source_tokens=tensor_source_tokens,
                    temperature=temperature,
                    n_sampling=n_sampling,
                    n_max_attempts=n_max_attempts,
                    max_len_a=max_len_a,
                    max_len_b=max_len_b,
                    target_layers_extraction=target_layers_extraction)
            else:
                try:
                    output_stack = self._sampling_multi_input(
                        source_text=source_text,
                        tensor_source_tokens=tensor_source_tokens,
                        n_sampling=n_sampling,
                        temperature=temperature,
                        max_len_a=max_len_a,
                        max_len_b=max_len_b,
                        target_layers_extraction=target_layers_extraction,
                        batch_size=batch_size)
                except (AssertionError, RuntimeError) as e:
                    if is_auto_recovery_sampling:
                        module_logger.warning(f"Assertion error occurred: {e}")
                        output_stack = self._sampling_single_input(
                            source_text=source_text,
                            tensor_source_tokens=tensor_source_tokens,
                            temperature=temperature,
                            n_sampling=n_sampling,
                            n_max_attempts=n_max_attempts,
                            max_len_a=max_len_a,
                            max_len_b=max_len_b,
                            target_layers_extraction=target_layers_extraction)
                    else:
                        raise e
                    # end if
                # end try-except
            # end if
        # end with

        return output_stack

    # ----------------------------------------------------------------
    # utils

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
    
    def get_all_possible_layers(self,
                                prefix_name_encoder: str = 'encoder',
                                prefix_name_decoder: str = 'decoder') -> ty.Tuple[ty.List, ty.List]:
        """
        Return:
            - seq_encoder_layers: list of encoder layers, `transformers.models.m2m_100.modeling_m2m_100.M2M100EncoderLayer`
            - seq_decoder_layers: list of decoder layers, `transformers.models.m2m_100.modeling_m2m_100.M2M100DecoderLayer`
        """
        index_encoder_layers = self.model.config.encoder_layers
        index_decoder_layers = self.model.config.decoder_layers

        seq_encoder_layers = list(self.model.model.encoder.layers)
        seq_decoder_layers = list(self.model.model.decoder.layers)

        seq_encoder_name = []
        for _i_layer, _layer_obj in enumerate(seq_encoder_layers):
            seq_encoder_name.append(f'{prefix_name_encoder}.{_i_layer}')
        # end for
        seq_decoder_name = []
        for _i_layer, _layer_obj in enumerate(seq_decoder_layers):
            seq_decoder_name.append(f'{prefix_name_decoder}.{_i_layer}')                
        # end for

        return seq_encoder_name, seq_decoder_name

    # @staticmethod
    # def _parse_layer_sequence(target_layers_extraction: ty.List[str]) -> ty.List[ty.Tuple[str, int]]:
    #     # parsing the `target_layers_extraction`
    #     _seq_target_layers = [_target_decoder for _target_decoder in target_layers_extraction if 'decoder' in _target_decoder]
    #     pattern = re.compile(r'decoder.([0-9]+)')
    #     seq_layer_number = []
    #     for _target_layer_name in _seq_target_layers:
    #         _match_res = pattern.search(_target_layer_name)
    #         if _match_res is not None:
    #             _val = int(_match_res.group(1))
    #             seq_layer_number.append([_target_layer_name, _val])
    #         # end if
    #     # end for

    #     return seq_layer_number

    def _extract_hidden_vector(self,
                               tensor_translated_sequence: torch.Tensor,
                               outputs: ty.Union[transformers.generation.utils.GenerateEncoderDecoderOutput, transformers.generation.utils.GenerateBeamEncoderDecoderOutput],
                               target_layers_extraction: ty.List[str],
                               is_batch_mode: bool = False,
                               n_batch_size: ty.Optional[int] = None
                               ) -> ty.Union[ty.Dict[str, torch.Tensor], ty.List[ty.Dict[str, torch.Tensor]]]:
        """
        When the output is `transformers.generation.utils.GenerateBeamEncoderDecoderOutput`, 
        the data structure is " Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor of shape (batch_size*num_beams*num_return_sequences, num_heads, generated_length, sequence_length)".
        https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput.decoder_attentions

        When the output is `transformers.generation.utils.GenerateEncoderDecoderOutput`,
        Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor of shape (batch_size, generated_length, hidden_size).
        https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput

        Return:
            a dict object when is_batch_mode = False. {"key of layer name": Tensor}. The tensor size is (T: tokens, D: dimensions).
            a list of dict object when is_batch_mode = True.
                A dict. object is {"key of layer name": Tensor}.
        """
        def _add_word_embedding_decoder(tensor_token_ids: torch.Tensor,
                                        d_layer_key2embedding: ty.Dict[str, ty.List[torch.Tensor]]
                                        ) -> ty.Dict[str, ty.List[torch.Tensor]]:
            # adding word embedding layer.
            tensor_word_embedding = self._get_word_embedding_decoder(tensor_token_ids)
            name_layer_word_embedding = self._get_decoder_word_embedding_layer_name()
            if len(tensor_word_embedding.shape) == 3:
                _tensor_word_embedding = tensor_word_embedding[0]
            elif len(tensor_word_embedding.shape) == 2:
                assert tensor_word_embedding.shape[0] == len(tensor_token_ids)
                _tensor_word_embedding = tensor_word_embedding
            else:
                raise RuntimeError(f'The word embedding shape is not defined pattern. input tensor shape -> {tensor_word_embedding.shape}')
            # end if
            d_layer_key2embedding[name_layer_word_embedding] = list(_tensor_word_embedding)

            return d_layer_key2embedding
        # end def


        if 'decoder_hidden_states' in outputs:
            """
            The data structure is Tuple(Tuple()).
            The 1st tuple represents a sequence of layers.
            The 2nd depth tuple represents a sequence of tokens.
            """
            assert target_layers_extraction is not None

            assert outputs.decoder_hidden_states is not None
            assert isinstance(outputs.decoder_hidden_states, tuple)
            n_layers = len(outputs.decoder_hidden_states) - 2  # one first element is for the word-embedding, one last element is for the layer normalization.

            seq_layer_number = self._parse_layer_sequence(target_layers_extraction)


            if isinstance(outputs, transformers.generation.utils.GenerateBeamEncoderDecoderOutput):
                dict_extraction = {_key_name: [] for _key_name in target_layers_extraction}

                _token_counts = len(outputs.decoder_hidden_states)
                _tuple_token: ty.Tuple
                for _tuple_token in outputs.decoder_hidden_states:
                    for _target_layer_name, _target_number in seq_layer_number:
                        _tensor_layer: torch.Tensor = _tuple_token[_target_number + 1]  # plus one for the word-embedding layer at the 1st
                        _tensor_target = _tensor_layer[0, 0, :]

                        dict_extraction[_target_layer_name].append(_tensor_target)
                    # end for
                # end for
            elif isinstance(outputs, transformers.generation.utils.GenerateEncoderDecoderOutput):
                _token_counts = len(outputs.decoder_hidden_states)

                if is_batch_mode:
                    assert n_batch_size is not None
                    dict_extraction = {}  # definiting it just for consistency.
                    return_obj = [{_key_name: [] for _key_name in target_layers_extraction} for __ in range(n_batch_size)]
                else:
                    dict_extraction = {_key_name: [] for _key_name in target_layers_extraction}
                    return_obj = []  # definiting it just for consistency.
                # end if

                _tuple_token: ty.Tuple
                for _tuple_token in outputs.decoder_hidden_states:
                    for _target_layer_name, _target_number in seq_layer_number:
                        _tensor_layer: torch.Tensor = _tuple_token[_target_number + 1]  # plus one for the word-embedding layer at the 1st
                        if is_batch_mode:
                            # in the batch mode, I extract samples and save it into the the sequence of dicts.
                            assert n_batch_size is not None
                            assert _tensor_layer.shape[0] == n_batch_size, f"The given batch size is not equal. `n_batch_size` = {n_batch_size} and tensor shape is {_tensor_layer.shape}"
                            # for-loop of processing over batch.
                            for _i_batch in range(n_batch_size):
                                _tensor_target = _tensor_layer[_i_batch, 0, :]
                                return_obj[_i_batch][_target_layer_name].append(_tensor_target)
                            # end for
                        else: 
                            _tensor_target = _tensor_layer[0, 0, :]
                            dict_extraction[_target_layer_name].append(_tensor_target)
                        # end if
                    # end for
                # end for
            else:
                raise TypeError()
            # end if
        
        # --------------------------------------------------------
        # post-process. Converting the list of tensor into a fixed size tensor.
        if is_batch_mode:
            seq_d_return = []
            assert len(return_obj) > 0
            for _d_obj in return_obj:
                # adding word emebdding tensor at the decoder side.
                _d_obj = _add_word_embedding_decoder(
                    tensor_translated_sequence, _d_obj)

                d_return = {}
                for _key_name, _list_tensor in _d_obj.items():
                    d_return[_key_name] = torch.stack(_list_tensor).cpu()
                # end if

                seq_d_return.append(d_return)
            # end for

            return seq_d_return
        else:
            d_return = {}

            # adding word emebdding tensor at the decoder side.            
            dict_extraction = _add_word_embedding_decoder(
                tensor_translated_sequence, dict_extraction) 

            for _key_name, _list_tensor in dict_extraction.items():
                d_return[_key_name] = torch.stack(_list_tensor).cpu()
            # end for
            
            return d_return
        # end if

    # -------------------------------------------------------------
    # Public Interfaces

    def translate_beam_search(self,
                    input_text: EvaluationTargetTranslationPair,
                    temperature: float = 1.0,
                    max_len_a: float = 0.0,
                    max_len_b: int = 200,
                    target_layers_extraction: ty.Optional[ty.List[str]] = None
                    ) -> TranslationResultContainer:
        exist_or_cache = self._is_exist_cache_or_fetch(
            input_text.sentence_id,
            tau_param=temperature,
            n_sampling=None)
        if isinstance(exist_or_cache, TranslationResultContainer):
            return exist_or_cache
        # end if

        source_text = input_text.source
        inputs = self.tokenizer(source_text, return_tensors="pt")
        source_tensor_tokens: torch.Tensor = inputs["input_ids"][0]

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        assert isinstance(tgt_lang_id, int), f"Unexpected type of the target language ID: {type(tgt_lang_id)}"

        max_length = self.__calculate_max_length(inputs, max_len_a, max_len_b)
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
            "output_hidden_states": True,
        }
        # Generate translation
        with torch.no_grad():
            model = self.model
            inputs = inputs.to(self.device)
            
            outputs: transformers.generation.utils.GenerateBeamEncoderDecoderOutput = model.generate(**inputs, **generation_kwargs)
        # end with

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

        # ------------------------------------------------------------
        # hidden vector extraction
        if target_layers_extraction is not None:
            d_hidden_layers = self._extract_hidden_vector(
                tensor_translated_sequence=tensor_token_ids,
                outputs=outputs, 
                target_layers_extraction=target_layers_extraction)
        else:
            d_hidden_layers = {}
        # end if
        # ------------------------------------------------------------
        # getting the word embedding
        assert isinstance(d_hidden_layers, dict)
        # tensor_word_embedding = self._get_word_embedding_decoder(tensor_token_ids)
        # name_layer_word_embedding = self._get_decoder_word_embedding_layer_name()
        # assert len(tensor_word_embedding.shape) == 2
        # assert tensor_word_embedding.shape[0] == len(tensor_token_ids)
        # d_hidden_layers[name_layer_word_embedding] = tensor_word_embedding

        # ------------------------------------------------------------        

        module_logger.debug(f"Translated: {translated_text}, Log Score: {_log_score}")
        
        return_obj = TranslationResultContainer(
            source_text=source_text,
            translation_text=translated_text,
            source_tensor_tokens=source_tensor_tokens,
            target_tensor_tokens=tensor_token_ids,
            source_language=self.src_lang,
            target_language=self.target_lang,
            log_probability_score=_log_score,
            dict_layer_embeddings=d_hidden_layers,
            argument_translation_conditions=generation_kwargs
        )
        
        if self.is_use_cache:
            self._save_cache(
                sentence_id=input_text.sentence_id, 
                translation_obj=return_obj,
                tau_param=temperature,
                n_sampling=None)
        # end if

        return return_obj

    def translate_sample_multiple_times(self,
                                        input_text: EvaluationTargetTranslationPair,
                                        temperature: float,
                                        n_sampling: int,
                                        max_len_a: float,
                                        max_len_b: int,
                                        n_max_attempts: int = 5,
                                        batch_size: int = 5,
                                        target_layers_extraction: ty.Optional[ty.List[str]] = None,
                                        is_sampling_in_iteration: bool = False,
                                        is_auto_recovery_sampling: bool = True                                        
                                        ) -> ty.List[TranslationResultContainer]:
        exists_or_cache = self._is_exist_cache_or_fetch(
            input_text.sentence_id,
            tau_param=temperature,
            n_sampling=n_sampling)
        if isinstance(exists_or_cache, list):
            return exists_or_cache
        else:
            source_text = input_text.source
            tensor_source_tokens = self.tokenizer(source_text, return_tensors="pt")
            seq_result = self._call_interface_sampling(
                source_text=source_text,
                tensor_source_tokens=tensor_source_tokens,
                temperature=temperature,
                n_sampling=n_sampling,
                max_len_a=max_len_a,
                max_len_b=max_len_b,
                n_max_attempts=n_max_attempts,
                batch_size=batch_size,
                target_layers_extraction=target_layers_extraction,
                is_sampling_in_iteration=is_sampling_in_iteration,
                is_auto_recovery_sampling=is_auto_recovery_sampling
            )
            if self.is_use_cache:
                self._save_cache(
                    sentence_id=input_text.sentence_id, 
                    tau_param=temperature,
                    n_sampling=n_sampling,
                    translation_obj=seq_result)
            # end if

            return seq_result
