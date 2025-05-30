import typing as ty
import re
import random
import copy
from functools import partial
from pathlib import Path

import torch
import numpy as np
import omegaconf

import logging
import tempfile

import fairseq
from fairseq.models.transformer import TransformerModel
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase, 
    TransformerDecoderLayer)
from fairseq.models.transformer.transformer_encoder import TransformerEncoderBase
from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase


from .module_base import (
    BaseTranslationModelHandlerVer2,
    TranslationResultContainer,
    EvaluationTargetTranslationPair)

from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...exceptions import ParameterSettingException


module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())



def attach_decoder_tracer(decoder: TransformerDecoderBase):
    original_forward = decoder.forward
    decoder._stored_hidden_states = []  # Will become a list of lists: [layer][timestep]

    def traced_forward(*args, **kwargs):
        kwargs['return_all_hiddens'] = True
        x, extra = original_forward(*args, **kwargs)
        inner_states = extra.get("inner_states", [])

        for i, layer_output in enumerate(inner_states):
            try:
                decoder._stored_hidden_states[i].append(layer_output.detach().cpu())
            except IndexError:
                raise IndexError(f'Attempted accessing {i}-th index. But list has only {decoder._stored_hidden_states} element.')
        return x, extra

    decoder.forward = traced_forward
    return decoder




class FairSeqTranslationModelHandlerVer2(BaseTranslationModelHandlerVer2):
    def __init__(self,
                 path_dir_fairseq_model: Path,
                 path_model_checkpoint: Path,
                 path_sentencepiece_model: Path,
                 is_select_gpu_flexible: bool = True,
                 path_cache_dir: ty.Optional[Path] = None, 
                 is_use_cache: bool = True,
                 random_seed: int = 42,
                 is_zlib_compress: bool = True,
                 is_save_convert_float16: bool = False):
        super().__init__(
            path_cache_dir=path_cache_dir,
            is_use_cache=is_use_cache,
            is_zlib_compress=is_zlib_compress,
            is_save_convert_float16=is_save_convert_float16)

        # self.model_encoder_decoder_mt = model_encoder_decoder_mt
        self.is_select_gpu_flexible: bool = is_select_gpu_flexible
        self.random_seed = random_seed

        assert path_dir_fairseq_model.exists()
        assert path_model_checkpoint.exists()
        assert path_sentencepiece_model.exists()

        self.path_dir_fairseq_model = path_dir_fairseq_model
        self.path_model_checkpoint = path_model_checkpoint
        self.path_sentencepiece_model = path_sentencepiece_model

        self.fairseq_interface_default_set = self._reload_fairseq_interface()  # I do not modify this modle. Just for reference.

        # self.decoder_initial_status: TransformerDecoderBase = copy.deepcopy(self.model_encoder_decoder_mt.models[0].decoder)
        
    # ---------------------------------------------
    # Private Methods

    # def __get_local_fairseq_interface(self) -> GeneratorHubInterface:
    #     """I get the local `GeneratorHubInterface`. 
    #     When I keep using the same `GeneratorHubInterface` object, an exception keep raising up.
    #     No idea why....but, making a local variable solves this issue."""
    #     if torch.cuda.is_available():
    #         # multiple GPUs
    #         if self.is_select_gpu_flexible:
    #             # select the less busy GPU
    #             device = torch.device(f"cuda:{self._get_less_busy_cuda_device()}")
    #         else:
    #             device = torch.device("cuda:0")
    #         # end if
    #     else:
    #         device = torch.device("cpu")
    #     # end if

    #     fairseq_interface = self.model_encoder_decoder_mt.to(device)

    #     return fairseq_interface
    
    def _reload_fairseq_interface(self) -> GeneratorHubInterface:
        fairseq_hub = TransformerModel.from_pretrained(
                self.path_dir_fairseq_model,
                checkpoint_file=self.path_model_checkpoint.as_posix(),
                data_name_or_path=self.path_dir_fairseq_model.as_posix(),
                bpe="sentencepiece",
                sentencepiece_model=self.path_sentencepiece_model
            )
        return fairseq_hub

    def _get_local_fairseq_interface_patched(self,
                                             fairseq_interface: GeneratorHubInterface
                                             ) -> GeneratorHubInterface:
        """I get the local `GeneratorHubInterface`. 
        When I keep using the same `GeneratorHubInterface` object, an exception keep raising up.
        No idea why....but, making a local variable solves this issue."""
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

        # fairseq_interface = self.model_encoder_decoder_mt.to(device)
        assert isinstance(fairseq_interface.models, torch.nn.modules.container.ModuleList)
        assert len(fairseq_interface.models) == 1
        assert isinstance(fairseq_interface.models[0].decoder, TransformerDecoderBase)
        # fairseq_interface.models[0].decoder = self.decoder_initial_status
        fairseq_interface.models[0].decoder = attach_decoder_tracer(fairseq_interface.models[0].decoder)
        fairseq_interface.models[0].decoder._stored_hidden_states = [[] for _ in range(len(fairseq_interface.models[0].decoder.layers) + 1)]

        fairseq_interface_device = fairseq_interface.to(device)

        return fairseq_interface_device


    # ------------------------------------------------------------
    # Semi Private Methods


    def _get_word_embedding_decoder(self,
                                    fairseq_interface: GeneratorHubInterface,
                                    token_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            word_embeddings = fairseq_interface.models[0].decoder.embed_tokens(token_tensor)  # type: ignore

            # Note: I leave the experimental code for getting the word embedding and positional encoding.
            # tokens = token_tensor.unsqueeze(0)
            # word_emb = fairseq_interface.models[0].decoder.embed_tokens(tokens)
            # pos_emb = fairseq_interface.models[0].decoder.embed_positions(tokens)
            # combined = fairseq_interface.models[0].decoder.dropout_module(word_emb + pos_emb)
            # combined = combined.transpose(0, 1)  # To [seq_len, batch, dim]
        # end with
        return word_embeddings


    # def _post_process_hidden_states(self,
    #                                 seq_decoder_layer_name: ty.List[str],
    #                                 decoder_states: ty.List[ty.Tuple[int, torch.Tensor]],
    #                                 target_layers_extraction: ty.Optional[ty.List[str]],
    #                                 ) -> ty.Dict[str, torch.Tensor]:
    #     """
    #     Return: {Key-of-layer-name: Tensor}. The Tensor is [T: tokens, D: DIms].
    #     """
    #     d_layer_id2tensor_calls = self._get_layer2tensor(decoder_states)

    #     _extraction_dict = {_layer_name: [] for _layer_name in seq_decoder_layer_name}

    #     for _ind_layer, _layer_name in enumerate(seq_decoder_layer_name):
    #         _seq_tensor_layer_called: ty.List[torch.Tensor] = d_layer_id2tensor_calls[_ind_layer]
    #         for _call_index, _tensor_hidden in enumerate(_seq_tensor_layer_called):
    #             _extraction_dict[_layer_name].append(_tensor_hidden[0, 0, :])
    #         # end for
    #     # end for
        
    #     if target_layers_extraction is None:
    #         _d_layer2hidden_tensor_token = {
    #             _layer_name: torch.stack(_seq_tensor) for _layer_name, _seq_tensor in _extraction_dict.items() 
    #         }
    #     else:
    #         _d_layer2hidden_tensor_token = {
    #             _layer_name: torch.stack(_seq_tensor) for _layer_name, _seq_tensor in _extraction_dict.items()
    #             if _layer_name in target_layers_extraction}
    #     # end if

    #     return _d_layer2hidden_tensor_token

    # @staticmethod    
    # def _get_layer2tensor(decoder_states: ty.List[ty.Tuple[int, torch.Tensor]]) -> ty.Dict[int, ty.List[torch.Tensor]]:
    #     """I want to make {layer-number: [tensor]}. The list is the size of tokens.
        
    #     Returns:
    #         {layer-id: [tensor-at-1st-call, tensor-at-2nd-call, ...]}.
    #         Here, `call` is an action that Python interpreter calls the target layer.
    #         The `call` sounds the similar concept with the token.
    #         But, in LLM and batch process, the token number is not consistent always.
    #         E.g. ["I am a good boy.", "I am a boy."].
    #         Thus, tensor shape can be different. However the number of calls are consistent. 
    #     """
    #     sqe_possible_layer_numbers = list(set([_t[0] for _t in decoder_states]))
    #     layer2list_tensors = {_l_id: [] for _l_id in sqe_possible_layer_numbers}
    #     # iteration over `decoder_states` in the sequence. The order should be the order of tokens per layer-id.
    #     _t_layer_vector: ty.Tuple[int, torch.Tensor]
    #     for _t_layer_vector in decoder_states:
    #         _layer_id: int = _t_layer_vector[0]
    #         layer2list_tensors[_layer_id].append(_t_layer_vector[1])
    #     # end for

    #     # assert that all layers has the same tokens
    #     _possible_calls = set([len(_vec_hidden) for _vec_hidden in layer2list_tensors.values()])
    #     assert len(_possible_calls) == 1, f"Expecting all layers have the same numbers to be called (of layer). However, inconsistent numbers -> {_possible_calls}"
        
    #     return layer2list_tensors
    # # end def    
    
    # def _post_process_hidden_states_batch_out(self,
    #                                           seq_decoder_layer_name: ty.List[str],
    #                                 decoder_states: ty.List[ty.Tuple[int, torch.Tensor]],
    #                                 translation_output_obj: ty.List[ty.List[ty.Dict]],
    #                                 target_layers_extraction: ty.Optional[ty.List[str]] = None
    #                                 ) -> ty.List[ty.Dict[str, torch.Tensor]]:
    #     """
    #     Return: {Key-of-layer-name: Tensor}. The Tensor is [T: tokens, D: DIms].
    #     """
    #     # ------------------------------------------------------
    #     # post-process the decoder states. `decoder_states`.

    #     # def _get_layer2tensor() -> ty.Dict[int, ty.List[torch.Tensor]]:
    #     #     """I want to make {layer-number: [tensor]}. The list is the size of tokens.
            
    #     #     Returns:
    #     #         {layer-id: [tensor-at-1st-call, tensor-at-2nd-call, ...]}.
    #     #         Here, `call` is an action that Python interpreter calls the target layer.
    #     #         The `call` sounds the similar concept with the token.
    #     #         But, in LLM and batch process, the token number is not consistent always.
    #     #         E.g. ["I am a good boy.", "I am a boy."].
    #     #         Thus, tensor shape can be different. However the number of calls are consistent. 
    #     #     """
    #     #     sqe_possible_layer_numbers = list(set([_t[0] for _t in decoder_states]))
    #     #     layer2list_tensors = {_l_id: [] for _l_id in sqe_possible_layer_numbers}
    #     #     # iteration over `decoder_states` in the sequence. The order should be the order of tokens per layer-id.
    #     #     _t_layer_vector: ty.Tuple[int, torch.Tensor]
    #     #     for _t_layer_vector in decoder_states:
    #     #         _layer_id: int = _t_layer_vector[0]
    #     #         layer2list_tensors[_layer_id].append(_t_layer_vector[1])
    #     #     # end for

    #     #     # assert that all layers has the same tokens
    #     #     _possible_calls = set([len(_vec_hidden) for _vec_hidden in layer2list_tensors.values()])
    #     #     assert len(_possible_calls) == 1, f"Expecting all layers have the same numbers to be called (of layer). However, inconsistent numbers -> {_possible_calls}"
            
    #     #     return layer2list_tensors
    #     # # end def

    #     d_layer_id2tensor_calls = self._get_layer2tensor(decoder_states)

    #     stack_extraction_dict = [{_layer_name: [] for _layer_name in seq_decoder_layer_name}] * len(translation_output_obj)

    #     d_ind_in_batch2token_length: ty.Dict[int, int] = {_ind_in_batch: len(_d_result[0]['tokens']) for _ind_in_batch, _d_result in enumerate(translation_output_obj)}

    #     for _ind_layer, _layer_name in enumerate(seq_decoder_layer_name):
    #         _seq_tensor_layer_called: ty.List[torch.Tensor] = d_layer_id2tensor_calls[_ind_layer]
    #         for _call_index, _tensor_hidden in enumerate(_seq_tensor_layer_called):
    #             # a pair of tensor does not hold tensor-shape of n-batch. E.g. I expect (1, 3, 512), but actual (1, 2, 512). This is due to the hook mechanism looking at local context.
    #             _in_batch_index_have_call = sorted([_batch_ind for _batch_ind, _length in d_ind_in_batch2token_length.items() if _call_index < _length])
    #             assert len(_in_batch_index_have_call) == _tensor_hidden.shape[1]
                
    #             for _in_batch_index, _hidden_vector in zip(_in_batch_index_have_call, _tensor_hidden[0]):
    #                 # accessing the `_in_batch_index`-th list element (dict), accesing key=`_layer_name` and adding the tensor `_hidden_vector`. 
    #                 stack_extraction_dict[_in_batch_index][_layer_name].append(_hidden_vector)
    #             # end for
    #         # end for
    #     # end for
        
    #     seq_return_obj = []
    #     _d_layer2seq_tensor: ty.Dict[str, ty.List[torch.Tensor]]
    #     for _d_layer2seq_tensor in stack_extraction_dict:  # type: ignore
    #         if target_layers_extraction is None:
    #             _d_layer2hidden_tensor_token = {
    #                 _layer_name: torch.stack(_seq_tensor) for _layer_name, _seq_tensor in _d_layer2seq_tensor.items()                    
    #             }
    #         else:
    #             _d_layer2hidden_tensor_token = {
    #                 _layer_name: torch.stack(_seq_tensor) for _layer_name, _seq_tensor in _d_layer2seq_tensor.items()
    #                 if _layer_name in target_layers_extraction}
    #         # end if

    #         seq_return_obj.append(_d_layer2hidden_tensor_token)
    #     # end for

    #     return seq_return_obj

    # ------------------------------
    # Sampling, Semi Private Methods

    def _sampling_multi_input(self,
                              source_text: str,
                              tensor_source_tokens: torch.Tensor,
                              temperature: float,
                              n_sampling: int,
                              max_len_a: float,
                              max_len_b: int,
                              batch_size: int = 100,
                              target_layers_extraction: ty.Optional[ty.List[str]] = None) -> ty.List[TranslationResultContainer]:

        # def one_sampling_trial(fairseq_interface: GeneratorHubInterface,
        #                        seq_input_tensor: ty.List[torch.Tensor],
        #                        argument_translation: ty.Dict,
        #                        random_seed: int):
        #     # ---------------------------------------
        #     # closure of extracting hidden layers.
        #     decoder_states: ty.List[ty.Tuple[int, torch.Tensor]] = []

        #     def capture_hidden_states(module, input, output: ty.Tuple, layer_index: int):
        #         # output shape: [batch_size * beam_size, tgt_len, hidden_dim]
        #         output_shape: torch.Tensor = output[0]
        #         # decoder_states.append( (layer_index, output_shape.clone().detach().cpu()) )
        #         decoder_states.append( (layer_index, output_shape) )
        #     # end def
        #     # ---------------------------------------

        #     # attaching the hook to the decoder layers
        #     hook_handles = []
        #     # This does not work. Causing CUDA related error.
        #     # for _i_module, _t_module in enumerate(fairseq_interface.models[0].decoder.layers.named_children()):
        #     for _i_module, _t_module in enumerate(fairseq_interface.models[0].decoder.layers):
        #         # __, module = _t_module
        #         module = _t_module
        #         if isinstance(module, (TransformerDecoderLayerBase, TransformerDecoderLayer)): # Adjust the class if needed
        #             hook_fn = partial(capture_hidden_states, layer_index=_i_module)
        #             hook_handle = module.register_forward_hook(hook_fn)
        #             hook_handles.append(hook_handle)
        #     # end for
        #     assert len(hook_handles) > 0, "No designed Layers are found in the decoder."

        #     with (torch.random.fork_rng(), torch.no_grad()):
        #         torch.manual_seed(random_seed)
        #         torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.

        #         translations = fairseq_interface.generate(
        #                 tokenized_sentences=seq_input_tensor, **argument_translation)  # type: ignore
        #     # end with

        #     # Note. The `decoder_states` is a list of tensors. The tensor shape is [1, batch-size, dim-size]

        #     return translations, decoder_states
        # # end def

        raise NotImplementedError('Batch translation (and hidden state extraction) is not available for FairSeq package.')
        
        def _post_process_hidden_states(seq_translation: ty.List) -> ty.List[ty.Dict[str, torch.Tensor]]:
            """Return: {'layer-name': (T-tokens, D-dims) }"""
            size_batch = len(seq_translation)

            __, seq_decoder_layer_name = self.get_all_possible_layers()
            seq_inner_states_layers = list_inner_states[1:]  # I skip the 1st layer (of word embedding + positional encoding)
            assert len(seq_inner_states_layers) == len(seq_decoder_layer_name)

            _extraction_dict = [{_layer_name: [] for _layer_name in seq_decoder_layer_name}] * size_batch

            for _ind_layer, _layer_name in enumerate(seq_decoder_layer_name):
                _seq_tokens_state_tensor: torch.Tensor = seq_inner_states_layers[_ind_layer]  # [ (1, batch-size, D) ]
                for _t_tokens, _tensor_hidden in enumerate(_seq_tokens_state_tensor):
                    for _index_in_batch in range(_tensor_hidden.shape[1]):
                        _extraction_dict[_index_in_batch][_layer_name].append(_tensor_hidden[0, _index_in_batch, :])
                # end for
            # end for
            
            seq_post_processed = []
            for _d_obj in _extraction_dict:
                if target_layers_extraction is None:                    
                    _d_layer2hidden_tensor_token = {
                        _layer_name: torch.stack(_seq_tensor).cpu() for _layer_name, _seq_tensor in _d_obj.items() 
                    }
                    seq_post_processed.append(_d_layer2hidden_tensor_token)
                else:
                    _d_layer2hidden_tensor_token = {
                        _layer_name: torch.stack(_seq_tensor).cpu() for _layer_name, _seq_tensor in _d_obj.items()
                        if _layer_name in target_layers_extraction}
                    seq_post_processed.append(_d_layer2hidden_tensor_token)
                # end if

            return seq_post_processed
        # end def

        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(self.random_seed)
        assert n_sampling < 10000, f"n_sampling={n_sampling} should be less than 10000."        
        seq_random_seed_values = _gen_random.sample(list(range(0, 9999)), k=n_sampling)

        batch_stack = []

        name_decoder_word_embedding = self._get_decoder_word_embedding_layer_name()

        seq_encoder_layer_name, seq_decoder_layer_name = self.get_all_possible_layers()
        for i in range(0, n_sampling, batch_size):
            _random_seed = seq_random_seed_values[i]

            module_logger.debug(f"Sampling {i} to {i + batch_size} / {n_sampling}")

            _current_batch_size = min(batch_size, n_sampling - i)

            _arguments_dict = dict(
                sampling=True,
                temperature=temperature,
                sampling_topk=-1.0,
                sampling_topp=1.0,
                max_len_a_mt=max_len_a,
                max_len_b_mt=max_len_b,
                beam=1
            )

            # I have to reset the interface everything....otherwise the internal states remains non-empty list.
            fairseq_interface = self._get_local_fairseq_interface_patched()
            
            # _seq_input_tensor = [tensor_source_tokens.to(fairseq_interface.device)] * _current_batch_size  # type: ignore
            _seq_input_tensor = [tensor_source_tokens] * _current_batch_size

            with (torch.random.fork_rng(), torch.no_grad()):
                torch.manual_seed(_random_seed)
                torch.cuda.manual_seed_all(_random_seed)  # if you are using multi-GPU.

                translations: ty.List = fairseq_interface.generate(
                        tokenized_sentences=_seq_input_tensor, **_arguments_dict)  # type: ignore
            # end with

            # 1st list of N-layers + 1. 2nd list of T-tokens. 3rd element is torch.Tensor.
            # note: +1 layer is at word_embedding (+ positional encoding)
            list_inner_states = fairseq_interface.models[0].decoder._stored_hidden_states
            
            _d_argument = copy.deepcopy(_arguments_dict)
            _d_argument['random_seed'] = _random_seed

            decoder2states = _post_process_hidden_states(translations)
            assert len(decoder2states) == len(translations)
            for _ind_in_batch, _t_translation_res in enumerate(translations):
                _tensor_translation = _t_translation_res[0]['tokens']
                _log_score = _t_translation_res[0]['score'].cpu().item()
                _translation_text = fairseq_interface.decode(_tensor_translation)
                _decoder_stats = decoder2states[_ind_in_batch]

                # converting tokens into the word embedding
                decoder_word_embedding = self._get_word_embedding_decoder(
                    fairseq_interface,
                    _tensor_translation)
                _decoder_stats[name_decoder_word_embedding] = decoder_word_embedding.cpu()

                return_obj = TranslationResultContainer(
                    source_text=source_text,
                    translation_text=_translation_text,
                    source_tensor_tokens=tensor_source_tokens.cpu(),
                    target_tensor_tokens=_tensor_translation.cpu(),
                    source_language='deu_Latn',
                    target_language='eng_Latn',
                    log_probability_score=_log_score,
                    dict_layer_embeddings=_decoder_stats,
                    argument_translation_conditions=_d_argument
                )
                batch_stack.append(return_obj)
        # end for

        return batch_stack        

    def _sampling_single_input(self,
                               source_text: str,
                               tensor_source_tokens: torch.Tensor,
                               temperature: float,
                               n_sampling: int,
                               max_len_a: float,
                               max_len_b: int,
                               n_max_attempts: int = 10,
                               target_layers_extraction: ty.Optional[ty.List[str]] = None,
                               reload_model_per_iteration: int = 10
                               ) -> ty.List[TranslationResultContainer]:
        """Generating translated token sequence using the stochastic sampling. 
        This method has a custom procedure to extract hidden state layer's vector.
        Due to this modification, I regularly reload the fairseq interface module, otherwise the stack list will be over the RAM.

        Args:
            reload_model_per_iteration: the interval of reloading the fairseq model.
        """

        def _post_process_hidden_states(list_inner_states, translations: ty.List) -> ty.Dict[str, torch.Tensor]:
            """Return: {'layer-name': (T-tokens, D-dims) }"""
            __, seq_decoder_layer_name = self.get_all_possible_layers()
            seq_inner_states_layers = list_inner_states[1:]  # I skip the 1st layer (of word embedding + positional encoding)
            assert len(seq_inner_states_layers) == len(seq_decoder_layer_name)

            _extraction_dict = {_layer_name: [] for _layer_name in seq_decoder_layer_name}

            for _ind_layer, _layer_name in enumerate(seq_decoder_layer_name):
                _seq_tokens_state_tensor: torch.Tensor = seq_inner_states_layers[_ind_layer]  # [ (1, beam-size, D) ]
                for _t_tokens, _tensor_hidden in enumerate(_seq_tokens_state_tensor):
                    _extraction_dict[_layer_name].append(_tensor_hidden[0, 0, :])
                # end for
            # end for
            
            if target_layers_extraction is None:
                _d_layer2hidden_tensor_token = {
                    _layer_name: torch.stack(_seq_tensor).cpu() for _layer_name, _seq_tensor in _extraction_dict.items() 
                }
            else:
                _d_layer2hidden_tensor_token = {
                    _layer_name: torch.stack(_seq_tensor).cpu() for _layer_name, _seq_tensor in _extraction_dict.items()
                    if _layer_name in target_layers_extraction}
            # end if

            n_tokens = len(translations[0]['tokens'])

            for _layer_name, _tensor in _d_layer2hidden_tensor_token.items():
                assert _tensor.shape[0] == n_tokens, f'Expected n-tokens: {n_tokens}, Extracted hidden state layer has tokens of {_tensor.shape[0]}'
            # end for
            return _d_layer2hidden_tensor_token
        # end def

        def _extract_layer_token_tensor_current_iteration(layer_inner_states_this_iteration: ty.List[ty.List[torch.Tensor]],
                                                          seq_memory_translated_token_size: ty.List[int]) -> ty.List[ty.List[torch.Tensor]]:
            seq_extracted = []
            for _l_layer in layer_inner_states_this_iteration:
                seq_extracted.append(_l_layer[sum(seq_memory_translated_token_size):])
            # end for
            return seq_extracted

        def one_sampling_trial(fairseq_interface: GeneratorHubInterface,
                               i_iteration: int):
            assert isinstance(fairseq_interface.models[0].decoder, TransformerDecoderBase)
            assert hasattr(fairseq_interface.models[0].decoder, '_stored_hidden_states')
            assert isinstance(fairseq_interface.models[0].decoder._stored_hidden_states, list)
            with (torch.random.fork_rng(), torch.no_grad()):
                _seed_value = seq_random_seed_values[i_iteration]
                torch.manual_seed(_seed_value)
                torch.cuda.manual_seed_all(_seed_value)  # if you are using multi-GPU.

                # fairseq_interface.models[0].decoder._stored_hidden_states = [[] for _ in range(len(fairseq_interface.models[0].decoder.layers) + 1)]
                _translations = fairseq_interface.generate(tokenized_sentences=tensor_source_tokens, **argument_translation)  # type: ignore
                list_inner_states_current = fairseq_interface.models[0].decoder._stored_hidden_states
                
                assert len(set([len(_list_tokens) for _list_tokens in list_inner_states_current])) == 1, f'Token Size must be consistent over hidden layers. However, inconsistent token numbers are detected -> {[len(_list_tokens) for _list_tokens in list_inner_states]}'
                
                # list_inner_states_this_iteration = list_inner_states_current[sum(seq_memory_translated_token_size):]
                list_inner_states_this_iteration = _extract_layer_token_tensor_current_iteration(list_inner_states_current, seq_memory_translated_token_size)
                assert isinstance(_translations[0], dict)
                n_tokens_translation = _translations[0]['tokens'].shape[0]
                seq_memory_translated_token_size.append(n_tokens_translation)

                for _l_layer in list_inner_states_this_iteration:
                    assert len(_l_layer) == n_tokens_translation, f'Expected n-tokens: {n_tokens_translation}, Extracted hidden state layer has tokens of {len(_l_layer)} at {i_random_seed_index}-th iteration.'
                # end for

                # post-process
                _decoder2states = _post_process_hidden_states(list_inner_states_this_iteration, _translations)
                # converting tokens into the word embedding
                decoder_word_embedding = self._get_word_embedding_decoder(
                    fairseq_interface, _translations[0]['tokens'])
                _decoder2states[name_decoder_word_embedding] = decoder_word_embedding.cpu()

                _d_argument = copy.deepcopy(argument_translation)
                _d_argument['random_seed'] = seq_random_seed_values[i_random_seed_index]

                return (_translations, _decoder2states, _d_argument)
        # end def


        output_stack = []
        i_error_attempt = 0
        i_random_seed_index = 0

        # making the random seed values from the `random_seed` parameter
        _gen_random = random.Random(self.random_seed)
        assert n_sampling < 99999, f"n_sampling={n_sampling} should be less than 10000."
        seq_random_seed_values = list(range(0, 99999))
        _gen_random.shuffle(seq_random_seed_values)

        argument_translation = dict(
            sampling=True,
            temperature=temperature,
            sampling_topk=-1.0,
            sampling_topp=-1.0,
            max_len_a_mt=max_len_a,
            max_len_b_mt=max_len_b,
            beam=1      
        )

        name_decoder_word_embedding = self._get_decoder_word_embedding_layer_name()

        fairseq_interface = self._reload_fairseq_interface()  # initial loading.
        fairseq_interface = self._get_local_fairseq_interface_patched(fairseq_interface)

        seq_memory_translated_token_size: ty.List[int] = []
        while len(output_stack) < n_sampling:
            try:
                _t_stack = one_sampling_trial(fairseq_interface, i_random_seed_index)
            except (AssertionError, RuntimeError, omegaconf.errors.UnsupportedValueType) as e:                    
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
                    fairseq_interface = self._reload_fairseq_interface()
                    fairseq_interface = self._get_local_fairseq_interface_patched(fairseq_interface)
                    seq_memory_translated_token_size = []

                    i_error_attempt += 1
                    i_random_seed_index += 1
                    continue
            else:
                output_stack.append(_t_stack)
                i_random_seed_index += 1
            # end if

            # reloading the interface and refreshing before the interface's internal stack beyond the RAM.
            if i_random_seed_index % reload_model_per_iteration == 0:
                fairseq_interface = self._reload_fairseq_interface()
                fairseq_interface = self._get_local_fairseq_interface_patched(fairseq_interface)
                seq_memory_translated_token_size = []
            # end if
        # end with

        # post-processing
        seq_output_obj = []
        for _index_in_batch, _t_exec_result in enumerate(output_stack):
            _obj_translations = _t_exec_result[0]
            _decoder_states = _t_exec_result[1]
            _d_argument = _t_exec_result[2]

            _n_tokens = len(_obj_translations[0]['tokens'])  # type: ignore
            _tensor_translation = _obj_translations[0]['tokens']

            # double-check tensor shape
            for _layer_key_name, _tensor in _decoder_states.items():
                assert _tensor.shape[0] == _n_tokens, f'Expected tensor shape is ({_n_tokens}, Dim). But found tensor is {_tensor.shape}. Exception. Layer={_layer_key_name} at index-of-batch={_index_in_batch}'
            # end for
    
            _log_score = _obj_translations[0]['score'].item()
            text_translation = fairseq_interface.decode(_obj_translations[0]['tokens'])  # type: ignore

            return_obj = TranslationResultContainer(
                source_text=source_text,
                translation_text=text_translation,
                source_tensor_tokens=tensor_source_tokens.cpu(),
                target_tensor_tokens=_obj_translations[0]['tokens'].cpu(),
                source_language='de_Latn',
                target_language='en_Latn',
                log_probability_score=_log_score,
                dict_layer_embeddings=_decoder_states,
                argument_translation_conditions=_d_argument
            )
            seq_output_obj.append(return_obj)
        # end for

        return seq_output_obj
    # end def
    
    def _call_fairseq_interface_stochastic(self, 
                                source_text: str,
                                tensor_source_tokens: torch.Tensor,
                                temperature: float,
                                n_sampling: int,
                                max_len_a: float,
                                max_len_b: int,
                                is_sampling_in_iteration: bool = False,
                                is_auto_recovery_sampling: bool = True,
                                n_max_attempts: int = 100,
                                batch_size: int = 100,
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
            # Note: Possible `generate` options are defined at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810 
            # `generate` method executes `inference_step` method of `TranslationTask` class.
            # `**kwargs` arguments are passed to `build_generator` method of `TranslationTask` class first,
            # and then, the generator object is passed to the `inference_step` method.
            # See `build_generator` API at here: https://fairseq.readthedocs.io/en/latest/tasks.html#fairseq.tasks.FairseqTask.build_generator
            # The args object is `fairseq.dataclass.configs.GenerationConfig`.
            # The `GenerationConfig` definition is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810
            output_stack = self._sampling_single_input(
                source_text=source_text,
                tensor_source_tokens=tensor_source_tokens,
                temperature=temperature,
                n_max_attempts=n_max_attempts,
                n_sampling=n_sampling,
                max_len_a=max_len_a,
                max_len_b=max_len_b,
                target_layers_extraction=target_layers_extraction)

            # Note: 2025-05-02. I found that fairseq package can not do the follwing at the same time.
            # batch sampling + hidden layer extraction. Thus, the following blocks are commented out.
            # if is_sampling_in_iteration:
            #     output_stack = self._sampling_single_input(
            #         source_text=source_text,
            #         tensor_source_tokens=tensor_source_tokens,
            #         temperature=temperature,
            #         n_max_attempts=n_max_attempts,
            #         n_sampling=n_sampling,
            #         max_len_a=max_len_a,
            #         max_len_b=max_len_b,
            #         target_layers_extraction=target_layers_extraction)
            # else:
            #     try:
            #         output_stack = self._sampling_multi_input(
            #             source_text=source_text,
            #             tensor_source_tokens=tensor_source_tokens,
            #             temperature=temperature,
            #             n_sampling=n_sampling,
            #             max_len_a=max_len_a,
            #             max_len_b=max_len_b,
            #             target_layers_extraction=target_layers_extraction,
            #             batch_size=batch_size)
            #     except (AssertionError, RuntimeError, omegaconf.errors.UnsupportedValueType) as e:
            #         if is_auto_recovery_sampling:
            #             module_logger.warning(f"Assertion error occurred: {e}")
            #             output_stack = self._sampling_single_input(
            #                 source_text=source_text,
            #                 tensor_source_tokens=tensor_source_tokens,
            #                 temperature=temperature,
            #                 n_max_attempts=n_max_attempts,
            #                 n_sampling=n_sampling,
            #                 max_len_a=max_len_a,
            #                 max_len_b=max_len_b,
            #                 target_layers_extraction=target_layers_extraction)
            #         else:
            #             raise e
            #         # end if
            #     # end try-except
            # # end if
        # end with

        return output_stack

    # ------------------------------------------------------------
    # Public Methods

    def get_all_possible_layers(self) -> ty.Tuple[ty.List, ty.List]:
        encoder_layers = []
        decoder_layers = []

        assert len(self.fairseq_interface_default_set.models) == 1, "This method assumes only one model (of encoder-decoder)." 
        transformer_encoder_obj: TransformerEncoderBase = self.fairseq_interface_default_set.models[0].encoder
        transformer_decoder_obj: TransformerDecoderBase = self.fairseq_interface_default_set.models[0].decoder

        for i, layer in enumerate(transformer_encoder_obj.layers):
            encoder_layers.append(f"encoder.{i}")
        # end for
        for i, layer in enumerate(transformer_decoder_obj.layers):
            decoder_layers.append(f"decoder.{i}")
        # end for

        # test accesing the layers

        return encoder_layers, decoder_layers

    def translate_beam_search(self,
                              input_text: EvaluationTargetTranslationPair,
                              temperature: float = 1.0,
                              max_len_a: float = 0.0,
                              max_len_b: int = 200,
                              target_layers_extraction: ty.Optional[ty.List[str]] = None,
                              ) -> TranslationResultContainer:
        """
        Args:

        Exception:
            `ParameterSettingException`
        """
        if self.is_use_cache:
            false_or_cache = self._is_exist_cache_or_fetch(
                input_text.sentence_id,
                tau_param=temperature,
                n_sampling=None)
            
            if isinstance(false_or_cache, TranslationResultContainer):
                return false_or_cache
            # end if
        # end if
        
        source_text = input_text.source

        # def translation_trial():
        #     # ---------------------------------------
        #     # closure of extracting hidden layers.
        #     decoder_states: ty.List[ty.Tuple[int, torch.Tensor]] = []

        #     # def capture_hidden_states(module, input, output: ty.Tuple):
        #     #     # output shape: [batch_size * beam_size, tgt_len, hidden_dim]
        #     #     output_shape: torch.Tensor = output[0]
        #     #     decoder_states.append(output_shape.detach().cpu())
        #     def capture_hidden_states(module, input, output: ty.Tuple, layer_index: int):
        #         # output shape: [batch_size * beam_size, tgt_len, hidden_dim]
        #         output_shape: torch.Tensor = output[0]
        #         decoder_states.append( (layer_index, output_shape.detach().cpu()) )
        #     # end def
        #     # ---------------------------------------

        #     fairseq_interface = self.__get_local_fairseq_interface()

        #     tensor_source_tokens = fairseq_interface.encode(source_text)

        #     # # attaching the hook to the decoder layers
        #     hook_handles = []
        #     for _i_module, _t_module in enumerate(fairseq_interface.models[0].decoder.layers.named_children()):
        #         __, module = _t_module
        #         if isinstance(module, (TransformerDecoderLayerBase, TransformerDecoderLayer)): # Adjust the class if needed
        #             hook_fn = partial(capture_hidden_states, layer_index=_i_module)
        #             hook_handle = module.register_forward_hook(hook_fn)
        #             hook_handles.append(hook_handle)
        #     # end for
        #     assert len(hook_handles) > 0, "No designed Layers are found in the decoder."


        #     max_len = (len(tensor_source_tokens) * max_len_a) + max_len_b
        #     argument_translations = dict(
        #         temperature=float(temperature),  # note: error occurs when the variable by numpy's. So, I forcelly put back to the Python's type.
        #         sampling=False, 
        #         beam=5,
        #         max_len_a=max_len_a,
        #         max_len_b=max_len_b + 1,
        #         # max_len=max_len
        #     )

        #     with torch.no_grad():            
        #         generation_output = fairseq_interface.generate(
        #             tensor_source_tokens, **argument_translations)  # type: ignore
        #     # end with

        #     return generation_output, decoder_states, tensor_source_tokens, argument_translations, fairseq_interface
        # # end def

        # generation_output, decoder_states, tensor_source_tokens, argument_translations, fairseq_interface = translation_trial()

        def _post_process_hidden_states() -> ty.Dict[str, torch.Tensor]:
            """Return: {'layer-name': (T-tokens, D-dims) }"""
            __, seq_decoder_layer_name = self.get_all_possible_layers()
            seq_inner_states_layers = list_inner_states[1:]  # I skip the 1st layer (of word embedding + positional encoding)
            assert len(seq_inner_states_layers) == len(seq_decoder_layer_name)

            _extraction_dict = {_layer_name: [] for _layer_name in seq_decoder_layer_name}

            for _ind_layer, _layer_name in enumerate(seq_decoder_layer_name):
                _seq_tokens_state_tensor: torch.Tensor = seq_inner_states_layers[_ind_layer]  # [ (1, beam-size, D) ]
                for _t_tokens, _tensor_hidden in enumerate(_seq_tokens_state_tensor):
                    _extraction_dict[_layer_name].append(_tensor_hidden[0, 0, :])
                # end for
            # end for
            
            if target_layers_extraction is None:
                _d_layer2hidden_tensor_token = {
                    _layer_name: torch.stack(_seq_tensor) for _layer_name, _seq_tensor in _extraction_dict.items() 
                }
            else:
                _d_layer2hidden_tensor_token = {
                    _layer_name: torch.stack(_seq_tensor) for _layer_name, _seq_tensor in _extraction_dict.items()
                    if _layer_name in target_layers_extraction}
            # end if

            return _d_layer2hidden_tensor_token
        # end def

        fairseq_interface = self._reload_fairseq_interface()
        fairseq_interface = self._get_local_fairseq_interface_patched(fairseq_interface)
        argument_translations = dict(
            temperature=float(temperature),  # note: error occurs when the variable by numpy's. So, I forcelly put back to the Python's type.
            sampling=False, 
            beam=5,
            max_len_a=max_len_a,
            max_len_b=max_len_b + 1,
            # max_len=max_len
        )
        tensor_source_tokens = fairseq_interface.encode(source_text)
        generation_output = fairseq_interface.generate(tensor_source_tokens, **argument_translations)  # type: ignore

        # 1st list of N-layers + 1. 2nd list of T-tokens. 3rd element is torch.Tensor.
        # note: +1 layer is at word_embedding (+ positional encoding)
        list_inner_states = fairseq_interface.models[0].decoder._stored_hidden_states

        # parsing the hidden states
        decoder2states = _post_process_hidden_states()

        # word embedding by the teacher forcing.
        tensor_translation: torch.Tensor = generation_output[0]['tokens']
        decoder_word_embedding = self._get_word_embedding_decoder(
            fairseq_interface,
            tensor_translation)
        decoder2states[self._get_decoder_word_embedding_layer_name()] = decoder_word_embedding.cpu()

        # Note: Hereafter, I look at only the 1st beam.
        assert len(generation_output) > 0, f"generation_output={generation_output}"
        assert "tokens" in generation_output[0], f"missing key 'tokens' in generation_output[0]={generation_output[0]}"
        assert isinstance(generation_output[0]['tokens'], torch.Tensor), f"generation_output[0]['tokens']={generation_output[0]['tokens']}"  # type: ignore

        text_translation = fairseq_interface.decode(generation_output[0]['tokens'])  # type: ignore
        
        # # ------------------------------------------------------

        _log_score = generation_output[0]['score'].item()

        return_obj = TranslationResultContainer(
            source_text=source_text,
            translation_text=text_translation,
            source_tensor_tokens=tensor_source_tokens.cpu(),
            target_tensor_tokens=tensor_translation.cpu(),
            source_language='deu_Latn',
            target_language='eng_Latn',
            log_probability_score=_log_score,
            dict_layer_embeddings=decoder2states,
            argument_translation_conditions=argument_translations
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
                                        n_sampling: int,
                                        temperature: float,
                                        max_len_a: float = 0.0,
                                        max_len_b: int = 200,
                                        n_max_attempts: int = 10,
                                        batch_size: int = 100,
                                        target_layers_extraction: ty.Optional[ty.List[str]] = None,
                                        is_sampling_in_iteration: bool = False,
                                        is_auto_recovery_sampling: bool = True,
                                        ) -> ty.List[TranslationResultContainer]:
        if self.is_use_cache:
            exists_or_cache = self._is_exist_cache_or_fetch(
                input_text.sentence_id,
                tau_param=temperature,
                n_sampling=n_sampling)
            if isinstance(exists_or_cache, list):
                return exists_or_cache
            # end if        

        source_text = input_text.source

        encoded_obj = self.fairseq_interface_default_set.encode(source_text)
        encoded_obj = encoded_obj.to(torch.int)

        seq_result = self._call_fairseq_interface_stochastic(
            source_text=source_text,
            tensor_source_tokens=encoded_obj,
            temperature=float(temperature),  # note: error occurs when the variable by numpy's. So, I forcelly put back to the Python's type.
            n_sampling=n_sampling,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            n_max_attempts=n_max_attempts,
            target_layers_extraction=target_layers_extraction,
            batch_size=batch_size,
            is_sampling_in_iteration=is_sampling_in_iteration,
            is_auto_recovery_sampling=is_auto_recovery_sampling)

        if self.is_use_cache:
            self._save_cache(
                sentence_id=input_text.sentence_id, 
                tau_param=temperature,
                n_sampling=n_sampling,
                translation_obj=seq_result)
        # end if

        return seq_result

