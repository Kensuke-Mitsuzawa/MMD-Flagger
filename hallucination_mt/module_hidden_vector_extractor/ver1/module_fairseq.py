import typing as ty
from pathlib import Path
import logging

import numpy as np
import torch

from fairseq.models.transformer import TransformerModel
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer.transformer_encoder import TransformerEncoderBase
from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase

from .module_base import (BaseVectorExtractor, ReturnTuple_method_extract_hidden_states)
from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler

module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())



class FairSeqVectorExtractor(BaseVectorExtractor):
    def __init__(self,
                 model: GeneratorHubInterface,
                 is_select_gpu_flexible: bool = False,
                 ) -> None:
        """
        """
        self.model = model        
        self.is_select_gpu_flexible = is_select_gpu_flexible

        _seq_encoder_layer_name, _seq_decoder_layer_name = self.get_all_possible_layers()
        self.encoder_layer_name = _seq_encoder_layer_name
        self.decoder_layer_name = _seq_decoder_layer_name

    def get_all_possible_layers(self) -> ty.Tuple[ty.List[str], ty.List[str]]:
        encoder_layers = []
        decoder_layers = []

        assert len(self.model.models) == 1, "This method assumes only one model (of encoder-decoder)." 
        transformer_encoder_obj: TransformerEncoderBase = self.model.models[0].encoder
        transformer_decoder_obj: TransformerEncoderBase = self.model.models[0].decoder

        for i, layer in enumerate(transformer_encoder_obj.layers):
            encoder_layers.append(f"encoder.layers.{i}")
        # end for
        for i, layer in enumerate(transformer_decoder_obj.layers):
            decoder_layers.append(f"decoder.layers.{i}")
        # end for

        # test accesing the layers

        return encoder_layers, decoder_layers


    def extract_hidden_states(self, 
                              source_text: str,
                              target_encoder_layers: ty.Optional[ty.List[int]] = None,
                              target_decoder_layers: ty.Optional[ty.List[int]] = None,
                              max_len_a_mt: float = 0.0,
                              max_len_b_mt: int = 200
                              ) -> ReturnTuple_method_extract_hidden_states:
        """A method of extracting the hidden states of the encoder and decoder layers.

        Args:
            source_text (str): The input text to be translated.
            target_encoder_layers (list, optional): The list of encoder layers to extract. Defaults to None.
            target_decoder_layers (list, optional): The list of decoder layers to extract. Defaults to None.
            max_len_a_mt (float, optional): The maximum length of the source text. Defaults to 0.0.
            max_len_b_mt (int, optional): The maximum length of the target text. Defaults to 200.

        Returns:
        """
        decoder_states = []

        def capture_hidden_states(module, input, output: ty.Tuple):
            # output shape: [batch_size * beam_size, tgt_len, hidden_dim]
            output_shape: torch.Tensor = output[0]
            decoder_states.append(output_shape.detach().cpu())

        def _encoder_post_process_tensor(seq_states: ty.List[torch.Tensor],
                                         seq_target_layers: ty.Optional[ty.List[int]],
                                         is_skip_embedding_layer: bool = True,
                                         ) -> ty.Dict[int, torch.Tensor]:
            """Post-process the tensor of Encoder hidden states.
            
            Args:
                is_skip_embedding_layer: If True, skip the embedding layer. If True, the layer number 0 is the word embedding layer.
            """
            d_layer2states = {}

            _i_layer = 0

            if is_skip_embedding_layer:
                _seq_states = seq_states[1:]
            else:
                _seq_states = seq_states
            # end if

            for _states in _seq_states:
                if seq_target_layers is not None and _i_layer not in seq_target_layers:
                    _i_layer += 1
                    continue
                # end if

                # _states is (Tokens, Batch, Dim)
                # I need to transpose it to (Batch, Tokens, Dim)
                if _states.is_nested:
                    _ = torch.nested.to_padded_tensor(_states, padding=0.0)
                    if len(_) == 3:
                        _states_record = _.squeeze(1)
                    else:
                        _states_record = _
                    # end if
                else:
                    if len(_states) == 3:
                        _states_record = _states.squeeze(1)
                    else:
                        _states_record = _states
                    # end if
                # end if
                d_layer2states[_i_layer] = _states_record.cpu()

                _i_layer += 1
            # end for
            return d_layer2states
        # end def

        # Check the device
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

        # -----------------------------------------------------------

        fairseq_interface = self.model.to(device)

        source_tokens_one = fairseq_interface.encode(source_text)
        source_tokens = source_tokens_one.unsqueeze(0)
        
        assert len(fairseq_interface.models) == 1, "This method assumes only one model (of encoder-decoder)." 
        
        # -----------------------------------------------------------
        # Encoder Processing
        # block of processing encoder.
        transformer_encoder_obj: TransformerEncoderBase = fairseq_interface.models[0].encoder  # type: ignore

        encoder_layer2states = {}
        with torch.no_grad():
            # Forward pass through the encoder
            encoder_out = transformer_encoder_obj.forward(source_tokens.to(device), 
                                                          return_all_hiddens=True)
            assert isinstance(encoder_out, dict), f"encoder_out={encoder_out}"
            assert "encoder_states" in encoder_out, f"missing key 'encoder_states' in encoder_out={encoder_out}"
            # tensor of (Tokens, Batch, Dim)
            seq_encoder_states: ty.List[torch.Tensor] = encoder_out["encoder_states"]
            encoder_layer2states = _encoder_post_process_tensor(seq_encoder_states, target_encoder_layers)
        # end with
    
        # -----------------------------------------------------------
        # Decoder Processing
        # attaching the hook to the decoder layers
        for layer in fairseq_interface.models[0].decoder.layers:  # type: ignore
            layer.register_forward_hook(capture_hidden_states)
        # end for

        with torch.no_grad():        
            generation_output = fairseq_interface.generate(source_tokens_one, # type: ignore
                                                           beam=5,
                                                           max_len_a=max_len_a_mt,
                                                           max_len_b=max_len_b_mt,)
            
            # Note: Hereafter, I look at only the 1st beam.
            assert len(generation_output) > 0, f"generation_output={generation_output}"
            assert "tokens" in generation_output[0], f"missing key 'tokens' in generation_output[0]={generation_output[0]}"
            assert isinstance(generation_output[0]['tokens'], torch.Tensor), f"generation_output[0]['tokens']={generation_output[0]['tokens']}"  # type: ignore

            text_translation = fairseq_interface.decode(generation_output[0]['tokens'])  # type: ignore
            
            # `decoder_states` is a list of tensors. The list length is (N-Tokens + 1) * N-Layers.
            _n_tokens = len(generation_output[0]['tokens'])  # type: ignore
            assert len(decoder_states) == (len(self.decoder_layer_name) * (_n_tokens + 1)), f"recorded decoder states -> {len(decoder_states)}. Expected -> {len(self.decoder_layer_name) * (_n_tokens + 1)}"  # (N-Tokens + 1), +1 is for the <BOS> token.

            # post-process the decoder states. `decoder_states`.
            decoder2states = {}  # key: decoder layer index, values: tensor of (Tokens, Dim) 

            _counter = 0
            for _ind_layer, _layer in enumerate(self.decoder_layer_name):
                _seq_tensor_states_layer = []  # I initialize it with list since the dim. size may not be fixed over the network. 
                for _ind_token in range(_n_tokens + 1):  # +1 is for the <BOS> token.
                    _vector_hidden_states = decoder_states[_counter]  # (Batch-size, Beam-size, Dim)
                    _seq_tensor_states_layer.append(_vector_hidden_states[0, 0, :])  # (Dim)
                    _counter += 1
                # end for
                _tensor_state_layer = torch.stack(_seq_tensor_states_layer, dim=0)  # (N-Layers, Dim)
                assert len(_tensor_state_layer) == (_n_tokens + 1)
                
                if target_decoder_layers is None:
                    decoder2states[_ind_layer] = _tensor_state_layer.cpu()
                elif _ind_layer in target_decoder_layers:
                    decoder2states[_ind_layer] = _tensor_state_layer.cpu()
                else:
                    pass  # do nothing.
                # end if
            # end for
        # end with

        return_tuple = ReturnTuple_method_extract_hidden_states(
            source_text=source_text,
            translated_text=text_translation,
            encoder_layer2states=encoder_layer2states,
            decoder_layer2states=decoder2states,
        )

        return return_tuple

    def extract_encoder_output(self,
                               source_text: str) -> ty.Dict[str, torch.Tensor]:
        """Extract encoder output from a sentence using a pre-trained Fairseq model.
        
        Returns:
            - dict: A dictionary containing:
                - token_ids: The token IDs of the input sentence.
                - encoder_out: The encoder output tensor.
        """
        # Check the device
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

        fairseq_interface = self.model.to(device)

        # The `encode` method is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/hub_utils.py#L246-L249
        # The method does: tokenize, apply_bpe, and binarize.
        tensor_source_tokens = fairseq_interface.encode(source_text).unsqueeze(0)
        tensor_source_tokens = tensor_source_tokens.to(device)

        # get the encoder object
        with torch.no_grad():
            transformer_encoder_obj: TransformerEncoderBase = fairseq_interface.models[0].encoder
            obj_encoder_out = transformer_encoder_obj.forward(tensor_source_tokens)
        # end with

        # this is the word-embedding layer. 
        encoder_embedding = obj_encoder_out['encoder_embedding'][0]
        # this is the encoder output. (Token, Batch, Embedding-Size)
        encoder_out = obj_encoder_out['encoder_out'][0]
        
        n_tokens = tensor_source_tokens.size(1)
        assert n_tokens == encoder_out.size(0), f"n_tokens={n_tokens}, encoder_out.size(0)={encoder_out.size(0)}"

        # Remove the second axis -> (Token, Embedding-Size)
        encoder_tensor_squeezed = encoder_out.squeeze(1)

        return {
            "token_ids": tensor_source_tokens.cpu()[0],
            "encoder_out": encoder_tensor_squeezed.cpu(),
        }


    def extract_word_embeddings(self, 
                                translated_text: str,
                                tensor_translated_text: ty.Optional[torch.Tensor] = None
                                ) -> torch.Tensor:
        """Extract word embeddings from a sentence using a pre-trained Fairseq model.
        
        
        Return:
            - torch.Tensor: The tensor of embeddings. The tensor is (seq_len, embed_dim).
        """
        _model = self.model.to('cpu')


        if tensor_translated_text is None:
            # Tokenize and convert to indices
            # tokenized = model.bpe.encode(sentence).split()
            # token_indices = [model.task.source_dictionary.index(token) for token in tokenized]
            token_indices = _model.encode(translated_text)
            token_indices = token_indices.to('cpu')
            token_tensor = torch.tensor(token_indices).unsqueeze(0)  # Batch size of 1
        else:
            # Use the provided tensor
            assert len(tensor_translated_text.shape) == 1, "The given tensor must be a sequence of token indices"
            token_tensor = tensor_translated_text
        # end if

        # # Check the device
        # if torch.cuda.is_available():
        #     # TODO: multiple GPUs
        #     device = torch.device("cuda:0")
        # else:
        #     device = torch.device("cpu")    
        # # end if
        # Convert token indices to embeddings
        with torch.no_grad():
            embeddings = _model.models[0].decoder.embed_tokens(token_tensor)
        # end with

        if tensor_translated_text is None:
            assert len(embeddings) == 1, f"Expected 1 tensor, got {len(embeddings)}"
            # the tensor is (1, seq_len, embed_dim). I extract (seq_len, embed_dim)
            return embeddings[0]
        else:
            return embeddings
        # end if

    def extract_word_embeddings_batch(self,
                                      seq_sentence: ty.List[str],
                                      target_lang: ty.Optional[str] = None,
                                      seq_token_id_tensor: ty.Optional[ty.List[torch.Tensor]] = None,
                                    ) -> ty.List[torch.Tensor]:
        seq_tesnor = []
        for _sentence in seq_sentence:
            _t = self.extract_word_embeddings(_sentence)
            seq_tesnor.append(_t)
        # end for
        return seq_tesnor

