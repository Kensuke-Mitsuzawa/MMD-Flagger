import typing as ty
from pathlib import Path
import logging

import numpy as np
import torch

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedModel)

from .module_base import (BaseVectorExtractor, OPTION_EMBEDDING_LAYERS, ReturnTuple_method_extract_hidden_states)
from ...module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ...module_translation_handler.ver1 import TransformersTranslationModelHandler

module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())




class TransformerVectorExtractor(BaseVectorExtractor):
    def __init__(self,
                 translation_handler: TransformersTranslationModelHandler
                 ) -> None:
        super().__init__()
        self.translation_handler = translation_handler
        self.is_select_gpu_flexible = translation_handler.is_select_gpu_flexible

        _seq_encoder_layer_name, _seq_decoder_layer_name = self.get_all_possible_layers()
        self.encoder_layer_name = _seq_encoder_layer_name
        self.decoder_layer_name = _seq_decoder_layer_name

    def get_all_possible_layers(self) -> ty.Tuple[ty.List, ty.List]:
        """
        Return:
            - seq_encoder_layers: list of encoder layers, `transformers.models.m2m_100.modeling_m2m_100.M2M100EncoderLayer`
            - seq_decoder_layers: list of decoder layers, `transformers.models.m2m_100.modeling_m2m_100.M2M100DecoderLayer`
        """
        index_encoder_layers = self.translation_handler.model.config.encoder_layers
        index_decoder_layers = self.translation_handler.model.config.decoder_layers

        seq_encoder_layers = list(self.translation_handler.model.model.encoder.layers)
        seq_decoder_layers = list(self.translation_handler.model.model.decoder.layers)        

        return seq_encoder_layers, seq_decoder_layers

    def extract_encoder_output_teacher_forcing(self,
                                               source_text: str,
                                               target_translation_text: str,
                                               target_encoder_layers: ty.Optional[ty.List[int]] = None,
                                               target_decoder_layers: ty.Optional[ty.List[int]] = None):
        inputs_tokens_obj = self.translation_handler.tokenizer(source_text, return_tensors="pt")
        inputs_tokens_obj.to(self.translation_handler.device)
        tgt_lang_id = self.translation_handler.tokenizer.convert_tokens_to_ids(self.translation_handler.target_lang)
        assert isinstance(tgt_lang_id, int), f"Unexpected type of the target language ID: {type(tgt_lang_id)}"

        # Tokenizer at the target language
        translation_target_lang = self.translation_handler.target_lang
        target_lang_tokenizer_obj = AutoTokenizer.from_pretrained(self.translation_handler.tokenizer.name_or_path, src_lang=translation_target_lang)
        target_tokens_obj = target_lang_tokenizer_obj(target_translation_text, return_tensors="pt")
        target_tokens_obj.to(self.translation_handler.device)

        with torch.no_grad():
            outputs = self.translation_handler.model(
                input_ids=inputs_tokens_obj["input_ids"],
                attention_mask=inputs_tokens_obj["attention_mask"],
                labels=target_tokens_obj["input_ids"],  # this enables teacher forcing
                output_hidden_states=True,
                return_dict=True)

            encoder_hidden_states = outputs.encoder_hidden_states  # tuple object. An element is a tensor. (Batch, N-Token, Dimension)
            decoder_hidden_states = outputs.decoder_hidden_states  # tuple object. An element is a tensor. (Batch, N-Token, Dimension)
        # end with

        assert len(encoder_hidden_states) == len(self.encoder_layer_name) + 1, \
            f"len(encoder_hidden_states)={len(encoder_hidden_states)}, len(self.seq_encoder_layers)={len(self.encoder_layer_name)}"
        assert len(decoder_hidden_states) == len(self.decoder_layer_name) + 1, \
            f"len(decoder_hidden_states)={len(decoder_hidden_states)}, len(self.seq_decoder_layers)={len(self.decoder_layer_name)}"
        
        


    def extract_hidden_states(self,
                              source_text: str,
                              target_encoder_layers: ty.Optional[ty.List[int]] = None,
                              target_decoder_layers: ty.Optional[ty.List[int]] = None,
                              generation_kwargs: ty.Optional[ty.Dict[str, ty.Any]] = None,
                              ) -> ReturnTuple_method_extract_hidden_states:
        # I have to cleanup the interface. 
        raise NotImplementedError()

        source_token_obj = self.translation_handler.tokenizer(source_text, return_tensors="pt")
        source_token_obj.to(self.translation_handler.device)

        # Generation parameters
        max_token_length = self.translation_handler._calculate_max_length(source_token_obj)

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



        dict_encoder_layer2hidden_vector = {}
        with torch.no_grad():
            encoder_outputs = self.translation_handler.model.model.encoder(
                input_ids=source_token_obj["input_ids"],
                attention_mask=source_token_obj["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            encoder_hidden_states = encoder_outputs.hidden_states  # tuple object. An element is a tensor. (Batch, N-Token, Dimension)

            for __ind_encoder_layer, layer in enumerate(encoder_hidden_states):
                if target_encoder_layers is None:
                    dict_encoder_layer2hidden_vector [__ind_encoder_layer] = layer
                elif __ind_encoder_layer in target_encoder_layers:
                    dict_encoder_layer2hidden_vector[__ind_encoder_layer] = layer
                # end if
            # end for
        # end with                

        dict_decoder_layer2hidden_vector = {}
        with torch.no_grad():
            generated_tokens = self.translation_handler.model.generate(
                source_token_obj["input_ids"],
                attention_mask=source_token_obj["attention_mask"],
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=False)  # optional

            decoder_hidden_states = generated_tokens.decoder_hidden_states
            # print(len(decoder_hidden_states))  # N + 1 for decoder

            for __ind_decoder_layer, layer in enumerate(decoder_hidden_states):
                if target_decoder_layers is None:
                    dict_decoder_layer2hidden_vector[__ind_decoder_layer] = layer
                elif __ind_decoder_layer in target_decoder_layers:
                    dict_decoder_layer2hidden_vector[__ind_decoder_layer] = layer
                # end if
            # end for
        # end with

        translated_text = self.translation_handler.tokenizer.decode(generated_tokens["sequences"][0], skip_special_tokens=True)

        return ReturnTuple_method_extract_hidden_states(
            source_text=source_text,
            translated_text=translated_text,
            encoder_layer2states=dict_encoder_layer2hidden_vector,
            decoder_layer2states=dict_decoder_layer2hidden_vector,
        )

        
    def extract_word_embeddings(self,
                                translated_text: str,
                                tensor_translated_text: ty.Optional[torch.Tensor] = None,
                                ) -> torch.Tensor:
        """
        Return:
            - embedding_vectors: torch.Tensor. The shape of (N-tokens, Dimension_of_vector)
        
        """

        if tensor_translated_text is None:
            # Tokenize the translated text for the target language
            inputs = self.translation_handler.tokenizer(translated_text, return_tensors="pt").to(self.translation_handler.model.device)
            # Get the input IDs
            input_ids = inputs['input_ids']
        else:
            input_ids = tensor_translated_text
            input_ids = input_ids.to(self.translation_handler.model.device)
        # end if

        # get decoder
        if hasattr(self.translation_handler.model, "decoder"):
            decoder_model = self.translation_handler.model.decoder
        elif hasattr(self.translation_handler.model, "get_decoder"):
            decoder_model = self.translation_handler.model.get_decoder()
        else:
            raise ValueError("The model does not have a decoder.")
        # end if

        # Note: the class is `"<class 'transformers.models.m2m_100.modeling_m2m_100.M2M100Decoder'>"`
        # Access the decoder's embedding layer
        assert hasattr(decoder_model, "embed_tokens"), "The decoder model does not have an embedding layer."
        decoder_embedding_layer = decoder_model.embed_tokens

        # Perform a forward pass through the embedding layer
        with torch.no_grad():
            embeddings = decoder_embedding_layer(input_ids).cpu()

        # The 'embeddings' tensor will have the shape (batch_size, sequence_length, embedding_dimension)
        # In this case, batch_size is 1.
        embedding_vectors = embeddings.squeeze(0)  # Remove the batch dimension

        # 'embedding_vectors' now has the shape (N_tokens, Dimension_of_vector)
        num_tokens = embedding_vectors.shape[0]
        vector_dimension = embedding_vectors.shape[1]

        if len(input_ids.shape) == 2:
            assert num_tokens == input_ids.shape[1], f"Number of tokens {num_tokens} does not match input IDs {input_ids.shape[0]}"            
        else:
            assert num_tokens == input_ids.shape[0], f"Number of tokens {num_tokens} does not match input IDs {input_ids.shape[0]}"

        return embedding_vectors

    def extract_word_embeddings_batch(self,
                                      seq_sentence: ty.List[str],
                                      seq_token_id_tensor: ty.Optional[ty.List[torch.Tensor]] = None,
                                    ) -> ty.List[torch.Tensor]:

        seq_tesnor = []

        if seq_token_id_tensor is not None:
            assert len(seq_sentence) == len(seq_token_id_tensor), \
                f"seq_sentence={len(seq_sentence)}, seq_token_id_tensor={len(seq_token_id_tensor)}"
            # If the token ID tensor is provided, use it
            for _sentence, _token_tensor in zip(seq_sentence, seq_token_id_tensor):
                _t = self.extract_word_embeddings(translated_text=_sentence,
                                                  tensor_translated_text=_token_tensor)
                seq_tesnor.append(_t)
            # end for
            return seq_tesnor
        else:
            for _sentence in seq_sentence:
                _t = self.extract_word_embeddings(translated_text=_sentence)
                seq_tesnor.append(_t)
            # end for
            return seq_tesnor
