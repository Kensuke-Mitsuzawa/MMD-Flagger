from pathlib import Path
import toml
import shutil
import pickle
import zlib
import torch

from hallucination_mt.module_translation_handler.ver2 import module_transformer_handler
from hallucination_mt.commons.data_models import EvaluationTargetTranslationPair


def test_TransformerTranslationModelHandlerVer2_fixed_random_seed(resource_path_root: Path):
    code_name_nllb = "facebook/nllb-200-distilled-600M"

    input_text = "Apfel"

    # the 1st translation handler
    translation_handler_1st = module_transformer_handler.TransformersTranslationModelHandlerVer2(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
    )


def test_TransformerTranslationModelHandlerVer2_sampling_single_input(resource_path_root: Path):
    code_name_nllb: str = "facebook/nllb-200-distilled-600M"

    input_text = "Apfel"

    translation_handler = module_transformer_handler.TransformersTranslationModelHandlerVer2(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        is_save_convert_float16=True
    )

    seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
    target_layers = seq_decoder[:2]

    tensor_source_tokens = translation_handler.tokenizer(text=input_text, return_tensors="pt")

    seq_result = translation_handler._sampling_single_input(
        source_text=input_text,
        tensor_source_tokens=tensor_source_tokens,
        n_sampling=5,
        max_len_a=1.0,
        max_len_b=5,
        temperature=0.5,
        target_layers_extraction=target_layers
    )

    name_layer_decoder_word_emb = translation_handler._get_decoder_word_embedding_layer_name()
    for _result in seq_result:
        assert _result.dict_layer_embeddings is not None
        assert isinstance(_result.dict_layer_embeddings, dict)
        for _key_name in target_layers:
            assert _key_name in _result.dict_layer_embeddings
            assert isinstance(_result.dict_layer_embeddings[_key_name], torch.Tensor)
        # end for
        assert name_layer_decoder_word_emb in _result.dict_layer_embeddings
    # end for


def test_TransformerTranslationModelHandlerVer2_sampling_multi_input(resource_path_root: Path):
    code_name_nllb: str = "facebook/nllb-200-distilled-600M"

    input_text = "Apfel"

    translation_handler = module_transformer_handler.TransformersTranslationModelHandlerVer2(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        is_save_convert_float16=True
    )

    seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
    target_layers = seq_decoder[:2]

    tensor_source_tokens = translation_handler.tokenizer(text=input_text, return_tensors="pt")

    seq_result = translation_handler._sampling_multi_input(
        source_text=input_text,
        tensor_source_tokens=tensor_source_tokens,
        n_sampling=5,
        max_len_a=1.0,
        max_len_b=5,
        temperature=0.5,
        target_layers_extraction=target_layers,
        batch_size=3
    )

    name_layer_decoder_word_emb = translation_handler._get_decoder_word_embedding_layer_name()
    for _result in seq_result:
        assert _result.dict_layer_embeddings is not None
        assert isinstance(_result.dict_layer_embeddings, dict)
        for _key_name in target_layers:
            assert _key_name in _result.dict_layer_embeddings
        # end for
        assert name_layer_decoder_word_emb in _result.dict_layer_embeddings
    # end for

def test_TransformerTranslationModelHandlerVer2_translate_beam_search(resource_path_root: Path):
    code_name_nllb: str = "facebook/nllb-200-distilled-600M"


    input_text = "Apfel"

    translation_handler = module_transformer_handler.TransformersTranslationModelHandlerVer2(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        is_save_convert_float16=True
    )

    seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
    target_layers = seq_decoder[:2] 

    input_text = EvaluationTargetTranslationPair(
        sentence_id='test',
        source=input_text,
        target=''
    )

    result = translation_handler.translate_beam_search(
        input_text=input_text,
        temperature=1.0,
        target_layers_extraction=target_layers
    )

    assert result.dict_layer_embeddings is not None
    for _key_name in target_layers:
        assert _key_name in result.dict_layer_embeddings
    # end for

    path_file = translation_handler._generate_cache_file_path(
        input_text.sentence_id,
        tau_parameter=1.0,
        n_sampling=None)
    assert path_file.exists()
    with path_file.open('rb') as f:
        _obj_load = pickle.loads(zlib.decompress(f.read()))
        obj = module_transformer_handler.TranslationResultContainer(**_obj_load)
        assert obj.dict_layer_embeddings is not None
        for _k, _v in obj.dict_layer_embeddings.items():
            assert isinstance(obj.dict_layer_embeddings[_k], torch.Tensor)
            assert obj.dict_layer_embeddings[_k].dtype == torch.float16
    # end with
    
    shutil.rmtree(translation_handler.path_cache_dir)


def test_TransformerTranslationModelHandlerVer2_translate_sample_multiple_times(resource_path_root: Path):
    code_name_nllb: str = "facebook/nllb-200-distilled-600M"


    input_text = "Apfel"

    translation_handler = module_transformer_handler.TransformersTranslationModelHandlerVer2(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        is_save_convert_float16=True
    )

    seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
    target_layers = seq_decoder[:2] 

    input_text = EvaluationTargetTranslationPair(
        sentence_id='test',
        source=input_text,
        target=''
    )

    seq_result = translation_handler.translate_sample_multiple_times(
        input_text=input_text,
        temperature=1.0,
        target_layers_extraction=target_layers,
        n_sampling=5,
        max_len_a=1.0,
        max_len_b=10,
        n_max_attempts=2
    )

    for result in seq_result:
        assert result.dict_layer_embeddings is not None
        for _key_name in target_layers:
            assert _key_name in result.dict_layer_embeddings
        # end for
    # end for

    path_file = translation_handler._generate_cache_file_path(
        input_text.sentence_id,
        tau_parameter=1.0,
        n_sampling=5)
    assert path_file.exists()
    with path_file.open('rb') as f:
        _obj_load = pickle.loads(zlib.decompress(f.read()))
        assert isinstance(_obj_load, list)
        seq_obj = [module_transformer_handler.TranslationResultContainer(**o) for o in _obj_load]
        for obj in seq_obj:
            assert obj.dict_layer_embeddings is not None
            for _k, _v in obj.dict_layer_embeddings.items():
                assert isinstance(obj.dict_layer_embeddings[_k], torch.Tensor)
                assert obj.dict_layer_embeddings[_k].dtype == torch.float16
        # end for
    # end with
    
    shutil.rmtree(translation_handler.path_cache_dir)
