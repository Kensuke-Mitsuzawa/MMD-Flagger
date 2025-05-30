from pathlib import Path
import toml
import zlib
import pickle
import shutil

from hallucination_mt.guerreiro_2023_wmt.utils_models.utils import load_model

from hallucination_mt.module_translation_handler.ver2 import module_fairseq_handler
from hallucination_mt.commons.data_models import EvaluationTargetTranslationPair


def test_FairSeqTranslationModelHandlerVer2_fixed_random_seed(resource_path_root: Path):
    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    model_encoder_decoder_mt = load_model(
        path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
        path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
        model_sentencepiece=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    input_text = "## $$ !! Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz"

    translation_handler_1st = module_fairseq_handler.FairSeqTranslationModelHandlerVer2(
        path_dir_fairseq_model=path_dir_model / "wmt18_de-en",
        path_model_checkpoint=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",
        random_seed=42
    )

    input_obj = EvaluationTargetTranslationPair(
        source=input_text,
        sentence_id='test',
        target=''
    )

    res_1st = translation_handler_1st.translate_sample_multiple_times(
        input_text=input_obj,
        n_sampling=5,
        temperature=1.0,
        max_len_a=1.0,
        max_len_b=10)
    
    translation_handler_2nd = module_fairseq_handler.FairSeqTranslationModelHandlerVer2(
        path_dir_fairseq_model=path_dir_model / "wmt18_de-en",
        path_model_checkpoint=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",
        random_seed=42,
        is_save_convert_float16=True
    )
    res_2nd = translation_handler_2nd.translate_sample_multiple_times(
        input_text=input_obj,
        n_sampling=5,
        temperature=1.0,
        max_len_a=1.0,
        max_len_b=10)

    assert len(res_1st) == len(res_2nd)
    for _1st, _2nd in zip(res_1st, res_2nd):
        assert _1st.translation_text == _2nd.translation_text
    # end for
    
    # another random seed
    translation_handler_seed_variant = module_fairseq_handler.FairSeqTranslationModelHandlerVer2(
        path_dir_fairseq_model=path_dir_model / "wmt18_de-en",
        path_model_checkpoint=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",
        random_seed=25,
        is_use_cache=False
    )
    res_variant = translation_handler_seed_variant.translate_sample_multiple_times(
        input_text=input_obj,
        n_sampling=5,
        temperature=1.0,
        max_len_a=1.0,
        max_len_b=10)
    assert len(res_1st) == len(res_variant)
    translation_variation_1st = set([_obj.translation_text for _obj in res_1st])
    translation_variation_variant = set([_obj.translation_text for _obj in res_variant])
    assert translation_variation_1st != translation_variation_variant


def test_FairSeqTranslationModelHandlerVer2_translate_sample_multiple_times(resource_path_root: Path):
    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    model_encoder_decoder_mt = load_model(
        path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
        path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
        model_sentencepiece=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    input_text = "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz"
    input_obj = EvaluationTargetTranslationPair(
        source=input_text,
        target='',
        sentence_id='test'
    )

    translation_handler = module_fairseq_handler.FairSeqTranslationModelHandlerVer2(
        path_dir_fairseq_model=path_dir_model / "wmt18_de-en",
        path_model_checkpoint=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",
        random_seed=42,
        is_save_convert_float16=True
    )
    seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
    target_layers_extraction = seq_decoder[:2]
    seq_result = translation_handler.translate_sample_multiple_times(input_text=input_obj,
                                                        n_sampling=5,
                                                        temperature=1.0,
                                                        max_len_a=1.0,
                                                        max_len_b=10,
                                                        target_layers_extraction=target_layers_extraction,
                                                        is_sampling_in_iteration=False)
    assert len(seq_result) == 5
    for obj in seq_result:
        assert isinstance(obj, module_fairseq_handler.TranslationResultContainer)
        assert obj.dict_layer_embeddings is not None
        for _layer_name in target_layers_extraction:
            assert _layer_name in obj.dict_layer_embeddings
        # end
        assert translation_handler._get_decoder_word_embedding_layer_name() in obj.dict_layer_embeddings
    # end for

    # check that the translation has variations. There should be variations by stochastic sampling.
    variation_translations = set([obj.translation_text for obj in seq_result])
    assert len(set(variation_translations)) > 1


def test_FairSeqTranslationModelHandlerVer2_sampling_single_input(resource_path_root: Path):
    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    # model_encoder_decoder_mt = load_model(
    #     path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
    #     path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
    #     model_sentencepiece=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    input_text = "Apfel"

    translation_handler = module_fairseq_handler.FairSeqTranslationModelHandlerVer2(
        path_dir_fairseq_model=path_dir_model / "wmt18_de-en",
        path_model_checkpoint=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model"
    )

    seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
    target_layers = seq_decoder[:2] 

    tensor_source_tokens = translation_handler.fairseq_interface_default_set.encode(input_text)

    seq_result = translation_handler._sampling_single_input(
        source_text=input_text,
        tensor_source_tokens=tensor_source_tokens,
        n_sampling=5,
        max_len_a=1.0,
        max_len_b=5,
        temperature=0.5,
        target_layers_extraction=target_layers
    )

    assert len(seq_result) == 5    
    for _result in seq_result:
        assert isinstance(_result, module_fairseq_handler.TranslationResultContainer)
        assert _result is not None
        assert _result.dict_layer_embeddings is not None
        for _key_name in target_layers:
            assert _key_name in _result.dict_layer_embeddings
        # end for
        assert translation_handler._get_decoder_word_embedding_layer_name() in _result.dict_layer_embeddings


# def test_FairSeqTranslationModelHandlerVer2_sampling_multi_input(resource_path_root: Path):
#     path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
#     assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

#     model_encoder_decoder_mt = load_model(
#         path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
#         path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
#         model_sentencepiece=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

#     input_text = "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz"

#     translation_handler = module_fairseq_handler.FairSeqTranslationModelHandlerVer2(
#         model_encoder_decoder_mt=model_encoder_decoder_mt,
#         random_seed=42
#     )

#     seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
#     target_layers = seq_decoder[:2]

#     tensor_source_tokens = translation_handler.model_encoder_decoder_mt.encode(input_text)

#     seq_result = translation_handler._sampling_multi_input(
#         source_text=input_text,
#         tensor_source_tokens=tensor_source_tokens,
#         n_sampling=6,
#         max_len_a=1.0,
#         max_len_b=5,
#         temperature=1.0,
#         target_layers_extraction=target_layers,
#         batch_size=3
#     )

#     assert len(seq_result) == 6

#     for _result in seq_result:
#         assert _result.dict_layer_embeddings is not None
#         for _key_name in target_layers:
#             assert _key_name in _result.dict_layer_embeddings
#         # end for

#         assert isinstance(_result.log_probability_score, float)
#         assert isinstance(_result.argument_translation_conditions, dict)
#         assert 'random_seed' in _result.argument_translation_conditions
#         assert translation_handler._get_decoder_word_embedding_layer_name() in _result.dict_layer_embeddings
#     # end for


def test_FairSeqTranslationModelHandlerVer2_translate_beam_search(resource_path_root: Path):

    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    model_encoder_decoder_mt = load_model(
        path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
        path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
        model_sentencepiece=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    input_text = "Apfel"
    input_obj = EvaluationTargetTranslationPair(
        source=input_text,
        target='',
        sentence_id='test'
    )


    translation_handler = module_fairseq_handler.FairSeqTranslationModelHandlerVer2(
        path_dir_fairseq_model=path_dir_model / "wmt18_de-en",
        path_model_checkpoint=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",
        is_save_convert_float16=True
    )

    seq_encoder, seq_decoder = translation_handler.get_all_possible_layers()
    target_layers = seq_decoder[:2] 
    result = translation_handler.translate_beam_search(
        input_text=input_obj,
        temperature=1.0,
        target_layers_extraction=target_layers
    )

    assert result.dict_layer_embeddings is not None
    for _key_name in target_layers:
        assert _key_name in result.dict_layer_embeddings
        assert translation_handler._get_decoder_word_embedding_layer_name() in result.dict_layer_embeddings
    # end for

    path_file = translation_handler._generate_cache_file_path(
        input_obj.sentence_id,
        tau_parameter=1.0,
        n_sampling=None)
    assert path_file.exists()
    with path_file.open('rb') as f:
        _obj_load = pickle.loads(zlib.decompress(f.read()))
        module_fairseq_handler.TranslationResultContainer(**_obj_load)
    # end with
    
    shutil.rmtree(translation_handler.path_cache_dir)

