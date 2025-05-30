from pathlib import Path
import toml

from hallucination_mt.module_translation_handler.ver1 import module_transformer_handler


def test_TransformersTranslationModelHandler_fixed_random_seed(resource_path_root: Path):
    path_config = resource_path_root / "config.toml"    
    assert path_config.exists(), f"path_config={path_config}"

    with open(path_config, "r") as f:
        config_obj = toml.load(f)
    # end with

    assert 'dataset_halomi' in config_obj, "Key 'dataset_halomi' not found in the config object."
    config_obj_halomi = config_obj['dataset_halomi']
    assert 'models' in config_obj_halomi, "Key 'models' not found in the config object."
    code_name_nllb: str = config_obj_halomi['models']['code_name_nllb']


    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

    # the 1st translation handler
    translation_handler_1st = module_transformer_handler.TransformersTranslationModelHandler(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        batch_size=5,
        max_len_a=1.0,
        max_len_b=10,
    )
    result_1st = translation_handler_1st.sample_multiple_times(
        input_text=input_text,
        temperature=1.0,
        n_sampling=10,
        is_sampling_in_iteration=False
    )

    # the 2nd translation handler
    translation_handler_2nd = module_transformer_handler.TransformersTranslationModelHandler(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        batch_size=5,
        max_len_a=1.0,
        max_len_b=10,
    )
    result_2nd = translation_handler_2nd.sample_multiple_times(
        input_text=input_text,
        temperature=1.0,
        n_sampling=10,
        is_sampling_in_iteration=False
    )

    # Compare the results
    seq_translation_1st = [_obj.translation_text for _obj in result_1st]
    seq_translation_2nd = [_obj.translation_text for _obj in result_2nd]    
    
    assert len(seq_translation_1st) == len(seq_translation_2nd), "The lengths of the translation results do not match."
    assert seq_translation_1st == seq_translation_2nd, "The translation results do not match."


def test_TransformersTranslationModelHandler_translate_beam_search(resource_path_root: Path):
    path_config = resource_path_root / "config.toml"    
    assert path_config.exists(), f"path_config={path_config}"

    with open(path_config, "r") as f:
        config_obj = toml.load(f)
    # end with

    assert 'dataset_halomi' in config_obj, "Key 'dataset_halomi' not found in the config object."
    config_obj_halomi = config_obj['dataset_halomi']
    assert 'models' in config_obj_halomi, "Key 'models' not found in the config object."
    code_name_nllb: str = config_obj_halomi['models']['code_name_nllb']


    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

    translation_handler = module_transformer_handler.TransformersTranslationModelHandler(
        src_lang="deu_Latn",
        target_lang="eng_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        batch_size=2,
        max_len_a=1.0,
        max_len_b=10
    )
    result = translation_handler.translate_beam_search(
        input_text=input_text,
        temperature=1.0,
    )
    assert isinstance(result.translation_text, str)
    assert isinstance(result.log_score, float)
