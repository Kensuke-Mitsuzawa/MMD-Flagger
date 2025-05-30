from pathlib import Path
import toml

from hallucination_mt.module_flagging import utils
from hallucination_mt.module_translation_handler.ver1 import module_fairseq_handler


def test_fixed_random_seed(resource_path_root: Path):
    path_config = resource_path_root / "config.toml"    
    assert path_config.exists(), f"path_config={path_config}"

    with open(path_config, "r") as f:
        config_obj = toml.load(f)
        assert "path_fairseq_model" in config_obj
        assert "path_dataset" in config_obj
    # end with

    config_obj_fairseq_model = config_obj["path_fairseq_model"]
    config_obj_dataset = config_obj["path_dataset"]

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=Path(config_obj_fairseq_model["path_fairseq_model_dir"]),
        path_fairseq_model_file=Path(config_obj_fairseq_model["path_fairseq_model_file"]),
        path_sentencepiece_model=Path(config_obj_fairseq_model["path_sentencepiece_model"])
    )

    handler = module_fairseq_handler.FaiseqTranslationModelHandler(
        model_encoder_decoder_mt=model_encoder_decoder_mt,
        is_sampling=True,
        n_sampling=25,
        random_seed=42)
    
    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

    sample_res_1st = handler.sample_multiple_times(
        input_text=input_text,
        temperature=0.5,
        n_sampling=25
    )
    seq_translation_1st = [_obj.target_text for _obj in sample_res_1st]
    sample_res_2nd = handler.sample_multiple_times(
        input_text=input_text,
        temperature=0.5,
        n_sampling=25
    )
    seq_translation_2nd = [_obj.target_text for _obj in sample_res_2nd]

    assert seq_translation_1st == seq_translation_2nd, f"seq_translation_1st={seq_translation_1st}, seq_translation_2nd={seq_translation_2nd}"
    # end assert


def test_random_seed_init(resource_path_root: Path):
    path_config = resource_path_root / "config.toml"    
    assert path_config.exists(), f"path_config={path_config}"

    with open(path_config, "r") as f:
        config_obj = toml.load(f)
        assert "path_fairseq_model" in config_obj
        assert "path_dataset" in config_obj
    # end with

    config_obj_fairseq_model = config_obj["path_fairseq_model"]
    config_obj_dataset = config_obj["path_dataset"]

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=Path(config_obj_fairseq_model["path_fairseq_model_dir"]),
        path_fairseq_model_file=Path(config_obj_fairseq_model["path_fairseq_model_file"]),
        path_sentencepiece_model=Path(config_obj_fairseq_model["path_sentencepiece_model"])
    )

    handler = module_fairseq_handler.FaiseqTranslationModelHandler(
        model_encoder_decoder_mt=model_encoder_decoder_mt,
        is_sampling=True,
        n_sampling=25,
        random_seed=-1)

    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."
    sample_res_1st = handler.sample_multiple_times(
        input_text=input_text,
        temperature=0.5,
        n_sampling=25
    )
    seq_translation_1st = [_obj.target_text for _obj in sample_res_1st]
    sample_res_2nd = handler.sample_multiple_times(
        input_text=input_text,
        temperature=0.5,
        n_sampling=25
    )
    seq_translation_2nd = [_obj.target_text for _obj in sample_res_2nd]
    assert seq_translation_1st != seq_translation_2nd, f"seq_translation_1st={seq_translation_1st}, seq_translation_2nd={seq_translation_2nd}"
    # end assert