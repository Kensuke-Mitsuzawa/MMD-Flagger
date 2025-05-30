from pathlib import Path
import toml

from hallucination_mt.module_flagging import utils
from hallucination_mt.baselines.mc_dropout.fairseq_handler.mc_dropout import DissimilarityMcDropOut


def test_mc_dropout(resource_path_root: Path):
    input_sentence = "Empfehlenswert gleich mit der Zimmerreservierung zu buchen!"

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
    
    dissimilarity_mc_dropout = DissimilarityMcDropOut(
        fairseq_interface=model_encoder_decoder_mt
    )
    result_obj = dissimilarity_mc_dropout.run_inference(
        source_text=input_sentence,
        num_samples=10
    )
    assert result_obj is not None, f"result_obj={result_obj}"
# end def