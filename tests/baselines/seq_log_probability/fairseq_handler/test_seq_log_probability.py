from pathlib import Path
import toml

from hallucination_mt.module_flagging import utils
from hallucination_mt.baselines.seq_log_probability import ComputeSequenceLogProbability


def test_seq_log_probability(resource_path_root: Path):
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

    compute_sequence_log_probability = ComputeSequenceLogProbability(
        fairseq_interface=model_encoder_decoder_mt
    )
    result_obj = compute_sequence_log_probability(
        source_text=input_sentence,
        temperature=1.0,
        beam=5,
        max_len_a=0.0,
        max_len_b=200
    )
    assert result_obj is not None, f"result_obj={result_obj}"
# end def