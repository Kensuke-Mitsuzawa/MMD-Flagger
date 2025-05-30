from pathlib import Path
import toml

from hallucination_mt.module_flagging import utils
from hallucination_mt.baselines.mc_dropout.fairseq_handler.flagger_mc_dropout import (
    FlaggerDisSimilarityMcDropOut, 
    OutputDisSimilarityMcDropOut,
)
from hallucination_mt.commons.data_models import EvaluationTargetTranslationPair
from hallucination_mt.guerreiro_2023_wmt.data_models.utils import load_dataset


def test_flagger_mc_dropout(resource_path_root: Path):
    input_sentence = "Empfehlenswert gleich mit der Zimmerreservierung zu buchen!"

    path_config = resource_path_root / "config.toml"
    # path_config = resource_path_root / "config_local.toml"    
    assert path_config.exists(), f"path_config={path_config}"

    with open(path_config, "r") as f:
        config_obj = toml.load(f)
        assert "path_fairseq_model" in config_obj
        assert "path_dataset" in config_obj
    # end with

    config_obj_fairseq_model = config_obj["path_fairseq_model"]
    config_obj_dataset = config_obj["path_dataset"]

    path_dataset_tsv = Path(config_obj_dataset["path_dataset_tsv"])
    seq_dataset = load_dataset(path_dataset_tsv, delimiter="\t")
    seq_correct_translation = [__record for __record in seq_dataset if __record.error_type == "correct"]


    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=Path(config_obj_fairseq_model["path_fairseq_model_dir"]),
        path_fairseq_model_file=Path(config_obj_fairseq_model["path_fairseq_model_file"]),
        path_sentencepiece_model=Path(config_obj_fairseq_model["path_sentencepiece_model"])
    )

    flagger = FlaggerDisSimilarityMcDropOut(
        fairseq_interface=model_encoder_decoder_mt,
        num_samples=2
    )

    seq_datasets = [
        EvaluationTargetTranslationPair(
            source=__r.source,
            target=__r.translation,
            sentence_id=str(__r.sentence_id)
        )
        for __r in seq_correct_translation[:5]]
    seq_distribution_dataset = flagger.compute_dataset_statistics(dataset=seq_datasets)
    threshold = flagger.get_flag_threshold(seq_distribution_dataset)
    eval_target = EvaluationTargetTranslationPair(
        source=input_sentence,
        target="",
        sentence_id="0"
    )
    out = flagger.flag(
        evaluation_target=eval_target,
        threshold=threshold)

    assert isinstance(out, OutputDisSimilarityMcDropOut), f"Expected OutputDisSimilarityMcDropOut, got {type(out)}"
    assert out.source_text == input_sentence, f"Expected {input_sentence}, got {out.source_text}"