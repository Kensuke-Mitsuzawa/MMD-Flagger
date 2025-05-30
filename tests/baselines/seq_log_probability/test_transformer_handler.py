from pathlib import Path
import toml

from hallucination_mt.module_flagging import utils
from hallucination_mt.baselines.seq_log_probability import TransformerFlaggerSeqLogProbability
from hallucination_mt.commons.data_models import EvaluationTargetTranslationPair
from hallucination_mt.module_translation_handler.ver1.module_transformer_handler import TransformersTranslationModelHandler
from hallucination_mt.dale_2023_halomi.load_dataset import load_dataset


def test_flagger_seq_log_probability(resource_path_root: Path):
    input_sentence = "Empfehlenswert gleich mit der Zimmerreservierung zu buchen!"

    path_dataset_tsv = resource_path_root / "eval_datasets/halomi_deu_eng_subset.tsv"
    seq_dataset = load_dataset(path_dataset_tsv)

    seq_dataset_de_en = [__r for __r in seq_dataset if __r.src_lang == "deu_Latn" and __r.tgt_lang == "eng_Latn"]
    assert len(seq_dataset_de_en) > 0, "seq_dataset_de_en is empty"

    translation_handler = TransformersTranslationModelHandler(
        src_lang="de_Latn",
        target_lang="en_Latn",
    )

    flagger = TransformerFlaggerSeqLogProbability(translation_handler)

    seq_datasets = [
        EvaluationTargetTranslationPair(
            source=__r.src_text,
            target=__r.tgt_text,
            sentence_id=str(__r.key_unique)
        )
        for __r in seq_dataset_de_en[:20]]
    seq_distribution_dataset = flagger.compute_dataset_log_probability(dataset=seq_datasets)
    threshold = flagger.get_flag_threshold(seq_log_probability=seq_distribution_dataset, percentile=0.4)
    out = flagger.flag(
        source_text=input_sentence,
        threshold=threshold)
