import logging
from dataclasses import dataclass
from pathlib import Path
import typing as ty
import tempfile
import shutil
import tqdm


import numpy as np
import torch
from fairseq.hub_utils import GeneratorHubInterface

from ....commons.data_models import EvaluationTargetTranslationPair
from .mc_dropout import OutputMcDropOut, DissimilarityMcDropOut, MeteorMetric, BaseMetric


logger = logging.getLogger(__name__)


class _RecordCache(ty.NamedTuple):
    sentence_id: str
    metric_class_name: str
    source_sentence: str
    translations: ty.List[str]
    avg_dis_similarity: float
    seq_dis_similarity: ty.List[float]

    @classmethod
    def from_dict(cls, dict_obj: ty.Dict[str, ty.Any]) -> "_RecordCache":
        return cls(sentence_id=dict_obj['sentence_id'],
                   metric_class_name=dict_obj['metric_class_name'],
                   source_sentence=dict_obj['source_sentence'],
                   translations=dict_obj['translations'],
                   avg_dis_similarity=dict_obj['avg_dis_similarity'],
                   seq_dis_similarity=dict_obj['seq_dis_similarity'])


@dataclass
class OutputDisSimilarityMcDropOut:
    source_text: str
    translations: ty.List[str]
    avg_dissimilarity: float
    threshold_dissimilarity: float
    is_hallucination: bool
    metric_class: str



class FlaggerDisSimilarityMcDropOut:
    def __init__(self, 
                 fairseq_interface: GeneratorHubInterface,
                 metric_obj: BaseMetric = MeteorMetric(),
                 path_dir_cache: ty.Optional[Path] = None,
                 num_samples: int = 10,
                 temperature_value: float = 1.0,
                 max_len_a: float = 0.0,
                 max_len_b: int = 200
                 ):
        self.executer_mc_drop_out = DissimilarityMcDropOut(
            metric_obj=metric_obj,
            fairseq_interface=fairseq_interface)

        if path_dir_cache is None:
            self.path_dir_cache = Path(tempfile.mkdtemp())
        else:
            assert Path(path_dir_cache).exists(), f"Path {path_dir_cache} does not exist"
            self.path_dir_cache = Path(path_dir_cache)
        # end if

        self.num_samples = num_samples
        self.temperature_value = temperature_value
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

    def _load_cache(self, sentence_id: str, source_text: str) -> _RecordCache:
        path_cache_file = self.path_dir_cache / f"{sentence_id}.pt"
        cache_obj = torch.load(path_cache_file)

        assert isinstance(cache_obj, dict), f"Expected dict, got {type(cache_obj)}"
        assert "source_sentence" in cache_obj, f"Expected key 'source_sentence' in {cache_obj.keys()}"
        assert "sentence_id" in cache_obj, f"Expected key 'sentence_id' in {cache_obj.keys()}"

        assert cache_obj['sentence_id'] == sentence_id, f"Expected source text {source_text}, got {cache_obj['source_sentence']}"
        assert cache_obj['source_sentence'] == source_text, f"Expected source text {source_text}, got {cache_obj['source_sentence']}"
        _cache_obj = _RecordCache.from_dict(cache_obj)
        return _cache_obj

    def compute_dataset_statistics(self, dataset: ty.List[EvaluationTargetTranslationPair]) -> ty.List[float]:
        """I compute the avg-dis-similarity for each record of the dataset.
        
        Returns:
            ty.List[float]: List of log probabilities. A distribution of the log probabilities.
        """
        seq_score = []
        # TODO: I want to introduce tqdm logger.
        for _record in dataset:
            path_cache_file = self.path_dir_cache / f"{_record.sentence_id}.pt"
            if path_cache_file.exists():
                record_cache = self._load_cache(_record.sentence_id, _record.source)
                seq_score.append(record_cache.avg_dis_similarity)
            else:
                __translated_obj = self.executer_mc_drop_out.run_inference(
                    _record.source,
                    num_samples=self.num_samples,
                    temperature_value=self.temperature_value,
                    max_len_a=self.max_len_a,
                    max_len_b=self.max_len_b,)
                _cache_obj = _RecordCache(
                    sentence_id=_record.sentence_id,
                    metric_class_name=__translated_obj.metric_class,
                    source_sentence=_record.source,
                    translations=__translated_obj.outputs,
                    avg_dis_similarity=__translated_obj.avg_dissimilarity,
                    seq_dis_similarity=__translated_obj.seq_dissimilarity)
                _cache_obj_dict =_cache_obj._asdict()

                torch.save(_cache_obj_dict, path_cache_file)
                seq_score.append(__translated_obj.avg_dissimilarity)
            # end if
        # end for
        return seq_score
    
    def get_flag_threshold(self, 
                           seq_dataset_statistics: ty.List[float],
                           percentile: float = 0.004,
                           is_top_percentile: bool = False) -> float:
        """I get the threshold for the flagging.
        
        Args:
            seq_log_probability (ty.List[float]): List of log probabilities.
            percentile (float, optional): Percentile. Defaults to 0.004 (0.4%), suggested by Guerreiro et al. (2023).
            Since the dissimilarity is a metric and a larger value indicates a higher uncertainty, I take (100 - top_percentile) as the threshold.
        """
        array_log_prob = np.array(seq_dataset_statistics)
        if is_top_percentile:
            _percentile = 100 - percentile
        else:
            _percentile = percentile
        # end if

        threshold = np.percentile(array_log_prob, _percentile)
        return threshold.item()

    def flag(self, 
             evaluation_target: EvaluationTargetTranslationPair,
             threshold: float, 
             is_hallucination_larger_threshold: bool = False,
             is_use_cache: bool = False,
             ) -> OutputDisSimilarityMcDropOut:
        
        if is_use_cache:
            path_cache_file = self.path_dir_cache / f"{evaluation_target.sentence_id}.pt"
            if path_cache_file.exists():
                record_cache = self._load_cache(evaluation_target.sentence_id, evaluation_target.source)
                __translated_obj = OutputMcDropOut(
                    source_text=record_cache.source_sentence,
                    reference_text=evaluation_target.target,
                    outputs=record_cache.translations,
                    avg_dissimilarity=record_cache.avg_dis_similarity,
                    metric_class=record_cache.metric_class_name,
                    seq_dissimilarity=record_cache.seq_dis_similarity,
                    temperature_value=self.temperature_value,
                    num_samples=self.num_samples)
            else:
                __translated_obj = None
        else:
            __translated_obj = None
        # end if

        if __translated_obj is None:
            __translated_obj = self.executer_mc_drop_out.run_inference(
                source_text=evaluation_target.source,
                num_samples=self.num_samples,
                temperature_value=self.temperature_value,
                max_len_a=self.max_len_a,
                max_len_b=self.max_len_b)
        # end if

        if is_hallucination_larger_threshold:
            is_hallucination = __translated_obj.avg_dissimilarity > threshold
        else:
            is_hallucination = __translated_obj.avg_dissimilarity < threshold
        # end if

        return OutputDisSimilarityMcDropOut(
            source_text=evaluation_target.source,
            translations=__translated_obj.outputs,
            avg_dissimilarity=__translated_obj.avg_dissimilarity,
            threshold_dissimilarity=threshold,
            is_hallucination=is_hallucination,
            metric_class=__translated_obj.metric_class)

