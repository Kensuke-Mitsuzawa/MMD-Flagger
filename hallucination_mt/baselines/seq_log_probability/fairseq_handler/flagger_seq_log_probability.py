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

from ..commons import _RecordCache, OutputLogProbabilityFlagger
from ....commons.data_models import EvaluationTargetTranslationPair
from .seq_log_probability import ComputeSequenceLogProbability


logger = logging.getLogger(__name__)




class FlaggerSeqLogProbability(object):
    def __init__(self, 
                 fairseq_interface: GeneratorHubInterface,
                 path_dir_cache: ty.Optional[Path] = None):
        self.executer_seq_log_probability = ComputeSequenceLogProbability(fairseq_interface)

        if path_dir_cache is None:
            self.path_dir_cache = Path(tempfile.mkdtemp())
        else:
            assert Path(path_dir_cache).exists(), f"Path {path_dir_cache} does not exist"
            self.path_dir_cache = Path(path_dir_cache)
        # end if

    def _load_cache(self, sentence_id: str, source_text: str) -> _RecordCache:
        path_cache_file = self.path_dir_cache / f"{sentence_id}.pt"
        cache_obj = torch.load(path_cache_file)

        assert isinstance(cache_obj, dict), f"Expected dict, got {type(cache_obj)}"
        assert "source_sentence" in cache_obj, f"Expected key 'source_sentence' in {cache_obj.keys()}"
        assert "log_probability" in cache_obj, f"Expected key 'log_probability' in {cache_obj.keys()}"
        assert "sentence_id" in cache_obj, f"Expected key 'sentence_id' in {cache_obj.keys()}"

        assert cache_obj['source_sentence'] == source_text, f"Expected source text {source_text}, got {cache_obj['source_sentence']}"
        return _RecordCache(
            sentence_id=cache_obj['sentence_id'], 
            source_sentence=cache_obj['source_sentence'], 
            log_probability=cache_obj['log_probability'],
            translation_text=cache_obj.get('translation_text', None)
            )

    def compute_dataset_log_probability(self, dataset: ty.List[EvaluationTargetTranslationPair]) -> ty.List[float]:
        """I compute the log probability for each record of the dataset.
        
        Returns:
            ty.List[float]: List of log probabilities. A distribution of the log probabilities.
        """
        seq_log_probabilities = []
        # TODO: I want to introduce tqdm logger.
        for _record in dataset:
            path_cache_file = self.path_dir_cache / f"{_record.sentence_id}.pt"
            if path_cache_file.exists():
                record_cache = self._load_cache(_record.sentence_id, _record.source)
                seq_log_probabilities.append(record_cache.log_probability)
            else:
                __translated_obj = self.executer_seq_log_probability(_record.source)
                cache_obj = {
                    "sentence_id": _record.sentence_id,
                    "source_sentence": _record.source,
                    "log_probability": __translated_obj.log_probability,
                    "translation_text": __translated_obj.translated_text
                }
                torch.save(cache_obj, path_cache_file)
                seq_log_probabilities.append(__translated_obj.log_probability)
            # end if
        # end for
        return seq_log_probabilities
    
    def get_flag_threshold(self, 
                           seq_log_probability: ty.List[float],
                           percentile: float = 0.004) -> float:
        """I get the threshold for the flagging.
        
        Args:
            seq_log_probability (ty.List[float]): List of log probabilities.
            percentile (float, optional): Percentile. Defaults to 0.004 (0.4%), suggested by Guerreiro et al. (2023).
            Since the log-propability indicates that a larger (close to 0.0) is more fluent, I take a small `percentile`  
        """
        array_log_prob = np.array(seq_log_probability)
        threshold = np.percentile(array_log_prob, percentile)
        return threshold.item()

    def flag(self, source_text: str, threshold: float, sentence_id: ty.Optional[str] = None) -> OutputLogProbabilityFlagger:
        if sentence_id is not None:
            path_cache_file = self.path_dir_cache / f"{sentence_id}.pt"
            if path_cache_file.exists():
                record_cache = self._load_cache(sentence_id, source_text)
            else:
                record_cache = None
            # end if
        else:
            record_cache = None
        # end if
        # 
        if record_cache is None:        
            __translated_obj = self.executer_seq_log_probability(source_text)
            is_hallucination = __translated_obj.log_probability < threshold
            translation_text = __translated_obj.translated_text
            log_score = __translated_obj.log_probability
        else:
            is_hallucination = record_cache.log_probability < threshold
            log_score = record_cache.log_probability
            translation_text = record_cache.translation_text
            assert log_score is not None, f"Log score is None for {source_text}"
        # end if

        return OutputLogProbabilityFlagger(source_text=source_text, 
                                           translated_text=translation_text,
                                           log_probability=log_score,
                                           threshold_probability=threshold,
                                           is_hallucination=is_hallucination)
