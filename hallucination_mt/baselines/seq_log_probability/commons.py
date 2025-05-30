import typing as ty
from dataclasses import dataclass


class _RecordCache(ty.NamedTuple):
    sentence_id: str
    source_sentence: str
    log_probability: float
    translation_text: ty.Optional[str] = None


@dataclass
class OutputLogProbabilityFlagger:
    source_text: str
    translated_text: str
    log_probability: float
    threshold_probability: float
    is_hallucination: bool
