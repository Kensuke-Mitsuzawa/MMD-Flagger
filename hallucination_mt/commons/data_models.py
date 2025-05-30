import typing as ty
import dataclasses


@dataclasses.dataclass
class EvaluationTargetTranslationPair:
    source: str
    target: str
    sentence_id: str

