import dataclasses
import typing as ty

@dataclasses.dataclass
class WMTDatasetRecord:
    sentence_id: ty.Union[str, int]
    source: str
    translation: str
    reference: str
    error_repetitions: int
    error_named_entities: int
    error_omission: int
    error_strong: int
    error_full: int
    extention_label: ty.Optional[str] = None  # a field to store the label of the extension
    error_type: ty.Optional[str] = None  # a field to store the type of error

    def __post_init__(self):
        error_total = self.error_repetitions + self.error_named_entities + self.error_omission + self.error_strong + self.error_full
        error_mt_error = self.error_named_entities + self.error_omission
        error_hallucination = self.error_repetitions + self.error_strong + self.error_full
        if error_total == 0:
            self.error_type = "correct"
        elif error_mt_error > 0 and error_hallucination == 0:
            self.error_type = "mt_error"
        elif error_hallucination > 0:
            self.error_type = "hallucination"
        else:
            raise ValueError("Invalid error type")
    # end def