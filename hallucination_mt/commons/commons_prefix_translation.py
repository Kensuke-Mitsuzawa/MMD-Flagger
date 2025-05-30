import typing as ty

import torch

from hallucination_mt.guerreiro_2023_wmt.data_models.data_models import WMTDatasetRecord


class ReturnFlagBeamSearch(ty.NamedTuple):
    is_hallucination: bool
    source_text: str
    h_tag: str
    h_no_tag: str
    prefix_meta_tag: str
    h_tag_before_modification: str
    is_success: bool


class AnalysisReturnTuple(ty.NamedTuple):
    sentence_id: str
    source_text: str
    h_tag: str
    h_no_tag: str

    error_label_truth: str
    is_hallucination_truth: bool
    class_hallucination_truth: str
# end class

class CachedAnalysisTuple(ty.NamedTuple):
    sentence_id: str
    record_obj_json: str  # json field
    source_text: str
    is_hallucination: bool
    h_tag: str
    h_no_tag: str
    prefix_meta_tag: str
    h_tag_before_modification: str
    is_success: bool
    record_obj: ty.Optional[ty.Dict] = None


class CachedAnalysisEncoderVectorTuple(ty.NamedTuple):
    sentence_id: str

    source_text_no_tag: str
    source_text_tag: str

    encode_vector_no_tag: torch.Tensor
    encode_vector_tag: torch.Tensor
    encode_vector_tag_no_modification: torch.Tensor

    distance_cosine: float

    token_ids_no_tag: ty.List[int]
    token_ids_tag: ty.List[int]

    is_hallucination: bool
    prefix_meta_tag: str
    is_success: bool
    record_obj: ty.Optional[ty.Union[ty.Dict, WMTDatasetRecord]] = None
    record_obj_json: ty.Optional[str] = None  # json field
