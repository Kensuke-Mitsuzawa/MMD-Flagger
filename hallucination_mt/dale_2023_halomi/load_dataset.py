import typing as ty
from pathlib import Path

import pandas as pd
import dataclasses

"""A module of treating the HALOMI dataset.

The Halomi dataset is at: https://github.com/facebookresearch/stopes/tree/main/demo/halomi

@article{dale2023halomi,
  title={HalOmi: A Manually Annotated Benchmark for Multilingual Hallucination and Omission Detection in Machine Translation},
  author={Dale, David and Voita, Elena and Lam, Janice and Hansanti, Prangthip and Ropers, Christophe and Kalbassi, Elahe and Gao, Cynthia and Barrault, Lo{\"\i}c and Costa-juss{\`a}, Marta R},
  journal={arXiv preprint arXiv:2305.11746},
  url={https://arxiv.org/abs/2305.11746},
  year={2023}
}
"""

# The class tags below are from the HALOMI dataset.
CLASS_HALL = ("1_No_hallucination", "2_Small_hallucination", "3_Partial_hallucination", "4_Full_hallucination")
CLASS_OMIT = ("1_No_omission", "2_Small_omission", "3_Partial_omission", "4_Full_omission")
# The classfication of the error types follow the Guerreiro et al. (2023) paper.
ERROR_TYPE = ("correct", "mt_error", "hallucination")


@dataclasses.dataclass
class HalomiDatasetRecord:
    index_id: int
    src_lang: str
    tgt_lang: str
    src_text: str    
    tgt_text: str
    class_hall: str
    class_omit: str
    is_hallucination: bool
    is_omission: bool
    error_type: str
    key_unique: ty.Optional[str] = None  # a field to store the unique key of the record

    def __post_init__(self):
        assert self.class_hall in CLASS_HALL, f"Invalid class_hall: {self.class_hall}. Expected one of {CLASS_HALL}."
        assert self.class_omit in CLASS_OMIT, f"Invalid class_omit: {self.class_omit}. Expected one of {CLASS_OMIT}."
        assert self.error_type in ERROR_TYPE, f"Invalid error_type: {self.error_type}. Expected one of {ERROR_TYPE}."

        self.key_unique = f"{self.src_lang}_{self.tgt_lang}_{self.index_id}"


def load_dataset(path_halomi_dataset_tsv: Path) -> ty.List[HalomiDatasetRecord]:
    assert path_halomi_dataset_tsv.exists(), f"Dataset file {path_halomi_dataset_tsv} does not exist."
    df_halomi = pd.read_csv(path_halomi_dataset_tsv, sep="\t")

    assert "src_text" in df_halomi.columns, "Column 'src_text' not found in the dataset."
    assert "mt_text" in df_halomi.columns, "Column 'src_text' not found in the dataset."
    assert "class_omit" in df_halomi.columns, "Column 'src_text' not found in the dataset."
    assert "class_hall" in df_halomi.columns, "Column 'src_text' not found in the dataset."

    assert len(df_halomi) > 0, "No records found in the dataset."

    seq_stack_record_obj = []

    seq_record = df_halomi.to_dict(orient="records")
    for _index_id, _record in enumerate(seq_record):
        _class_hall = _record["class_hall"]
        assert _class_hall in CLASS_HALL, f"Invalid class_hall: {_class_hall}. Expected one of {CLASS_HALL}."

        _class_omit = _record["class_omit"]
        assert _class_omit in CLASS_OMIT, f"Invalid class_omit: {_class_omit}. Expected one of {CLASS_OMIT}."

        if int(_class_hall[0]) > 1:
            _is_hallucination = True
        else:
            _is_hallucination = False
        # end if

        if int(_class_omit[0]) > 1:
            _is_omission = True
        else:
            _is_omission = False
        # end if

        if _is_hallucination:
            _error_type = "hallucination"
        elif _is_omission:
            _error_type = "mt_error"
        else:
            _error_type = "correct"
        # end if

        assert _error_type in ERROR_TYPE, f"Invalid error_type: {_error_type}. Expected one of {ERROR_TYPE}."


        _obj_record = HalomiDatasetRecord(
            index_id=_index_id,
            src_lang=_record["src_lang"],
            tgt_lang=_record["tgt_lang"],
            src_text=_record["src_text"],
            tgt_text=_record["mt_text"],
            class_hall=_class_hall,
            class_omit=_class_omit,
            is_hallucination=_is_hallucination,
            is_omission=_is_omission,
            error_type=_error_type
        )
        seq_stack_record_obj.append(_obj_record)
    # end for

    return seq_stack_record_obj
# end def
