import typing as ty
from ....guerreiro_2023_wmt.data_models.data_models import WMTDatasetRecord
from ....dale_2023_halomi.load_dataset import HalomiDatasetRecord

DEFAULT_GROUND_TRUTH_SETTINGS = (
    "hallucination", 
    "hallucination+mt-error", 
    "mt-error", 
    "error_named_entity",
    "error_omission",
    "error_full",
    "error_strong",
    "error_repetitions",
    "2_Small_hallucination",
    "3_Partial_hallucination",
    "4_Full_hallucination"
)



def _get_ground_truth_labels(seq_dataset_record: ty.List, dataset_name: str, label_setting: str) -> ty.Dict[str, int]:
    """Obtaining ground truth labels."""
    def _process_lfan_hall_dataset(seq_dataset_record: ty.List[WMTDatasetRecord]):
        assert label_setting in DEFAULT_GROUND_TRUTH_SETTINGS, f"Unsupported label setting: {label_setting}"
        dict_config_name2label = {}
        for dataset_record in seq_dataset_record:
            _sentence_id = str(dataset_record.sentence_id)
            if label_setting == "hallucination":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.error_type == "hallucination" else 0
            elif label_setting == "hallucination+mt-error":
                dict_config_name2label[_sentence_id] = 1 if (dataset_record.error_type == "hallucination") or (dataset_record.error_type == "mt-error") else 0
            elif label_setting == "mt-error":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.error_type == "mt-error" else 0
            elif label_setting == "error_full":
                dict_config_name2label[_sentence_id] = dataset_record.error_full
            elif label_setting == "error_strong":
                dict_config_name2label[_sentence_id] = dataset_record.error_strong
            elif label_setting == "error_repetitions":
                dict_config_name2label[_sentence_id] = dataset_record.error_repetitions
            elif label_setting == "error_omission":
                dict_config_name2label[_sentence_id] = dataset_record.error_omission
            elif label_setting == "error_named_entity":
                dict_config_name2label[_sentence_id] = dataset_record.error_named_entities
            else:
                raise ValueError(f"Unsupported label setting: {label_setting}")
            # end if
        # end for
        assert len(dict_config_name2label) > 0, "No records found."
        return dict_config_name2label
    # end def

    def _process_halomi_dataset(seq_dataset_record: ty.List[HalomiDatasetRecord]):
        assert label_setting in DEFAULT_GROUND_TRUTH_SETTINGS, f"Unsupported label setting: {label_setting}"
        dict_config_name2label = {}
        for dataset_record in seq_dataset_record:
            _sentence_id = dataset_record.key_unique
            # ---------------------------------------------------------
            # Using the Halomi Dataset Labels System
            if label_setting == "2_Small_hallucination":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.class_hall == "2_Small_hallucination" else 0
            elif label_setting == "3_Partial_hallucination":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.class_hall == "3_Partial_hallucination" else 0
            elif label_setting == "4_Full_hallucination":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.class_hall == "4_Full_hallucination" else 0

            # ---------------------------------------------------------
            # Using the Labels of the LFAN-HALL.
            elif label_setting == "hallucination":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.error_type == "hallucination" else 0
            elif label_setting == "hallucination+mt-error":
                dict_config_name2label[_sentence_id] = 1 if (dataset_record.error_type == "hallucination") or (dataset_record.error_type == "mt-error") else 0
            elif label_setting == "mt-error":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.error_type == "mt-error" else 0
            elif label_setting == "error_full":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.class_hall == "4_Full_hallucination" else 0
            elif label_setting == "error_strong":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.class_hall == "3_Partial_hallucination" else 0
            elif label_setting == "error_repetitions":
                # Note: there is no corresponding class for repetition in the dataset.
                dict_config_name2label[_sentence_id] = 0
            elif label_setting == "error_omission":
                dict_config_name2label[_sentence_id] = 1 if dataset_record.is_omission else 0
            elif label_setting == "error_named_entity":
                # Note: there is no corresponding class for repetition in the dataset.
                dict_config_name2label[_sentence_id] = 0
            else:
                raise ValueError(f"Unsupported label setting: {label_setting}")
            # end if
        # end for
        assert len(dict_config_name2label) > 0, "No records found."
        return dict_config_name2label
    # end def


    if dataset_name == 'halomi':
        return _process_halomi_dataset(seq_dataset_record)  # type: ignore
    elif dataset_name == 'lfan_hall':
        return _process_lfan_hall_dataset(seq_dataset_record)  # type: ignore
    else:
        raise ValueError(f"Unsupported dataset record type: {dataset_name}")
    # end if