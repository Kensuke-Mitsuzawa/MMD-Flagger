from pathlib import Path
import typing as ty

import pandas as pd
import csv
import numpy as np

from .data_models import WMTDatasetRecord

import logging
logger = logging.getLogger(__name__)


def load_dataset(path_original_dataset_csv: Path, delimiter: str = ',', encoding: str = 'utf-8') -> ty.List[WMTDatasetRecord]:
    def __load_csv():
        csv_file = open(path_original_dataset_csv, 'r', encoding=encoding)
        csv_reader = csv.reader(csv_file, delimiter=delimiter, quotechar='"')
        
        stack = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            # end if 
            
            if len(row) != 9:
                logger.error(f"Row {i} has {len(row)} columns, expected 9 columns. Skip this instance.")
                continue
            else:        
                assert len(row) == 9, f"Row {i} has {len(row)} columns, expected 9 columns" 
                record_obj = WMTDatasetRecord(
                    sentence_id=int(row[0]),
                    source=row[1],
                    translation=row[2],
                    reference=row[3],
                    error_repetitions=int(row[4]),
                    error_named_entities=int(row[5]),
                    error_omission=int(row[6]),
                    error_strong=int(row[7]),
                    error_full=int(row[8])
                )
                stack.append(record_obj)
            # end if
        # end for
        logger.info(f"Loaded {len(stack)} records from {path_original_dataset_csv}")
        return stack
    # end def


    def __load_excel():
        df_dataset = pd.read_excel(path_original_dataset_csv)
        seq_records = df_dataset.to_dict('records')
        
        assert len(seq_records) > 0, f"Expected at least one record, got {len(seq_records)}"

        stack = []
        for __record in seq_records:
            __record['extention_label'] = __record['Evaluation']
            del __record['Evaluation']
            record_obj = WMTDatasetRecord(**__record)
            stack.append(record_obj)
        # end for
        logger.info(f"Loaded {len(stack)} records from {path_original_dataset_csv}")

        return stack
    # end def

    # detecting the extension
    if path_original_dataset_csv.suffix == ".csv" or path_original_dataset_csv.suffix == ".tsv":
        return __load_csv()
    elif path_original_dataset_csv.suffix == ".xlsx":
        return __load_excel()
    else:
        raise ValueError(f"Unsupported file extension: {path_original_dataset_csv.suffix}")
    # end if


def classify_error_category_rough(seq_records: ty.List[WMTDatasetRecord]) -> ty.List[str]:
    """I classify the error label (of the given dataset) into any of readable categories."""
    # for the classification, Refer the Dataset paper: https://aclanthology.org/2023.eacl-main.75.pdf
    
    # if all attributes are zero -> "correct"
    # if any of repetitions, strong-unsupport,full-unsupport -> "hallucination"
    # if any of named-entities, omission -> "MT-error"

    # I extract the error labels first and convert into the array.
    array_label = np.zeros((len(seq_records), 5))
    for __i, _record in enumerate(seq_records):
        assert isinstance(_record, WMTDatasetRecord), f"Expected WMTDatasetRecord, got {type(_record)}"
        # assert "error_repetitions" in _record, f"Expected 'error_repetitions' in the record, got {_record.keys()}"
        # assert "error_named_entities" in _record, f"Expected 'error_named_entities' in the record, got {_record.keys()}"
        # assert "error_omission" in _record, f"Expected 'error_omission' in the record, got {_record.keys()}"
        # assert "error_strong" in _record, f"Expected 'error_strong' in the record, got {_record.keys()}"
        # assert "error_full" in _record, f"Expected 'error_full' in the record, got {_record.keys()}"
        
        array_label[__i, 0] = _record.error_named_entities
        array_label[__i, 1] = _record.error_omission
        array_label[__i, 2] = _record.error_repetitions        
        array_label[__i, 3] = _record.error_strong
        array_label[__i, 4] = _record.error_full
    # end for
    
    # I classify the error labels
    # Get Hallucination
    ind_hallucination = np.where((array_label[:, 2] == 1) | (array_label[:, 3] == 1) | (array_label[:, 4] == 1))[0]
    # Get MT-errors (but NOT hallucination)
    __ind_mt_errors = np.where((array_label[:, 0] == 1) | (array_label[:, 1] == 1))[0]
    ind_mt_errors = sorted(list(set(__ind_mt_errors) - set(ind_hallucination)))  # removing the hallucination errors
    # correct translations
    ind_correct = sorted(list(set(range(len(seq_records))) - set(ind_hallucination) - set(ind_mt_errors)))
    
    # double-check
    assert len(ind_hallucination) + len(ind_mt_errors) + len(ind_correct) == len(seq_records), f"Expected {len(seq_records)} records, got {len(ind_hallucination) + len(ind_mt_errors) + len(ind_correct)}"
    # double-check: Correct-label MUST have all zeros
    for __i in ind_correct:
        assert np.all(array_label[__i] == 0), f"Expected all zeros, got {array_label[__i]}"
    # end for
    
    # I assign the labels
    stack_labels = []
    for __i in range(len(seq_records)):
        __label: str
        if __i in ind_hallucination:
            __label = "hallucination"
        elif __i in ind_mt_errors:
            __label = "MT-error"
        elif __i in ind_correct:
            __label = "correct"
        else:
            raise ValueError(f"Invalid index: {__i}")
        # end if
        stack_labels.append(__label)
    # end for
    assert len(stack_labels) == len(seq_records), f"Expected {len(seq_records)} labels, got {len(stack_labels)}"
    
    return stack_labels
# end def
