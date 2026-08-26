import typing as ty
from pydantic import BaseModel, Field, ConfigDict

import torch
import numpy as np
import hashlib
import dataclasses
import pickle
import json
from copy import deepcopy

from mmd_tst_variable_detector.kernels.commons import KernelMatrixObject

from .db_keys import SampleSetUniqueId, RecordObjMmdFlaggerIntermedTableUniqueId

"""A module for a record model of saving the intermediate values in MMD-flagger."""

TableTypes = ty.Literal['mmd_trajectory', 'mmd_matrix_temperature', 'kernel_matrix']


class MmdEstimatorConfig(BaseModel):
    mmd_estimator_class: str
    kernel_class: str
    mmed_estimator_args: ty.Dict[str, ty.Any]
    kernel_args: ty.Dict[str, ty.Any]  # TODO: should I make it a class? percentile-length-scale etc.?
    
    def to_dict(self) -> ty.Dict:
        return self.model_dump()


class RecordObjMmdFlaggerIntermed(BaseModel):
    """A record object to save the intermediate values of MMD-flagger."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---- fields to form an unique key ----
    unique_id_y_hyp: ty.Optional[ty.Union[str, SampleSetUniqueId]] = Field(description="an unique label of `SampleSet` object. None when the record is for the 'mmd_matrix_temperature'.")
    unique_ids_y_sto: ty.List[ty.Union[str, SampleSetUniqueId]] = Field(description="an unique label of `SampleSet` object.")
    hash_y_hyp: str = Field()
    hash_y_sto: str = Field()
    temperature_sequences: ty.List[float] = Field(description="a sequence of temperature parameter.")
    mmd_estimator_config: ty.Dict = Field(description="`MmdEstimatorConfig.to_dict()`")
    # ---- fields to form an unique key ----

    table_type: TableTypes = Field(description="the table name that this record object is saved.")
    
    blob_object: bytes = Field(description="MMD-trajectory array or MMD-matrix or Kernel-matrix-object.")

    global_unique_id: ty.Optional[ty.Union[str, RecordObjMmdFlaggerIntermedTableUniqueId]] = None 

    def _form_unique_key_dict(self) -> ty.Dict:
        dict_mmd_estm = self.mmd_estimator_config
        temp = self.temperature_sequences

        d_unique_id = dict(
            unique_id_y_hyp=self.unique_id_y_hyp,
            unique_ids_y_sto=self.unique_ids_y_sto,
            temperature_sequences=temp,
            mmd_estimator_config=dict_mmd_estm,
            hash_y_hyp=self.hash_y_hyp,
            hash_y_sto=self.hash_y_sto)

        return d_unique_id

    def _get_unique_id(self, d_unique_id: ty.Dict) -> str:
        # ---- forming the unique id ----
        hash_id = hashlib.sha256(json.dumps(d_unique_id, sort_keys=True).encode('utf8')).hexdigest()

        return hash_id
    
    def to_dict(self) -> ty.Dict:
        d_unique_id = self._form_unique_key_dict()
        hash_id = self._get_unique_id(d_unique_id)

        d = deepcopy(d_unique_id)
        d |= dict(blob_object=self.blob_object, global_unique_id=hash_id)

        return d

    @classmethod
    def get_unique_id(cls,
                      unique_id_y_hyp: ty.Optional[str],
                      unique_ids_y_sto: ty.List[str],
                      temperature_sequences: ty.List[float],
                      mmd_estimator_config: ty.Dict) -> str:
        # ---- forming the unique id ----
        d_unique_id = dict(
            unique_id_y_hyp=unique_id_y_hyp,
            unique_ids_y_sto=unique_ids_y_sto,
            temperature_sequences=temperature_sequences,
            mmd_estimator_config=mmd_estimator_config)

        hash_id = hashlib.sha256(json.dumps(d_unique_id, sort_keys=True).encode('utf8')).hexdigest()

        return hash_id
