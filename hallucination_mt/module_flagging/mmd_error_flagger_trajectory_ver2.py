import typing as ty
import numpy.typing as npt
import torch
import logging
import tqdm
import dataclasses
import json
import sys
import sqlite3
import tempfile
import io
from pathlib import Path

import zlib



import GPUtil

import numpy as np
from fairseq.hub_utils import GeneratorHubInterface

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel

from ..commons.data_models import EvaluationTargetTranslationPair

from ..module_assessments.custom_tqdm_handler import TqdmLoggingHandler
from ..guerreiro_2023_wmt.utils_models.utils import (
    load_model, 
    extract_word_embeddings,
    extract_word_embeddings_batch
)

from ..module_assessments.module_management_db import module_sqlite3_handler
from ..module_translation_handler.ver1.module_fairseq_handler import (
    FaiseqTranslationModelHandler,
    DecodedTranslationObject,
    ParameterSettingException)
from ..module_translation_handler.ver1.module_transformer_handler import (
    TransformersTranslationModelHandler,
    GeneratedTranslationObject
)
from ..module_hidden_vector_extractor.ver1.module_fairseq import FairSeqVectorExtractor
from ..module_hidden_vector_extractor.ver1.module_transformers import TransformerVectorExtractor
from .module_classify_trajectory import module_classify_rule_base



module_logger = logging.getLogger(__name__)
# a special logger for tqdm
tqdm_logger = logging.getLogger(f'{__name__}.tqdm')
tqdm_logger.addHandler(TqdmLoggingHandler())


# ------------------------------
# Internal Data Models

@dataclasses.dataclass
class _SampledTranslationObject:
    tau_parameter: float
    n_sample: int
    translation_text: ty.Optional[ty.List[str]] = None
    embedding_tensor: ty.Optional[torch.Tensor] = None
    mmd_distance: ty.Optional[float] = None
    is_success: bool = True


@dataclasses.dataclass
class _CacheDatabaseRecordTranslation:
    record_key_id: str
    sentence_id: str
    n_sampling: int
    temperature: float
    source_text: str
    translation_set_json: str
    is_success: bool = True

    def to_dict(self) -> ty.Dict[str, ty.Any]:
        obj = dataclasses.asdict(self)
        return obj

    @classmethod
    def from_dict(cls, dict_obj: ty.Dict[str, ty.Any]) -> "_CacheDatabaseRecordTranslation":
        return cls(**dict_obj)


@dataclasses.dataclass
class _CacheDatabaseRecordEmbeddingTensor:
    record_key_id: str
    sentence_id: str
    n_sampling: int
    temperature: float
    embedding_layer_name: str
    embedding_tensor_bytes: ty.Optional[bytes]  # compressed byte string of the tensor
    embedding_tensor: ty.Optional[torch.Tensor]

    def __post_init__(self):
        if self.embedding_tensor_bytes is None:
            assert self.embedding_tensor is not None, "Invalid embedding tensor."
            self.embedding_tensor_bytes = self.tensor_to_bytes(self.embedding_tensor)
        # end if

    @staticmethod
    def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        """Serializes a PyTorch tensor to a bytes sequence."""
        numpy_array = tensor.cpu().numpy()  # Move tensor to CPU for NumPy
        byte_stream = io.BytesIO()
        np.save(byte_stream, numpy_array)
        numpy_bytes = byte_stream.getvalue()
        compressed_data = zlib.compress(numpy_bytes)
        return compressed_data

    @staticmethod
    def bytes_to_tensor(compressed_data: bytes) -> torch.Tensor:
        """Deserializes a bytes sequence back to a PyTorch tensor."""
        decompressed_data = zlib.decompress(compressed_data)
        byte_stream = io.BytesIO(decompressed_data)
        numpy_array = np.load(byte_stream)
        tensor = torch.from_numpy(numpy_array)
        return tensor
    
    def to_dict(self) -> ty.Dict[str, ty.Any]:
        obj = dataclasses.asdict(self)
        obj["embedding_tensor"] = None
        return obj

    @classmethod
    def from_dict(cls, dict_obj: ty.Dict[str, ty.Any]) -> "_CacheDatabaseRecordEmbeddingTensor":
        assert dict_obj["embedding_tensor_bytes"] is not None, "Invalid embedding tensor bytes."
        dict_obj["embedding_tensor"] = cls.bytes_to_tensor(dict_obj["embedding_tensor_bytes"])
        return cls(**dict_obj)


@dataclasses.dataclass
class _CacheDatabaseRecordEmbeddingTensor_Y:
    record_key_id: str
    sentence_id: str
    embedding_layer_name: str
    embedding_tensor_bytes: ty.Optional[bytes]
    embedding_tensor: ty.Optional[torch.Tensor]

    def __post_init__(self):
        if self.embedding_tensor_bytes is None:
            assert self.embedding_tensor is not None, "Invalid embedding tensor."
            self.embedding_tensor_bytes = self.tensor_to_bytes(self.embedding_tensor)
        # end if

    @staticmethod
    def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        """Serializes a PyTorch tensor to a bytes sequence."""
        numpy_array = tensor.cpu().numpy()  # Move tensor to CPU for NumPy
        byte_stream = io.BytesIO()
        np.save(byte_stream, numpy_array)
        numpy_bytes = byte_stream.getvalue()
        compressed_data = zlib.compress(numpy_bytes)
        return compressed_data

    @staticmethod
    def bytes_to_tensor(compressed_data: bytes) -> torch.Tensor:
        """Deserializes a bytes sequence back to a PyTorch tensor."""
        decompressed_data = zlib.decompress(compressed_data)
        byte_stream = io.BytesIO(decompressed_data)
        numpy_array = np.load(byte_stream)
        tensor = torch.from_numpy(numpy_array)
        return tensor

    def to_dict(self) -> ty.Dict[str, ty.Any]:
        obj = dataclasses.asdict(self)
        obj["embedding_tensor"] = None
        return obj

    @classmethod
    def from_dict(cls, dict_obj: ty.Dict[str, ty.Any]) -> "_CacheDatabaseRecordEmbeddingTensor_Y":
        assert dict_obj["embedding_tensor_bytes"] is not None, "Invalid embedding tensor bytes."
        dict_obj["embedding_tensor"] = cls.bytes_to_tensor(dict_obj["embedding_tensor_bytes"])
        return cls(**dict_obj)
# end def

PossibleDataRecordType = ty.Union[_CacheDatabaseRecordTranslation, _CacheDatabaseRecordEmbeddingTensor, _CacheDatabaseRecordEmbeddingTensor_Y]

# ------------------------------


@dataclasses.dataclass
class MmdErrorFlagResult:
    evaluation_pair: EvaluationTargetTranslationPair
    n_sample: int

    tau_parameter: ty.List[float]
    mmd_distances: ty.List[float]

    tensor_given_translation: torch.Tensor
    tensor_hypothesis_translation: torch.Tensor  # (T: the number of tau param, N: the number of sampling, D: embedding-size)
    
    hypothesis_translation: ty.List[ty.Optional[ty.List[str]]]

    trajectory_shape: str
    is_hallucination: bool


class MmdErrorFlaggerTrajectoryVer2(object):
    def __init__(self,
                 translation_handler: ty.Union[FaiseqTranslationModelHandler, TransformersTranslationModelHandler],
                 vector_extractor: ty.Union[FairSeqVectorExtractor, TransformerVectorExtractor],
                 mmd_estimator: ty.Optional[QuadraticMmdEstimator] = None,
                 seq_calibration_text: ty.Optional[ty.List[str]] = None,
                 mode_preprocess: str = "avg",
                 median_options: str = "dimensionwise",
                 median_heuristic_operation: str = "median",
                 path_cache_dir: ty.Optional[Path] = None,
                 file_name_cache_database_translation: str = "cache_database_translation.db",
                 file_name_cache_database_embedding: str = "cache_database_embedding.db",
                 trajectory_rule: str = 'v1',
                 trajectory_rule_smoothing: str = 'no_filter',
                 trajectory_rule_smoothing_window: ty.Optional[int] = None,
                 is_use_gpu: bool = True,
                 ):
        # assert isinstance(model_encoder_decoder_mt, GeneratorHubInterface), "model_encoder_decoder_mt must be an instance of fairseq.hub_utils.GeneratorHubInterface"
        # self.model_encoder_decoder_mt = model_encoder_decoder_mt

        assert isinstance(translation_handler, (FaiseqTranslationModelHandler, TransformersTranslationModelHandler)), \
            "translation_handler must be an instance of FaiseqTranslationModelHandler or TransformersTranslationModelHandler"
        assert isinstance(vector_extractor, (FairSeqVectorExtractor, TransformerVectorExtractor)), \
            "vector_extractor must be an instance of FairSeqVectorExtractor or TransformerVectorExtractor"
        
        self.translation_handler = translation_handler
        self.vector_extractor = vector_extractor

        self.mode_preprocess = mode_preprocess
        self.median_options = median_options
        self.median_heuristic_operation = median_heuristic_operation

        # ------------------------------------------------------
        # Setting the MMD estimator
        if mmd_estimator is None:
            assert seq_calibration_text is not None, "seq_calibration_dataset is required to initialize mmd estimator."
            self.mmd_estimator = self.__init_mmd_estimator(seq_calibration_text)
        else:
            self.mmd_estimator = mmd_estimator
        # end if

        if is_use_gpu and torch.cuda.is_available():
            _device_id_gpu = self._get_less_busy_cuda_device()
            self.torch_device = torch.device(f"cuda:{_device_id_gpu}")
            self.mmd_estimator.to(self.torch_device)
            self.is_use_gpu = True
        else:
            self.torch_device = torch.device("cpu")
            self.is_use_gpu = False
        # end if

        # ------------------------------------------------------
        # Setting the cache directory
        if path_cache_dir is None:
            self.path_cache_dir = Path(tempfile.mkdtemp())
        else:
            self.path_cache_dir = path_cache_dir
        # end if        
        self.path_cache_dir.mkdir(parents=True, exist_ok=True)
        _path_cache_database_translation = self.path_cache_dir / file_name_cache_database_translation
        _path_cache_database_embedding = self.path_cache_dir / file_name_cache_database_embedding
        # self.path_cache_database = _path_cache_database
        self.cache_database_handler_translation, self.cache_database_handler_embedding = self._set_cache_database(
            _path_cache_database_translation,
            _path_cache_database_embedding)
        self.path_cache_database_translation = _path_cache_database_translation
        self.path_cache_database_embedding = _path_cache_database_embedding
        # ------------------------------------------------------

        assert trajectory_rule_smoothing in module_classify_rule_base.POSSIBLE_FILTERS, f"Invalid type trajectory shape: {trajectory_rule_smoothing}"
        self.trajectory_rule = trajectory_rule
        self.trajectory_rule_smoothing = trajectory_rule_smoothing
        self.trajectory_rule_smoothing_window = trajectory_rule_smoothing_window


    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]

    # ------------------------------------------------------
    # methods about a cache database

    def _set_cache_database(self, 
                            path_cache_database_translation: Path,
                            path_cache_database_embedding: Path
                            ) -> ty.Tuple[module_sqlite3_handler.DBHandlerExp, module_sqlite3_handler.DBHandlerExp]:
        """I set the cache database."""
        sqlite3_conn_translation = sqlite3.connect(path_cache_database_translation)
        module_sqlite3_handler.create_table_from_table_definition(
            conn=sqlite3_conn_translation,
            table_record_class=_CacheDatabaseRecordTranslation,
            primary_key="record_key_id")
        sqlite3_conn_translation.close()

        sqlite3_conn_embedding = sqlite3.connect(path_cache_database_embedding)
        module_sqlite3_handler.create_table_from_table_definition(
            conn=sqlite3_conn_embedding,
            table_record_class=_CacheDatabaseRecordEmbeddingTensor,
            primary_key="record_key_id")
        module_sqlite3_handler.create_table_from_table_definition(
            conn=sqlite3_conn_embedding,
            table_record_class=_CacheDatabaseRecordEmbeddingTensor_Y,
            primary_key="record_key_id")
        sqlite3_conn_embedding.close()
        
        db_handler_translation = module_sqlite3_handler.DBHandlerExp(path_cache_database_translation)
        db_handler_embedding = module_sqlite3_handler.DBHandlerExp(path_cache_database_embedding)
        return db_handler_translation, db_handler_embedding
        
    def _save_cache_records(self, record_obj: PossibleDataRecordType) -> None:
        if isinstance(record_obj, _CacheDatabaseRecordTranslation):
            self.cache_database_handler_translation.insert(
                table_name=_CacheDatabaseRecordTranslation.__name__,
                data=record_obj.to_dict())
        elif isinstance(record_obj, _CacheDatabaseRecordEmbeddingTensor):
            self.cache_database_handler_embedding.insert(
                table_name=_CacheDatabaseRecordEmbeddingTensor.__name__,
                data=record_obj.to_dict())
        elif isinstance(record_obj, _CacheDatabaseRecordEmbeddingTensor_Y):
            self.cache_database_handler_embedding.insert(
                table_name=_CacheDatabaseRecordEmbeddingTensor_Y.__name__,
                data=record_obj.to_dict())
        else:
            raise ValueError(f"Invalid record object: {record_obj}")

    def _fetch_cache_records(self, primary_key: str, table_name: str) -> ty.Optional[PossibleDataRecordType]:
        if table_name == _CacheDatabaseRecordTranslation.__name__:
            _is_record_exist = self.cache_database_handler_translation.is_record_exists(
                table_name=_CacheDatabaseRecordTranslation.__name__,
                exp_key=primary_key,
                is_partially_search=False,
                primary_key="record_key_id")
            if _is_record_exist:
                _obj_record_obj = self.cache_database_handler_translation.get_record_key(
                    table_name=_CacheDatabaseRecordTranslation.__name__,
                    exp_key=primary_key,
                    primary_key_field="record_key_id")
                assert _obj_record_obj is not None, f"Invalid record object: {primary_key}"
                record_obj = _CacheDatabaseRecordTranslation.from_dict(_obj_record_obj)
                return record_obj
            else:
                return None
            # end if
        elif table_name == _CacheDatabaseRecordEmbeddingTensor.__name__:
            _is_record_exist = self.cache_database_handler_embedding.is_record_exists(
                table_name=_CacheDatabaseRecordEmbeddingTensor.__name__,
                exp_key=primary_key,
                is_partially_search=False,
                primary_key="record_key_id")
            if _is_record_exist:
                _obj_record_obj = self.cache_database_handler_embedding.get_record_key(
                    table_name=_CacheDatabaseRecordEmbeddingTensor.__name__,
                    exp_key=primary_key,
                    primary_key_field="record_key_id",)
                assert _obj_record_obj is not None, f"Invalid record object: {primary_key}"
                record_obj = _CacheDatabaseRecordEmbeddingTensor.from_dict(_obj_record_obj)

                return record_obj
            else:
                return None
        elif table_name == _CacheDatabaseRecordEmbeddingTensor_Y.__name__:
            _is_record_exist = self.cache_database_handler_embedding.is_record_exists(
                table_name=_CacheDatabaseRecordEmbeddingTensor_Y.__name__,
                exp_key=primary_key,
                is_partially_search=False,
                primary_key="record_key_id")
            if _is_record_exist:
                _obj_record_obj = self.cache_database_handler_embedding.get_record_key(
                    table_name=_CacheDatabaseRecordEmbeddingTensor_Y.__name__,
                    exp_key=primary_key,
                    primary_key_field="record_key_id")
                assert _obj_record_obj is not None, f"Invalid record object: {primary_key}"
                record_obj = _CacheDatabaseRecordEmbeddingTensor_Y.from_dict(_obj_record_obj)

                return record_obj
            else:
                return None
        else:
            raise ValueError(f"Invalid table name: {table_name}")
        # end if

    @staticmethod
    def _get_cache_database_key(sentence_id: str,
                                temperature: float,
                                n_sampling: int) -> str:
        return f"{sentence_id}_{temperature}_{n_sampling}"

    # ------------------------------------------------------

    def _calibrate_kernel_function(
            self,
            seq_calibration_text: ty.List[str]
            ) -> QuadraticKernelGaussianKernel:
        
        seq_embedding_tensor = self.vector_extractor.extract_word_embeddings_batch(seq_calibration_text)

        # pre-processing of tensors.
        # __a_emb -> a_emb, a fixed shape
        calibration_emb_fixed = self._preprocess_tensors(seq_embedding_tensor)

        if self.median_options == 'single':
            _is_dimension_wise = False
        elif self.median_options == 'dimensionwise':
            _is_dimension_wise = True
        else:
            raise ValueError(f"Invalid median options: {self.median_options}")
        # end if

        kernel_func_obj = QuadraticKernelGaussianKernel(
            is_dimension_median_heuristic=_is_dimension_wise,
            heuristic_operation=self.median_heuristic_operation,
            ard_weights=torch.ones(calibration_emb_fixed.shape[1])
        )
        module_logger.debug("Computing length scale using the calibration set...")
        if _is_dimension_wise:
            # TODO: there is the safe guard avoiding L2(x, x).
            tensor_length_scale = kernel_func_obj._get_median_dim(
                x=calibration_emb_fixed,
                y=calibration_emb_fixed,
                is_safe_guard_same_xy=False)
        else:
            tensor_length_scale = kernel_func_obj._get_median_single(
                x=calibration_emb_fixed,
                y=calibration_emb_fixed)
        # end if
        module_logger.debug("Done computing the length scale...")    
        assert tensor_length_scale is not None

        # set the computed length-scale to the kernel object.
        kernel_func_obj.bandwidth = torch.nn.Parameter(tensor_length_scale, requires_grad=False)
        kernel_func_obj.ard_weights = torch.nn.Parameter(torch.ones(calibration_emb_fixed.shape[1]), requires_grad=False)

        return kernel_func_obj

    def __init_mmd_estimator(self, seq_calibration_text: ty.List[str]) -> QuadraticMmdEstimator:
        kernel_func_obj = self._calibrate_kernel_function(seq_calibration_text)
        mmd_estimator = QuadraticMmdEstimator(kernel_func_obj, variance_term='sutherland_2017')

        return mmd_estimator
    
    def _compute_mmd_distance(self,
                             tensor_original_translation: torch.Tensor,
                             tensor_new_translation: torch.Tensor
                             ) -> float:
        """I want to compute the MMD distance between the original and the new translation."""
        is_same_tensor = torch.equal(tensor_original_translation, tensor_new_translation)
        if is_same_tensor:
            return 0.0
        # end if

        if self.is_use_gpu:
            tensor_original_translation = tensor_original_translation.to(self.torch_device)
            tensor_new_translation = tensor_new_translation.to(self.torch_device)
        # end if

        with torch.no_grad():
            # distance_mmd = self.mmd_estimator.forward(tensor_original_translation, tensor_new_translation)
            distance_mmd = self.mmd_estimator.forward(tensor_original_translation, tensor_new_translation)
        # end with

        return distance_mmd.mmd.cpu().item()    

    # ------------------------------------------------------

    def _preprocess_tensors(self,
                            seq_tensor: ty.List[torch.Tensor]
                            ) -> torch.Tensor:
        """
        Args:
            seq_tensor: The list of tensors. Each tensor is (T: number of tokens, embed_dim).

        Returns:
            torch.Tensor: The tensor is (N: the num. of documents, D_emb: embedding-size).
        """
        def mode_avg(document_tensor: torch.Tensor) -> torch.Tensor:
            """I want to compute the average of the tensor.
            
            Args:
                document_tensor: The tensor is (T: number of tokens, embed_dim).
            """
            assert len(document_tensor.shape) == 2, f"Expected 2D tensor, got {len(document_tensor.shape)}"
            return torch.mean(document_tensor, dim=0)
        # end mode_avg

        if self.mode_preprocess == "avg":
            return torch.stack([mode_avg(_t) for _t in seq_tensor], dim=0)
        else:
            raise ValueError(f"Invalid mode preprocess: {self.mode_preprocess}")

    def flag_hallucination_one_record(self,
                                      eval_target: EvaluationTargetTranslationPair,
                                      candidate_temperature_parameters: npt.NDArray[np.float32],
                                      n_sampling: int,
                                      n_max_attempts: int = 5) -> MmdErrorFlagResult:
        """
        I want to flag the hallucination of the given translation.

        Args:
            n_max_attempts: The number of maximum attempts to sample the translation.
                FairSeq may fail to sample the translation with a lower \tau value. 
                In this case, I try to sample the translation again.
        """
        assert len(candidate_temperature_parameters) > 0, "Empty candidate temperature parameters."
        assert n_sampling > 0, "Invalid n_sampling value."
        if self.trajectory_rule_smoothing == "savgol_filter":
            assert self.trajectory_rule_smoothing_window is not None
            assert len(candidate_temperature_parameters) > self.trajectory_rule_smoothing_window, \
                (f"`candidate_temperature_parameters` must be > window_length_filter",
                 f"len(candidate_temperature_parameters) == {len(candidate_temperature_parameters)}, window_length_filter == {self.trajectory_rule_smoothing_window}")
        elif self.trajectory_rule_smoothing == "rolling_mean":
            assert self.trajectory_rule_smoothing_window is not None            
            assert len(candidate_temperature_parameters) + 2 > self.trajectory_rule_smoothing_window, \
                (f"`candidate_temperature_parameters` must be > window_length_filter + 2.",
                 f"len(candidate_temperature_parameters) == {len(candidate_temperature_parameters)}, window_length_filter == {self.trajectory_rule_smoothing_window}")
        # end if
                
        seq_sampled_translation_obj: ty.List[_SampledTranslationObject] = []
        # TODO: I wanna do the parallel processing.
        # For each temperature parameter, I generate the translation hypothesis.
        for _tau_param in candidate_temperature_parameters:
            # Check the cache database. Loading it if exists.
            _primary_key_cache_db = self._get_cache_database_key(
                sentence_id=eval_target.sentence_id, temperature=_tau_param, n_sampling=n_sampling)
            _cache_record_translation = self._fetch_cache_records(_primary_key_cache_db, _CacheDatabaseRecordTranslation.__name__)

            if _cache_record_translation is None:
                try:
                    # Sampling with the temperature parameter.
                    _seq_translated_obj = self.translation_handler.sample_multiple_times(
                        input_text=eval_target.source, 
                        temperature=float(_tau_param),
                        n_max_attempts=n_max_attempts,
                        n_sampling=n_sampling)
                except ParameterSettingException as e:
                    module_logger.error(f"Error in sampling: {e}")
                    # Saving the cache.
                    _cache_db_record = _CacheDatabaseRecordTranslation(
                        record_key_id=_primary_key_cache_db,
                        sentence_id=eval_target.sentence_id,
                        n_sampling=n_sampling,
                        temperature=float(_tau_param),
                        source_text=eval_target.source,
                        translation_set_json='',
                        is_success=False)
                    self._save_cache_records(_cache_db_record)                    
                    _sampled_translation_obj = _SampledTranslationObject(
                        tau_parameter=float(_tau_param),
                        n_sample=n_sampling,
                        translation_text=None,
                        is_success=False)
                else:
                    # Collecting the translated text.
                    if isinstance(_seq_translated_obj[0], DecodedTranslationObject):
                        _seq_translated_text = [_obj.target_text for _obj in _seq_translated_obj]  # type: ignore
                    elif isinstance(_seq_translated_obj[0], GeneratedTranslationObject):
                        _seq_translated_text = [_obj.translation_text for _obj in _seq_translated_obj]  # type: ignore
                    else:
                        raise Exception(f"Invalid object: {_seq_translated_obj[0]}")
                    # end if

                    # Saving the cache.
                    _cache_db_record = _CacheDatabaseRecordTranslation(
                        record_key_id=_primary_key_cache_db,
                        sentence_id=eval_target.sentence_id,
                        n_sampling=n_sampling,
                        temperature=float(_tau_param),
                        source_text=eval_target.source,
                        translation_set_json=json.dumps(_seq_translated_text))
                    self._save_cache_records(_cache_db_record)
                    _sampled_translation_obj = _SampledTranslationObject(
                        tau_parameter=float(_tau_param),
                        n_sample=n_sampling,
                        translation_text=_seq_translated_text,
                        is_success=True)                    
            else:
                assert isinstance(_cache_record_translation, _CacheDatabaseRecordTranslation)
                _translation_text = json.loads(_cache_record_translation.translation_set_json)
                _sampled_translation_obj = _SampledTranslationObject(
                    tau_parameter=_cache_record_translation.temperature,
                    n_sample=_cache_record_translation.n_sampling,
                    translation_text=_translation_text,
                    is_success=bool(_cache_record_translation.is_success))
            # end if
            seq_sampled_translation_obj.append(_sampled_translation_obj)
        # end for
        
        # TODO I want to do the parallel processing.
        # For each translation, I convert the text into the embedding.
        for _sampled_obj in seq_sampled_translation_obj:
            # Check the cache database. Loading it if exists.
            _primary_key_cache_db = self._get_cache_database_key(
                sentence_id=eval_target.sentence_id, 
                temperature=_sampled_obj.tau_parameter, 
                n_sampling=_sampled_obj.n_sample)
            _cache_record_embedding = self._fetch_cache_records(_primary_key_cache_db, _CacheDatabaseRecordEmbeddingTensor.__name__)

            if _cache_record_embedding is None:
                assert isinstance(_sampled_obj, _SampledTranslationObject), f"Invalid object: {_sampled_obj}"

                if _sampled_obj.is_success is False:
                    # when the translation is failed, I set the None object. 
                    _sampled_obj.embedding_tensor = None
                else:
                    assert _sampled_obj.translation_text is not None, f"Invalid object: {_sampled_obj}" 
                    _h_enb_unfixed = self.vector_extractor.extract_word_embeddings_batch(
                        seq_sentence=_sampled_obj.translation_text,
                    )
                    _h_emb_fixed = self._preprocess_tensors(_h_enb_unfixed)
                    _sampled_obj.embedding_tensor = _h_emb_fixed

                    # Saving the cache.
                    _cache_db_record = _CacheDatabaseRecordEmbeddingTensor(
                        record_key_id=_primary_key_cache_db,
                        sentence_id=eval_target.sentence_id,
                        n_sampling=n_sampling,
                        temperature=float(_sampled_obj.tau_parameter),
                        embedding_layer_name='decoder.embed_tokens',
                        embedding_tensor=_h_emb_fixed,
                        embedding_tensor_bytes=None)
                    self._save_cache_records(_cache_db_record)
            else:
                assert isinstance(_cache_record_embedding, _CacheDatabaseRecordEmbeddingTensor)
                _sampled_obj.embedding_tensor = _cache_record_embedding.embedding_tensor
            # end if
        # end for

        # I make the embedding tensor of the given translation.
        _primary_key_cache_db = f"{eval_target.sentence_id}_Y"
        _cache_record_embedding_y = self._fetch_cache_records(_primary_key_cache_db, _CacheDatabaseRecordEmbeddingTensor_Y.__name__)
        if _cache_record_embedding_y is None:
            y_emb_unfixed = self.vector_extractor.extract_word_embeddings(translated_text=eval_target.target)
            y_emb_fixed = self._preprocess_tensors([y_emb_unfixed] * 2)
            
            # Saving the cache.
            _cache_db_record = _CacheDatabaseRecordEmbeddingTensor_Y(
                record_key_id=_primary_key_cache_db,
                sentence_id=eval_target.sentence_id,
                embedding_layer_name='decoder.embed_tokens',                
                embedding_tensor=y_emb_fixed,
                embedding_tensor_bytes=None)
            self._save_cache_records(_cache_db_record)
        else:
            assert isinstance(_cache_record_embedding_y, _CacheDatabaseRecordEmbeddingTensor_Y)
            y_emb_fixed = _cache_record_embedding_y.embedding_tensor
            assert isinstance(y_emb_fixed, torch.Tensor), f"Invalid cache: {_cache_record_embedding_y}"
        # end if

        # For each pair of (\tau, H), I compute the MMD distance between the original translation and the hypothesis.
        for _sampled_obj in seq_sampled_translation_obj:
            if _sampled_obj.is_success is False:
                # when the translation is failed, I set the nan value.
                _sampled_obj.mmd_distance = np.nan
            else:
                assert _sampled_obj.embedding_tensor is not None, f"Invalid object: {_sampled_obj}"
                _sampled_obj.mmd_distance = self._compute_mmd_distance(
                    tensor_original_translation=y_emb_fixed,
                    tensor_new_translation=_sampled_obj.embedding_tensor)
            # end if
        # end for

        array_tau = np.array([_obj.tau_parameter for _obj in seq_sampled_translation_obj])
        array_mmd = np.array([_obj.mmd_distance for _obj in seq_sampled_translation_obj])

        if len(array_mmd) == 0:
            raise RuntimeError("Empty MMD distances.")
        # end if
        if len(array_tau) == 0:
            raise RuntimeError("Empty tau parameters.")
        # end if
        if len(array_tau) != len(array_mmd):
            raise RuntimeError(f"Inconsistent lengths. array_tau={len(array_tau)}, array_mmd={len(array_mmd)}")
        # end if

        # Available Data, X: temperature, Y: MMD distance MMD(y, H), where H is the hypothesis (translation).
        shape_function = module_classify_rule_base.classify_function_shape(
            x=array_tau, 
            y=array_mmd,
            rule_version=self.trajectory_rule,
            type_filter=self.trajectory_rule_smoothing,
            window_length=self.trajectory_rule_smoothing_window,)
        
        if shape_function == 'saddle-point':
            _is_hallucination = True
        elif shape_function == 'monotonic-increasing':
            _is_hallucination = False
        else:
            raise Exception(f"Unknown trajectory shape: {shape_function}")
        # end if

        # I want to save the tensor of (N-sample, D: embedding-size).
        # There is None value sometimes when the translation is failed.
        # If so, I set a tensor of 0.0 values.
        __tensor_hypothesis_translation = []
        for _obj in seq_sampled_translation_obj:
            if _obj.is_success is False:
                assert len(y_emb_fixed.shape) == 2, f"Invalid shape: {y_emb_fixed.shape}"
                assert isinstance(y_emb_fixed.shape[1], int), f"Invalid shape: {y_emb_fixed.shape}"
                __size_embedding_tensor = y_emb_fixed.shape[1]
                __tensor_hypothesis_translation.append(torch.zeros(n_sampling, __size_embedding_tensor))
            else:
                __tensor_hypothesis_translation.append(_obj.embedding_tensor)
            # end if
        # end for
        tensor_hypothesis_translation = torch.stack(__tensor_hypothesis_translation)
        assert isinstance(tensor_hypothesis_translation, torch.Tensor), f"Invalid tensor: {tensor_hypothesis_translation}"

        result_obj = MmdErrorFlagResult(
            evaluation_pair=eval_target,
            n_sample=n_sampling,
            tau_parameter=array_tau.tolist(),
            mmd_distances=array_mmd.tolist(),
            tensor_given_translation=y_emb_fixed,
            tensor_hypothesis_translation=tensor_hypothesis_translation,
            hypothesis_translation=[_obj.translation_text for _obj in seq_sampled_translation_obj],
            trajectory_shape=shape_function,
            is_hallucination=_is_hallucination
        )

        return result_obj
