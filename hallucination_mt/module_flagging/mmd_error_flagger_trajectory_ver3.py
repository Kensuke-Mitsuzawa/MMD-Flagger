import typing as ty
import numpy.typing as npt
import torch
import logging
import tqdm
import dataclasses
import tempfile
import pickle
import json
import copy
from pathlib import Path

import zlib
import GPUtil

import numpy as np

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel

from ..commons.data_models import EvaluationTargetTranslationPair

from ..module_assessments.custom_tqdm_handler import TqdmLoggingHandler

from ..module_assessments.module_management_db import module_sqlite3_handler
from ..exceptions import ParameterSettingException
from ..module_hidden_vector_extractor.ver2.module_base import (
    BaseVectorExtractorVer2,
    TranslationResultContainer
)
from .module_classify_trajectory import module_classify_rule_base
from .module_mmd_error_flagger_trajectory_ver3 import (
    TensorPreprocessorVer1,
    MmdEstimatorInitialiserVer1
)

from ..logger_module import formatter

module_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
module_logger.addHandler(handler)

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
    source_text: str

    temperature: float
    translation_mode: str
    n_sampling: ty.Optional[int]
    
    translation_result_container: ty.Optional[TranslationResultContainer]
    is_success: bool

    def __post_init__(self):
        assert self.translation_mode in ('beam', 'stochastic')


@dataclasses.dataclass
class MmdErrorFlagResultVer3:
    evaluation_pair: EvaluationTargetTranslationPair
    n_sample: int

    tau_parameter: ty.List[float]
    mmd_distances: ty.List[float]
    variance_mmd: ty.List[float]  # empiriclly estimated variance of MMD. See Sutherland, 2017.
    test_power_approximation: ty.List[float]  # empiriclly approximation of Test-Power regarding Two Sample Testing. See Sutherland, 2017.
    
    translation_stable: str
    hypothesis_translation: ty.List[ty.Optional[ty.List[str]]]

    trajectory_shape: str
    is_hallucination: bool
# end class




class MmdErrorFlaggerTrajectoryVer3(object):
    def __init__(self,
                 vector_extractor: BaseVectorExtractorVer2,
                 mmd_estimator: QuadraticMmdEstimator,
                 tensor_preprocessor: TensorPreprocessorVer1,
                 # ------------------------------
                 # options about generating vectors
                mode_target_embedding_layer: ty.Optional[str] = None,
                 # ------------------------------
                 # translation options
                 option_translation_max_a: float = 0.0,
                 option_translation_max_b: int = 200,
                 option_is_sampling_in_iteration: bool = False,  # batch sampling option (of stochastic). If True, no-batch, False then batch.
                 # ------------------------------
                 # saving cache file
                 path_cache_dir: ty.Optional[Path] = None,
                 # ------------------------------
                 # flagging rule configurations
                 trajectory_rule: str = 'v1',
                 trajectory_rule_smoothing: str = 'no_filter',
                 trajectory_rule_smoothing_window: ty.Optional[int] = None,
                 # ------------------------------
                 is_use_gpu: bool = True,
                 target_layers_extraction: ty.Optional[ty.List[str]] = None  # list of layer name of extract and save. If None, I save all.
                 ):
        assert isinstance(vector_extractor, (BaseVectorExtractorVer2, )), \
            "vector_extractor must be an instance of FairSeqVectorExtractor or TransformerVectorExtractor"
        
        self.vector_extractor = vector_extractor

        self.tensor_preprocessor = tensor_preprocessor

        # translation options
        self.option_translation_max_a = option_translation_max_a
        self.option_translation_max_b = option_translation_max_b
        self.option_is_sampling_in_iteration = option_is_sampling_in_iteration

        # when the given `target_layers_extraction` is None. I extract and save all possible layers.
        if target_layers_extraction is None:
            __, _seq_decoder_layers = self.vector_extractor.translation_handler.get_all_possible_layers()
            _seq_decoder_layers.append(self.vector_extractor.translation_handler._get_decoder_word_embedding_layer_name())
            self.target_layers_extraction = _seq_decoder_layers
        else:
            assert len(target_layers_extraction) > 0
            self.target_layers_extraction = target_layers_extraction
        # end if

        # ------------------------------------------------------
        self.mode_target_embedding_layer: str = self.vector_extractor.translation_handler._get_decoder_word_embedding_layer_name() if mode_target_embedding_layer is None else mode_target_embedding_layer

        # self.__check_mmd_estimator_and_parameter(mmd_estimator)
        assert self.mode_target_embedding_layer in self.target_layers_extraction, \
            f'The specified layer name {self.mode_target_embedding_layer} does not exist in the defined layers -> {self.target_layers_extraction}'
        # ------------------------------------------------------
        # fixing the GPU device for the exec. speed.
        if is_use_gpu and torch.cuda.is_available():
            _device_id_gpu = self._get_less_busy_cuda_device()
            self.torch_device = torch.device(f"cuda:{_device_id_gpu}")
            self.is_use_gpu = True
        else:
            self.torch_device = torch.device("cpu")
            self.is_use_gpu = False
        # end if

        # ------------------------------------------------------
        # Case Gaussian Kernel and dimension-wise
        # TODO Check if the length scale vector size == required vector size.
        self.mmd_estimator = mmd_estimator.to(self.torch_device)

        # ------------------------------------------------------
        # Setting the cache directory
        # self.cache_file_name = cache_file_name
        if path_cache_dir is None:
            self.path_cache_dir = Path(tempfile.mkdtemp())
        else:
            self.path_cache_dir = path_cache_dir
        # end if        
        self.path_cache_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------
        # setting attributes of trajectory classifier

        assert trajectory_rule_smoothing in module_classify_rule_base.POSSIBLE_FILTERS, f"Invalid type trajectory shape: {trajectory_rule_smoothing}"
        self.trajectory_rule = trajectory_rule
        self.trajectory_rule_smoothing = trajectory_rule_smoothing
        self.trajectory_rule_smoothing_window = trajectory_rule_smoothing_window
        # ------------------------------------------------------

    # --------------------------------------------
    # check case

    # def __check_mmd_estimator_and_parameter(self, mmd_estimator: ty.Optional[QuadraticMmdEstimator]):
    #     """When the MMD estimator is given, then `mode_max_token_length_vector_concat` should be the number."""
    #     if mmd_estimator is not None:
    #         if self.mode_vector_preprocess == 'concat':
    #             assert isinstance(self.mode_max_token_length_vector_concat, int), '`mode_max_token_length_vector_concat` must be the integer number when the mmd estimator is given. Check the file "configs.json" at the MMD file.'
    #         # end if
    #         # note: no need to set the number of the 'avg' mode.
    #     # end if


    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]


    @staticmethod
    def _get_cache_database_key(sentence_id: str,
                                temperature: float,
                                n_sampling: int) -> str:
        return f"{sentence_id}_{temperature}_{n_sampling}"

    # ------------------------------------------------------
    
    def _compute_mmd_distance(self,
                             tensor_original_translation: torch.Tensor,
                             tensor_new_translation: torch.Tensor
                             ) -> ty.Tuple[float, float, float]:
        """I want to compute the MMD distance between the original and the new translation."""
        is_same_tensor = torch.equal(tensor_original_translation, tensor_new_translation)
        if is_same_tensor:
            return 0.0, np.nan, np.nan
        # end if

        tensor_original_translation = tensor_original_translation.to(self.torch_device)
        tensor_new_translation = tensor_new_translation.to(self.torch_device)

        with torch.no_grad():
            # distance_mmd = self.mmd_estimator.forward(tensor_original_translation, tensor_new_translation)
            distance_mmd_obj = self.mmd_estimator.forward(tensor_original_translation, tensor_new_translation)
        # end with

        distance_mmd = distance_mmd_obj.mmd.cpu().item()
        variance_mmd = distance_mmd_obj.variance.cpu().item()
        assert distance_mmd_obj.ratio is not None
        test_power_approx = distance_mmd_obj.ratio.cpu().item()

        return distance_mmd, variance_mmd, test_power_approx    

    def flag_hallucination_one_record(self,
                                      eval_target: EvaluationTargetTranslationPair,
                                      candidate_temperature_parameters: ty.List[float],
                                      n_sampling: int,
                                      n_max_attempts: int = 5,
                                      batch_size_sampling: int = 5,
                                      is_skip_min_tau_unavailable: bool = True) -> MmdErrorFlagResultVer3:
        """
        I want to flag the hallucination of the given translation.

        Args:
            n_max_attempts: The number of maximum attempts to sample the translation.
                FairSeq may fail to sample the translation with a lower \tau value. 
                In this case, I try to sample the translation again.
            is_skip_min_tau_unavailable: True then skip the flagging when the MMD of min(tau) is unavailable.
                Empirically, MMD of tau=0.1 is the strong factor to classify.  
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

        # -----------------------------------------------------------------------
        # closures for stochastic sampling utils

        # def _save_stochastic_cache_translation(tau_param: float,
        #                                        translation_results: ty.List[TranslationResultContainer]):
        #     """Saving the file into the following path
            
        #     'stochastic'/`sentence-id`/`n_sampling`/`tau_value`
        #     """
        #     _key_path = self._generate_file_path_stochastic(
        #         sentence_id=eval_target.sentence_id,
        #         n_sampling=n_sampling,
        #         tau_param=tau_param)
        #     path_file = self.path_cache_dir / _key_path
        #     path_file.parent.mkdir(parents=True, exist_ok=True)
            
        #     seq_obj_dict = [_obj._asdict() for _obj in translation_results]

        #     pickled_data = pickle.dumps(seq_obj_dict)
        #     compressed_data_zlib = zlib.compress(pickled_data)
        #     with open(path_file, "wb") as f:
        #         f.write(compressed_data_zlib)
        #     # end with

        # def _open_stochastic_cache_translation(tau_param: float) -> ty.List[TranslationResultContainer]:
        #     file_path = self.path_cache_dir / self._generate_file_path_stochastic(
        #         sentence_id=eval_target.sentence_id,
        #         n_sampling=n_sampling,
        #         tau_param=tau_param)

        #     with file_path.open('rb') as f:
        #         obj_saved = pickle.loads(zlib.decompress(f.read()))
        #     # end with
            
        #     return_seq = [TranslationResultContainer(**_obj) for _obj in obj_saved]
        #     return return_seq

        def _form_vector_tensor(translation_result: ty.List[TranslationResultContainer]) -> torch.Tensor:
            """I collect list of translation container. I extract tensor object. I form a fixed shape tensor."""
            stack_tensor = []
            for _container in translation_result:
                assert isinstance(_container.dict_layer_embeddings, dict)
                assert self.mode_target_embedding_layer in _container.dict_layer_embeddings
                stack_tensor.append(_container.dict_layer_embeddings[self.mode_target_embedding_layer])
            # end for
            tensor_formed = self.tensor_preprocessor.preprocess_tensors(
                seq_tensor=stack_tensor,
                is_calibration_mode=False)
            return tensor_formed
        
        def _run_stochastic_translation_and_extract_vector(input_text: EvaluationTargetTranslationPair, 
                                                           tau_param: float) -> ty.List[TranslationResultContainer]:
            
            _seq_translated_obj = self.vector_extractor.translation_stochatstic_sampling(
                input_text=input_text,
                temperature=tau_param,
                n_sampling=n_sampling,
                max_len_a=self.option_translation_max_a,
                max_len_b=self.option_translation_max_b,
                n_max_attempts=10,
                batch_size=batch_size_sampling,
                target_layers_extraction=self.target_layers_extraction,
                is_auto_recovery_sampling=True,
                is_sampling_in_iteration=self.option_is_sampling_in_iteration)
            return _seq_translated_obj

        def _fetch_cache_or_run_stochastic_translation() -> ty.Tuple[ty.List[float], ty.List[ty.Optional[torch.Tensor]], ty.List[ty.Optional[ty.List[str]]]]:
            """I check the existing cache file. 
            If it exists, I load the cache file. Otherwise, I run the translation"""

            # source_text = eval_target.source
            seq_tau_param = []
            seq_tensor = []
            seq_set_translation_text: ty.List[ty.Optional[ty.List[str]]] = []
            for _tau_param in candidate_temperature_parameters:
                _tau_param = round(_tau_param, 3)  # make round the temperature number. Too detail float number makes an error.
                try:
                    seq_translation_stack = _run_stochastic_translation_and_extract_vector(
                        input_text=eval_target,
                        tau_param=_tau_param
                    )
                except Exception as e:
                    module_logger.error(f'Encountering translation error. But, it continues to the next iteration. tau={_tau_param}. The original message is {e}')
                    seq_tau_param.append(_tau_param)
                    seq_tensor.append(None)
                    seq_set_translation_text.append(None)
                    continue
                # end try

                tensor_vector = _form_vector_tensor(seq_translation_stack)
                seq_tau_param.append(_tau_param)
                seq_tensor.append(tensor_vector)
                seq_set_translation_text.append([_obj.translation_text for _obj in seq_translation_stack])
            # end for
            assert len(seq_tau_param) == len(seq_tensor)
            assert len([_e for _e in seq_tensor if _e is not None]) > 0, f"All `seq_tensor` are None. All translations failed at all tau parameters. Check."
            return seq_tau_param, seq_tensor, seq_set_translation_text
        # end def

        # -----------------------------------------------------------------------
        # closures for beam search utils

        def _fetch_cache_or_run_beam_search_translation(tau_param=1.0) -> ty.Tuple[torch.Tensor, str]:
            translation_result = self.vector_extractor.translation_handler.translate_beam_search(
                input_text=eval_target,
                temperature=tau_param,
                max_len_a=self.option_translation_max_a,
                max_len_b=self.option_translation_max_b,
                target_layers_extraction=self.target_layers_extraction
            )

            assert isinstance(translation_result.dict_layer_embeddings, dict)
            assert self.mode_target_embedding_layer in translation_result.dict_layer_embeddings, f'The specified layer name {self.mode_target_embedding_layer} is not listed in {translation_result.dict_layer_embeddings.keys()}'
            
            tensor_translation = translation_result.dict_layer_embeddings[self.mode_target_embedding_layer]

            return tensor_translation, translation_result.translation_text
        # end def

        def _compute_mmd_distance(tau_param: float, 
                                  tensor_beam_search: torch.Tensor, 
                                  tensor_stochastic_sampling: torch.Tensor
                                  ) -> ty.Tuple[float, float, float, float]:
            # TODO: delete
            # reshaping the tensor of the stochastic sampling results.
            # fixed_tensor_stochastic_sampling = self._preprocess_tensors(
            #     seq_tensor=tensor_stochastic_sampling,
            #     mode_vector_preprocess=self.mode_vector_preprocess,
            #     mode_max_token_length_vector_concat=self.mode_max_token_length_vector_concat)
            _mmd_distance, _mmd_var, _test_power_approx = self._compute_mmd_distance(
                tensor_original_translation=tensor_beam_search,
                tensor_new_translation=tensor_stochastic_sampling)
            return tau_param, _mmd_distance, _mmd_var, _test_power_approx
        # end def

        def _get_tau_and_mmd_sequence(fixed_tensor_beam_search: torch.Tensor,
                                      seq_tau_sequence: ty.List[float], 
                                      seq_tensor_tau: ty.List[ty.Optional[torch.Tensor]]
                                      ) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            seq_tau = []
            seq_mmd = []
            seq_variance = []
            seq_test_power = []
            for _tau_param, _tensor_stochastic in zip(seq_tau_sequence, seq_tensor_tau):
                if _tensor_stochastic is None:
                    seq_mmd.append(None)
                    seq_variance.append(None)
                    seq_test_power.append(None)
                    seq_tau.append(_tau_param)
                else:
                    _tau_param, _mmd_distance, _var_mmd, _test_power = _compute_mmd_distance(
                        tau_param=_tau_param, 
                        tensor_beam_search=fixed_tensor_beam_search, 
                        tensor_stochastic_sampling=_tensor_stochastic)
                    seq_tau.append(_tau_param)
                    seq_mmd.append(_mmd_distance)
                    seq_variance.append(_var_mmd)
                    seq_test_power.append(_test_power)
            # end for

            array_mmd_non_none = [_e for _e in seq_mmd if _e is not None]
            if len(array_mmd_non_none) == 0:
                raise RuntimeError("Empty MMD distances.")
            # end if
            if len(seq_tau) == 0:
                raise RuntimeError("Empty tau parameters.")
            # end if
            if len(seq_tau) != len(seq_mmd):
                raise RuntimeError(f"Inconsistent lengths. array_tau={len(seq_tau)}, array_mmd={len(seq_mmd)}")
            # end if
            if is_skip_min_tau_unavailable:
                _index_min_tau = np.argsort(np.array(seq_tau))[0]
                if seq_mmd[_index_min_tau] is None:
                    raise RuntimeError(f'MMD at min(tau)={min(seq_tau)} is unavailable (None). The value is a strong factor, thus Exception. Hint: set is_skip_min_tau_unavailable=False')
                # end if
            # end if


            array_mmd_non_none = [(_tau, _mmd, _var, _test) for _tau, _mmd, _var, _test in zip(seq_tau, seq_mmd, seq_variance, seq_test_power) if _mmd is not None]

            array_tau = np.array([_t[0] for _t in array_mmd_non_none])
            array_mmd = np.array([_t[1] for _t in array_mmd_non_none])
            array_var = np.array([_t[2] for _t in array_mmd_non_none])
            array_test_power = np.array([_t[3] for _t in array_mmd_non_none])

            return array_tau, array_mmd, array_var, array_test_power
        
        def _classify_trajectory(array_tau: np.ndarray, array_mmd: np.ndarray):
            assert len(array_mmd) == len(array_tau)

            array_mmd_no_nan = array_mmd[~np.isnan(array_mmd)]
            array_tau_no_nan = array_tau[~np.isnan(array_mmd)]

            # Available Data, X: temperature, Y: MMD distance MMD(y, H), where H is the hypothesis (translation).
            shape_function = module_classify_rule_base.classify_function_shape(
                x=array_tau_no_nan, 
                y=array_mmd_no_nan,
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

            return _is_hallucination, shape_function
        # end def

        # the prepared tensor. A list of tensor, (n-token, dim). The list length is the length of temperature.
        seq_tau_sequence, seq_tensor_tau, seq_set_translations = _fetch_cache_or_run_stochastic_translation()
        # the prepared tensor is (n-token, dim)
        tensor_beam_search, translation_stable = _fetch_cache_or_run_beam_search_translation()

        # reshaping the tensor of the beam search.
        fixed_tensor_beam_search = self.tensor_preprocessor.preprocess_tensors(
            seq_tensor=[tensor_beam_search] * 2)

        array_tau, array_mmd, array_var, array_test_power_approx = _get_tau_and_mmd_sequence(
            fixed_tensor_beam_search,
            seq_tau_sequence, 
            seq_tensor_tau)

        
        is_hallucination, shape_function = _classify_trajectory(array_tau, array_mmd)


        result_obj = MmdErrorFlagResultVer3(
            evaluation_pair=eval_target,
            n_sample=n_sampling,
            tau_parameter=array_tau.tolist(),
            mmd_distances=array_mmd.tolist(),
            variance_mmd=array_var.tolist(),
            test_power_approximation=array_test_power_approx.tolist(),
            translation_stable=translation_stable,
            hypothesis_translation=seq_set_translations,
            trajectory_shape=shape_function,
            is_hallucination=is_hallucination
        )

        return result_obj
