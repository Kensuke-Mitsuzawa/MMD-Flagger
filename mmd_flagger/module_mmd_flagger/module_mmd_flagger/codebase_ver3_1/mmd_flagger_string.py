# ----- Ver3 Flagger String Based -----
import typing as ty
import torch
import dataclasses
import tempfile
from pathlib import Path

import GPUtil

import numpy as np

from mmd_tst_variable_detector.mmd_estimator import (
    QuadraticMmdEstimator, 
    MmdValues,
    KernelMatrixObject,
    QuadraticKernelMatrixContainer)
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from ..module_rules_mmd_trajectory import module_classify_rule_base

from ...module_kernels.string_based_gaussian_kernel import StringBasedGaussianKernel

from ....utils.utils_gpu_status import is_cuda_usable

from .models import MmdErrorFlagResultVer3

from ...module_kernels.module_distance.module_preprocessing import nltk_preprocess_text


class MmdErrorFlaggerTrajectoryVer3StringBased(object):
    def __init__(self,
                 mmd_estimator: QuadraticMmdEstimator,
                 # saving cache file
                 path_cache_dir: ty.Optional[Path] = None,
                 # ------------------------------
                 # flagging rule configurations
                 trajectory_rule: str = 'v1',
                 trajectory_rule_smoothing: str = 'no_filter',
                 trajectory_rule_smoothing_window: ty.Optional[int] = None,
                 # ------------------------------
                 is_use_gpu: bool = True):
        self.mmd_estimator = mmd_estimator

        assert isinstance(self.mmd_estimator.kernel_obj, StringBasedGaussianKernel)
        
        # fixing the GPU device for the exec. speed.
        if is_use_gpu and is_cuda_usable():
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

    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]


    def flag_hallucination(self,
                           hypothesis_sequences: ty.List[str],
                           tau2stochastic_sequences: ty.Dict[float, ty.List[str]],
                           is_skip_min_tau_unavailable: bool = True,
                           is_add_kernel_matrix_object: bool = False,
                           tau_zero: ty.Optional[float] = None) -> MmdErrorFlagResultVer3:
        """
        I want to flag the hallucination of the given hypothesis token-sequences v.s. token-sequences per temperature parameter.

        Args:
            n_max_attempts: The number of maximum attempts to sample the translation.
                FairSeq may fail to sample the translation with a lower \tau value. 
                In this case, I try to sample the translation again.
            is_skip_min_tau_unavailable: True then skip the flagging when the MMD of min(tau) is unavailable.
                Empirically, MMD of tau=0.1 is the strong factor to classify.  
            is_add_kernel_matrix_object: if True, saving K_xx, K_yy, K_xy during MMD calculation.
        """
        candidate_temperature_parameters = list(tau2stochastic_sequences.keys())
        assert len(candidate_temperature_parameters) > 0, "Empty candidate temperature parameters."
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

        def _get_mmd_distance(tau_param: float, 
                              text_hypothesis: ty.List[str], 
                              text_stochastic_sampling: ty.List[str],
                              is_add_kernel_matrix_object: bool = False
                              ) -> ty.Tuple[float, MmdValues]:
            
            is_same_tensor = text_hypothesis == text_stochastic_sampling
            if is_same_tensor or len(text_stochastic_sampling) <= 1:
                return tau_param, MmdValues(mmd=torch.tensor([0.0]), variance=torch.tensor([np.nan]), ratio=torch.tensor([np.nan]), kernel_matrix_obj=None)
            # end if

            # ---- text preprocessing: document string -> token sequence ----
            set_doc_tokens_hypothesis = nltk_preprocess_text(text_hypothesis)
            set_doc_tokens_stochastic = nltk_preprocess_text(text_stochastic_sampling)
            # ----

            # exceptional case operation: when x list is a single element, make it copy.
            if len(set_doc_tokens_hypothesis) == 1:
                set_doc_tokens_hypothesis = set_doc_tokens_hypothesis + set_doc_tokens_hypothesis
            # end if

            with torch.no_grad():
                # distance_mmd = self.mmd_estimator.forward(tensor_original_translation, tensor_new_translation)
                distance_mmd_obj = self.mmd_estimator.forward(set_doc_tokens_hypothesis, set_doc_tokens_stochastic, is_add_kernel_matrix_object=is_add_kernel_matrix_object)
            # end with
            
            return (tau_param, distance_mmd_obj)
        # end def

        def _get_tau_and_mmd_sequence(text_hypothesis: ty.List[str],
                                      seq_tau_sequence: ty.List[float], 
                                      text_tau: ty.List[ty.List[ty.Optional[str]]],
                                      is_add_kernel_matrix_object: bool = False
                                      ) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ty.List[KernelMatrixObject]]:
            seq_tau = []
            seq_mmd = []
            seq_variance = []
            seq_test_power = []
            seq_kernel_obj: ty.List[KernelMatrixObject] = []
            for _tau_param, _text_stochastic in zip(seq_tau_sequence, text_tau):
                if _text_stochastic is None:
                    seq_mmd.append(None)
                    seq_variance.append(None)
                    seq_test_power.append(None)
                    seq_tau.append(_tau_param)
                else:
                    _tau_param, _mmd_values = _get_mmd_distance(
                        text_hypothesis=text_hypothesis,
                        text_stochastic_sampling=_text_stochastic,
                        tau_param=_tau_param,
                        is_add_kernel_matrix_object=is_add_kernel_matrix_object)
                    seq_tau.append(_tau_param)
                    seq_mmd.append(_mmd_values.mmd.cpu().numpy())
                    seq_variance.append(_mmd_values.variance.cpu().numpy())
                    seq_test_power.append(None if _mmd_values.ratio is None else _mmd_values.ratio.cpu().numpy())
                    
                    if is_add_kernel_matrix_object:
                        assert _mmd_values.kernel_matrix_obj is not None
                        seq_kernel_obj.append(_mmd_values.kernel_matrix_obj)
                    # end if
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

            return array_tau, array_mmd, array_var, array_test_power, seq_kernel_obj
        
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
                window_length=self.trajectory_rule_smoothing_window,
                tau_zero=tau_zero)
            
            if shape_function == 'saddle-point':
                _is_hallucination = True
            elif shape_function == 'monotonic-increasing':
                _is_hallucination = False
            else:
                raise Exception(f"Unknown trajectory shape: {shape_function}")
            # end if

            return _is_hallucination, shape_function
        # end def
        
        seq_tau_sequence = candidate_temperature_parameters
        seq_tokens_seq_tau: ty.List[ty.List[ty.Optional[str]]] = [tau2stochastic_sequences.get(_tau, None) for _tau in seq_tau_sequence]

        array_tau, array_mmd, array_var, array_test_power_approx, seq_kernel_obj = _get_tau_and_mmd_sequence(
            text_hypothesis=hypothesis_sequences,
            seq_tau_sequence=seq_tau_sequence,
            text_tau=seq_tokens_seq_tau,
            is_add_kernel_matrix_object=is_add_kernel_matrix_object)
        
        is_hallucination, shape_function = _classify_trajectory(array_tau, array_mmd)

        # -------------------------------------------------
        # if `seq_kernel_obj` is there, convert all to the cpu mode.
        kernel_containers = []
        for _kernel_container_obj in seq_kernel_obj:
            # `QuadraticKernelMatrixContainer`
            assert isinstance(_kernel_container_obj.kernel_matrix_container, QuadraticKernelMatrixContainer)
            _kernel_container_obj.kernel_matrix_container.k_xx = _kernel_container_obj.kernel_matrix_container.k_xx.cpu()
            _kernel_container_obj.kernel_matrix_container.k_yy = _kernel_container_obj.kernel_matrix_container.k_yy.cpu()
            _kernel_container_obj.kernel_matrix_container.k_xy = _kernel_container_obj.kernel_matrix_container.k_xy.cpu()                        
        
            kernel_containers.append(dataclasses.asdict(_kernel_container_obj))
        # end for
        # -------------------------------------------------

        result_obj = MmdErrorFlagResultVer3(
            n_sample=-1,
            tau_parameter=array_tau.tolist(),
            mmd_distances=array_mmd.tolist(),
            variance_mmd=array_var.tolist(),
            test_power_approximation=array_test_power_approx.tolist(),
            trajectory_shape=shape_function,
            is_hallucination=is_hallucination,
            kernel_containers=kernel_containers
        )

        return result_obj

