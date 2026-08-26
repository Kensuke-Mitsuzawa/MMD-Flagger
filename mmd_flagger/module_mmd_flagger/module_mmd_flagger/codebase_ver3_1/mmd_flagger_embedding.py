"""Migrated from the previous codebase. I keep the consistent Ver. to the class name."""
import typing as ty
import numpy.typing as npt
import torch
import logging
import dataclasses
import tempfile
from pathlib import Path

from pydantic import BaseModel

import GPUtil

import numpy as np

from mmd_tst_variable_detector.mmd_estimator import (
    QuadraticMmdEstimator, 
    MmdValues,
    KernelMatrixObject,
    QuadraticKernelMatrixContainer)
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from ..module_rules_mmd_trajectory import module_classify_rule_base

from .models import MmdErrorFlagResultVer3

from ....utils.utils_gpu_status import is_cuda_usable

module_logger = logging.getLogger(__name__)


class MmdErrorFlaggerTrajectoryVer3(object):
    def __init__(self,
                #  vector_extractor: BaseVectorExtractorVer2,
                 mmd_estimator: QuadraticMmdEstimator,
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
                 ):
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

    # --------------------------------------------
    # check case


    @staticmethod
    def _get_less_busy_cuda_device() -> int:
        gpu_device_info = GPUtil.getGPUs()
        seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
        gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
        return gpu_id_less_busy[0]

    # ------------------------------------------------------
    
    def _compute_estimate_mmd_distance(self,
                             tensor_original_translation: torch.Tensor,
                             tensor_new_translation: torch.Tensor,
                             is_add_kernel_matrix_object: bool = False
                             ) -> MmdValues:
        """I want to compute the MMD distance between the original and the new translation.
        
        Returns: calculated MMD values     
        """
        is_same_tensor = torch.equal(tensor_original_translation, tensor_new_translation)
        if is_same_tensor:
            module_logger.warning("The original and new translation are the same. Returning MmdValues with 0.0 MMD.")
            return MmdValues(mmd=torch.tensor([0.0]), variance=torch.tensor([np.nan]), ratio=torch.tensor([np.nan]), kernel_matrix_obj=None)
        # end if

        tensor_original_translation = tensor_original_translation.to(self.torch_device)
        tensor_new_translation = tensor_new_translation.to(self.torch_device)

        # Ensure we have at least 2 samples for the dataset side to avoid ZeroDivisionError
        if tensor_original_translation.shape[0] == 1:
            module_logger.debug("Duplicating dataset sample to avoid ZeroDivisionError in MMD calculation.")
            tensor_original_translation = torch.cat([tensor_original_translation, tensor_original_translation], dim=0)
        # end if

        # # bug fix: 2025-06-30, unmatched torch.data type.
        # if tensor_original_translation.dtype != tensor_new_translation.dtype:
        #     tensor_original_translation = tensor_original_translation.to(torch.float16)
        #     tensor_new_translation = tensor_new_translation.to(torch.float16)
        # # end if
        # bug fix: 2025-09-22. Overflow comes when I use fp16.
        # Background: Using Dotproduct kernel, the kernel matrices tend to have large values.
        tensor_original_translation = torch.nan_to_num(tensor_original_translation.to(torch.float32), nan=0.0)
        tensor_new_translation = torch.nan_to_num(tensor_new_translation.to(torch.float32), nan=0.0)

        with torch.no_grad():
            if hasattr(self.mmd_estimator.kernel_obj, "bandwidth"):
                if torch.isnan(self.mmd_estimator.kernel_obj.bandwidth).any() or (self.mmd_estimator.kernel_obj.bandwidth == 0).any():
                    print(f"DEBUG: bandwidth is {self.mmd_estimator.kernel_obj.bandwidth}")
                # end
            # end if
            
            try:
                distance_mmd_obj = self.mmd_estimator.forward(tensor_original_translation, tensor_new_translation, is_add_kernel_matrix_object=is_add_kernel_matrix_object)            
            except ZeroDivisionError:
                # This happens when m=1 or n=1 and variance_term="sutherland_2017"
                module_logger.warning(f"ZeroDivisionError during MMD calculation (m={tensor_original_translation.shape[0]}, n={tensor_new_translation.shape[0]}). Likely due to insufficient samples for variance estimation.")
                distance_mmd_obj = MmdValues(
                    mmd=torch.tensor([0.0]), 
                    variance=torch.tensor([np.nan]), 
                    ratio=torch.tensor([np.nan]), 
                    kernel_matrix_obj=None
                )
            # end try

            # Robust post-processing of MmdValues to avoid negative variance or NaN ratio
            if distance_mmd_obj is not None:
                orig_var = distance_mmd_obj.variance
                orig_ratio = distance_mmd_obj.ratio

                # Sutherland U-statistic variance can be negative or zero. Clamp to small positive value.
                if torch.isnan(orig_var) or torch.isinf(orig_var) or orig_var.item() <= 0.0:
                    new_variance = torch.tensor([1e-8], device=orig_var.device, dtype=orig_var.dtype)
                else:
                    new_variance = orig_var

                safe_mmd = torch.nan_to_num(distance_mmd_obj.mmd, nan=0.0)

                # Recompute ratio if either variance was clamped or original ratio is NaN/Inf
                if torch.isnan(orig_ratio) or torch.isinf(orig_ratio) or new_variance.item() == 1e-8:
                    new_ratio = safe_mmd / torch.sqrt(new_variance)
                else:
                    new_ratio = orig_ratio

                distance_mmd_obj = MmdValues(
                    mmd=safe_mmd,
                    variance=new_variance,
                    ratio=new_ratio,
                    kernel_matrix_obj=distance_mmd_obj.kernel_matrix_obj
                )
        # end with

        return distance_mmd_obj

    def flag_hallucination(self,
                           processed_embedding_hypothesis: torch.Tensor,
                           tau2processed_embedding_samples: ty.Dict[float, torch.Tensor],
                           is_skip_min_tau_unavailable: bool = True,
                           is_add_kernel_matrix_object: bool = False,
                           tau_zero: ty.Optional[float] = None) -> MmdErrorFlagResultVer3:
        """
        I want to flag the hallucination of the given tensor (of hypothesis) v.s. tensor (of set of samples).

        Args:
            n_max_attempts: The number of maximum attempts to sample the translation.
                FairSeq may fail to sample the translation with a lower \tau value. 
                In this case, I try to sample the translation again.
            is_skip_min_tau_unavailable: True then skip the flagging when the MMD of min(tau) is unavailable.
                Empirically, MMD of tau=0.1 is the strong factor to classify.  
            is_add_kernel_matrix_object: if True, saving K_xx, K_yy, K_xy during MMD calculation.
        """
        candidate_temperature_parameters = list(tau2processed_embedding_samples.keys())
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
                              tensor_beam_search: torch.Tensor, 
                              tensor_stochastic_sampling: torch.Tensor,
                              is_add_kernel_matrix_object: bool = False
                                  ) -> ty.Tuple[float, MmdValues]:
            if tensor_stochastic_sampling.shape[0] <= 1:
                # Avoid ZeroDivisionError in variance
                _mmd_values = MmdValues(mmd=torch.tensor([0.0]), variance=torch.tensor([np.nan]), ratio=torch.tensor([np.nan]), kernel_matrix_obj=None)
            else:
                _mmd_values = self._compute_estimate_mmd_distance(
                    tensor_original_translation=tensor_beam_search,
                    tensor_new_translation=tensor_stochastic_sampling,
                    is_add_kernel_matrix_object=is_add_kernel_matrix_object)
            return (tau_param, _mmd_values)
        # end def

        def _get_tau_and_mmd_sequence(fixed_tensor_beam_search: torch.Tensor,
                                      seq_tau_sequence: ty.List[float], 
                                      seq_tensor_tau: ty.List[ty.Optional[torch.Tensor]],
                                      is_add_kernel_matrix_object: bool = False
                                      ) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ty.List[KernelMatrixObject]]:
            seq_tau = []
            seq_mmd = []
            seq_variance = []
            seq_test_power = []
            seq_kernel_obj: ty.List[KernelMatrixObject] = []
            for _tau_param, _tensor_stochastic in zip(seq_tau_sequence, seq_tensor_tau):
                if _tensor_stochastic is None:
                    seq_mmd.append(None)
                    seq_variance.append(None)
                    seq_test_power.append(None)
                    seq_tau.append(_tau_param)
                else:
                    _tau_param, _mmd_values = _get_mmd_distance(
                        tau_param=_tau_param, 
                        tensor_beam_search=fixed_tensor_beam_search, 
                        tensor_stochastic_sampling=_tensor_stochastic,
                        is_add_kernel_matrix_object=is_add_kernel_matrix_object)

                    # ---- logging block ----
                    module_logger.debug(f"MMD for tau={_tau_param}: {_mmd_values.mmd.item()}")
                    module_logger.debug(f"Variance for tau={_tau_param}: {_mmd_values.variance.item()}")
                    module_logger.debug(f"Ratio for tau={_tau_param}: {_mmd_values.ratio.item()}")
                    # ---- logging block ----
                    
                    seq_tau.append(_tau_param)
                    _mmd_val = _mmd_values.mmd.item()
                    if np.isnan(_mmd_val):
                        module_logger.debug(f"[NaN detected] MMD is NaN for tau={_tau_param}. "
                                            f"Hypothesis mag: {fixed_tensor_beam_search.abs().mean().item():.2e}, "
                                            f"Sample mag: {_tensor_stochastic.abs().mean().item():.2e}")
                    seq_mmd.append(_mmd_val)
                    seq_variance.append(_mmd_values.variance.item())
                    seq_test_power.append(None if _mmd_values.ratio is None else _mmd_values.ratio.item())
                    
                    if is_add_kernel_matrix_object:
                        if _mmd_values.kernel_matrix_obj is not None:
                            seq_kernel_obj.append(_mmd_values.kernel_matrix_obj)
                        else:
                            module_logger.debug(f"Kernel matrix object is None for tau={_tau_param}. Skipping.")
                        # end if
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

            array_mmd = array_mmd.flatten()
            array_mmd_no_nan = array_mmd[~np.isnan(array_mmd)]
            array_tau_no_nan = array_tau[~np.isnan(array_mmd)]
            
            if len(array_mmd_no_nan) == 0:
                _msg = (f"MMD trajectory is empty after NaN removal. "
                        f"All {len(array_mmd)} MMD values were NaN. "
                        f"This often indicates numerical overflow in the kernel (e.g. DotProductKernel) "
                        f"due to large embedding magnitudes.")
                module_logger.error(_msg)
                raise RuntimeError(_msg)

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
        seq_tensor_tau = [tau2processed_embedding_samples.get(_tau, None) for _tau in seq_tau_sequence]

        array_tau, array_mmd, array_var, array_test_power_approx, seq_kernel_obj = _get_tau_and_mmd_sequence(
            fixed_tensor_beam_search=processed_embedding_hypothesis,
            seq_tau_sequence=seq_tau_sequence, 
            seq_tensor_tau=seq_tensor_tau,
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

