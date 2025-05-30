import typing as ty
import logging
import copy
import tempfile
import json
import GPUtil

from pathlib import Path

import torch
import numpy as np

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.kernels.gaussian_kernel import (
    QuadraticKernelGaussianKernel,
    DistributedFunctionArg,
    L2Distance,
    KernelMatrixObject,
    QuadraticKernelMatrixContainer)


from ...commons.data_models import EvaluationTargetTranslationPair
from .tensor_preprocessor import TensorPreprocessorVer1
from ...module_hidden_vector_extractor.ver2.module_base import BaseVectorExtractorVer2
from .dot_product_kernel import DotProductKernel

from ...logger_module import formatter


module_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
module_logger.addHandler(handler)


file_name_mmd_estimator: str = 'mmd_estimator.pt'
file_name_mmd_estimator_config: str = 'configs.json'


KERNEL_TYPE = ('gaussian', 'dot')
KERNEL_LENGTH_SCALE_MEDIAN_OPTION = ('dimensionwise', 'single', 'no-gaussian')


# -------------------------------------------------------------------------------

def _get_less_busy_cuda_device() -> int:
    gpu_device_info = GPUtil.getGPUs()
    seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
    gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
    return gpu_id_less_busy[0]



def compute_length_scale_dataset_dimension_d(args: DistributedFunctionArg,
                                             percentile: int,
                                             torch_device: torch.device) -> ty.Tuple[int, torch.Tensor]:
    """Ptivate API. Compute a length scale for a dimension.
    
    Parameters
    ----------
    args: DistributedFunctionArg
        
    Returns
    -------
    A tuple (int, torch.Tensor).
    
    torch.Tensor
        A length scale for a dimension.
    """
    x_projected_m = args.x_projected_m
    y_projected_m = args.y_projected_m
    n_dimension = args.n_dimension

    d_function = args.distance_function
    heuristic_operation = args.heuristic_operation

    x_projected_m = x_projected_m.to(torch_device)
    y_projected_m = y_projected_m.to(torch_device)

    __d_container = d_function(x_projected_m, y_projected_m, True)
    a_m = __d_container.d_xx
    b_m = __d_container.d_yy
    c_m = __d_container.d_xy
    

    tensor_combined = torch.cat([a_m, b_m, c_m])
    gamma_m = torch.quantile(tensor_combined, percentile / 100.0)
   
    # if heuristic_operation == 'median':
    #     gamma_m: torch.Tensor = torch.median(torch.cat([a_m, b_m, c_m]))
    # elif heuristic_operation == 'mean':
    #     raise NotImplementedError("The mean operation is commented out.")
    #     gamma_m: torch.Tensor = torch.mean(torch.cat([a_m, b_m, c_m]))
    # else:
    #     raise Exception(f'heuristic_operation == {heuristic_operation} does not exist.')
    # # end if

    gamma_m = gamma_m.cpu()

    return args.index_dimension, gamma_m * n_dimension


def euclidean_distances_squared(X, Y=None):
    if Y is None:
        Y = X
    # Compute squared norms of X and Y
    X_norm = (X**2).sum(dim=1).unsqueeze(1)  # shape: (n_samples_X, 1)
    Y_norm = (Y**2).sum(dim=1).unsqueeze(0)  # shape: (1, n_samples_Y)
    
    # Compute the squared Euclidean distance matrix
    distances_squared = X_norm + Y_norm - 2.0 * torch.matmul(X, Y.T)
    
    # Clamp to zero to avoid negative distances due to numerical issues
    distances_squared = torch.clamp(distances_squared, min=0.0)
    return distances_squared
# end def


def get_median_heuristic_single(
        distance_module: L2Distance,
        x: torch.Tensor,
        y: torch.Tensor,
        percentile: int) -> torch.Tensor:

    sample_concat = torch.cat([x, y])
    # d2_matrix_torch = distance_module.compute_distance(sample_concat, sample_concat)  # computing L2 distance matrix. for debug.
    d2_torch = euclidean_distances_squared(sample_concat)
    
    # matrix_shape_torch = d2_torch.shape
    # distance_matrix_torch = d2_torch[torch.triu_indices(matrix_shape_torch[0], matrix_shape_torch[0], 1)]
    
    med_sqdist_torch = torch.quantile(d2_torch, q=(percentile / 100))
    bandwidth_torch = torch.sqrt(med_sqdist_torch / 2)
    bandwidth_log = torch.log(bandwidth_torch)

    return bandwidth_log



class QuadraticKernelGaussianKernelCustom(QuadraticKernelGaussianKernel):
    """Kernel Definition class of replacing the length scale related operations.
    The original class defines only two operations {median, mean}"""
    def __init__(self,
                 bandwidth_percentile: int,
                 is_dimension_median_heuristic: bool,
                 bandwidth: ty.Optional[torch.Tensor] = None,
                 ard_weights: ty.Optional[torch.Tensor] = None,
                 ard_weight_shape: ty.Optional[ty.Tuple[int, ...]] = None,
                 target_cuda_device_id: ty.Optional[int] = None
                 ):
        super().__init__(
            bandwidth=bandwidth,
            is_dimension_median_heuristic=is_dimension_median_heuristic,
            ard_weight_shape=ard_weight_shape,
            ard_weights=ard_weights)
        self.bandwidth_percentile = bandwidth_percentile

        # setting the GPU device
        if torch.cuda.is_available():
            if target_cuda_device_id is None:
                target_cuda_device_id = _get_less_busy_cuda_device()
            # end if
            self.torch_device = torch.device(target_cuda_device_id)
            self.distance_module.to(self.torch_device)
        else:
            self.torch_device = torch.device('cpu')
            self.distance_module.to(self.torch_device)
        # end if

    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor,
                           percentile: int) -> torch.Tensor:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'
        Args:
            x: (samples, features)
            y: (samples, features)
            minimum_sample: a minimum value for sampling.
            heuristic_operation: 'median' or 'mean'
        Returns:
            computed median
        """
        x_projected = torch.mul(self.ard_weights, x)  # elementwise product of ARD weight and x
        y_projected = torch.mul(self.ard_weights, y)  # elementwise product of ARD weight and y

        x_projected = x_projected.to(self.torch_device)
        y_projected = y_projected.to(self.torch_device)
        
        with torch.no_grad():
            bandwidth = get_median_heuristic_single(self.distance_module, 
                                                    x_projected, 
                                                    y_projected, 
                                                    percentile)

        return bandwidth.cpu()
    
        # comment out. The following code uses the numpy based implementation. The exec. speed is slower.
        # samp = torch.cat([x_projected, y_projected])
        # np_reps = samp.detach().cpu().numpy()
        # d2 = euclidean_distances(np_reps, squared=True)

        # assert self.heuristic_operation == 'median'

        # comment out: 2025-05-06
        # if self.heuristic_operation == 'median':
        #     med_sqdist = np.median(d2[np.triu_indices_from(d2, k=1)])
        # elif self.heuristic_operation == 'mean':
        #     med_sqdist = np.mean(d2[np.triu_indices_from(d2, k=1)])
        # else:
        #     raise Exception(f'No heuristic_operation == {self.heuristic_operation}.')
        # # end if

        # distance_matrix = d2[np.triu_indices_from(d2, k=1)]
        # med_sqdist = np.percentile(distance_matrix, q=percentile)

        # bandwidth = np.sqrt(med_sqdist / 2)

        # if self.is_force_cutoff:
        #     bandwidth = self.adjust_auto_median_heuristic(x, y, bandwidth)
        # # end if

        # del samp, d2, med_sqdist
        # # end if

        # return torch.tensor([bandwidth])

    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        is_completion_missing: bool = True,
                        is_safe_guard_same_xy: bool = True
                        ) -> ty.Optional[torch.Tensor]:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'
        Args:
            x: (Samples, M)
            y: (Samples, M)
            ard_weight: ARD weight initialized.
        Returns:
            Median heuristic term with M where M is a dimension size.
        """
        with torch.no_grad():
            if self.distance_module.coordinate_size == 1:
                x_projected = torch.mul(self.ard_weights, x)  # elementwise product of ARD weight and x
                y_projected = torch.mul(self.ard_weights, y)  # elementwise product of ARD weight and y
            else:
                # comment: element-wise-product((N, |S|, C), (|S|))
                x_projected = torch.einsum('ijk,j->ijk', x, self.ard_weights)
                y_projected = torch.einsum('ijk,j->ijk', y, self.ard_weights)
                # double check multiplication.
                x_projected[0][0] = x[0][0] * self.ard_weights[0]
                y_projected[0][0] = y[0][0] * self.ard_weights[0]
                x_projected[-1][-1] = x[-1][-1] * self.ard_weights[-1]
                y_projected[-1][-1] = y[-1][-1] * self.ard_weights[-1]
            # end if
        # end with
                        
        # return None if all same.
        if is_safe_guard_same_xy:
            diff_xy = x_projected - y_projected
            # comment: all elements are zero. Hence returning None.
            if torch.count_nonzero(diff_xy) == 0:
                return None
            # end if
        # end if
        
        median_heuristic = torch.zeros((x.shape[1],))
        __shape = x_projected.shape
        n_dimension = x.shape[1]
        
        # ----------------------------------------------
        # arguments of task function
        __task_arguments = []
        for __m in range(0, n_dimension):
            if len(__shape) == 2:
                __x_projected_m = torch.reshape(x_projected[:, __m], shape=(len(x_projected), 1))
                __y_projected_m = torch.reshape(y_projected[:, __m], shape=(len(x_projected), 1))
            else:
                __x_projected_m = x_projected[:, __m]
                __y_projected_m = y_projected[:, __m]
            # end if
            __args_obj = DistributedFunctionArg(
                index_dimension=__m, 
                n_dimension=n_dimension, 
                x_projected_m=__x_projected_m,
                y_projected_m=__y_projected_m,
                heuristic_operation=self.heuristic_operation,
                distance_function=self.distance_module.compute_distance)
            __task_arguments.append(__args_obj)
        # end for
        # ----------------------------------------------
        # execution
        
        seq_length_scale = [compute_length_scale_dataset_dimension_d(
            __args, 
            percentile=self.bandwidth_percentile,
            torch_device=self.torch_device) for __args in __task_arguments]
        # below is the original code. I comment out the dask related becasue I wanna keep the code simple.
        # if self.dask_client is None:
        #     seq_length_scale = [compute_length_scale_dataset_dimension_d(__args) for __args in __task_arguments]
        # else:
        #     dask_client = self.dask_client
        #     assert dask_client is not None, 'dask_client is None.' and isinstance(dask_client, Client)
        #     task_queue = dask_client.map(compute_length_scale_dataset_dimension_d, __task_arguments)
        #     seq_length_scale = dask_client.gather(task_queue)
        # # end if
        # ----------------------------------------------
        for __tuple_length_scale in seq_length_scale:
            __index_dimension = __tuple_length_scale[0]
            __gamma_m = __tuple_length_scale[1]
            median_heuristic[__index_dimension] = __gamma_m
        # end for
        # ----------------------------------------------
        # post process
        res_value = torch.reshape(median_heuristic, self.ard_weights.shape)

        if self.distance_module.coordinate_size == 1:
            assert res_value.shape == x.shape[1:] == y.shape[1:]
        else:
            assert (res_value.shape[0], self.distance_module.coordinate_size) == x.shape[1:] == y.shape[1:]
        # end if

        if self.is_force_cutoff and self.ratio_cutoff == -1:
            ratio_cutoff = self.select_lower_bound_auto(x_projected, y_projected, res_value)
            self.ratio_cutoff = ratio_cutoff
        else:
            ratio_cutoff = self.ratio_cutoff
        # end if
        if self.is_force_cutoff:
            res_value = self.execute_force_cutoff(res_value, ratio_cutoff)
        # end if


        if is_completion_missing:
            # comment: when too few samples. Can be all 0.0. So, I skip the if block when too few samples.
            if len(x) < 50 and len(y) < 50:
                pass
            else:
                assert torch.count_nonzero(res_value) > 0, \
                    f'Kernel length scaling function encountered zero vales for all dimensions. Hint: changing kernel configuration to heuristic_operation="mean". Current heuristic_operation={self.heuristic_operation}'
                value_replace = torch.min(res_value[res_value != 0])
                res_value[res_value == 0] = value_replace
            # end if
        # endif
        return res_value.detach()
    
    # ------------------------------------------------
    # computing kernel matrix.

    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: ty.Optional[torch.Tensor] = None
                                      ) -> KernelMatrixObject:
        """
        Args:
            bandwidth: a bandwidth (length-scale). A scalar shape tensor.
            The value must be after `torch.log`. Otherwise, the `gamma` in this function will be `inf`.
        """
        # comment: I do not maintain this method anymore. Multi-dim length scale is the default.
        # Basically, I do not need this method anymore.
        x = torch.mul(self.ard_weights, x)
        y = torch.mul(self.ard_weights, y)

        if bandwidth is None:
            bandwidth = self.bandwidth
            assert bandwidth is not None
        # end if
        sigma = torch.exp(bandwidth)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        # torch.t() is transpose function. torch.dot() is only for vectors. For 2nd tensors, "mm".
        # xx = torch.mm(x, torch.t(x))
        # xy = torch.mm(x, torch.t(y))
        # yy = torch.mm(y, torch.t(y))

        # x_sqnorms = torch.diagonal(xx, offset=0)
        # y_sqnorms = torch.diagonal(yy, offset=0)

        d_container = self.distance_module.compute_distance(x, y, False)

        k_xy = torch.exp(-1 * gamma * d_container.d_xy)
        k_xx = torch.exp(-1 * gamma * d_container.d_xx)
        k_yy = torch.exp(-1 * gamma * d_container.d_yy)

        k_container = QuadraticKernelMatrixContainer(k_xx, k_yy, k_xy)
        return KernelMatrixObject(kernel_computation_type=self.kernel_computation_type, x_size=len(x), y_size=len(y),
                                  kernel_matrix_container=k_container)    


# -------------------------------------------------------------------------------


def load_mmd_estimator(path_saved_dir: Path,
                       file_name_mmd_estimator_config: str = file_name_mmd_estimator_config,
                       file_name_mmd_estimator: str = file_name_mmd_estimator
                       ) -> ty.Optional[ty.Tuple[QuadraticMmdEstimator, ty.Dict]]:
    path_mmd_model = path_saved_dir / file_name_mmd_estimator

    if not path_mmd_model.exists():
        return None
    # end if

    mmd_estimator = torch.load(path_mmd_model)

    path_config = path_saved_dir / file_name_mmd_estimator_config
    
    with path_config.open('r') as f:
        config_dict = json.loads(f.read())
    # end with

    return mmd_estimator, config_dict



class MmdEstimatorInitialiserVer1(object):
    def __init__(
                  self,
                  tensor_preprocessor: TensorPreprocessorVer1,
                  vector_extractor: BaseVectorExtractorVer2,
                  mode_target_embedding_layer: str,
                  kernel_type: str = "gaussian",
                  kernel_length_scale_percentile: int = 50,  # percentile values of selecting. 50th is the median.
                  kernel_length_scale_median_option: str = "dimensionwise",
                  path_cache_dir: ty.Optional[Path] = None,
                  file_name_mmd_estimator: str = file_name_mmd_estimator,
                  file_name_mmd_estimator_config: str = file_name_mmd_estimator_config,
                  option_translation_max_a: float = 0.0,
                  option_translation_max_b: int = 200
                  ):
            assert kernel_type in KERNEL_TYPE, f'argument must be in {KERNEL_TYPE}. Given -> {kernel_type}'
            assert kernel_length_scale_median_option in KERNEL_LENGTH_SCALE_MEDIAN_OPTION, f'argument must be in {KERNEL_LENGTH_SCALE_MEDIAN_OPTION}. Give -> {kernel_length_scale_median_option}'

            self.vector_extractor = vector_extractor
            self.tensor_preprocessor = tensor_preprocessor

            self.mode_target_embedding_layer = mode_target_embedding_layer

            self.kernel_type = kernel_type
            self.kernel_length_scale_percentile = kernel_length_scale_percentile
            self.kernel_length_scale_median_option = kernel_length_scale_median_option

            if path_cache_dir is None:
                self.path_cache_dir = Path(tempfile.mkdtemp())
                self.path_cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.path_cache_dir = path_cache_dir
            # end if
            self.file_name_mmd_estimator = file_name_mmd_estimator
            self.file_name_mmd_estimator_config = file_name_mmd_estimator_config

            self.option_translation_max_a = option_translation_max_a
            self.option_translation_max_b = option_translation_max_b

            self.mmd_estimator: QuadraticMmdEstimator
    
    # ------------------------------------------------------------------
    # semi private

    def _calibrate_kernel_function(
            self,
            seq_calibration_text: ty.List[EvaluationTargetTranslationPair],
            mode_target_embedding_layer: str,
            kernel_length_scale_median_option: str,
            target_layers_extraction: ty.Optional[ty.List[str]] = None
            ) -> QuadraticKernelGaussianKernelCustom:
        """Calculating the median values over the set of vectors and set it as the length scale of the gaussian kernel.
        Args:
            seq_calibration_text: a set of text to be used for the {median/mean} heuristic.
        """
        # Extracting hidden vectors.
        # Note: I have to run the translation and extract the hidden vectors.
        # 2025-04-29: Hard to get the hidden vectors with the teacher-forcing.
        seq_embedding_stack = []
        for _record_calibration in seq_calibration_text:
            _obj_translation = self.vector_extractor.translation_handler.translate_beam_search(
                input_text=_record_calibration,
                temperature=1.0,
                max_len_a=self.option_translation_max_a,
                max_len_b=self.option_translation_max_b,
                target_layers_extraction=target_layers_extraction)
            assert isinstance(_obj_translation.dict_layer_embeddings, dict)
            assert mode_target_embedding_layer in _obj_translation.dict_layer_embeddings

            seq_embedding_stack.append(_obj_translation.dict_layer_embeddings[mode_target_embedding_layer])
        # end for

        # pre-processing of tensors.
        # __a_emb -> a_emb, a fixed shape
        calibration_emb_fixed = self.tensor_preprocessor.preprocess_tensors(
            seq_tensor=seq_embedding_stack,
            is_calibration_mode=True)

        if 'single' in kernel_length_scale_median_option:
            _is_dimension_wise = False
        elif 'dimensionwise' in kernel_length_scale_median_option:
            _is_dimension_wise = True
        else:
            raise ValueError(f"Invalid median options: {kernel_length_scale_median_option}")
        # end if

        kernel_func_obj = QuadraticKernelGaussianKernelCustom(
            bandwidth_percentile=self.kernel_length_scale_percentile,
            is_dimension_median_heuristic=_is_dimension_wise,
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
                y=calibration_emb_fixed,
                percentile=self.kernel_length_scale_percentile)
        # end if
        module_logger.debug("Done computing the length scale...")    
        assert tensor_length_scale is not None

        # set the computed length-scale to the kernel object.
        kernel_func_obj.bandwidth = torch.nn.Parameter(tensor_length_scale, requires_grad=False)
        kernel_func_obj.ard_weights = torch.nn.Parameter(torch.ones(calibration_emb_fixed.shape[1]), requires_grad=False)

        return kernel_func_obj

    # ------------------------------------------------------------------
    # public 

    def init_gaussian_kernel_mmd_estimator(self, 
                                         seq_calibration_text: ty.List[EvaluationTargetTranslationPair],
                                         target_layers_extraction: ty.Optional[ty.List[str]] = None,
                                         ) -> QuadraticMmdEstimator:
        """Initialising the Gaussian Kernel using the given text."""
        if target_layers_extraction is None:
            __, target_layers_extraction = self.vector_extractor.translation_handler.get_all_possible_layers()
        # end if

        kernel_func_obj = self._calibrate_kernel_function(
            seq_calibration_text=seq_calibration_text,
            mode_target_embedding_layer=self.mode_target_embedding_layer,
            kernel_length_scale_median_option=self.kernel_length_scale_median_option,
            target_layers_extraction=target_layers_extraction)
        mmd_estimator = QuadraticMmdEstimator(kernel_func_obj, variance_term='sutherland_2017')

        self.mmd_estimator = mmd_estimator

        return mmd_estimator

    def init_dot_product_kernel_mmd_estimator(self, 
                                              seq_calibration_text: ty.List[EvaluationTargetTranslationPair],
                                              target_layers_extraction: ty.Optional[ty.List[str]] = None,
                                              ) -> QuadraticMmdEstimator:
        """Initialising the Gaussian Kernel using the given text."""
        if target_layers_extraction is None:
            __, target_layers_extraction = self.vector_extractor.translation_handler.get_all_possible_layers()
        # end if

        if self.tensor_preprocessor.mode_vector_preprocess == "concat":
            # when "cocat" & "max_calibration", I have to know the max. token size. Note that dot-product kernel does not use length-scale. I just need the number of tokens.
            if self.tensor_preprocessor.mode_max_token_length_vector_concat == "max_calibration":
                # Extracting hidden vectors.
                # Note: I have to run the translation and extract the hidden vectors.
                # 2025-04-29: Hard to get the hidden vectors with the teacher-forcing.
                seq_embedding_stack = []
                for _record_calibration in seq_calibration_text:
                    _obj_translation = self.vector_extractor.translation_handler.translate_beam_search(
                        input_text=_record_calibration,
                        temperature=1.0,
                        max_len_a=self.option_translation_max_a,
                        max_len_b=self.option_translation_max_b,
                        target_layers_extraction=target_layers_extraction)
                    assert isinstance(_obj_translation.dict_layer_embeddings, dict)
                    assert self.mode_target_embedding_layer in _obj_translation.dict_layer_embeddings

                    seq_embedding_stack.append(_obj_translation.dict_layer_embeddings[self.mode_target_embedding_layer])
                # end for

                # pre-processing of tensors. Setting the max-token size in the method.
                calibration_emb_fixed = self.tensor_preprocessor.preprocess_tensors(
                    seq_tensor=seq_embedding_stack,
                    is_calibration_mode=True)
            elif self.tensor_preprocessor.mode_max_token_length_vector_concat == "fixed":
                # the fixed token size is already set. No need to do.
                pass
            # end if
        # end if

        _kernel_obj = DotProductKernel(ard_weight_shape=(1,))  # `ard_weight_shape` is a dummy variable. Not used in the computation at all.
        mmd_estimator = QuadraticMmdEstimator(_kernel_obj, variance_term='sutherland_2017')

        self.mmd_estimator = mmd_estimator
        return mmd_estimator

    def get_mmd_estimator(self,
                          seq_calibration_text: ty.List[EvaluationTargetTranslationPair],
                          target_layers_extraction: ty.Optional[ty.List[str]] = None
                          ) -> QuadraticMmdEstimator:
        """an interface API to obtain an MMD estimator object"""
        if self.kernel_type == 'dot':
            return self.init_dot_product_kernel_mmd_estimator(
                seq_calibration_text=seq_calibration_text,
                target_layers_extraction=target_layers_extraction)
        elif self.kernel_type == 'gaussian':
            return self.init_gaussian_kernel_mmd_estimator(
                seq_calibration_text=seq_calibration_text,
                target_layers_extraction=target_layers_extraction)
        else:
            raise ValueError(f'No case for {self.kernel_type}')

    def save_mmd_estimator(self) -> Path:
        assert self.mmd_estimator is not None, f'self.mmd_estimator is not set yet.'
        _path_model_file = self.path_cache_dir / self.file_name_mmd_estimator
        _path_configs = self.path_cache_dir / self.file_name_mmd_estimator_config

        torch.save(copy.deepcopy(self.mmd_estimator).cpu(), _path_model_file)

        config_dict = dict(
            kernel_type=self.kernel_type,
            kernel_length_scale_setting=self.kernel_length_scale_percentile,
            kernel_length_scale_median_option=self.kernel_length_scale_median_option,
            mode_vector_preprocess=self.tensor_preprocessor.mode_vector_preprocess,
            mode_target_embedding_layer=self.mode_target_embedding_layer,
            mode_max_token_length_vector_concat=self.tensor_preprocessor.mode_max_token_length_vector_concat,
            option_max_token_length=self.tensor_preprocessor.option_max_token_length
        )
        
        with _path_configs.open('w') as f:
            f.write(json.dumps(config_dict))
        # end with

        return _path_model_file
            