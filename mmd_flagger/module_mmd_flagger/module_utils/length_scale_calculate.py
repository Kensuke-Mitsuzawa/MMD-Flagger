import typing as ty
import logging
import tempfile

from pathlib import Path

import torch


from ..module_kernels.gaussian_kernel import QuadraticKernelGaussianKernelCustom

from .tensor_preprocessor import BaseTensorPreprocessor


module_logger = logging.getLogger(__name__)



KERNEL_LENGTH_SCALE_MEDIAN_OPTION = ty.Literal['dimensionwise', 'single']


class KernelLengthScaleCalculator(object):
    """Constructor of MMD-Estimator based on embedding vectors. No need to going through a generation model."""
    def __init__(self,
                 tensor_preprocessor: BaseTensorPreprocessor,
                 kernel_length_scale_percentile: int = 50,  # percentile values of selecting. 50th is the median.
                 path_cache_dir: ty.Optional[Path] = None):

            self.kernel_length_scale_percentile = kernel_length_scale_percentile

            if path_cache_dir is None:
                self.path_cache_dir = Path(tempfile.mkdtemp())
                self.path_cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.path_cache_dir = path_cache_dir
            # end if
            self.tensor_preprocessor = tensor_preprocessor
    
    # ------------------------------------------------------------------
    # public methods

    def get_length_scale(
            self,
            seq_embedding_stack: ty.List[torch.Tensor],
            kernel_length_scale_median_option: KERNEL_LENGTH_SCALE_MEDIAN_OPTION,
            ) -> torch.Tensor:
        """Calculating the median values over the set of vectors and set it as the length scale of the gaussian kernel.
        Args:
            seq_calibration_text: a set of text to be used for the {median/mean} heuristic.
        """
        calibration_emb_fixed = torch.nan_to_num(self.tensor_preprocessor.preprocess_tensors(
            seq_tensor=seq_embedding_stack,
            is_calibration_mode=True), nan=0.0)

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

        # Sanitize and clamp length scale to avoid extremely small or NaN values
        tensor_length_scale = torch.nan_to_num(tensor_length_scale, nan=1.0)
        tensor_length_scale = torch.clamp(tensor_length_scale, min=1e-4)

        # set the computed length-scale to the kernel object.
        # kernel_func_obj.bandwidth = torch.nn.Parameter(tensor_length_scale, requires_grad=False)
        # kernel_func_obj.ard_weights = torch.nn.Parameter(torch.ones(calibration_emb_fixed.shape[1]), requires_grad=False)

        return tensor_length_scale
