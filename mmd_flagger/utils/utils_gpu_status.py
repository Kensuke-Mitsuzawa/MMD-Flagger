import typing as ty
import torch
import GPUtil
import logging
import os

module_logger = logging.getLogger(__name__)

# Cache for the usability check to avoid redundant probes
_IS_CUDA_USABLE_CACHE: ty.Optional[bool] = None

def get_less_busy_cuda_device() -> ty.Optional[int]:
    if not is_cuda_usable():
        return None
    else:
        try:
            gpu_device_info = GPUtil.getGPUs()
            if not gpu_device_info:
                return 0 # Fallback to device 0 if GPUtil fails but CUDA is usable
            seq_tuple_gpu_memory_utils = [(gpu_obj.id, gpu_obj.memoryUtil) for gpu_obj in gpu_device_info]
            gpu_id_less_busy = sorted(seq_tuple_gpu_memory_utils, key=lambda x: x[1])[0]
            return gpu_id_less_busy[0]
        except Exception as e:
            module_logger.warning(f"Error querying GPU status with GPUtil: {e}. Defaulting to device 0.")
            return 0
# end def

def is_cuda_usable() -> bool:
    """
    Robust check for CUDA usability. 
    Returns False if:
    - CUDA is not available.
    - VRAM is <= 2.1GB (Resource safety).
    - Architecture is incompatible (detects 'no kernel image' error).
    - Environment variable FORCE_CPU is set.
    """
    global _IS_CUDA_USABLE_CACHE
    if _IS_CUDA_USABLE_CACHE is not None:
        return _IS_CUDA_USABLE_CACHE

    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        module_logger.info("CUDA disabled via FORCE_CPU environment variable.")
        _IS_CUDA_USABLE_CACHE = False
        return False

    if not torch.cuda.is_available():
        _IS_CUDA_USABLE_CACHE = False
        return False

    try:
        # 1. Check VRAM size (Resource Safety)
        # Using 2.1GB as threshold as requested by the user for their 2GB machine
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb <= 2.1:
            module_logger.warning(f"Low VRAM detected ({vram_gb:.2f}GB). Forcing CPU mode for stability.")
            _IS_CUDA_USABLE_CACHE = False
            return False

        # 2. Architecture Probe (detects 'no kernel image' or other driver mismatches)
        # We perform a tiny operation to trigger a kernel launch.
        # This will raise a RuntimeError or AcceleratorError if the GPU is incompatible (e.g. sm_61 on sm_70+ build).
        probe_tensor = torch.zeros(1).cuda()
        del probe_tensor
        
        _IS_CUDA_USABLE_CACHE = True
        return True
        
    except (RuntimeError, Exception) as e:
        # Catching specific 'no kernel image' or other CUDA initialization errors
        error_msg = str(e)
        if "no kernel image" in error_msg or "binary" in error_msg:
            module_logger.warning(f"Incompatible GPU architecture detected: {error_msg}. Falling back to CPU.")
        else:
            module_logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU for stability.")
        
        _IS_CUDA_USABLE_CACHE = False
        return False
# end def