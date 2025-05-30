import torch
import torch.nn.functional as F
import numpy as np

import typing as ty

import tqdm
import logging

logger = logging.getLogger(__name__)


def pad_arrays(arrays: ty.List[np.ndarray], max_length=None) -> np.ndarray:
    if max_length is None:
        max_length = max(len(arr) for arr in arrays)
    
    padded_arrays = np.array([np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in arrays])
    return padded_arrays    


def pad_tensor(tensor, target_size, pad_value=0):
    """
    Pads a tensor to the target size with the specified pad value.

    Args:
        tensor (torch.Tensor): The tensor to pad.
        target_size (int): The target size for the second dimension.
        pad_value (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: The padded tensor.
    """
    current_size = tensor.size(1)
    if current_size < target_size:
        padding = (0, target_size - current_size)
        tensor = F.pad(tensor, padding, "constant", pad_value)
    return tensor


def pad_tensor_1d(tensor: torch.Tensor, target_size: int, pad_value: int = 0):
    """
    Pads a tensor to the target size with the specified pad value.

    Args:
        tensor (torch.Tensor): The tensor to pad.
        target_size (int): The target size for the second dimension.
        pad_value (int, optional): The value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: The padded tensor.
    """
    current_size = tensor.size(0)
    if current_size < target_size:
        padding = target_size - current_size
        tensor = F.pad(tensor, (0, padding), "constant", pad_value)
    return tensor


def shape_dataset_tensor(
        seq_document_vectors: ty.List[ty.List[torch.Tensor]],
        max_token_size: ty.Optional[int]
        ) -> torch.Tensor:
    """Shape the dataset tensor to have the same length of tokens.
    
    Args:
        seq_document_vectors (List[List[torch.Tensor]]): The list of document vectors.
            A document consists of a list of token vectors.
        max_token_size (int): The maximum token size.
        max_document_vector_size (int): The maximum document vector size.
    """
    if max_token_size is not None:
        logger.info(f"Max token size: {max_token_size}")
        seq_document_vectors = [seq[:max_token_size] for seq in seq_document_vectors]
    # end if

    # I want to make the 2D tensor flatten.
    seq_document_flatten = [torch.flatten(seq) for seq in seq_document_vectors]

    # get the max length of the document vector
    max_vector_size = max([seq.shape[0] for seq in seq_document_flatten])

    # I want to pad the tensor to have the same length.
    array_document_pad = pad_arrays(seq_document_flatten, max_length=max_vector_size)

    return torch.from_numpy(array_document_pad).float()



