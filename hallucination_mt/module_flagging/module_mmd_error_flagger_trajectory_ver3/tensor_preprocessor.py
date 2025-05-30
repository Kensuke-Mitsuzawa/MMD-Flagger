import typing as ty

import torch


MODE_VECTOR_PREPROCESS = ('avg', 'concat')
MODE_MAX_TOKEN_LENGTH_VECTOR_CONCAT = ('max_calibration', 'fixed', None)


class TensorPreprocessorVer1(object):
    def __init__(
            self,
            mode_vector_preprocess: str = "avg",
            mode_max_token_length_vector_concat: ty.Optional[str] = "max_calibration",
            option_max_token_length: int = -1
            ):
        assert mode_max_token_length_vector_concat in MODE_MAX_TOKEN_LENGTH_VECTOR_CONCAT
        if mode_max_token_length_vector_concat == 'fixed':
            assert option_max_token_length != -1
        # end if

        assert mode_vector_preprocess in MODE_VECTOR_PREPROCESS


        self.mode_vector_preprocess = mode_vector_preprocess
        self.mode_max_token_length_vector_concat = mode_max_token_length_vector_concat
        self.option_max_token_length = option_max_token_length

    def preprocess_tensors(self,
                            seq_tensor: ty.List[torch.Tensor],
                            is_calibration_mode: bool = False
                            ) -> torch.Tensor:
        """
        Args:
            seq_tensor: The list of tensors. Each tensor is (T: number of tokens, embed_dim).

        Returns:
            torch.Tensor: The tensor is (N: the num. of documents, D_emb: embedding-size).
            mode_vector_preprocess: mode. See MODE_VECTOR_PREPROCESS for the available options.
            mode_max_token_length_vector_concat: mode. See MODE_MAX_TOKEN_LENGTH_VECTOR_CONCAT for the available options.
        """
        def mode_avg(document_tensor: torch.Tensor) -> torch.Tensor:
            """I want to compute the average of the tensor.
            
            Args:
                document_tensor: The tensor is (T: number of tokens, embed_dim).
            """
            try:
                assert len(document_tensor.shape) == 2, f"Expected 2D tensor, got {len(document_tensor.shape)}"
            except AssertionError:
                # TODO delete
                print()
            return torch.mean(document_tensor, dim=0)
        # end mode_avg

        def truncate_and_pad_vector(document_tensor: torch.Tensor, _max_token_length: int) -> torch.Tensor:
            """
            Args:
                document_tensor (torch.Tensor): The tensor is (T: token-size, D_emb: embedding-size).
            """            
            assert len(document_tensor.shape) == 2
            if _max_token_length < document_tensor.shape[0]:
                return document_tensor[:_max_token_length, :]
            else:
                # padding of the vector of tokens to the speficied size.
                _diff_size: int = _max_token_length - document_tensor.shape[0]
                _dim_size: int = document_tensor.shape[1]
                _tensor_pad = torch.zeros(size=(_diff_size, _dim_size))
                tensor_pad = torch.cat([document_tensor, _tensor_pad])
                assert tensor_pad.shape == (_max_token_length, _dim_size)
                return tensor_pad
            # end if
        # end def

        if self.mode_vector_preprocess == "avg":
            return torch.stack([mode_avg(_t.cpu()) for _t in seq_tensor], dim=0)  # moving to cpu, cause the tensor may from various cuda devices.
        elif self.mode_vector_preprocess == "concat":
            if isinstance(self.mode_max_token_length_vector_concat, int):  # fixed token size.
                _seq_token_vector_truncate = torch.stack([torch.flatten(truncate_and_pad_vector(_v, self.option_max_token_length)).cpu() for _v in seq_tensor])  # moving to cpu, cause the tensor may from various cuda devices.
                
                assert len(_seq_token_vector_truncate.shape) == 2
                return _seq_token_vector_truncate
            elif isinstance(self.mode_max_token_length_vector_concat, str):
                if self.mode_max_token_length_vector_concat == 'max_calibration':
                    if is_calibration_mode:
                        """when `is_calibration_mode`, I calculate the longest token size."""
                        _max_token_length = max([_s.shape[0] for _s in seq_tensor])
                        _seq_token_vector_pad = torch.stack([torch.flatten(truncate_and_pad_vector(_v, _max_token_length)) for _v in seq_tensor])

                        self.option_max_token_length = _max_token_length

                        assert len(_seq_token_vector_pad.shape) == 2
                        return _seq_token_vector_pad
                    else:
                        """when `is_calibration_mode=False`, I refer to the number of self.option_max_token_length"""
                        assert self.option_max_token_length is not None
                        _seq_token_vector_pad = torch.stack([torch.flatten(truncate_and_pad_vector(_v, self.option_max_token_length)) for _v in seq_tensor])
                        assert len(_seq_token_vector_pad.shape) == 2
                        return _seq_token_vector_pad
                    # end if
                elif self.mode_max_token_length_vector_concat == 'fixed':
                    _seq_token_vector_pad = torch.stack([torch.flatten(truncate_and_pad_vector(_v, self.option_max_token_length)) for _v in seq_tensor])
                    
                    return _seq_token_vector_pad
                else:
                    raise ValueError()
                # end if
            else:
                raise ValueError()
            # end if
        else:
            raise ValueError(f"Invalid mode preprocess: {self.mode_vector_preprocess}")
