import typing as ty

import torch
import numpy as np
import collections


from mmd_tst_variable_detector.distance_module.base import BaseDistanceModule, DistanceContainer

from .module_preprocessing import StringFeatureDictionary

class JaccardDistanceModule(BaseDistanceModule):
    def __init__(self, 
                 feature_dictionary: StringFeatureDictionary,
                 data_point_shape: ty.Tuple[int, ...] = (1,)) -> None:
        super().__init__(data_point_shape)
        self.coordinate_size: int
        self.feature_dictionary = feature_dictionary

        self.register_buffer(
            'device_tracker', 
            torch.empty(0)
        )

    @classmethod
    def from_documents(cls, documents: ty.List[str], is_proprocessed: bool = False):
        """
        documents: a sequence of tokens.
        """
        vocab_dict = StringFeatureDictionary.from_documents(documents, is_proprocessed=is_proprocessed)

        return JaccardDistanceModule(feature_dictionary=vocab_dict, data_point_shape=(1,))

    def get_current_device(self) -> torch.device:
        """
        Returns the torch.device object by checking a registered buffer.
        This is reliable because buffers are always moved by model.to().
        """
        return self.device_tracker.device

    def get_hyperparameters(self) -> ty.Dict[str, ty.Any]:
        return super().get_hyperparameters()

    def _get_binary_matrix(self, samples_x: ty.List[ty.List[str]], samples_y: ty.List[ty.List[str]]) -> ty.Tuple[np.ndarray, ty.Dict[int, str]]:
        """obtaining a binary matrix X.
        An element of X represents an existence of a token.
        I concatenate the two sets into once matrix for the computational efficiency.
        
        Args:
            samples_x: a list (sentences) of lists (tokens).
                [[token, token, ...], [token, token, ...]]
            samples_y: the same for samples_x.

        Returns:
            x_array: np.ndarray (n-sample-x + n-sample-y, Dim-vocab)
        """
        dict_index2sample_label = {}
        dict_index2sample_label.update({_i: 'x' for _i in range(len(samples_x))})
        dict_index2sample_label.update({len(samples_x) + _i: 'y' for _i in range(len(samples_y))})        

        x_array = np.zeros(shape=(len(dict_index2sample_label), self.feature_dictionary.get_size()))
        for _doc_i, _doc in enumerate(samples_x + samples_y):
            _counter_obj = collections.Counter(_doc)
            for _vocab, _freq in _counter_obj.items():
                if _vocab in self.feature_dictionary.vocab_dict:
                    _feat_id = self.feature_dictionary.vocab_dict[_vocab]
                else:
                    _feat_id = self.feature_dictionary.vocab_dict[self.feature_dictionary.unk_default]
                # end if
                x_array[_doc_i][_feat_id] = _freq
            # end for
        # end for

        return x_array, dict_index2sample_label

    @staticmethod
    def _compute_jaccard_distance(x_array: torch.Tensor) -> torch.Tensor:
        # ----- Compute the Intersection Matrix (I) using Matrix Multiplication
        # I is an n x n matrix of counts of shared '1's.
        i_array = x_array @ x_array.T

        # ------ Compute the Cardinality Vector (C) (sum of '1's in each row)
        # C is an n x 1 vector.
        c_array = x_array.sum(dim=1)

        # Compute the Union Matrix (U) using Broadcasting
        # c_array.unsqueeze(1) is (n x 1), c_array.unsqueeze(0) is (1 x n). 
        # Adding them creates an (n x n) matrix where U_ij = C_i + C_j.
        # Then subtract I.
        u_array = c_array.unsqueeze(1) + c_array.unsqueeze(0) - i_array

        # Compute Jaccard Similarity Matrix (J)
        # Use torch.div and torch.nan_to_num to handle the case where U_ij = 0 (0/0 -> 1.0 similarity)
        J_similarity = torch.div(i_array, u_array)
        J_similarity = torch.nan_to_num(J_similarity, nan=1.0) # Sets 0/0 case (two empty sets) to 1.0

        # Compute Jaccard Distance Matrix (D_J)
        D_jaccard = 1.0 - J_similarity        

        return D_jaccard

    def extract_sub_matrices(self,
                             d_jaccard: torch.Tensor,
                             dict_index2sample_label: ty.Dict[int, str]
                             ) -> DistanceContainer:
        indices_x = torch.tensor([_k for _k, _v in dict_index2sample_label.items() if _v == 'x'])
        indices_y = torch.tensor([_k for _k, _v in dict_index2sample_label.items() if _v == 'y'])

        # D_XX: A rows, A columns
        d_xx = d_jaccard[indices_x.unsqueeze(1), indices_x]
        # D_YY: B rows, B columns
        d_yy = d_jaccard[indices_y.unsqueeze(1), indices_y]

        # D_XY: A rows, B columns
        d_xy = d_jaccard[indices_x.unsqueeze(1), indices_y]
        # D_yx: B rows, A columns
        d_yx = d_jaccard[indices_y.unsqueeze(1), indices_x]
        assert torch.equal(d_yx.T, d_xy)


        res = DistanceContainer(
            d_xx=d_xx.to(self.get_current_device()),
            d_yy=d_yy.to(self.get_current_device()),
            d_xy=d_xy.to(self.get_current_device()))

        return res

    @staticmethod
    def check_data_type(x: ty.List[ty.List[str]]):
        assert len(x) > 0, f'The input samples has zero length. {x}'

        def check_one_doc(one_doc: ty.List[str]) -> bool:
            assert isinstance(one_doc, list), f'One document must be a list of tokens. Input->{one_doc}'
            assert all([isinstance(_t, str) for _t in one_doc]), f'One document must be consisted of tokens (str). Input -> {one_doc}.'

            return True
        # end def

        assert all([check_one_doc(one_doc) for one_doc in x])

    def compute_distance(self, 
                         x: ty.List[ty.List[str]], 
                         y: ty.List[ty.List[str]], 
                         is_compute_length_scale: bool = False  # not used.
                         ) -> DistanceContainer:
        """Compute distance between x and y."""
        self.check_data_type(x)
        self.check_data_type(y)

        x_array, dict_index2sample_label = self._get_binary_matrix(x, y)
        d_matrix_jaccard = self._compute_jaccard_distance(torch.from_numpy(x_array))

        d_container = self.extract_sub_matrices(d_matrix_jaccard, dict_index2sample_label)

        return d_container