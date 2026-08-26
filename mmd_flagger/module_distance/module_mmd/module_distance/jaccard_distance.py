import typing as ty

import torch
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from mmd_tst_variable_detector.distance_module.base import BaseDistanceModule, DistanceContainer

from .text_preprocessing import nltk_preprocess_text


class JaccardDistanceModule(BaseDistanceModule):
    def __init__(self, data_point_shape: ty.Tuple[int, ...] = (1,)) -> None:
        super().__init__(data_point_shape)
        self.coordinate_size: int
        self.vectorizer: CountVectorizer

        self.register_buffer(
            'device_tracker', 
            torch.empty(0)
        )

    def get_current_device(self) -> torch.device:
        """
        Returns the torch.device object by checking a registered buffer.
        This is reliable because buffers are always moved by model.to().
        """
        return self.device_tracker.device

    def get_hyperparameters(self) -> ty.Dict[str, ty.Any]:
        return super().get_hyperparameters()
    
    def get_vectorizer(self, samples_all: ty.List[ty.List[str]]) -> None:
        def re_join_tokens(one_token_sequence: ty.List[str]) -> str:
            return ' '.join(one_token_sequence)
        # end def
        all_samples_text = [re_join_tokens(_one_sequence) for _one_sequence in samples_all]

        vectorizer = CountVectorizer(binary=False)
        X_sparse = vectorizer.fit_transform(all_samples_text)
        self.vectorizer = vectorizer


    def _get_binary_matrix(self, samples_x: ty.List[ty.List[str]], samples_y: ty.List[ty.List[str]]) -> ty.Tuple[np.ndarray, ty.Dict[int, str], ty.List[str]]:
        """obtaining a binary matrix X.
        An element of X represents an existence of a token.
        I concatenate the two sets into once matrix for the computational efficiency.
        """
        def re_join_tokens(one_token_sequence: ty.List[str]) -> str:
            return ' '.join(one_token_sequence)
        # end def

        assert isinstance(self.vectorizer, CountVectorizer), f"vectorizer is not ready to use. Call first `get_vectorizer()`"

        dict_index2sample_label = {}
        dict_index2sample_label.update({_i: 'x' for _i in range(len(samples_x))})
        dict_index2sample_label.update({len(samples_x) + _i: 'y' for _i in range(len(samples_y))})        

        all_samples_text = [re_join_tokens(_one_sequence) for _one_sequence in samples_x + samples_y]

        X_sparse = self.vectorizer.fit_transform(all_samples_text)

        # Convert to Dense NumPy Matrix (X)
        x_array = X_sparse.toarray()

        vocabulary = list(self.vectorizer.get_feature_names_out())

        return x_array, dict_index2sample_label, vocabulary

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

    def compute_distance(self, 
                         x: ty.List[ty.List[str]], 
                         y: ty.List[ty.List[str]], 
                         is_compute_length_scale: bool = False  # not used.
                         ) -> DistanceContainer:
        """Compute distance between x and y."""
        x_array, dict_index2sample_label, vocabulary = self._get_binary_matrix(x, y)
        d_matrix_jaccard = self._compute_jaccard_distance(torch.from_numpy(x_array))

        d_container = self.extract_sub_matrices(d_matrix_jaccard, dict_index2sample_label)

        return d_container