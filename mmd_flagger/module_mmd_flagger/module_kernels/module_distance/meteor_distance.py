import typing as ty

import torch
import numpy as np

from nltk.translate import meteor_score


from mmd_tst_variable_detector.distance_module.base import BaseDistanceModule, DistanceContainer



class MeteorDistanceModule(BaseDistanceModule):
    def __init__(self,
                 data_point_shape: ty.Tuple[int, ...] = (1,)) -> None:
        super().__init__(data_point_shape)
        self.coordinate_size: int

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

    @staticmethod
    def compute_triangular_similarity_matrix(docs: ty.List[ty.List[str]]) -> np.ndarray:
        """
        Computes a full symmetric similarity matrix (S_XX or S_YY) 
        by only calculating the lower triangular elements (S[i, j] where i > j)
        and then reflecting the results.
        """
        N = len(docs)
        S_sym = np.zeros((N, N), dtype=np.float64)

        # Calculate the diagonal (Self-Similarity)
        # The similarity of a sequence to itself is always 1.0 (perfect match).
        np.fill_diagonal(S_sym, 1.0) 

        # Calculate the lower triangle elements (i > j)
        # This avoids redundant calculation since S[i, j] = S[j, i]
        for i in range(N):
            for j in range(i):
                # Calculate METEOR score for the unique pair (docs[i], docs[j])
                # similarity = get_meteor_similarity(docs[i], docs[j])
                similarity = meteor_score.meteor_score([docs[i]], docs[j])
                
                # Place the score in both the lower (S[i, j]) and upper (S[j, i]) triangle
                # This simultaneously computes the full symmetric matrix
                S_sym[i, j] = similarity
                S_sym[j, i] = similarity
            # end for
        # end for

        return S_sym

    def _compute_meteor(self, set_x_tokens: ty.List[ty.List[str]], set_y_tokens: ty.List[ty.List[str]]) -> DistanceContainer:
        matrix_sim_xx_half: np.ndarray = self.compute_triangular_similarity_matrix(set_x_tokens)
        matrix_sim_yy_half: np.ndarray = self.compute_triangular_similarity_matrix(set_y_tokens)
        matrix_sim_xy_full = torch.zeros(len(set_x_tokens), len(set_y_tokens))

        # ---- xy -----
        for _i, _x_sample  in enumerate(set_x_tokens):
            for _j, _y_sample in enumerate(set_y_tokens):
                _score = meteor_score.meteor_score([_x_sample], _y_sample)
                matrix_sim_xy_full[_i, _j] = _score
            # end for
        # end for
        # ------
        d_xy = 1.0 - matrix_sim_xy_full
        d_xx = 1.0 - matrix_sim_xx_half
        d_yy = 1.0 - matrix_sim_yy_half

        current_device = self.get_current_device()

        res = DistanceContainer(
            d_xx=torch.from_numpy(d_xx).to(current_device),
            d_yy=torch.from_numpy(d_yy).to(current_device),
            d_xy=d_xy.to(current_device)
        )

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
        """Compute distance between x and y.
        
        Args:
            x: a list of tokens. One (1st level) list represents a sentence. The 2nd level list represents a sequence of tokens.
            y: the same as x.
        """
        self.check_data_type(x)
        self.check_data_type(y)

        d_matrix_jaccard = self._compute_meteor(x, y)
        
        return d_matrix_jaccard