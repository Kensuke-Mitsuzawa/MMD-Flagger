import typing as ty
import enum
import itertools

import numpy as np
import torch
import networkx as nx

from langchain_core.outputs import Generation


from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator, MmdValues

from ....utils import DecodingConfig
from ....module_langchain.module_langchain_hidden_layers import models

from .module_distance import JaccardDistanceModule, MeteorDistanceModule, nltk_preprocess_text
from .string_based_gaussian_kernel import StringBasedGaussianKernel, InputTypeString

import logging
logger = logging.getLogger(__name__)


class KernelChoice(str, enum.Enum):
    StringGaussianJaccard = "string_jaccard_distance_gaussian"
    StringGaussianMeteor = "string_meteor_distance_gaussian"    


def get_all_outcomes(seq_text_outcomes: ty.List[Generation]) -> InputTypeString:
    seq_documents_tokens = nltk_preprocess_text(seq_text_outcomes)

    return seq_documents_tokens


def initialize_mmd_estimator(
        kernel_choice: KernelChoice,
        dict_sample_set: ty.Optional[ty.Dict[DecodingConfig, ty.List[str]]]=None,
        seq_text_outcomes: ty.Optional[ty.List[Generation]] = None) -> QuadraticMmdEstimator:

    assert (dict_sample_set is not None) or (seq_text_outcomes is not None)
    if dict_sample_set is not None and seq_text_outcomes is None:
        seq_text = list(dict_sample_set.values())
        seq_documents = list(itertools.chain.from_iterable(seq_text))
        _seq_texts = get_all_outcomes(seq_documents)            
    elif dict_sample_set is None and seq_text_outcomes is not None:
        _seq_texts = get_all_outcomes(seq_text_outcomes)
    else:
        raise ValueError()
    # end if

    if kernel_choice == KernelChoice.StringGaussianJaccard:
        _distance_module_jaccard = JaccardDistanceModule()
        _distance_module_jaccard.get_vectorizer(_seq_texts)
        _kernel_module = StringBasedGaussianKernel(_distance_module_jaccard)
        bandwidth = _kernel_module._get_median_single(_seq_texts, _seq_texts)
        assert hasattr(_distance_module_jaccard, "vectorizer")
        _kernel_module.bandwidth = torch.nn.Parameter(bandwidth, requires_grad=False)
        
        _mmd_estimator = QuadraticMmdEstimator(_kernel_module)
    elif kernel_choice == KernelChoice.StringGaussianMeteor:
        _distance_module_meteor = MeteorDistanceModule()
        _kernel_module = StringBasedGaussianKernel(_distance_module_meteor)
        bandwidth = _kernel_module._get_median_single(_seq_texts, _seq_texts)
        _kernel_module.bandwidth = torch.nn.Parameter(bandwidth, requires_grad=False)
        
        _mmd_estimator = QuadraticMmdEstimator(_kernel_module)
    else:
        raise ValueError()
    # end if

    return _mmd_estimator


class ReturnObj_construct_graph_matrix(ty.NamedTuple):
    adjacency_matrix: np.ndarray
    node_labels: ty.List[DecodingConfig]
    mmd_statistics: ty.Dict[ty.Tuple[int, int], MmdValues]

    def create_networkx_graph(self) -> nx.Graph:
        """
        Converts an adjacency matrix and its labels into a NetworkX graph.

        Args:
            adjacency_matrix: The (N x N) symmetric matrix of edge weights (distances).
            node_labels: A list of DecodingConfig objects corresponding to the matrix indices.

        Returns:
            A NetworkX Graph object with weighted edges.
        """
        # Create a graph directly from the numpy matrix.
        # Edge weights from the matrix are automatically stored in the 'weight' attribute.
        G = nx.from_numpy_array(self.adjacency_matrix)

        # Create a mapping from integer node indices (0, 1, 2...) to descriptive string labels.
        # e.g., {0: "top_k(50)", 1: "top_k(10)", 2: "beam_search(5)"}
        label_mapping = {
            i: f"{label.method}({label.model_dump_json()})" if label.model_dump_json() is not None else label.method
            for i, label in enumerate(self.node_labels)
        }

        # Relabel the nodes in the graph to be descriptive strings instead of integers.
        nx.relabel_nodes(G, label_mapping, copy=False)

        logger.debug("\nSuccessfully created NetworkX graph.")
        return G


# The v1 is based on token sequences.
def construct_graph_matrix(dict_sample_set: ty.Dict[DecodingConfig, ty.List[str]],
                           mmd_estimator: QuadraticMmdEstimator) -> ReturnObj_construct_graph_matrix:
    # calculate the adjacency-matrix

    node_labels = list(dict_sample_set.keys())
    n_nodes = len(node_labels)

    # Initialize the outputs
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    mmd_statistics = {}

    logger.debug(f"Constructing graph for {n_nodes} decoding configurations...")

    # Use itertools.combinations_with_replacement to efficiently iterate through
    # all unique pairs of nodes (i, j) where i <= j.
    for i, j in itertools.combinations_with_replacement(range(n_nodes), 2):
        if i == j:
            continue
        # end if

        config_i = node_labels[i]
        config_j = node_labels[j]

        # Retrieve the corresponding sample sets
        samples_i = dict_sample_set[config_i]
        samples_j = dict_sample_set[config_j]


        samples_i_tokens = nltk_preprocess_text(samples_i)
        samples_j_tokens = nltk_preprocess_text(samples_j)

        if len(samples_i_tokens) == 1:
            samples_i_tokens = samples_i_tokens + samples_i_tokens
        if len(samples_j_tokens) == 1:
            samples_j_tokens = samples_j_tokens + samples_j_tokens
        # end if

        # Compute the MMD distance and stats
        logger.debug(f"  Calculating MMD between node {i} ({config_i.method}) and node {j} ({config_j.method})...")
        mmd_value_obj = mmd_estimator.forward(samples_i_tokens, samples_j_tokens)

        # Populate the symmetric matrix
        adjacency_matrix[i, j] = mmd_value_obj.mmd
        adjacency_matrix[j, i] = mmd_value_obj.mmd

        # Store the auxiliary statistics
        mmd_statistics[(i, j)] = mmd_value_obj
    # end for

    logger.debug("Graph construction complete.")

    return ReturnObj_construct_graph_matrix(adjacency_matrix, node_labels, mmd_statistics)

