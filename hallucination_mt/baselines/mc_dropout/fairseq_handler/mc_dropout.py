from pathlib import Path
import typing as ty
import dataclasses

import abc

import torch
import numpy as np
from fairseq.models.transformer import TransformerModel
from fairseq.modules import FairseqDropout
from fairseq.hub_utils import GeneratorHubInterface


import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize


class BaseMetric(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self,
                 reference: str,
                 seq_hypothesis: ty.List[str]) -> ty.List[float]:
        pass


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        # setup nltk tokenizer
        nltk.download('punkt_tab')
        nltk.download('wordnet')

    def __call__(self,
                 reference: str,
                 seq_hypothesis: ty.List[str]) -> ty.List[float]:
        """Compute METEOR over multiple MC Dropout samples."""
        scores = []
        for _hyp in seq_hypothesis:
            assert isinstance(_hyp, str), f"_hyp must be a string"
            _seq_tokens_hyp = word_tokenize(_hyp)
            assert isinstance(_seq_tokens_hyp, list), f"_seq_tokens must be a list"

            _seq_tokens_ref = word_tokenize(reference)
            assert isinstance(_seq_tokens_ref, list), f"_seq_tokens must be a list"

            # _score = meteor_score.stem_match([_seq_tokens_ref], _seq_tokens_hyp)
            _score = meteor_score.meteor_score([_seq_tokens_ref], _seq_tokens_hyp)
            scores.append(_score)
        # end for

        return scores


# ----------------------------------------------------------------------


@dataclasses.dataclass
class OutputMcDropOut:
    source_text: str
    reference_text: str
    outputs: ty.List[str]
    avg_dissimilarity: float
    metric_class: str
    seq_dissimilarity: ty.List[float]
    temperature_value: float
    num_samples: int




class DissimilarityMcDropOut(object):
    def __init__(self,
                 fairseq_interface: GeneratorHubInterface,
                 metric_obj: BaseMetric = MeteorMetric()) -> None:
        self.metric_obj = metric_obj
        self.fairseq_interface = fairseq_interface

        self.fairseq_interface.eval()

    @staticmethod
    def switch_dropout(fairseq_interface: GeneratorHubInterface) -> GeneratorHubInterface:
        model_decoder = fairseq_interface.models[0].decoder
        
        is_dropout_enabled = False
        for name, module in model_decoder.named_modules():
            if isinstance(module, FairseqDropout):
                # module.training = True
                module.apply_during_inference = True
                assert module.apply_during_inference == True, f"Failed to set apply_during_inference to True"
                is_dropout_enabled = True
            # end if
        # end for

        assert is_dropout_enabled == True, f"Failed to enable dropout during inference"
        return fairseq_interface
    
    @staticmethod
    def device_cuda(fairseq_interface: GeneratorHubInterface) -> GeneratorHubInterface:
        # Check the device
        if torch.cuda.is_available():
            # TODO: multiple GPUs
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")    
        # end if

        fairseq_interface = fairseq_interface.to(device)

        return fairseq_interface

    def run_inference(self, 
                      source_text: str, 
                      num_samples: int = 10,
                      temperature_value: float = 1.0,
                      max_len_a: float = 0.0,
                      max_len_b: int = 200,
                      n_beam: int = 5,       
                      reference_text: ty.Optional[str] = None) -> OutputMcDropOut:
        if reference_text is None:
            # setting the reference as the translation without dropout and Beam search (default)
            assert isinstance(source_text, str), f"source_text must be a string"
            __reference_text = self.fairseq_interface.translate(sentences=[source_text], 
                                                              beam=n_beam, 
                                                              temperature=temperature_value, 
                                                              max_len_a=max_len_a, 
                                                              max_len_b=max_len_b)  # type: ignore
            reference_text = __reference_text[0]
            assert reference_text is not None, f"Failed to get reference text"
        else:
            reference_text = reference_text    
        # end if

        fairseq_interface_dropout = self.switch_dropout(self.fairseq_interface)
        fairseq_interface_dropout = self.device_cuda(fairseq_interface_dropout)

        outputs = []
        for _ in range(num_samples):
            __output = fairseq_interface_dropout.translate([source_text],
                                                         beam=n_beam, 
                                                         temperature=temperature_value, 
                                                         max_len_a=max_len_a, 
                                                         max_len_b=max_len_b)
            output = __output[0]
            outputs.append(output)
        # end for

        seq_metrics = []
        # compute metric
        seq_metrics = self.metric_obj(reference=reference_text, seq_hypothesis=outputs)
        # end for


        output = OutputMcDropOut(source_text=source_text,
                                 reference_text=reference_text,
                                 outputs=outputs,
                                 avg_dissimilarity=float(np.mean(seq_metrics).item()),
                                 metric_class=self.metric_obj.__class__.__name__,
                                 seq_dissimilarity=seq_metrics,
                                 num_samples=num_samples,
                                 temperature_value=temperature_value)
        return output
