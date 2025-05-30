from pathlib import Path
import typing as ty
import dataclasses

import abc

import torch
import numpy as np
from fairseq.hub_utils import GeneratorHubInterface



@dataclasses.dataclass
class OutputLogProbability:
    source_text: str
    translated_text: str
    log_probability: float
    temperature_value: float
    beam_value: int
    max_len_a_value: float
    max_len_b_value: float




class ComputeSequenceLogProbability(object):
    def __init__(self,
                 fairseq_interface: GeneratorHubInterface
                 ) -> None:
        self.fairseq_interface = fairseq_interface

        self.fairseq_interface.eval()

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

    def __call__(self,
                 source_text: str,
                 temperature: float = 1.0,
                 beam: int = 1,
                 max_len_a: float = 0.0,
                 max_len_b: float = 200) -> OutputLogProbability:
        fairseq_interface = self.device_cuda(self.fairseq_interface)

        with torch.no_grad():
            tensor_source_tokens = fairseq_interface.encode(source_text)                
            translation_obj = fairseq_interface.generate(
                tokenized_sentences=tensor_source_tokens,  # type: ignore
                sampling=False,
                temperature=temperature,
                beam=beam,
                max_len_a_mt=max_len_a,
                max_len_b_mt=max_len_b)
            assert len(translation_obj) == beam, f"len(translation_obj)={len(translation_obj)}"
            assert isinstance(translation_obj, list), f"type(translation_obj)={type(translation_obj)}"
            assert isinstance(translation_obj[0], dict), f"type(translation_obj[0])={type(translation_obj[0])}"
            # I take the top-beam translation
            tensor_tokens = translation_obj[0]['tokens']
            translation_text = fairseq_interface.decode(tensor_tokens)  # type: ignore

            assert 'score' in translation_obj[0], f"translation_obj[0]={translation_obj[0]}"
            assert isinstance(translation_obj[0]['score'], torch.Tensor), f"type(translation_obj[0]['score'])={type(translation_obj[0]['score'])}"
            score_log_probability = translation_obj[0]['score'].item()

        out_obj = OutputLogProbability(
            source_text=source_text,
            translated_text=translation_text,
            log_probability=score_log_probability,
            temperature_value=temperature,
            beam_value=beam,
            max_len_a_value=max_len_a,
            max_len_b_value=max_len_b
        )

        return out_obj