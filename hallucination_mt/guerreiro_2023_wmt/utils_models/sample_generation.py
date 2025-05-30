import torch
import typing as ty

from fairseq.hub_utils import GeneratorHubInterface


"""A module for sampling multiple times with the same input text."""


class DecodedTranslationObject(ty.NamedTuple):
    source_tensor_tokens: torch.Tensor  # tensor of token-ids
    target_tensor_tokens: torch.Tensor  # tensor of token-ids
    tensor_attention: torch.Tensor  # tensor of attention weights (source-len x target-len)
    score: float  # ????
    positional_scores: torch.Tensor  # tensor of scores (target-len)
    target_text: str  # decoded text

    def __str__(self):
        return f"Translation: {self.target_text}"


def function_sample_multiple_times(fairseq_interface: GeneratorHubInterface,
                                   input_text: str,
                                   n_sampling: int,
                                   temperature: float,
                                   sampling_topk: int = -1,
                                   sampling_topp: float = -1.0,
                                   is_sampling_in_iteration: bool = False) -> ty.List[DecodedTranslationObject]:
    """A custom function to sample multiple times with the same input text.
    This function conducts tokenization just one time.
    
    Args:
        is_sampling_in_iteration: If True, the sampling is executed in the iteration.
            This flag is for saving the RAM or GPU memory.
            However, the execution speed will be slower.
    """
    # Check the device
    if torch.cuda.is_available():
        # TODO: multiple GPUs
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")    
    # end if

    fairseq_interface.eval()
    fairseq_interface = fairseq_interface.to(device)

    # The `encode` method is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/hub_utils.py#L246-L249
    # The method does: tokenize, apply_bpe, and binarize.
    tensor_source_tokens = fairseq_interface.encode(input_text)

    with torch.no_grad():
        # Note: Possible `generate` options are defined at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810 
        # `generate` method executes `inference_step` method of `TranslationTask` class.
        # `**kwargs` arguments are passed to `build_generator` method of `TranslationTask` class first,
        # and then, the generator object is passed to the `inference_step` method.
        # See `build_generator` API at here: https://fairseq.readthedocs.io/en/latest/tasks.html#fairseq.tasks.FairseqTask.build_generator
        # The args object is `fairseq.dataclass.configs.GenerationConfig`.
        # The `GenerationConfig` definition is at https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/dataclass/configs.py#L810
        
        
        # Note about the `generate` outcomes.
        # The outcome comes from `generate` method of `fairseq.sequence_generator.SequenceGenerator`: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L189
        # The outcome object is from the method `_generate`: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L206
        # However, no documentations available.
        # The outcome object seems to be: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L717
        # I list-up keys of the object: 
        # - `tokens`: The tokenized output Torch.Tensor.
        # - `score`: ????.
        # - `attention`: The attention weights, (src_len x tgt_len).
        # - `alignment`: ?????
        # - `positional_scores`: The score. The tensor size is the same as `tokens`.
            # The definition seems to be: https://github.com/facebookresearch/fairseq/blob/ecbf110e1eb43861214b05fa001eff584954f65a/fairseq/sequence_generator.py#L668-L672

        if is_sampling_in_iteration:
            output_stack = []
            for _ in range(n_sampling):
                translations = fairseq_interface.generate(
                    tokenized_sentences=tensor_source_tokens,
                    sampling=True,
                    temperature=temperature,
                    sampling_topk=sampling_topk,
                    sampling_topp=sampling_topp,
                    beam=1)
                output_stack.append(translations)
            # end for
        else:
            seq_input_tensor = [tensor_source_tokens] * n_sampling
            translations = fairseq_interface.generate(
                tokenized_sentences=seq_input_tensor,
                sampling=True,
                temperature=temperature,
                sampling_topk=sampling_topk,
                sampling_topp=sampling_topp,
                beam=1)
            output_stack = translations
        # end if
    # end with

    # decoding from the token-id -> text
    seq_decoded_objects = []
    for __object in output_stack:
        assert len(__object) == 1, f"Unexpected length of the output: {len(__object)}"
        assert len(__object[0]) > 0, f"Unexpected length of the output: {len(__object[0])}"
        assert isinstance(__object[0], dict), f"Unexpected type of the output: {type(__object[0])}"
        __dict_obj = __object[0]
        tensor_tokens = __dict_obj['tokens'].cpu()
        tensor_attention = __dict_obj['attention'].cpu()
        score = __dict_obj['score'].cpu()
        positional_scores = __dict_obj['positional_scores'].cpu()

        translation_text = fairseq_interface.decode(tensor_tokens)

        __decoded_obj = DecodedTranslationObject(
            source_tensor_tokens=tensor_source_tokens,
            target_tensor_tokens=tensor_tokens,
            tensor_attention=tensor_attention,
            score=score,
            positional_scores=positional_scores,
            target_text=translation_text)
        # Note: I guess `score` is a perplexity value.
        # Since `score = sum(positional_scores) / len(positional_scores)`
        seq_decoded_objects.append(__decoded_obj)
    # end for

    return seq_decoded_objects
# end function
