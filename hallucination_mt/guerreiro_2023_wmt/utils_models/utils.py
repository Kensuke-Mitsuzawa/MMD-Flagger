import typing as ty
from pathlib import Path

import torch

from fairseq.models.transformer import TransformerModel
from fairseq.hub_utils import GeneratorHubInterface



def load_model(path_fairseq_model_dir: Path, 
               path_fairseq_model_file: Path,
               model_sentencepiece: Path) -> GeneratorHubInterface:
    # Load the pre-trained Fairseq model
    model = TransformerModel.from_pretrained(
        path_fairseq_model_dir,
        checkpoint_file=path_fairseq_model_file.as_posix(),
        data_name_or_path=path_fairseq_model_dir.as_posix(),
        bpe="sentencepiece",
        sentencepiece_model=model_sentencepiece
    )

    return model



def extract_word_embeddings(model: GeneratorHubInterface, 
                            sentence: str) -> torch.Tensor:
    """Extract word embeddings from a sentence using a pre-trained Fairseq model.
    
    
    Return:
        - torch.Tensor: The tensor of embeddings. The tensor is (seq_len, embed_dim).
    """
    _model = model.to('cpu')

    # Tokenize and convert to indices
    # tokenized = model.bpe.encode(sentence).split()
    # token_indices = [model.task.source_dictionary.index(token) for token in tokenized]
    token_indices = _model.encode(sentence)
    token_indices = token_indices.to('cpu')

    # # Check the device
    # if torch.cuda.is_available():
    #     # TODO: multiple GPUs
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")    
    # # end if
    # Convert token indices to embeddings
    with torch.no_grad():
        token_tensor = torch.tensor(token_indices).unsqueeze(0)  # Batch size of 1
        embeddings = _model.models[0].decoder.embed_tokens(token_tensor)
    # end with
    
    assert len(embeddings) == 1, f"Expected 1 tensor, got {len(embeddings)}"
    # the tensor is (1, seq_len, embed_dim). I extract (seq_len, embed_dim)
    return embeddings[0]



def extract_word_embeddings_batch(model: GeneratorHubInterface, 
                                  seq_sentence: ty.List[str]
                                  ) -> ty.List[torch.Tensor]:
    seq_tesnor = []
    for _sentence in seq_sentence:
        _t = extract_word_embeddings(model, _sentence)
        seq_tesnor.append(_t)
    # end for
    return seq_tesnor
