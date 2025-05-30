from pathlib import Path
import toml

import torch

from hallucination_mt.module_flagging import utils
from hallucination_mt.module_translation_handler.ver1 import module_fairseq_handler

from hallucination_mt.module_hidden_vector_extractor.ver1.module_fairseq import FairSeqVectorExtractor


def test_extract_hidden_states(resource_path_root: Path):
    path_dir_model = resource_path_root / "model_guerreiro_2023"
    assert path_dir_model.exists(), f"path_dir_fairseq_model={path_dir_model}"

    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
        path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

    fairseq_vector_extractor = FairSeqVectorExtractor(model=model_encoder_decoder_mt)
    seq_layer_encoder, seq_layer_decoder = fairseq_vector_extractor.get_all_possible_layers()

    assert len(seq_layer_encoder) > 0, "seq_layer_encoder is empty"
    assert len(seq_layer_decoder) > 0, "seq_layer_decoder is empty"

    # Test. Specifying the layers to extract
    return_tuple_obj = fairseq_vector_extractor.extract_hidden_states(source_text=input_text,
                                                   target_encoder_layers=[1, 2],
                                                   target_decoder_layers=[1, 2],)
    
    assert isinstance(return_tuple_obj.decoder_layer2states, dict) and isinstance(return_tuple_obj.encoder_layer2states, dict)
    assert list(return_tuple_obj.decoder_layer2states.keys()) == [1, 2]
    assert list(return_tuple_obj.encoder_layer2states.keys()) == [1, 2]

    # Test. Extracting all layers
    return_tuple_obj = fairseq_vector_extractor.extract_hidden_states(source_text=input_text,
                                                    target_encoder_layers=None,
                                                    target_decoder_layers=None,)
    assert isinstance(return_tuple_obj.decoder_layer2states, dict) and isinstance(return_tuple_obj.encoder_layer2states, dict)
    assert len(return_tuple_obj.decoder_layer2states) == len(seq_layer_decoder)
    assert len(return_tuple_obj.encoder_layer2states) == len(seq_layer_encoder)


def test_get_all_possible_layers(resource_path_root: Path):
    path_dir_model = resource_path_root / "model_guerreiro_2023"
    assert path_dir_model.exists(), f"path_dir_fairseq_model={path_dir_model}"

    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
        path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

    fairseq_vector_extractor = FairSeqVectorExtractor(model=model_encoder_decoder_mt)
    seq_layer_encoder, seq_layer_decoder = fairseq_vector_extractor.get_all_possible_layers()

    assert len(seq_layer_encoder) > 0, "seq_layer_encoder is empty"
    assert len(seq_layer_decoder) > 0, "seq_layer_decoder is empty"



def test_extract_encoder_output(resource_path_root: Path):
    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
        path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

    fairseq_vector_extractor = FairSeqVectorExtractor(
        model=model_encoder_decoder_mt,
    )

    # Extract encoder output by the source text (string)
    tensor_encoder_output_string = fairseq_vector_extractor.extract_encoder_output(
        source_text=input_text
    )
    assert tensor_encoder_output_string is not None, "tensor_encoder_output_string is None"
    assert "token_ids" in tensor_encoder_output_string, "token_ids not in tensor_encoder_output_string"
    assert "encoder_out" in tensor_encoder_output_string, "encoder_out not in tensor_encoder_output_string"

    assert len(tensor_encoder_output_string['token_ids']) == len(tensor_encoder_output_string['encoder_out'])
    assert len(tensor_encoder_output_string["encoder_out"].shape) == 2


def test_extract_word_embeddings(resource_path_root: Path):

    path_dir_model = Path(resource_path_root / "model_guerreiro_2023")
    assert path_dir_model.exists(), f"path_dir_model={path_dir_model}"

    model_encoder_decoder_mt = utils.load_fairseq_model(
        path_fairseq_model_dir=path_dir_model / "wmt18_de-en",
        path_fairseq_model_file=path_dir_model / "checkpoint_best.pt",
        path_sentencepiece_model=path_dir_model / "sentencepiece_models/sentencepiece.joint.bpe.model",)

    handler = module_fairseq_handler.FaiseqTranslationModelHandler(
        model_encoder_decoder_mt=model_encoder_decoder_mt,
        is_sampling=True,
        n_sampling=25,
        random_seed=-1)

    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."
    sample_res_1st = handler.sample_multiple_times(
        input_text=input_text,
        temperature=0.5,
        n_sampling=25
    )

    fairseq_vector_extractor = FairSeqVectorExtractor(
        model=model_encoder_decoder_mt,
    )
    # Extract word embeddings by the translated text (string)
    tensor_word_embedding_string = fairseq_vector_extractor.extract_word_embeddings(
        translated_text=sample_res_1st[0].target_text
    )
    # Extract word embeddings by the translated text (tensor)
    tensor_word_embedding_tensor = fairseq_vector_extractor.extract_word_embeddings(
        translated_text=sample_res_1st[0].target_text,
        tensor_translated_text=sample_res_1st[0].target_tensor_tokens
    )
    assert torch.equal(tensor_word_embedding_string, tensor_word_embedding_tensor)
    