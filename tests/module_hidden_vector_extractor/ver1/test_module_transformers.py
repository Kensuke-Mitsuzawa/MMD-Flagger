from pathlib import Path
import toml
import torch

from hallucination_mt.module_translation_handler.ver1 import module_transformer_handler
from hallucination_mt.module_hidden_vector_extractor.ver1 import module_transformers


def test_extract_encoder_output_teacher_forcing(resource_path_root: Path):
    code_name_nllb = "facebook/nllb-200-distilled-600M"

    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."
    target_text = "Finally, I have my previously hacked Youtube channel back."

    translation_handler = module_transformer_handler.TransformersTranslationModelHandler(
        src_lang="deu_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        batch_size=2,
        max_len_a=1.0,
        max_len_b=10,
        target_lang="eng_Latn",
    )

    hidden_vector_extractor = module_transformers.TransformerVectorExtractor(
        translation_handler)
    hidden_obj = hidden_vector_extractor.extract_encoder_output_teacher_forcing(
        source_text=input_text,
        target_translation_text=target_text,
        target_encoder_layers=None,
        target_decoder_layers=None)
    




# def test_extract_hidden_states(resource_path_root: Path):
#     code_name_nllb = "facebook/nllb-200-distilled-600M"

#     input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

#     translation_handler = module_transformer_handler.TransformersTranslationModelHandler(
#         src_lang="deu_Latn",
#         model_name=code_name_nllb,
#         random_seed=42,
#         batch_size=2,
#         max_len_a=1.0,
#         max_len_b=10,
#         target_lang="eng_Latn",
#     )

#     hidden_vector_extractor = module_transformers.TransformerVectorExtractor(
#         translation_handler)
#     hidden_obj = hidden_vector_extractor.extract_hidden_states(
#         source_text=input_text,
#         target_encoder_layers=None,
#         target_decoder_layers=None
#     )


def test_extract_word_embeddings(resource_path_root: Path):

    code_name_nllb = "facebook/nllb-200-distilled-600M"

    input_text = "Eeeeeendlich habe ich meinen vormals gehackten Youtubekanal wieder."

    translation_handler = module_transformer_handler.TransformersTranslationModelHandler(
        src_lang="deu_Latn",
        model_name=code_name_nllb,
        random_seed=42,
        batch_size=2,
        max_len_a=1.0,
        max_len_b=10,
        target_lang="eng_Latn",
    )
    result = translation_handler.translate_beam_search(
        input_text=input_text,
        temperature=1.0,
    )
    assert isinstance(result.translation_text, str)
    assert isinstance(result.log_score, float)

    hidden_vector_extractor = module_transformers.TransformerVectorExtractor(
        translation_handler)
    tensor_emb_vector = hidden_vector_extractor.extract_word_embeddings(
        translated_text=result.translation_text,
        tensor_translated_text=result.tensor_translation_tokens
    )
    assert isinstance(tensor_emb_vector, torch.Tensor)
