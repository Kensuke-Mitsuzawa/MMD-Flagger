import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mmd_flagger.module_features.interface_feature_extraction import InterfaceFeatureExtraction
from mmd_flagger.module_features.models import GenerationInfoDict
from mmd_flagger.module_features.module_feature_extraction import (
    HiddenStatesExtractor,
    WordEmbeddingExtractor,
    LapEigvalsExtractor,
    AttentionEigenValsExtractor,
)


def test_interface_feature_extraction():
    prompt = "Hello, how are you?"
    response = "I'm fine, thank you."
    
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", attn_implementation="eager")

    interface = InterfaceFeatureExtraction(tokenizer_target=tokenizer, model_target=model)
    features = interface.extract(prompt, response)

    assert isinstance(features, GenerationInfoDict)
    assert features.prompt_input_ids.ndim == 1
    assert features.generated_token_ids.ndim == 1
    
    n_prompt = features.prompt_input_ids.shape[0]
    n_response = features.generated_token_ids.shape[0]
    n_total = n_prompt + n_response

    assert len(features.layer_hidden_states) > 0
    for layer_idx, h_state in features.layer_hidden_states.items():
        assert h_state.shape[0] == n_total

    assert features.attention_matrix.ndim == 4
    assert features.attention_matrix.shape[2] == n_total
    assert features.attention_matrix.shape[3] == n_total
    assert features.word_embeddings_generated_tokens.shape[0] == n_response

    # Verify downstream feature extractors work with the extracted features
    hs_outputs = HiddenStatesExtractor().extract(features)
    assert len(hs_outputs) == len(features.layer_hidden_states)

    we_outputs = WordEmbeddingExtractor().extract(features)
    assert len(we_outputs) == 1

    lap_outputs = LapEigvalsExtractor().extract(features)
    assert len(lap_outputs) == 1

    att_outputs = AttentionEigenValsExtractor().extract(features)
    assert len(att_outputs) == 1


if __name__ == "__main__":
    test_interface_feature_extraction()

