from pathlib import Path

from hallucination_mt.guerreiro_2023_wmt.utils_models.utils import load_model


def load_fairseq_model(path_fairseq_model_dir: Path,
                    path_fairseq_model_file: Path,
                    path_sentencepiece_model: Path):
    assert path_fairseq_model_dir.exists(), f"Directory not found: {path_fairseq_model_dir}"
    assert path_fairseq_model_file.exists(), f"File not found: {path_fairseq_model_file}"
    assert path_sentencepiece_model.exists(), f"File not found: {path_sentencepiece_model}"

    model = load_model(
        path_fairseq_model_dir,
        path_fairseq_model_file,
        path_sentencepiece_model,
    )
    # Set model to evaluation mode
    model.eval()

    return model
