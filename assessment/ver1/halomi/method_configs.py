import typing as ty
from pathlib import Path
from dataclasses import asdict, dataclass
import dataclasses

from hallucination_mt.module_hidden_vector_extractor.ver1.module_base import OPTION_EMBEDDING_LAYERS


@dataclass
class MmdErrorFlaggerTrajectoryVer2:
    # algorithm config
    ## A list of n_sampling for each temperature
    option_n_translation_sampling: ty.List[int]
    ## A list of \tau parameter sequence.
    option_temperature_sequence: ty.List[ty.List[float]]

    ## embedding option
    option_embedding_layer: ty.List[str]

    ## Trajectory Rules
    option_trajectory_rule_versions: ty.List[str]
    option_trajectory_rule_smoothing: ty.List[str]
    option_trajectory_rule_smoothing_window: ty.List[int]

    # default config fields
    approach_name: str = "MmdErrorFlaggerTrajectoryVer2"
    
    nllb_model_name: str = "facebook/nllb-200-distilled-600M"

    max_len_a: float = 0.0
    max_len_b: int = 200

    def __post_init__(self):
        for __option_embedding_layer in self.option_embedding_layer:
            assert __option_embedding_layer in OPTION_EMBEDDING_LAYERS, f"embedding_layer={__option_embedding_layer} is not in {OPTION_EMBEDDING_LAYERS}"
        # end for

    def to_dict(self):
        _obj = asdict(self)
        return _obj


@dataclass
class MmdErrorFlaggerVer1:
    path_fairseq_model_dir: Path
    path_fairseq_model_file: Path
    path_sentencepiece_model: Path

    # algorithm config
    n_translation_sampling: int    
    set_temperature_low: ty.List[float]
    set_temperature_high: ty.List[float]

    # default config fields
    approach_name: str = "MmdErrorFlaggerVer1"
    
    max_len_a: float = 0.0
    max_len_b: int = 200

    def __post_init__(self):
        assert self.path_fairseq_model_dir.exists(), f"Fairseq model directory not found at {self.path_fairseq_model_dir}"
        assert self.path_fairseq_model_file.exists(), f"Fairseq model file not found at {self.path_fairseq_model_file}"
        assert self.path_sentencepiece_model.exists(), f"Sentencepiece model not found at {self.path_sentencepiece_model}"

    def to_dict(self):
        _obj = asdict(self)
        _obj["path_fairseq_model_dir"] = _obj["path_fairseq_model_dir"].as_posix()
        _obj["path_fairseq_model_file"] = _obj["path_fairseq_model_file"].as_posix()
        _obj["path_sentencepiece_model"] = _obj["path_sentencepiece_model"].as_posix()
        return _obj


# ---------------------------------------------------------
# Baselines

@dataclass
class Raunak2021ApproachConfig:
    path_python37_exec: Path

    approach_name: str = "OscillatoryDetectionRaunak2021"
    
    ngram_size: int = 2
    count_threshold: int = 10
    difference_threshold: int = 5
    min_length_threshold: int = 10

    def __post_init__(self):
        assert self.path_python37_exec.exists(), f"Python 3.7 executable not found at {self.path_python37_exec}"

    def to_dict(self):
        _obj = asdict(self)
        _obj["path_python37_exec"] = _obj["path_python37_exec"].as_posix()
        return _obj


@dataclass
class Guerreiro2023SeqLogprob:
    nllb_model_name: str = "facebook/nllb-200-distilled-600M"

    max_len_a: float = 0.0
    max_len_b: int = 200

    # algorithm config
    percentile_threshold: float = 40

    # default config fields
    approach_name: str = "Guerreiro2023SeqLogprob"

    def __post_init__(self):
        pass

    def to_dict(self):
        _obj = asdict(self)
        return _obj


@dataclass
class Guerreiro2023MCDSim:
    path_fairseq_model_dir: Path
    path_fairseq_model_file: Path
    path_sentencepiece_model: Path

    # algorithm config
    num_samples: int = 10
    temperature_value: float = 1.0
    percentile: float = 0.004  # 0.4%. Taking smaller 0.4% values.

    # default config fields
    approach_name: str = "Guerreiro2023MCDSim"
        
    max_len_a: float = 0.0
    max_len_b: int = 200

    def __post_init__(self):
        assert self.path_fairseq_model_dir.exists(), f"Fairseq model directory not found at {self.path_fairseq_model_dir}"
        assert self.path_fairseq_model_file.exists(), f"Fairseq model file not found at {self.path_fairseq_model_file}"
        assert self.path_sentencepiece_model.exists(), f"Sentencepiece model not found at {self.path_sentencepiece_model}"

    def to_dict(self):
        _obj = asdict(self)
        _obj["path_fairseq_model_dir"] = _obj["path_fairseq_model_dir"].as_posix()
        _obj["path_fairseq_model_file"] = _obj["path_fairseq_model_file"].as_posix()
        _obj["path_sentencepiece_model"] = _obj["path_sentencepiece_model"].as_posix()
        return _obj
