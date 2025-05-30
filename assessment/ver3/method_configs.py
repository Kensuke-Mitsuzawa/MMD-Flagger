import typing as ty
from pathlib import Path
from dataclasses import asdict, dataclass
import dataclasses
import re


@dataclass
class ModelConfigHalomi:
    nllb_model_name: str = "facebook/nllb-200-distilled-600M"


@dataclass
class ModelConfigLfanHall:
    path_fairseq_model_dir: Path
    path_fairseq_model_file: Path
    path_sentencepiece_model: Path

    batch_size: int = 25

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



@dataclass
class TranslationConfig:
    path_cache_translator: Path

    candidates_n_sampling_stochastic: ty.List[int]
    candidates_temperature_parameter: ty.List[float]

    max_len_a: float = 0.0
    max_len_b: int = 200

    temperature_beam: float = 1.0

    is_save_convert_fp16: bool = False

    option_target_sentence_id: ty.Optional[ty.List[str]] = None  # a list of sentence id that you want to process in priority.

    option_embedding_layer: ty.Optional[ty.List[str]] = None


@dataclass
class MmdErrorFlaggerTrajectoryVer3Config:
    # algorithm config
    ## A list of n_sampling for each temperature
    option_n_translation_sampling: ty.List[int]
    ## A list of \tau parameter sequence.
    option_temperature_sequence: ty.List[ty.List[float]]

    ## embedding option
    option_vector_preprocess: ty.List[str]  # 'avg', 'concat'
    option_max_token_length_vector_concat: ty.List[str]  # "max_calibration" or "fixed" (a fixed max tokens.)
    option_max_token_length: ty.List[int]  # use it when "option_max_token_length_vector_concat" includes "fixed". If "max_calibration", the script automatically sets max_token_length = -1.
    option_embedding_layer: ty.List[str]
    
    # kernel option
    option_kernel_type: ty.List[str]  # gaussian or dot
    option_kernel_gaussian_length_scale: ty.List[int]  # option to set the length scale for a Gaussian Kernel  
    option_kernel_gaussian_length_scale_computation: ty.List[str]  # either "single" or "dimensionwise".

    ## Trajectory Rules
    option_trajectory_rule_versions: ty.List[str]
    option_trajectory_rule_smoothing: ty.List[str]
    option_trajectory_rule_smoothing_window: ty.List[int]

    # default config fields
    approach_name: str = "MmdErrorFlaggerTrajectoryVer3"
    
    # @staticmethod
    # def _update_option_max_token_length_vector_concat(option_max_token_length_vector_concat) -> ty.List[ty.Union[str, int]]:
    #     seq_replaced = []
    #     for _option in option_max_token_length_vector_concat:
    #         if isinstance(_option, str):
    #             _match_res = re.match(r'^[0-9]+$', _option)
    #             if _match_res is None:
    #                 seq_replaced.append(_option)
    #             else:
    #                 seq_replaced.append(int(_match_res.group(0)))
    #             # end if
    #         else:
    #             seq_replaced.append(_option)
    #     # end for
    #     return seq_replaced

    def __post_init__(self):
        assert set(self.option_vector_preprocess).issubset(set(['avg', 'concat']))
        # self.option_max_token_length_vector_concat = self._update_option_max_token_length_vector_concat(
        #     self.option_max_token_length_vector_concat)
        if self.option_max_token_length == []:
            self.option_max_token_length = [-1]  # temporaly adhoc solution.


    def to_dict(self):
        _obj = asdict(self)
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
