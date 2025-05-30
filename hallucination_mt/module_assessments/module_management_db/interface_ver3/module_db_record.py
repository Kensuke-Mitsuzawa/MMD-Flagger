import typing as ty
import dataclasses
import json


@dataclasses.dataclass
class DbTableRecordRaunak2021:
    config_name: str
    sentence_id: str
    approach_name: str
    flagging_label: bool
    flagging_argument_json: str  # field to save the extra information for the flagging approach
    record_id: ty.Optional[str] = None  # unique key. Automatically set.

    @classmethod
    def get_record_id(cls,
                      config_name: str,
                      approach_name: str,
                      _sentence_id: str) -> str:
        _db_record = f'{config_name}/{approach_name}/{_sentence_id}'
        return _db_record
    
    def __post_init__(self):
        self.record_id = self.get_record_id(self.config_name, self.approach_name, self.sentence_id)


@dataclasses.dataclass
class DbTableRecordGuerreiro2023SeqLogProb:
    config_name: str
    sentence_id: str
    log_probability: float
    log_probability_threshold: float
    flagging_label: bool
    flagging_argument_json: str
    record_id: ty.Optional[str] = None

    @classmethod
    def get_record_id(cls,
                      config_name: str,
                      _sentence_id: str) -> str:
        _db_record = f'{config_name}/{_sentence_id}'
        return _db_record
    
    def __post_init__(self):
        self.record_id = self.get_record_id(self.config_name, self.sentence_id)


@dataclasses.dataclass
class DbTableRecordGuerreiro2023McDSIM:
    config_name: str
    sentence_id: str
    score: float
    score_threshold: float
    flagging_label: bool
    flagging_argument_json: str
    record_id: ty.Optional[str] = None

    @classmethod
    def get_record_id(cls,
                      config_name: str,
                      _sentence_id: str) -> str:
        _db_record = f'{config_name}/{_sentence_id}'
        return _db_record
    
    def __post_init__(self):
        self.record_id = self.get_record_id(self.config_name, self.sentence_id)



@dataclasses.dataclass
class DbTableRecordProposalMmdFlaggerTrajectoryVer3:
    approach_name: str

    # fields of dataset attributes.
    dataset_name: str
    sentence_id: str
    source_language_code: str
    target_language_code: str

    # fields of computing distances.
    n_sampling: int
    tau_sequence: str  # Json field

    # fields of vector pre-processing.
    # I set columns rather than json string since these columns are interest of research.
    mode_vector_preprocess: str
    mode_target_embedding_layer: str
    mode_max_token_length_vector_concat: str
    mode_max_token_length: int

    # other possible fields.
    args_trajectory_options_json: str  # about trajectory rule
    args_kernel_options_json: str
    args_translation_options_json: str

    # fields of flagging.
    flagging_label: bool
    flagging_argument_json: str

    # fields of database, primary key.
    record_id: ty.Optional[str] = None

    def _generate_record_id(self) -> str:
        str_tau_sequence = json.dumps(self.tau_sequence)
        _db_record_id = f'{self.dataset_name}/{self.sentence_id}/{self.approach_name}/{self.n_sampling}/{str_tau_sequence}/{self.mode_vector_preprocess}/{self.mode_target_embedding_layer}/{self.mode_max_token_length_vector_concat}/{self.mode_max_token_length}/{self.args_trajectory_options_json}/{self.args_kernel_options_json}'
        return _db_record_id
    
    def _set_db_record_id(self) -> None:
        id = self._generate_record_id()
        self.record_id = id

    # @classmethod
    # def get_record_id(cls,
    #                   sentence_id: str,
    #                   dataset_name: str,
    #                   approach_name: str,
    #                   n_sampling: int,
    #                   tau_sequence: ty.List[float],
    #                   mode_vector_preprocess: str,
    #                   mode_target_embedding_layer: str,
    #                   mode_max_token_length_vector_concat: ty.Union[str, int],
    #                   args_trajectory_options_json: str,
    #                   args_kernel_options_json: str,
    #                   args_translation_options_json: str) -> str:
    #     str_tau_sequence = json.dumps(tau_sequence)
    #     _db_record = f'{dataset_name}/{sentence_id}/{approach_name}/{n_sampling}/{str_tau_sequence}/{mode_vector_preprocess}/{mode_target_embedding_layer}/{mode_max_token_length_vector_concat}/{args_trajectory_options_json}/{args_kernel_options_json}/{args_translation_options_json}'
    #     return _db_record
    
    def __post_init__(self):
        self._set_db_record_id()
        