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
class DbTableRecordProposalMmdFlaggerVer1:
    config_name: str
    sentence_id: str
    approach_name: str
    temperature_low: float
    temperature_high: float

    flagging_label: ty.Optional[bool]
    flagging_argument_json: str

    record_id: ty.Optional[str] = None
    is_success: bool = True  # when the translation is failed, this field is set to False

    @classmethod
    def get_record_id(cls,
                      config_name: str,
                      approach_name: str,
                      temperature_low: float,
                      temperature_high: float,
                      _sentence_id: str) -> str:
        _db_record = f'{config_name}/{approach_name}/{temperature_low}-{temperature_high}/{_sentence_id}'
        return _db_record
    
    def __post_init__(self):
        self.record_id = self.get_record_id(
            self.config_name, 
            self.approach_name, 
            self.temperature_low,
            self.temperature_high,
            self.sentence_id)


@dataclasses.dataclass
class DbTableRecordProposalMmdFlaggerTrajectoryVer1:
    config_name: str
    sentence_id: str
    approach_name: str

    n_sampling: int
    tau_sequence: str  # Json field

    trajectory_rule: str
    trajectory_rule_smoothing: str
    trajectory_rule_smoothing_window: ty.Optional[int]
    trajectory_rule_options_json: str

    flagging_label: bool
    flagging_argument_json: str

    record_id: ty.Optional[str] = None

    @classmethod
    def get_record_id(cls,
                      config_name: str,
                      approach_name: str,
                      n_sampling: int,
                      tau_sequence: ty.List[float],
                      _sentence_id: str,
                      trajectory_rule: str,
                      trajectory_rule_smoothing: str,
                      trajectory_rule_smoothing_window: ty.Optional[int]) -> str:
        str_tau_sequence = json.dumps(tau_sequence)
        _db_record = f'{config_name}/{approach_name}/{n_sampling}/{trajectory_rule}/{trajectory_rule_smoothing}/{trajectory_rule_smoothing_window}/{str_tau_sequence}/{_sentence_id}'
        return _db_record
    
    def __post_init__(self):
        if isinstance(self.tau_sequence, str):
            seq_tau_sequence = json.loads(self.tau_sequence)
        elif isinstance(self.tau_sequence, list):
            seq_tau_sequence = self.tau_sequence
        else:
            raise ValueError(f"Unknown type for tau_sequence: {type(self.tau_sequence)}")
        # end if

        self.record_id = self.get_record_id(
            self.config_name, 
            self.approach_name, 
            self.n_sampling,
            seq_tau_sequence,
            self.sentence_id,
            trajectory_rule=self.trajectory_rule,
            trajectory_rule_smoothing=self.trajectory_rule_smoothing,
            trajectory_rule_smoothing_window=self.trajectory_rule_smoothing_window)


@dataclasses.dataclass
class DbTableRecordProposalMmdFlaggerTrajectoryVer2:

    config_name: str
    approach_name: str

    # fields of dataset attributes.
    dataset_name: str
    sentence_id: str
    source_language_code: str
    target_language_code: str

    # fields of computing distances.
    n_sampling: int
    tau_sequence: str  # Json field
    embedding_option: str

    # fields of trajectory rule.
    trajectory_rule: str
    trajectory_rule_smoothing: str
    trajectory_rule_smoothing_window: ty.Optional[int]
    trajectory_rule_options_json: str

    # fields of flagging.
    flagging_label: bool
    flagging_argument_json: str

    # fields of database, primary key.
    record_id: ty.Optional[str] = None

    @classmethod
    def get_record_id(cls,
                      config_name: str,
                      approach_name: str,
                      n_sampling: int,
                      embedding_option: str,
                      tau_sequence: ty.List[float],
                      _sentence_id: str,
                      trajectory_rule: str,
                      trajectory_rule_smoothing: str,
                      trajectory_rule_smoothing_window: ty.Optional[int]) -> str:
        str_tau_sequence = json.dumps(tau_sequence)
        _db_record = f'{config_name}/{approach_name}/{n_sampling}/{embedding_option}/{trajectory_rule}/{trajectory_rule_smoothing}/{trajectory_rule_smoothing_window}/{str_tau_sequence}/{_sentence_id}'
        return _db_record
    
    def __post_init__(self):
        if isinstance(self.tau_sequence, str):
            seq_tau_sequence = json.loads(self.tau_sequence)
        elif isinstance(self.tau_sequence, list):
            seq_tau_sequence = self.tau_sequence
        else:
            raise ValueError(f"Unknown type for tau_sequence: {type(self.tau_sequence)}")
        # end if

        self.record_id = self.get_record_id(
            config_name=self.config_name, 
            approach_name=self.approach_name, 
            n_sampling=self.n_sampling,
            embedding_option=self.embedding_option,
            tau_sequence=seq_tau_sequence,
            _sentence_id=self.sentence_id,
            trajectory_rule=self.trajectory_rule,
            trajectory_rule_smoothing=self.trajectory_rule_smoothing,
            trajectory_rule_smoothing_window=self.trajectory_rule_smoothing_window)
