import typing as ty
from hallucination_mt.module_assessments.module_management_db.interface_ver3.module_db_record import (
    DbTableRecordProposalMmdFlaggerTrajectoryVer3
)



class _KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(ty.NamedTuple):
    tau_sequence: str
    n_sampling: int
    mode_vector_preprocess: str
    mode_target_embedding_layer: str
    mode_max_token_length_vector_concat: str
    mode_max_token_length: int
    args_kernel_options_json: str
    args_trajectory_options_json: str

    def generate_code_name(self) -> str:
        # Note: Do not use '/' for the separator. The key file may be used as the file path.
        def _group_sampling_options() -> str:
            return f"{self.tau_sequence}-{self.n_sampling}-{self.mode_target_embedding_layer}"


        def _group_name_vector_preprocess() -> str:
            return f'{self.mode_vector_preprocess}-{self.mode_max_token_length_vector_concat}-{self.mode_max_token_length}'

        # def _group_name_kernel_options():
        #     pass

        __key_name = f'{_group_sampling_options()}-{_group_name_vector_preprocess()}-{self.args_kernel_options_json}-{self.args_trajectory_options_json}'
        return __key_name


# sort and aggregate the records by the field `tau_sequence` & `embedding_option` & `n_sampling` & filter options.
def _func_aggregation_key_DbTableRecordProposalMmdFlaggerTrajectoryVer3(record: DbTableRecordProposalMmdFlaggerTrajectoryVer3) -> _KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3[str, ...]:
    # return record['tau_sequence'], record['n_sampling'], record['embedding_option'], record['trajectory_rule'], record['trajectory_rule_smoothing'], record['trajectory_rule_smoothing_window']
    return _KeyAggregationDbTableRecordProposalMmdFlaggerTrajectoryVer3(
        record.tau_sequence,
        record.n_sampling,
        record.mode_vector_preprocess,
        record.mode_target_embedding_layer,
        record.mode_max_token_length_vector_concat,
        record.mode_max_token_length,
        record.args_kernel_options_json,
        record.args_trajectory_options_json
    )    
# end def 