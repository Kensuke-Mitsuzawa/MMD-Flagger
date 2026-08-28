import typing as ty
from pathlib import Path
import duckdb
import pickle
import dataclasses
import hashlib
import logging

from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd

import torch

from mmd_tst_variable_detector.mmd_estimator import QuadraticMmdEstimator, KernelMatrixObject

from .module_kernels.string_based_gaussian_kernel import BaseStringBasedKernels

from ..utils import utils_gpu_status
from ..utils.duck_db_helper import get_duckdb_connection

# module: cache DB.
from .module_models.model_db_record_obj import (
    TableTypes, 
    RecordObjMmdFlaggerIntermed,
    MmdEstimatorConfig)
from .module_models.model_sample_set_container import SampleSet
# module: the main algorithm of MMD-flagger.
from .module_mmd_flagger.codebase_ver3_1 import (
    MmdErrorFlaggerTrajectoryVer3,
    MmdErrorFlaggerTrajectoryVer3StringBased,
    MmdErrorFlagResultVer3
)
# module: measure scores about the MMD-flagger results.
from .module_mmd_flagger.module_scoring import (
    score_matrix_outlier,
    score_trajectory,
)
from .module_utils.db_handler_backend import DbHandlerMMDFlaggerInterface

logger = logging.getLogger(__name__)

PossibleMetricsMMDFlaggerInterface = ty.Union[score_matrix_outlier.PossibleMetricsMmdVariability, score_trajectory.PossibleScoreMetrics]

class EstimateReturnObject(ty.NamedTuple):
    score: float
    mmd_matrix: score_matrix_outlier.MmdMatrix
    mmd_traj: MmdErrorFlagResultVer3

    def render_mmd_trajectory(self, ax_obj: Axes, name_feature: ty.Optional[str] = None) -> ty.Tuple[Axes]:
        """render the MMD-trajectory.
        
        x-axis: MMD-value.
        y-axis: temperature parameters.

        Return: tuple of (Axes, feature_name).
        """

        df_plot = pd.DataFrame({
            'x_vals': self.mmd_traj.tau_parameter,
            'y_vals': self.mmd_traj.mmd_distances,
        })

        sns.lineplot(data=df_plot, x='x_vals', y='y_vals', ax=ax_obj, label=name_feature)
        
        return ax_obj
# end class




class MMDFlagger(object):
    def __init__(self,
                 mmd_estimator: QuadraticMmdEstimator,
                 backend_db: ty.Optional[Path | duckdb.DuckDBPyConnection | DbHandlerMMDFlaggerInterface],
                 is_use_cache: bool = True) -> None:
        # ---- attributes ----
        self.mmd_flagger: ty.Union[MmdErrorFlaggerTrajectoryVer3, MmdErrorFlaggerTrajectoryVer3StringBased]
        # ---- attributes ----
        self.is_use_cache = is_use_cache

        if backend_db is None:
            self.db_handler = None
        else:
            if isinstance(backend_db, Path):
                self.db_handler = DbHandlerMMDFlaggerInterface(get_duckdb_connection(str(backend_db), read_only=False))
            elif isinstance(backend_db, duckdb.DuckDBPyConnection):
                self.db_handler = DbHandlerMMDFlaggerInterface(backend_db)
            elif isinstance(backend_db, DbHandlerMMDFlaggerInterface):
                self.db_handler = backend_db
            else:
                raise TypeError("unexpected type to `backend_db`.")
            # end if
        # end if

        self.mmd_estimator = mmd_estimator
        
        mmed_estimator_args = dataclasses.asdict(mmd_estimator.get_hyperparameters())
        kernel_args = mmd_estimator.kernel_obj.get_hyperparameters()
        self.mmd_estimator_config = MmdEstimatorConfig(
            mmd_estimator_class=type(self.mmd_estimator).__name__,
            kernel_class=type(self.mmd_estimator.kernel_obj).__name__,
            mmed_estimator_args=mmed_estimator_args,
            kernel_args=kernel_args
        )

        self._set_mmd_flagger()

    # ---- semi-private methods ----

    def _set_mmd_flagger(self):
        if isinstance(self.mmd_estimator.kernel_obj, BaseStringBasedKernels):
            self.mmd_flagger = MmdErrorFlaggerTrajectoryVer3StringBased(self.mmd_estimator)
        else:
            self.mmd_flagger = MmdErrorFlaggerTrajectoryVer3(self.mmd_estimator)
        # end if

    # ---- semi-private methods ----

    # ---- DB operation related ----

    def _get_unique_db_key(self,
                           Y_hyp: ty.Optional[SampleSet], 
                           Y_sto: ty.List[SampleSet]) -> str:
        temperature_sequences = [o.temperature_parameter.as_float() for o in Y_sto]  # type: ignore
        u_id = RecordObjMmdFlaggerIntermed.get_unique_id(
            unique_id_y_hyp=None if Y_hyp is None else Y_hyp.get_unique_id(), 
            unique_ids_y_sto=[o.get_unique_id() for o in Y_sto], 
            temperature_sequences=temperature_sequences, 
            mmd_estimator_config=self.mmd_estimator_config.model_dump())

        return u_id

    def check_cash_db_record(self, 
                             Y_hyp: ty.Optional[SampleSet], 
                             Y_sto: ty.List[SampleSet], 
                             target_table: TableTypes) -> bool:
        """Check if the corresponding record exists"""
        if self.db_handler is None:
            return False
        else:
            u_id = self._get_unique_db_key(Y_hyp, Y_sto)
            is_exist = self.db_handler.is_record_exist(global_identifier=u_id, table_name=target_table)

            return is_exist
        
    def fetch_cache_db_record(self,
                             Y_hyp: ty.Optional[SampleSet], 
                             Y_sto: ty.List[SampleSet],
                             target_table: TableTypes) -> ty.Optional[score_matrix_outlier.MmdMatrix | KernelMatrixObject | MmdErrorFlagResultVer3]:
        if self.db_handler is None:
            return None

        u_id = self._get_unique_db_key(Y_hyp, Y_sto)
        record_obj = self.db_handler.fetch_record(u_id, target_table)
        if record_obj is None:
            return None
        # end if

        serial_obj: ty.Dict = pickle.loads(record_obj.blob_object)

        if target_table == 'mmd_matrix_temperature':
            obj = score_matrix_outlier.MmdMatrix(**serial_obj)

            return obj
        elif target_table == 'kernel_matrix':
            obj = KernelMatrixObject(**serial_obj)

            return obj
        elif target_table == 'mmd_trajectory':
            obj = MmdErrorFlagResultVer3(**serial_obj)
            
            return obj
        else:
            raise NotImplementedError(f'Unknown case when table_name = {target_table}')        
    
    def post_to_db(self,
                   Y_hyp: ty.Optional[SampleSet], 
                   Y_sto: ty.List[SampleSet],
                   generated_obj: score_matrix_outlier.MmdMatrix | MmdErrorFlagResultVer3 | KernelMatrixObject,
                   target_table: TableTypes) -> None:
        if self.db_handler is None:
            return None
        else:
            u_id = self._get_unique_db_key(Y_hyp, Y_sto)
            temperature_sequences = [o.temperature_parameter.as_float() for o in Y_sto]  # type: ignore

            if isinstance(generated_obj, score_matrix_outlier.MmdMatrix):
                obj_serial = generated_obj._asdict()
            elif isinstance(generated_obj, KernelMatrixObject):
                obj_serial = dataclasses.asdict(generated_obj)
            else:
                obj_serial = generated_obj.model_dump()
            # end if
            blob_bytes = pickle.dumps(obj_serial)
            pickle.loads(blob_bytes)  # doubkle-check

            if Y_hyp is None:
                feat_y_hyp = None
            else:
                try:
                    # embedding feature
                    feat_y_hyp = Y_hyp.get_embedding_samples()
                except Exception:
                    # text feature
                    feat_y_hyp = Y_hyp.get_text_samples()
                # end if
            hash_y_hyp = hashlib.sha256(pickle.dumps(feat_y_hyp)).hexdigest()
            # end if

            assert Y_sto is not None
            try:
                # embedding feature
                feat_y_sto = [_sto.get_embedding_samples() for _sto in Y_sto]
            except Exception:
                # text feature
                feat_y_sto = [_sto.get_text_samples() for _sto in Y_sto]
            # end if
            hash_y_sto = hashlib.sha256(pickle.dumps(feat_y_sto)).hexdigest()
            
            record = RecordObjMmdFlaggerIntermed(
                unique_id_y_hyp=None if Y_hyp is None else Y_hyp.get_unique_id(),
                unique_ids_y_sto=[o.get_unique_id() for o in Y_sto],
                hash_y_hyp=hash_y_hyp,
                hash_y_sto=hash_y_sto,
                temperature_sequences=temperature_sequences,
                mmd_estimator_config=self.mmd_estimator_config.model_dump(),
                table_type=target_table,
                blob_object=blob_bytes,
                global_unique_id=u_id
            )
            self.db_handler.post_record(record, target_table_name=target_table)


    # ---- DB operation related ----    

    def do_main_mmd_flagger(self,
                            Y_hyp: SampleSet, 
                            Y_sto: ty.List[SampleSet]) -> MmdErrorFlagResultVer3:
        """The method of calling the main algorithm of MMD-flagger."""
        if isinstance(self.mmd_flagger, MmdErrorFlaggerTrajectoryVer3StringBased):
            logger.debug(f"MMD-flagger (string-based) is called.")
            tau2stochastic_sequences = {o.temperature_parameter.as_float(): o.get_text_samples() for o in Y_sto}  # type: ignore
            res_mmd_flagger = self.mmd_flagger.flag_hallucination(
                hypothesis_sequences=Y_hyp.get_text_samples(),
                tau2stochastic_sequences=tau2stochastic_sequences,
                is_add_kernel_matrix_object=True)
        else:
            logger.debug(f"MMD-flagger (embedding-based) is called.")
            assert isinstance(self.mmd_flagger, MmdErrorFlaggerTrajectoryVer3)
            _seq_emb_hyp = torch.stack(Y_hyp.get_embedding_samples(), dim=0)
            tau2processed_embedding_samples = {o.temperature_parameter.as_float(): torch.stack(o.get_embedding_samples(), dim=0) for o in Y_sto}  # type: ignore

            # ---- logging block ----
            logger.debug(f"Hypothesis embedding shape: {_seq_emb_hyp.shape}")
            for _tau, _seq_emb_sto in tau2processed_embedding_samples.items():
                logger.debug(f"Stochastic embedding shape: {_seq_emb_sto.shape}")
            # end for
            # ---- logging block ----

            res_mmd_flagger = self.mmd_flagger.flag_hallucination(
                processed_embedding_hypothesis=_seq_emb_hyp,
                tau2processed_embedding_samples=tau2processed_embedding_samples,
                is_add_kernel_matrix_object=True)
        # end if
        return res_mmd_flagger

    
    def calculate_mmd_trajectory(self,
                                 Y_hyp: SampleSet, 
                                 Y_sto: ty.List[SampleSet]) -> MmdErrorFlagResultVer3:
        is_exist = self.check_cash_db_record(Y_hyp, Y_sto, target_table='mmd_trajectory')
        if self.is_use_cache is False:
            mmd_traj = None
        else:
            if is_exist:
                mmd_traj: ty.Optional[MmdErrorFlagResultVer3] = self.fetch_cache_db_record(Y_hyp=Y_hyp, Y_sto=Y_sto, target_table='mmd_trajectory')  # type: ignore
            else:
                mmd_traj = None
            # end if
        # end if

        if mmd_traj is None:
            mmd_traj = self.do_main_mmd_flagger(Y_hyp, Y_sto)

            if self.db_handler is not None:
                self.post_to_db(Y_hyp=Y_hyp, Y_sto=Y_sto, generated_obj=mmd_traj, target_table='mmd_trajectory')
            # end if
        # end if

        return mmd_traj

    def calculate_mmd_matrix_temperature(self,
                                         Y_sto: ty.List[SampleSet]) -> score_matrix_outlier.MmdMatrix:
        if self.is_use_cache is False:
            mmd_matrix = None
        else:
            is_exist = self.check_cash_db_record(None, Y_sto, target_table='mmd_matrix_temperature')
            if is_exist:
                mmd_matrix: ty.Optional[score_matrix_outlier.MmdMatrix] = self.fetch_cache_db_record(Y_hyp=None, Y_sto=Y_sto, target_table='mmd_matrix_temperature')  # type: ignore
            else:
                mmd_matrix = None
            # end if
        # end if

        if mmd_matrix is None:
            device_no = utils_gpu_status.get_less_busy_cuda_device()

            if device_no is None:
                _torch_device_signiture = f"cpu"
            else:
                _torch_device_signiture = f'cuda:{device_no}'
            # end if

            mmd_matrix = score_matrix_outlier._compute_distance_inner_stochastic_samples(
                mmd_estimator=self.mmd_estimator,
                temperature2tensor=Y_sto,
                torch_device=torch.device(_torch_device_signiture)
            )

            if self.db_handler is not None:
                self.post_to_db(Y_hyp=None, Y_sto=Y_sto, generated_obj=mmd_matrix, target_table='mmd_matrix_temperature')
        # end if
        return mmd_matrix
    
    def score(
        self,
        mmd_traj: MmdErrorFlagResultVer3,
        mmd_matrix: score_matrix_outlier.MmdMatrix,
        scoring_method: PossibleMetricsMMDFlaggerInterface
        ) -> float:
        
        if scoring_method in ty.get_args(score_trajectory.PossibleScoreMetrics):
            return score_trajectory.score(
                mmd_traj, 
                scoring_method  # type: ignore
            )
        elif scoring_method in ty.get_args(score_matrix_outlier.PossibleMetricsMmdVariability):
            _mmd_traj = score_matrix_outlier.MmdTrajectory.from_MmdErrorFlagResultVer3(mmd_traj)
            return score_matrix_outlier.score_mmd_trajectory_variability(
                mmd_matrix=mmd_matrix,
                mmd_trajectory=_mmd_traj,
                metric_name=scoring_method  # type: ignore
            )
        else:
            raise NotImplementedError(f'No scoring method named {scoring_method}')
        # end if

    def estimate(self, 
                 Y_hyp: SampleSet, 
                 Y_sto: ty.List[SampleSet],
                 scoring_method: PossibleMetricsMMDFlaggerInterface) -> EstimateReturnObject:
        assert len(Y_sto) > 0, "`Y_sto` is empty. `Y_sto` must be given."
        mmd_traj = self.calculate_mmd_trajectory(Y_hyp=Y_hyp, Y_sto=Y_sto)
        mmd_matrix = self.calculate_mmd_matrix_temperature(Y_sto=Y_sto)

        obj = EstimateReturnObject(
            score=self.score(mmd_traj, mmd_matrix, scoring_method),
            mmd_matrix=mmd_matrix,
            mmd_traj=mmd_traj
        )
        return obj
