import typing as ty

import duckdb

from ..module_models.model_db_record_obj import TableTypes, RecordObjMmdFlaggerIntermed
# ---- DuckDB Utils -----
from ...utils.duck_db_helper import (
    reconstruct_sql_record_for_pydantic,
    generate_duckdb_schema,
)
from ..module_models import RecordObjMmdFlaggerIntermedTableUniqueId


class DbHandlerMMDFlaggerInterface:
    def __init__(self,
                 duck_db_connection: duckdb.DuckDBPyConnection) -> None:
        self.duck_db_connection = duck_db_connection
        self.init_tables()

    def init_tables(self):
        for _table_name in ty.get_args(TableTypes):
            table_scheme = generate_duckdb_schema(RecordObjMmdFlaggerIntermed, table_name=_table_name)
            self.duck_db_connection.execute(table_scheme)
        # end for

    def is_record_exist(self, 
                        global_identifier: ty.Union[str, RecordObjMmdFlaggerIntermedTableUniqueId],
                        table_name: TableTypes) -> bool:
        _c = self.duck_db_connection.cursor()
        _c.execute(f"SELECT count(*) FROM {table_name} WHERE global_unique_id = ?", [global_identifier])
        _r = _c.fetchone()
        if _r is None:
            return False
        elif _r[0] == 0:
            return False
        else:
            return True
    # end def


    def post_record(self, 
                    record: RecordObjMmdFlaggerIntermed,
                    target_table_name: TableTypes, 
                    is_check: bool = True):
        # Insert the DataFrame into DuckDB
        assert record is not None

        if is_check:
            assert record.global_unique_id is not None
            if self.is_record_exist(record.global_unique_id, table_name=target_table_name):
                return None
            else:
                pass
        # end if

        # record_df = pd.Series(record.model_dump()).to_frame().T
        _obj_dict = record.model_dump()
        
        _keys = ', '.join(_obj_dict.keys())
        _place_holder = ', '.join(['?'] * len(_obj_dict.keys()))
        _values = _obj_dict.values()
        
        query = f"INSERT INTO {target_table_name} ({_keys}) VALUES ({_place_holder})"
        self.duck_db_connection.execute(query, _values)
        self.duck_db_connection.commit()

    def fetch_record(self, 
                     global_identifier: str, 
                     target_table_name: TableTypes) -> ty.Optional[RecordObjMmdFlaggerIntermed]:
        """fetch record from the DB.
        """
        _sql = f"SELECT * FROM {target_table_name} WHERE global_unique_id = '{global_identifier}';"
        
        _c = self.duck_db_connection.cursor()
        _c.execute(_sql)

        _row, _cols = _c.fetchone(), _c.description
        _col_name = [_c[0] for _c in _cols]
        _col_dtype = [_c[1] for _c in _cols]

        if _row is None or len(_row) == 0:
            return None
        # end if

        pydantic_obj = reconstruct_sql_record_for_pydantic(_row, _col_name, _col_dtype, RecordObjMmdFlaggerIntermed)
            
        assert isinstance(pydantic_obj, RecordObjMmdFlaggerIntermed)
        return pydantic_obj




