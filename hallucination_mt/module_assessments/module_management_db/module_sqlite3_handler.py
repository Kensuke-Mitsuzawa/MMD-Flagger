import sqlite3
import typing as ty
import torch
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

"""A module for handling sqlite3 database operations."""


def map_python_type_to_sqlite(python_type):
    # Handle Optional types
    nullable = False
    # if ty.get_origin(python_type) is ty.Union:
    if hasattr(python_type, '__origin__') and python_type.__origin__ is ty.Union:    
        # args = ty.get_args(python_type)
        args = python_type.__args__
        # Check if it's Optional (Union[type, NoneType])
        if len(args) == 2 and type(None) in args:
            python_type = args[0] if args[1] is type(None) else args[1]
            nullable = True
        # end if
    # end if
            
    if python_type == str:
        sqlite_type = "TEXT"
    elif python_type == int:
        sqlite_type = "INTEGER"
    elif python_type == float:
        sqlite_type = "REAL"
    # -------------------------------------------------------------------------
    elif python_type == ty.List[int]:
        sqlite_type = "TEXT"
    elif python_type == ty.List[str]:
        sqlite_type = "TEXT"
    elif python_type == ty.List[float]:
        sqlite_type = "TEXT"
    elif python_type == ty.List[ty.Dict]:
        sqlite_type = "TEXT"
    elif python_type == ty.Dict:
        sqlite_type = "TEXT"
    # -------------------------------------------------------------------------
    elif python_type == bool:
        sqlite_type = "BOOLEAN"
    elif python_type == bytes:
        sqlite_type = "BLOB"
    elif python_type == torch.Tensor:
        sqlite_type = "BLOB"
    else:
        raise ValueError(f"Unsupported type: {python_type}")

    if nullable:
        sqlite_type += " NULL"
    else:
        sqlite_type += " NOT NULL"
    # end if

    return sqlite_type

def create_table_from_table_definition(conn: sqlite3.Connection, 
                                       table_record_class: ty.Any, 
                                       primary_key: ty.Optional[str] = None):
    table_name = table_record_class.__name__
    dict_name2type = ty.get_type_hints(table_record_class)
    
    columns = []
    for field_name, field_type in dict_name2type.items():
        sqlite_type = map_python_type_to_sqlite(field_type)
        column_def = f"{field_name} {sqlite_type}"
        if primary_key and field_name == primary_key:
            column_def += " PRIMARY KEY"
        columns.append(column_def)

    columns_str = ", ".join(columns)
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});"

    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    conn.commit()

# -------------------------------------------------------------------------


class DBHandlerExp(object):
    def __init__(self, 
                 db_path: Path):
        self.db_path = db_path
        self.conn: ty.Optional[sqlite3.Connection] = None
        self.conn = self.__init_db_conn()
    
    def __init_db_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def __del__(self):
        if self.conn:
            self.conn.close()
        
    def insert(self, table_name: str, data: ty.Dict):
        if not self.conn:
            self.conn = self.__init_db_conn()
        # end if
        
        cursor = self.conn.cursor()
        fields = data.keys()
        values = tuple(data.values())
        
        fields_str = ", ".join(fields)
        placeholders = ", ".join(["?" for _ in fields])
        
        sql = f"INSERT INTO {table_name} ({fields_str}) VALUES ({placeholders})"
        try:
            cursor.execute(sql, values)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logger.error(f"Error: {e}")
            raise sqlite3.IntegrityError(e)
        except Exception as e:
            raise ValueError(
                (f"Unknown error -> {e}.",
                 f"SQL-Query -> {sql}.",
                 f"Values -> {values}"))
        
    def insert_many(self, table_name: str, data_list: ty.List[ty.Dict]):
        if not self.conn:
            self.conn = self.__init_db_conn()
        # end if
        
        cursor = self.conn.cursor()
        fields = data_list[0].keys()
        placeholders = ", ".join(["?" for _ in fields])
        
        for data in data_list:
            values = tuple(data.values())
            sql = f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({placeholders})"
            try:
                cursor.execute(sql, values)
                self.conn.commit()
            except sqlite3.IntegrityError as e:
                logger.error(f"Error: {e}")
                raise sqlite3.IntegrityError(e)
            except Exception as e:
                breakpoint()
                raise ValueError(f"Unknown error -> {e}")
        # end for

    def get_record_key(self, 
                       table_name: str,
                       exp_key: str,
                       primary_key_field: str) -> ty.Optional[ty.Dict]:
        if not self.conn:
            self.conn = self.__init_db_conn()
        # end if
        
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} WHERE {primary_key_field}=?", (exp_key,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        else:
            return None

    def get_all(self, table_name: str) -> ty.List[ty.Any]:
        if not self.conn:
            self.conn = self.__init_db_conn()
        # end if
        
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        __rows = cursor.fetchall()
        rows = [dict(row) for row in __rows]
        return rows
    
    def get_all_keys(self, table_name: str, primary_key_field: str) -> ty.List[str]:
        if not self.conn:
            self.conn = self.__init_db_conn()
        # end if
        
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {primary_key_field} FROM {table_name}")
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    
    def is_record_exists(self, 
                         table_name: str, 
                         exp_key: str,
                         is_partially_search: bool,
                         primary_key: str = 'exp_key') -> bool:
        if not self.conn:
            self.conn = self.__init_db_conn()
        # end if
        
        cursor = self.conn.cursor()
        if is_partially_search:
            cursor.execute(f"SELECT count({primary_key}) FROM {table_name} WHERE {primary_key} LIKE ?", (f"{exp_key}%",))
        else:
            cursor.execute(f"SELECT count({primary_key}) FROM {table_name} WHERE {primary_key}=?", (exp_key,))
        # end if
        row = cursor.fetchone()
        return True if row[0] > 0 else False