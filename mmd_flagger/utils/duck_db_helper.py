import typing as ty
import json
import inspect
import time
import random
import logging
import duckdb

from pydantic import BaseModel
from duckdb.sqltypes import DuckDBPyType

logger = logging.getLogger(__name__)

def get_duckdb_connection(db_path: str, read_only: bool = False, max_retries: int = 10) -> duckdb.DuckDBPyConnection:
    """Helper to get a DuckDB connection with retries for locking issues."""
    retry_count = 0
    base_delay = 0.1
    effective_read_only = False  # Start by trying read-write to avoid mixing configs inside same process
    
    while retry_count < max_retries:
        try:
            return duckdb.connect(db_path, read_only=effective_read_only)
        except (duckdb.IOException, duckdb.ConnectionException) as e:
            err_msg = str(e).lower()
            if "different configuration" in err_msg:
                # Fallback to whatever mode is already active (usually read-only)
                effective_read_only = not effective_read_only
                continue
            
            if "lock" in err_msg or "used by another process" in err_msg:
                if read_only and not effective_read_only:
                    effective_read_only = True
                    continue
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to acquire DuckDB lock after {max_retries} retries: {db_path}")
                    raise e
                # Exponential backoff with jitter
                delay = base_delay * (2 ** retry_count) + random.uniform(0, 0.1)
                logger.warning(f"DuckDB lock conflict for {db_path}. Retrying in {delay:.2f}s... (Attempt {retry_count}/{max_retries})")
                time.sleep(delay)
            elif "does not exist" in err_msg and read_only:
                # Re-raise so the caller can handle missing file in read-only mode
                raise e
            else:
                # Unexpected error
                raise e
    
    # Should not reach here
    return duckdb.connect(db_path, read_only=read_only)

def reconstruct_sql_record_for_pydantic(
    row_tuple: ty.Optional[ty.Tuple[ty.Any,...]], 
    column_names: ty.List[str],
    column_dtypes: ty.List[DuckDBPyType], 
    model_class: ty.Type[BaseModel]) -> ty.Optional[BaseModel]:
    """
    Takes a raw SQL row (tuple) and column names, deserializes JSON fields, 
    and instantiates the target Pydantic BaseModel object.

    Args:
        row_tuple: A single row fetched from DuckDB (e.g., via cursor.fetchone()).
        column_names: The names of the columns corresponding to the row_tuple.
        model_class: The target Pydantic BaseModel class (e.g., RecordHallucinationEvalDataset).

    Returns:
        A validated instance of the Pydantic model.
    """
    if row_tuple is None:
        return None
        
    # 1. Combine column names and tuple values into a basic dict
    raw_record_dict = dict(zip(column_names, row_tuple))
    dict_c_name2d_type = dict(zip(column_names, column_dtypes))

    # 2. Identify and deserialize JSON fields based on the model's structure
    reconstructed_data = {}
    
    # Iterate through the fields of the TARGET Pydantic model
    for field_name, field_info in model_class.model_fields.items():
        value = raw_record_dict.get(field_name)
        _column_type: DuckDBPyType = dict_c_name2d_type[field_name]

        if _column_type == 'JSON':
            if value is None:
                value = None
            else:
                assert isinstance(value, str), f'field_name={field_name} is specified to JSON type. But, the saved type is {value}. It must be str type.'
                value = json.loads(value)
            # end if
        # end if
        
        # Default: Assign the raw value (works for simple types like int, float, str)
        reconstructed_data[field_name] = value

    # 3. Final Validation and Instantiation
    # Pydantic takes the reconstructed dictionary and performs all type casting and validation.
    return model_class.model_validate(reconstructed_data)


def is_str_included(type_hint: ty.Any) -> bool:
    """
    Checks if a type hint (including Unions) contains the string type.

    True cases,
    >> ty.Union[str, float, int]

    False case,
    >> ty.Union[ty.List[str], ty.Dict[str, str]]
    """
    # 1. Handle simple type first
    if type_hint is str:
        return True

    # 2. Check the origin (e.g., if it's a Union or List)
    origin = ty.get_origin(type_hint)

    # 3. If it's a Union, get the components (args)
    if origin is ty.Union:
        args = ty.get_args(type_hint)
        
        # Check if the 'str' type is among the components
        if str in args:
            return True
        # end if
    return False


def map_type_to_sql(python_type: ty.Any) -> str:
    """Maps Python/Pydantic types to suitable DuckDB SQL types."""
    # Handle optional types (typing.Optional[T] wraps T)
    if ty.get_origin(python_type) is ty.Union:
        args = ty.get_args(python_type)
        # Assuming Optional[T] is Union[T, NoneType]
        if type(None) in args:
            # Recursively check the inner type
            base_type = next(arg for arg in args if arg is not type(None))
            # DuckDB fields are NULLABLE by default, so we just return the base type map
            return map_type_to_sql(base_type)

    # 1. Direct Python Type Mapping
    is_union_contain_str = is_str_included(python_type)
    if python_type in (str, ty.Literal, ty.Optional[str]) or is_union_contain_str:
        return "VARCHAR"
    if python_type in (int, ty.Optional[int]):
        return "BIGINT"
    if python_type in (float, ty.Optional[float]):
        return "DOUBLE"
    if python_type in (bool, ty.Optional[bool]):
        return "BOOLEAN"
    if python_type in (bytes,):
        return "BLOB"

    # 2. Complex Pydantic/Typing Objects Mapping
    # DuckDB handles JSON structures natively for fast read/write,
    # so we convert all dicts and lists to JSON strings (VARCHAR) in the table.
    if python_type in (ty.Dict, ty.Any, ty.Dict[str, ty.Any], ty.Optional[ty.Dict[str, ty.Any]]):
        return "JSON" # Will store as JSON string

    if ty.get_origin(python_type) is list or ty.get_origin(python_type) is ty.List:
        return "JSON" # Will store as JSON string
        
    # 3. Nested Pydantic Models (The Configuration)
    if inspect.isclass(python_type) and issubclass(python_type, BaseModel):
        return "JSON" # Will store as JSON string

    return "VARCHAR" # Default fallback for custom or complex types


def generate_duckdb_schema(model: ty.Type[BaseModel],  
                           table_name: str) -> str:
    #    schema_name: str = "llm_decoding_comparison",
    """
    Automatically generates a DuckDB CREATE TABLE SQL statement from a Pydantic model.
    All nested structures (Dicts, Lists, other BaseModels) are stored as JSON (VARCHAR).
    """
    column_definitions = []

    # Iterate over the fields defined in the Pydantic model
    for field_name, field in model.model_fields.items():
        # Get the Python type hint
        python_type = field.annotation
        
        # Determine the SQL type
        sql_type = map_type_to_sql(python_type)
        
        # Check if the field is marked as a primary key in json_schema_extra
        is_primary = False
        if field.json_schema_extra and isinstance(field.json_schema_extra, dict):
            is_primary = field.json_schema_extra.get("is_primary_key", False)
        
        pk_constraint = "PRIMARY KEY" if is_primary else ""

        # Determine if the field is NOT NULL
        # NOTE: Pydantic fields without Optional are implicitly required (NOT NULL)
        is_required = field.is_required
        null_constraint = "NOT NULL" if is_required and "Optional" not in str(python_type) and not is_primary else ""
        
        # Create the column definition string
        column_def = f"    {field_name} {sql_type} {pk_constraint} {null_constraint}".replace("  ", " ").strip()
        column_definitions.append(column_def)
    # end for

    # Join all column definitions
    columns_sql = ",\n".join(column_definitions)

    # Construct the final SQL statement
    sql_statement = f"""
    -- SQL Schema generated automatically from Pydantic model {model.__name__}
    CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"""


    return sql_statement