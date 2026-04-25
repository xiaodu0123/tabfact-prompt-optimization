""" Custom tool functions for ReAct agent (introduced in https://aclanthology.org/2025.findings-naacl.244/)
"""
import json
from typing import Any, Optional

# Retrieval tools
def get_column_by_name(table_data: list[list[str]], column_name: str) -> str:
    """
    Retrieve the column values by column name.

    Args:
        table_data (list[list[str]]): The table data as a list of lists. First row is header.
        column_name (str): The name of the column to retrieve.
    
    """
    if not table_data or column_name not in table_data[0]:
        return f"Error: Column '{column_name}' does not exist in the table."
    column_index = table_data[0].index(column_name)
    column_data = [row[column_index] for row in table_data]
    
    return json.dumps(column_data, ensure_ascii=False)

def get_column_cell_value(column_data: list[str], row_index: int) -> str:
    """
    Retrieve the cell value from a column by row index.

    Args:
        column_data (list[str]): The column data as a list.
        row_index (int): The index of the row to retrieve.
    """
    if row_index < 0 or row_index >= len(column_data):
        return "Error: Row index out of bounds."
    
    return column_data[row_index]

def get_row_index_by_value(table_data: list[list[str]], value: str) -> str:
    """
    Retrieve the indices of the rows that contain the specified value in the first column.

    Args:
        table_data (list[list[str]]): The table data as a list of lists. First row is header.
        value (str): The value to search for in the first column.
    """
    if not table_data or len(table_data) < 2:
        return "Error: The table is empty or has no data rows."

    row_indices = []
    for index, row in enumerate(table_data, start=1):
        if row[0] == value:
            row_indices.append(index)

    if not row_indices:
        return "Error: Value not found in the first column."
    
    return json.dumps(row_indices)

def get_row_by_name(table_data: list[list[str]], row_name: str) -> str:
    """
    Retrieve the row values by row name from the first column.
    """
    if not table_data or len(table_data) < 2:
        return "Error: The table is empty or has no data rows."

    for row in table_data:
        if row[0] == row_name:
            return json.dumps(row, ensure_ascii=False)

    return "Error: Row name not found in the first column."

def get_column_by_index(table_data: list[list[str]], column_index: int) -> str:
    """
    Retrieve the column values by column index.

    Args:
        table_data (list[list[str]]): The table data as a list of lists. First row is header.
        column_index (int): The index of the column to retrieve.
    """
    if not table_data or column_index < 0 or column_index >= len(table_data[0]):
        return f"Error: Column index {column_index} is out of bounds."

    column_data = [row[column_index] for row in table_data]
    return json.dumps(column_data, ensure_ascii=False)

# Math tools
def equal_to(value1: str, value2: str) -> str:
    """
    Check if two values are equal.

    Args:
        value1 (str): The first value.
        value2 (str): The second value.
    """
    # Direct string comparison
    if value1 == value2:
        return "True"
    
    # Numeric comparison
    try:
        return str(float(value1) == float(value2))
    except ValueError:
        return "False"

def subtract(value1: float, value2: float) -> float:
    """
    Subtract two numeric values.

    Args:
        value1 (float): The first value.
        value2 (float): The second value.
    """
    return value1 - value2

def divide(value1: float, value2: float):
    """
    Divide two numeric values.

    Args:
        value1 (float): The numerator.
        value2 (float): The denominator.
    """
    if value2 == 0:
        return "Error: Division by zero."
    return value1 / value2

def add(value1: float, value2: float) -> float:
    """
    Add two numeric values.

    Args:
        value1 (float): The first value.
        value2 (float): The second value.
    """
    return value1 + value2

# Define tool combinations for experiments
top_three_tools = [get_column_by_name, get_column_cell_value, get_row_index_by_value]

top_five_tools = [
    get_column_by_name, 
    get_column_cell_value,
    get_row_index_by_value,
    get_row_by_name,
    equal_to
]

all_tools = [
    get_column_by_name,
    get_column_by_index,
    get_column_cell_value,
    get_row_index_by_value,
    get_row_by_name,
    equal_to,
    subtract,
    divide,
    add
]

TOOL_COMBINATIONS = {
    "top3_tools": top_three_tools,
    "top5_tools": top_five_tools,
    "all_tools": all_tools,
    "get_column_by_name": [get_column_by_name],
    "get_column_cell_value": [get_column_cell_value],
    "get_row_index_by_value": [get_row_index_by_value],
    "get_row_by_name": [get_row_by_name],
    "equal_to": [equal_to]   
}