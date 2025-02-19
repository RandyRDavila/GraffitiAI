# candidate_operations.py
import numpy as np

def identity_func(df, col):
    """Return the specified column unchanged."""
    return df[col]

def square_func(df, col):
    """Return the square of the specified column."""
    return df[col] ** 2

def floor_func(df, col):
    """Return the floor of the specified column."""
    return np.floor(df[col])

def ceil_func(df, col):
    """Return the ceiling of the specified column."""
    return np.ceil(df[col])

def add_ratio_func(df, col, ratio):
    """Return the specified column with a constant ratio added."""
    return df[col] + float(ratio)

def sub_ratio_func(df, col, ratio):
    """Return the specified column with a constant ratio subtracted."""
    return df[col] - float(ratio)

def multiply_ratio_func(df, col, ratio):
    """Return the specified column multiplied by a constant ratio."""
    return float(ratio) * df[col]

def add_columns_func(df, col1, col2):
    """Return the sum of two columns."""
    return df[col1] + df[col2]

def subtract_columns_func(df, col1, col2):
    """Return the difference (col1 - col2)."""
    return df[col1] - df[col2]

def subtract_columns_func_reversed(df, col1, col2):
    """Return the difference (col2 - col1)."""
    return df[col2] - df[col1]

def multiply_columns_func(df, col1, col2):
    """Return the product of two columns."""
    return df[col1] * df[col2]

def max_columns_func(df, col1, col2):
    """Return the elementwise maximum of two columns."""
    return np.maximum(df[col1], df[col2])

def min_columns_func(df, col1, col2):
    """Return the elementwise minimum of two columns."""
    return np.minimum(df[col1], df[col2])

def abs_diff_columns_func(df, col1, col2):
    """Return the absolute difference of two columns."""
    return np.abs(df[col1] - df[col2])

def safe_division_func(df, col1, col2):
    """Return col1 divided by col2 (assumes col2 has no zeros)."""
    return df[col1] / df[col2]

def safe_division_func_reversed(df, col1, col2):
    """Return col2 divided by col1 (assumes col1 has no zeros)."""
    return df[col2] / df[col1]

def mod_func(df, col1, col2):
    """Return the modulus of col1 by col2."""
    return df[col1] % df[col2]

def sqrt_func(df, col):
    """Return the square root of the specified column."""
    return np.sqrt(df[col])
