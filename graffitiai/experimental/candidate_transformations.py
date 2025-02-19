# candidate_transformations.py
import numpy as np

__all__ = [
    "floor_transform",
    "ceil_transform",
    "add_ratio_transform",
    "sub_ratio_transform",
    "multiply_ratio_transform",
    "sqrt_transform"
]

def floor_transform(df, base_func):
    """Apply floor to the output of a candidate function."""
    return np.floor(base_func(df))

def ceil_transform(df, base_func):
    """Apply ceil to the output of a candidate function."""
    return np.ceil(base_func(df))

def add_ratio_transform(df, base_func, ratio):
    """Add a constant ratio to the output of a candidate function."""
    return base_func(df) + float(ratio)

def sub_ratio_transform(df, base_func, ratio):
    """Subtract a constant ratio from the output of a candidate function."""
    return base_func(df) - float(ratio)

def multiply_ratio_transform(df, base_func, ratio):
    """Multiply the output of a candidate function by a constant ratio."""
    return float(ratio) * base_func(df)

def sqrt_transform(df, base_func):
    """Take the square root of the output of a candidate function."""
    return np.sqrt(base_func(df))
