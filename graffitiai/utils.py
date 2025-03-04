import ast
import numpy as np
import pandas as pd

__all__ = [
    'convert_conjecture_dicts',
    'is_list_string',
    'convert_list_string',
    'convert_list_to_array',
    'convert_and_no_pad',
    'convert_and_pad',
    'median_absolute_deviation',
    'compute_statistics',
    'expand_statistics',
    'linear_function_to_string',
    'format_keyword',
    'format_conjecture_keyword',
]

def convert_conjecture_dicts(conjecture_reps, target, hypothesis=None, default_bound_type='lower'):
    """
    Convert conjecture representations into a list of BoundConjecture objects.

    Parameters:
        conjecture_reps (dict or list): Either a dictionary whose keys are bound types (e.g., 'lower')
            and values are lists of conjecture dictionaries, or a list of conjecture dictionaries.
        target (str): The target column (e.g., 'radius').
        hypothesis (str, optional): An optional hypothesis (e.g., a boolean column name).
        default_bound_type (str): If conjecture_reps is a list, this bound type will be assigned to all entries.

    Returns:
        List[BoundConjecture]: A list of BoundConjecture objects created from the representations.
    """
    from graffitiai.base import BoundConjecture
    bound_conjectures = []

    if isinstance(conjecture_reps, dict):
        for bound_type, conj_list in conjecture_reps.items():
            for conj in conj_list:
                candidate_expr = conj.get('rhs_str')
                candidate_func = conj.get('func')
                complexity = conj.get('complexity')
                touch = conj.get('touch', None)

                bc = BoundConjecture(
                    target=target,
                    candidate_expr=candidate_expr,
                    candidate_func=candidate_func,
                    bound_type=bound_type,
                    hypothesis=hypothesis,
                    complexity=complexity
                )
                bc.touch = touch
                bound_conjectures.append(bc)
    elif isinstance(conjecture_reps, list):
        # Assume all entries are of the default bound type.
        for conj in conjecture_reps:
            candidate_expr = conj.get('rhs_str')
            candidate_func = conj.get('func')
            complexity = conj.get('complexity')
            touch = conj.get('touch', None)

            bc = BoundConjecture(
                target=target,
                candidate_expr=candidate_expr,
                candidate_func=candidate_func,
                bound_type=default_bound_type,
                hypothesis=hypothesis,
                complexity=complexity
            )
            bc.touch = touch
            bound_conjectures.append(bc)
    else:
        raise ValueError("conjecture_reps must be a dictionary or a list")

    return bound_conjectures

def is_list_string(x):
    """Return True if x is a string that can be parsed as a list or tuple."""
    try:
        val = ast.literal_eval(x)
        return isinstance(val, (list, tuple))
    except Exception:
        return False

def convert_list_string(x):
    """Convert a string representation of a list to an actual list.
       Returns None if conversion fails.
    """
    try:
        val = ast.literal_eval(x)
        if isinstance(val, (list, tuple)):
            return list(val)
    except Exception:
        pass
    return None

def convert_list_to_array(lst):
    """Convert a list to a numpy array if possible."""
    if is_list_string(lst):
        lst = convert_list_string(lst)
        return np.array(lst)
    elif isinstance(lst, list):
        return np.array(lst)
    elif isinstance(lst, np.ndarray):
        return lst
    else:
        return None

def convert_and_no_pad(data):
    """Convert a pandas series of lists to numpy arrays without padding."""
    data = data.apply(convert_list_string)
    return data.apply(convert_list_to_array)

def convert_and_pad(data, pad_value = 0):
    """Convert a pandas series of lists to numpy arrays and pad them
    with a specified value.
    """
    # data = data.apply(convert_list_string)
    max_len = data.apply(len).max()
    return data.apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant', constant_values=pad_value))

def median_absolute_deviation(lst):
    """Compute the Median Absolute Deviation (MAD)."""
    median = np.median(lst)
    abs_deviation = np.abs(lst - median)
    return np.median(abs_deviation)

def compute_statistics(lst):
    """Compute various statistics for a list."""
    data = {}
    data['length'] = len(lst)
    data['min'] = np.min(lst)
    data['max'] = np.max(lst)
    data['range'] = data['max'] - data['min']
    data['mean'] = np.mean(lst)
    data['median'] = np.median(lst)
    data['variance'] = np.var(lst)
    data['abs_dev'] = np.abs(lst - data['median'])
    data['std_dev'] = np.std(lst)
    data['median_absolute_deviation'] = median_absolute_deviation(lst)
    data['count_non_zero'] = np.count_nonzero(lst)
    data['count_zero'] = data['length'] - data['count_non_zero']

    # --- Zero-Centric Properties ---
    data['mostly_zeros'] = (data['count_zero'] / data['length'] >= 0.7)
    def max_contiguous_zeros(seq):
        max_count = count = 0
        for x in seq:
            if x == 0:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count
    max_zeros = max_contiguous_zeros(lst)
    data['zeros_clustered'] = (max_zeros >= 0.5 * data['count_zero']) if data['count_zero'] > 0 else False

    # --- Cumulative Sum Property ---
    def first_index_half_cumsum(seq):
        tot = sum(seq)
        if tot <= 0:
            return None
        half = tot / 2.0
        run = 0
        for idx, val in enumerate(seq):
            run += val
            if run >= half:
                return idx
        return None
    data['first_index_half_cumsum'] = first_index_half_cumsum(lst)

    # Even and odd counts
    data['count_even'] = sum(1 for x in lst if x % 2 == 0)
    data['count_odd'] = data['length'] - data['count_even']
    data['unique_count'] = len(set(lst))
    return data

def expand_statistics(column, df):
    """Expand a column of statistics into separate columns."""

    # Apply the function to each row and expand into separate columns
    stats_df = df[column].apply(compute_statistics).apply(pd.Series)
    stats_df.columns = [f"{col}({column})" for col in stats_df.columns]
    df = df.join(stats_df)
    return df

def linear_function_to_string(W_values, other_invariants, b_value):
    terms = []
    for coeff, var in zip(W_values, other_invariants):
        # Skip terms with zero coefficient.
        if coeff == 0:
            continue

        # Format coefficient: omit "1*" for 1, and "-1*" for -1.
        if coeff == 1:
            term = f"{var}"
        elif coeff == -1:
            term = f"-{var}"
        else:
            term = f"{coeff}*{var}"
        terms.append(term)

    # Add the constant term if it's non-zero.
    if b_value != 0 or not terms:
        terms.append(str(b_value))

    # Join terms with " + " and fix signs.
    result = " + ".join(terms)
    # Replace sequences like "+ -": "a + -b" becomes "a - b"
    result = result.replace("+ -", "- ")
    return result

def format_keyword(keyword: str) -> str:
    # Replace underscores with spaces and capitalize each word.
    return keyword.title()

def format_conjecture_keyword(hypothesis: str, target: str, other_invariants: list) -> str:
    # Format the hypothesis, target, and other invariants as a keyword.
    keywords = []
    if hypothesis:
        keywords.append(format_keyword(hypothesis))
    keywords.append(format_keyword(target))
    for inv in other_invariants:
        keywords.append(format_keyword(inv))
    return keywords
