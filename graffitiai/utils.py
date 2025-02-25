import pulp
import math
import ast
import numpy as np
import pandas as pd
from fractions import Fraction
from itertools import combinations


__all__ = [
    'make_upper_linear_conjecture',
    'make_lower_linear_conjecture',
    'make_all_linear_conjectures',
    'make_all_linear_conjectures_range',
    'hazel_heuristic',
    'morgan_heuristic',
    'weak_smokey',
    'filter_false_conjectures',
    'strong_smokey',
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
    'filter_upper_candidates',
    'filter_lower_candidates',
]

def make_upper_linear_conjecture(
        df,
        target_invariant,
        other_invariants,
        hyp="object",
        b_upper_bound=None,
        b_lower_bound=None,
        W_upper_bound=10,
        W_lower_bound=-10,
    ):
    """
    Constructs an upper-bound conjecture: target_invariant <= candidate,
    by solving an LP that finds weights (W) and an intercept (b) so that, for all rows,
    candidate >= target. Returns a BoundConjecture instance (or None if infeasible).
    """
    from graffitiai.base import BoundConjecture
    pulp.LpSolverDefault.msg = 0

    # Filter data by the hypothesis condition.
    df_filtered = df[df[hyp] == True]
    true_objects = df_filtered["name"].tolist()

    complexity = len(other_invariants)
    # For upper bound: choose the row having maximum target for each combination of other invariants.
    extreme_points = df_filtered.loc[df_filtered.groupby(other_invariants)[target_invariant].idxmax()]

    # Extract data.
    X = [extreme_points[other].tolist() for other in other_invariants]
    Y = extreme_points[target_invariant].tolist()
    num_instances = len(Y)

    # Define LP variables.
    W = [pulp.LpVariable(f"w_{i+1}", W_lower_bound, W_upper_bound) for i in range(complexity)]
    b = pulp.LpVariable("b")
    W_abs = [pulp.LpVariable(f"W_abs_{i+1}", lowBound=0) for i in range(complexity)]

    # Initialize the LP.
    prob = pulp.LpProblem("Generate_Upper_Bound_Conjecture", pulp.LpMinimize)

    # Objective: minimize the total slack.
    prob += pulp.lpSum([
        (pulp.lpSum(W[i] * X[i][j] for i in range(complexity)) + b - Y[j])
        for j in range(num_instances)
    ])

    # Constraints: candidate >= Y[j] for each instance.
    for j in range(num_instances):
        prob += pulp.lpSum([W[i] * X[i][j] for i in range(complexity)]) + b >= Y[j]

    # Enforce absolute value constraints.
    for i in range(complexity):
        prob += W_abs[i] >= W[i]
        prob += W_abs[i] >= -W[i]
    prob += pulp.lpSum(W_abs) >= 1e-6  # at least one nonzero W

    # Optionally set bounds on b.
    if b_upper_bound is not None:
        prob += b <= b_upper_bound
    if b_lower_bound is not None:
        prob += b >= b_lower_bound

    prob.solve()

    if prob.status != 1:
        return None

    W_values = [w.varValue for w in W]
    b_value = b.varValue

    if any(not isinstance(w, (int, float)) for w in W_values) or any(math.isinf(x) for x in W_values):
        return None
    if all(abs(x) < 1e-6 for x in W_values):
        return None

    # Convert values to fractions.
    W_values = [Fraction(w).limit_denominator(10) for w in W_values]
    b_value = Fraction(b_value).limit_denominator(10)

    # Compute "sharp" instances (rows where equality holds).
    X_true = [df_filtered[other].tolist() for other in other_invariants]
    Y_true = df_filtered[target_invariant].tolist()
    sharp_instances = {
        true_objects[j]
        for j in range(len(Y_true))
        if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
    }
    touch_number = len(sharp_instances)

    # Build the candidate expression string.
    candidate_expr = " + ".join([f"{W_values[i]}*({other_invariants[i]})" for i in range(complexity)])
    if b_value >= 0:
        candidate_expr += f" + {b_value}"
    else:
        candidate_expr += f" - {abs(b_value)}"

    # Create the candidate function.
    # (Convert fractions to float for numerical evaluation.)
    candidate_func = lambda df, W_vals=W_values, cols=other_invariants, b_val=b_value: \
        sum(float(W_vals[i]) * df[cols[i]] for i in range(len(cols))) + float(b_val)

    # Create a BoundConjecture instance with bound_type "upper" (target <= candidate).
    bc = BoundConjecture(
        target=target_invariant,
        candidate_expr=candidate_expr,
        candidate_func=candidate_func,
        bound_type="upper",
        hypothesis=hyp,
        complexity=complexity
    )
    # Set the computed "touch" value.
    bc.touch = touch_number
    return bc


def make_lower_linear_conjecture(
        df,
        target_invariant,
        other_invariants,
        hyp="object",
        b_upper_bound=None,
        b_lower_bound=None,
        W_upper_bound=10,
        W_lower_bound=-10,
    ):
    """
    Constructs a lower-bound conjecture: target_invariant >= candidate,
    by solving an LP that finds weights (W) and an intercept (b) so that, for all rows,
    candidate <= target. Returns a BoundConjecture instance (or None if infeasible).
    """
    from graffitiai.base import BoundConjecture
    pulp.LpSolverDefault.msg = 0

    # Filter data by the hypothesis condition.
    df_filtered = df[df[hyp] == True]
    true_objects = df_filtered["name"].tolist()

    complexity = len(other_invariants)
    # For lower bound: choose the row having minimum target for each combination.
    extreme_points = df_filtered.loc[df_filtered.groupby(other_invariants)[target_invariant].idxmin()]

    X = [extreme_points[other].tolist() for other in other_invariants]
    Y = extreme_points[target_invariant].tolist()
    num_instances = len(Y)

    # Define LP variables.
    W = [pulp.LpVariable(f"w_{i+1}", W_lower_bound, W_upper_bound) for i in range(complexity)]
    b = pulp.LpVariable("b")
    W_abs = [pulp.LpVariable(f"W_abs_{i+1}", lowBound=0) for i in range(complexity)]

    # Initialize the LP for maximization.
    prob = pulp.LpProblem("Generate_Lower_Bound_Conjecture", pulp.LpMaximize)

    # Objective: maximize the total slack.
    prob += pulp.lpSum([
        (pulp.lpSum(W[i] * X[i][j] for i in range(complexity)) + b - Y[j])
        for j in range(num_instances)
    ])

    # Constraints: candidate <= Y[j] for each instance.
    for j in range(num_instances):
        prob += pulp.lpSum([W[i] * X[i][j] for i in range(complexity)]) + b <= Y[j]

    for i in range(complexity):
        prob += W_abs[i] >= W[i]
        prob += W_abs[i] >= -W[i]
    prob += pulp.lpSum(W_abs) >= 1e-6

    if b_upper_bound is not None:
        prob += b <= b_upper_bound
    if b_lower_bound is not None:
        prob += b >= b_lower_bound

    prob.solve()

    if prob.status != 1:
        return None

    W_values = [w.varValue for w in W]
    b_value = b.varValue

    if any(not isinstance(w, (int, float)) for w in W_values) or any(math.isinf(x) for x in W_values):
        return None
    if all(abs(x) < 1e-6 for x in W_values):
        return None

    W_values = [Fraction(w).limit_denominator(10) for w in W_values]
    b_value = Fraction(b_value).limit_denominator(10)

    X_true = [df_filtered[other].tolist() for other in other_invariants]
    Y_true = df_filtered[target_invariant].tolist()
    sharp_instances = {
        true_objects[j]
        for j in range(len(Y_true))
        if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
    }
    touch_number = len(sharp_instances)

    candidate_expr = " + ".join([f"{W_values[i]}*({other_invariants[i]})" for i in range(complexity)])
    if b_value >= 0:
        candidate_expr += f" + {b_value}"
    else:
        candidate_expr += f" - {abs(b_value)}"

    candidate_func = lambda df, W_vals=W_values, cols=other_invariants, b_val=b_value: \
        sum(float(W_vals[i]) * df[cols[i]] for i in range(len(cols))) + float(b_val)

    # For lower bound, use bound_type "lower" (target >= candidate).
    bc = BoundConjecture(
        target=target_invariant,
        candidate_expr=candidate_expr,
        candidate_func=candidate_func,
        bound_type="lower",
        hypothesis=hyp,
        complexity=complexity
    )
    bc.touch = touch_number
    return bc


def make_all_linear_conjectures(
        df,
        target_invariant,
        other_invariants,
        properties,
        complexity=2,
        lower_b_max=None,
        upper_b_max=None,
        lower_b_min=None,
        upper_b_min=None,
    ):
    """
    Generate linear conjectures with a specified complexity (k-combinations of invariants).
    Returns two lists: (upper_conjectures, lower_conjectures) as BoundConjecture instances.
    """
    upper_conjectures = []
    lower_conjectures = []

    valid_invariants = [inv for inv in other_invariants if inv != target_invariant]

    for combo in combinations(valid_invariants, complexity):
        for prop in properties:
            upper_conj = make_upper_linear_conjecture(
                df,
                target_invariant,
                list(combo),
                hyp=prop,
                b_upper_bound=upper_b_max,
                b_lower_bound=upper_b_min
            )
            if upper_conj:
                upper_conjectures.append(upper_conj)

            lower_conj = make_lower_linear_conjecture(
                df,
                target_invariant,
                list(combo),
                hyp=prop,
                b_upper_bound=lower_b_max,
                b_lower_bound=lower_b_min
            )
            if lower_conj:
                lower_conjectures.append(lower_conj)

    return upper_conjectures, lower_conjectures


def make_all_linear_conjectures_range(
        df,
        target_invariant,
        other_invariants,
        properties,
        complexity_range=(1, 1),
        lower_b_max=None,
        upper_b_max=None,
        lower_b_min=None,
        upper_b_min=None,
        W_upper_bound=10,
        W_lower_bound=-10,
        progress_bar=None  # Accept an external progress bar
):
    """
    Generate linear conjectures over a range of complexities.
    Returns two lists: (upper_conjectures, lower_conjectures) as BoundConjecture instances.
    """
    upper_conjectures = []
    lower_conjectures = []

    valid_invariants = [inv for inv in other_invariants if inv != target_invariant]
    # Adjust range to be inclusive.
    lower, upper = complexity_range
    upper += 1

    for complexity in range(lower, upper):
        for combo in combinations(valid_invariants, complexity):
            for prop in properties:
                upper_conj = make_upper_linear_conjecture(
                    df, target_invariant, list(combo), hyp=prop,
                    b_upper_bound=upper_b_max, b_lower_bound=upper_b_min,
                    W_upper_bound=W_upper_bound, W_lower_bound=W_lower_bound,
                )
                if upper_conj:
                    upper_conjectures.append(upper_conj)

                lower_conj = make_lower_linear_conjecture(
                    df, target_invariant, list(combo), hyp=prop,
                    b_upper_bound=lower_b_max, b_lower_bound=lower_b_min,
                    W_upper_bound=W_upper_bound, W_lower_bound=W_lower_bound,
                )
                if lower_conj:
                    lower_conjectures.append(lower_conj)

                if progress_bar:
                    progress_bar.update(1)

    return upper_conjectures, lower_conjectures




def hazel_heuristic(conjectures, min_touch=0):
    """
    Filters and sorts a list of conjectures based on touch number.

    This heuristic:
    - Removes duplicate conjectures.
    - Removes conjectures that never attain equality (touch <= min_touch).
    - Sorts the remaining conjectures in descending order of touch number.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter and sort.
    min_touch : int, optional
        The minimum touch number required for a conjecture to be retained (default is 0).

    Returns
    -------
    list of Conjecture
        The sorted list of conjectures with the highest touch numbers.
    """
    # Remove duplicate conjectures.
    conjectures = list(set(conjectures))

    # Remove conjectures never attaining equality.
    conjectures = [conj for conj in conjectures if conj.touch > min_touch]

    # Sort the conjectures by touch number.
    conjectures.sort(key=lambda x: x.touch, reverse=True)

    # Return the sorted list of conjectures.
    return conjectures


def morgan_heuristic(conjectures):
    """
    Removes redundant conjectures based on generality.

    A conjecture is considered redundant if another conjecture has the same conclusion
    and a more general hypothesis (i.e., its true_object_set is a superset of the redundant one).

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.

    Returns
    -------
    list of Conjecture
        A list with redundant conjectures removed.
    """
    new_conjectures = conjectures.copy()

    for conj_one in conjectures:
        for conj_two in new_conjectures.copy():  # Make a copy for safe removal
            # Avoid comparing the conjecture with itself
            if conj_one != conj_two:
                # Check if conclusions are the same and conj_one's hypothesis is more general
                if conj_one.conclusion == conj_two.conclusion and conj_one.hypothesis > conj_two.hypothesis:
                    new_conjectures.remove(conj_two)  # Remove the less general conjecture (conj_two)

    return new_conjectures


def weak_smokey(conjectures):
    """
    Selects conjectures based on equality and distinct sharp objects.

    This heuristic:
    - Starts with the conjecture having the highest touch number.
    - Retains conjectures that either satisfy equality or introduce new sharp objects.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.

    Returns
    -------
    list of Conjecture
        A list of strong conjectures with distinct or new sharp objects.
    """
    # Start with the conjecture that has the highest touch number (first in the list).
    conj = conjectures[0]

    # Initialize the list of strong conjectures with the first conjecture.
    strong_conjectures = [conj]

    # Get the set of sharp objects (i.e., objects where the conjecture holds as equality) for the first conjecture.
    sharp_objects = conj.sharps

    # Iterate over the remaining conjectures in the list.
    for conj in conjectures[1:]:
        if conj.is_equal():
            strong_conjectures.append(conj)
            sharp_objects = sharp_objects.union(conj.sharps)
        else:
            # Check if the current conjecture shares the same sharp objects as any already selected strong conjecture.
            if any(conj.sharps.issuperset(known.sharps) for known in strong_conjectures):
                # If it does, add the current conjecture to the list of strong conjectures.
                strong_conjectures.append(conj)
                # Update the set of sharp objects to include the newly discovered sharp objects.
                sharp_objects = sharp_objects.union(conj.sharps)
            # Otherwise, check if the current conjecture introduces new sharp objects (objects where the conjecture holds).
            elif conj.sharps - sharp_objects != set():
                # If new sharp objects are found, add the conjecture to the list.
                strong_conjectures.append(conj)
                # Update the set of sharp objects to include the newly discovered sharp objects.
                sharp_objects = sharp_objects.union(conj.sharps)

    # Return the list of strong, non-redundant conjectures.
    return strong_conjectures


def strong_smokey(conjectures):
    """
    Selects conjectures that strongly subsume others based on sharp objects.

    This heuristic:
    - Starts with the conjecture having the highest touch number.
    - Retains conjectures whose sharp objects are supersets of previously selected conjectures.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.

    Returns
    -------
    list of Conjecture
        A list of conjectures with non-redundant, strongly subsuming sharp objects.
    """
    # Start with the conjecture that has the highest touch number (first in the list).
    conj = conjectures[0]

    # Initialize the list of strong conjectures with the first conjecture.
    strong_conjectures = [conj]

    # Get the set of sharp objects (i.e., objects where the conjecture holds as equality) for the first conjecture.
    sharp_objects = conj.sharps

    # Iterate over the remaining conjectures in the list.
    for conj in conjectures[1:]:
        if conj.is_equal():
            strong_conjectures.append(conj)
        else:
            # Check if the current conjecture set of sharp objects is a superset of any already selected strong conjecture.
            if any(conj.sharps.issuperset(known.sharps) for known in strong_conjectures):
                # If it does, add the current conjecture to the list of strong conjectures.
                strong_conjectures.append(conj)
                sharp_objects = sharp_objects.union(conj.sharps)

    # Return the list of strong, non-redundant conjectures.
    return strong_conjectures


def filter_false_conjectures(conjectures, df):
    """
    Filters conjectures to remove those with counterexamples in the provided data.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.
    df : pandas.DataFrame
        The DataFrame containing graph data.

    Returns
    -------
    list of Conjecture
        A list of conjectures with no counterexamples in the DataFrame.
    """
    new_conjectures = []
    for conj in conjectures:
        if conj.false_objects(df).empty:
            new_conjectures.append(conj)
    return new_conjectures



def strong_smokey(conjectures, df):
    """
    Given a list of BoundConjecture objects and a DataFrame (representing the graphs
    that satisfy the hypothesis), this heuristic removes any conjecture that is
    "dominated" by another—that is, if for every row the candidate function of one
    conjecture is no better than that of another (and is strictly worse in at least one row),
    then it is removed.

    For a 'lower' bound conjecture (target >= candidate), a conjecture A is dominated by
    conjecture B if for every row B.candidate >= A.candidate and in at least one row B.candidate > A.candidate.

    For an 'upper' bound conjecture (target <= candidate), A is dominated by B if for every row
    B.candidate <= A.candidate and in at least one row B.candidate < A.candidate.

    Parameters
    ----------
    conjectures : list of BoundConjecture
        The list of conjectures to filter.
    df : pandas.DataFrame
        The DataFrame on which to evaluate the candidate functions (typically, the rows that satisfy the hypothesis).

    Returns
    -------
    list of BoundConjecture
        The list of conjectures after removing those that are dominated.
    """
    strong = []
    n = len(conjectures)
    for i, conj_i in enumerate(conjectures):
        series_i = conj_i.candidate_func(df)
        dominated = False
        for j, conj_j in enumerate(conjectures):
            if i == j:
                continue
            series_j = conj_j.candidate_func(df)
            if conj_i.bound_type == 'lower':
                # For lower bounds: we want a candidate that is as high as possible.
                # If every value of candidate_j is greater than or equal to candidate_i,
                # and strictly greater for at least one row, then candidate_i is dominated.
                if (series_j >= series_i).all() and (series_j > series_i).any():
                    dominated = True
                    break
            else:
                # For upper bounds: we want a candidate that is as low as possible.
                if (series_j <= series_i).all() and (series_j < series_i).any():
                    dominated = True
                    break
        if not dominated:
            strong.append(conj_i)
    return strong


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
    return data

def expand_statistics(column, df):
    """Expand a column of statistics into separate columns."""

    # Apply the function to each row and expand into separate columns
    stats_df = df[column].apply(compute_statistics).apply(pd.Series)
    stats_df.columns = [f"{col}(p_vector)" for col in stats_df.columns]
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

def filter_upper_candidates(candidates, knowledge_table):
    """
    For each candidate upper-bound conjecture in candidates, evaluate its candidate function on the
    entire knowledge table (it internally filters by its hypothesis) and keep the candidate if there is
    at least one row where its value is strictly lower than every other candidate's value.
    """
    # Evaluate each candidate's function on the full table.
    cand_values = {cand: cand.candidate_func(knowledge_table) for cand in candidates}
    accepted = []
    # Assume that all candidate functions return a pandas Series with the same index.
    for cand in candidates:
        series = cand_values[cand]
        keep = False
        # Iterate row-by-row; if this candidate is strictly lower than all others on any row, we keep it.
        for idx in series.index:
            val = series.loc[idx]
            if all(val < cand_values[other].loc[idx] for other in candidates if other != cand):
                keep = True
                break
        if keep:
            accepted.append(cand)
    return accepted

def filter_lower_candidates(candidates, knowledge_table):
    """
    For each candidate lower-bound conjecture in candidates, evaluate its candidate function on the
    entire knowledge table and keep the candidate if there is at least one row where its value is strictly
    higher than every other candidate's value.
    """
    cand_values = {cand: cand.candidate_func(knowledge_table) for cand in candidates}
    accepted = []
    for cand in candidates:
        series = cand_values[cand]
        keep = False
        for idx in series.index:
            val = series.loc[idx]
            if all(val > cand_values[other].loc[idx] for other in candidates if other != cand):
                keep = True
                break
        if keep:
            accepted.append(cand)
    return accepted