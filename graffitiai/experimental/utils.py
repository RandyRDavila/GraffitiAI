from fractions import Fraction
import pandas as pd
import pulp
import math
import sympy as sp


__all__ = [
    'OptimistConjecture',
    'make_upper_linear_conjecture',
    'make_lower_linear_conjecture',
    'linear_function_to_string',
    'upper_bound_acceptance',
    'lower_bound_acceptance',
    'hazel_heuristic',
    'morgan_heuristic',
    'dalmatian_upper_bound_acceptance',
    'dalmatian_lower_bound_acceptance',
    'probability_distribution',
]

class OptimistConjecture:
    def __init__(
            self,
            hypothesis_function,
            target_function,
            other_function,
            operator,
            hypothesis_string=None,
            lhs_string=None,
            rhs_string=None,
            touch=None,
            true_objects=None,
            bound_callable=None,
            candidate_func=None,
            sharp_instances=None,
            rank=None,
        ):
        """
        Represents a mathematical conjecture of the form:
            If hypothesis_function, then target_function operator other_function.

        Parameters:
            hypothesis_function: callable that tests the hypothesis.
            target_function: callable representing the target expression.
            other_function: callable representing the other expression.
            operator: one of '<=', '>=', or '=='.
            touch (optional): attribute to store additional data (e.g. count of instances).
            true_objects (optional): set of objects for which the conjecture holds,
                                     used for comparing generality.
        """
        if operator not in ("<=", ">=", "=="):
            raise ValueError("Operator must be one of '<=', '>=', or '=='")
        self.hypothesis_function = hypothesis_function
        self.target_function = target_function
        self.other_function = other_function
        self.operator = operator
        self.hypothesis_string = hypothesis_string
        self.lhs_string = lhs_string
        self.rhs_string = rhs_string
        self.touch = touch  # additional attribute from BoundConjecture
        self.true_objects = true_objects  # may be used for is_less_general_than
        self._set_conclusion_string()
        self.bound_callable = bound_callable  # may be used for filtering
        self.candidate_func = candidate_func  # may be used for filtering
        self.sharp_instances = sharp_instances  # may be used for filtering
        self.rank = rank  # may be used for filtering

        # We use the string representation as our 'conclusion'
        self.conclusion = self.__repr__()

    def __call__(self, obj):
        """
        Evaluates the conjecture on an object.
        If the hypothesis is False, returns True (vacuous truth).
        Otherwise, evaluates the target and other functions and compares them.
        """
        if not self.hypothesis_function(obj):
            return True  # vacuously true if the hypothesis doesn't hold

        target_val = self.target_function(obj)
        other_val = self.other_function(obj)

        if self.operator == "<=":
            return target_val <= other_val
        elif self.operator == ">=":
            return target_val >= other_val
        elif self.operator == "==":
            return target_val == other_val

    def __repr__(self):
        # Attempt to use the function names; if not available, fall back to their repr.
        hyp_name = self.hypothesis_string if self.hypothesis_string else getattr(self.hypothesis_function, "__name__", repr(self.hypothesis_function))
        return f"For any {hyp_name}, {self.conclusion_string}"

    def _set_conclusion_string(self):
        target_name = self.lhs_string if self.lhs_string else getattr(self.target_function, "__name__", repr(self.target_function))
        other_name = self.rhs_string if self.rhs_string else getattr(self.other_function, "__name__", repr(self.other_function))
        self.conclusion_string = f"{target_name} {self.operator} {other_name}, with equality on {self.touch} instances."

    def __hash__(self):
        # Use the repr as the hash for consistency.
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, OptimistConjecture):
            return repr(self) == repr(other)
        return False

    def is_less_general_than(self, other):
        """
        Check if this conjecture is less general than another.

        A conjecture is considered less general if:
          1. They have the same conclusion (i.e. their string representation is equal), and
          2. Its true_objects (the set of objects for which it holds) is a strict subset
             of the other's true_objects.

        Raises a ValueError if either conjecture does not have a defined true_objects attribute.
        """
        if self.conclusion != other.conclusion:
            return False
        # If the hypothesis is absent for one, we cannot determine specificity.
        if self.hypothesis_function is None:
            return False
        if other.hypothesis_function is None:
            return True
        if self.true_objects is None or other.true_objects is None:
            raise ValueError("Both conjectures must have 'true_objects' defined to compare generality.")
        return self.true_objects < other.true_objects


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


def upper_bound_acceptance(conj1, conj2, df):
    return (conj1.candidate_func(df) == conj2.candidate_func(df)).all() or (conj1.candidate_func(df) > conj2.candidate_func(df)).any()

def lower_bound_acceptance(conj1, conj2, df):
    return (conj1.candidate_func(df) == conj2.candidate_func(df)).all() or (conj1.candidate_func(df) < conj2.candidate_func(df)).any()

def make_upper_linear_conjecture(
        df,
        hypothesis_function,
        target_function,
        other_functions,
        b_upper_bound=None,
        b_lower_bound=None,
        W_upper_bound=None,
        W_lower_bound=None,
    ):
    pulp.LpSolverDefault.msg = 0


    # Get the function names as strings.
    hyp = getattr(hypothesis_function, "__name__", repr(hypothesis_function))
    target_invariant = getattr(target_function, "__name__", repr(target_function))
    other_invariants = [getattr(other_function, "__name__", repr(other_function)) for other_function in other_functions]

    # Filter data for the hypothesis condition.
    df = df[df[hyp] == True]
    true_objects = df["name"].tolist()

    # Complexity is the number of other invariants being considered.
    complexity = len(other_invariants)

    # Preprocess the data to find the maximum Y for each X for the upper bound.
    extreme_points = df.loc[df.groupby(other_invariants)[target_invariant].idxmax()]

    # Extract the data for the upper bound problem.
    X = [extreme_points[other].tolist() for other in other_invariants]
    Y = extreme_points[target_invariant].tolist()
    num_instances = len(Y)

    # Define LP variables.
    W = [pulp.LpVariable(f"w_{i+1}", lowBound=W_lower_bound, upBound=W_upper_bound) for i in range(complexity)]
    b = pulp.LpVariable("b")

    # Auxiliary absolute value variables for W.
    W_abs = [pulp.LpVariable(f"W_abs_{i+1}", lowBound=0) for i in range(complexity)]

    # Initialize the LP.
    prob = pulp.LpProblem("Generate_Upper_Bound_Conjecture", pulp.LpMinimize)

    # Define the objective function.
    prob += pulp.lpSum(
        [(pulp.lpSum(W[i] * X[i][j] for i in range(complexity)) + b - Y[j]) for j in range(num_instances)]
    )

    # Define the LP constraints.
    for j in range(num_instances):
        prob += pulp.lpSum([W[i] * X[i][j] for i in range(complexity)]) + b >= Y[j]

    # Enforce W_abs[i] >= |W[i]|
    for i in range(complexity):
        prob += W_abs[i] >= W[i]
        prob += W_abs[i] >= -W[i]

    # Ensure at least one W value is nonzero.
    prob += pulp.lpSum(W_abs) >= 1e-6

    # Set bounds on the intercept term, if provided.
    if b_upper_bound is not None:
        prob += b <= b_upper_bound
    if b_lower_bound is not None:
        prob += b >= b_lower_bound

    # Solve the LP.
    prob.solve()

    # No feasible solution found.
    if prob.status != 1:
        return None

    # Extract variable values.
    W_values = [w.varValue for w in W]
    b_value = b.varValue

    # Handle invalid values.
    if any(not isinstance(w, (int, float)) for w in W_values) or any(math.isinf(x) for x in W_values):
        return None

    # Avoid solutions with all-zero slopes.
    if all(abs(x) < 1e-6 for x in W_values):
        return None

    # Convert to fractions for better interpretability.
    W_values = [Fraction(w).limit_denominator(10) for w in W_values]
    b_value = Fraction(b_value).limit_denominator(10)

    # Compute sharp instances.
    X_true = [df[other].tolist() for other in other_invariants]
    Y_true = df[target_invariant].tolist()

    # Collect sharp instances.
    sharp_instances = {
        true_objects[j]
        for j in range(len(Y_true))
        if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
    }
    touch = len(sharp_instances)

    if touch == 0:
        rank = 0
    else:
        # ---------------------------------------------
        # 1.  Build the matrix of invariant vectors
        # ---------------------------------------------
        sharp_df = df[df["name"].isin(sharp_instances)]

        # Choose the columns you want in the matrix.
        # • Most geometric approaches (e.g., facets of a convex hull)
        #   use only the 'other_invariants' coordinates.
        # • If you also need the target value append it:
        #     cols = other_invariants + [target_invariant]
        cols = other_invariants

        # ---------------------------------------------
        # 2.  Convert to a SymPy Matrix (exact arithmetic)
        # ---------------------------------------------

        # Helper to coerce Python Fractions / ints / floats to SymPy Rationals
        M = sp.Matrix(
            [[sp.nsimplify(x) for x in row]         # nsimplify preserves fractions exactly
            for row in sharp_df[cols].values]
        )

        # ---------------------------------------------
        # 3.  Compute the rank
        # ---------------------------------------------
        rank = M.rank()            # exact row-rank over ℚ

        # If you prefer a quick floating-point answer:
        # import numpy as np
        # rank = np.linalg.matrix_rank(sharp_df[cols].to_numpy(dtype=float))

        # ---------------------------------------------
        # 4.  (Optional) store it on the Conjecture object
        # ---------------------------------------------
        # dim = rank - 1               # rank-1 if you want affine dimension

    # Define a callable representative of the right-hand side of the inequality.
    rhs_function = lambda x: sum(W_values[i] * other_functions[i](x) for i in range(complexity)) + b_value

    # Build a string representation of the right-hand side of the inequality.
    rhs_string = linear_function_to_string(W_values, other_invariants, b_value)

    def candidate_function(df):
        df = df[df[hyp] == True]
        return sum(W_values[i] * df[other_invariants[i]] for i in range(complexity)) + b_value

    # bound_callable ensures that for all rows, the target invariant is at most the candidate function's value.
    def bound_callable(df):
        df = df[df[hyp] == True]
        return (df[target_invariant] <= candidate_function(df)).all()

    return OptimistConjecture(
        hypothesis_function,
        target_function,
        rhs_function,
        operator="<=",
        touch=touch,
        true_objects=set(true_objects),
        hypothesis_string=hyp,
        lhs_string=target_invariant,
        rhs_string=rhs_string,
        bound_callable=bound_callable,
        candidate_func=candidate_function,
        sharp_instances=sharp_instances,
        rank=rank,
    )

def make_lower_linear_conjecture(
        df,
        hypothesis_function,
        target_function,
        other_functions,
        b_upper_bound=None,
        b_lower_bound=None,
        W_upper_bound=None,    # Example numeric default
        W_lower_bound=None,   # Example numeric default
    ):
    pulp.LpSolverDefault.msg = 0

    # Get the function names as strings.
    hyp = getattr(hypothesis_function, "__name__", repr(hypothesis_function))
    target_invariant = getattr(target_function, "__name__", repr(target_function))
    other_invariants = [getattr(other_function, "__name__", repr(other_function)) for other_function in other_functions]

    # Filter data for the hypothesis condition
    df = df[df[hyp] == True]
    true_objects = df["name"].tolist()

    # Complexity is the number of other invariants being considered
    complexity = len(other_invariants)

    # Preprocess the data to find the minimum Y for each X for the lower bound
    extreme_points = df.loc[df.groupby(other_invariants)[target_invariant].idxmin()]

    # Extract the data for the lower bound problem
    X = [extreme_points[other].tolist() for other in other_invariants]
    Y = extreme_points[target_invariant].tolist()
    num_instances = len(Y)

    # Define LP variables
    W = [pulp.LpVariable(f"w_{i+1}", lowBound=W_lower_bound, upBound=W_upper_bound) for i in range(complexity)]
    b = pulp.LpVariable("b")

    # Auxiliary absolute value variables for W
    W_abs = [pulp.LpVariable(f"W_abs_{i+1}", lowBound=0) for i in range(complexity)]

    # Initialize the LP
    prob = pulp.LpProblem("Generate_Lower_Bound_Conjecture", pulp.LpMaximize)

    # Define the objective function
    prob += pulp.lpSum(
        [(pulp.lpSum(W[i] * X[i][j] for i in range(complexity)) + b - Y[j]) for j in range(num_instances)]
    )

    # Constraints: ensure the candidate is a valid lower bound
    for j in range(num_instances):
        prob += pulp.lpSum([W[i] * X[i][j] for i in range(complexity)]) + b <= Y[j]

    # Enforce W_abs[i] >= |W[i]|
    for i in range(complexity):
        prob += W_abs[i] >= W[i]
        prob += W_abs[i] >= -W[i]

    # Ensure at least one W value is nonzero
    prob += pulp.lpSum(W_abs) >= 1e-6

    # Set bounds on the intercept term, if provided
    if b_upper_bound is not None:
        prob += b <= b_upper_bound
    if b_lower_bound is not None:
        prob += b >= b_lower_bound

    # Solve the LP
    prob.solve()

    # No feasible solution found
    if prob.status != 1:
        return None

    # Extract variable values
    W_values = [w.varValue for w in W]
    b_value = b.varValue

    # Handle invalid values
    if any(not isinstance(w, (int, float)) for w in W_values) or any(math.isinf(x) for x in W_values):
        return None

    # Avoid solutions with all-zero slopes
    if all(abs(x) < 1e-6 for x in W_values):
        return None

    # Convert to fractions for better interpretability
    W_values = [Fraction(w).limit_denominator(10) for w in W_values]
    b_value = Fraction(b_value).limit_denominator(10)

    # Compute sharp instances
    X_true = [df[other].tolist() for other in other_invariants]
    Y_true = df[target_invariant].tolist()

    # Get sharp instances of the inequality.
    sharp_instances = {
        true_objects[j]
        for j in range(len(Y_true))
        if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
    }
    touch = len(sharp_instances)

    if touch == 0:
        rank = 0
    else:
        # ---------------------------------------------
        # 1.  Build the matrix of invariant vectors
        # ---------------------------------------------
        sharp_df = df[df["name"].isin(sharp_instances)]

        # Choose the columns you want in the matrix.
        # • Most geometric approaches (e.g., facets of a convex hull)
        #   use only the 'other_invariants' coordinates.
        # • If you also need the target value append it:
        #     cols = other_invariants + [target_invariant]
        cols = other_invariants

        # ---------------------------------------------
        # 2.  Convert to a SymPy Matrix (exact arithmetic)
        # ---------------------------------------------

        # Helper to coerce Python Fractions / ints / floats to SymPy Rationals
        M = sp.Matrix(
            [[sp.nsimplify(x) for x in row]         # nsimplify preserves fractions exactly
            for row in sharp_df[cols].values]
        )

        # ---------------------------------------------
        # 3.  Compute the rank
        # ---------------------------------------------
        rank = M.rank()            # exact row-rank over ℚ

        # If you prefer a quick floating-point answer:
        # import numpy as np
        # rank = np.linalg.matrix_rank(sharp_df[cols].to_numpy(dtype=float))

        # ---------------------------------------------
        # 4.  (Optional) store it on the Conjecture object
        # ---------------------------------------------
        # dim = rank - 1               # rank-1 if you want affine dimension

    # Define a callable for the right-hand side of the inequality.
    rhs_function = lambda x: sum(W_values[i] * other_functions[i](x) for i in range(complexity)) + b_value

    # Build a string representation of the right-hand side of the inequality.
    rhs_string = linear_function_to_string(W_values, other_invariants, b_value)

    hypothesis_string = getattr(hypothesis_function, "__name__", repr(hypothesis_function))
    target_string = getattr(target_function, "__name__", repr(target_function))

    def candidate_function(df):
        df = df[df[hyp] == True]
        return sum(W_values[i] * df[other_invariants[i]] for i in range(complexity)) + b_value

    def bound_callable(df):
        df = df[df[hyp] == True]
        return (df[target_invariant] >= candidate_function(df)).all()

    return OptimistConjecture(
        hypothesis_function,
        target_function,
        rhs_function,
        operator=">=",
        touch=touch,
        true_objects=set(true_objects),
        hypothesis_string=hyp,
        lhs_string=target_invariant,
        rhs_string=rhs_string,
        bound_callable=bound_callable,
        candidate_func=candidate_function,
        sharp_instances=sharp_instances,
        rank=rank,
    )


def hazel_heuristic(conjectures, df, min_touch = 0):
    """
    Remove duplicate conjectures and sort by touch.
    """
    conjectures = [conj for conj in set(conjectures) if conj.bound_callable(df)]
    conjectures = [conj for conj in conjectures if conj.touch >= min_touch]
    conjectures = sorted(conjectures, key=lambda x: x.touch, reverse=True)
    return conjectures

def morgan_heuristic(conjectures):
    """
    Given a list of BoundConjecture objects, returns only the most general conjectures.
    Conjectures are grouped by their conclusion, and within each group, a conjecture is
    removed if its hypothesis (i.e. its set of true_objects) is strictly more specific
    than that of another with the same conclusion.
    """
    # Group conjectures by their conclusion
    groups = {}
    for conj in conjectures:
        groups.setdefault(conj.conclusion_string, []).append(conj)

    filtered = []
    for group in groups.values():
        # In each group, keep only the "maximal" (most general) conjectures.
        # A conjecture is less general if there exists another in the group that it is
        # less general than (using its is_less_general_than method).
        most_general = []
        for conj in group:
            if not any(
                other is not conj and conj.is_less_general_than(other)
                for other in group
            ):
                most_general.append(conj)
        filtered.extend(most_general)

    return filtered



def dalmatian_upper_bound_acceptance(candidates, df, hypothesis):
    """
    For each candidate upper-bound conjecture in candidates, evaluate its candidate function on the
    knowledge table filtered by hypothesis and keep the candidate if there is at least one row where its
    value is strictly lower than every other candidate's value.
    """
    # Filter the DataFrame based on the hypothesis.
    df = df[df[hypothesis] == True]

    # Compute candidate values for each candidate and build a DataFrame.
    cand_values = {str(cand): cand.candidate_func(df) for cand in candidates}
    cand_df = pd.DataFrame(cand_values)

    accepted = []
    # For each candidate, compute the row-wise minimum of the other candidates.
    for cand in candidates:
        col_name = str(cand)
        # For each row, compute the minimum among all other candidates.
        other_min = cand_df.drop(columns=[col_name]).min(axis=1)
        # Create a boolean mask where the candidate's value is strictly less than the other candidates' minimum.
        mask = cand_df[col_name] < other_min
        # If there's at least one row where this candidate is strictly lower, accept it.
        if mask.any():
            accepted.append(cand)
    return accepted

def dalmatian_lower_bound_acceptance(candidates, df, hypothesis):
    """
    For each candidate lower-bound conjecture in candidates, evaluate its candidate function on the
    knowledge table filtered by hypothesis and keep the candidate if there is at least one row where its
    value is strictly higher than every other candidate's value.
    """
    # Filter the DataFrame based on the hypothesis.
    df = df[df[hypothesis] == True]

    # Compute candidate values for each candidate and build a DataFrame.
    cand_values = {str(cand): cand.candidate_func(df) for cand in candidates}
    cand_df = pd.DataFrame(cand_values)

    accepted = []
    # For each candidate, compute the row-wise maximum of the other candidates.
    for cand in candidates:
        col_name = str(cand)
        # For each row, compute the maximum among all other candidates.
        other_max = cand_df.drop(columns=[col_name]).max(axis=1)
        # Create a boolean mask where the candidate's value is strictly greater than the other candidates' maximum.
        mask = cand_df[col_name] > other_max
        # If there's at least one row where this candidate is strictly higher, accept it.
        if mask.any():
            accepted.append(cand)
    return accepted


# sampling for comparison helper function
def probability_distribution(optimist, hypothesis_function, target_function, num_features=4):
    import random
    import numpy as np

    target = getattr(target_function, "__name__", repr(target_function))
    hyp = getattr(hypothesis_function, "__name__", repr(hypothesis_function))

    df = optimist.knowledge_table.copy()
    df = df[df[hyp] == True]

    numeric_cols = df.select_dtypes(include=['number']).columns

    df = df[numeric_cols]

    # Ensure the target exists in the dataframe
    if target not in df.columns:
        raise ValueError(f"Target {target} not in numeric columns!")

    df = df.fillna(0)

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Fix potential NaN values and normalize
    X = X.fillna(0)
    X = (X - X.mean()) / X.std()
    y = y.fillna(0)
    y = (y - y.mean()) / y.std()

    # Compute absolute Pearson correlation
    correlations = X.corrwith(y).abs()

    # Convert correlations into a probability distribution
    prob_dist = correlations / correlations.sum()

    # Remove problematic entries
    prob_dist = prob_dist.replace([np.inf, -np.inf], np.nan).dropna()

    # Renormalize
    prob_dist /= prob_dist.sum()

    # Now sample safely:
    return list(set(random.choices(
        population=prob_dist.index,
        weights=prob_dist.values,
        k=num_features,
    )))
