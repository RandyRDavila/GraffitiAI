import pulp
import math
from fractions import Fraction
from itertools import combinations

from .conjecture_class import Hypothesis, MultiLinearConclusion, MultiLinearConjecture

def make_upper_linear_conjecture(
        df,
        target_invariant,
        other_invariants,
        hyp="object",
        b_upper_bound=None,
        b_lower_bound=None,
    ):
    pulp.LpSolverDefault.msg = 0

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

    # Define LP variables
    W = [pulp.LpVariable(f"w_{i+1}", -1000, 1000) for i in range(complexity)]
    b = pulp.LpVariable("b")

    # Auxiliary absolute value variables for W
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

    # Ensure at least one W value is nonzero
    prob += pulp.lpSum(W_abs) >= 1e-6

    # Set bounds on the intercept term, if provided
    if b_upper_bound:
        prob += b <= b_upper_bound
    if b_lower_bound:
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

    sharp_instances = {
        true_objects[j]
        for j in range(len(Y_true))
        if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
    }
    touch_number = len(sharp_instances)

    # Create conjecture
    hypothesis = Hypothesis(hyp, true_object_set=true_objects)
    conclusion = MultiLinearConclusion(target_invariant, "<=", W_values, other_invariants, b_value)

    return MultiLinearConjecture(hypothesis, conclusion, touch_number, sharp_instances)

def make_lower_linear_conjecture(
        df,
        target_invariant,
        other_invariants,
        hyp="object",
        b_upper_bound=None,
        b_lower_bound=None,
    ):
    pulp.LpSolverDefault.msg = 0

    # Filter data for the hypothesis condition
    df = df[df[hyp] == True]
    true_objects = df["name"].tolist()

    # Complexity is the number of other invariants being considered
    complexity = len(other_invariants)

    # Preprocess the data to find the minimum Y for each X for the upper bound
    extreme_points = df.loc[df.groupby(other_invariants)[target_invariant].idxmin()]

    # Extract the data for the upper and lower bound problems
    X = [extreme_points[other].tolist() for other in other_invariants]
    Y = extreme_points[target_invariant].tolist()

    num_instances = len(Y)

    # Define LP variables
    W = [pulp.LpVariable(f"w_{i+1}", -1000, 1000) for i in range(complexity)]
    b = pulp.LpVariable("b")

    # Auxiliary absolute value variables for W
    W_abs = [pulp.LpVariable(f"W_abs_{i+1}", lowBound=0) for i in range(complexity)]

    # Initialize the LP
    prob = pulp.LpProblem("Generate_Lower_Bound_Conjecture", pulp.LpMaximize)

    # Define the objective function
    prob += pulp.lpSum(
        [(pulp.lpSum(W[i] * X[i][j] for i in range(complexity)) + b - Y[j]) for j in range(num_instances)]
    )

    # Constraints
    for j in range(num_instances):
        prob += pulp.lpSum([W[i] * X[i][j] for i in range(complexity)]) + b <= Y[j]

    # Enforce W_abs[i] >= |W[i]|
    for i in range(complexity):
        prob += W_abs[i] >= W[i]
        prob += W_abs[i] >= -W[i]

    # Ensure at least one W value is nonzero
    prob += pulp.lpSum(W_abs) >= 1e-6

    # Set bounds on the intercept term, if provided
    if b_upper_bound:
        prob += b <= b_upper_bound
    if b_lower_bound:
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

    sharp_instances = {
        true_objects[j]
        for j in range(len(Y_true))
        if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
    }
    touch_number = len(sharp_instances)

    # Create conjecture
    hypothesis = Hypothesis(hyp, true_object_set=true_objects)
    conclusion = MultiLinearConclusion(target_invariant, ">=", W_values, other_invariants, b_value)

    return MultiLinearConjecture(hypothesis, conclusion, touch_number, sharp_instances)


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

    :param df: The data frame containing the invariant data.
    :param target_invariant: The name/key of the target invariant.
    :param other_invariants: A list of other invariants from which to form combinations.
    :param properties: A list of 'hypotheses' or properties to incorporate in the conjecture.
    :param complexity: The number 'k' of invariants to combine in each conjecture.
    :return: Two lists: (upper_conjectures, lower_conjectures).
    """

    upper_conjectures = []
    lower_conjectures = []

    # Exclude the target_invariant from our "other invariants" to mimic the original logic
    valid_invariants = [inv for inv in other_invariants if inv != target_invariant]

    # Generate all k-combinations from the valid invariants
    for combo in combinations(valid_invariants, complexity):
        for prop in properties:
            # Generate the "upper" conjecture
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

            # Generate the "lower" conjecture
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
        progress_bar=None  # Accept an external progress bar
):
    upper_conjectures = []
    lower_conjectures = []

    valid_invariants = [inv for inv in other_invariants if inv != target_invariant]
    complexity_range = (complexity_range[0], complexity_range[1] + 1)

    for complexity in range(*complexity_range):
        for combo in combinations(valid_invariants, complexity):
            for prop in properties:
                upper_conj = make_upper_linear_conjecture(
                    df, target_invariant, list(combo), hyp=prop,
                    b_upper_bound=upper_b_max, b_lower_bound=upper_b_min
                )
                if upper_conj:
                    upper_conjectures.append(upper_conj)

                lower_conj = make_lower_linear_conjecture(
                    df, target_invariant, list(combo), hyp=prop,
                    b_upper_bound=lower_b_max, b_lower_bound=lower_b_min
                )
                if lower_conj:
                    lower_conjectures.append(lower_conj)

                if progress_bar:
                    progress_bar.update(1)  # Update the shared progress bar

    return upper_conjectures, lower_conjectures
