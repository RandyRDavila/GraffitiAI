import pulp
from fractions import Fraction
from itertools import combinations

from .conjecture_class import Hypothesis, MultiLinearConclusion, MultiLinearConjecture


def make_upper_linear_conjecture(
        df,
        target_invariant,
        other_invariants,
        hyp = "object",
    ):

    pulp.LpSolverDefault.msg = 0

    # Filter data for the hypothesis condition.
    df = df[df[hyp] == True]
    true_objects = df["name"].tolist()

    # Preprocess the data to find the maximum Y for each X for the upper bound
    df_upper = df.loc[df.groupby(other_invariants)[target_invariant].idxmax()]

    # Extract the data for the upper and lower bound problems
    Xs_upper = [df_upper[other].tolist() for other in other_invariants]
    Y_upper = df_upper[target_invariant].tolist()

    ws_upper = [pulp.LpVariable(f"w_upper{i+1}") for i in range(len(other_invariants))]  # Weights for upper bound
    b_upper = pulp.LpVariable("b_upper")


    # Initialize the LP, say "prob".
    prob = pulp.LpProblem("Test_Problem", pulp.LpMinimize)


    # Define the objective function.
    prob += pulp.lpSum(
        [(ws_upper[i] * Xs_upper[i][j] for i in range(len(other_invariants))) + b_upper - Y_upper[j]
         for j in range(len(Y_upper))]
    )


    # Define the LP constraints.
    # Upper bound constraints (maximize equality on max Y values)
    for j in range(len(Y_upper)):
        prob += pulp.lpSum([ws_upper[i] * Xs_upper[i][j] for i in range(len(other_invariants))]) + b_upper >= Y_upper[j]
        # prob += pulp.lpSum([Xs_upper[i][j] for i in range(len(other_invariants))]) >= b_upper/12

    # Solve the LP.
    prob.solve()

    # Solve the MIP
    prob.solve()

    if prob.status != 1:
        print("No feasible solution found.")
        return None
    else:
        weights_upper = [Fraction(w.varValue).limit_denominator(10) for w in ws_upper]
        b_upper_value = Fraction(b_upper.varValue).limit_denominator(10)

        Xs_true_upper = [df[other].tolist() for other in other_invariants]
        Y_true_upper = df[target_invariant].tolist()
        # Xs_true_lower = [df[other].tolist() for other in other_invariants]
        # Y_true_lower = df[target_invariant].tolist()
        # Compute the number of instances of equality - the touch number of the conjecture.
        touch_set_upper = set([true_objects[j] for j in range(len(Y_true_upper)) if
                            Y_true_upper[j] == sum(weights_upper[i] * Xs_true_upper[i][j] for i in range(len(other_invariants))) + b_upper_value])

        touch_upper = len(touch_set_upper)

        # Create the hypothesis and conclusion objects for both upper and lower bounds.
        hypothesis = Hypothesis(hyp, true_object_set=true_objects)
        upper_conclusion = MultiLinearConclusion(target_invariant, "<=", weights_upper, other_invariants, b_upper_value)

        # Return the full conjecture object (not the conclusion directly).
        return MultiLinearConjecture(hypothesis, upper_conclusion, touch_upper, touch_set_upper)

def make_lower_linear_conjecture(
        df,
        target_invariant,
        other_invariants,
        hyp="object",
    ):
    pulp.LpSolverDefault.msg = 0

    # Filter data for the hypothesis condition.
    df = df[df[hyp] == True]
    true_objects = df["name"].tolist()

    # Preprocess the data to find the maximum Y for each X for the upper bound
    df_upper = df.loc[df.groupby(other_invariants)[target_invariant].idxmin()]

    # Extract the data for the upper and lower bound problems
    Xs_upper = [df_upper[other].tolist() for other in other_invariants]
    Y_upper = df_upper[target_invariant].tolist()

    ws_upper = [pulp.LpVariable(f"w_upper{i+1}") for i in range(len(other_invariants))]  # Weights for upper bound
    b_upper = pulp.LpVariable("b_upper")

    # Initialize the LP
    prob = pulp.LpProblem("Test_Problem", pulp.LpMaximize)

    # Define the objective function using pulp.lpSum
    prob += pulp.lpSum(
        [(ws_upper[i] * Xs_upper[i][j] for i in range(len(other_invariants))) + b_upper - Y_upper[j]
         for j in range(len(Y_upper))]
    )

    # Define the LP constraints
    # Upper bound constraints (maximize equality on max Y values)
    for j in range(len(Y_upper)):
        prob += pulp.lpSum([ws_upper[i] * Xs_upper[i][j] for i in range(len(other_invariants))]) + b_upper <= Y_upper[j]

        # prob += pulp.lpSum([Xs_upper[i][j] for i in range(len(other_invariants))]) >= b_upper/12

    # Solve the LP
    prob.solve()

    if prob.status != 1:
        print("No feasible solution found.")
        return None
    else:
        weights_upper = [Fraction(w.varValue).limit_denominator(10) for w in ws_upper]
        b_upper_value = Fraction(b_upper.varValue).limit_denominator(10)

        Xs_true_upper = [df[other].tolist() for other in other_invariants]
        Y_true_upper = df[target_invariant].tolist()
        Xs_true_lower = [df[other].tolist() for other in other_invariants]
        Y_true_lower = df[target_invariant].tolist()
        # Compute the number of instances of equality - the touch number of the conjecture.
        touch_set_upper = set([true_objects[j] for j in range(len(Y_true_upper)) if
                            Y_true_upper[j] == sum(weights_upper[i] * Xs_true_upper[i][j] for i in range(len(other_invariants))) + b_upper_value])

        touch_upper = len(touch_set_upper)

        # Create the hypothesis and conclusion objects for both upper and lower bounds.
        hypothesis = Hypothesis(hyp, true_object_set=true_objects)
        upper_conclusion = MultiLinearConclusion(target_invariant, ">=", weights_upper, other_invariants, b_upper_value)

        # Return the full conjecture object (not the conclusion directly).
        return MultiLinearConjecture(hypothesis, upper_conclusion, touch_upper, touch_set_upper)


# def make_all_linear_conjectures(df, target_invariant, other_invariants, properties, complexity=2):

#     upper_conjectures = []
#     lower_conjectures = []
#     seen_pairs = []
#     if complexity == 2:
#         for other1, other2 in combinations(other_invariants, 2):
#             set_pair = set([other1, other2])
#             if set_pair not in seen_pairs:
#                 seen_pairs.append(set_pair)
#                 for prop in properties:
#                     # Ensure that neither of the 'other' invariants equals the target_invariant.
#                     if other1 != target_invariant and other2 != target_invariant:
#                         # Generate the conjecture for this combination of two invariants.
#                         upper_conj = make_upper_linear_conjecture(df, target_invariant, [other1, other2], hyp=prop)
#                         if upper_conj:
#                             upper_conjectures.append(upper_conj)
#                         lower_conj = make_lower_linear_conjecture(df, target_invariant, [other1, other2], hyp=prop)
#                         if lower_conj:
#                             lower_conjectures.append(lower_conj)
#     elif complexity == 1:
#         for other in other_invariants:
#             for prop in properties:
#                 if other != target_invariant:
#                     upper_conj = make_upper_linear_conjecture(df, target_invariant, [other], hyp=prop)
#                     if upper_conj:
#                         upper_conjectures.append(upper_conj)
#                         lower_conj = make_lower_linear_conjecture(df, target_invariant, [other], hyp=prop)
#                     if lower_conj:
#                         lower_conjectures.append(lower_conj)

#     return upper_conjectures, lower_conjectures


def make_all_linear_conjectures(df, target_invariant, other_invariants, properties, complexity=2):
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
            upper_conj = make_upper_linear_conjecture(df, target_invariant, list(combo), hyp=prop)
            if upper_conj:
                upper_conjectures.append(upper_conj)

            # Generate the "lower" conjecture
            lower_conj = make_lower_linear_conjecture(df, target_invariant, list(combo), hyp=prop)
            if lower_conj:
                lower_conjectures.append(lower_conj)

    return upper_conjectures, lower_conjectures
