from itertools import combinations
from tqdm import tqdm  # Import tqdm as a function for progress bars
import numpy as np
import pandas as pd
import pulp
import math
from fractions import Fraction
from graffitiai.base import BaseConjecturer, BoundConjecture

__all__ = [
    "GraffitiAI",
]

class GraffitiAI(BaseConjecturer):

    def update_accepted_upper_candidates(self):
        # Step 1: Add the new candidate if it beats at least one row
        if not self.accepted_upper_conjectures:
            return []
        else:
            # Step 2: Remove candidates that no longer have the property
            updated_candidates = []
            for i, candidate in enumerate(self.accepted_upper_conjectures):
                candidate_values = candidate.candidate_func(self.knowledge_table)

                # Build a DataFrame of the other candidates' values
                other_funcs = [
                    other.candidate_func(self.knowledge_table)
                    for j, other in enumerate(self.accepted_upper_conjectures) if j != i
                ]
                if other_funcs:  # Only compare if there is at least one other candidate
                    other_values = pd.concat(other_funcs, axis=1)
                    min_others = other_values.min(axis=1)

                    # Keep candidate only if it is lower than all others on at least one row
                    if (candidate_values < min_others).any():
                        updated_candidates.append(candidate)
                else:
                    # If candidate is the only one, it automatically qualifies.
                    updated_candidates.append(candidate)

            self.accepted_upper_conjectures = updated_candidates

    def update_accepted_lower_candidates(self):
        # Add the new candidate if it beats at least one row.
        if not self.accepted_lower_conjectures:
            return []
        else:
            # Now, remove candidates that no longer satisfy the lower-bound property.
            updated_candidates = []
            for i, candidate in enumerate(self.accepted_lower_conjectures):
                candidate_values = candidate.candidate_func(self.knowledge_table)

                # Gather the values of all other candidates.
                other_values_list = [
                    other.candidate_func(self.knowledge_table)
                    for j, other in enumerate(self.accepted_lower_conjectures) if j != i
                ]

                if other_values_list:  # Only compare if there is at least one other candidate.
                    other_values = pd.concat(other_values_list, axis=1)
                    max_others = other_values.max(axis=1)

                    # Keep the candidate only if it is strictly greater than the max of all others for at least one row.
                    if (candidate_values > max_others).any():
                        updated_candidates.append(candidate)
                else:
                    # If it's the only candidate, it automatically qualifies.
                    updated_candidates.append(candidate)

            self.accepted_lower_conjectures = updated_candidates

    def conjecture(
        self,
        target_invariant=None,
        target_invariants=None,
        other_invariants=None,
        hypothesis=None,
        complexity_range=(2, 3),
        lower_b_max=None,
        upper_b_max=None,
        lower_b_min=None,
        upper_b_min=None,
        W_lower_bound=-10,
        W_upper_bound=10,
    ):
        from graffitiai.utils import filter_upper_candidates, filter_lower_candidates
        if other_invariants is None:
            other_invariants = self.numerical_columns
        if hypothesis is None:
            hypothesis = self.boolean_columns

        targets = [target_invariant] if target_invariant else (target_invariants or self.numerical_columns)

        total_iterations = sum(
            sum(
                len(list(combinations([inv for inv in other_invariants if inv != target], complexity))) * len(hypothesis)
                for complexity in range(complexity_range[0], complexity_range[1] + 1)
            )
            for target in targets
        )
        if total_iterations == 0:
            return

        with tqdm(total=total_iterations, desc="Generating Conjectures", leave=True) as pbar:
            for target_invariant in targets:
                upper_bounds = []
                lower_bounds = []
                valid_invariants = [inv for inv in other_invariants if inv != target_invariant]
                comp_range = range(complexity_range[0], complexity_range[1] + 1)
                for prop in hypothesis:
                    # Pre-filter the DataFrame once per hypothesis.
                    filtered_df = self.knowledge_table[self.knowledge_table[prop] == True]
                    self.accepted_upper_conjectures = []
                    self.accepted_lower_conjectures = []
                    for complexity in comp_range:
                        for combo in combinations(valid_invariants, complexity):
                            combo_invariants = list(combo)
                            candidate_upper_bound_conjecture = self.make_upper_linear_conjecture(
                                filtered_df,
                                target_invariant,
                                combo_invariants,
                                hyp=prop,
                                b_upper_bound=upper_b_max,
                                b_lower_bound=upper_b_min,
                                W_upper_bound=W_upper_bound,
                                W_lower_bound=W_lower_bound,
                            )
                            if candidate_upper_bound_conjecture is not None:
                                self.accepted_upper_conjectures.append(candidate_upper_bound_conjecture)

                            candidate_lower_bound_conjecture = self.make_lower_linear_conjecture(
                                filtered_df,
                                target_invariant,
                                combo_invariants,
                                hyp=prop,
                                b_upper_bound=lower_b_max,
                                b_lower_bound=lower_b_min,
                                W_upper_bound=W_upper_bound,
                                W_lower_bound=W_lower_bound,
                            )
                            if candidate_lower_bound_conjecture is not None:
                                self.accepted_lower_conjectures.append(candidate_lower_bound_conjecture)

                            pbar.update(1)
                    # First, filter out any candidates that fail the basic bound check.
                    basic_upper = [x for x in set(self.accepted_upper_conjectures) if x(self.knowledge_table)]
                    basic_lower = [x for x in set(self.accepted_lower_conjectures) if x(self.knowledge_table)]
                    # Now apply the dalmatian acceptance criteria.
                    filtered_upper = filter_upper_candidates(basic_upper, self.knowledge_table)
                    filtered_lower = filter_lower_candidates(basic_lower, self.knowledge_table)
                    upper_bounds += filtered_upper
                    lower_bounds += filtered_lower

                self.conjectures[target_invariant] = {
                    "upper": sorted(upper_bounds, key=lambda x: x.touch, reverse=True),
                    "lower": sorted(lower_bounds, key=lambda x: x.touch, reverse=True),
                }
        return self.conjectures

    def write_on_the_wall(self, target_invariants=None):
        """
        Display generated upper and lower conjectures for specified target invariants.

        Args:
            target_invariants (list, optional): List of target invariants to display.
                If None, displays conjectures for all invariants.

        Example:
            >>> ai.write_on_the_wall(target_invariants=['independence_number'])
        """
        from pyfiglet import Figlet
        fig = Figlet(font='slant')

        # Print the main title.
        title = fig.renderText("Graffiti AI")
        print(title)
        print("Author: Randy R. Davila, PhD")
        print("Automated Conjecturing since 2017")
        print()
        print('-' * 50)
        print()

        if not hasattr(self, 'conjectures') or not self.conjectures:
            print("No conjectures generated yet!")
            return

        # If no specific target invariants are provided, use all available.
        if target_invariants is None:
            target_invariants = list(self.conjectures.keys())

        # Iterate through each target invariant.
        for target in target_invariants:

            conj_data = self.conjectures.get(target, {})
            upper_conj = conj_data.get("upper", [])
            lower_conj = conj_data.get("lower", [])

            # Print upper bound conjectures.
            print("Upper Bound Conjectures:")
            if upper_conj:
                for i, conj in enumerate(upper_conj, start=1):
                    print(f"Conjecture {i}: {conj}. Touch: {conj.touch}")
                    print()
            else:
                print("  None")
            print()
            print()
            # Print lower bound conjectures.
            print("Lower Bound Conjectures:")
            if lower_conj:
                for i, conj in enumerate(lower_conj, start=1):
                    print(f"Conjecture {i}: {conj}. Touch: {conj.touch}")
                    print()
            else:
                print("  None")


    def make_upper_linear_conjecture(
            self,
            df,
            target_invariant,
            other_invariants,
            hyp="object",
            b_upper_bound=None,
            b_lower_bound=None,
            W_upper_bound=None,
            W_lower_bound=None,
        ):
        from graffitiai.utils import linear_function_to_string
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

        # Define LP variables.
        W = [pulp.LpVariable(f"w_{i+1}", W_lower_bound, W_upper_bound) for i in range(complexity)]
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

        sharp_instances = {
            true_objects[j]
            for j in range(len(Y_true))
            if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
        }
        touch = len(sharp_instances)

        def candidate_function(df):
            df = df[df[hyp] == True]
            return sum(W_values[i] * df[other_invariants[i]] for i in range(complexity)) + b_value

        # bound_callable ensures that for all rows, the target invariant is at most the candidate function's value.
        def bound_callable(df):
            df = df[df[hyp] == True]
            return (df[target_invariant] <= candidate_function(df)).all()

        conclusion = f"{linear_function_to_string(W_values, other_invariants, b_value)}"

        return BoundConjecture(
            target_invariant,
            conclusion,
            candidate_function,
            bound_type='upper',
            touch=touch,
            sharp_instances=sharp_instances,
            hypothesis=hyp,
            conclusion=conclusion,
            callable=bound_callable
        )

    def make_lower_linear_conjecture(
            self,
            df,
            target_invariant,
            other_invariants,
            hyp="object",
            b_upper_bound=None,
            b_lower_bound=None,
            W_upper_bound=None,    # Example numeric default
            W_lower_bound=None,   # Example numeric default
        ):
        from graffitiai.utils import linear_function_to_string
        pulp.LpSolverDefault.msg = 0

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
        W = [pulp.LpVariable(f"w_{i+1}", W_lower_bound, W_upper_bound) for i in range(complexity)]
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

        sharp_instances = {
            true_objects[j]
            for j in range(len(Y_true))
            if Y_true[j] == sum(W_values[i] * X_true[i][j] for i in range(complexity)) + b_value
        }
        touch = len(sharp_instances)

        def candidate_function(df):
            df = df[df[hyp] == True]
            return sum(W_values[i] * df[other_invariants[i]] for i in range(complexity)) + b_value

        # Define a callable that checks the lower bound condition on the DataFrame
        def bound_callable(df):
            df = df[df[hyp] == True]
            return (df[target_invariant] >= candidate_function(df)).all()

        conclusion = f"{linear_function_to_string(W_values, other_invariants, b_value)}"

        return BoundConjecture(
            target_invariant,
            conclusion,
            candidate_function,
            bound_type='lower',
            touch=touch,
            sharp_instances=sharp_instances,
            hypothesis=hyp,
            conclusion=conclusion,
            callable=bound_callable
        )
