from itertools import combinations
from tqdm import tqdm  # Import tqdm as a function for progress bars
import pulp
import math
from fractions import Fraction
from graffitiai.base import BaseConjecturer, BoundConjecture

__all__ = [
    "GraffitiAI",
]

class GraffitiAI(BaseConjecturer):

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
        min_touch=0,
    ):
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
                # Retrieve previously stored conjectures for this target, if any.
                if target_invariant in self.conjectures:
                    old_upper = self.conjectures[target_invariant].get("upper", [])
                    old_lower = self.conjectures[target_invariant].get("lower", [])
                    old_equals = self.conjectures[target_invariant].get("equals", [])
                else:
                    old_upper = []
                    old_lower = []
                    old_equals = []

                new_upper = []
                new_lower = []
                new_equals = []
                valid_invariants = [inv for inv in other_invariants if inv != target_invariant and inv in self.numerical_columns]
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
                                if candidate_upper_bound_conjecture.bound_type == 'equals':
                                    new_equals.append(candidate_upper_bound_conjecture)
                                else:
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
                                if candidate_lower_bound_conjecture.bound_type == 'equals':
                                    new_equals.append(candidate_lower_bound_conjecture)
                                else:
                                    self.accepted_lower_conjectures.append(candidate_lower_bound_conjecture)

                            pbar.update(1)
                    # Apply the Hazel heuristic to the accepted upper and lower bounds.
                    basic_upper = self.hazel_heuristic(self.accepted_upper_conjectures, min_touch=min_touch)
                    basic_lower = self.hazel_heuristic(self.accepted_lower_conjectures, min_touch=min_touch)

                    # Now apply the Dalmatian acceptance criteria.
                    filtered_upper = self.dalmatian_upper_bound_acceptance(basic_upper, filtered_df)
                    filtered_lower = self.dalmatian_lower_bound_acceptance(basic_lower, filtered_df)
                    new_upper += filtered_upper
                    new_lower += filtered_lower

                # Merge old and new conjectures.
                merged_upper = self.hazel_heuristic(old_upper + new_upper)
                merged_lower = self.hazel_heuristic(old_lower + new_lower)

                # Apply the Morgan heuristic to the union so that previously good conjectures are retained.
                final_upper = self.morgan_heuristic(merged_upper)
                final_lower = self.morgan_heuristic(merged_lower)

                # Filter equals
                new_equals = list(set(new_equals))
                equal_conjectures = self.morgan_heuristic(old_equals + new_equals)
                equal_conjectures = self.morgan_heuristic(equal_conjectures)

                # Remove equal conjectures from the final lists.
                final_upper = [conj for conj in final_upper if conj not in equal_conjectures]
                final_lower = [conj for conj in final_lower if conj not in equal_conjectures]

                # Update the stored conjectures.
                self.conjectures[target_invariant] = {
                    "upper": sorted(final_upper, key=lambda x: x.touch, reverse=True),
                    "lower": sorted(final_lower, key=lambda x: x.touch, reverse=True),
                    "equals": equal_conjectures,
                }
        return self.conjectures

    def hazel_heuristic(self, conjectures, min_touch = 0):
        """
        Remove duplicate conjectures and sort by touch.
        """
        conjectures = [conj for conj in set(conjectures) if conj(self.knowledge_table)]
        conjectures = [conj for conj in conjectures if conj.touch >= min_touch]
        conjectures = sorted(conjectures, key=lambda x: x.touch, reverse=True)
        return conjectures

    def morgan_heuristic(self, conjectures):
        """
        Given a list of BoundConjecture objects, returns only the most general conjectures.
        Conjectures are grouped by their conclusion, and within each group, a conjecture is
        removed if its hypothesis (i.e. its set of true_objects) is strictly more specific
        than that of another with the same conclusion.
        """
        # Group conjectures by their conclusion
        groups = {}
        for conj in conjectures:
            groups.setdefault(conj.conclusion, []).append(conj)

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

    def dalmatian_upper_bound_acceptance(self, candidates, df):
        """
        For each candidate upper-bound conjecture in candidates, evaluate its candidate function on the
        entire knowledge table (it internally filters by its hypothesis) and keep the candidate if there is
        at least one row where its value is strictly lower than every other candidate's value.
        """
        # Evaluate each candidate's function on the full table.
        cand_values = {cand: cand.candidate_func(df) for cand in candidates}
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

    def dalmatian_lower_bound_acceptance(self, candidates, df):
        """
        For each candidate lower-bound conjecture in candidates, evaluate its candidate function on the
        entire knowledge table and keep the candidate if there is at least one row where its value is strictly
        higher than every other candidate's value.
        """
        cand_values = {cand: cand.candidate_func(df) for cand in candidates}
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

    def write_on_the_wall(self, target_invariants=None, search=False):
        """
        Display generated upper and lower conjectures for specified target invariants,
        with a more detailed and user-friendly view, including:
        - Percentage of hypothesis objects that are sharp.
        - Neatly formatted sharp instances in columns.
        - Analysis of common properties among the sharp instances (if any exist).

        Args:
            target_invariants (list, optional): List of target invariants to display.
                If None, displays conjectures for all invariants.

        Example:
            >>> ai.write_on_the_wall(target_invariants=['independence_number'])
        """
        from pyfiglet import Figlet
        import pandas as pd
        fig = Figlet(font='slant')

        # Helper: Get subset of rows corresponding to sharp instances.
        def get_sharp_subset(df, sharp_ids):
            """
            If 'name' is a column in df, filter rows where df['name'] is in sharp_ids;
            otherwise, assume sharp_ids are indices.
            """
            if 'name' in df.columns:
                return df[df['name'].isin(sharp_ids)]
            else:
                return df.loc[sharp_ids]

        # Helper: Format a list of sharp instances into columns.
        def format_sharp_instances(instances, num_columns=4, indent="    "):
            items = sorted(str(item) for item in instances)
            if not items:
                return ""
            max_width = max(len(item) for item in items)
            rows = (len(items) + num_columns - 1) // num_columns
            formatted_rows = []
            for row in range(rows):
                row_items = []
                for col in range(num_columns):
                    idx = col * rows + row
                    if idx < len(items):
                        row_items.append(items[idx].ljust(max_width))
                formatted_rows.append(indent + "   ".join(row_items))
            return "\n".join(formatted_rows)

        # Helper: Find common constant boolean properties.
        def find_common_boolean_properties(df, sharp_ids, boolean_columns):
            subset = get_sharp_subset(df, sharp_ids)
            common_props = {}
            for col in boolean_columns:
                unique_vals = subset[col].unique()
                if len(unique_vals) == 1:
                    common_props[col] = unique_vals[0]
            return common_props

        # Helper: Find common numeric properties.
        def find_common_numeric_properties(df, sharp_ids, numeric_columns):
            subset = get_sharp_subset(df, sharp_ids)
            common_props = {}
            for col in numeric_columns:
                values = subset[col].dropna()
                props = []
                if (values == 0).all():
                    props.append("all zero")
                # if (values != 0).all():
                #     props.append("all nonzero")
                # Check even/odd if the column is integer-like.
                # if pd.api.types.is_integer_dtype(values) or all(float(v).is_integer() for v in values):
                #     if (values % 2 == 0).all():
                #         props.append("even")
                #     if (values % 2 == 1).all():
                #         props.append("odd")
                common_props[col] = props
            return common_props

        # Helper: Find common inequalities among numeric columns.
        def find_common_inequalities(df, sharp_ids, numeric_columns):
            subset = get_sharp_subset(df, sharp_ids)
            common_ineq = []
            n = len(numeric_columns)
            for i in range(n):
                for j in range(i + 1, n):
                    col1, col2 = numeric_columns[i], numeric_columns[j]
                    if (subset[col1] < subset[col2]).all():
                        common_ineq.append((col1, '<', col2))
                    elif (subset[col1] > subset[col2]).all():
                        common_ineq.append((col1, '>', col2))
            return common_ineq

        # Print a fancy title.
        title = fig.renderText("Graffiti AI")
        print(title)
        print("Author: Randy R. Davila, PhD")
        print("Automated Conjecturing since 2017")
        print("=" * 80)
        print()

        if not hasattr(self, 'conjectures') or not self.conjectures:
            print("No conjectures generated yet!")
            return

        # Use all available target invariants if none are provided.
        if target_invariants is None:
            target_invariants = list(self.conjectures.keys())

        # Iterate through each target invariant.
        for target in target_invariants:
            conj_data = self.conjectures.get(target, {})
            upper_conj = conj_data.get("upper", [])
            lower_conj = conj_data.get("lower", [])
            equal_conj = conj_data.get("equals", [])

            print(f"Target Invariant: {target}")
            print("-" * 40)

            # Display equal conjectures if any exist.
            if equal_conj:
                print("\nEqual Conjectures:")
                for i, conj in enumerate(equal_conj, start=1):
                    print(f"\nConjecture {i}:")
                    print("------------")
                    print(f"Statement: {conj.full_expr}")
                    print("Details:")
                    print(f"  Keywords:")
                    for keyword in conj.keywords:
                        print(f"    {keyword}")
                    print(f"  Target Invariant: {conj.target}")
                    print(f"  Bound Type: {conj.bound_type}")
                    print(f"  Hypothesis: Any {conj.hypothesis.replace('_', ' ')}")
                    print(f"  Conclusion: {conj._set_conclusion()}")

            # Display Upper Bound Conjectures.
            print("\nUpper Bound Conjectures:")
            if upper_conj:
                for i, conj in enumerate(upper_conj, start=1):
                    print(f"\nConjecture {i}:")
                    print("------------")
                    print(f"Statement: {conj.full_expr}")
                    print("Details:")
                    print(f"  Keywords:")
                    for keyword in conj.keywords:
                        print(f"    {keyword}")
                    print(f"  Target Invariant: {conj.target}")
                    print(f"  Bound Type: {conj.bound_type}")
                    print(f"  Hypothesis: Any {conj.hypothesis.replace('_', ' ')}")
                    print(f"  Conclusion: {conj._set_conclusion()}")
                    if hasattr(conj, 'complexity') and conj.complexity is not None:
                        print(f"  Complexity: {conj.complexity}")
                    if conj.touch > 0:
                        if conj.touch > 1:
                            print(f"  Sharp on {conj.touch} objects.")
                        else:
                            print("  Sharp on 1 object.")
                    else:
                        print("  Inequality is strict.")
                    if hasattr(conj, 'sharp_instances') and conj.sharp_instances:
                        print("  Sharp Instances:")
                        print(format_sharp_instances(conj.sharp_instances, num_columns=4))
                        if search:
                            # If knowledge_table is available, analyze common properties.
                            if hasattr(self, 'knowledge_table'):
                                sharp_ids = list(conj.sharp_instances)
                                common_bool = (find_common_boolean_properties(self.knowledge_table, sharp_ids, self.boolean_columns)
                                            if hasattr(self, 'boolean_columns') else {})
                                common_numeric = (find_common_numeric_properties(self.knowledge_table, sharp_ids, self.original_numerical_columns)
                                                if hasattr(self, 'numerical_columns') else {})
                                # common_ineq = (find_common_inequalities(self.knowledge_table, sharp_ids, self.numerical_columns)
                                #             if hasattr(self, 'numerical_columns') else [])
                                if common_bool or common_numeric:
                                    print("  Common properties among sharp instances:")
                                    if common_bool:
                                        print("    Constant boolean columns:")
                                        for col, val in common_bool.items():
                                            print(f"      {col} == {val}")

                                    if common_numeric:
                                        print("    Common numeric properties:")
                                        for col, props in common_numeric.items():
                                            if props:
                                                print(f"      {col}: {', '.join(props)}")
                                    else:
                                        print("    Common numeric properties:")
                                        print("      None")
                                    # if common_ineq:
                                    #     print("    Common inequalities:")
                                    #     for col1, rel, col2 in common_ineq:
                                    #         print(f"      {col1} {rel} {col2}")
                                else:
                                    print("  No common properties found among sharp instances.")
                    # Calculate and display the percentage of hypothesis objects that are sharp.
                    if hasattr(self, 'knowledge_table') and conj.hypothesis in self.knowledge_table.columns:
                        hyp_df = self.knowledge_table[self.knowledge_table[conj.hypothesis] == True]
                        total_hyp = len(hyp_df)
                        if total_hyp > 0:
                            percent_sharp = 100 * conj.touch / total_hyp
                            print(f"  Percentage of hypothesis objects that are sharp: {percent_sharp:.1f}%")
                        else:
                            print("  No objects satisfy the hypothesis.")
            else:
                print("  None")

            # Display Lower Bound Conjectures.
            print("\nLower Bound Conjectures:")
            if lower_conj:
                for i, conj in enumerate(lower_conj, start=1):
                    print(f"\nConjecture {i}:")
                    print("------------")
                    print(f"Statement: {conj.full_expr}")
                    print("Details:")
                    print(f"  Keywords:")
                    for keyword in conj.keywords:
                        print(f"    {keyword}")
                    print(f"  Target Invariant: {conj.target}")
                    print(f"  Bound Type: {conj.bound_type}")
                    print(f"  Hypothesis: Any {conj.hypothesis.replace('_', ' ')}")
                    print(f"  Conclusion: {conj._set_conclusion()}")
                    if hasattr(conj, 'complexity') and conj.complexity is not None:
                        print(f"  Complexity: {conj.complexity}")
                    if conj.touch > 0:
                        if conj.touch > 1:
                            print(f"  Sharp on {conj.touch} objects.")
                        else:
                            print("  Sharp on 1 object.")
                    else:
                        print("  Inequality is strict.")
                    if hasattr(conj, 'sharp_instances') and conj.sharp_instances:
                        print("  Sharp Instances:")
                        print(format_sharp_instances(conj.sharp_instances, num_columns=4))
                        if search:
                            if hasattr(self, 'knowledge_table'):
                                sharp_ids = list(conj.sharp_instances)
                                common_bool = (find_common_boolean_properties(self.knowledge_table, sharp_ids, self.boolean_columns)
                                            if hasattr(self, 'boolean_columns') else {})
                                common_numeric = (find_common_numeric_properties(self.knowledge_table, sharp_ids, self.original_numerical_columns)
                                                if hasattr(self, 'numerical_columns') else {})
                                # common_ineq = (find_common_inequalities(self.knowledge_table, sharp_ids, self.numerical_columns)
                                #             if hasattr(self, 'numerical_columns') else [])
                                if common_bool or common_numeric:
                                    print("  Common properties among sharp instances:")
                                    if common_bool:
                                        print("    Constant boolean columns:")
                                        for col, val in common_bool.items():
                                            print(f"      {col} == {val}")
                                    if common_numeric:
                                        print("    Common numeric properties:")
                                        for col, props in common_numeric.items():
                                            if props:
                                                print(f"      {col}: {', '.join(props)}")
                                    else:
                                        print("    Common numeric properties:")
                                        print("      None")
                                    # if common_ineq:
                                    #     print("    Common inequalities:")
                                    #     for col1, rel, col2 in common_ineq:
                                    #         print(f"      {col1} {rel} {col2}")
                                else:
                                    print("  No common properties found among sharp instances.")
                    if hasattr(self, 'knowledge_table') and conj.hypothesis in self.knowledge_table.columns:
                        hyp_df = self.knowledge_table[self.knowledge_table[conj.hypothesis] == True]
                        total_hyp = len(hyp_df)
                        if total_hyp > 0:
                            percent_sharp = 100 * conj.touch / total_hyp
                            print(f"  Percentage of hypothesis objects that are sharp: {percent_sharp:.1f}%")
                        else:
                            print("  No objects satisfy the hypothesis.")
            else:
                print("  None")

            print("\n" + "=" * 80 + "\n")

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
        from graffitiai.utils import linear_function_to_string, format_conjecture_keyword
        pulp.LpSolverDefault.msg = 0

        # Get keywords
        keywords = format_conjecture_keyword(hyp, target_invariant, other_invariants)

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

        bound_type = 'equals' if touch == len(true_objects) else 'upper'

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
            bound_type=bound_type,
            touch=touch,
            sharp_instances=sharp_instances,
            hypothesis=hyp,
            conclusion=conclusion,
            callable=bound_callable,
            true_objects=set(true_objects),
            keywords=keywords,
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
        from graffitiai.utils import linear_function_to_string, format_conjecture_keyword
        pulp.LpSolverDefault.msg = 0

        # Get keywords
        keywords = format_conjecture_keyword(hyp, target_invariant, other_invariants)

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

        bound_type = 'equals' if touch == len(true_objects) else 'lower'

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
            bound_type=bound_type,
            touch=touch,
            sharp_instances=sharp_instances,
            hypothesis=hyp,
            conclusion=conclusion,
            callable=bound_callable,
            true_objects=set(true_objects),
            keywords=keywords,
        )


def get_sharp_subset(df, sharp_ids):
    """
    Given a DataFrame and a collection of sharp instance identifiers,
    return the subset of rows where the 'name' column is in sharp_ids if 'name' exists,
    otherwise assume sharp_ids are indices.
    """
    if 'name' in df.columns:
        return df[df['name'].isin(sharp_ids)]
    else:
        return df.loc[sharp_ids]
