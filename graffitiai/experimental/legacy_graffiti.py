# legacy_graffiti.py
import time
from functools import partial
from itertools import combinations, product
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging

# Import candidate operations.
from graffitiai.experimental.candidate_operations import (
    identity_func, square_func, floor_func, ceil_func,
    add_ratio_func, sub_ratio_func, multiply_ratio_func,
    add_columns_func, subtract_columns_func, subtract_columns_func_reversed,
    multiply_columns_func, max_columns_func, min_columns_func,
    abs_diff_columns_func, safe_division_func, safe_division_func_reversed,
    mod_func, sqrt_func
)

# Import candidate transformations.
from graffitiai.experimental.candidate_transformations import (
    floor_transform, ceil_transform,
    add_ratio_transform, sub_ratio_transform, multiply_ratio_transform,
    sqrt_transform
)

__all__ = [
    "LegacyGraffiti",
    "NeuralGraffiti",
]

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class LegacyGraffiti:
    def __init__(self, df, target_invariant, bound_type='lower', filter_property=None, time_limit=None):
        self.df_full = df.copy()
        if filter_property is not None:
            self.df = df[df[filter_property] == True].copy()
            self.hypothesis_str = filter_property
        else:
            self.df = df.copy()
            self.hypothesis_str = None

        self.target = target_invariant
        self.bound_type = bound_type  # 'lower' or 'upper'
        self.time_limit = time_limit  # in seconds

        # Candidate columns: numeric (but not boolean) and not the target.
        self.candidate_cols = [
            col for col in self.df.columns
            if col != target_invariant and
               pd.api.types.is_numeric_dtype(self.df[col]) and
               not pd.api.types.is_bool_dtype(self.df[col])
        ]

        self.accepted_conjectures = []
        self.max_complexity = 7

        # A list of Fraction constants used in ratio operations.
        from fractions import Fraction
        self.ratios = [
            Fraction(1, 10), Fraction(3, 10), Fraction(7, 10), Fraction(9, 10),
            Fraction(1, 9), Fraction(2, 9), Fraction(4, 9), Fraction(5, 9),
            Fraction(7, 9), Fraction(8, 9), Fraction(10, 9),
            Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(9, 8),
            Fraction(1, 7), Fraction(2, 7), Fraction(3, 7), Fraction(4, 7), Fraction(5, 7),
            Fraction(6, 7), Fraction(8, 7), Fraction(9, 7),
            Fraction(1, 6), Fraction(5, 6), Fraction(7, 6),
            Fraction(1, 5), Fraction(2, 5), Fraction(3, 5), Fraction(4, 5), Fraction(6, 5),
            Fraction(7, 5), Fraction(8, 5), Fraction(9, 5),
            Fraction(1, 4),
            Fraction(1, 3), Fraction(2, 3), Fraction(4, 3), Fraction(5, 3),
            Fraction(7, 3), Fraction(8, 3), Fraction(10, 3),
            Fraction(1, 2), Fraction(3, 2), Fraction(5, 2), Fraction(7, 2), Fraction(9, 2),
            Fraction(1, 1), Fraction(2, 1),
             # Fraction(3, 1), Fraction(4, 1),
        ]

    # -------------------- Candidate Generation --------------------
    def _generate_candidates_unary(self, col):
        # Use list comprehensions to generate all unary candidates quickly.
        base = [
            (f"{col}", partial(identity_func, col=col)),
            (f"({col})^2", partial(square_func, col=col)),
            (f"floor({col})", partial(floor_func, col=col)),
            (f"ceil({col})", partial(ceil_func, col=col))
        ]
        # Include ratio multiplication, addition, and subtraction.
        mult_candidates = [(f"{col} * {ratio}", partial(multiply_ratio_func, col=col, ratio=ratio))
                           for ratio in self.ratios]
        add_candidates = [(f"({col}) + {ratio}", partial(add_ratio_func, col=col, ratio=ratio))
                          for ratio in self.ratios]
        sub_candidates = [(f"({col}) - {ratio}", partial(sub_ratio_func, col=col, ratio=ratio))
                          for ratio in self.ratios]
        return base + mult_candidates + add_candidates + sub_candidates

    def _generate_candidates_binary(self, col1, col2):
        # Use a list comprehension where possible.
        candidates = [
            (f"({col1} + {col2})", partial(add_columns_func, col1=col1, col2=col2)),
            (f"({col1} - {col2})", partial(subtract_columns_func, col1=col1, col2=col2)),
            (f"({col2} - {col1})", partial(subtract_columns_func_reversed, col1=col1, col2=col2)),
            (f"{col1} * {col2}", partial(multiply_columns_func, col1=col1, col2=col2)),
            (f"max({col1}, {col2})", partial(max_columns_func, col1=col1, col2=col2)),
            (f"min({col1}, {col2})", partial(min_columns_func, col1=col1, col2=col2)),
            (f"abs({col1} - {col2})", partial(abs_diff_columns_func, col1=col1, col2=col2)),
            (f"{col1}*{col2}", partial(multiply_columns_func, col1=col1, col2=col2))
        ]
        if (self.df[col2] == 0).sum() == 0:
            candidates.append((f"({col1} / {col2})", partial(safe_division_func, col1=col1, col2=col2)))
        if (self.df[col1] == 0).sum() == 0:
            candidates.append((f"({col2} / {col1})", partial(safe_division_func_reversed, col1=col1, col2=col2)))
        # Add modulus candidate.
        candidates.append((f"({col1} mod {col2})", partial(mod_func, col1=col1, col2=col2)))

        return candidates

    # def _generate_candidates_trinary(self, col1, col2, col3):
    #     # Use a list comprehension where possible.
    #     candidates = [
    #         (f"({col1} + {col2} + {col3})", partial(add_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"({col1} - {col2} - {col3})", partial(subtract_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"({col2} - {col1} - {col3})", partial(subtract_columns_func_reversed, col1=col1, col2=col2, col3=col3)),
    #         (f"{col1} * {col2} * {col3}", partial(multiply_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"max({col1}, {col2}, {col3})", partial(max_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"min({col1}, {col2}, {col3})", partial(min_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"abs({col1} - {col2} - {col3})", partial(abs_diff_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"{col1}*{col2}*{col3}", partial(multiply_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"({col1} + {col2}) * {col3}", partial(multiply_columns_func, col1=col1, col2=col2, col3=col3)),
    #         (f"({col1} - {col2}) * {col3}", partial(multiply_columns_func, col1=col1, col2=col2, col3=col3)),
    #     ]
    #     return candidates

    def _generate_candidates_complex_mod_sqrt(self):
        # Use combinations from itertools.
        candidates = []
        for a, n, d in combinations(self.candidate_cols, 3):
            def candidate_func(df, a=a, n=n, d=d):
                mod_val = mod_func(df, n, d)
                one_plus = 1 + mod_val
                product_val = df[a] * one_plus
                sqrt_val = np.sqrt(product_val)
                return np.ceil(sqrt_val)
            expr_str = f"CEIL(sqrt({a} * (1 + ({n} mod {d}))))"
            candidates.append((expr_str, candidate_func))
        return candidates

    def _generate_candidates_max_of_three(self):
        # Use combinations from itertools.
        candidates = []
        for a, n, d in combinations(self.candidate_cols, 3):
            def candidate_func(df, a=a, n=n, d=d):
                return np.maximum(df[a], np.maximum(df[n], df[d]))
            expr_str = f"max({a}, max({n}, {d}))"
            candidates.append((expr_str, candidate_func))
        return candidates

    def _generate_candidates_min_of_three(self):
        # Use combinations from itertools.
        candidates = []
        for a, n, d in combinations(self.candidate_cols, 3):
            def candidate_func(df, a=a, n=n, d=d):
                return np.minimum(df[a], np.minimum(df[n], df[d]))
            expr_str = f"min({a}, min({n}, {d}))"
            candidates.append((expr_str, candidate_func))
        return candidates

    def _generate_candidates_linear_combination(self, num_terms=2):
        """
        Generate candidates of the form:
          ratio1*inv1 + (or -) ratio2*inv2 + (or -) ratio3*inv3
        using the identity candidate for each invariant.
        """
        candidates = []
        # Build base candidates using each candidate column.
        base_candidates = [(f"{col}", partial(identity_func, col=col)) for col in self.candidate_cols]
        for combo in combinations(base_candidates, num_terms):
            for ratios in product(self.ratios, repeat=num_terms):
                for signs in product([1, -1], repeat=num_terms):
                    # Build the expression string.
                    expr_parts = [f"{'' if s > 0 else '-'}{ratio}*({expr})"
                                  for (expr, _), ratio, s in zip(combo, ratios, signs)]
                    candidate_expr = " + ".join(expr_parts).replace("+ -", "- ")
                    # Define a candidate function that sums the contributions.
                    def candidate_func(df, combo=combo, ratios=ratios, signs=signs):
                        total = 0
                        for (_, func_i), ratio, s in zip(combo, ratios, signs):
                            total += s * float(ratio) * func_i(df)
                        return total
                    candidates.append((candidate_expr, candidate_func))
        return candidates

    # -------------------- Candidate Transformations --------------------
    def _with_floor_ceil(self, candidate):
        base_rhs, base_func = candidate
        return [(f"floor({base_rhs})", partial(floor_transform, base_func=base_func)),
                (f"ceil({base_rhs})", partial(ceil_transform, base_func=base_func))]

    def _with_ratio_addition(self, candidate):
        base_rhs, base_func = candidate
        return [(f"({base_rhs}) + {ratio}", partial(add_ratio_transform, base_func=base_func, ratio=ratio))
                for ratio in self.ratios]

    def _with_ratio_subtraction(self, candidate):
        base_rhs, base_func = candidate
        return [(f"({base_rhs}) - {ratio}", partial(sub_ratio_transform, base_func=base_func, ratio=ratio))
                for ratio in self.ratios]

    def _with_ratio_multiplication(self, candidate):
        base_rhs, base_func = candidate
        return [(f"{ratio}*({base_rhs})", partial(multiply_ratio_transform, base_func=base_func, ratio=ratio))
                for ratio in self.ratios]

    def _expand_candidate(self, candidate):
        variants = {candidate[0]: candidate}
        for transform_func in [self._with_floor_ceil, self._with_ratio_multiplication,
                               self._with_ratio_subtraction, self._with_ratio_addition]:
            for cand in transform_func(candidate):
                variants.setdefault(cand[0], cand)
        return list(variants.values())

    # -------------------- Search Loop --------------------
    def search(self):
        start_time = time.time()
        new_found = True
        logging.info("Starting the search process...")
        # Cache some attributes locally.
        df = self.df
        target = self.target
        bound_type = self.bound_type
        while new_found:
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                logging.info("Time limit reached. Halting search.")
                break
            new_found = False
            # Loop through complexities.
            for complexity in range(1, self.max_complexity + 1):
                logging.info(f"Generating candidates for complexity level {complexity}...")
                candidates = []
                if complexity == 1:
                    for col in self.candidate_cols:
                        candidates.extend(self._generate_candidates_unary(col))
                elif complexity == 2:
                    for col1 in self.candidate_cols:
                        for col2 in self.candidate_cols:
                            if col1 != col2:
                                candidates.extend(self._generate_candidates_binary(col1, col2))
                # elif complexity == 3:
                    #
                    # candidates.extend(self._generate_candidates_max_of_three())
                    # candidates.extend(self._generate_candidates_min_of_three())
                    # for col1, col2, col3 in combinations(self.candidate_cols, 3):
                    #     candidates.extend(self._generate_candidates_trinary(col1, col2, col3))
                    # Also include linear combinations of 3 terms.
                    # lin_candidates = self._generate_candidates_linear_combination(num_terms=2)
                    # for cand in lin_candidates:
                    #     candidates.extend(self._expand_candidate(cand))
                else:
                    logging.debug("Complexity level not yet implemented.")

                logging.info(f"Generated {len(candidates)} candidates for complexity {complexity}.")
                if not candidates:
                    continue

                with tqdm(total=len(candidates), desc=f"Complexity {complexity}", leave=True) as pbar:
                    for rhs_str, func in candidates:
                        if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                            logging.info("Time limit reached during candidate evaluation. Halting search.")
                            new_found = False
                            break
                        try:
                            candidate_series = func(df)
                        except Exception as e:
                            logging.warning(f"Skipping candidate {rhs_str} due to error: {e}")
                            pbar.update(1)
                            continue
                        pbar.set_postfix(candidate=rhs_str)
                        pbar.update(1)
                        if not self._inequality_holds(candidate_series):
                            continue
                        if not self._is_significant(candidate_series):
                            continue
                        logging.info(f"Candidate accepted: {rhs_str}")
                        self._record_conjecture(complexity, rhs_str, func)
                        new_found = True
                        break
                if new_found:
                    break
            if not new_found:
                logging.info("No further significant conjectures found within the maximum complexity.")
                break

    # -------------------- Evaluation Helpers --------------------
    def _inequality_holds(self, candidate_series):
        target_series = self.df[self.target]
        if self.bound_type == 'lower':
            return (target_series >= candidate_series).all()
        else:
            return (target_series <= candidate_series).all()

    def _is_significant(self, candidate_series):
        current_bound = self._compute_current_bound()
        if self.bound_type == 'lower':
            diff = candidate_series - current_bound
        else:
            diff = current_bound - candidate_series
        return (diff > 0).any()

    def _compute_current_bound(self):
        if not self.accepted_conjectures:
            return pd.Series(-np.inf if self.bound_type == 'lower' else np.inf, index=self.df.index)
        bounds = []
        for conj in self.accepted_conjectures:
            try:
                b = conj['func'](self.df)
                bounds.append(b)
            except Exception as e:
                print("Error computing accepted bound:", conj['full_expr_str'], e)
        df_bounds = pd.concat(bounds, axis=1)
        return df_bounds.max(axis=1) if self.bound_type == 'lower' else df_bounds.min(axis=1)

    def _record_conjecture(self, complexity, rhs_str, func):
        if self.hypothesis_str:
            if self.bound_type == 'lower':
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≥ {rhs_str}."
            else:
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≤ {rhs_str}."
        else:
            full_expr_str = f"{self.target} ≥ {rhs_str}" if self.bound_type == 'lower' else f"{self.target} ≤ {rhs_str}"
        new_conj = {
            'complexity': complexity,
            'rhs_str': rhs_str,
            'full_expr_str': full_expr_str,
            'func': func,
            'bound_type': self.bound_type
        }
        try:
            candidate_series = func(self.df)
        except Exception as e:
            print("Error evaluating candidate during record:", e)
            candidate_series = None
        touches = int((self.df[self.target] == candidate_series).sum()) if candidate_series is not None else 0
        new_conj['touch'] = touches
        self.accepted_conjectures.append(new_conj)
        print(f"Accepted conjecture (complexity {complexity}, touch {touches}): {full_expr_str}")
        self._prune_conjectures()

    def _prune_conjectures(self):
        new_conjectures = []
        removed_conjectures = []
        n = len(self.accepted_conjectures)
        for i in range(n):
            conj_i = self.accepted_conjectures[i]
            try:
                series_i = conj_i['func'](self.df)
            except Exception as e:
                print("Error evaluating conjecture for pruning:", e)
                continue
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                try:
                    series_j = self.accepted_conjectures[j]['func'](self.df)
                except Exception as e:
                    continue
                if self.bound_type == 'lower':
                    if ((series_j >= series_i).all() and (series_j > series_i).any()):
                        dominated = True
                        break
                else:
                    if ((series_j <= series_i).all() and (series_j < series_i).any()):
                        dominated = True
                        break
            if not dominated:
                new_conjectures.append(conj_i)
            else:
                removed_conjectures.append(conj_i)
        if removed_conjectures:
            print("Pruning conjectures:")
            for rem in removed_conjectures:
                print("Removed:", rem['full_expr_str'])
        self.accepted_conjectures = new_conjectures


# # legacy_graffiti.py
# import time
# from functools import partial
# from itertools import combinations, product
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import logging
# import pulp
# from fractions import Fraction

# # Import candidate operations.
# from graffitiai.experimental.candidate_operations import (
#     identity_func, square_func, floor_func, ceil_func,
#     add_ratio_func, sub_ratio_func, multiply_ratio_func,
#     add_columns_func, subtract_columns_func, subtract_columns_func_reversed,
#     multiply_columns_func, max_columns_func, min_columns_func,
#     abs_diff_columns_func, safe_division_func, safe_division_func_reversed,
#     mod_func, sqrt_func
# )

# # Import candidate transformations.
# from graffitiai.experimental.candidate_transformations import (
#     floor_transform, ceil_transform,
#     add_ratio_transform, sub_ratio_transform, multiply_ratio_transform,
#     sqrt_transform
# )

# __all__ = ["LegacyGraffiti"]

# logging.basicConfig(
#     level=logging.INFO,  # Set to DEBUG for more detailed output
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# # --- LP Solver Function ---
# def solve_optimal_linear_combination(target, invariants, include_intercept=False):
#     """
#     Solve for coefficients a_i in the candidate expression:
#       E(x) = sum_i a_i * invariants[i](x) [+ b]
#     so that for every row x:
#       target(x) - E(x) >= epsilon,
#     and maximize epsilon.

#     Filters out rows with NaN or infinite values.

#     Returns:
#        coeffs: list of optimal coefficients (floats) (if include_intercept, last element is b)
#        epsilon: optimal margin (float)
#     """
#     pulp.LpSolverDefault.msg = 0
#     # Filter out non-finite rows.
#     mask = np.isfinite(target)
#     for inv in invariants:
#         mask = mask & np.isfinite(inv)
#     if not np.any(mask):
#         return None, None
#     target = target[mask]
#     invariants = [inv[mask] for inv in invariants]

#     n = len(invariants)
#     N = len(target)
#     prob = pulp.LpProblem("OptimalLinearCombination", pulp.LpMaximize)
#     a_vars = [pulp.LpVariable(f"a_{i}", lowBound=None, cat="Continuous") for i in range(n)]
#     if include_intercept:
#         b = pulp.LpVariable("b", lowBound=None, cat="Continuous")
#     epsilon = pulp.LpVariable("epsilon", lowBound=None, cat="Continuous")
#     prob += epsilon, "Maximize minimum margin"
#     for j in range(N):
#         expr = pulp.lpSum(a_vars[i] * invariants[i][j] for i in range(n))
#         if include_intercept:
#             expr += b
#         prob += target[j] - expr >= epsilon, f"row_{j}"
#     prob.solve()
#     if pulp.LpStatus[prob.status] != "Optimal":
#         return None, None
#     coeffs = [pulp.value(var) for var in a_vars]
#     if include_intercept:
#         coeffs.append(pulp.value(b))
#     eps_value = pulp.value(epsilon)
#     return coeffs, eps_value

# # --- LegacyGraffiti Class ---
# class LegacyGraffiti:
#     def __init__(self, df, target_invariant, bound_type='lower', filter_property=None, time_limit=None):
#         self.df_full = df.copy()
#         if filter_property is not None:
#             self.df = df[df[filter_property] == True].copy()
#             self.hypothesis_str = filter_property
#         else:
#             self.df = df.copy()
#             self.hypothesis_str = None

#         self.target = target_invariant
#         self.bound_type = bound_type  # 'lower' or 'upper'
#         self.time_limit = time_limit  # in seconds

#         # Candidate columns: numeric columns (non-boolean) other than the target.
#         self.candidate_cols = [
#             col for col in self.df.columns
#             if col != target_invariant and
#                pd.api.types.is_numeric_dtype(self.df[col]) and
#                not pd.api.types.is_bool_dtype(self.df[col])
#         ]

#         self.accepted_conjectures = []
#         self.max_complexity = 7

#         # List of Fraction ratios.
#         self.ratios = [
#             Fraction(1, 10), Fraction(3, 10), Fraction(7, 10), Fraction(9, 10),
#             Fraction(1, 9), Fraction(2, 9), Fraction(4, 9), Fraction(5, 9),
#             Fraction(7, 9), Fraction(8, 9), Fraction(10, 9),
#             Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(9, 8),
#             Fraction(1, 7), Fraction(2, 7), Fraction(3, 7), Fraction(4, 7), Fraction(5, 7),
#             Fraction(6, 7), Fraction(8, 7), Fraction(9, 7),
#             Fraction(1, 6), Fraction(5, 6), Fraction(7, 6),
#             Fraction(1, 5), Fraction(2, 5), Fraction(3, 5), Fraction(4, 5), Fraction(6, 5),
#             Fraction(7, 5), Fraction(8, 5), Fraction(9, 5),
#             Fraction(1, 4),
#             Fraction(1, 3), Fraction(2, 3), Fraction(4, 3), Fraction(5, 3),
#             Fraction(7, 3), Fraction(8, 3), Fraction(10, 3),
#             Fraction(1, 2), Fraction(3, 2), Fraction(5, 2), Fraction(7, 2), Fraction(9, 2),
#             Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
#         ]

#     # -------------------- Candidate Generation Methods --------------------
#     def _generate_candidates_unary(self, col):
#         base = [
#             (f"{col}", partial(identity_func, col=col)),
#             (f"({col})^2", partial(square_func, col=col)),
#             (f"floor({col})", partial(floor_func, col=col)),
#             (f"ceil({col})", partial(ceil_func, col=col))
#         ]
#         mult_candidates = [(f"{col} * {ratio}", partial(multiply_ratio_func, col=col, ratio=ratio))
#                            for ratio in self.ratios]
#         add_candidates = [(f"({col}) + {ratio}", partial(add_ratio_func, col=col, ratio=ratio))
#                           for ratio in self.ratios]
#         sub_candidates = [(f"({col}) - {ratio}", partial(sub_ratio_func, col=col, ratio=ratio))
#                           for ratio in self.ratios]
#         return base + mult_candidates + add_candidates + sub_candidates

#     def _generate_candidates_binary(self, col1, col2):
#         candidates = [
#             (f"({col1} + {col2})", partial(add_columns_func, col1=col1, col2=col2)),
#             (f"({col1} - {col2})", partial(subtract_columns_func, col1=col1, col2=col2)),
#             (f"({col2} - {col1})", partial(subtract_columns_func_reversed, col1=col1, col2=col2)),
#             (f"{col1} * {col2}", partial(multiply_columns_func, col1=col1, col2=col2)),
#             (f"max({col1}, {col2})", partial(max_columns_func, col1=col1, col2=col2)),
#             (f"min({col1}, {col2})", partial(min_columns_func, col1=col1, col2=col2)),
#             (f"abs({col1} - {col2})", partial(abs_diff_columns_func, col1=col1, col2=col2)),
#             (f"{col1}*{col2}", partial(multiply_columns_func, col1=col1, col2=col2))
#         ]
#         if (self.df[col2] == 0).sum() == 0:
#             candidates.append((f"({col1} / {col2})", partial(safe_division_func, col1=col1, col2=col2)))
#         if (self.df[col1] == 0).sum() == 0:
#             candidates.append((f"({col2} / {col1})", partial(safe_division_func_reversed, col1=col1, col2=col2)))
#         candidates.append((f"({col1} mod {col2})", partial(mod_func, col1=col1, col2=col2)))
#         return candidates

#     def _generate_candidates_complex_mod_sqrt(self):
#         candidates = []
#         for a, n, d in combinations(self.candidate_cols, 3):
#             def candidate_func(df, a=a, n=n, d=d):
#                 mod_val = mod_func(df, n, d)
#                 one_plus = 1 + mod_val
#                 product_val = df[a] * one_plus
#                 sqrt_val = np.sqrt(product_val)
#                 return np.ceil(sqrt_val)
#             expr_str = f"CEIL(sqrt({a} * (1 + ({n} mod {d}))))"
#             candidates.append((expr_str, candidate_func))
#         return candidates

#     def _generate_candidates_linear_combination_optimal(self, candidate_columns, num_terms, include_intercept=False):
#         invariants = [self.df[col].values for col in candidate_columns]
#         target_vals = self.df[self.target].values
#         coeffs, eps = solve_optimal_linear_combination(target_vals, invariants, include_intercept=include_intercept)
#         if coeffs is None or eps is None or eps <= 0:
#             return None
#         if include_intercept:
#             frac_coeffs = [Fraction(c).limit_denominator() for c in coeffs[:-1]]
#             intercept = Fraction(coeffs[-1]).limit_denominator()
#         else:
#             frac_coeffs = [Fraction(c).limit_denominator() for c in coeffs]
#         expr_parts = []
#         for col, frac in zip(candidate_columns, frac_coeffs):
#             if frac != 0:
#                 expr_parts.append(f"{frac}*({col})")
#         if include_intercept:
#             expr_parts.append(f"{intercept}")
#         candidate_expr = " + ".join(expr_parts)
#         def candidate_func(df):
#             total = 0
#             for col, c in zip(candidate_columns, coeffs[:-1] if include_intercept else coeffs):
#                 total += c * df[col]
#             if include_intercept:
#                 total += coeffs[-1]
#             return total
#         return candidate_expr, candidate_func

#     # -------------------- Candidate Transformations --------------------
#     def _with_floor_ceil(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"floor({base_rhs})", partial(floor_transform, base_func=base_func)),
#                 (f"ceil({base_rhs})", partial(ceil_transform, base_func=base_func))]

#     def _with_ratio_addition(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"({base_rhs}) + {ratio}", partial(add_ratio_transform, base_func=base_func, ratio=ratio))
#                 for ratio in self.ratios]

#     def _with_ratio_subtraction(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"({base_rhs}) - {ratio}", partial(sub_ratio_transform, base_func=base_func, ratio=ratio))
#                 for ratio in self.ratios]

#     def _with_ratio_multiplication(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"{ratio}*({base_rhs})", partial(multiply_ratio_transform, base_func=base_func, ratio=ratio))
#                 for ratio in self.ratios]

#     def _expand_candidate(self, candidate):
#         variants = {candidate[0]: candidate}
#         for transform_func in [self._with_floor_ceil, self._with_ratio_multiplication,
#                                self._with_ratio_subtraction, self._with_ratio_addition]:
#             for cand in transform_func(candidate):
#                 variants.setdefault(cand[0], cand)
#         return list(variants.values())

#     # -------------------- Search Loop --------------------
#     def search(self):
#         start_time = time.time()
#         new_found = True
#         logging.info("Starting the search process...")
#         df = self.df
#         if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#             logging.info("Time limit reached. Halting search.")
#             return
#         while new_found:
#             if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                 logging.info("Time limit reached. Halting search.")
#                 break
#             new_found = False
#             for complexity in range(1, self.max_complexity + 1):
#                 logging.info(f"Generating candidates for complexity {complexity}...")
#                 candidates = []
#                 if complexity == 1:
#                     # LP-based candidate for each column with intercept
#                     for col in self.candidate_cols:
#                         lp_candidate = self._generate_candidates_linear_combination_optimal((col,), 1, include_intercept=True)
#                         if lp_candidate is not None:
#                             candidates.append(lp_candidate)
#                     # Enumerated unary candidates.
#                     for col in self.candidate_cols:
#                         candidates.extend(self._generate_candidates_unary(col))
#                 elif complexity == 2:
#                     # For each pair, add LP-based candidate (with intercept)
#                     for combo in combinations(self.candidate_cols, 2):
#                         lp_candidate = self._generate_candidates_linear_combination_optimal(combo, 2, include_intercept=True)
#                         if lp_candidate is not None:
#                             candidates.append(lp_candidate)
#                     # Also add enumerated binary candidates.
#                     for col1, col2 in combinations(self.candidate_cols, 2):
#                         candidates.extend(self._generate_candidates_binary(col1, col2))
#                 elif complexity == 3:
#                     # Include mod-sqrt candidates.
#                     candidates.extend(self._generate_candidates_complex_mod_sqrt())
#                     # And LP-based candidates for triplets.
#                     for combo in combinations(self.candidate_cols, 3):
#                         lp_candidate = self._generate_candidates_linear_combination_optimal(combo, 3, include_intercept=True)
#                         if lp_candidate is not None:
#                             candidates.append(lp_candidate)
#                 else:
#                     logging.debug("Complexity level not yet implemented.")
#                 logging.info(f"Generated {len(candidates)} candidates for complexity {complexity}.")
#                 if not candidates:
#                     continue
#                 with tqdm(total=len(candidates), desc=f"Complexity {complexity}", leave=True) as pbar:
#                     for rhs_str, func in candidates:
#                         if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                             logging.info("Time limit reached during candidate evaluation. Halting search.")
#                             new_found = False
#                             break
#                         try:
#                             candidate_series = func(df)
#                         except Exception as e:
#                             logging.warning(f"Skipping candidate {rhs_str} due to error: {e}")
#                             pbar.update(1)
#                             continue
#                         pbar.set_postfix(candidate=rhs_str)
#                         pbar.update(1)
#                         if not self._inequality_holds(candidate_series):
#                             continue
#                         if not self._is_significant(candidate_series):
#                             continue
#                         logging.info(f"Candidate accepted: {rhs_str}")
#                         self._record_conjecture(complexity, rhs_str, func)
#                         new_found = True
#                         break
#                 if new_found:
#                     break
#             if not new_found:
#                 logging.info("No further significant conjectures found within the maximum complexity.")
#                 break

#     # -------------------- Evaluation Helpers --------------------
    # def _inequality_holds(self, candidate_series):
    #     target_series = self.df[self.target]
    #     if self.bound_type == 'lower':
    #         return (target_series >= candidate_series).all()
    #     else:
    #         return (target_series <= candidate_series).all()

    # def _is_significant(self, candidate_series):
    #     current_bound = self._compute_current_bound()
    #     if self.bound_type == 'lower':
    #         diff = candidate_series - current_bound
    #     else:
    #         diff = current_bound - candidate_series
    #     return (diff > 0).any()

    # def _compute_current_bound(self):
    #     if not self.accepted_conjectures:
    #         return pd.Series(-np.inf if self.bound_type == 'lower' else np.inf, index=self.df.index)
    #     bounds = []
    #     for conj in self.accepted_conjectures:
    #         try:
    #             b = conj['func'](self.df)
    #             bounds.append(b)
    #         except Exception as e:
    #             print("Error computing accepted bound:", conj['full_expr_str'], e)
    #     df_bounds = pd.concat(bounds, axis=1)
    #     return df_bounds.max(axis=1) if self.bound_type == 'lower' else df_bounds.min(axis=1)

    # def _record_conjecture(self, complexity, rhs_str, func):
    #     if self.hypothesis_str:
    #         if self.bound_type == 'lower':
    #             full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≥ {rhs_str}."
    #         else:
    #             full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≤ {rhs_str}."
    #     else:
    #         full_expr_str = f"{self.target} ≥ {rhs_str}" if self.bound_type == 'lower' else f"{self.target} ≤ {rhs_str}"
    #     new_conj = {
    #         'complexity': complexity,
    #         'rhs_str': rhs_str,
    #         'full_expr_str': full_expr_str,
    #         'func': func,
    #         'bound_type': self.bound_type
    #     }
    #     try:
    #         candidate_series = func(self.df)
    #     except Exception as e:
    #         print("Error evaluating candidate during record:", e)
    #         candidate_series = None
    #     touches = int((self.df[self.target] == candidate_series).sum()) if candidate_series is not None else 0
    #     new_conj['touch'] = touches
    #     self.accepted_conjectures.append(new_conj)
    #     print(f"Accepted conjecture (complexity {complexity}, touch {touches}): {full_expr_str}")
    #     self._prune_conjectures()

    # def _prune_conjectures(self):
    #     new_conjectures = []
    #     removed_conjectures = []
    #     n = len(self.accepted_conjectures)
    #     for i in range(n):
    #         conj_i = self.accepted_conjectures[i]
    #         try:
    #             series_i = conj_i['func'](self.df)
    #         except Exception as e:
    #             print("Error evaluating conjecture for pruning:", e)
    #             continue
    #         dominated = False
    #         for j in range(n):
    #             if i == j:
    #                 continue
    #             try:
    #                 series_j = self.accepted_conjectures[j]['func'](self.df)
    #             except Exception as e:
    #                 continue
    #             if self.bound_type == 'lower':
    #                 if ((series_j >= series_i).all() and (series_j > series_i).any()):
    #                     dominated = True
    #                     break
    #             else:
    #                 if ((series_j <= series_i).all() and (series_j < series_i).any()):
    #                     dominated = True
    #                     break
    #         if not dominated:
    #             new_conjectures.append(conj_i)
    #         else:
    #             removed_conjectures.append(conj_i)
    #     if removed_conjectures:
    #         print("Pruning conjectures:")
    #         for rem in removed_conjectures:
    #             print("Removed:", rem['full_expr_str'])
    #     self.accepted_conjectures = new_conjectures

# # neural_graffiti.py
# import time
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, InputLayer
# from tensorflow.keras.optimizers import Adam
# from functools import partial
# from itertools import combinations, product
# from tqdm import tqdm
# import logging
# import pulp
# from fractions import Fraction
# import matplotlib.pyplot as plt

# # Import candidate operations.
# from graffitiai.experimental.candidate_operations import (
#     identity_func, square_func, floor_func, ceil_func,
#     add_ratio_func, sub_ratio_func, multiply_ratio_func,
#     add_columns_func, subtract_columns_func, subtract_columns_func_reversed,
#     multiply_columns_func, max_columns_func, min_columns_func,
#     abs_diff_columns_func, safe_division_func, safe_division_func_reversed,
#     mod_func, sqrt_func
# )

# # Import candidate transformations.
# from graffitiai.experimental.candidate_transformations import (
#     floor_transform, ceil_transform,
#     add_ratio_transform, sub_ratio_transform, multiply_ratio_transform,
#     sqrt_transform
# )



# logging.basicConfig(
#     level=logging.INFO,  # Change to DEBUG for more detailed logs.
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# # --- LP Solver Function ---
# def solve_optimal_linear_combination(target, invariants, include_intercept=False):
#     # Suppress solver output.
#     pulp.LpSolverDefault.msg = 0
#     # Filter out non-finite rows.
#     mask = np.isfinite(target)
#     for inv in invariants:
#         mask = mask & np.isfinite(inv)
#     if not np.any(mask):
#         return None, None
#     target = target[mask]
#     invariants = [inv[mask] for inv in invariants]

#     n = len(invariants)
#     N = len(target)
#     prob = pulp.LpProblem("OptimalLinearCombination", pulp.LpMaximize)
#     a_vars = [pulp.LpVariable(f"a_{i}", lowBound=None, cat="Continuous") for i in range(n)]
#     if include_intercept:
#         b = pulp.LpVariable("b", lowBound=None, cat="Continuous")
#     epsilon = pulp.LpVariable("epsilon", lowBound=None, cat="Continuous")
#     prob += epsilon, "Maximize minimum margin"
#     for j in range(N):
#         expr = pulp.lpSum(a_vars[i] * invariants[i][j] for i in range(n))
#         if include_intercept:
#             expr += b
#         prob += target[j] - expr >= epsilon, f"row_{j}"
#     prob.solve()
#     if pulp.LpStatus[prob.status] != "Optimal":
#         return None, None
#     coeffs = [pulp.value(var) for var in a_vars]
#     if include_intercept:
#         coeffs.append(pulp.value(b))
#     eps_value = pulp.value(epsilon)
#     return coeffs, eps_value

# # --- NeuralGraffiti Class ---
# class NeuralGraffiti:
#     def __init__(self, df, target, top_k_ratio=0.5, selection_method="soft",
#                 hidden_units=[64, 32], learning_rate=0.001,
#                 epochs=50, batch_size=32, random_state=42, time_limit=None,
#                 bound_type='lower', filter_property=None):
#         self.df = df.copy()
#         self.target = target
#         self.top_k_ratio = top_k_ratio
#         self.selection_method = selection_method
#         self.hidden_units = hidden_units
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.random_state = random_state
#         self.time_limit = time_limit
#         self.bound_type = bound_type
#         np.random.seed(self.random_state)
#         tf.random.set_seed(self.random_state)
#         self.model = None
#         self.feature_importance = None
#         self.sampling_probabilities = None
#         self.feature_cols = None
#         # Always set hypothesis_str even if filter_property is None.
#         self.hypothesis_str = filter_property if filter_property is not None else None
#         self.accepted_conjectures = []
#         self.max_complexity = 7
#         from fractions import Fraction
#         # self.ratios = [ ... ]  # your list of Fraction ratios here.
#         self.ratios = [
#                 Fraction(1, 10), Fraction(3, 10), Fraction(7, 10), Fraction(9, 10),
#                 Fraction(1, 9), Fraction(2, 9), Fraction(4, 9), Fraction(5, 9),
#                 Fraction(7, 9), Fraction(8, 9), Fraction(10, 9),
#                 Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(9, 8),
#                 Fraction(1, 7), Fraction(2, 7), Fraction(3, 7), Fraction(4, 7), Fraction(5, 7),
#                 Fraction(6, 7), Fraction(8, 7), Fraction(9, 7),
#                 Fraction(1, 6), Fraction(5, 6), Fraction(7, 6),
#                 Fraction(1, 5), Fraction(2, 5), Fraction(3, 5), Fraction(4, 5), Fraction(6, 5),
#                 Fraction(7, 5), Fraction(8, 5), Fraction(9, 5),
#                 Fraction(1, 4),
#                 Fraction(1, 3), Fraction(2, 3), Fraction(4, 3), Fraction(5, 3),
#                 Fraction(7, 3), Fraction(8, 3), Fraction(10, 3),
#                 Fraction(1, 2), Fraction(3, 2), Fraction(5, 2), Fraction(7, 2), Fraction(9, 2),
#                 Fraction(1, 1), Fraction(2, 1)
#             ]
#         self.candidate_cols = [col for col in self.df.columns
#                        if col != self.target and
#                           pd.api.types.is_numeric_dtype(self.df[col]) and
#                           not pd.api.types.is_bool_dtype(self.df[col])]


#     # ---------- Neural Network Pipeline ----------
#     def build_and_train_model(self):
#         # Exclude target from features.
#         feature_cols = [col for col in self.df.columns if col != self.target]
#         self.feature_cols = feature_cols
#         # Ensure features and target are numeric.
#         self.df[feature_cols] = self.df[feature_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
#         self.df[self.target] = pd.to_numeric(self.df[self.target], errors='coerce').astype(np.float32)
#         # Debug print.
#         print("Data types after conversion:")
#         print(self.df.dtypes)
#         X = self.df[feature_cols].values
#         y = self.df[self.target].values
#         model = Sequential()
#         # Use "shape" per warning.
#         model.add(InputLayer(shape=(X.shape[1],)))
#         for units in self.hidden_units:
#             model.add(Dense(units, activation='relu'))
#         model.add(Dense(1, activation='linear'))
#         model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
#         model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
#         self.model = model
#         logging.info("Neural network trained.")

#     def extract_feature_importances(self):
#         """
#         Extract feature importance using the sum of absolute weights from the first Dense layer with weights.
#         """
#         # Find the first layer with weights.
#         weights = None
#         for layer in self.model.layers:
#             w = layer.get_weights()
#             if w:
#                 weights = w[0]
#                 break
#         if weights is None:
#             raise ValueError("No weights found in model layers.")
#         # Compute importance as the sum of absolute weights for each input feature.
#         importance = np.sum(np.abs(weights), axis=1)
#         self.feature_importance = importance
#         probabilities = importance / np.sum(importance)
#         self.sampling_probabilities = probabilities
#         sorted_idx = np.argsort(probabilities)[::-1]
#         self.selected_features = [self.feature_cols[i] for i in sorted_idx]
#         logging.info("Feature importances extracted.")
#         return self.selected_features, self.sampling_probabilities

#     def visualize_feature_importance(self):
#         df_vis = pd.DataFrame({
#             "Feature": self.feature_cols,
#             "Importance": self.feature_importance,
#             "Probability": self.sampling_probabilities
#         }).sort_values(by="Importance", ascending=False)
#         plt.figure(figsize=(10, 5))
#         plt.bar(df_vis["Feature"], df_vis["Importance"])
#         plt.title("Feature Importance")
#         plt.xlabel("Feature")
#         plt.ylabel("Importance")
#         plt.xticks(rotation=45)
#         plt.show()
#         plt.figure(figsize=(10, 5))
#         plt.bar(df_vis["Feature"], df_vis["Probability"])
#         plt.title("Sampling Probability Distribution")
#         plt.xlabel("Feature")
#         plt.ylabel("Probability")
#         plt.xticks(rotation=45)
#         plt.show()

#     def get_candidate_columns(self):
#         """
#         Returns candidate feature names based on the selection method.
#         """
#         if self.selection_method == "hard":
#             top_k = max(1, int(self.top_k_ratio * len(self.selected_features)))
#             return self.selected_features[:top_k]
#         elif self.selection_method == "weighted":
#             k = max(1, int(self.top_k_ratio * len(self.selected_features)))
#             return list(np.random.choice(self.selected_features, size=k, replace=False, p=self.sampling_probabilities))
#         else:
#             return self.selected_features

#     # ---------- LegacyGraffiti Candidate Generation Methods ----------
#     def _generate_candidates_unary(self, col):
#         base = [
#             (f"{col}", partial(identity_func, col=col)),
#             (f"({col})^2", partial(square_func, col=col)),
#             (f"floor({col})", partial(floor_func, col=col)),
#             (f"ceil({col})", partial(ceil_func, col=col))
#         ]
#         mult_candidates = [(f"{col} * {ratio}", partial(multiply_ratio_func, col=col, ratio=ratio))
#                            for ratio in self.ratios]
#         add_candidates = [(f"({col}) + {ratio}", partial(add_ratio_func, col=col, ratio=ratio))
#                           for ratio in self.ratios]
#         sub_candidates = [(f"({col}) - {ratio}", partial(sub_ratio_func, col=col, ratio=ratio))
#                           for ratio in self.ratios]
#         return base + mult_candidates + add_candidates + sub_candidates

#     def _generate_candidates_binary(self, col1, col2):
#         candidates = [
#             (f"({col1} + {col2})", partial(add_columns_func, col1=col1, col2=col2)),
#             (f"({col1} - {col2})", partial(subtract_columns_func, col1=col1, col2=col2)),
#             (f"({col2} - {col1})", partial(subtract_columns_func_reversed, col1=col1, col2=col2)),
#             (f"{col1} * {col2}", partial(multiply_columns_func, col1=col1, col2=col2)),
#             (f"max({col1}, {col2})", partial(max_columns_func, col1=col1, col2=col2)),
#             (f"min({col1}, {col2})", partial(min_columns_func, col1=col1, col2=col2)),
#             (f"abs({col1} - {col2})", partial(abs_diff_columns_func, col1=col1, col2=col2)),
#             (f"{col1}*{col2}", partial(multiply_columns_func, col1=col1, col2=col2))
#         ]
#         if (self.df[col2] == 0).sum() == 0:
#             candidates.append((f"({col1} / {col2})", partial(safe_division_func, col1=col1, col2=col2)))
#         if (self.df[col1] == 0).sum() == 0:
#             candidates.append((f"({col2} / {col1})", partial(safe_division_func_reversed, col1=col1, col2=col2)))
#         candidates.append((f"({col1} mod {col2})", partial(mod_func, col1=col1, col2=col2)))
#         return candidates

#     def _generate_candidates_complex_mod_sqrt(self):
#         candidates = []
#         for a, n, d in combinations(self.candidate_cols, 3):
#             def candidate_func(df, a=a, n=n, d=d):
#                 mod_val = mod_func(df, n, d)
#                 one_plus = 1 + mod_val
#                 product_val = df[a] * one_plus
#                 sqrt_val = np.sqrt(product_val)
#                 return np.ceil(sqrt_val)
#             expr_str = f"CEIL(sqrt({a} * (1 + ({n} mod {d}))))"
#             candidates.append((expr_str, candidate_func))
#         return candidates

#     def _generate_candidates_linear_combination_optimal(self, candidate_columns, include_intercept=False):
#         invariants = [self.df[col].values for col in candidate_columns]
#         target_vals = self.df[self.target].values
#         coeffs, eps = solve_optimal_linear_combination(target_vals, invariants, include_intercept=include_intercept)
#         if coeffs is None or eps is None or eps <= 0:
#             return None
#         if include_intercept:
#             frac_coeffs = [Fraction(c).limit_denominator() for c in coeffs[:-1]]
#             intercept = Fraction(coeffs[-1]).limit_denominator()
#         else:
#             frac_coeffs = [Fraction(c).limit_denominator() for c in coeffs]
#         expr_parts = []
#         for col, frac in zip(candidate_columns, frac_coeffs):
#             if frac != 0:
#                 expr_parts.append(f"{frac}*({col})")
#         if include_intercept:
#             expr_parts.append(f"{intercept}")
#         candidate_expr = " + ".join(expr_parts)
#         def candidate_func(df):
#             total = 0
#             for col, c in zip(candidate_columns, coeffs[:-1] if include_intercept else coeffs):
#                 total += c * df[col]
#             if include_intercept:
#                 total += coeffs[-1]
#             return total
#         return candidate_expr, candidate_func

#     # ---------- Candidate Transformations ----------
#     def _with_floor_ceil(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"floor({base_rhs})", partial(floor_transform, base_func=base_func)),
#                 (f"ceil({base_rhs})", partial(ceil_transform, base_func=base_func))]

#     def _with_ratio_addition(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"({base_rhs}) + {ratio}", partial(add_ratio_transform, base_func=base_func, ratio=ratio))
#                 for ratio in self.ratios]

#     def _with_ratio_subtraction(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"({base_rhs}) - {ratio}", partial(sub_ratio_transform, base_func=base_func, ratio=ratio))
#                 for ratio in self.ratios]

#     def _with_ratio_multiplication(self, candidate):
#         base_rhs, base_func = candidate
#         return [(f"{ratio}*({base_rhs})", partial(multiply_ratio_transform, base_func=base_func, ratio=ratio))
#                 for ratio in self.ratios]

#     def _expand_candidate(self, candidate):
#         variants = {candidate[0]: candidate}
#         for transform_func in [self._with_floor_ceil, self._with_ratio_multiplication,
#                                self._with_ratio_subtraction, self._with_ratio_addition]:
#             for cand in transform_func(candidate):
#                 variants.setdefault(cand[0], cand)
#         return list(variants.values())

#     def _generate_candidates_nested(self, base_candidate):
#         """
#         Given a base candidate expression (a tuple of (expr, func)),
#         apply additional transformations (e.g., combine with another feature)
#         to build a nested candidate.
#         """
#         candidates = []
#         # For instance, combine the base candidate with a unary candidate from a top feature.
#         for col in self.get_candidate_columns():
#             # Skip if the column is already part of the expression (optional)
#             new_expr = f"({base_candidate[0]}) + ({col})"
#             new_func = lambda df, base_func=base_candidate[1], col=col: base_func(df) + df[col]
#             candidates.append((new_expr, new_func))
#             # You could also try subtraction, multiplication, or wrap with floor/ceil.
#             new_expr2 = f"floor(({base_candidate[0]}) + ({col}))"
#             new_func2 = lambda df, base_func=base_candidate[1], col=col: np.floor(base_func(df) + df[col])
#             candidates.append((new_expr2, new_func2))
#         return candidates

#     # ---------- Search Loop ----------
#     def search(self):
#         start_time = time.time()
#         new_found = True
#         logging.info("Starting the search process...")
#         df = self.df
#         while new_found:
#             if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                 logging.info("Time limit reached. Halting search.")
#                 break
#             new_found = False
#             for complexity in range(1, self.max_complexity + 1):
#                 logging.info(f"Generating candidates for complexity {complexity}...")
#                 candidates = []
#                 # Use candidate space reduction via neural guidance.
#                 candidate_columns = self.get_candidate_columns()
#                 if complexity == 1:
#                     # LP-based candidate for each selected feature.
#                     for col in candidate_columns:
#                         lp_candidate = self._generate_candidates_linear_combination_optimal((col,), include_intercept=True)
#                         if lp_candidate is not None:
#                             candidates.append(lp_candidate)
#                     # Plus enumerated unary candidates.
#                     for col in candidate_columns:
#                         candidates.extend(self._generate_candidates_unary(col))
#                 elif complexity == 2:
#                     # For each pair from candidate_columns.
#                     for combo in combinations(candidate_columns, 2):
#                         lp_candidate = self._generate_candidates_linear_combination_optimal(combo, include_intercept=True)
#                         if lp_candidate is not None:
#                             candidates.append(lp_candidate)
#                     for col1, col2 in combinations(candidate_columns, 2):
#                         candidates.extend(self._generate_candidates_binary(col1, col2))
#                 elif complexity == 3:
#                     candidates.extend(self._generate_candidates_complex_mod_sqrt())
#                     for combo in combinations(candidate_columns, 3):
#                         lp_candidate = self._generate_candidates_linear_combination_optimal(combo, include_intercept=True)
#                         if lp_candidate is not None:
#                             candidates.append(lp_candidate)
#                         # In complexity >= 3, after generating candidates as before:
#                     # For each candidate from lower complexity (e.g., an accepted candidate),
#                     # generate nested candidates.
#                     for base in self.accepted_conjectures:
#                         nested = self._generate_candidates_nested((base['rhs_str'], base['func']))
#                         candidates.extend(nested)
#                 else:
#                     logging.debug("Complexity level not yet implemented.")
#                 logging.info(f"Generated {len(candidates)} candidates for complexity {complexity}.")
#                 if not candidates:
#                     continue
#                 with tqdm(total=len(candidates), desc=f"Complexity {complexity}", leave=True) as pbar:
#                     for rhs_str, func in candidates:
#                         if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                             logging.info("Time limit reached during candidate evaluation. Halting search.")
#                             new_found = False
#                             break
#                         try:
#                             candidate_series = func(df)
#                         except Exception as e:
#                             logging.warning(f"Skipping candidate {rhs_str} due to error: {e}")
#                             pbar.update(1)
#                             continue
#                         pbar.set_postfix(candidate=rhs_str)
#                         pbar.update(1)
#                         if not self._inequality_holds(candidate_series):
#                             continue
#                         if not self._is_significant(candidate_series):
#                             continue
#                         logging.info(f"Candidate accepted: {rhs_str}")
#                         self._record_conjecture(complexity, rhs_str, func)
#                         new_found = True
#                         break
#                 if new_found:
#                     break
#             if not new_found:
#                 logging.info("No further significant conjectures found within the maximum complexity.")
#                 break

#     def get_candidate_columns(self):
#         """
#         Returns candidate feature names based on the selection method.
#         """
#         if self.selection_method == "hard":
#             top_k = max(1, int(self.top_k_ratio * len(self.selected_features)))
#             return self.selected_features[:top_k]
#         elif self.selection_method == "weighted":
#             k = max(1, int(self.top_k_ratio * len(self.selected_features)))
#             return list(np.random.choice(self.selected_features, size=k, replace=False, p=self.sampling_probabilities))
#         else:
#             return self.selected_features

#     # ---------- Evaluation Helpers ----------
#     def _inequality_holds(self, candidate_series):
#         target_series = self.df[self.target]
#         if self.bound_type == 'lower':
#             return (target_series >= candidate_series).all()
#         else:
#             return (target_series <= candidate_series).all()

#     def _is_significant(self, candidate_series):
#         current_bound = self._compute_current_bound()
#         if self.bound_type == 'lower':
#             diff = candidate_series - current_bound
#         else:
#             diff = current_bound - candidate_series
#         return (diff > 0).any()

#     def _compute_current_bound(self):
#         if not self.accepted_conjectures:
#             return pd.Series(-np.inf if self.bound_type == 'lower' else np.inf, index=self.df.index)
#         bounds = []
#         for conj in self.accepted_conjectures:
#             try:
#                 b = conj['func'](self.df)
#                 bounds.append(b)
#             except Exception as e:
#                 print("Error computing accepted bound:", conj['full_expr_str'], e)
#         df_bounds = pd.concat(bounds, axis=1)
#         return df_bounds.max(axis=1) if self.bound_type == 'lower' else df_bounds.min(axis=1)

#     def _record_conjecture(self, complexity, rhs_str, func):
#         if self.hypothesis_str:
#             if self.bound_type == 'lower':
#                 full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≥ {rhs_str}."
#             else:
#                 full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≤ {rhs_str}."
#         else:
#             full_expr_str = f"{self.target} ≥ {rhs_str}" if self.bound_type == 'lower' else f"{self.target} ≤ {rhs_str}"
#         new_conj = {
#             'complexity': complexity,
#             'rhs_str': rhs_str,
#             'full_expr_str': full_expr_str,
#             'func': func,
#             'bound_type': self.bound_type
#         }
#         try:
#             candidate_series = func(self.df)
#         except Exception as e:
#             print("Error evaluating candidate during record:", e)
#             candidate_series = None
#         touches = int((self.df[self.target] == candidate_series).sum()) if candidate_series is not None else 0
#         new_conj['touch'] = touches
#         self.accepted_conjectures.append(new_conj)
#         print(f"Accepted conjecture (complexity {complexity}, touch {touches}): {full_expr_str}")
#         self._prune_conjectures()

#     def _prune_conjectures(self):
#         new_conjectures = []
#         removed_conjectures = []
#         n = len(self.accepted_conjectures)
#         for i in range(n):
#             conj_i = self.accepted_conjectures[i]
#             try:
#                 series_i = conj_i['func'](self.df)
#             except Exception as e:
#                 print("Error evaluating conjecture for pruning:", e)
#                 continue
#             dominated = False
#             for j in range(n):
#                 if i == j:
#                     continue
#                 try:
#                     series_j = self.accepted_conjectures[j]['func'](self.df)
#                 except Exception as e:
#                     continue
#                 if self.bound_type == 'lower':
#                     if ((series_j >= series_i).all() and (series_j > series_i).any()):
#                         dominated = True
#                         break
#                 else:
#                     if ((series_j <= series_i).all() and (series_j < series_i).any()):
#                         dominated = True
#                         break
#             if not dominated:
#                 new_conjectures.append(conj_i)
#             else:
#                 removed_conjectures.append(conj_i)
#         if removed_conjectures:
#             print("Pruning conjectures:")
#             for rem in removed_conjectures:
#                 print("Removed:", rem['full_expr_str'])
#         self.accepted_conjectures = new_conjectures

#     def write_on_the_wall(self):
#         print("Accepted Conjectures:")
#         for conj in self.accepted_conjectures:
#             print(conj['full_expr_str'])





import re
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shap
import time
import warnings
import ast
import numpy as np
import statistics
from scipy.stats import gmean, hmean, skew  # Requires SciPy


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


def safe_mode(lst):
    """Return mode if a unique mode exists, otherwise return None."""
    try:
        return statistics.mode(lst)
    except Exception:
        return None

def median_absolute_deviation(lst):
    """Compute the median absolute deviation."""
    med = np.median(lst)
    return np.median([abs(x - med) for x in lst])

def safe_harmonic_mean(lst):
    """Compute the harmonic mean if possible; returns None if any element is zero."""
    try:
        # hmean will throw an error if any value is zero.
        return hmean(lst)
    except Exception:
        return None

def is_prime(n):
    """Return True if n is a prime number."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

def number_of_prime_entries(lst):
    """Return the number of prime entries in the list."""
    return len([x for x in lst if is_prime(x)])

def compute_aggregates(series):
    """
    Given a pandas Series where each element is a list of numbers,
    compute various aggregate invariants.
    Returns a dictionary with keys as aggregate names and values as the aggregate Series.
    """
    aggregates = {}
    valid_entries = series.dropna()
    if valid_entries.empty:
        return aggregates

    aggregates['count'] = valid_entries.apply(lambda lst: len(lst) if lst else None)
    aggregates['mean'] = valid_entries.apply(lambda lst: np.mean(lst) if lst else None)
    aggregates['median'] = valid_entries.apply(lambda lst: np.median(lst) if lst else None)
    aggregates['mode'] = valid_entries.apply(lambda lst: safe_mode(lst) if lst else None)
    aggregates['std'] = valid_entries.apply(lambda lst: np.std(lst) if lst else None)
    aggregates['min'] = valid_entries.apply(lambda lst: min(lst) if lst else None)
    aggregates['max'] = valid_entries.apply(lambda lst: max(lst) if lst else None)
    aggregates['sum'] = valid_entries.apply(lambda lst: sum(lst) if lst else None)
    # aggregates['second_largest_value'] = valid_entries.apply(lambda lst: sorted(lst)[-2] if len(lst) > 1 else None)
    # aggregates['second_smallest_value'] = valid_entries.apply(lambda lst: sorted(lst)[1] if len(lst) > 1 else None)
    # aggregates['skewness'] = valid_entries.apply(lambda lst: skew(lst) if len(lst) > 2 else None)
    aggregates['nonzero_count'] = valid_entries.apply(lambda lst: len([x for x in lst if x != 0]) if lst else None)
    # aggregates['positive_count'] = valid_entries.apply(lambda lst: len([x for x in lst if x > 0]) if lst else None)
    aggregates['negative_count'] = valid_entries.apply(lambda lst: len([x for x in lst if x < 0]) if lst else None)
    aggregates['unique_count'] = valid_entries.apply(lambda lst: len(set(lst)) if lst else None)
    # aggregates['nonzero_unique_count'] = valid_entries.apply(lambda lst: len(set(x for x in lst if x != 0)) if lst else None)
    # aggregates['positive_unique_count'] = valid_entries.apply(lambda lst: len(set(x for x in lst if x > 0)) if lst else None)
    # aggregates['negative_unique_count'] = valid_entries.apply(lambda lst: len(set(x for x in lst if x < 0)) if lst else None)
    aggregates['nonzero_mean'] = valid_entries.apply(lambda lst: np.mean([x for x in lst if x != 0]) if lst else None)
    aggregates['prime_count'] = valid_entries.apply(lambda lst: number_of_prime_entries(lst) if lst else None)

    aggregates['unique_ratio'] = valid_entries.apply(lambda lst: len(set(lst)) / len(lst) if lst else None)

    # Additional aggregates:
    aggregates['range'] = valid_entries.apply(lambda lst: max(lst) - min(lst) if lst else None)
    aggregates['25th_percentile'] = valid_entries.apply(lambda lst: np.percentile(lst, 25) if lst else None)
    aggregates['75th_percentile'] = valid_entries.apply(lambda lst: np.percentile(lst, 75) if lst else None)
    aggregates['IQR'] = valid_entries.apply(lambda lst: np.percentile(lst, 75) - np.percentile(lst, 25) if lst else None)
    aggregates['geom_mean'] = valid_entries.apply(lambda lst: gmean(lst) if lst and all(x > 0 for x in lst) else None)
    aggregates['harm_mean'] = valid_entries.apply(lambda lst: safe_harmonic_mean(lst) if lst and all(x > 0 for x in lst) else None)
    aggregates['sum_of_squares'] = valid_entries.apply(lambda lst: sum(x**2 for x in lst) if lst else None)
    aggregates['median_absolute_deviation'] = valid_entries.apply(lambda lst: median_absolute_deviation(lst) if lst else None)
    aggregates['CV'] = valid_entries.apply(lambda lst: np.std(lst) / np.mean(lst) if lst and np.mean(lst) != 0 else None)
    aggregates['mode_frequency'] = valid_entries.apply(lambda lst: lst.count(safe_mode(lst)) if safe_mode(lst) is not None else None)
    aggregates['zero_count'] = valid_entries.apply(lambda lst: lst.count(0) if lst else None)

    return aggregates

def detect_and_create_invariants(df):
    """
    For each object-type column in the DataFrame, try to detect whether
    it contains string representations of lists. If so, compute aggregate
    invariants and add them as new columns, and also expand the list entries
    into separate columns.
    """
    # Mapping of aggregate keys to descriptive names.
    descriptive_mapping = {
        'count': 'count',
        'mean': 'mean',
        'median': 'median',
        'mode': 'mode',
        'std': 'standard_deviation',
        'min': 'minimum',
        'max': 'maximum',
        'sum': 'sum',
        'range': 'range',
        '25th_percentile': '25th_percentile',
        '75th_percentile': '75th_percentile',
        'IQR': 'interquartile_range',
        'geom_mean': 'geometric_mean',
        'harm_mean': 'harmonic_mean',
        'sum_of_squares': 'sum_of_squares',
        'median_absolute_deviation': 'median_absolute_deviation',
        'CV': 'coefficient_of_variation',
        'mode_frequency': 'mode_frequency',
        'zero_count': 'zero_count',
        'second_largest_value': 'second_largest_value',
        'second_small_value': 'second_smallest_value',
        'skewness': 'skewness',
        'nonzero_count': 'nonzero_count',
        'positive_count': 'positive_count',
        'negative_count': 'negative_count',
        'unique_count': 'unique_count',
        'nonzero_unique_count': 'nonzero_unique_count',
        'positive_unique_count': 'positive_unique_count',
        'negative_unique_count': 'negative_unique_count',
        'nonzero_mean': 'nonzero_mean',
        'number_of_primes_in': 'number_of_primes_in',
        'unique_ratio': 'unique_ratio'
    }

    for col in df.columns:
        # Only consider object-type columns.
        if df[col].dtype == 'object':
            # Sample some non-null entries.
            sample = df[col].dropna().head(10)
            if sample.empty:
                continue
            # Count how many entries in the sample are list-like.
            count = sum(is_list_string(x) for x in sample)
            if count >= len(sample) / 2:
                print(f"Column '{col}' appears to be a list. Creating aggregate invariants and expanding column...")
                # Convert the entire column.
                parsed_series = df[col].apply(convert_list_string)

                # Compute aggregate invariants.
                aggregates = compute_aggregates(parsed_series)
                for agg_key, agg_series in aggregates.items():
                    # Create a new column with a descriptive name.
                    new_col_name = f"{descriptive_mapping.get(agg_key, agg_key)}({col})"
                    df[new_col_name] = agg_series
                    print(f"Created aggregate column: {new_col_name}")

                # Determine the maximum list length in the column.
                max_len = parsed_series.apply(lambda x: len(x) if isinstance(x, list) else 0).max()
                # Create a new column for each index position.
                for i in range(max_len):
                    new_col_name = f"{col}[{i}]"
                    df[new_col_name] = parsed_series.apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else 0)
                    print(f"Created expanded column: {new_col_name}")
    return df


# from graffitai.base import BaseConjecturer

class BaseConjecturer:
    """
    BaseConjecturer is the core abstract class for generating mathematical conjectures.
    It provides common functionality for:
      - loading and preprocessing data,
      - managing the internal data (knowledge_table),
      - applying heuristics to refine conjectures,
      - displaying and saving conjectures.

    Subclasses should override the `conjecture()` method with a concrete algorithm.

    Attributes:
        knowledge_table (pd.DataFrame): Data table used for conjecturing.
        conjectures (dict): Stores generated conjectures.
        bad_columns (list): Columns containing non-numerical or non-boolean entries.
        numerical_columns (list): List of numerical columns (excluding booleans).
        boolean_columns (list): List of boolean columns.
    """
    def __init__(self, knowledge_table=None):
        self.knowledge_table = knowledge_table
        if knowledge_table is not None:
            self.update_invariant_knowledge()
        self.conjectures = {}

    def read_csv(self, path_to_csv, drop=None):
        """
        Load data from a CSV file and preprocess it.

        - Standardizes column names.
        - Ensures a 'name' column exists.
        - Warns if non-numerical/boolean columns are present.
        - Adds a default boolean column if none exists.
        """
        self.knowledge_table = pd.read_csv(path_to_csv)
        self.bad_columns = []

        # Standardize column names.
        original_columns = self.knowledge_table.columns
        self.knowledge_table.columns = [
            re.sub(r'\W+', '_', col.strip().lower()) for col in original_columns
        ]
        print("Standardized column names:")
        for original, new in zip(original_columns, self.knowledge_table.columns):
            if original != new:
                print(f"  '{original}' -> '{new}'")

        # Ensure a 'name' column exists.
        if 'name' not in self.knowledge_table.columns:
            n = len(self.knowledge_table)
            self.knowledge_table['name'] = [f'O{i+1}' for i in range(n)]
            print(f"'name' column missing. Created default names: O1, O2, ..., O{n}.")

        # Identify problematic columns.
        for column in self.knowledge_table.columns:
            if column == 'name':  # Skip the name column.
                continue
            if not pd.api.types.is_numeric_dtype(self.knowledge_table[column]) and \
               not pd.api.types.is_bool_dtype(self.knowledge_table[column]):
                warnings.warn(f"Column '{column}' contains non-numerical and non-boolean entries.")
                self.bad_columns.append(column)

        # Add a default boolean column if none exist.
        boolean_columns = [
            col for col in self.knowledge_table.columns
            if pd.api.types.is_bool_dtype(self.knowledge_table[col])
        ]
        if not boolean_columns:
            self.knowledge_table['object'] = True
            print("No boolean columns found. Added default column 'object' with all values set to True.")

        self.update_invariant_knowledge()
        if drop:
            self.drop_columns(drop)

        self.knowledge_table = detect_and_create_invariants(self.knowledge_table)

    def add_row(self, row_data):
        """
        Add a new row of data to the knowledge_table.

        Args:
            row_data (dict): A mapping from column names to data values.

        Raises:
            ValueError: If the knowledge_table isn’t initialized or unexpected keys are found.
        """
        if self.knowledge_table is None:
            raise ValueError("Knowledge table is not initialized. Load or create a dataset first.")

        unexpected_keys = [key for key in row_data.keys() if key not in self.knowledge_table.columns]
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in row_data: {unexpected_keys}. Allowed columns: {list(self.knowledge_table.columns)}")

        complete_row = {col: row_data.get(col, None) for col in self.knowledge_table.columns}
        self.knowledge_table = pd.concat(
            [self.knowledge_table, pd.DataFrame([complete_row])],
            ignore_index=True
        )
        print(f"Row added successfully: {complete_row}")
        self.update_invariant_knowledge()

    def update_invariant_knowledge(self):
        """
        Update internal records of which columns are numerical or boolean and track any bad columns.
        """
        self.bad_columns = []
        for column in self.knowledge_table.columns:
            if column == 'name':
                continue
            if not pd.api.types.is_numeric_dtype(self.knowledge_table[column]) and \
               not pd.api.types.is_bool_dtype(self.knowledge_table[column]):
                warnings.warn(f"Column '{column}' contains non-numerical and non-boolean entries.")
                self.bad_columns.append(column)
        self.numerical_columns = [
            col for col in self.knowledge_table.columns
            if pd.api.types.is_numeric_dtype(self.knowledge_table[col]) and not pd.api.types.is_bool_dtype(self.knowledge_table[col])
        ]
        self.boolean_columns = [
            col for col in self.knowledge_table.columns
            if pd.api.types.is_bool_dtype(self.knowledge_table[col])
        ]

    def drop_columns(self, columns):
        """
        Drop the specified columns from the knowledge_table.

        Args:
            columns (list): List of column names to remove.
        """
        self.knowledge_table = self.knowledge_table.drop(columns, axis=1)
        print(f"Columns dropped: {columns}")
        self.update_invariant_knowledge()


    def conjecture(self, **kwargs):
        """
        Generate conjectures. This method is intended to be overridden by subclasses
        with specific conjecturing logic.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError("Subclasses must implement the conjecture() method.")

    def write_on_the_wall(self, target_invariants=None):
        """
        Generate conjectures. This method is intended to be overridden by subclasses
        with specific conjecturing logic.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError("Subclasses must implement the write_on_the_wall() method.")


# Assume BaseConjecturer is defined elsewhere and imported.
# from base_conjecturer import BaseConjecturer

class NeuralGraffiti(BaseConjecturer):
    """
    NeuralGraffiti uses a neural network (TensorFlow/Keras) to reduce the search
    space for generating mathematical conjectures by learning which numerical invariants
    are most relevant to bounding a target invariant under a given boolean hypothesis.

    For each hypothesis, it:
      - Filters the data to rows where the hypothesis holds.
      - Trains an MLP to predict the target invariant from the other numerical invariants.
      - Uses SHAP to extract feature importances and converts them to a probability distribution.
      - For each complexity level (number of invariants in a candidate expression),
        samples invariants according to the learned distribution.
      - Generates candidate bounds and validates them using helper methods.
      - Records acceptable conjectures.
    """
    def __init__(self, knowledge_table=None, bound_type='lower'):
        """
        Initialize NeuralGraffiti.

        Args:
            knowledge_table (pd.DataFrame, optional): The data table for conjecturing.
            bound_type (str): Either 'lower' or 'upper' to indicate the type of bound.
        """
        super().__init__(knowledge_table)
        self.bound_type = bound_type  # 'lower' for lower bounds; 'upper' for upper bounds.
        self.accepted_conjectures = []  # List to store accepted conjectures.
        self.target = None              # Will hold the target invariant name.
        self.hypothesis_str = ""        # Current hypothesis being processed.

    def _train_nn_and_get_probability_distribution(self, df, target_invariant):
        # Select numerical features excluding the target invariant.
        df = df.dropna(subset=[target_invariant])
        features = [col for col in self.numerical_columns if col != target_invariant]
        if len(features) == 0:
            return {}
        X = df[features].values
        y = df[target_invariant].values

        # Check for NaNs in the data.
        if np.isnan(X).any() or np.isnan(y).any():
            warnings.warn("Data contains NaN values. Please handle them before training.")
            return {}

        # Normalize the data.
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Build a simple MLP model with a reduced learning rate.
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(len(features),)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='mse')

        # Optionally add early stopping if needed.
        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

        # Train the model.
        history = model.fit(X, y, epochs=1000, batch_size=32, verbose=1, callbacks=[early_stop])

        # Use SHAP to explain model predictions.
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        # Average the absolute SHAP values for each feature.
        avg_shap = np.abs(shap_values.values).mean(axis=0)

        # Check for NaN or zero-sum conditions and assign uniform probabilities if needed.
        if np.isnan(avg_shap).any() or np.sum(avg_shap) == 0:
            warnings.warn("SHAP values contain NaN or sum to zero; using uniform probability distribution.")
            probabilities = np.ones_like(avg_shap) / len(avg_shap)
        else:
            probabilities = avg_shap / np.sum(avg_shap)

        # Map each feature name to its corresponding probability.
        prob_dict = {feature: prob for feature, prob in zip(features, probabilities)}
        return prob_dict


    def _sample_invariants(self, prob_dict, k):
        """
        Samples k invariants from the provided probability distribution (without replacement).

        Args:
            prob_dict (dict): Mapping of invariant names to probability.
            k (int): Number of invariants to sample.

        Returns:
            list: Sampled invariant names.
        """
        keys = list(prob_dict.keys())
        # If k exceeds the available keys, adjust.
        if k > len(keys):
            k = len(keys)
        probs = np.array([prob_dict[key] for key in keys])
        sampled = np.random.choice(keys, size=k, replace=False, p=probs)
        return list(sampled)

    def _inequality_holds(self, candidate_series, target_series):
        """
        Checks whether the candidate bound holds on the entire dataset.

        Args:
            candidate_series (pd.Series): Candidate bound values.
            target_series (pd.Series): The target invariant values.

        Returns:
            bool: True if the inequality holds, False otherwise.
        """
        if self.bound_type == 'lower':
            return (target_series >= candidate_series).all()
        else:
            return (target_series <= candidate_series).all()

    def _compute_current_bound(self, target_series):
        """
        Computes the current best bound from accepted conjectures. If none are accepted,
        returns a series of -inf (for lower bound) or +inf (for upper bound).

        Args:
            target_series (pd.Series): The target invariant values.

        Returns:
            pd.Series: The current bound values.
        """
        if not self.accepted_conjectures:
            return pd.Series(-np.inf if self.bound_type == 'lower' else np.inf,
                             index=target_series.index)
        bounds = []
        for conj in self.accepted_conjectures:
            try:
                b = conj['func'](self.knowledge_table)
                bounds.append(b)
            except Exception as e:
                warnings.warn(f"Error computing bound from conjecture '{conj['full_expr_str']}': {e}")
        if len(bounds) == 0:
            return pd.Series(-np.inf if self.bound_type == 'lower' else np.inf,
                             index=target_series.index)
        df_bounds = pd.concat(bounds, axis=1)
        if self.bound_type == 'lower':
            return df_bounds.max(axis=1)
        else:
            return df_bounds.min(axis=1)

    def _is_significant(self, candidate_series, target_series):
        """
        Checks if the candidate bound is significantly better than the current bound.

        Args:
            candidate_series (pd.Series): Candidate bound values.
            target_series (pd.Series): The target invariant values.

        Returns:
            bool: True if the candidate is significant, False otherwise.
        """
        current_bound = self._compute_current_bound(target_series)
        if self.bound_type == 'lower':
            diff = candidate_series - current_bound
        else:
            diff = current_bound - candidate_series
        return (diff > 0).any()

    def _record_conjecture(self, complexity, rhs_str, func):
        """
        Records an accepted conjecture.

        Args:
            complexity (int): Complexity level (number of invariants used).
            rhs_str (str): A string representation of the right-hand side expression.
            func (callable): A function that, given a DataFrame, returns a series representing
                             the candidate bound.
        """
        if self.hypothesis_str:
            if self.bound_type == 'lower':
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≥ {rhs_str}."
            else:
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} ≤ {rhs_str}."
        else:
            full_expr_str = f"{self.target} ≥ {rhs_str}" if self.bound_type == 'lower' else f"{self.target} ≤ {rhs_str}"
        new_conj = {
            'complexity': complexity,
            'rhs_str': rhs_str,
            'full_expr_str': full_expr_str,
            'func': func,
            'bound_type': self.bound_type
        }
        try:
            candidate_series = func(self.knowledge_table)
        except Exception as e:
            warnings.warn(f"Error evaluating candidate during record: {e}")
            candidate_series = None
        touches = int((self.knowledge_table[self.target] == candidate_series).sum()) if candidate_series is not None else 0
        new_conj['touch'] = touches
        self.accepted_conjectures.append(new_conj)
        print(f"Accepted conjecture (complexity {complexity}, touch {touches}): {full_expr_str}")
        self._prune_conjectures()

    def _prune_conjectures(self):
        """
        Prunes dominated conjectures from the accepted list.
        """
        new_conjectures = []
        removed_conjectures = []
        n = len(self.accepted_conjectures)
        for i in range(n):
            conj_i = self.accepted_conjectures[i]
            try:
                series_i = conj_i['func'](self.knowledge_table)
            except Exception as e:
                warnings.warn(f"Error evaluating conjecture for pruning: {e}")
                continue
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                try:
                    series_j = self.accepted_conjectures[j]['func'](self.knowledge_table)
                except Exception as e:
                    continue
                if self.bound_type == 'lower':
                    if ((series_j >= series_i).all() and (series_j > series_i).any()):
                        dominated = True
                        break
                else:
                    if ((series_j <= series_i).all() and (series_j < series_i).any()):
                        dominated = True
                        break
            if not dominated:
                new_conjectures.append(conj_i)
            else:
                removed_conjectures.append(conj_i)
        if removed_conjectures:
            print("Pruning conjectures:")
            for rem in removed_conjectures:
                print("Removed:", rem['full_expr_str'])
        self.accepted_conjectures = new_conjectures

    def conjecture(self, target_invariant=None, hypothesis=None, complexity_range=(1, 3), time_limit=10):
        """
        Generate conjectures for tight upper/lower bounds on the target invariant.

        Keyword Args:
            target_invariant (str): The column name of the numerical invariant to bound.
            hypothesis (list of str): List of boolean column names to use as hypotheses.
            complexity_range (tuple): A tuple (lower_combn, upper_combn) defining the range of complexity.
            time_limit (int): Maximum running time in minutes.
        """
        if target_invariant is None or hypothesis is None:
            raise ValueError("Both 'target_invariant' and 'hypothesis' must be provided.")
        self.target = target_invariant
        start_time = time.time()

        # Loop over each boolean hypothesis.
        for hyp in hypothesis:
            self.hypothesis_str = hyp
            # Filter rows where the current hypothesis is True.
            filtered_df = self.knowledge_table[self.knowledge_table[hyp] == True]
            if filtered_df.empty:
                print(f"No rows satisfy hypothesis '{hyp}'. Skipping.")
                continue

            # Train the neural network and get the probability distribution over invariants.
            prob_dict = self._train_nn_and_get_probability_distribution(filtered_df, target_invariant)
            if not prob_dict:
                print(f"No numerical invariants available for hypothesis '{hyp}' after excluding target '{target_invariant}'.")
                continue

            # Loop over complexity levels.
            for k in range(complexity_range[0], complexity_range[1] + 1):
                # Check if we have exceeded the time limit.
                if time.time() - start_time > time_limit * 60:
                    print("Time limit reached. Stopping conjecture generation.")
                    return

                try:
                    sampled_invariants = self._sample_invariants(prob_dict, k)
                except Exception as e:
                    warnings.warn(f"Sampling error for complexity {k} under hypothesis '{hyp}': {e}")
                    continue

                # Generate a candidate bound expression using the sampled invariants.
                # For demonstration, we define the candidate as:
                #   - lower bound: the minimum of the sampled invariants.
                #   - upper bound: the maximum of the sampled invariants.
                if self.bound_type == 'lower':
                    candidate_func = lambda df, inv=sampled_invariants: df[inv].min(axis=1)
                    rhs_str = "min(" + ", ".join(sampled_invariants) + ")"
                else:
                    candidate_func = lambda df, inv=sampled_invariants: df[inv].max(axis=1)
                    rhs_str = "max(" + ", ".join(sampled_invariants) + ")"

                # Evaluate the candidate on the entire dataset.
                target_series = self.knowledge_table[target_invariant]
                candidate_series = candidate_func(self.knowledge_table)

                # Validate the candidate bound.
                if not self._inequality_holds(candidate_series, target_series):
                    print(f"Candidate bound '{rhs_str}' does not hold under hypothesis '{hyp}'.")
                    continue
                if not self._is_significant(candidate_series, target_series):
                    print(f"Candidate bound '{rhs_str}' is not significant under hypothesis '{hyp}'.")
                    continue

                # Record the candidate as an accepted conjecture.
                self._record_conjecture(k, rhs_str, candidate_func)

    def write_on_the_wall(self):
        """
        Display the accepted conjectures.
        """
        if not self.accepted_conjectures:
            print("No conjectures generated.")
            return
        print("Accepted Conjectures:")
        for conj in self.accepted_conjectures:
            print(conj['full_expr_str'])
