
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm  # progress bar
from fractions import Fraction  # for representing ratios
import time  # for the time limit
from tqdm import tqdm


from graffitiai import TxGraffiti
from graffitiai.experimental.expression_generator import generate_expressions, simplify

__all__ = [
    "GraffitiII",
    "Graffiti",
    "Christy",
]


class GraffitiII(TxGraffiti):

    def __init__(self):
        super().__init__()


    def conjecture(
            self,
            target_invariant,
            other_invariants = [],
            hypothesis = None,
            constants = [1, 2],
            search_depth=2,
            touch_number_threshold=2,
    ):

        df = self.knowledge_table

        # Generate and simplify the expressions.
        expressions = generate_expressions(other_invariants, max_depth=search_depth, constants=constants)
        simplified_expressions = [simplify(expr) for expr in expressions]

        # Remove duplicates.
        unique_expressions = list(set(simplified_expressions))

        if hypothesis is not None:
            df = df[df[hypothesis]]

        upper_bound_expressions = []
        lower_bound_expressions = []
        for expr in unique_expressions:
            try:
                evaluated = expr.eval(df)
                # Ensure evaluated is a Series.
                if np.isscalar(evaluated):
                    evaluated = pd.Series(evaluated, index=df.index)
                # Check if the expression is an upper bound.
                if (df[target_invariant] <= evaluated).all():
                    upper_bound_expressions.append(expr)
                # Check if the expression is a lower bound.
                if (df[target_invariant] >= evaluated).all():
                    lower_bound_expressions.append(expr)
            except Exception as e:
                print(f"Skipping expression {expr} due to evaluation error: {e}")

        upper_bound_conjectures = []
        for expr in upper_bound_expressions:
            try:
                evaluated = expr.eval(df)
                if np.isscalar(evaluated):
                    evaluated = pd.Series(evaluated, index=df.index)
                # Identify rows where the evaluated expression equals the target_invariant.
                touch_mask = (df[target_invariant] == evaluated)
                touch_number = touch_mask.sum()
                touch_set = list(df.index[touch_mask])
                # Create the conjecture tuple.
                conjecture = (target_invariant, "<=", expr, touch_number, touch_set)
                upper_bound_conjectures.append(conjecture)
            except Exception as e:
                # print(f"Skipping conjecture for expression {expr} due to evaluation error: {e}")
                pass # Skip the conjecture if evaluation fails.

        upper_bound_conjectures.sort(key=lambda x: x[3], reverse=True)
        upper_bound_conjectures = [conj for conj in upper_bound_conjectures if conj[3] >= touch_number_threshold]

        final_upper_bound_conjectures = []
        instances = df.index.tolist()
        instances = set(instances)

        # print("------------------------")
        # print("GRAFFITI II Upper Bound Conjectures:")
        # print("------------------------")
        for conjecture in upper_bound_conjectures:
            target_invariant, bound, expr, touch_number, touch_set = conjecture
            touch_set = set(touch_set)
            if not instances.intersection(touch_set) == set():
                # print(f"Conjecture. For every {hypothesis}, {target_invariant} {bound} {expr} | Touch Number: {touch_number} \n")
                final_upper_bound_conjectures.append(conjecture)
                instances = instances - touch_set
        print()

        lower_bound_conjectures = []
        for expr in lower_bound_expressions:
            try:
                evaluated = expr.eval(df)
                if np.isscalar(evaluated):
                    evaluated = pd.Series(evaluated, index=df.index)
                # Identify rows where the evaluated expression equals the target_invariant.
                touch_mask = (df[target_invariant] == evaluated)
                touch_number = touch_mask.sum()
                touch_set = list(df.index[touch_mask])
                # Create the conjecture tuple.
                conjecture = (target_invariant, ">=", expr, touch_number, touch_set)
                lower_bound_conjectures.append(conjecture)
            except Exception as e:
                # print(f"Skipping conjecture for expression {expr} due to evaluation error: {e}")
                pass # Skip the expression if it cannot be evaluated.

        lower_bound_conjectures.sort(key=lambda x: x[3], reverse=True)
        lower_bound_conjectures = [conj for conj in lower_bound_conjectures if conj[3] >= touch_number_threshold]

        final_lower_bound_conjectures = []
        instances = df.index.tolist()
        instances = set(instances)

        # print("------------------------")
        # print("GRAFFITI II Lower Bound Conjectures:")
        # print("------------------------")
        for conjecture in lower_bound_conjectures:
            target_invariant, bound, expr, touch_number, touch_set = conjecture
            touch_set = set(touch_set)
            if not instances.intersection(touch_set) == set():
                # print(f"Conjecture. For every {hypothesis}, {target_invariant} {bound} {expr} | Touch Number: {touch_number} \n")
                final_lower_bound_conjectures.append(conjecture)
                instances = instances - touch_set
        conjectures = {"upper": final_upper_bound_conjectures, "lower": final_lower_bound_conjectures}
        self.conjectures[target_invariant] = conjectures

    def write_on_the_wall(self, target_invariant):
        print("------------------------")
        print(f"GRAFFITI II {target_invariant} Conjectures:")
        print("------------------------")
        for bound in ["upper", "lower"]:
            print(f"{bound.capitalize()} Bound Conjectures:")
            for conjecture in self.conjectures[target_invariant][bound]:
                target_invariant, bound, expr, touch_number, touch_set = conjecture
                print(f"Conjecture. {target_invariant} {bound} {expr} | Touch Number: {touch_number} \n")
            print()






class Graffiti:

    def __init__(self, df, target_invariant, bound_type='lower', filter_property=None, time_limit=None):
        """
        Parameters:
          df: pandas DataFrame containing the invariants and boolean properties.
          target_invariant: name of the column whose bound we wish to conjecture.
          bound_type: 'lower' (interpreted as target >= candidate) or 'upper' (target <= candidate).
          filter_property: optional boolean column name; if provided, only rows with True are used.
          time_limit: maximum search time in seconds (or None for no limit).
        """
        self.df_full = df.copy()
        if filter_property is not None:
            self.df = df[df[filter_property] == True].copy()
            self.hypothesis_str = filter_property
        else:
            self.df = df.copy()
            self.hypothesis_str = None

        self.target = target_invariant
        self.bound_type = bound_type
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
            Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
        ]

    # ---------------------------------------------------------------------------
    # Helper: Compute the current bound from accepted conjectures.
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Check if the candidate inequality holds.
    # ---------------------------------------------------------------------------
    def _inequality_holds(self, candidate_series):
        target_series = self.df[self.target]
        if self.bound_type == 'lower':
            return (target_series >= candidate_series).all()
        else:
            return (target_series <= candidate_series).all()

    # ---------------------------------------------------------------------------
    # Check if the candidate is significant (i.e. improves over current bound).
    # ---------------------------------------------------------------------------
    def _is_significant(self, candidate_series):
        current_bound = self._compute_current_bound()
        if self.bound_type == 'lower':
            diff = candidate_series - current_bound
        else:
            diff = current_bound - candidate_series
        return (diff > 0).any()

    # ---------------------------------------------------------------------------
    # Record a new conjecture.
    # ---------------------------------------------------------------------------
    def _record_conjecture(self, complexity, rhs_str, func):
        if self.hypothesis_str:
            if self.bound_type == 'lower':
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} >= {rhs_str}."
            else:
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} <= {rhs_str}."
        else:
            full_expr_str = f"{self.target} >= {rhs_str}" if self.bound_type == 'lower' else f"{self.target} <= {rhs_str}"
        new_conj = {
            'complexity': complexity,
            'rhs_str': rhs_str,
            'full_expr_str': full_expr_str,
            'func': func
        }
        # Compute the "touch" (number of rows where equality holds).
        candidate_series = func(self.df)
        touches = int((self.df[self.target] == candidate_series).sum())
        new_conj['touch'] = touches

        self.accepted_conjectures.append(new_conj)
        print(f"Accepted conjecture (complexity {complexity}, touch {touches}): {full_expr_str}")
        self._prune_conjectures()

    # ---------------------------------------------------------------------------
    # Prune conjectures that are dominated by others.
    # ---------------------------------------------------------------------------
    def _prune_conjectures(self):
        new_conjectures = []
        removed_conjectures = []
        n = len(self.accepted_conjectures)
        for i in range(n):
            conj_i = self.accepted_conjectures[i]
            series_i = conj_i['func'](self.df)
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                series_j = self.accepted_conjectures[j]['func'](self.df)
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

    # ---------------------------------------------------------------------------
    # -- Transformation helper functions --
    # Each takes a candidate (rhs_str, func) and yields variants.
    # ---------------------------------------------------------------------------
    def _with_floor_ceil(self, candidate):
        base_rhs, base_func = candidate
        yield (f"floor({base_rhs})", lambda df, base_func=base_func: np.floor(base_func(df)))
        yield (f"ceil({base_rhs})", lambda df, base_func=base_func: np.ceil(base_func(df)))

    def _with_ratio_addition(self, candidate):
        base_rhs, base_func = candidate
        for ratio in self.ratios:
            yield (f"({base_rhs}) + {ratio}",
                   lambda df, base_func=base_func, ratio=ratio: base_func(df) + float(ratio))

    def _with_ratio_subtraction(self, candidate):
        base_rhs, base_func = candidate
        for ratio in self.ratios:
            yield (f"({base_rhs}) - {ratio}",
                   lambda df, base_func=base_func, ratio=ratio: base_func(df) - float(ratio))

    def _with_ratio_multiplication(self, candidate):
        base_rhs, base_func = candidate
        for ratio in self.ratios:
            yield (f"{ratio}*({base_rhs})",
                   lambda df, base_func=base_func, ratio=ratio: float(ratio) * base_func(df))

    def _expand_candidate(self, candidate):
        """
        Given a base candidate (rhs_str, func), apply a set of transformations and
        return the list of unique candidate variants (based on their string representation).
        """
        variants = {candidate[0]: candidate}
        for transform in [self._with_floor_ceil, self._with_ratio_multiplication,
                          self._with_ratio_subtraction, self._with_ratio_addition]:
            for cand in transform(candidate):
                # Avoid duplicates (based on the expression string).
                if cand[0] not in variants:
                    variants[cand[0]] = cand
        return list(variants.values())

    # ---------------------------------------------------------------------------
    # -- Candidate Generation: Complexity 1 and 2 (Unary candidates) --
    # ---------------------------------------------------------------------------
    def _generate_candidates_complexity1(self):
        # Basic candidate: each invariant column by itself.
        return [(col, lambda df, col=col: df[col]) for col in self.candidate_cols]

    def _generate_candidates_unary(self, col):
        """
        Generate several unary transformations for a given column.
        Used in complexity 2.
        """
        candidates = []
        candidates.append((f"{col}", lambda df, col=col: df[col]))
        candidates.append((f"({col})^2", lambda df, col=col: df[col]**2))
        candidates.append((f"floor({col})", lambda df, col=col: np.floor(df[col])))
        candidates.append((f"ceil({col})", lambda df, col=col: np.ceil(df[col])))
        for ratio in self.ratios:
            # Multiplication with ratio (with floor and ceil already handled later)
            expr = f"{ratio}*({col})"
            candidates.append((expr, lambda df, col=col, ratio=ratio: float(ratio) * df[col]))
            # Addition and subtraction variants.
            candidates.append((f"({col}) + {ratio}", lambda df, col=col, ratio=ratio: df[col] + float(ratio)))
            candidates.append((f"({col}) - {ratio}", lambda df, col=col, ratio=ratio: df[col] - float(ratio)))
        return candidates

    def _generate_candidates_complexity2(self):
        candidates = []
        for col in self.candidate_cols:
            candidates.extend(self._generate_candidates_unary(col))
        return candidates

    # ---------------------------------------------------------------------------
    # -- Candidate Generation: Complexity 3 and 4 (Binary candidates) --
    # ---------------------------------------------------------------------------
    def _generate_candidates_binary(self, col1, col2):
        base = [
            (f"({col1} + {col2})", lambda df, col1=col1, col2=col2: df[col1] + df[col2]),
            (f"({col1} - {col2})", lambda df, col1=col1, col2=col2: df[col1] - df[col2]),
            (f"({col2} - {col1})", lambda df, col1=col1, col2=col2: df[col2] - df[col1]),
            (f"{col1} * {col2}", lambda df, col1=col1, col2=col2: df[col1] * df[col2]),
            (f"max({col1}, {col2})", lambda df, col1=col1, col2=col2: np.maximum(df[col1], df[col2])),
            (f"min({col1}, {col2})", lambda df, col1=col1, col2=col2: np.minimum(df[col1], df[col2])),
            (f"abs({col1} - {col2})", lambda df, col1=col1, col2=col2: np.abs(df[col1] - df[col2])),
            (f"{col1}*{col2}", lambda df, col1=col1, col2=col2: df[col1] * df[col2]),
        ]
        # Add safe division candidates.
        if (self.df[col2] == 0).sum() == 0:
            base.append((f"({col1} / {col2})", lambda df, col1=col1, col2=col2: df[col1] / df[col2]))
        if (self.df[col1] == 0).sum() == 0:
            base.append((f"({col2} / {col1})", lambda df, col1=col1, col2=col2: df[col2] / df[col1]))
        return base

    def _generate_candidates_complexity3(self):
        candidates = []
        for col1, col2 in combinations(self.candidate_cols, 2):
            for cand in self._generate_candidates_binary(col1, col2):
                candidates.extend(self._expand_candidate(cand))
        return candidates

    def _generate_candidates_binary_complex4(self, col1, col2):
        base = [
            (f"({col1} + {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] + df[col2])**2),
            (f"({col1} - {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] - df[col2])**2)
        ]
        # Only allow sqrt if the sum is always nonnegative.
        if (self.df[col1] + self.df[col2] < 0).sum() == 0:
            base.append((f"sqrt({col1} + {col2})", lambda df, col1=col1, col2=col2: np.sqrt(df[col1] + df[col2])))
        return base

    def _generate_candidates_complexity4(self):
        candidates = []
        for col1, col2 in combinations(self.candidate_cols, 2):
            for cand in self._generate_candidates_binary_complex4(col1, col2):
                candidates.extend(self._expand_candidate(cand))
        return candidates

    # ---------------------------------------------------------------------------
    # -- Candidate Generation: Complexity 5 (Powers of accepted conjectures) --
    # ---------------------------------------------------------------------------
    def _generate_candidates_complexity5(self):
        candidates = []
        if not self.accepted_conjectures:
            return candidates
        for accepted in self.accepted_conjectures:
            for exponent in [2, 3]:
                new_rhs = f"({accepted['rhs_str']})^{exponent}"
                func = lambda df, f_old=accepted['func'], exponent=exponent: f_old(df) ** exponent
                candidates.append((new_rhs, func))
        return candidates

    # ---------------------------------------------------------------------------
    # -- Candidate Generation: Complexity 6 (Using constants) --
    # ---------------------------------------------------------------------------
    def _generate_candidates_complexity6(self):
        candidates = []
        for col in self.candidate_cols:
            for c in self.ratios:
                expr = f"{c}*({col})"
                func = lambda df, col=col, c=c: float(c) * df[col]
                candidates.append((expr, func))
        return candidates

    # ---------------------------------------------------------------------------
    # -- Candidate Generation: Complexity 7 (Recursive combinations) --
    # ---------------------------------------------------------------------------
    def _generate_candidates_recursive(self, max_depth):
        if max_depth == 1:
            return self._generate_candidates_complexity1()
        else:
            prev_candidates = self._generate_candidates_recursive(max_depth - 1)
            atomic_candidates = self._generate_candidates_complexity1()
            new_candidates = []
            operators = [
                ("+", lambda a, b: a + b),
                ("-", lambda a, b: a - b),
                ("*", lambda a, b: a * b)
                # Note: Division can be added with care.
            ]
            for cand1 in prev_candidates:
                for cand2 in atomic_candidates:
                    for op_sym, op_func in operators:
                        new_expr = f"({cand1[0]}) {op_sym} ({cand2[0]})"
                        new_func = lambda df, f1=cand1[1], f2=cand2[1], op=op_func: op(f1(df), f2(df))
                        new_candidates.append((new_expr, new_func))
            return prev_candidates + new_candidates

    def _generate_candidates_complexity7(self):
        rec_candidates = self._generate_candidates_recursive(max_depth=3)
        candidates = []
        for expr, func in rec_candidates:
            # Use a rough measure of “complexity” (number of operators)
            op_count = expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/')
            if op_count >= 2:
                candidates.extend(self._expand_candidate((expr, func)))
        return candidates

    # ---------------------------------------------------------------------------
    # Public candidate-generation entry point.
    # ---------------------------------------------------------------------------
    def generate_candidates(self, complexity):
        if complexity == 1:
            return self._generate_candidates_complexity1()
        elif complexity == 2:
            return self._generate_candidates_complexity2()
        elif complexity == 3:
            return self._generate_candidates_complexity3()
        elif complexity == 4:
            return self._generate_candidates_complexity4()
        elif complexity == 5:
            return self._generate_candidates_complexity5()
        elif complexity == 6:
            return self._generate_candidates_complexity6()
        elif complexity == 7:
            return self._generate_candidates_complexity7()
        else:
            return []

    # ---------------------------------------------------------------------------
    # Main search loop.
    # ---------------------------------------------------------------------------
    def search(self):
        start_time = time.time()
        new_conjecture_found = True
        while new_conjecture_found:
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                print("Time limit reached. Halting search.")
                break
            new_conjecture_found = False
            for complexity in range(1, self.max_complexity + 1):
                candidates = self.generate_candidates(complexity)
                for rhs_str, func in tqdm(candidates, desc=f"Complexity {complexity}", leave=False):
                    if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                        print("Time limit reached during candidate evaluation. Halting search.")
                        new_conjecture_found = False
                        break
                    try:
                        candidate_series = func(self.df)
                    except Exception as e:
                        print(f"Skipping candidate {rhs_str} due to error: {e}")
                        continue
                    if not self._inequality_holds(candidate_series):
                        continue
                    if not self._is_significant(candidate_series):
                        continue
                    self._record_conjecture(complexity, rhs_str, func)
                    new_conjecture_found = True
                    break
                if new_conjecture_found:
                    break
            if not new_conjecture_found:
                print("No further significant conjectures found within the maximum complexity.")
                break

    # ---------------------------------------------------------------------------
    # Get accepted conjectures sorted by "touch" (descending).
    # ---------------------------------------------------------------------------
    def get_accepted_conjectures(self):
        return sorted(self.accepted_conjectures, key=lambda c: c['touch'], reverse=True)

    # ---------------------------------------------------------------------------
    # Write conjectures to the “wall.”
    # ---------------------------------------------------------------------------
    def write_on_the_wall(self):
        print("GRAFFITI conjectures:")
        print("------------------------")
        for conj in self.get_accepted_conjectures():
            print(f"Conjecture: {conj['full_expr_str']} (touch {conj['touch']})")


# import pandas as pd
# import numpy as np
# import time
# from itertools import combinations


# class Christy:
#     def __init__(self, df, target, candidate_antecedents=None, candidate_properties=None, time_limit=None):
#         self.df = df.copy()
#         self.target = target
#         self.time_limit = time_limit

#         # If candidate antecedents not provided, generate from numeric (non-boolean) columns.
#         if candidate_antecedents is None:
#             num_cols = [col for col in df.columns
#                         if col != target
#                         and pd.api.types.is_numeric_dtype(df[col])
#                         and not pd.api.types.is_bool_dtype(df[col])]
#             # Base antecedents: the raw columns.
#             base_candidates = [(col, lambda df, col=col: df[col]) for col in num_cols]

#             # Define self.ratios (as in Graffiti)
#             self.ratios = [
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
#                 Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
#             ]

#             # Generate additional candidates:
#             # 1. Multiplication: ratio * (column)
#             # 2. Addition: (column) + ratio
#             ratio_candidates = []
#             for col in num_cols:
#                 for ratio in self.ratios:
#                     # Multiplication candidate.
#                     ratio_candidates.append((
#                         f"{ratio}*({col})",
#                         lambda df, col=col, ratio=ratio: float(ratio) * df[col]
#                     ))
#                     # Addition candidate.
#                     ratio_candidates.append((
#                         f"({col}) + {ratio}",
#                         lambda df, col=col, ratio=ratio: df[col] + float(ratio)
#                     ))
#             self.candidate_antecedents = base_candidates + ratio_candidates
#         else:
#             if not hasattr(candidate_antecedents, '__iter__'):
#                 raise ValueError("candidate_antecedents must be an iterable of (expression, function) tuples.")
#             self.candidate_antecedents = candidate_antecedents

#         # Determine candidate boolean properties.
#         if candidate_properties is None:
#             candidate_props = []
#             # (1) Use boolean columns.
#             bool_cols = [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]
#             for col in bool_cols:
#                 candidate_props.append((col, lambda df, col=col: df[col]))
#             # (2) Also add candidate equalities between numeric invariants.
#             num_cols = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
#             for col1, col2 in combinations(num_cols, 2):
#                 expr = f"({col1} == {col2})"
#                 candidate_props.append((expr, lambda df, col1=col1, col2=col2: df[col1] == df[col2]))
#             self.candidate_properties = candidate_props
#         else:
#             self.candidate_properties = candidate_properties

#         # List of accepted conjectures.
#         self.accepted_conjectures = []

#     def _implication_holds(self, antecedent_series, prop_series):
#         """
#         Checks whether for every row where target >= antecedent holds,
#         the property is True.
#         Returns False if no rows satisfy the antecedent.
#         """
#         condition = self.df[self.target] >= antecedent_series
#         if condition.sum() == 0:
#             return False
#         return prop_series[condition].all()

#     def _record_conjecture(self, ant_str, prop_str, ant_func, prop_func):
#         """
#         Record the conjecture along with its support (the number of rows where
#         the antecedent fires) and print it.
#         """
#         ant_series = ant_func(self.df)
#         support = int((self.df[self.target] >= ant_series).sum())
#         full_expr = f"If {self.target} >= {ant_str}, then {prop_str}  [support: {support}]"
#         new_conj = {
#             "antecedent_str": ant_str,
#             "prop_str": prop_str,
#             "full_expr": full_expr,
#             "ant_func": ant_func,
#             "prop_func": prop_func,
#             "support": support,
#         }
#         self.accepted_conjectures.append(new_conj)
#         print("Accepted conjecture:", full_expr)
#         self._prune_conjectures()

#     def _prune_conjectures(self):
#         """
#         If two conjectures have the same property (consequent) but different antecedents,
#         keep only the one with the higher support.
#         """
#         pruned = []
#         unique_conjs = {}
#         for conj in self.accepted_conjectures:
#             key = conj["prop_str"]
#             if key in unique_conjs:
#                 if conj["support"] > unique_conjs[key]["support"]:
#                     pruned.append(unique_conjs[key])
#                     unique_conjs[key] = conj
#                 else:
#                     pruned.append(conj)
#             else:
#                 unique_conjs[key] = conj
#         if pruned:
#             print("Pruned conjectures:")
#             for p in pruned:
#                 print("Removed:", p["full_expr"])
#         self.accepted_conjectures = list(unique_conjs.values())

#     def search(self):
#         """
#         Loop over candidate antecedents and candidate properties.
#         For each pair, if for every row where target >= antecedent holds the property is True,
#         record the rule (conjecture) along with its support.
#         """
#         import time  # For time limit checks.
#         start_time = time.time()
#         for ant_str, ant_func in self.candidate_antecedents:
#             try:
#                 ant_series = ant_func(self.df)
#             except Exception as e:
#                 print(f"Error evaluating antecedent '{ant_str}':", e)
#                 continue

#             for prop_str, prop_func in self.candidate_properties:
#                 try:
#                     prop_series = prop_func(self.df)
#                 except Exception as e:
#                     print(f"Error evaluating property '{prop_str}':", e)
#                     continue

#                 if self._implication_holds(ant_series, prop_series):
#                     self._record_conjecture(ant_str, prop_str, ant_func, prop_func)

#                 if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                     print("Time limit reached, stopping search.")
#                     return

#     def get_accepted_conjectures(self):
#         return self.accepted_conjectures

#     def write_on_the_wall(self):
#         print("Christy conjectures:")
#         print("--------------------")
#         for conj in self.accepted_conjectures:
#             print(conj["full_expr"])

# from fractions import Fraction
# from itertools import combinations
# import pandas as pd
# import numpy as np

# class Christy:
#     def __init__(self, df, target, candidate_antecedents=None, candidate_properties=None, time_limit=None):
#         self.df = df.copy()
#         self.target = target
#         self.time_limit = time_limit

#         # If candidate antecedents are not provided, we generate them.
#         if candidate_antecedents is None:
#             # Use numeric columns (excluding booleans and the target)
#             num_cols = [col for col in df.columns
#                         if col != target
#                         and pd.api.types.is_numeric_dtype(df[col])
#                         and not pd.api.types.is_bool_dtype(df[col])]

#             # Base candidates: the raw columns.
#             base_candidates = [(col, lambda df, col=col: df[col]) for col in num_cols]

#             # Define self.ratios exactly as in Graffiti.
#             self.ratios = [
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
#                 Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
#             ]

#             # Ratio candidates: multiply, add, and subtract each column by a constant.
#             ratio_candidates = []
#             for col in num_cols:
#                 for ratio in self.ratios:
#                     ratio_candidates.append((
#                         f"{ratio}*({col})",
#                         lambda df, col=col, ratio=ratio: float(ratio) * df[col]
#                     ))
#                     ratio_candidates.append((
#                         f"({col}) + {ratio}",
#                         lambda df, col=col, ratio=ratio: df[col] + float(ratio)
#                     ))
#                     ratio_candidates.append((
#                         f"({col}) - {ratio}",
#                         lambda df, col=col, ratio=ratio: df[col] - float(ratio)
#                     ))

#             # Complexity3 hypothesis candidates: combine pairs of columns with various operations.
#             complexity3_candidates = self._generate_candidates_complexity3_hypothesis(num_cols)

#             # Final list of candidate antecedents.
#             self.candidate_antecedents = base_candidates + ratio_candidates + complexity3_candidates
#         else:
#             if not hasattr(candidate_antecedents, '__iter__'):
#                 raise ValueError("candidate_antecedents must be an iterable of (expression, function) tuples.")
#             self.candidate_antecedents = candidate_antecedents

#         # Candidate boolean properties:
#         if candidate_properties is None:
#             candidate_props = []
#             # (1) Boolean columns.
#             bool_cols = [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]
#             for col in bool_cols:
#                 candidate_props.append((col, lambda df, col=col: df[col]))
#             # (2) Candidate equalities between numeric columns.
#             num_cols_all = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
#             for col1, col2 in combinations(num_cols_all, 2):
#                 expr = f"({col1} == {col2})"
#                 candidate_props.append((expr, lambda df, col1=col1, col2=col2: df[col1] == df[col2]))
#             self.candidate_properties = candidate_props
#         else:
#             self.candidate_properties = candidate_properties

#         # List to store accepted conjectures.
#         self.accepted_conjectures = []

#     # --- Transformation functions for applying self.ratios ---
#     def _with_ratio_addition(self, candidate):
#         base_rhs, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"({base_rhs}) + {ratio}",
#                    lambda df, base_func=base_func, ratio=ratio: base_func(df) + float(ratio))
#     def _with_ratio_subtraction(self, candidate):
#         base_rhs, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"({base_rhs}) - {ratio}",
#                    lambda df, base_func=base_func, ratio=ratio: base_func(df) - float(ratio))
#     def _with_ratio_multiplication(self, candidate):
#         base_rhs, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"{ratio}*({base_rhs})",
#                    lambda df, base_func=base_func, ratio=ratio: float(ratio) * base_func(df))
#     def _expand_candidate(self, candidate):
#         """
#         Given a base candidate (expression, function), apply the ratio-based
#         transformations and return the list of unique candidate variants.
#         """
#         variants = {candidate[0]: candidate}
#         for transform in [self._with_ratio_addition, self._with_ratio_subtraction, self._with_ratio_multiplication]:
#             for cand in transform(candidate):
#                 if cand[0] not in variants:
#                     variants[cand[0]] = cand
#         return list(variants.values())

#     # --- Complexity3 hypothesis candidate generation ---
#     def _generate_candidates_complexity3_hypothesis(self, num_cols):
#         """
#         For each pair of numeric columns, generate candidates using:
#           - Multiplication: x*y
#           - Addition: x+y
#           - Division: x/y (if denominator is safe) and y/x (if safe)
#           - Minimum and Maximum: min(x, y) and max(x, y)
#         Then, expand each candidate by applying ratio-based transformations.
#         """
#         candidates = []
#         for col1, col2 in combinations(num_cols, 2):
#             # Multiplication candidate.
#             candidates.append((
#                 f"({col1} * {col2})",
#                 lambda df, col1=col1, col2=col2: df[col1] * df[col2]
#             ))
#             # Addition candidate.
#             candidates.append((
#                 f"({col1} + {col2})",
#                 lambda df, col1=col1, col2=col2: df[col1] + df[col2]
#             ))
#             # Division candidates (if safe).
#             if (self.df[col2] == 0).sum() == 0:
#                 candidates.append((
#                     f"({col1} / {col2})",
#                     lambda df, col1=col1, col2=col2: df[col1] / df[col2]
#                 ))
#             if (self.df[col1] == 0).sum() == 0:
#                 candidates.append((
#                     f"({col2} / {col1})",
#                     lambda df, col1=col1, col2=col2: df[col2] / df[col1]
#                 ))
#             # Minimum candidate.
#             candidates.append((
#                 f"min({col1}, {col2})",
#                 lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).min(axis=1)
#             ))
#             # Maximum candidate.
#             candidates.append((
#                 f"max({col1}, {col2})",
#                 lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).max(axis=1)
#             ))
#         # For each base candidate, expand it using ratio-based transformations.
#         expanded = []
#         for cand in candidates:
#             expanded.extend(self._expand_candidate(cand))
#         return expanded

#     # --- Implication and Conjecture Recording ---
#     def _implication_holds(self, antecedent_series, prop_series):
#         condition = self.df[self.target] >= antecedent_series
#         if condition.sum() == 0:
#             return False
#         return prop_series[condition].all()

#     def _record_conjecture(self, ant_str, prop_str, ant_func, prop_func):
#         ant_series = ant_func(self.df)
#         support = int((self.df[self.target] >= ant_series).sum())
#         full_expr = f"If {self.target} >= {ant_str}, then {prop_str}  [support: {support}]"
#         new_conj = {
#             "antecedent_str": ant_str,
#             "prop_str": prop_str,
#             "full_expr": full_expr,
#             "ant_func": ant_func,
#             "prop_func": prop_func,
#             "support": support,
#         }
#         self.accepted_conjectures.append(new_conj)
#         print("Accepted conjecture:", full_expr)
#         self._prune_conjectures()

#     def _prune_conjectures(self):
#         pruned = []
#         unique_conjs = {}
#         for conj in self.accepted_conjectures:
#             key = conj["prop_str"]
#             if key in unique_conjs:
#                 if conj["support"] > unique_conjs[key]["support"]:
#                     pruned.append(unique_conjs[key])
#                     unique_conjs[key] = conj
#                 else:
#                     pruned.append(conj)
#             else:
#                 unique_conjs[key] = conj
#         if pruned:
#             print("Pruned conjectures:")
#             for p in pruned:
#                 print("Removed:", p["full_expr"])
#         self.accepted_conjectures = list(unique_conjs.values())

#     def search(self):
#         import time
#         start_time = time.time()
#         for ant_str, ant_func in self.candidate_antecedents:
#             try:
#                 ant_series = ant_func(self.df)
#             except Exception as e:
#                 print(f"Error evaluating antecedent '{ant_str}':", e)
#                 continue
#             for prop_str, prop_func in self.candidate_properties:
#                 try:
#                     prop_series = prop_func(self.df)
#                 except Exception as e:
#                     print(f"Error evaluating property '{prop_str}':", e)
#                     continue
#                 if self._implication_holds(ant_series, prop_series):
#                     self._record_conjecture(ant_str, prop_str, ant_func, prop_func)
#                 if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                     print("Time limit reached, stopping search.")
#                     return

#     def get_accepted_conjectures(self):
#         return self.accepted_conjectures

#     def write_on_the_wall(self):
#         print("Christy conjectures:")
#         print("--------------------")
#         for conj in self.accepted_conjectures:
#             print(conj["full_expr"])

# from fractions import Fraction
# from itertools import combinations
# import pandas as pd
# import numpy as np

# class Christy:
#     def __init__(self, df, target, bound_type='lower', candidate_antecedents=None, candidate_properties=None, time_limit=None):
#         self.df = df.copy()
#         self.target = target
#         self.bound_type = bound_type  # 'lower' for target >= antecedent, 'upper' for target <= antecedent
#         self.time_limit = time_limit

#         # Generate candidate antecedents if not provided.
#         if candidate_antecedents is None:
#             num_cols = [col for col in df.columns
#                         if col != target and pd.api.types.is_numeric_dtype(df[col])
#                         and not pd.api.types.is_bool_dtype(df[col])]
#             base_candidates = [(col, lambda df, col=col: df[col]) for col in num_cols]

#             self.ratios = [
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
#                 Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
#             ]

#             ratio_candidates = []
#             for col in num_cols:
#                 for ratio in self.ratios:
#                     ratio_candidates.append((
#                         f"{ratio}*({col})",
#                         lambda df, col=col, ratio=ratio: float(ratio) * df[col]
#                     ))
#                     ratio_candidates.append((
#                         f"({col}) + {ratio}",
#                         lambda df, col=col, ratio=ratio: df[col] + float(ratio)
#                     ))
#                     ratio_candidates.append((
#                         f"({col}) - {ratio}",
#                         lambda df, col=col, ratio=ratio: df[col] - float(ratio)
#                     ))

#             complexity3_candidates = self._generate_candidates_complexity3_hypothesis(num_cols)

#             self.candidate_antecedents = base_candidates + ratio_candidates + complexity3_candidates
#         else:
#             if not hasattr(candidate_antecedents, '__iter__'):
#                 raise ValueError("candidate_antecedents must be an iterable of (expression, function) tuples.")
#             self.candidate_antecedents = candidate_antecedents

#         # Candidate boolean properties (unchanged):
#         if candidate_properties is None:
#             candidate_props = []
#             bool_cols = [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]
#             for col in bool_cols:
#                 candidate_props.append((col, lambda df, col=col: df[col]))
#             num_cols_all = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])
#                              and not pd.api.types.is_bool_dtype(df[col])]
#             for col1, col2 in combinations(num_cols_all, 2):
#                 expr = f"({col1} = {col2})"
#                 candidate_props.append((expr, lambda df, col1=col1, col2=col2: df[col1] == df[col2]))
#             self.candidate_properties = candidate_props
#         else:
#             self.candidate_properties = candidate_properties

#         self.accepted_conjectures = []

#     # (Transformation functions and complexity-3 candidate generation remain the same.)
#     def _with_ratio_addition(self, candidate):
#         base_expr, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"({base_expr}) + {ratio}",
#                    lambda df, base_func=base_func, ratio=ratio: base_func(df) + float(ratio))
#     def _with_ratio_subtraction(self, candidate):
#         base_expr, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"({base_expr}) - {ratio}",
#                    lambda df, base_func=base_func, ratio=ratio: base_func(df) - float(ratio))
#     def _with_ratio_multiplication(self, candidate):
#         base_expr, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"{ratio}*({base_expr})",
#                    lambda df, base_func=base_func, ratio=ratio: float(ratio) * base_func(df))
#     def _expand_candidate(self, candidate):
#         variants = {candidate[0]: candidate}
#         for transform in [self._with_ratio_addition, self._with_ratio_subtraction, self._with_ratio_multiplication]:
#             for cand in transform(candidate):
#                 if cand[0] not in variants:
#                     variants[cand[0]] = cand
#         return list(variants.values())
#     def _generate_candidates_complexity3_hypothesis(self, num_cols):
#         candidates = []
#         for col1, col2 in combinations(num_cols, 2):
#             candidates.append((
#                 f"({col1} * {col2})",
#                 lambda df, col1=col1, col2=col2: df[col1] * df[col2]
#             ))
#             candidates.append((
#                 f"({col1} + {col2})",
#                 lambda df, col1=col1, col2=col2: df[col1] + df[col2]
#             ))
#             if (self.df[col2] == 0).sum() == 0:
#                 candidates.append((
#                     f"({col1} / {col2})",
#                     lambda df, col1=col1, col2=col2: df[col1] / df[col2]
#                 ))
#             if (self.df[col1] == 0).sum() == 0:
#                 candidates.append((
#                     f"({col2} / {col1})",
#                     lambda df, col1=col1, col2=col2: df[col2] / df[col1]
#                 ))
#             candidates.append((
#                 f"min({col1}, {col2})",
#                 lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).min(axis=1)
#             ))
#             candidates.append((
#                 f"max({col1}, {col2})",
#                 lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).max(axis=1)
#             ))
#         expanded = []
#         for cand in candidates:
#             expanded.extend(self._expand_candidate(cand))
#         return expanded

#     # --- Implication Checking and Conjecture Recording ---
#     def _implication_holds(self, antecedent_series, prop_series):
#         if self.bound_type == 'lower':
#             condition = self.df[self.target] >= antecedent_series
#         else:  # 'upper'
#             condition = self.df[self.target] <= antecedent_series
#         if condition.sum() == 0:
#             return False
#         return prop_series[condition].all()

#     def _record_conjecture(self, ant_str, prop_str, ant_func, prop_func):
#         ant_series = ant_func(self.df)
#         if self.bound_type == 'lower':
#             support = int((self.df[self.target] >= ant_series).sum())
#             bound_symbol = ">="
#         else:
#             support = int((self.df[self.target] <= ant_series).sum())
#             bound_symbol = "<="
#         full_expr = f"If {self.target} {bound_symbol} {ant_str}, then {prop_str}  [support: {support}]"
#         new_conj = {
#             "antecedent_str": ant_str,
#             "prop_str": prop_str,
#             "full_expr": full_expr,
#             "ant_func": ant_func,
#             "prop_func": prop_func,
#             "support": support,
#         }
#         self.accepted_conjectures.append(new_conj)
#         print("Accepted conjecture:", full_expr)
#         self._prune_conjectures()

    # def _prune_conjectures(self):
    #     pruned = []
    #     unique_conjs = {}
    #     for conj in self.accepted_conjectures:
    #         key = conj["prop_str"]
    #         if key in unique_conjs:
    #             if conj["support"] > unique_conjs[key]["support"]:
    #                 pruned.append(unique_conjs[key])
    #                 unique_conjs[key] = conj
    #             else:
    #                 pruned.append(conj)
    #         else:
    #             unique_conjs[key] = conj
    #     if pruned:
    #         print("Pruned conjectures:")
    #         for p in pruned:
    #             print("Removed:", p["full_expr"])
    #     self.accepted_conjectures = list(unique_conjs.values())
    # def _prune_conjectures(self):
    #     """
    #     For conjectures with the same property (consequent), keep only the one with the highest support.
    #     If support is equal, then (optionally) choose the one with the best bound – that is,
    #     - for lower-bound hypotheses, the candidate whose antecedent (when evaluated on the full dataset)
    #         has the highest average value, and
    #     - for upper-bound hypotheses, the candidate whose antecedent has the lowest average value.
    #     """
    #     pruned = []
    #     unique_conjs = {}
    #     for conj in self.accepted_conjectures:
    #         key = conj["prop_str"]
    #         if key in unique_conjs:
    #             current = unique_conjs[key]
    #             if conj["support"] > current["support"]:
    #                 pruned.append(current)
    #                 unique_conjs[key] = conj
    #             elif conj["support"] == current["support"]:
    #                 # Compute a summary measure (e.g., mean) of the antecedent series.
    #                 current_bound = current["ant_func"](self.df).mean()
    #                 new_bound = conj["ant_func"](self.df).mean()
    #                 # For lower-bound hypotheses, higher bound is better;
    #                 # for upper-bound hypotheses, lower bound is better.
    #                 if self.bound_type == 'lower':
    #                     if new_bound > current_bound:
    #                         pruned.append(current)
    #                         unique_conjs[key] = conj
    #                     else:
    #                         pruned.append(conj)
    #                 else:  # self.bound_type == 'upper'
    #                     if new_bound < current_bound:
    #                         pruned.append(current)
    #                         unique_conjs[key] = conj
    #                     else:
    #                         pruned.append(conj)
    #             else:
    #                 pruned.append(conj)
    #         else:
    #             unique_conjs[key] = conj
    #     if pruned:
    #         print("Pruned conjectures:")
    #         for p in pruned:
    #             print("Removed:", p["full_expr"])
    #     self.accepted_conjectures = list(unique_conjs.values())


    # def search(self):
    #     import time
    #     start_time = time.time()
    #     for ant_str, ant_func in self.candidate_antecedents:
    #         try:
    #             ant_series = ant_func(self.df)
    #         except Exception as e:
    #             print(f"Error evaluating antecedent '{ant_str}':", e)
    #             continue
    #         for prop_str, prop_func in self.candidate_properties:
    #             try:
    #                 prop_series = prop_func(self.df)
    #             except Exception as e:
    #                 print(f"Error evaluating property '{prop_str}':", e)
    #                 continue
    #             if self._implication_holds(ant_series, prop_series):
    #                 self._record_conjecture(ant_str, prop_str, ant_func, prop_func)
    #             if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
    #                 print("Time limit reached, stopping search.")
    #                 return

    # def get_accepted_conjectures(self):
    #     return self.accepted_conjectures

    # def write_on_the_wall(self):
    #     print("Christy conjectures:")
    #     print("--------------------")
    #     for conj in self.accepted_conjectures:
    #         print(conj["full_expr"])






from fractions import Fraction
from itertools import combinations
import pandas as pd
import numpy as np
from collections import defaultdict

class Christy:
    def __init__(self, df, target, bound_type='lower', candidate_antecedents=None, candidate_properties=None, time_limit=None):
        self.df = df.copy()
        self.target = target
        self.bound_type = bound_type  # 'lower' for target >= antecedent, 'upper' for target <= antecedent
        self.time_limit = time_limit

        # If candidate antecedents are not provided, we generate them.
        if candidate_antecedents is None:
            # Use numeric columns (excluding booleans and the target)
            num_cols = [col for col in df.columns
                        if col != target
                        and pd.api.types.is_numeric_dtype(df[col])
                        and not pd.api.types.is_bool_dtype(df[col])]

            # Base candidates: the raw columns.
            base_candidates = [(col, lambda df, col=col: df[col]) for col in num_cols]

            # Define self.ratios exactly as in Graffiti.
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
                Fraction(1, 1), Fraction(2, 1), #Fraction(3, 1), #Fraction(4, 1),
            ]

            # Ratio candidates: multiply, add, and subtract each column by a constant.
            ratio_candidates = []
            for col in num_cols:
                for ratio in self.ratios:
                    ratio_candidates.append((
                        f"{ratio}*({col})",
                        lambda df, col=col, ratio=ratio: float(ratio) * df[col]
                    ))
                    # ratio_candidates.append((
                    #     f"({col}) + {ratio}",
                    #     lambda df, col=col, ratio=ratio: df[col] + float(ratio)
                    # ))
                    # ratio_candidates.append((
                    #     f"({col}) - {ratio}",
                    #     lambda df, col=col, ratio=ratio: df[col] - float(ratio)
                    # ))

            # Complexity3 hypothesis candidates: combine pairs of columns with various operations.
            complexity3_candidates = self._generate_candidates_complexity3_hypothesis(num_cols)

            # Final list of candidate antecedents.
            self.candidate_antecedents = base_candidates + ratio_candidates + complexity3_candidates
        else:
            if not hasattr(candidate_antecedents, '__iter__'):
                raise ValueError("candidate_antecedents must be an iterable of (expression, function) tuples.")
            self.candidate_antecedents = candidate_antecedents

        # Candidate boolean properties:
        if candidate_properties is None:
            candidate_props = []
            # (1) Boolean columns.
            bool_cols = [col for col in df.columns if pd.api.types.is_bool_dtype(df[col])]
            for col in bool_cols:
                candidate_props.append((col, lambda df, col=col: df[col]))
            # (2) Candidate equalities between numeric columns (only non-boolean).
            num_cols_all = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])
                            and not pd.api.types.is_bool_dtype(df[col])]
            for col1, col2 in combinations(num_cols_all, 2):
                expr = f"({col1} = {col2})"
                candidate_props.append((expr, lambda df, col1=col1, col2=col2: df[col1] == df[col2]))
            self.candidate_properties = candidate_props
        else:
            self.candidate_properties = candidate_properties

        # List to store accepted conjectures.
        self.accepted_conjectures = []

    # --- Transformation functions for applying self.ratios ---
    def _with_ratio_addition(self, candidate):
        base_expr, base_func = candidate
        for ratio in self.ratios:
            yield (f"({base_expr}) + {ratio}",
                   lambda df, base_func=base_func, ratio=ratio: base_func(df) + float(ratio))
    def _with_ratio_subtraction(self, candidate):
        base_expr, base_func = candidate
        for ratio in self.ratios:
            yield (f"({base_expr}) - {ratio}",
                   lambda df, base_func=base_func, ratio=ratio: base_func(df) - float(ratio))
    def _with_ratio_multiplication(self, candidate):
        base_expr, base_func = candidate
        for ratio in self.ratios:
            yield (f"{ratio}*({base_expr})",
                   lambda df, base_func=base_func, ratio=ratio: float(ratio) * base_func(df))
    def _expand_candidate(self, candidate):
        """
        Apply the ratio-based transformations and return unique candidate variants.
        """
        variants = {candidate[0]: candidate}
        for transform in [self._with_ratio_multiplication]:
            for cand in transform(candidate):
                if cand[0] not in variants:
                    variants[cand[0]] = cand
        return list(variants.values())

    # --- Complexity3 hypothesis candidate generation ---
    def _generate_candidates_complexity3_hypothesis(self, num_cols):
        """
        For each pair of numeric columns, generate candidates using:
          - Multiplication: x * y
          - Addition: x + y
          - Division: x/y (if safe) and y/x (if safe)
          - Minimum: min(x, y)
          - Maximum: max(x, y)
        Then, expand each candidate using ratio-based transformations.
        """
        candidates = []
        for col1, col2 in combinations(num_cols, 2):
            # Multiplication.
            candidates.append((
                f"({col1} * {col2})",
                lambda df, col1=col1, col2=col2: df[col1] * df[col2]
            ))
            # Addition.
            candidates.append((
                f"({col1} + {col2})",
                lambda df, col1=col1, col2=col2: df[col1] + df[col2]
            ))
            # Division: x/y (if safe).
            if (self.df[col2] == 0).sum() == 0:
                candidates.append((
                    f"({col1} / {col2})",
                    lambda df, col1=col1, col2=col2: df[col1] / df[col2]
                ))
            # Division: y/x (if safe).
            if (self.df[col1] == 0).sum() == 0:
                candidates.append((
                    f"({col2} / {col1})",
                    lambda df, col1=col1, col2=col2: df[col2] / df[col1]
                ))
            # Minimum.
            candidates.append((
                f"min({col1}, {col2})",
                lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).min(axis=1)
            ))
            # Maximum.
            candidates.append((
                f"max({col1}, {col2})",
                lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).max(axis=1)
            ))
        # Expand each candidate with ratio-based transformations.
        expanded = []
        for cand in candidates:
            expanded.extend(self._expand_candidate(cand))
        return expanded

    # --- Implication Checking and Conjecture Recording ---
    def _implication_holds(self, antecedent_series, prop_series):
        """
        For lower-bound hypotheses, checks that every row where target >= antecedent has prop_series True;
        for upper-bound, uses <=.
        """
        if self.bound_type == 'lower':
            condition = self.df[self.target] >= antecedent_series
        else:
            condition = self.df[self.target] <= antecedent_series
        if condition.sum() == 0:
            return False
        return prop_series[condition].all()

    def _record_conjecture(self, ant_str, prop_str, ant_func, prop_func):
        ant_series = ant_func(self.df)
        if self.bound_type == 'lower':
            support = int((self.df[self.target] >= ant_series).sum())
            bound_symbol = ">="
        else:
            support = int((self.df[self.target] <= ant_series).sum())
            bound_symbol = "<="
        full_expr = f"If {self.target} {bound_symbol} {ant_str}, then {prop_str}  [support: {support}]"
        new_conj = {
            "antecedent_str": ant_str,
            "prop_str": prop_str,
            "full_expr": full_expr,
            "ant_func": ant_func,
            "prop_func": prop_func,
            "support": support,
        }
        self.accepted_conjectures.append(new_conj)
        print("Accepted conjecture:", full_expr)
        self._prune_conjectures()

    def _prune_conjectures(self):
        """
        Prune duplicate conjectures: For conjectures with the same property, keep only the one with higher support.
        """
        pruned = []
        unique_conjs = {}
        for conj in self.accepted_conjectures:
            key = conj["prop_str"]
            if key in unique_conjs:
                if conj["support"] > unique_conjs[key]["support"]:
                    pruned.append(unique_conjs[key])
                    unique_conjs[key] = conj
                else:
                    pruned.append(conj)
            else:
                unique_conjs[key] = conj
        if pruned:
            print("Pruned conjectures:")
            for p in pruned:
                print("Removed:", p["full_expr"])
        self.accepted_conjectures = list(unique_conjs.values())

    # --- New: Consolidate Conjectures ---
    def consolidate_conjectures(self):
        """
        Group accepted conjectures by their antecedent and support.
        If multiple conjectures share the same antecedent and support,
        consolidate their properties into a single statement stating that the properties are equivalent.
        """
        groups = defaultdict(list)
        for conj in self.accepted_conjectures:
            key = (conj['antecedent_str'], conj['support'])
            groups[key].append(conj['prop_str'])

        bound_symbol = ">=" if self.bound_type == 'lower' else "<="
        consolidated = []
        for (ant, support), props in groups.items():
            props = sorted(set(props))
            if len(props) > 1:
                # Consolidate as an equivalence: p₁ ⇔ p₂ ⇔ ... ⇔ pₙ
                eq_props = " ⇔ ".join(props)
                consolidated.append(f"Conjecture. If {self.target} {bound_symbol} {ant}, then ({eq_props}) [support: {support}]")
            else:
                consolidated.append(f"Conjecture. If {self.target} {bound_symbol} {ant}, then {props[0]} [support: {support}]")
        return consolidated

    def search(self):
        import time
        start_time = time.time()
        for ant_str, ant_func in self.candidate_antecedents:
            try:
                ant_series = ant_func(self.df)
            except Exception as e:
                print(f"Error evaluating antecedent '{ant_str}':", e)
                continue
            for prop_str, prop_func in self.candidate_properties:
                try:
                    prop_series = prop_func(self.df)
                except Exception as e:
                    print(f"Error evaluating property '{prop_str}':", e)
                    continue
                if self._implication_holds(ant_series, prop_series):
                    self._record_conjecture(ant_str, prop_str, ant_func, prop_func)
                if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                    print("Time limit reached, stopping search.")
                    return

    def get_accepted_conjectures(self):
        return self.accepted_conjectures

    def write_on_the_wall(self):
        # print("Christy conjectures:")
        # print("--------------------")
        # for conj in self.accepted_conjectures:
        #     conj_temp = conj["full_expr"]
        #     print(f"Conjecture. {conj_temp}")
        #     print()
        print("Christy Conjectures:")
        print("--------------------")
        for c in self.consolidate_conjectures():
            print(c)
            print()







# import pandas as pd
# import numpy as np
# from fractions import Fraction
# from itertools import combinations
# import time
# from tqdm import tqdm

# class Christy:
#     def __init__(self, df, target, candidate_antecedents=None, candidate_properties=None, time_limit=None):
#         """
#         Parameters:
#           df: pandas DataFrame containing your data.
#           target: Name of the target column (used for numerical comparison).
#           candidate_antecedents: Optional iterable of (expression, function) tuples.
#               If None, antecedents are generated from numeric columns (excluding booleans).
#           candidate_properties: Optional iterable of (expression, function) tuples for boolean properties.
#               If None, candidate properties are generated from boolean columns and equalities between numeric columns.
#           time_limit: Optional search time limit (in seconds).
#         """
#         self.df = df.copy()
#         self.target = target
#         self.time_limit = time_limit

#         # Define candidate columns: numeric columns (excluding booleans and the target)
#         self.candidate_cols = [
#             col for col in self.df.columns
#             if col != target and pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_bool_dtype(self.df[col])
#         ]

#         # Define the set of ratios (exactly as in Graffiti).
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

#         # IMPORTANT: Define accepted_conjectures BEFORE generating candidates that might reference it.
#         self.accepted_conjectures = []  # This will store accepted rules.

#         # Generate candidate antecedents.
#         if candidate_antecedents is None:
#             self.candidate_antecedents = []
#             self.candidate_antecedents.extend(self._generate_candidates_complexity1())
#             self.candidate_antecedents.extend(self._generate_candidates_complexity2())
#             self.candidate_antecedents.extend(self._generate_candidates_complexity3())
#             self.candidate_antecedents.extend(self._generate_candidates_complexity4())
#             self.candidate_antecedents.extend(self._generate_candidates_complexity5())
#             self.candidate_antecedents.extend(self._generate_candidates_complexity6())
#             self.candidate_antecedents.extend(self._generate_candidates_complexity7())
#         else:
#             if not hasattr(candidate_antecedents, '__iter__'):
#                 raise ValueError("candidate_antecedents must be an iterable of (expression, function) tuples.")
#             self.candidate_antecedents = candidate_antecedents

#         # Generate candidate properties.
#         if candidate_properties is None:
#             candidate_props = []
#             # 1. Use boolean columns.
#             bool_cols = [col for col in self.df.columns if pd.api.types.is_bool_dtype(self.df[col])]
#             for col in bool_cols:
#                 candidate_props.append((col, lambda df, col=col: df[col]))
#             # 2. Also add candidate equalities between numeric columns.
#             for col1, col2 in combinations(self.candidate_cols, 2):
#                 expr = f"({col1} == {col2})"
#                 candidate_props.append((expr, lambda df, col1=col1, col2=col2: df[col1] == df[col2]))
#             self.candidate_properties = candidate_props
#         else:
#             self.candidate_properties = candidate_properties

#     # --- Transformation functions (same as in Graffiti) ---
#     def _with_floor_ceil(self, candidate):
#         base_rhs, base_func = candidate
#         yield (f"floor({base_rhs})", lambda df, base_func=base_func: np.floor(base_func(df)))
#         yield (f"ceil({base_rhs})", lambda df, base_func=base_func: np.ceil(base_func(df)))

#     def _with_ratio_addition(self, candidate):
#         base_rhs, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"({base_rhs}) + {ratio}",
#                    lambda df, base_func=base_func, ratio=ratio: base_func(df) + float(ratio))

#     def _with_ratio_subtraction(self, candidate):
#         base_rhs, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"({base_rhs}) - {ratio}",
#                    lambda df, base_func=base_func, ratio=ratio: base_func(df) - float(ratio))

#     def _with_ratio_multiplication(self, candidate):
#         base_rhs, base_func = candidate
#         for ratio in self.ratios:
#             yield (f"{ratio}*({base_rhs})",
#                    lambda df, base_func=base_func, ratio=ratio: float(ratio) * base_func(df))

#     def _expand_candidate(self, candidate):
#         """
#         Given a base candidate (expression, function), apply a set of transformations and
#         return the list of unique candidate variants (based on their expression string).
#         """
#         variants = {candidate[0]: candidate}
#         for transform in [self._with_floor_ceil, self._with_ratio_multiplication,
#                           self._with_ratio_subtraction, self._with_ratio_addition]:
#             for cand in transform(candidate):
#                 if cand[0] not in variants:
#                     variants[cand[0]] = cand
#         return list(variants.values())

#     # --- Candidate Generation for antecedents ---
#     # Complexity 1: Basic candidate: each numeric column.
#     def _generate_candidates_complexity1(self):
#         return [(col, lambda df, col=col: df[col]) for col in self.candidate_cols]

#     # Complexity 2: Unary operations.
#     def _generate_candidates_unary(self, col):
#         candidates = []
#         candidates.append((f"{col}", lambda df, col=col: df[col]))
#         candidates.append((f"({col})^2", lambda df, col=col: df[col]**2))
#         candidates.append((f"floor({col})", lambda df, col=col: np.floor(df[col])))
#         candidates.append((f"ceil({col})", lambda df, col=col: np.ceil(df[col])))
#         for ratio in self.ratios:
#             expr = f"{ratio}*({col})"
#             candidates.append((expr, lambda df, col=col, ratio=ratio: float(ratio) * df[col]))
#             candidates.append((f"({col}) + {ratio}", lambda df, col=col, ratio=ratio: df[col] + float(ratio)))
#             candidates.append((f"({col}) - {ratio}", lambda df, col=col, ratio=ratio: df[col] - float(ratio)))
#         return candidates

#     def _generate_candidates_complexity2(self):
#         candidates = []
#         for col in self.candidate_cols:
#             candidates.extend(self._generate_candidates_unary(col))
#         return candidates

#     # Complexity 3: Binary combinations (using two columns).
#     def _generate_candidates_binary(self, col1, col2):
#         base = [
#             (f"({col1} + {col2})", lambda df, col1=col1, col2=col2: df[col1] + df[col2]),
#             (f"({col1} - {col2})", lambda df, col1=col1, col2=col2: df[col1] - df[col2]),
#             (f"({col2} - {col1})", lambda df, col1=col1, col2=col2: df[col2] - df[col1]),
#             (f"{col1} * {col2}", lambda df, col1=col1, col2=col2: df[col1] * df[col2]),
#             (f"max({col1}, {col2})", lambda df, col1=col1, col2=col2: np.maximum(df[col1], df[col2])),
#             (f"min({col1}, {col2})", lambda df, col1=col1, col2=col2: np.minimum(df[col1], df[col2])),
#             (f"abs({col1} - {col2})", lambda df, col1=col1, col2=col2: np.abs(df[col1] - df[col2])),
#             (f"{col1}*{col2}", lambda df, col1=col1, col2=col2: df[col1] * df[col2]),
#         ]
#         # Add safe division candidates.
#         if (self.df[col2] == 0).sum() == 0:
#             base.append((f"({col1} / {col2})", lambda df, col1=col1, col2=col2: df[col1] / df[col2]))
#         if (self.df[col1] == 0).sum() == 0:
#             base.append((f"({col2} / {col1})", lambda df, col1=col1, col2=col2: df[col2] / df[col1]))
#         return base

#     def _generate_candidates_complexity3(self):
#         candidates = []
#         for col1, col2 in combinations(self.candidate_cols, 2):
#             for cand in self._generate_candidates_binary(col1, col2):
#                 candidates.extend(self._expand_candidate(cand))
#         return candidates

#     # Complexity 4: Binary combinations with additional transformations.
#     def _generate_candidates_binary_complex4(self, col1, col2):
#         base = [
#             (f"({col1} + {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] + df[col2])**2),
#             (f"({col1} - {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] - df[col2])**2)
#         ]
#         if (self.df[col1] + self.df[col2] < 0).sum() == 0:
#             base.append((f"sqrt({col1} + {col2})", lambda df, col1=col1, col2=col2: np.sqrt(df[col1] + df[col2])))
#         return base

#     def _generate_candidates_complexity4(self):
#         candidates = []
#         for col1, col2 in combinations(self.candidate_cols, 2):
#             for cand in self._generate_candidates_binary_complex4(col1, col2):
#                 candidates.extend(self._expand_candidate(cand))
#         return candidates

#     # Complexity 5: Powers of accepted conjectures.
#     def _generate_candidates_complexity5(self):
#         candidates = []
#         if not self.accepted_conjectures:
#             return candidates
#         for accepted in self.accepted_conjectures:
#             for exponent in [2, 3]:
#                 new_rhs = f"({accepted['rhs_str']})^{exponent}"
#                 func = lambda df, f_old=accepted['func'], exponent=exponent: f_old(df) ** exponent
#                 candidates.append((new_rhs, func))
#         return candidates

#     # Complexity 6: Using constants.
#     def _generate_candidates_complexity6(self):
#         candidates = []
#         for col in self.candidate_cols:
#             for c in self.ratios:
#                 expr = f"{c}*({col})"
#                 func = lambda df, col=col, c=c: float(c) * df[col]
#                 candidates.append((expr, func))
#         return candidates

#     # Complexity 7: Recursive combinations.
#     def _generate_candidates_recursive(self, max_depth):
#         if max_depth == 1:
#             return self._generate_candidates_complexity1()
#         else:
#             prev_candidates = self._generate_candidates_recursive(max_depth - 1)
#             atomic_candidates = self._generate_candidates_complexity1()
#             new_candidates = []
#             operators = [
#                 ("+", lambda a, b: a + b),
#                 ("-", lambda a, b: a - b),
#                 ("*", lambda a, b: a * b)
#             ]
#             for cand1 in prev_candidates:
#                 for cand2 in atomic_candidates:
#                     for op_sym, op_func in operators:
#                         new_expr = f"({cand1[0]}) {op_sym} ({cand2[0]})"
#                         new_func = lambda df, f1=cand1[1], f2=cand2[1], op=op_func: op(f1(df), f2(df))
#                         new_candidates.append((new_expr, new_func))
#             return prev_candidates + new_candidates

#     def _generate_candidates_complexity7(self):
#         rec_candidates = self._generate_candidates_recursive(max_depth=3)
#         candidates = []
#         for expr, func in rec_candidates:
#             op_count = expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/')
#             if op_count >= 2:
#                 candidates.extend(self._expand_candidate((expr, func)))
#         return candidates

#     def generate_candidates(self, complexity):
#         if complexity == 1:
#             return self._generate_candidates_complexity1()
#         elif complexity == 2:
#             return self._generate_candidates_complexity2()
#         elif complexity == 3:
#             return self._generate_candidates_complexity3()
#         elif complexity == 4:
#             return self._generate_candidates_complexity4()
#         elif complexity == 5:
#             return self._generate_candidates_complexity5()
#         elif complexity == 6:
#             return self._generate_candidates_complexity6()
#         elif complexity == 7:
#             return self._generate_candidates_complexity7()
#         else:
#             return []

#     # --- Conditional Conjecture Search ---
#     def _implication_holds(self, antecedent_series, prop_series):
#         """
#         Check whether for every row where target >= antecedent holds the property is True.
#         Returns False if no row satisfies the antecedent.
#         """
#         condition = self.df[self.target] >= antecedent_series
#         if condition.sum() == 0:
#             return False
#         return prop_series[condition].all()

#     def _record_conjecture(self, ant_str, prop_str, ant_func, prop_func):
#         ant_series = ant_func(self.df)
#         support = int((self.df[self.target] >= ant_series).sum())
#         full_expr = f"If {self.target} >= {ant_str}, then {prop_str}  [support: {support}]"
#         new_conj = {
#             "rhs_str": ant_str,
#             "prop_str": prop_str,
#             "full_expr": full_expr,
#             "func": ant_func,
#             "prop_func": prop_func,
#             "support": support
#         }
#         self.accepted_conjectures.append(new_conj)
#         print("Accepted conjecture:", full_expr)
#         self._prune_conjectures()

#     def _prune_conjectures(self):
#         pruned = []
#         unique = {}
#         for conj in self.accepted_conjectures:
#             key = conj["prop_str"]
#             if key in unique:
#                 if conj["support"] > unique[key]["support"]:
#                     pruned.append(unique[key])
#                     unique[key] = conj
#                 else:
#                     pruned.append(conj)
#             else:
#                 unique[key] = conj
#         if pruned:
#             print("Pruned conjectures:")
#             for p in pruned:
#                 print("Removed:", p["full_expr"])
#         self.accepted_conjectures = list(unique.values())

#     def search(self):
#         """
#         Loop over all candidate antecedents and candidate properties.
#         For each pair, if for every row where target >= antecedent holds the property is True,
#         record the conjecture.
#         """
#         start_time = time.time()
#         # Use a progress bar for antecedents.
#         for ant_str, ant_func in tqdm(self.candidate_antecedents, desc="Evaluating Antecedents"):
#             try:
#                 ant_series = ant_func(self.df)
#             except Exception as e:
#                 print(f"Error evaluating antecedent '{ant_str}':", e)
#                 continue
#             # Use a nested progress bar for candidate properties.
#             for prop_str, prop_func in tqdm(self.candidate_properties, desc="Evaluating Properties", leave=False):
#                 try:
#                     prop_series = prop_func(self.df)
#                 except Exception as e:
#                     print(f"Error evaluating property '{prop_str}':", e)
#                     continue
#                 if self._implication_holds(ant_series, prop_series):
#                     self._record_conjecture(ant_str, prop_str, ant_func, prop_func)
#                 if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                     print("Time limit reached, stopping search.")
#                     return

#     def get_accepted_conjectures(self):
#         return self.accepted_conjectures

#     def write_on_the_wall(self):
#         print("Christy conjectures:")
#         print("--------------------")
#         for conj in self.accepted_conjectures:
#             print(conj["full_expr"])