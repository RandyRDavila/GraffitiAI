
import pandas as pd
import numpy as np
from fractions import Fraction
from itertools import combinations
import time
from tqdm import tqdm

from graffitiai.base import BaseConjecturer
from graffitiai.utils import convert_conjecture_dicts

__all__ = ["Graffiti"]

import pandas as pd
import numpy as np
from fractions import Fraction
from itertools import combinations
import time
from tqdm import tqdm
from graffitiai.base import BaseConjecturer
from graffitiai.utils import convert_conjecture_dicts

__all__ = ["Graffiti"]

class Graffiti(BaseConjecturer):
    """
    Graffiti generates bound conjectures of the form:
         target >= candidate  (if bound_type=='lower')
         target <= candidate  (if bound_type=='upper')

    This consolidated class performs all tasks:
      - Sets the data (optionally filtered by a hypothesis column)
      - Computes candidate columns (numeric, non-boolean, excluding target)
      - Generates candidate expressions at various complexities
      - Evaluates candidates to check if they hold as inequalities
      - Records and prunes accepted conjectures, and then stores them in a dictionary
    """
    def __init__(self, knowledge_table=None, max_complexity=7):
        """
        Parameters:
          knowledge_table: pandas DataFrame containing the invariants and properties.
                           It can be loaded later if not provided.
          max_complexity: maximum candidate complexity to consider in search.
        """
        super().__init__(knowledge_table)
        self.conjectures = {}           # Stores final conjectures per target
        self.accepted_conjectures = []    # Working list during search
        self.max_complexity = max_complexity
        self.target = None              # Name of target column
        self.bound_type = None          # 'lower' or 'upper'
        self.hypothesis_str = None      # Optional filtering column name
        self.df = None                  # Working DataFrame (filtered if needed)
        self.candidate_cols = []        # Numeric, non-boolean candidate columns
        # A list of Fraction constants used in candidate transformations.
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
        self.ratio_pairs = list(combinations(self.ratios, 2))
        # Precompute whether each candidate column contains any zeros.
        self._cache_zero_flags()

    def _cache_zero_flags(self):
        self.zero_flags = {col: (self.df[col] == 0).any() for col in self.candidate_cols}

    # --------------------------------------------------------------------------
    # Setup and Data Filtering
    # --------------------------------------------------------------------------
    def _setup_data(self, target, filter_property=None):
        if self.knowledge_table is None:
            raise ValueError("No knowledge table available. Please load a DataFrame.")
        if filter_property is not None:
            self.df = self.knowledge_table[self.knowledge_table[filter_property] == True].copy()
            self.hypothesis_str = filter_property
        else:
            self.df = self.knowledge_table.copy()
            self.hypothesis_str = None
        self.target = target
        self.candidate_cols = [
            col for col in self.df.columns
            if col != target and
               pd.api.types.is_numeric_dtype(self.df[col]) and
               not pd.api.types.is_bool_dtype(self.df[col])
        ]

    # --------------------------------------------------------------------------
    # Candidate Transformation Helpers (unchanged)
    # --------------------------------------------------------------------------
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
        variants = {candidate[0]: candidate}
        for transform in [self._with_floor_ceil, self._with_ratio_multiplication,
                          self._with_ratio_subtraction, self._with_ratio_addition]:
            for cand in transform(candidate):
                if cand[0] not in variants:
                    variants[cand[0]] = cand
        return list(variants.values())

    # --------------------------------------------------------------------------
    # Candidate Generation Methods (unchanged for complexities 1-2 and others)
    # --------------------------------------------------------------------------
    def _generate_candidates_complexity1(self):
        base = [(col, lambda df, col=col: df[col]) for col in self.candidate_cols]
        scaled = [
            (f"{ratio}*({col})", lambda df, col=col, ratio=ratio: float(ratio) * df[col])
            for col in self.candidate_cols for ratio in self.ratios
        ]
        return base + scaled

    def _generate_candidates_unary(self, col):
        base_candidates = [
            (f"({col})^2", lambda df, col=col: df[col] ** 2),
            (f"floor({col})", lambda df, col=col: np.floor(df[col])),
            (f"ceil({col})", lambda df, col=col: np.ceil(df[col])),
        ]
        plus_candidates = [
            (f"{ratio2}*({col}) + {ratio}",
             lambda df, col=col, ratio=ratio, ratio2=ratio2: float(ratio2) * df[col] + float(ratio))
            for ratio, ratio2 in self.ratio_pairs
        ]
        minus_candidates = [
            (f"{ratio2}*({col}) - {ratio}",
             lambda df, col=col, ratio=ratio, ratio2=ratio2: float(ratio2) * df[col] - float(ratio))
            for ratio, ratio2 in self.ratio_pairs
        ]
        return base_candidates + plus_candidates + minus_candidates

    def _generate_candidates_complexity2(self):
        return [cand for col in self.candidate_cols for cand in self._generate_candidates_unary(col)]

    def _generate_candidates_binary(self, col1, col2):
        base = [
            (f"({col1} + {col2})", lambda df, col1=col1, col2=col2: df[col1] + df[col2]),
            (f"({col1} - {col2})", lambda df, col1=col1, col2=col2: df[col1] - df[col2]),
            (f"({col2} - {col1})", lambda df, col1=col1, col2=col2: df[col2] - df[col1]),
            (f"{col1} * {col2}", lambda df, col1=col1, col2=col2: df[col1] * df[col2]),
            (f"max({col1}, {col2})", lambda df, col1=col1, col2=col2: np.maximum(df[col1], df[col2])),
            (f"min({col1}, {col2})", lambda df, col1=col1, col2=col2: np.minimum(df[col1], df[col2])),
            (f"abs({col1} - {col2})", lambda df, col1=col1, col2=col2: np.abs(df[col1] - df[col2])),
        ]
        if not self.zero_flags.get(col2, (self.df[col2] == 0).any()):
            base.append((f"({col1} / {col2})", lambda df, col1=col1, col2=col2: df[col1] / df[col2]))
        if not self.zero_flags.get(col1, (self.df[col1] == 0).any()):
            base.append((f"({col2} / {col1})", lambda df, col1=col1, col2=col2: df[col2] / df[col1]))
        return base

    def _generate_candidates_complexity3(self):
        """
        Generate candidate expressions combining 3 candidate columns using binary operations.
        For each triple (col1, col2, col3) from candidate_cols, we form expressions of the form:
            ((col1 op1 col2) op2 col3)
        for operators op1 and op2 in {+, -, *}.
        Each candidate is then expanded via _expand_candidate to generate additional variants.
        """
        candidates = []
        operators = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b)
        ]
        for col1, col2, col3 in combinations(self.candidate_cols, 3):
            for op1_sym, op1 in operators:
                for op2_sym, op2 in operators:
                    # Build the expression string.
                    expr = f"(({col1} {op1_sym} {col2}) {op2_sym} {col3})"
                    # Use default arguments to capture the current values.
                    func = lambda df, col1=col1, col2=col2, col3=col3, op1=op1, op2=op2: op2(op1(df[col1], df[col2]), df[col3])
                    # Expand candidate using your transformation helpers.
                    for variant in self._expand_candidate((expr, func)):
                        candidates.append(variant)
        return candidates


    # You can leave complexities 4-7 as they are (or apply similar grouping if needed).

    # --------------------------------------------------------------------------
    # Early-Pruning Helpers for Candidate Groups (New)
    # --------------------------------------------------------------------------
    def _evaluate_candidate_variants(self, variants, complexity):
        """
        Given a list of candidate variants (each a tuple (expr, func)) from the same base group,
        compute an aggregate metric and sort them so that for upper bounds the candidate with the smallest
        metric (i.e. tightest) is evaluated first and for lower bounds the one with the highest metric is evaluated first.

        For upper bounds, we use candidate_series.max() (lower is better).
        For lower bounds, we use candidate_series.mean() (higher is better).
        Returns True if a candidate in the group is accepted.
        """
        metrics = []
        for expr, func in variants:
            try:
                candidate_series = func(self.df)
                if self.bound_type == 'upper':
                    metric = candidate_series.max()
                else:
                    # For lower bounds, use the mean value as a measure of overall strength.
                    metric = candidate_series.mean()
            except Exception:
                metric = np.inf if self.bound_type == 'upper' else -np.inf
            metrics.append((metric, expr, func))

        if self.bound_type == 'upper':
            metrics.sort(key=lambda x: x[0])
        else:
            metrics.sort(key=lambda x: x[0], reverse=True)

        for metric, expr, func in metrics:
            try:
                candidate_series = func(self.df)
            except Exception:
                continue
            # The candidate is accepted if it satisfies the inequality and is significant.
            if self._inequality_holds(candidate_series) and self._is_significant(candidate_series):
                self._record_conjecture(complexity, expr, func)
                return True  # Early exit: accepted one candidate from this group.
        return False


    def _process_binary_candidates(self, complexity):
        """
        Process binary candidates by grouping all variants from each candidate pair.
        For each pair (col1, col2), generate base binary candidates, expand them into a group,
        and then evaluate the group using early pruning.
        """
        for col1, col2 in combinations(self.candidate_cols, 2):
            base_candidates = self._generate_candidates_binary(col1, col2)
            for base in base_candidates:
                group = self._expand_candidate(base)
                if self._evaluate_candidate_variants(group, complexity):
                    # If one variant in the group is accepted, skip other variants for this pair.
                    break

    def generate_candidates(self, complexity):
        if complexity == 1:
            return self._generate_candidates_complexity1()
        elif complexity == 2:
            return self._generate_candidates_complexity2()
        elif complexity == 3:
            # Here _generate_candidates_complexity3 returns a list of groups (lists of variants)
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


    # --------------------------------------------------------------------------
    # Candidate Evaluation and Conjecture Recording (Unchanged)
    # --------------------------------------------------------------------------
    def _process_unary_candidates(self, complexity):
        """
        For each candidate column, generate all its unary candidate variants and
        evaluate them using early pruning. A progress bar is displayed over the candidate columns.
        """
        from tqdm import tqdm  # ensure tqdm is imported
        for col in tqdm(self.candidate_cols, desc=f"Complexity {complexity} (unary groups)"):
            variants = self._generate_candidates_unary(col)
            self._evaluate_candidate_variants(variants, complexity)



    def _compute_current_bound(self):
        if not self.accepted_conjectures:
            init_val = -np.inf if self.bound_type == 'lower' else np.inf
            return pd.Series(init_val, index=self.df.index)
        bounds = []
        for conj in self.accepted_conjectures:
            try:
                b = conj['func'](self.df)
                bounds.append(b)
            except Exception as e:
                print("Error computing accepted bound:", conj.get('full_expr_str', '?'), e)
        df_bounds = pd.concat(bounds, axis=1)
        return df_bounds.max(axis=1) if self.bound_type == 'lower' else df_bounds.min(axis=1)

    def _inequality_holds(self, candidate_series):
        target_series = self.df[self.target]
        if self.bound_type == 'lower':
            return (target_series >= candidate_series).all()
        else:
            return (target_series <= candidate_series).all()

    def _is_significant(self, candidate_series):
        """
        A candidate is significant if there is at least one row for which
        the new candidate improves upon every accepted candidate.

        For upper bounds: candidate_series must be strictly lower than all
        accepted candidates on at least one row.

        For lower bounds: candidate_series must be strictly higher than all
        accepted candidates on at least one row.
        """
        # If there are no accepted conjectures yet, then the candidate is automatically significant.
        if not self.accepted_conjectures:
            return True

        # Create a list of boolean Series, one for each accepted candidate.
        comparisons = []
        for conj in self.accepted_conjectures:
            accepted_values = conj['func'](self.df)
            if self.bound_type == 'upper':
                # Candidate is better if its value is lower than the accepted one.
                comparisons.append(candidate_series < accepted_values)
            else:
                # For lower bounds, candidate is better if its value is higher.
                comparisons.append(candidate_series > accepted_values)

        # Combine the comparisons along the row axis: a row is "improved" only if the candidate
        # is strictly better than every accepted candidate for that row.
        # (We use np.logical_and.reduce to get the elementwise "and" across all accepted candidates.)
        all_better = np.logical_and.reduce(np.vstack([comp.values for comp in comparisons]))

        # The candidate is significant if there is at least one row where this holds.
        return all_better.any()


    def _record_conjecture(self, complexity, rhs_str, func):
        if self.hypothesis_str:
            if self.bound_type == 'lower':
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} >= {rhs_str}."
            else:
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} <= {rhs_str}."
        else:
            full_expr_str = f"{self.target} >= {rhs_str}" if self.bound_type == 'lower' else f"{self.target} <= {rhs_str}"
        candidate_series = func(self.df)
        touches = int((self.df[self.target] == candidate_series).sum())
        new_conj = {
            'complexity': complexity,
            'rhs_str': rhs_str,
            'full_expr_str': full_expr_str,
            'func': func,
            'touch': touches
        }
        self.accepted_conjectures.append(new_conj)
        print(f"Accepted conjecture (complexity {complexity}, touch {touches}): {full_expr_str}")
        self._prune_conjectures()

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

    # --------------------------------------------------------------------------
    # Main Search Loop and Public Entry Point
    # --------------------------------------------------------------------------
    def search(self, time_limit_seconds):
        start_time = time.time()
        new_conjecture_found = True
        while new_conjecture_found:
            if time_limit_seconds is not None and (time.time() - start_time) >= time_limit_seconds:
                print("Time limit reached. Halting search.")
                break
            new_conjecture_found = False
            for complexity in range(1, self.max_complexity + 1):
                if complexity == 2:
                    # Process unary candidates by groups (per column)
                    self._process_unary_candidates(complexity)
                    # If at least one conjecture was accepted, mark new_conjecture_found.
                    if self.accepted_conjectures:
                        new_conjecture_found = True
                        break

                elif complexity == 3:
                    self._process_binary_candidates(complexity)
                    if self.accepted_conjectures:
                        new_conjecture_found = True
                        break
                else:
                    candidates = self.generate_candidates(complexity)
                    if not candidates:
                        continue
                    with tqdm(total=len(candidates), desc=f"Complexity {complexity}", leave=True) as pbar:
                        for rhs_str, func in candidates:
                            if time_limit_seconds is not None and (time.time() - start_time) >= time_limit_seconds:
                                print("Time limit reached during candidate evaluation. Halting search.")
                                new_conjecture_found = False
                                break
                            try:
                                candidate_series = func(self.df)
                            except Exception as e:
                                pbar.write(f"Skipping candidate {rhs_str} due to error: {e}")
                                pbar.update(1)
                                continue
                            pbar.set_postfix(candidate=rhs_str)
                            pbar.update(1)
                            if not self._inequality_holds(candidate_series):
                                continue
                            if not self._is_significant(candidate_series):
                                continue
                            self._record_conjecture(complexity, rhs_str, func)
                            new_conjecture_found = True
                            break  # Stop the current complexity round after one valid candidate
                    if new_conjecture_found:
                        break
            if not new_conjecture_found:
                print("No further significant conjectures found within the maximum complexity.")
                break



    def conjecture(self, target, hypothesis=None, bound_type='lower', filter_property=None, time_limit_minutes=1):
        self._setup_data(target, filter_property=hypothesis if hypothesis is not None else filter_property)
        self.bound_type = bound_type
        self.accepted_conjectures = []  # Clear previous conjectures.
        self.search(time_limit_seconds=time_limit_minutes * 60)
        processed = convert_conjecture_dicts(self.accepted_conjectures, target, hypothesis=hypothesis, default_bound_type=bound_type)
        self.conjectures[target] = {bound_type: processed}

    def write_on_the_wall(self, target=None):
        for bound_type in ['lower', 'upper']:
            if target is not None:
                if target not in self.conjectures:
                    print(f"No conjectures available for target: {target}")
                    return
                conj_list = self.conjectures[target].get(bound_type, [])
                if not conj_list:
                    print(f"No {bound_type} conjectures for target: {target}")
                    return
                sorted_conjectures = sorted(conj_list, key=lambda c: c.touch, reverse=True)
                print(f"GRAFFITI conjectures for {target} ({bound_type}):")
                print("------------------------")
                for conj in sorted_conjectures:
                    print(f"Conjecture: {conj}")
            else:
                for tgt, bound_dict in self.conjectures.items():
                    conj_list = bound_dict.get(bound_type, [])
                    if not conj_list:
                        continue
                    sorted_conjectures = sorted(conj_list, key=lambda c: c.touch, reverse=True)
                    print(f"GRAFFITI conjectures for {tgt} ({bound_type}):")
                    print("------------------------")
                    for conj in sorted_conjectures:
                        print(f"Conjecture: {conj}")






# import pandas as pd
# import numpy as np
# from fractions import Fraction
# from itertools import combinations
# import time
# from tqdm import tqdm


# from graffitiai.base import BaseConjecturer, BoundConjecture
# from graffitiai.utils import convert_conjecture_dicts

# __all__ = ["Graffiti"]




# class Graffiti(BaseConjecturer):
#     def __init__(self, knowledge_table=None):
#         # If no knowledge_table is provided, it can be loaded later using read_csv.
#         super().__init__(knowledge_table)

#     def conjecture(self, target, hypothesis=None, bound_type='lower', filter_property=None, time_limit_minutes=1):
#         if hypothesis is not None:
#             df = self.knowledge_table.copy()
#             df = df[df[hypothesis] == True].copy()
#             fajtlowicz = LegacyGraffiti(df, target, bound_type, filter_property, time_limit=time_limit_minutes * 60)
#         else:
#             fajtlowicz = LegacyGraffiti(self.knowledge_table, target, bound_type, filter_property, time_limit=time_limit_minutes * 60)

#         fajtlowicz.search()
#         conjectures = convert_conjecture_dicts(fajtlowicz.accepted_conjectures, target, hypothesis=hypothesis)
#         self.conjectures[target] = {bound_type: conjectures}

#     def write_on_the_wall(self, target=None, bound_type='lower'):
#         # If a specific target is provided, print only its conjectures.
#         if target is not None:
#             if target not in self.conjectures:
#                 print(f"No conjectures available for target: {target}")
#                 return
#             conj_list = self.conjectures[target].get(bound_type, [])
#             if not conj_list:
#                 print(f"No {bound_type} conjectures for target: {target}")
#                 return
#             sorted_conjectures = sorted(conj_list, key=lambda c: c.touch, reverse=True)
#             print(f"GRAFFITI conjectures for {target} ({bound_type}):")
#             print("------------------------")
#             for conj in sorted_conjectures:
#                 print(f"Conjecture: {conj}")
#         else:
#             # If no target is provided, iterate over all targets.
#             for tgt, bound_dict in self.conjectures.items():
#                 conj_list = bound_dict.get(bound_type, [])
#                 if not conj_list:
#                     continue
#                 sorted_conjectures = sorted(conj_list, key=lambda c: c.touch, reverse=True)
#                 print(f"GRAFFITI conjectures for {tgt} ({bound_type}):")
#                 print("------------------------")
#                 for conj in sorted_conjectures:
#                     print(f"Conjecture: {conj}")



# class LegacyGraffiti:

#     def __init__(self, df, target_invariant, bound_type='lower', filter_property=None, time_limit=None):
#         """
#         Parameters:
#           df: pandas DataFrame containing the invariants and boolean properties.
#           target_invariant: name of the column whose bound we wish to conjecture.
#           bound_type: 'lower' (interpreted as target >= candidate) or 'upper' (target <= candidate).
#           filter_property: optional boolean column name; if provided, only rows with True are used.
#           time_limit: maximum search time in seconds (or None for no limit).
#         """
#         self.df_full = df.copy()
#         if filter_property is not None:
#             self.df = df[df[filter_property] == True].copy()
#             self.hypothesis_str = filter_property
#         else:
#             self.df = df.copy()
#             self.hypothesis_str = None

#         self.target = target_invariant
#         self.bound_type = bound_type
#         self.time_limit = time_limit  # in seconds

#         # Candidate columns: numeric (but not boolean) and not the target.
#         self.candidate_cols = [
#             col for col in self.df.columns
#             if col != target_invariant and
#                pd.api.types.is_numeric_dtype(self.df[col]) and
#                not pd.api.types.is_bool_dtype(self.df[col])
#         ]

#         self.accepted_conjectures = []
#         self.max_complexity = 7

#         # A list of Fraction constants used in ratio operations.
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

#     # ---------------------------------------------------------------------------
#     # Helper: Compute the current bound from accepted conjectures.
#     # ---------------------------------------------------------------------------
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

#     # ---------------------------------------------------------------------------
#     # Check if the candidate inequality holds.
#     # ---------------------------------------------------------------------------
#     def _inequality_holds(self, candidate_series):
#         target_series = self.df[self.target]
#         if self.bound_type == 'lower':
#             return (target_series >= candidate_series).all()
#         else:
#             return (target_series <= candidate_series).all()

#     # ---------------------------------------------------------------------------
#     # Check if the candidate is significant (i.e. improves over current bound).
#     # ---------------------------------------------------------------------------
#     def _is_significant(self, candidate_series):
#         current_bound = self._compute_current_bound()
#         if self.bound_type == 'lower':
#             diff = candidate_series - current_bound
#         else:
#             diff = current_bound - candidate_series
#         return (diff > 0).any()

#     # ---------------------------------------------------------------------------
#     # Record a new conjecture.
#     # ---------------------------------------------------------------------------
#     def _record_conjecture(self, complexity, rhs_str, func):
#         if self.hypothesis_str:
#             if self.bound_type == 'lower':
#                 full_expr_str = f"For any {self.hypothesis_str}, {self.target} >= {rhs_str}."
#             else:
#                 full_expr_str = f"For any {self.hypothesis_str}, {self.target} <= {rhs_str}."
#         else:
#             full_expr_str = f"{self.target} >= {rhs_str}" if self.bound_type == 'lower' else f"{self.target} <= {rhs_str}"
#         new_conj = {
#             'complexity': complexity,
#             'rhs_str': rhs_str,
#             'full_expr_str': full_expr_str,
#             'func': func
#         }
#         # Compute the "touch" (number of rows where equality holds).
#         candidate_series = func(self.df)
#         touches = int((self.df[self.target] == candidate_series).sum())
#         new_conj['touch'] = touches

#         self.accepted_conjectures.append(new_conj)
#         print(f"Accepted conjecture (complexity {complexity}, touch {touches}): {full_expr_str}")
#         self._prune_conjectures()

#     # ---------------------------------------------------------------------------
#     # Prune conjectures that are dominated by others.
#     # ---------------------------------------------------------------------------
#     def _prune_conjectures(self):
#         new_conjectures = []
#         removed_conjectures = []
#         n = len(self.accepted_conjectures)
#         for i in range(n):
#             conj_i = self.accepted_conjectures[i]
#             series_i = conj_i['func'](self.df)
#             dominated = False
#             for j in range(n):
#                 if i == j:
#                     continue
#                 series_j = self.accepted_conjectures[j]['func'](self.df)
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

#     # ---------------------------------------------------------------------------
#     # -- Transformation helper functions --
#     # Each takes a candidate (rhs_str, func) and yields variants.
#     # ---------------------------------------------------------------------------
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
#         Given a base candidate (rhs_str, func), apply a set of transformations and
#         return the list of unique candidate variants (based on their string representation).
#         """
#         variants = {candidate[0]: candidate}
#         for transform in [self._with_floor_ceil, self._with_ratio_multiplication,
#                           self._with_ratio_subtraction, self._with_ratio_addition]:
#             for cand in transform(candidate):
#                 # Avoid duplicates (based on the expression string).
#                 if cand[0] not in variants:
#                     variants[cand[0]] = cand
#         return list(variants.values())

#     # ---------------------------------------------------------------------------
#     # -- Candidate Generation: Complexity 1 and 2 (Unary candidates) --
#     # ---------------------------------------------------------------------------
#     def _generate_candidates_complexity1(self):
#         # Basic candidate: each invariant column by itself.
#         return [(col, lambda df, col=col: df[col]) for col in self.candidate_cols]

#     def _generate_candidates_unary(self, col):
#         """
#         Generate several unary transformations for a given column.
#         Used in complexity 2.
#         """
#         candidates = []
#         candidates.append((f"{col}", lambda df, col=col: df[col]))
#         candidates.append((f"({col})^2", lambda df, col=col: df[col]**2))
#         candidates.append((f"floor({col})", lambda df, col=col: np.floor(df[col])))
#         candidates.append((f"ceil({col})", lambda df, col=col: np.ceil(df[col])))
#         for ratio in self.ratios:
#             # Multiplication with ratio (with floor and ceil already handled later)
#             expr = f"{ratio}*({col})"
#             candidates.append((expr, lambda df, col=col, ratio=ratio: float(ratio) * df[col]))
#             # Addition and subtraction variants.
#             candidates.append((f"({col}) + {ratio}", lambda df, col=col, ratio=ratio: df[col] + float(ratio)))
#             candidates.append((f"({col}) - {ratio}", lambda df, col=col, ratio=ratio: df[col] - float(ratio)))
#         return candidates

#     def _generate_candidates_complexity2(self):
#         candidates = []
#         for col in self.candidate_cols:
#             candidates.extend(self._generate_candidates_unary(col))
#         return candidates

#     # ---------------------------------------------------------------------------
#     # -- Candidate Generation: Complexity 3 and 4 (Binary candidates) --
#     # ---------------------------------------------------------------------------
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

#     def _generate_candidates_binary_complex4(self, col1, col2):
#         base = [
#             (f"({col1} + {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] + df[col2])**2),
#             (f"({col1} - {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] - df[col2])**2)
#         ]
#         # Only allow sqrt if the sum is always nonnegative.
#         if (self.df[col1] + self.df[col2] < 0).sum() == 0:
#             base.append((f"sqrt({col1} + {col2})", lambda df, col1=col1, col2=col2: np.sqrt(df[col1] + df[col2])))
#         return base

#     def _generate_candidates_complexity4(self):
#         candidates = []
#         for col1, col2 in combinations(self.candidate_cols, 2):
#             for cand in self._generate_candidates_binary_complex4(col1, col2):
#                 candidates.extend(self._expand_candidate(cand))
#         return candidates

#     # ---------------------------------------------------------------------------
#     # -- Candidate Generation: Complexity 5 (Powers of accepted conjectures) --
#     # ---------------------------------------------------------------------------
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

#     # ---------------------------------------------------------------------------
#     # -- Candidate Generation: Complexity 6 (Using constants) --
#     # ---------------------------------------------------------------------------
#     def _generate_candidates_complexity6(self):
#         candidates = []
#         for col in self.candidate_cols:
#             for c in self.ratios:
#                 expr = f"{c}*({col})"
#                 func = lambda df, col=col, c=c: float(c) * df[col]
#                 candidates.append((expr, func))
#         return candidates

#     # ---------------------------------------------------------------------------
#     # -- Candidate Generation: Complexity 7 (Recursive combinations) --
#     # ---------------------------------------------------------------------------
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
#                 # Note: Division can be added with care.
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
#             # Use a rough measure of “complexity” (number of operators)
#             op_count = expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/')
#             if op_count >= 2:
#                 candidates.extend(self._expand_candidate((expr, func)))
#         return candidates

#     # ---------------------------------------------------------------------------
#     # Public candidate-generation entry point.
#     # ---------------------------------------------------------------------------
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
#         # elif complexity == 7:
#         #     return self._generate_candidates_complexity7()
#         else:
#             return []

#     # ---------------------------------------------------------------------------
#     # Main search loop.
#     # ---------------------------------------------------------------------------
#     def search(self):
#         start_time = time.time()
#         new_conjecture_found = True
#         while new_conjecture_found:
#             # Check time limit before starting a new round
#             if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                 print("Time limit reached. Halting search.")
#                 break
#             new_conjecture_found = False
#             # Loop through complexities from 1 up to max_complexity
#             for complexity in range(1, self.max_complexity + 1):
#                 candidates = self.generate_candidates(complexity)
#                 if not candidates:
#                     continue  # No candidates for this complexity level
#                 # Create a progress bar for the current set of candidates
#                 with tqdm(total=len(candidates), desc=f"Complexity {complexity}", leave=True) as pbar:
#                     for rhs_str, func in candidates:
#                         # Check time limit before evaluating each candidate
#                         if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
#                             print("Time limit reached during candidate evaluation. Halting search.")
#                             new_conjecture_found = False
#                             break
#                         try:
#                             candidate_series = func(self.df)
#                         except Exception as e:
#                             pbar.write(f"Skipping candidate {rhs_str} due to error: {e}")
#                             pbar.update(1)
#                             continue

#                         # Update the progress bar to show the current candidate
#                         pbar.set_postfix(candidate=rhs_str)
#                         pbar.update(1)

#                         if not self._inequality_holds(candidate_series):
#                             continue
#                         if not self._is_significant(candidate_series):
#                             continue

#                         # If we found a valid candidate, record the conjecture
#                         self._record_conjecture(complexity, rhs_str, func)
#                         new_conjecture_found = True
#                         break  # Exit loop after a valid candidate is found
#                 if new_conjecture_found:
#                     break  # Exit complexity loop if a valid candidate is found
#             if not new_conjecture_found:
#                 print("No further significant conjectures found within the maximum complexity.")
#                 break


#     # ---------------------------------------------------------------------------
#     # Get accepted conjectures sorted by "touch" (descending).
#     # ---------------------------------------------------------------------------
#     def get_accepted_conjectures(self):
#         return self.accepted_conjectures
#         # return sorted(self.accepted_conjectures, key=lambda c: c['touch'], reverse=True)

#     # ---------------------------------------------------------------------------
#     # Write conjectures to the “wall.”
#     # ---------------------------------------------------------------------------
#     def write_on_the_wall(self):
#         print("GRAFFITI conjectures:")
#         print("------------------------")
#         for conj in self.get_accepted_conjectures():
#             print(f"Conjecture: {conj['full_expr_str']} (touch {conj['touch']})")




# # class Graffiti(BaseConjecturer):
# #     """
# #     Graffiti generates bound conjectures (conclusions) of the form:
# #          target invariant ≥ candidate   (if bound_type=='lower')
# #       or target invariant ≤ candidate   (if bound_type=='upper').

# #     Accepted conjectures are stored as BoundConjecture objects.
# #     All heavy processing is deferred to the conjecture() method.
# #     """

# #     def __init__(self):
# #         """
# #         Minimal constructor. All heavy work is done in conjecture().
# #         """
# #         super().__init__()
# #         self.accepted_conjectures = []
# #         self.conjectures = {}
# #         self.candidate_antecedents = None
# #         self.candidate_properties = None
# #         self.ratios = None
# #         self.candidate_cols = None  # Will be computed when data is loaded
# #         self.max_complexity = 2
# #     def _generate_candidate_components(self, target, candidate_antecedents=None):
# #         """
# #         Generate candidate antecedents from the cleaned DataFrame (self.knowledge_table).
# #         Also computes self.candidate_cols as all numeric, non-boolean columns excluding the target.
# #         """
# #         # Compute candidate columns.
# #         self.candidate_cols = [
# #             col for col in self.knowledge_table.columns
# #             if col != target and
# #                pd.api.types.is_numeric_dtype(self.knowledge_table[col]) and
# #                not pd.api.types.is_bool_dtype(self.knowledge_table[col])
# #         ]

# #         if candidate_antecedents is None:
# #             # Base candidates: raw columns.
# #             base_candidates = [(col, lambda df, col=col: df[col]) for col in self.candidate_cols]

# #             # Define ratios.
# #             self.ratios = [
# #                 Fraction(1, 10), Fraction(3, 10), Fraction(7, 10), Fraction(9, 10),
# #                 Fraction(1, 9),  Fraction(2, 9),  Fraction(4, 9),  Fraction(5, 9),
# #                 Fraction(7, 9),  Fraction(8, 9),  Fraction(10, 9),
# #                 Fraction(1, 8),  Fraction(3, 8),  Fraction(5, 8),  Fraction(7, 8),  Fraction(9, 8),
# #                 Fraction(1, 7),  Fraction(2, 7),  Fraction(3, 7),  Fraction(4, 7),  Fraction(5, 7),
# #                 Fraction(6, 7),  Fraction(8, 7),  Fraction(9, 7),
# #                 Fraction(1, 6),  Fraction(5, 6),  Fraction(7, 6),
# #                 Fraction(1, 5),  Fraction(2, 5),  Fraction(3, 5),  Fraction(4, 5),  Fraction(6, 5),
# #                 Fraction(7, 5),  Fraction(8, 5),  Fraction(9, 5),
# #                 Fraction(1, 4),
# #                 Fraction(1, 3),  Fraction(2, 3),  Fraction(4, 3),  Fraction(5, 3),
# #                 Fraction(7, 3),  Fraction(8, 3),  Fraction(10, 3),
# #                 Fraction(1, 2),  Fraction(3, 2),  Fraction(5, 2),  Fraction(7, 2),  Fraction(9, 2),
# #                 Fraction(1, 1),  Fraction(2, 1)
# #             ]

# #             # Ratio candidates: multiply each candidate column by a ratio.
# #             ratio_candidates = []
# #             for col in self.candidate_cols:
# #                 for ratio in self.ratios:
# #                     # Use default parameters to capture current col and ratio.
# #                     ratio_candidates.append((
# #                         f"{ratio}*({col})",
# #                         lambda df, col=col, ratio=ratio: float(ratio) * df[col]
# #                     ))

# #             # Complexity-3 candidates: combine pairs of candidate columns.
# #             complexity3_candidates = self._generate_candidates_complexity3_hypothesis(self.candidate_cols)

# #             self.candidate_antecedents = base_candidates + ratio_candidates + complexity3_candidates
# #         else:
# #             self.candidate_antecedents = candidate_antecedents

# #     def _generate_candidates_complexity3_hypothesis(self, cols):
# #         """
# #         For each pair in cols, generate candidate functions using basic operations.
# #         """
# #         candidates = []
# #         for col1, col2 in combinations(cols, 2):
# #             candidates.append((
# #                 f"({col1} * {col2})",
# #                 lambda df, col1=col1, col2=col2: df[col1] * df[col2]
# #             ))
# #             candidates.append((
# #                 f"({col1} + {col2})",
# #                 lambda df, col1=col1, col2=col2: df[col1] + df[col2]
# #             ))
# #             if (self.knowledge_table[col2] == 0).sum() == 0:
# #                 candidates.append((
# #                     f"({col1} / {col2})",
# #                     lambda df, col1=col1, col2=col2: df[col1] / df[col2]
# #                 ))
# #             if (self.knowledge_table[col1] == 0).sum() == 0:
# #                 candidates.append((
# #                     f"({col2} / {col1})",
# #                     lambda df, col1=col1, col2=col2: df[col2] / df[col1]
# #                 ))
# #             candidates.append((
# #                 f"min({col1}, {col2})",
# #                 lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).min(axis=1)
# #             ))
# #             candidates.append((
# #                 f"max({col1}, {col2})",
# #                 lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).max(axis=1)
# #             ))
# #         return candidates

# #     def _is_significant(self, candidate_series):
# #         """
# #         Candidate is significant if, compared to the current bound, it improves for at least one row.
# #         """
# #         current_bound = self._compute_current_bound()
# #         if self.bound_type == 'lower':
# #             diff = candidate_series - current_bound
# #         else:
# #             diff = current_bound - candidate_series
# #         return (diff > 0).any()

# #     def _record_conjecture(self, complexity, candidate_expr, candidate_func):
# #         """
# #         Create a new BoundConjecture from the candidate.
# #         Before appending, check if a conjecture with the same full_expr already exists.
# #         Then, compute its touch and add it if it's new.
# #         """
# #         bc = BoundConjecture(
# #             target=self.target,
# #             candidate_expr=candidate_expr,
# #             candidate_func=candidate_func,
# #             bound_type=self.bound_type,
# #             hypothesis=self.hypothesis_str,
# #             complexity=complexity
# #         )
# #         bc.compute_touch(self.knowledge_table)
# #         # Check for duplicates based on the formatted expression.
# #         for existing in self.accepted_conjectures:
# #             if existing.full_expr == bc.full_expr:
# #                 # Duplicate found; do not add.
# #                 return
# #         print(f"Accepted conjecture (complexity {complexity}, touch {bc.touch}): {bc.full_expr}")
# #         self.accepted_conjectures.append(bc)
# #         self._prune_conjectures()

# #     def _prune_conjectures(self):
# #         """
# #         Remove conjectures that are dominated by others.
# #         Dominance is determined by comparing the candidate function evaluations.
# #         """
# #         unique = {}
# #         for conj in self.accepted_conjectures:
# #             # Use the full expression as key.
# #             key = conj.full_expr
# #             if key not in unique or conj.touch > unique[key].touch:
# #                 unique[key] = conj
# #         pruned = list(unique.values())
# #         if len(pruned) < len(self.accepted_conjectures):
# #             print("Pruning conjectures:")
# #             removed = set(self.accepted_conjectures) - set(pruned)
# #             for rem in removed:
# #                 print("Removed:", rem.full_expr)
# #         self.accepted_conjectures = pruned

# #     def generate_candidates(self, complexity):
# #         """
# #         Public entry point for generating candidate antecedents.
# #         For complexity 1, return each candidate column.
# #         For higher complexities, generate unary, binary, etc.
# #         """
# #         if complexity == 1:
# #             return [(col, lambda df, col=col: df[col]) for col in self.candidate_cols]
# #         elif complexity == 2:
# #             candidates = []
# #             for col in self.candidate_cols:
# #                 candidates.extend(self._generate_candidates_unary(col))
# #             return candidates
# #         elif complexity == 3:
# #             return self._generate_candidates_complexity3()
# #         else:
# #             return []

# #     def _generate_candidates_unary(self, col):
# #         candidates = []
# #         candidates.append((f"{col}", lambda df, col=col: df[col]))
# #         candidates.append((f"({col})^2", lambda df, col=col: df[col]**2))
# #         candidates.append((f"floor({col})", lambda df, col=col: np.floor(df[col])))
# #         candidates.append((f"ceil({col})", lambda df, col=col: np.ceil(df[col])))
# #         for ratio in self.ratios:
# #             expr = f"{ratio}*({col})"
# #             candidates.append((expr, lambda df, col=col, ratio=ratio: float(ratio) * df[col]))
# #         return candidates

# #     def _generate_candidates_complexity3(self):
# #         candidates = []
# #         for col1, col2 in combinations(self.candidate_cols, 2):
# #             for cand in self._generate_candidates_binary(col1, col2):
# #                 # You may expand further if desired.
# #                 candidates.append(cand)
# #         return candidates

# #     def _generate_candidates_binary(self, col1, col2):
# #         base = [
# #             (f"({col1} + {col2})", lambda df, col1=col1, col2=col2: df[col1] + df[col2]),
# #             (f"({col1} - {col2})", lambda df, col1=col1, col2=col2: df[col1] - df[col2]),
# #             (f"({col2} - {col1})", lambda df, col1=col1, col2=col2: df[col2] - df[col1]),
# #             (f"{col1} * {col2}", lambda df, col1=col1, col2=col2: df[col1] * df[col2])
# #         ]
# #         if (self.knowledge_table[col2] == 0).sum() == 0:
# #             base.append((f"({col1} / {col2})", lambda df, col1=col1, col2=col2: df[col1] / df[col2]))
# #         if (self.knowledge_table[col1] == 0).sum() == 0:
# #             base.append((f"({col2} / {col1})", lambda df, col1=col1, col2=col2: df[col2] / df[col1]))
# #         return base

# #     def _compute_current_bound(self):
# #         """
# #         Compute the current bound from accepted conjectures.
# #         If none are accepted, return -∞ for lower bound or +∞ for upper bound.
# #         """
# #         if not self.accepted_conjectures:
# #             return pd.Series(-np.inf if self.bound_type == 'lower' else np.inf,
# #                              index=self.knowledge_table.index)
# #         bounds = []
# #         for conj in self.accepted_conjectures:
# #             try:
# #                 b_val = conj.candidate_func(self.knowledge_table)
# #                 bounds.append(b_val)
# #             except Exception as e:
# #                 print("Error computing bound for", conj.full_expr, e)
# #         df_bounds = pd.concat(bounds, axis=1)
# #         return df_bounds.max(axis=1) if self.bound_type == 'lower' else df_bounds.min(axis=1)

# #     def search(self):
# #         """
# #         Main search loop.
# #         Iterates over candidate antecedents at increasing complexity.
# #         For each candidate, if the candidate inequality holds for all rows and is significant
# #         (i.e. at least one row attains equality), record it as a BoundConjecture.
# #         Stops if the time limit is reached.
# #         """
# #         start_time = time.time()
# #         new_found = True
# #         while new_found:
# #             if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
# #                 print("Time limit reached. Halting search.")
# #                 break
# #             new_found = False
# #             for complexity in range(1, self.max_complexity + 1):
# #                 candidates = self.generate_candidates(complexity)
# #                 for candidate_expr, candidate_func in tqdm(candidates, desc=f"Complexity {complexity}", leave=False):
# #                     if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
# #                         print("Time limit reached during candidate evaluation. Halting search.")
# #                         new_found = False
# #                         break
# #                     try:
# #                         candidate_series = candidate_func(self.knowledge_table)
# #                     except Exception as e:
# #                         print(f"Skipping candidate {candidate_expr} due to error: {e}")
# #                         continue
# #                     # Check that candidate inequality holds.
# #                     if self.bound_type == 'lower':
# #                         if not (self.knowledge_table[self.target] >= candidate_series).all():
# #                             continue
# #                     else:
# #                         if not (self.knowledge_table[self.target] <= candidate_series).all():
# #                             continue
# #                     # Check significance: require at least one row of equality.
# #                     touches = int((self.knowledge_table[self.target] == candidate_series).sum())
# #                     if touches == 0:
# #                         continue
# #                     self._record_conjecture(complexity, candidate_expr, candidate_func)
# #                     new_found = True
# #                     break
# #                 if new_found:
# #                     break
# #             if not new_found:
# #                 print("No further significant conjectures found within the maximum complexity.")
# #                 break

# #     def get_accepted_conjectures(self):
# #         """
# #         Returns the accepted conjectures (BoundConjecture objects) sorted in descending order by touch.
# #         """
# #         return sorted(self.accepted_conjectures, key=lambda c: c.touch, reverse=True)

# #     def write_on_the_wall(self):
# #         """
# #         Print the accepted conjectures.
# #         """
# #         print("Graffiti Conjectures:")
# #         print("---------------------")
# #         for conj in self.get_accepted_conjectures():
# #             print(conj)

# #     def save_conjectures_to_pdf(self, file_name="conjectures.pdf"):
# #         """
# #         Save the accepted conjectures to a PDF file.
# #         """
# #         from reportlab.lib.pagesizes import letter
# #         from reportlab.pdfgen import canvas
# #         from datetime import datetime

# #         pdf = canvas.Canvas(file_name, pagesize=letter)
# #         width, height = letter
# #         pdf.setFont("Helvetica-Bold", 16)
# #         pdf.drawString(50, height - 50, "Generated Conjectures")
# #         pdf.setFont("Helvetica", 10)
# #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# #         pdf.drawString(50, height - 70, f"Generated on: {timestamp}")
# #         y_position = height - 100
# #         pdf.setFont("Helvetica", 10)
# #         for i, conj in enumerate(self.get_accepted_conjectures(), 1):
# #             line = f"{i}. {conj.full_expr} (touch: {conj.touch})"
# #             if y_position < 50:
# #                 pdf.showPage()
# #                 y_position = height - 50
# #             pdf.drawString(50, y_position, line)
# #             y_position -= 20
# #         pdf.save()
# #         print(f"Conjectures saved to {file_name}")

# #     def conjecture(self, target, bound_type='lower', time_limit_minutes=1,
# #                    csv_path=None, df=None, candidate_antecedents=None, filter_property=None):
# #         """
# #         Main entry point for generating conjectures.
# #         Performs all heavy work:
# #           - Reads data (if csv_path or df provided),
# #           - Sets the target and bound type,
# #           - Sets the time limit,
# #           - Optionally filters the data (if filter_property provided),
# #           - Generates candidate antecedents,
# #           - Runs the search loop,
# #           - Stores results in self.conjectures.
# #         """
# #         # Read data if provided.
# #         if csv_path is not None:
# #             self.read_csv(csv_path)
# #         elif df is not None:
# #             self.knowledge_table = df.copy()
# #         # TODO: check for empty knowledge_table

# #         self.target = target
# #         self.bound_type = bound_type
# #         self.time_limit = time_limit_minutes * 60

# #         # Filter data if needed.
# #         if filter_property is not None:
# #             self.knowledge_table = self.knowledge_table[self.knowledge_table[filter_property] == True].copy()
# #             self.hypothesis_str = filter_property
# #         else:
# #             self.hypothesis_str = None

# #         # Generate candidate components (this computes candidate_cols as well).
# #         self._generate_candidate_components(target, candidate_antecedents)

# #         # Clear previous conjectures.
# #         self.accepted_conjectures = []

# #         # Run the search.
# #         self.search()

# #         # Store results.
# #         self.conjectures = {target: {"implications": self.get_accepted_conjectures()}}
