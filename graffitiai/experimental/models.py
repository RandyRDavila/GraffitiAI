
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm  # progress bar
from fractions import Fraction  # for representing ratios
import time  # for the time limit

from graffitiai import TxGraffiti
from graffitiai.experimental.expression_generator import generate_expressions, simplify

__all__ = [
    "GraffitiII",
    "Graffiti",
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
            self.hypothesis_str = filter_property  # store the filter column as hypothesis
        else:
            self.df = df.copy()
            self.hypothesis_str = None

        self.target = target_invariant
        self.bound_type = bound_type
        self.time_limit = time_limit  # time limit in seconds

        # Choose candidate columns: numeric (but not boolean) and not the target.
        self.candidate_cols = [
            col for col in self.df.columns
            if col != target_invariant and
               pd.api.types.is_numeric_dtype(self.df[col]) and
               not pd.api.types.is_bool_dtype(self.df[col])
        ]

        self.accepted_conjectures = []
        # We now use complexities 1 through 7.
        self.max_complexity = 7

    def _compute_current_bound(self):
        if not self.accepted_conjectures:
            if self.bound_type == 'lower':
                return pd.Series(-np.inf, index=self.df.index)
            else:
                return pd.Series(np.inf, index=self.df.index)
        bounds = []
        for conj in self.accepted_conjectures:
            try:
                b = conj['func'](self.df)
                bounds.append(b)
            except Exception as e:
                print("Error computing accepted bound:", conj['full_expr_str'], e)
        if self.bound_type == 'lower':
            return pd.concat(bounds, axis=1).max(axis=1)
        else:
            return pd.concat(bounds, axis=1).min(axis=1)

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
            return (diff > 0).any()
        else:
            diff = current_bound - candidate_series
            return (diff > 0).any()

    def _record_conjecture(self, complexity, rhs_str, func):
        if self.hypothesis_str:
            if self.bound_type == 'lower':
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} >= {rhs_str}."
            else:
                full_expr_str = f"For any {self.hypothesis_str}, {self.target} <= {rhs_str}."
        else:
            if self.bound_type == 'lower':
                full_expr_str = f"{self.target} >= {rhs_str}."
            else:
                full_expr_str = f"{self.target} <= {rhs_str}."
        new_conj = {
            'complexity': complexity,
            'rhs_str': rhs_str,
            'full_expr_str': full_expr_str,
            'func': func
        }
        # Compute the touch number: count rows where target equals candidate value.
        candidate_series = func(self.df)
        touches = int((self.df[self.target] == candidate_series).sum())
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
            series_i = conj_i['func'](self.df)
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                conj_j = self.accepted_conjectures[j]
                series_j = conj_j['func'](self.df)
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

    # Helper: Given a candidate (rhs_str, func), return floor and ceiling variants.
    def _with_floor_ceil(self, candidate):
        base_rhs, base_func = candidate
        floor_candidate = (f"floor({base_rhs})", lambda df, base_func=base_func: np.floor(base_func(df)))
        ceil_candidate = (f"ceil({base_rhs})", lambda df, base_func=base_func: np.ceil(base_func(df)))
        return [candidate, floor_candidate, ceil_candidate]

    # ----------------------------
    # Complexity 1: Single candidate invariant.
    # ----------------------------
    def _generate_candidates_complexity1(self):
        candidates = []
        for col in self.candidate_cols:
            rhs_str = f"{col}"
            func = lambda df, col=col: df[col]
            candidates.append((rhs_str, func))
        return candidates

    # ----------------------------
    # Complexity 2: Unary operations plus ratio multiplication.
    # ----------------------------
    def _generate_candidates_complexity2(self):
        candidates = []
        for col in self.candidate_cols:
            # candidates.append((f"sqrt({col})", lambda df, col=col: np.sqrt(df[col])))
            candidates.append((f"({col})^2", lambda df, col=col: df[col]**2))
            candidates.append((f"floor({col})", lambda df, col=col: np.floor(df[col])))
            candidates.append((f"ceil({col})", lambda df, col=col: np.ceil(df[col])))
            # Expanded set of ratio multiplicative candidates.
            ratios = [Fraction(1,10), Fraction(1,8), Fraction(1,6), Fraction(1,4),
                      Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(3,4),
                      Fraction(1,1), Fraction(4,3), Fraction(3,2), Fraction(2,1),
                      Fraction(3,1), Fraction(4,1), Fraction(5,1)]
            for ratio in ratios:
                base_expr = f"{ratio}*({col})"
                func = lambda df, col=col, ratio=ratio: float(ratio) * df[col]
                candidates.append((base_expr, func))
                candidates.append((f"floor({base_expr})", lambda df, col=col, ratio=ratio: np.floor(float(ratio)*df[col])))
                candidates.append((f"ceil({base_expr})", lambda df, col=col, ratio=ratio: np.ceil(float(ratio)*df[col])))

            for ratio in ratios:
                base_expr = f"({col}) + {ratio}"
                func = lambda df, col=col, ratio=ratio:  df[col] + float(ratio)
                candidates.append((base_expr, func))

            for ratio in ratios:
                base_expr = f"({col}) - {ratio}"
                func = lambda df, col=col, ratio=ratio:  df[col] - float(ratio)
                candidates.append((base_expr, func))

        return candidates

    # ----------------------------
    # Complexity 3: Binary operations (using 2 invariants) with floor/ceil variants.
    # Now we also add max and min operations.
    # ----------------------------
    def _generate_candidates_complexity3(self):
        candidates = []
        for col1, col2 in combinations(self.candidate_cols, 2):
            base_candidates = [
                (f"{col1} + {col2}", lambda df, col1=col1, col2=col2: df[col1] + df[col2]),
                (f"{col1} - {col2}", lambda df, col1=col1, col2=col2: df[col1] - df[col2]),
                (f"{col2} - {col1}", lambda df, col1=col1, col2=col2: df[col2] - df[col1]),
                (f"{col1} * {col2}", lambda df, col1=col1, col2=col2: df[col1] * df[col2]),
                (f"max({col1}, {col2})", lambda df, col1=col1, col2=col2: np.maximum(df[col1], df[col2])),
                (f"min({col1}, {col2})", lambda df, col1=col1, col2=col2: np.minimum(df[col1], df[col2]))
            ]
            # Division candidates (if safe)
            if (self.df[col2] == 0).sum() == 0:
                base_candidates.append((f"{col1} / {col2}", lambda df, col1=col1, col2=col2: df[col1] / df[col2]))
            if (self.df[col1] == 0).sum() == 0:
                base_candidates.append((f"{col2} / {col1}", lambda df, col1=col1, col2=col2: df[col2] / df[col1]))
            for cand in base_candidates:
                candidates.extend(self._with_floor_ceil(cand))
        return candidates

    # ----------------------------
    # Complexity 4: Combined operations on 2 invariants, with floor/ceil variants.
    # ----------------------------
    def _generate_candidates_complexity4(self):
        candidates = []
        for col1, col2 in combinations(self.candidate_cols, 2):
            base_candidates = [
                (f"({col1} + {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] + df[col2])**2),
                (f"2*({col1} + {col2})", lambda df, col1=col1, col2=col2: 2*(df[col1] + df[col2])),
                (f"({col1} + {col2})/2", lambda df, col1=col1, col2=col2: (df[col1] + df[col2]) / 2)
            ]
            if (self.df[col1] + self.df[col2] < 0).sum() == 0:
                base_candidates.append((f"sqrt({col1} + {col2})", lambda df, col1=col1, col2=col2: np.sqrt(df[col1] + df[col2])))
            base_candidates.append((f"({col1} - {col2})^2", lambda df, col1=col1, col2=col2: (df[col1] - df[col2])**2))
            for cand in base_candidates:
                candidates.extend(self._with_floor_ceil(cand))
        return candidates

    # ----------------------------
    # Complexity 5: Powers of previously accepted conjectures.
    # ----------------------------
    def _generate_candidates_complexity5(self):
        candidates = []
        if not self.accepted_conjectures:
            return candidates
        for accepted in self.accepted_conjectures:
            for exponent in [2, 3]:
                new_rhs = f"({accepted['rhs_str']})^{exponent}"
                func = lambda df, func_old=accepted['func'], exponent=exponent: func_old(df)**exponent
                candidates.append((new_rhs, func))
        return candidates

    # ----------------------------
    # Complexity 6: Incorporate additive and multiplicative constants (using Fraction constants).
    # ----------------------------
    def _generate_candidates_complexity6(self):
        candidates = []
        constants = [Fraction(1,10), Fraction(1,8), Fraction(1,6), Fraction(1,4),
                     Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(3,4),
                     Fraction(1,1), Fraction(4,3), Fraction(3,2), Fraction(2,1),
                     Fraction(3,1), Fraction(4,1), Fraction(5,1)]
        for col in self.candidate_cols:
            for c in constants:
                base_expr = f"{c}*({col})"
                func = lambda df, col=col, c=c: float(c)*df[col]
                candidates.append((base_expr, func))
                candidates.append((f"floor({base_expr})", lambda df, col=col, c=c: np.floor(float(c)*df[col])))
                candidates.append((f"ceil({base_expr})", lambda df, col=col, c=c: np.ceil(float(c)*df[col])))
                candidates.append((f"({col}) + {c}", lambda df, col=col, c=c: df[col] + float(c)))
                candidates.append((f"({col}) - {c}", lambda df, col=col, c=c: df[col] - float(c)))
        return candidates

    # ----------------------------
    # Complexity 7: Recursive combination to allow three or more invariants.
    # ----------------------------
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
                # Division could be added with appropriate checks.
            ]
            for cand1 in prev_candidates:
                for cand2 in atomic_candidates:
                    for op_sym, op_func in operators:
                        new_expr = f"({cand1[0]}) {op_sym} ({cand2[0]})"
                        new_func = lambda df, cand1=cand1, cand2=cand2, op_func=op_func: op_func(cand1[1](df), cand2[1](df))
                        new_candidates.append((new_expr, new_func))
            return prev_candidates + new_candidates

    def _generate_candidates_complexity7(self):
        rec_candidates = self._generate_candidates_recursive(max_depth=3)
        new_candidates = []
        for expr, func in rec_candidates:
            op_count = expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/')
            if op_count >= 2:  # roughly indicates at least three invariants involved.
                new_candidates.extend(self._with_floor_ceil((expr, func)))
        return new_candidates

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

    def search(self):
        start_time = time.time()
        new_conjecture_found = True
        while new_conjecture_found:
            # Check time limit if set.
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                print("Time limit reached. Halting search.")
                break
            new_conjecture_found = False
            for complexity in range(1, self.max_complexity + 1):
                candidates = self.generate_candidates(complexity)
                for rhs_str, func in tqdm(candidates, desc=f"Complexity {complexity}", leave=False):
                    # Check time limit inside the loop as well.
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

    def get_accepted_conjectures(self):
        return sorted(self.accepted_conjectures, key=lambda c: c['touch'], reverse=True)

    def write_on_the_wall(self):
        conjectures = self.get_accepted_conjectures()
        print("GRAFFITI conjectures:")
        print("------------------------")
        for conj in conjectures:
            print(f"Conjecture. {conj['full_expr_str']} (touch {conj['touch']})")
