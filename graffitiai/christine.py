from collections import defaultdict
import pandas as pd
from itertools import combinations
from fractions import Fraction
import time

# Import the ImplicationConjecture from your conjectures module.
from graffitiai.base import ImplicationConjecture
from graffitiai.base import BaseConjecturer  # Assuming BaseConjecturer is defined elsewhere.

__all__ = ["Christine"]


class Christine(BaseConjecturer):
    """
    Christine is a specialized conjecturer that searches for implication‐based conjectures.

    Given a target column and a bound direction, it searches for candidate antecedents (functions on
    numeric columns) and candidate properties (from boolean columns or equalities among numeric columns)
    such that an implication holds:

       If target {>= or <=} antecedent then property holds.

    All data cleaning is handled by the inherited BaseConjecturer.read_csv method.
    All heavy initialization (e.g. reading the CSV, setting a time limit, candidate generation)
    is deferred to the conjecture() method.
    """
    def __init__(self):
        super().__init__()
        self.accepted_conjectures = []
        self.conjectures = {}
        # Candidate components will be generated later.
        self.candidate_antecedents = None
        self.candidate_properties = None

    def _generate_candidate_components(self, target, candidate_antecedents=None, candidate_properties=None):
        # Candidate Antecedents.
        if candidate_antecedents is None:
            num_cols = [col for col in self.knowledge_table.columns
                        if col != target and
                           pd.api.types.is_numeric_dtype(self.knowledge_table[col]) and
                           not pd.api.types.is_bool_dtype(self.knowledge_table[col])]
            base_candidates = [(col, lambda df, col=col: df[col]) for col in num_cols]

            self.ratios = [
                Fraction(1, 10), Fraction(3, 10), Fraction(7, 10), Fraction(9, 10),
                Fraction(1, 9),  Fraction(2, 9),  Fraction(4, 9),  Fraction(5, 9),
                Fraction(7, 9),  Fraction(8, 9),  Fraction(10, 9),
                Fraction(1, 8),  Fraction(3, 8),  Fraction(5, 8),  Fraction(7, 8),  Fraction(9, 8),
                Fraction(1, 7),  Fraction(2, 7),  Fraction(3, 7),  Fraction(4, 7),  Fraction(5, 7),
                Fraction(6, 7),  Fraction(8, 7),  Fraction(9, 7),
                Fraction(1, 6),  Fraction(5, 6),  Fraction(7, 6),
                Fraction(1, 5),  Fraction(2, 5),  Fraction(3, 5),  Fraction(4, 5),  Fraction(6, 5),
                Fraction(7, 5),  Fraction(8, 5),  Fraction(9, 5),
                Fraction(1, 4),
                Fraction(1, 3),  Fraction(2, 3),  Fraction(4, 3),  Fraction(5, 3),
                Fraction(7, 3),  Fraction(8, 3),  Fraction(10, 3),
                Fraction(1, 2),  Fraction(3, 2),  Fraction(5, 2),  Fraction(7, 2),  Fraction(9, 2),
                Fraction(1, 1),  Fraction(2, 1),
            ]
            ratio_candidates = []
            for col in num_cols:
                for ratio in self.ratios:
                    ratio_candidates.append((
                        f"{ratio}*({col})",
                        lambda df, col=col, ratio=ratio: float(ratio) * df[col]
                    ))
            complexity3_candidates = self._generate_candidates_complexity3_hypothesis(num_cols)
            self.candidate_antecedents = base_candidates + ratio_candidates + complexity3_candidates
        else:
            self.candidate_antecedents = candidate_antecedents

        # Candidate Properties.
        if candidate_properties is None:
            candidate_props = []
            bool_cols = [col for col in self.knowledge_table.columns
                         if pd.api.types.is_bool_dtype(self.knowledge_table[col])]
            for col in bool_cols:
                candidate_props.append((col, lambda df, col=col: df[col]))
            num_cols_all = [col for col in self.knowledge_table.columns
                            if col != target and pd.api.types.is_numeric_dtype(self.knowledge_table[col])
                            and not pd.api.types.is_bool_dtype(self.knowledge_table[col])]
            for col1, col2 in combinations(num_cols_all, 2):
                expr = f"({col1} = {col2})"
                candidate_props.append((expr, lambda df, col1=col1, col2=col2: df[col1] == df[col2]))
            self.candidate_properties = candidate_props
        else:
            self.candidate_properties = candidate_properties

    def _generate_candidates_complexity3_hypothesis(self, num_cols):
        candidates = []
        for col1, col2 in combinations(num_cols, 2):
            candidates.append((
                f"({col1} * {col2})",
                lambda df, col1=col1, col2=col2: df[col1] * df[col2]
            ))
            candidates.append((
                f"({col1} + {col2})",
                lambda df, col1=col1, col2=col2: df[col1] + df[col2]
            ))
            if (self.knowledge_table[col2] == 0).sum() == 0:
                candidates.append((
                    f"({col1} / {col2})",
                    lambda df, col1=col1, col2=col2: df[col1] / df[col2]
                ))
            if (self.knowledge_table[col1] == 0).sum() == 0:
                candidates.append((
                    f"({col2} / {col1})",
                    lambda df, col1=col1, col2=col2: df[col2] / df[col1]
                ))
            candidates.append((
                f"min({col1}, {col2})",
                lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).min(axis=1)
            ))
            candidates.append((
                f"max({col1}, {col2})",
                lambda df, col1=col1, col2=col2: pd.concat([df[col1], df[col2]], axis=1).max(axis=1)
            ))
        return candidates

    def _implication_holds(self, antecedent_series, prop_series):
        if self.bound_type == 'lower':
            condition = self.knowledge_table[self.target] >= antecedent_series
        else:
            condition = self.knowledge_table[self.target] <= antecedent_series
        if condition.sum() == 0:
            return False
        return prop_series[condition].all()

    def _record_conjecture(self, ant_str, prop_str, ant_func, prop_func, min_support=10, min_touch=1):
        """
        Create and record an ImplicationConjecture if the implication holds.
        Stronger acceptance criteria are imposed: the candidate must have a support of at least
        min_support and a touch value of at least min_touch. If a duplicate exists with the same antecedent
        and property, it is replaced if the new one has higher support.
        """
        new_conj = ImplicationConjecture(
            target=self.target,
            antecedent_expr=ant_str,
            property_expr=prop_str,
            ant_func=ant_func,
            prop_func=prop_func,
            bound_type=self.bound_type
            # Optionally, pass hypothesis and complexity here.
        )
        new_conj.compute_support(self.knowledge_table)
        new_conj.compute_touch(self.knowledge_table)

        # Apply stronger acceptance criteria:
        if new_conj.support < min_support:
            # print("Rejected conjecture due to insufficient support:", new_conj.full_expr)
            return
        if new_conj.touch < min_touch:
            # print("Rejected conjecture due to insufficient touch:", new_conj.full_expr)
            return

        # Check for duplicates.
        duplicate_found = False
        for idx, existing in enumerate(self.accepted_conjectures):
            if (existing.antecedent_expr == new_conj.antecedent_expr and
                existing.property_expr == new_conj.property_expr):
                duplicate_found = True
                if new_conj.support > existing.support:
                    self.accepted_conjectures[idx] = new_conj
                    print("Replaced duplicate with a tighter conjecture:", new_conj.full_expr)
                else:
                    print("Duplicate found; keeping the existing conjecture:", existing.full_expr)
                break

        if not duplicate_found:
            self.accepted_conjectures.append(new_conj)
            print("Accepted conjecture:", new_conj.full_expr)

        self._prune_conjectures()

    def _prune_conjectures(self):
        """
        Prune duplicate conjectures using a composite key: (antecedent_expr, property_expr, support).
        """
        unique_conjs = {}
        for conj in self.accepted_conjectures:
            key = (conj.antecedent_expr, conj.property_expr, conj.support)
            if key not in unique_conjs or conj.support > unique_conjs[key].support:
                unique_conjs[key] = conj
        before = len(self.accepted_conjectures)
        self.accepted_conjectures = list(unique_conjs.values())
        after = len(self.accepted_conjectures)
        if after < before:
            print(f"Pruned duplicate conjectures: reduced from {before} to {after}.")

    def consolidate_conjectures(self):
        """
        Consolidate accepted conjectures by grouping those with the same antecedent and support.
        Returns a list of consolidated conjecture strings.
        """
        groups = defaultdict(list)
        for conj in self.accepted_conjectures:
            key = (conj.antecedent_expr, conj.support)
            groups[key].append(conj.property_expr)
        bound_symbol = ">=" if self.bound_type == 'lower' else "<="
        consolidated = []
        for (ant, support), props in groups.items():
            props = sorted(set(props))
            if len(props) > 1:
                eq_props = " ⇔ ".join(props)
                consolidated.append(f"Conjecture. If {self.target} {bound_symbol} {ant}, then ({eq_props}) [support: {support}]")
            else:
                consolidated.append(f"Conjecture. If {self.target} {bound_symbol} {ant}, then {props[0]} [support: {support}]")
        return consolidated

    def search(self):
        """
        The main search loop.
        Iterates over candidate antecedents and candidate properties.
        For each pair, if the implication holds, record the conjecture.
        Stops if the time limit is reached.
        """
        start_time = time.time()
        for ant_str, ant_func in self.candidate_antecedents:
            try:
                ant_series = ant_func(self.knowledge_table)
            except Exception as e:
                print(f"Error evaluating antecedent '{ant_str}':", e)
                continue
            for prop_str, prop_func in self.candidate_properties:
                try:
                    prop_series = prop_func(self.knowledge_table)
                except Exception as e:
                    print(f"Error evaluating property '{prop_str}':", e)
                    continue
                if self._implication_holds(ant_series, prop_series):
                    self._record_conjecture(ant_str, prop_str, ant_func, prop_func)
                if time.time() - start_time >= self.time_limit:
                    print("Time limit reached, stopping search.")
                    return

    def get_accepted_conjectures(self):
        return self.accepted_conjectures

    def write_on_the_wall(self):
        from pyfiglet import Figlet
        fig = Figlet(font='slant')

        # Print the main title.
        title = fig.renderText("Graffiti AI: Christine")
        print(title)

        if not hasattr(self, 'conjectures') or not self.conjectures:
            print("No conjectures generated yet!")
            return
        for c in self.consolidate_conjectures():
            print(c)
            print()

    def conjecture(self, target, bound_type='lower', time_limit_minutes=1,
                   csv_path=None, df=None, candidate_antecedents=None, candidate_properties=None):
        """
        The main entry point for generating conjectures.
        All heavy work occurs here:
          - The data is read (if needed),
          - The target and bound type are set,
          - The time limit is set,
          - Candidate antecedents and properties are generated,
          - The search is run,
          - Results are stored in self.conjectures.

        Parameters:
            target (str): The target column.
            bound_type (str): 'lower' or 'upper'.
            time_limit_minutes (float): Time limit in minutes.
            csv_path (str, optional): Path to a CSV file.
            df (pd.DataFrame, optional): A DataFrame with the data.
            candidate_antecedents (iterable, optional): Overrides default antecedent candidates.
            candidate_properties (iterable, optional): Overrides default property candidates.
        """
        # Read data if provided.
        if csv_path is not None:
            self.read_csv(csv_path)
        elif df is not None:
            self.knowledge_table = df.copy()

        self.target = target
        self.bound_type = bound_type
        self.time_limit = time_limit_minutes * 60

        self._generate_candidate_components(target, candidate_antecedents, candidate_properties)
        self.accepted_conjectures = []
        self.search()
        self.conjectures = {target: {"implications": self.get_accepted_conjectures()}}

