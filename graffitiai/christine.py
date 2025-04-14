from collections import defaultdict
import pandas as pd
from itertools import combinations
from fractions import Fraction
import time
import re
from tqdm import tqdm

# Import the ImplicationConjecture from your conjectures module.
from graffitiai.base import ImplicationConjecture
from graffitiai.base import BaseConjecturer  # Assuming BaseConjecturer is defined elsewhere.

# Utility functions.
def extract_multiplier(antecedent_expr):
    # This regex captures a fraction like '1/8' in the antecedent expression.
    match = re.search(r'(\d+/\d+)', antecedent_expr)
    if match:
        return Fraction(match.group(1))
    return None

def prune_weaker_conjectures(conjectures, df, tolerance=0.05):
    """
    Prune conjectures by grouping them based on property_expr and support.
    If any candidate in a group has support equal to the total number of rows
    (i.e. the condition holds on all rows) and its multiplier equals 1,
    that candidate is preferred.

    Parameters:
      conjectures: list of ImplicationConjecture objects.
      df: the knowledge_table DataFrame.
      tolerance: used for comparing support differences if needed.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for conj in conjectures:
        key = (conj.property_expr, conj.support)
        grouped[key].append(conj)
    pruned = []
    n_rows = len(df)
    for group in grouped.values():
        # Look for candidates with full support.
        full_support_candidates = [c for c in group if c.support == n_rows]
        if full_support_candidates:
            # If one of these has multiplier 1, choose it.
            best = None
            for c in full_support_candidates:
                multiplier = extract_multiplier(c.antecedent_expr)
                if multiplier is not None and multiplier == 1:
                    best = c
                    break
            # If none has an exact multiplier 1, take the first candidate with full support.
            if best is None:
                best = full_support_candidates[0]
            pruned.append(best)
        else:
            # Otherwise, if the group has multiple candidates, you can compare support differences
            # or simply choose the one with the highest multiplier.
            if len(group) > 1:
                sorted_group = sorted(group, key=lambda c: extract_multiplier(c.antecedent_expr) or 0)
                best = sorted_group[-1]
                pruned.append(best)
            else:
                pruned.append(group[0])
    return pruned


def prune_by_rhs_variable(conjectures):
    groups = defaultdict(list)
    for conj in conjectures:
        match = re.search(r'>=\s*[\d/]+\*\(\s*([a-zA-Z_]+)\s*\)', conj.antecedent_expr)
        rhs_var = match.group(1) if match else None
        key = (conj.property_expr, rhs_var)
        groups[key].append(conj)
    pruned = []
    for group in groups.values():
        best_conj = max(group, key=lambda c: c.support)
        pruned.append(best_conj)
    return pruned

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
        new_conj = ImplicationConjecture(
            target=self.target,
            antecedent_expr=ant_str,
            property_expr=prop_str,
            ant_func=ant_func,
            prop_func=prop_func,
            bound_type=self.bound_type
        )
        new_conj.compute_support(self.knowledge_table)
        new_conj.compute_touch(self.knowledge_table)

        if new_conj.support < min_support or new_conj.touch < min_touch:
            return

        duplicate_found = False
        for idx, existing in enumerate(self.accepted_conjectures):
            if (existing.antecedent_expr == new_conj.antecedent_expr and
                existing.property_expr == new_conj.property_expr):
                duplicate_found = True
                if new_conj.support > existing.support:
                    self.accepted_conjectures[idx] = new_conj
                    tqdm.write("Replaced duplicate with a tighter conjecture: " + new_conj.full_expr)
                else:
                    tqdm.write("Duplicate found; keeping the existing conjecture: " + existing.full_expr)
                break

        if not duplicate_found:
            self.accepted_conjectures.append(new_conj)
            # tqdm.write("Accepted conjecture: " + new_conj.full_expr)

        self._prune_conjectures()

    def _prune_conjectures(self):
        unique_conjs = {}
        for conj in self.accepted_conjectures:
            key = (conj.antecedent_expr, conj.property_expr, conj.support)
            if key not in unique_conjs or conj.support > unique_conjs[key].support:
                unique_conjs[key] = conj
        before = len(self.accepted_conjectures)
        self.accepted_conjectures = list(unique_conjs.values())
        after = len(self.accepted_conjectures)
        if after < before:
            tqdm.write(f"Pruned duplicate conjectures: reduced from {before} to {after}.")
        # Use the modified pruning function here:
        self.accepted_conjectures = prune_weaker_conjectures(self.accepted_conjectures, self.knowledge_table)
        self.accepted_conjectures = prune_by_rhs_variable(self.accepted_conjectures)

    def filter_general_properties(self, prop_percentages, prop_funcs):
        """
        Given a dict mapping property_expr -> percentage and a corresponding dict mapping
        property_expr -> property function (which returns a boolean series when applied to the knowledge_table),
        filter out any property that is a strict subset of another property.
        Only perform the subset check if a boolean series can be obtained for both properties.
        """
        general_props = dict(prop_percentages)  # Make a copy.
        for propA in list(prop_percentages.keys()):
            for propB in list(prop_percentages.keys()):
                if propA == propB:
                    continue
                # Attempt to get boolean series for both properties.
                # For propA:
                if propA in self.knowledge_table.columns:
                    seriesA = self.knowledge_table[propA]
                elif propA in prop_funcs:
                    seriesA = prop_funcs[propA](self.knowledge_table)
                else:
                    seriesA = None
                # For propB:
                if propB in self.knowledge_table.columns:
                    seriesB = self.knowledge_table[propB]
                elif propB in prop_funcs:
                    seriesB = prop_funcs[propB](self.knowledge_table)
                else:
                    seriesB = None

                # Only compare if both series are available.
                if seriesA is not None and seriesB is not None:
                    if ((seriesA) & (~seriesB)).sum() == 0:
                        supportA = seriesA.sum()
                        supportB = seriesB.sum()
                        if supportA < supportB:
                            general_props.pop(propA, None)
                            break
        return general_props

    def consolidate_conjectures(self):
        """
        Consolidate accepted conjectures by grouping those with the same antecedent and support.
        For each group, for each boolean property compute:
            (# rows satisfying both antecedent and property) / (# rows satisfying property) * 100.
        Only properties with coverage >= 50% are retained.
        Then, filter out properties that are strict subsets of a more general property.
        Finally, create two kinds of consolidated entries:
        A. If-and-only-if conjectures for properties with 100% coverage,
        B. Regular conjectures for the remaining properties.
        Groups with no property meeting the threshold are skipped.
        Additionally, if a boolean column exists for which every row is True,
        its name is prepended (as "For any <col>,") to the antecedent.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for conj in self.accepted_conjectures:
            key = (conj.antecedent_expr, conj.support)
            groups[key].append(conj)

        # Determine a global true column if one exists.
        global_true = self.global_type
        global_prefix = f"For any {global_true}, " if global_true is not None else ""

        consolidated_if_and_only = []  # if-and-only-if entries
        consolidated_regular = []       # regular entries

        for (ant_expr, support), conj_group in groups.items():
            # Determine the group-specific comparison symbol.
            group_bound_symbol = ">=" if self.bound_type == 'lower' else "<="
            if self.bound_type == 'upper' and conj_group and conj_group[0].is_exact_equality(self.knowledge_table):
                group_bound_symbol = "="

            # Build dictionaries mapping property_expr to its coverage percentage and store property functions.
            prop_percentages = {}
            prop_funcs = {}
            for conj in conj_group:
                if conj.property_expr not in prop_percentages:
                    ant_series = conj.ant_func(self.knowledge_table)
                    if self.bound_type == 'lower':
                        condition = self.knowledge_table[self.target] >= ant_series
                    else:
                        condition = self.knowledge_table[self.target] <= ant_series
                    prop_series = conj.prop_func(self.knowledge_table)
                    total_prop = prop_series.sum()  # Assumes booleans (True=1, False=0)
                    if total_prop == 0:
                        percentage = 0
                    else:
                        common = (prop_series & condition).sum()
                        percentage = common / total_prop * 100
                    # Only include properties with coverage >= 50%
                    if percentage >= 20:
                        prop_percentages[conj.property_expr] = percentage
                        prop_funcs[conj.property_expr] = conj.prop_func

            if not prop_percentages:
                continue

            # Filter out properties that are strict subsets of a more general property.
            prop_percentages = self.filter_general_properties(prop_percentages, prop_funcs)

            # Separate properties into if-and-only-if (100%) and regular (>=50% but <100%).
            props_if_and_only = {}
            props_regular = {}
            for prop_expr, pct in prop_percentages.items():
                if abs(pct - 100) == 0.0:
                    props_if_and_only[prop_expr] = pct
                else:
                    props_regular[prop_expr] = pct

            # Build consolidated string for if-and-only-if conjectures.
            if props_if_and_only:
                if len(props_if_and_only) > 1:
                    prop_list = " ⇔ ".join(props_if_and_only.keys())
                else:
                    prop_list = list(props_if_and_only.keys())[0]
                if_and_only_str = (
                    f"Conjecture (if and only if):\n"
                    f"{global_prefix}{self.target} {group_bound_symbol} {ant_expr}\n"
                    f"if and only if {prop_list}\n"
                    f"[support: {support}]"
                )
                consolidated_if_and_only.append((100, len(props_if_and_only), if_and_only_str))

            # Build consolidated string for regular conjectures.
            if props_regular:
                prop_strs = []
                for prop_expr, pct in sorted(props_regular.items()):
                    prop_strs.append(f"{prop_expr} ({pct:.0f}%)")
                if len(prop_strs) > 1:
                    conclusion_str = "\n".join("    " + prop for prop in prop_strs)
                else:
                    conclusion_str = prop_strs[0]
                max_pct = max(props_regular.values()) if props_regular else 0
                count_booleans = len(prop_strs)
                reg_str = (
                    f"Conjecture:\n"
                    f"For any {self.global_type}, if {self.target} {group_bound_symbol} {ant_expr},\n"
                    f"then:\n{conclusion_str}\n"
                    f"[support: {support}]"
                )
                consolidated_regular.append((max_pct, count_booleans, reg_str))

        consolidated_if_and_only.sort(key=lambda x: (x[0], x[1]), reverse=True)
        consolidated_regular.sort(key=lambda x: (x[0], x[1]), reverse=True)

        all_entries = [s for _, _, s in consolidated_if_and_only] + [s for _, _, s in consolidated_regular]
        return all_entries

    def search(self):
        """
        The main search loop using a time-based progress bar.
        The progress bar's total is set to the time limit (in seconds) and updated based on elapsed time.
        """
        start_time = time.time()
        pbar = tqdm(total=self.time_limit, desc=f"Searching {self.bound_type} conjectures (time-based)")
        last_time = start_time

        for ant_str, ant_func in self.candidate_antecedents:
            try:
                ant_series = ant_func(self.knowledge_table)
            except Exception as e:
                for _ in self.candidate_properties:
                    current_time = time.time()
                    dt = current_time - last_time
                    pbar.update(dt)
                    last_time = current_time
                continue

            for prop_str, prop_func in self.candidate_properties:
                current_time = time.time()
                dt = current_time - last_time
                pbar.update(dt)
                last_time = current_time

                if current_time - start_time >= self.time_limit:
                    pbar.set_description("Time limit reached, stopping search.")
                    pbar.close()
                    return

                try:
                    prop_series = prop_func(self.knowledge_table)
                except Exception as e:
                    continue

                if self._implication_holds(ant_series, prop_series):
                    self._record_conjecture(ant_str, prop_str, ant_func, prop_func)
        pbar.close()

    def get_accepted_conjectures(self):
        return self.accepted_conjectures

    def write_on_the_wall(self):
        from pyfiglet import Figlet
        fig = Figlet(font='slant')
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
