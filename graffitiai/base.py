import pandas as pd
import numpy as np
from itertools import combinations, permutations
from tqdm import tqdm

from graffitiai.utils import (
    is_list_string,
    convert_and_no_pad,
    convert_and_pad,
    expand_statistics,
)

__all__ = [
    "BoundConjecture",
    "BaseConjecturer",
]

class BoundConjecture:
    """
    Represents a bound conjecture of the form:
       If (hypothesis), then (target bound_type candidate_expr)
    where bound_type is 'lower' (target ≥ candidate) or 'upper' (target ≤ candidate).
    """
    def __init__(
            self,
            target,
            candidate_expr,
            candidate_func,
            bound_type='lower',
            hypothesis=None,
            complexity=None,
            touch=None,
            sharp_instances=None,
            conclusion=None,
            callable=None,
            true_objects=None,
            keywords=None,
        ):
        self.target = target
        self.candidate_expr = candidate_expr
        self.candidate_func = candidate_func
        self.bound_type = bound_type
        self.hypothesis = hypothesis
        self.complexity = complexity
        self.touch = touch
        self.full_expr = self.format_full_expression()
        self.conclusion = self._set_conclusion()
        self.sharp_instances = sharp_instances
        self.conclusion = conclusion
        self.callable = callable
        self.true_objects = true_objects
        self.keywords = keywords

    @staticmethod
    def simplify_expression(expr: str) -> str:
        """
        Simplify an expression string by removing terms that are 0 or 0*(something),
        including cases where the term is preceded by a minus sign.
        The expression is assumed to have terms joined by " + ".
        """
        # Split the expression into terms by the plus sign.
        terms = [t.strip() for t in expr.split('+')]
        nonzero_terms = []
        for term in terms:
            # Remove spaces and then remove a leading '-' (if any) for checking.
            compact = term.replace(" ", "")
            if compact.startswith('-'):
                compact = compact[1:]
            # Skip the term if it's exactly "0" or starts with "0*"
            if compact == "0" or compact.startswith("0*"):
                continue
            nonzero_terms.append(term)
        # If all terms are filtered out, return "0".
        if not nonzero_terms:
            return "0"
        return " + ".join(nonzero_terms)

    def format_full_expression(self):
        # Simplify the candidate expression before formatting.
        simplified_expr = BoundConjecture.simplify_expression(self.candidate_expr)
        if self.hypothesis:
            if self.bound_type == 'lower':
                return f"For any {self.hypothesis}, {self.target} ≥ {simplified_expr}"
            elif self.bound_type == 'upper':
                return f"For any {self.hypothesis}, {self.target} ≤ {simplified_expr}"
            else:
                return f"For any {self.hypothesis}, {self.target} = {simplified_expr}"
        else:
            if self.bound_type == 'lower':
                return f"{self.target} ≥ {simplified_expr}"
            elif self.bound_type == 'upper':
                return f"{self.target} ≤ {simplified_expr}"
            else:
                return f"{self.target} = {simplified_expr}"

    def _set_conclusion(self):
        # This method is a placeholder. In a real implementation, this might evaluate the conclusion.
        if self.bound_type == 'lower':
            return f"{self.target} ≥ {self.candidate_expr}"
        elif self.bound_type == 'upper':
            return f"{self.target} ≤ {self.candidate_expr}"
        else:
            return f"{self.target} = {self.candidate_expr}"

    def evaluate(self, df):
        """Evaluate the candidate function on the given DataFrame."""
        return self.candidate_func(df)

    def __call__(self, df):
        return self.callable(df)

    def compute_touch(self, df):
        """Compute how many rows satisfy equality between the target and candidate."""
        candidate_series = self.evaluate(df)
        self.touch = int((df[self.target] == candidate_series).sum())
        return self.touch

    def get_sharp_objects(self, df):
        """
        Compute and return the set of row identifiers (using the 'name' column if available,
        or the index otherwise) where the candidate equals the target.
        """
        candidate_series = self.evaluate(df)
        target_series = df[self.target]
        if "name" in df.columns:
            return set(df.loc[target_series == candidate_series, "name"])
        else:
            return set(df.index[target_series == candidate_series])

    def false_objects(self, df):
        """
        Returns the subset of rows where the conjecture does NOT hold.
        For a 'lower' bound, these are rows where target < candidate;
        for an 'upper' bound, rows where target > candidate.
        """
        candidate_series = self.evaluate(df)
        target_series = df[self.target]
        if self.bound_type == 'lower':
            false_mask = target_series < candidate_series
        else:
            false_mask = target_series > candidate_series
        return df[false_mask]

    def is_less_general_than(self, other):
        """
        Check if this conjecture is less general than another.
        A conjecture is less general if it has the same conclusion
        as the other but a more specific hypothesis (i.e., its true_objects
        is a strict subset of the other's true_objects).
        """
        if self.conclusion != other.conclusion:
            return False
        elif self.hypothesis is None:
            return False
        elif other.hypothesis is None:
            return True
        return self.true_objects < other.true_objects

    def __hash__(self):
        # Use the full expression (which captures target, candidate_expr, hypothesis, etc.)
        return hash(self.full_expr)

    def __eq__(self, other):
        if not isinstance(other, BoundConjecture):
            return False
        return self.full_expr == other.full_expr

    def __str__(self):
        return f"{self.full_expr}"

    def __repr__(self):
        return f"BoundConjecture({self.full_expr!r}, touch={self.touch}, complexity={self.complexity})"


class ImplicationConjecture:
    """
    Represents an implication-based conjecture of the form:

      If target {≥ or ≤} antecedent_expr, then property_expr holds.

    For example:
      If zero_forcing_number ≤ min_degree, then (connected_zero_forcing_number = min_degree)
      but if the data shows that zero_forcing_number ≥ min_degree always, then the hypothesis forces
      zero_forcing_number = min_degree.

    Attributes:
        target (str): The target invariant (e.g. "number_of_6_gons").
        antecedent_expr (str): A human-readable expression for the antecedent.
        property_expr (str): A human-readable expression for the property (conclusion).
        ant_func (callable): A function that accepts a DataFrame and returns a Series for the antecedent.
        prop_func (callable): A function that accepts a DataFrame and returns a boolean Series for the property.
        bound_type (str): Either 'lower' (using ≥) or 'upper' (using ≤).
        hypothesis (str, optional): An optional additional condition.
        complexity (int, optional): A measure of the conjecture’s complexity.
        support (int, optional): The number of rows where the antecedent holds.
        full_expr (str): A fully formatted human‑readable expression.
        touch (int, optional): The number of rows where the implication is satisfied.
    """
    def __init__(self, target, antecedent_expr, property_expr, ant_func, prop_func,
                 bound_type='lower', hypothesis=None, complexity=None, support=None):
        self.target = target
        self.antecedent_expr = antecedent_expr
        self.property_expr = property_expr
        self.ant_func = ant_func
        self.prop_func = prop_func  # Stored for later use (e.g. in filtering)
        self.bound_type = bound_type
        self.hypothesis = hypothesis
        self.complexity = complexity
        self.support = support
        self.touch = None  # to be computed later (e.g. with compute_touch)
        # Initially format full_expr without DataFrame-specific info.
        self.full_expr = self.format_full_expression()

    def format_full_expression(self):
        """
        Format the full human-readable expression for the conjecture using the default inequality.
        This does not yet account for data-driven equality.
        """
        symbol = ">=" if self.bound_type == 'lower' else "<="
        expr = f"If {self.target} {symbol} {self.antecedent_expr}, then {self.property_expr}"
        if self.support is not None:
            expr += f" [support: {self.support}]"
        if self.hypothesis:
            expr = f"For any {self.hypothesis}, " + expr
        return expr

    def is_exact_equality(self, df):
        """
        For an upper-bound conjecture (hypothesis: target <= antecedent):
        If the data shows that target >= antecedent on all rows, then target = antecedent.
        Similarly, for a lower-bound conjecture (hypothesis: target >= antecedent):
        If the data shows that target <= antecedent on all rows, then target = antecedent.
        """
        ant_series = self.ant_func(df)
        target_series = df[self.target]
        if self.bound_type == 'upper':
            # Hypothesis is target <= ant_series.
            # If data shows target_series >= ant_series always, then equality holds.
            return (target_series >= ant_series).all()
        else:  # bound_type == 'lower'
            # Hypothesis is target >= ant_series.
            # If data shows target_series <= ant_series always, then equality holds.
            return (target_series <= ant_series).all()


    def get_full_expression(self, df):
        """
        Returns the full human-readable expression for the conjecture using the appropriate
        comparison symbol. For upper-bound conjectures, if the data forces equality, uses "=".
        """
        if self.bound_type == 'upper' and self.is_exact_equality(df):
            symbol = "="
        else:
            symbol = ">=" if self.bound_type == 'lower' else "<="
        expr = f"If {self.target} {symbol} {self.antecedent_expr}, then {self.property_expr}"
        if self.support is not None:
            expr += f" [support: {self.support}]"
        if self.hypothesis:
            expr = f"For any {self.hypothesis}, " + expr
        return expr

    def compute_support(self, df):
        """
        Compute the support of the antecedent: the number of rows where the condition holds.
        """
        ant_series = self.ant_func(df)
        if self.bound_type == 'lower':
            condition = df[self.target] >= ant_series
        else:
            condition = df[self.target] <= ant_series
        self.support = int(condition.sum())
        return self.support

    def get_property_series(self, df):
        """
        Returns a boolean Series representing the property by applying the stored property function.
        """
        return self.prop_func(df)

    def evaluate(self, df):
        """
        Evaluate the implication on the DataFrame.
        Returns the boolean Series for the property evaluated on rows where the antecedent holds.
        """
        ant_series = self.ant_func(df)
        prop_series = self.prop_func(df)
        if self.bound_type == 'lower':
            condition = df[self.target] >= ant_series
        else:
            condition = df[self.target] <= ant_series
        return prop_series[condition]

    def compute_touch(self, df):
        """
        Compute the "touch" value: the number of rows where both the antecedent holds and the property is True.
        """
        ant_series = self.ant_func(df)
        prop_series = self.prop_func(df)
        if self.bound_type == 'lower':
            condition = (df[self.target] >= ant_series) & (prop_series)
        else:
            condition = (df[self.target] <= ant_series) & (prop_series)
        self.touch = int(condition.sum())
        return self.touch

    def __hash__(self):
        return hash(self.full_expr)

    def __eq__(self, other):
        if not isinstance(other, ImplicationConjecture):
            return False
        return self.full_expr == other.full_expr

    def __str__(self):
        return f"{self.full_expr} (touch: {self.touch}, complexity: {self.complexity})"

    def __repr__(self):
        return f"ImplicationConjecture({self.full_expr!r}, touch={self.touch}, complexity={self.complexity})"



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
        self.accepted_conjectures = []

    def read_csv(self, path_to_csv, drop_columns=None, standard_columns=True):
        """
        Load data from a CSV file and preprocess it.

        - Standardizes column names.
        - Ensures a 'name' column exists.
        - Warns if non-numerical/boolean columns are present.
        - Adds a default boolean column if none exists.
        """
        self.knowledge_table = pd.read_csv(path_to_csv)
        self.bad_columns = []

        if standard_columns:
            self.knowledge_table.columns = (
            self.knowledge_table.columns
            .str.strip()
            .str.lower()
            .str.replace(r'\W+', '_', regex=True)
        )

        # Ensure a 'name' column exists.
        if 'name' not in self.knowledge_table.columns:
            n = len(self.knowledge_table)
            self.knowledge_table['name'] = [f'O{i+1}' for i in range(n)]

        # Add a default boolean column if none exist.
        self.numerical_columns = self.knowledge_table.select_dtypes(include=['number']).columns.tolist()
        self.original_numerical_columns = self.knowledge_table.select_dtypes(include=['number']).columns.tolist()
        self.boolean_columns = self.knowledge_table.select_dtypes(include='bool').columns.tolist()

        if not self.boolean_columns:
            self.knowledge_table['object'] = True

        self.global_type = self.get_global_true_column()
        self.update_invariant_knowledge()

        if drop_columns:
            self.drop_columns(drop_columns)

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
        self.update_invariant_knowledge()

    def update_invariant_knowledge(self):
        """
        Update internal records of which columns are numerical or boolean and track any bad columns.
        """
        self.numerical_columns = self.knowledge_table.select_dtypes(include=['number']).columns.tolist()
        self.boolean_columns = self.knowledge_table.select_dtypes(include='bool').columns.tolist()

    def drop_columns(self, columns):
        """
        Drop the specified columns from the knowledge_table.

        Args:
            columns (list): List of column names to remove.
        """
        self.knowledge_table = self.knowledge_table.drop(columns, axis=1)
        self.original_numerical_columns = self.knowledge_table.select_dtypes(include=['number']).columns.tolist()
        self.update_invariant_knowledge()

    def find_columns_of_lists(self):
        """
        Find columns that contain lists in the
        knowledge_table and return their names.
        """
        list_columns = []
        for column in self.knowledge_table.columns:
            if self.knowledge_table[column].apply(lambda x: isinstance(x, list)).any():
                list_columns.append(column)
            if is_list_string(self.knowledge_table[column]):
                list_columns.append(column)
        return list_columns

    def vectorize(self, columns, pad=False):
        """
        Convert columns that contain lists in the knowledge_table
        to arrays and return the modified DataFrame.
        """
        for column in columns:
            if pad:
                self.knowledge_table[column] = convert_and_pad(self.knowledge_table[column])
            else:
                self.knowledge_table[column] = convert_and_no_pad(self.knowledge_table[column])

        return self.knowledge_table

    def add_statistics(self, columns):
        """
        Compute statistics for the specified columns in the knowledge_table.
        """
        for column in columns:
            self.knowledge_table = expand_statistics(column, self.knowledge_table)
        return self.knowledge_table

    def set_boolean_columns(self):
        """
        Return the boolean columns in the knowledge_table.
        """
        self.boolean_columns = self.knowledge_table.select_dtypes(include='bool').columns.tolist()
        return self.boolean_columns

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

    def propose_conjecture(self, new_conjecture):
        """
        Propose a new BoundConjecture and integrate it into the existing set.

        The method works as follows:
        1. Retrieves the current conjectures for the new conjecture's target invariant and bound type.
        2. Merges the new conjecture with the existing ones.
        3. Applies the Morgan heuristic to keep only the most general conjectures.
        4. Determines whether the new conjecture was accepted (i.e. it survives filtering).
        5. Returns a tuple (accepted, message) that informs the user of the outcome and details
            any conjectures that were filtered out.

        Note: The user is expected to supply a properly formed BoundConjecture.
            You might consider a helper function to build one from user inputs.
        """
        target = new_conjecture.target
        bound_type = new_conjecture.bound_type  # expected to be 'upper' or 'lower'

        # Retrieve current conjectures for the target invariant and bound type, if they exist.
        current_conjectures = []
        if target in self.conjectures and bound_type in self.conjectures[target]:
            current_conjectures = self.conjectures[target][bound_type]

        # Merge the new conjecture with the existing ones.
        merged_conjectures = current_conjectures + [new_conjecture]

        # Apply the Morgan heuristic to filter to the most general conjectures.
        filtered_conjectures = self.morgan_heuristic(merged_conjectures)

        # Check if the new conjecture is among the accepted (most general) ones.
        if new_conjecture in filtered_conjectures:
            accepted = True
            # Determine which previously stored conjectures were filtered out.
            filtered_out = [c for c in current_conjectures if c not in filtered_conjectures]
            message = (f"New conjecture accepted. "
                    f"The following {len(filtered_out)} conjecture(s) were filtered out: {filtered_out}")
        else:
            accepted = False
            message = "New conjecture rejected: it is less general than an existing conjecture."

        # Update the stored conjectures for this target invariant.
        if target not in self.conjectures:
            self.conjectures[target] = {}
        self.conjectures[target][bound_type] = filtered_conjectures

        return accepted, message

    def set_complexity(self, avoid_columns=[], max_complexity=3):
        """
        Generate new columns of increased complexity from numerical invariants.

        For each base invariant (numerical column not in avoid_columns), create:
          - Complexity 1:
              inv^2, max(1, inv), floor(inv), ceil(inv),
              sqrt(inv) [if all values >= 0],
              log(inv) [if all values > 0].
          - Complexity 2:
              For every pair (inv1, inv2) from the current set (base + complexity 1):
                  min(inv1, inv2), max(inv1, inv2),
                  inv1*inv2,
                  inv1/inv2 [if inv2 is never 0],
                  sqrt(inv1 + inv2) [if (inv1+inv2) is nonnegative],
                  (inv1 + inv2)^2.
              (For division, we use each ordered pair so that the denominator role is explicit.)
          - Complexity 3 (if max_complexity == 3):
              For every triple (inv1, inv2, inv3) from the current set (including complexity 1 and 2):
                  (inv1 + inv2)/inv3 [if inv3 is never 0],
                  (inv1 + inv2)*inv3,
                  (inv1 - inv2)/inv3 [if inv3 is never 0].

        New columns are added to self.knowledge_table.
        """

        # Base invariants: numerical columns not in avoid_columns.
        base_invariants = [col for col in self.numerical_columns if col not in avoid_columns]
        current_invariants = base_invariants.copy()

        # For ease of notation.
        df = self.knowledge_table

        # ---- Complexity 1 ----
        new_columns = {}
        for col in tqdm(base_invariants, desc="Processing Complexity 1 features"):
            col_data = df[col]
            # Square: inv^2
            new_name = f"({col})^2"
            new_columns[new_name] = col_data ** 2

            # max(1, inv)
            new_name = f"max(1, {col})"
            new_columns[new_name] = np.maximum(1, col_data)

            # floor(inv)
            new_name = f"floor({col})"
            new_columns[new_name] = np.floor(col_data)

            # ceil(inv)
            new_name = f"ceil({col})"
            new_columns[new_name] = np.ceil(col_data)

            # sqrt(inv) if defined (all entries nonnegative)
            if (col_data >= 0).all():
                new_name = f"sqrt({col})"
                new_columns[new_name] = np.sqrt(col_data)

            # log(inv) if defined (all entries > 0)
            if (col_data > 0).all():
                new_name = f"log({col})"
                new_columns[new_name] = np.log(col_data)

        # Add all complexity 1 columns to our invariant set.
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

        # ---- Complexity 2 ----
        new_cols = {}
        # For symmetric operations, use unordered pairs.
        combo_list = list(combinations(current_invariants, 2))
        for col1, col2 in tqdm(combo_list, desc="Processing Complexity 2 pairs"):
            # min(inv1, inv2)
            new_name = f"min({col1}, {col2})"
            new_cols[new_name] = np.minimum(df[col1], df[col2])

            # max(inv1, inv2)
            new_name = f"max({col1}, {col2})"
            new_cols[new_name] = np.maximum(df[col1], df[col2])

            # Product: inv1 * inv2
            new_name = f"({col1} * {col2})"
            new_cols[new_name] = df[col1] * df[col2]

            # sqrt(inv1 + inv2) if (inv1 + inv2) is nonnegative on all rows.
            if ((df[col1] + df[col2]) >= 0).all():
                new_name = f"sqrt({col1} + {col2})"
                new_cols[new_name] = np.sqrt(df[col1] + df[col2])

            # (inv1 + inv2)^2
            new_name = f"({col1}+ {col2})^2"
            new_cols[new_name] = (df[col1] + df[col2]) ** 2

        perm_total = len(current_invariants) * (len(current_invariants) - 1)
        for col1, col2 in tqdm(permutations(current_invariants, 2), total=perm_total, desc="Processing Complexity 2 divisions"):
            if (df[col2] != 0).all():
                new_name = f"({col1} / {col2})"
                new_cols[new_name] = df[col1] / df[col2]

        # Update invariant set with complexity 2 columns.
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

        # ---- Complexity 3 (optional) ----
        if max_complexity == 3:
            new_cols = {}
            triple_list = list(combinations(current_invariants, 3))
            for col1, col2, col3 in tqdm(triple_list, desc="Processing Complexity 3 triples"):
                # (inv1 + inv2)/inv3 and (inv1 - inv2)/inv3 require inv3 to be safe.
                if (df[col3] != 0).all():
                    new_name = f"[({col1} + {col2}) / {col3}]"
                    new_cols[new_name] = (df[col1] + df[col2]) / df[col3]

                    new_name = f"[({col1} - {col2}) / {col3}]"
                    new_cols[new_name] = (df[col1] - df[col2]) / df[col3]
                # (inv1 + inv2)*inv3 (no division, so no check needed)
                new_name = f"[({col1} + {col2}) * {col3}]"
                new_cols[new_name] = (df[col1] + df[col2]) * df[col3]

            # Update invariant set with complexity 3 columns.
            df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

        # Finally, update the invariant knowledge in case new columns need to be tracked.
        self.knowledge_table = df
        self.update_invariant_knowledge()

    def get_global_true_column(self):
        """
        Returns the first boolean column in the knowledge_table that is True for every row.
        If no such column exists, returns None.
        """
        for col in self.knowledge_table.columns:
            if pd.api.types.is_bool_dtype(self.knowledge_table[col]) and self.knowledge_table[col].all():
                return col
        return None
