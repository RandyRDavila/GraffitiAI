import pandas as pd
import re
import warnings

# These functions are assumed to be defined elsewhere in your package.
# from graffitiai.heuristics import hazel_heuristic, morgan_heuristic, weak_smokey
# from graffitiai.conjectures import make_all_linear_conjectures_range

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
    def __init__(self, target, candidate_expr, candidate_func, bound_type='lower',
                 hypothesis=None, complexity=None):
        self.target = target
        self.candidate_expr = candidate_expr
        self.candidate_func = candidate_func
        self.bound_type = bound_type
        self.hypothesis = hypothesis
        self.complexity = complexity
        self.touch = None  # Will be computed later (e.g. with compute_touch)
        self.full_expr = self.format_full_expression()
        self.conclusion = self._set_conclusion()

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
            else:
                return f"For any {self.hypothesis}, {self.target} ≤ {simplified_expr}"
        else:
            if self.bound_type == 'lower':
                return f"{self.target} ≥ {simplified_expr}"
            else:
                return f"{self.target} ≤ {simplified_expr}"

    def _set_conclusion(self):
        # This method is a placeholder. In a real implementation, this might evaluate the conclusion.
        if self.bound_type == 'lower':
            return f"{self.target} ≥ {self.candidate_expr}"
        else:
            return f"{self.target} ≤ {self.candidate_expr}"

    def evaluate(self, df):
        """Evaluate the candidate function on the given DataFrame."""
        return self.candidate_func(df)

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

    def __hash__(self):
        # Use the full expression (which captures target, candidate_expr, hypothesis, etc.)
        return hash(self.full_expr)

    def __eq__(self, other):
        if not isinstance(other, BoundConjecture):
            return False
        return self.full_expr == other.full_expr

    def __str__(self):
        return f"{self.full_expr} (touch: {self.touch}, complexity: {self.complexity})"

    def __repr__(self):
        return f"BoundConjecture({self.full_expr!r}, touch={self.touch}, complexity={self.complexity})"


class ImplicationConjecture:
    """
    Represents an implication-based conjecture of the form:

      If target {≥ or ≤} antecedent_expr, then property_expr holds.

    For example:

      If number_of_vertices ≥ 1*(index) + 2/3*(number_of_edges) - 3, then some property holds,
      or more specifically:

      If sum_p_gons_with_p>6 ≥ radius, then number_of_6_gons = 0.

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
        self.prop_func = prop_func
        self.bound_type = bound_type
        self.hypothesis = hypothesis
        self.complexity = complexity
        self.support = support
        self.touch = None  # to be computed later (e.g. with compute_touch)
        self.full_expr = self.format_full_expression()

    def format_full_expression(self):
        """Format the full human-readable expression for the conjecture."""
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

    def read_csv(self, path_to_csv):
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
