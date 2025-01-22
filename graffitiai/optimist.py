
import pandas as pd
import warnings
import re

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from importlib.resources import files

from .optimization import make_all_linear_conjectures
from .heuristics import hazel_heuristic, morgan_heuristic, weak_smokey


class Optimist:
    """
    The Optimist class is designed for generating mathematical conjectures
    based on properties of graphs or other mathematical objects. It allows users
    to load data, identify possible invariants and hypotheses, generate conjectures,
    and save results.

    Attributes:
        knowledge_table (pd.DataFrame): The data table containing graph or object properties.
        conjectures (dict): A dictionary to store generated conjectures.

    Example:
        >>> from graffitiai import Optimist
        >>> optimist = Optimist()
        >>> optimist.load_sample_3_regular_polytope_data() # sample dataset included in the package
        >>> optimist.describe_invariants_and_hypotheses()
        >>> optimist.conjecture(
        ...     target_invariant='independence_number',
        ...     other_invariants=['n', 'matching_number'],
        ...     hypothesis='cubic_polytope',
        ...     complexity=2,
        ...     show=True
        ... )
        >>> optimist.save_conjectures_to_pdf("my_conjectures.pdf")
    """
    def __init__(self, knowledge_table=None):
        self.knowledge_table = knowledge_table
        self.conjectures = {}

    def load_sample_3_regular_polytope_data(self):
        """Load the sample 3-regular polytope dataset."""
        dataset_path = files("graffitiai.data").joinpath("sample_3_regular_polytope_data.csv")
        self.knowledge_table = pd.read_csv(dataset_path)
        print("Sample 3-regular polytope dataset loaded successfully.")

    def read_csv(self, path_to_csv):
        """
        Load data from a CSV file and preprocess it for conjecturing.

        Standardizes column names, ensures a 'name' column exists, and adds a
        default boolean column if no boolean columns are present.

        Args:
            path_to_csv (str): The path to the CSV file.

        Example:
            >>> optimist = Optimist()
            >>> optimist.read_csv("data/graph_properties.csv")
            Standardized column names:
              'Vertex Count' -> 'vertex_count'
              'Average Degree' -> 'average_degree'
            'name' column missing. Created default names: O1, O2, ..., On.
        """
        # Read the CSV file
        self.knowledge_table = pd.read_csv(path_to_csv)

        # Standardize column names
        original_columns = self.knowledge_table.columns
        self.knowledge_table.columns = [
            re.sub(r'\W+', '_', col.strip().lower()) for col in original_columns
        ]
        print("Standardized column names:")
        for original, new in zip(original_columns, self.knowledge_table.columns):
            if original != new:
                print(f"  '{original}' -> '{new}'")

        # Check for 'name' column and create it if missing
        if 'name' not in self.knowledge_table.columns:
            n = len(self.knowledge_table)
            self.knowledge_table['name'] = [f'O{i+1}' for i in range(n)]
            print(f"'name' column missing. Created default names: O1, O2, ..., O{n}.")

        # Warn for non-numerical or non-boolean entries
        for column in self.knowledge_table.columns:
            if column == 'name':  # Skip the 'name' column
                continue
            if not pd.api.types.is_numeric_dtype(self.knowledge_table[column]) and \
               not pd.api.types.is_bool_dtype(self.knowledge_table[column]):
                warnings.warn(f"Column '{column}' contains non-numerical and non-boolean entries.")

        # Add a default boolean column if no boolean columns exist
        boolean_columns = [
            col for col in self.knowledge_table.columns
            if pd.api.types.is_bool_dtype(self.knowledge_table[col])
        ]
        if not boolean_columns:
            self.knowledge_table['object'] = True
            print("No boolean columns found. Added default column 'object' with all values set to True.")


    def add_row(self, row_data):
        """
        Add a new row of data to the knowledge_table.

        Args:
            row_data (dict): A dictionary where keys are column names and values are the corresponding data.

        Raises:
            ValueError: If `knowledge_table` is not initialized or if `row_data` has keys not in the current columns.

        Example:
            >>> optimist = Optimist()
            >>> optimist.load_sample_3_regular_polytope_data()
            >>> new_row = {
            ...     "name": "new_object",
            ...     "n": 12,
            ...     "matching_number": 6,
            ...     "independence_number": 5,
            ...     "cubic_polytope": True,
            ...     "average_shortest_path_length": 2.1,
            ... }
            >>> optimist.add_row(new_row)
        """
        if self.knowledge_table is None:
            raise ValueError("Knowledge table is not initialized. Load or create a dataset first.")

        # Check for unexpected keys
        unexpected_keys = [key for key in row_data.keys() if key not in self.knowledge_table.columns]
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in row_data: {unexpected_keys}. Allowed columns: {list(self.knowledge_table.columns)}")

        # Fill in missing columns with defaults
        complete_row = {col: row_data.get(col, None) for col in self.knowledge_table.columns}

        # Append the new row
        self.knowledge_table = pd.concat(
            [self.knowledge_table, pd.DataFrame([complete_row])],
            ignore_index=True
        )
        print(f"Row added successfully: {complete_row}")


    def get_possible_invariants(self):
        """
        Identify numerical columns suitable for conjectures.

        Returns:
            list: A list of column names that have numerical data, excluding boolean columns.

        Example:
            >>> optimist.get_possible_invariants()
            ['n', 'matching_number', 'independence_number', ...]
        """
        if self.knowledge_table is None:
            raise ValueError("Knowledge table is not loaded. Please load data first.")

        # Identify numerical columns, excluding boolean columns
        numerical_columns = [
            col for col in self.knowledge_table.columns
            if pd.api.types.is_numeric_dtype(self.knowledge_table[col]) and
            not pd.api.types.is_bool_dtype(self.knowledge_table[col])
        ]
        return numerical_columns

    def get_possible_hypotheses(self):
        """
        Identify boolean columns suitable for hypotheses.

        Returns:
            list: A list of column names that have boolean data.

        Example:
            >>> optimist.get_possible_hypotheses()
            ['cubic_polytope', 'object']
        """
        if self.knowledge_table is None:
            raise ValueError("Knowledge table is not loaded. Please load data first.")

        boolean_columns = [
            col for col in self.knowledge_table.columns
            if pd.api.types.is_bool_dtype(self.knowledge_table[col])
        ]
        return boolean_columns

    def describe_invariants_and_hypotheses(self):
        """
        Print the possible numerical invariants and boolean hypotheses.

        This method summarizes the columns in the knowledge table that can
        be used for conjecturing or as hypotheses.

        Example:
            >>> optimist.describe_invariants_and_hypotheses()
            Possible Numerical Invariants (for conjecturing):
              - n
              - matching_number
              - independence_number
              ...
            Possible Boolean Hypotheses:
              - cubic_polytope
              - object
        """
        print("Possible Numerical Invariants (for conjecturing):")
        for col in self.get_possible_invariants():
            print(f"  - {col}")

        print("\nPossible Boolean Hypotheses:")
        for col in self.get_possible_hypotheses():
            print(f"  - {col}")

    def apply_heuristics(self, conjectures, min_touch=0, use_morgan=True, use_smokey=True):
        """
        Apply heuristics to refine a list of conjectures.

        Args:
            conjectures (list): A list of conjectures to refine.
            min_touch (int): Minimum number of instances of equality to keep a conjecture.
            use_morgan (bool): Whether to apply the Morgan heuristic.
            use_smokey (bool): Whether to apply the Smokey heuristic.

        Returns:
            list: A list of refined conjectures.

        Example:
            >>> refined_conjectures = optimist.apply_heuristics(conjectures, min_touch=5)
        """
        if not conjectures:
            return []

        conjectures = hazel_heuristic(conjectures, min_touch=min_touch)
        if use_morgan:
            conjectures = morgan_heuristic(conjectures)
        if use_smokey:
            conjectures = weak_smokey(conjectures)
        return conjectures

    def conjecture(
            self,
            target_invariant,
            other_invariants,
            hypothesis,
            complexity=2,
            show=False,
            min_touch=0,
            use_morgan=True,
            use_smokey=True
    ):
        """
        Generate conjectures for a specified target invariant.

        Args:
            target_invariant (str): The column to conjecture bounds for.
            other_invariants (list): Columns to use in forming conjectures.
            hypothesis (str): The boolean column representing the hypothesis.
            complexity (int): Maximum complexity of the conjectures.
            show (bool): Whether to display conjectures in the console.
            min_touch (int): Minimum touch number for conjectures.
            use_morgan (bool): Whether to apply the Morgan heuristic.
            use_smokey (bool): Whether to apply the Smokey heuristic.

        Example:
            >>> optimist.conjecture(
            ...     target_invariant='independence_number',
            ...     other_invariants=['n', 'matching_number'],
            ...     hypothesis='cubic_polytope',
            ...     complexity=2,
            ...     show=True
            ... )
        """
        upper_conjectures, lower_conjectures = make_all_linear_conjectures(
            self.knowledge_table, target_invariant, other_invariants, hypothesis, complexity=complexity
        )

        # Apply heuristics
        upper_conjectures = self.apply_heuristics(upper_conjectures, min_touch, use_morgan, use_smokey)
        lower_conjectures = self.apply_heuristics(lower_conjectures, min_touch, use_morgan, use_smokey)

        if show:
            print("Upper Conjectures:")
            for i, conj in enumerate(upper_conjectures):
                print(f"  {i+1}. {conj} (Equality: {conj.touch} times)")
            print("\nLower Conjectures:")
            for i, conj in enumerate(lower_conjectures):
                print(f"  {i+1}. {conj} (Equality: {conj.touch} times)")

        self.conjectures[target_invariant] = {
            "upper": upper_conjectures,
            "lower": lower_conjectures
        }

    def write_on_the_wall(self, target_invariants=None):
        """
        Display generated conjectures for specified target invariants.

        Args:
            target_invariants (list, optional): List of target invariants to display. If None,
                displays conjectures for all invariants.

        Example:
            >>> optimist.write_on_the_wall(target_invariants=['independence_number'])
        """
        upper_conjectures = []
        lower_conjectures = []

        # Gather conjectures
        if target_invariants is not None:
            for target_invariant in target_invariants:
                upper_conjectures += self.conjectures.get(target_invariant, {}).get("upper", [])
                lower_conjectures += self.conjectures.get(target_invariant, {}).get("lower", [])
        else:
            for target_invariant, results in self.conjectures.items():
                upper_conjectures += results["upper"]
                lower_conjectures += results["lower"]

        # Apply heuristics if there are multiple conjectures
        if len(target_invariants or self.conjectures.keys()) > 1:
            upper_conjectures = self.apply_heuristics(upper_conjectures)
            lower_conjectures = self.apply_heuristics(lower_conjectures)

        # Format output
        def format_conjectures(conjectures, title):
            if not conjectures:
                print(f"{title}:\n  None\n")
                return
            print(f"{title}:")
            for i, conj in enumerate(conjectures):
                print(f"  {i+1}. {conj} (Equality: {conj.touch} times)")
            print("-" * 50)

        format_conjectures(upper_conjectures, "Upper Conjectures")
        format_conjectures(lower_conjectures, "Lower Conjectures")

    def save_conjectures_to_pdf(self, file_name="conjectures.pdf", target_invariants=None):
        """
        Save conjectures to a PDF file with the date and time.

        Args:
            file_name (str): Name of the output PDF file.
            target_invariants (list, optional): List of target invariants to include. If None,
                includes all conjectures.

        Example:
            >>> optimist.save_conjectures_to_pdf("my_conjectures.pdf")
        """
        # Prepare the conjectures
        upper_conjectures = []
        lower_conjectures = []

        if target_invariants is not None:
            for target_invariant in target_invariants:
                upper_conjectures += self.conjectures.get(target_invariant, {}).get("upper", [])
                lower_conjectures += self.conjectures.get(target_invariant, {}).get("lower", [])
        else:
            for target_invariant, results in self.conjectures.items():
                upper_conjectures += results["upper"]
                lower_conjectures += results["lower"]

        # Create the PDF
        pdf = canvas.Canvas(file_name, pagesize=letter)
        width, height = letter

        # Add title and date/time
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, height - 50, "Generated Conjectures")
        pdf.setFont("Helvetica", 10)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.drawString(50, height - 70, f"Generated on: {timestamp}")

        # Add conjectures
        y_position = height - 100
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_position, "Upper Conjectures:")
        y_position -= 20
        pdf.setFont("Helvetica", 10)
        if upper_conjectures:
            for i, conj in enumerate(upper_conjectures, start=1):
                if y_position < 50:  # Start a new page if we run out of space
                    pdf.showPage()
                    y_position = height - 50
                pdf.drawString(50, y_position, f"{i}. {conj} (Equality: {conj.touch} times)")
                y_position -= 20
        else:
            pdf.drawString(50, y_position, "None")
            y_position -= 20

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_position, "Lower Conjectures:")
        y_position -= 20
        pdf.setFont("Helvetica", 10)
        if lower_conjectures:
            for i, conj in enumerate(lower_conjectures, start=1):
                if y_position < 50:  # Start a new page if we run out of space
                    pdf.showPage()
                    y_position = height - 50
                pdf.drawString(50, y_position, f"{i}. {conj} (Equality: {conj.touch} times)")
                y_position -= 20
        else:
            pdf.drawString(50, y_position, "None")

        # Save the PDF
        pdf.save()
        print(f"Conjectures saved to {file_name}")
