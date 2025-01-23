# GraffitiAI

GraffitiAI is a Python package for automated mathematical conjecturing, inspired by the legacy of GRAFFITI. It provides tools for exploring relationships between mathematical invariants and properties, with a focus on graph theory and polytopes. This package supports generating conjectures, applying heuristics, and visualizing results.

## Features
- Load and preprocess datasets with ease.
- Identify possible invariants and hypotheses for conjecturing.
- Generate upper and lower bounds for a target invariant.
- Apply customizable heuristics to refine conjectures.
- Export results to PDF for presentation and sharing.
- Includes a sample dataset of 3-regular polytopes for experimentation.

---

## Installation

To install GraffitiAI, use `pip`:

```bash
# Install GraffitiAI with pip
pip install graffitiai
```

---

## Quick Start

Here's a simple example to get you started:

### Using the Built-in Dataset
```python
from graffitiai import Optimist

# Initialize the Optimist instance
optimist = Optimist()

# Load the sample dataset
optimist.load_sample_3_regular_polytope_data()

# Describe available invariants and hypotheses
optimist.describe_invariants_and_hypotheses()

# Generate conjectures
optimist.conjecture(
    target_invariant='independence_number',
    other_invariants=['n', 'matching_number'],
    hypothesis=['cubic_polytope'],
    complexity=2,
    show=True
)

# Save conjectures to a PDF
optimist.save_conjectures_to_pdf("conjectures.pdf")
```

### Using a Custom CSV File
```python
from graffitiai import Optimist

# Initialize the Optimist instance
optimist = Optimist()

# Load a custom dataset
optimist.read_csv("path_to_your_data.csv")

# Describe available invariants and hypotheses
optimist.describe_invariants_and_hypotheses()

# Generate conjectures
optimist.conjecture(
    target_invariant='your_target_invariant',
    other_invariants=['invariant1', 'invariant2'],
    hypothesis=['your_hypothesis_column'],
    complexity=2,
    show=True
)

# Save conjectures to a PDF
optimist.save_conjectures_to_pdf("custom_conjectures.pdf")
```

---

## API Reference

### `class Optimist`

#### Methods

1. **`load_sample_3_regular_polytope_data()`**
   Loads the included sample dataset of 3-regular polytopes into the knowledge table.

2. **`read_csv(path_to_csv)`**
   Loads a dataset from a CSV file, standardizing column names and ensuring compatibility for conjecturing.

3. **`get_possible_invariants()`**
   Returns a list of numerical columns suitable for conjecturing.

4. **`get_possible_hypotheses()`**
   Returns a list of boolean columns suitable for hypotheses.

5. **`describe_invariants_and_hypotheses()`**
   Prints the possible invariants and hypotheses from the knowledge table.

6. **`conjecture(target_invariant, other_invariants, hypothesis, complexity=2, show=False, min_touch=0, use_morgan=True, use_smokey=True)`**
   Generates conjectures for a given target invariant using the specified invariants and hypotheses.

7. **`write_on_the_wall(target_invariants=None)`**
   Displays generated conjectures for specified or all target invariants.

8. **`save_conjectures_to_pdf(file_name="conjectures.pdf", target_invariants=None)`**
   Exports generated conjectures to a PDF file, including the date and time of generation.

---

## Sample Dataset

GraffitiAI includes a sample dataset of 3-regular polytopes. It contains properties such as independence number, domination number, and average shortest path length, among others. This dataset is ideal for experimenting with the package's features.

---

## Contributing

Contributions are welcome! If you have suggestions, find bugs, or want to add features, feel free to create an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

GraffitiAI is inspired by the pioneering work of GRAFFITI and built using the ideas of *TxGraffiti*.

### Author

Randy R. Davila, PhD

