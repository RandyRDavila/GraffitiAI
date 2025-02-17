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

# Quick Start

Here's a simple example to get you started:

## TxGraffiti
```python
from graffitiai import TxGraffiti

# Initialize TxGraffiti
ai = TxGraffiti()

# Load a custom dataset
ai.read_csv("<path_to_your_data>.csv")

# Describe available invariants and hypotheses
ai.describe_invariants_and_hypotheses()

# Generate conjectures
ai.conjecture(
    target_invariants=[
        "zero_forcing_number",
        "total_domination_number",
    ],
    other_invariants=[
        "independence_number",
        "diameter",
        "radius",
        "domination_number"
    ],
    hypothesis=[
      "a_connected_cubic_and_diamond_free_graph",
      "a_connected_and_cubic_graph_which_is_not_k_4",
   ],
    complexity_range=(1, 3),
    lower_b_max=None,
    upper_b_max=2,
)

# Write the conjectures to the wall!
ai.write_on_the_wall()

# Save conjectures to a PDF
ai.save_conjectures_to_pdf("custom_conjectures.pdf")
```

## Graffiti
```python
from graffitiai import Graffiti

# Initialize Graffiti
ai = Graffiti()

# Read in data
ai.read_csv("https://raw.githubusercontent.com/RandyRDavila/GraffitiAI/refs/heads/main/graffitiai/data/data_437.csv")

# Drop unwanted columns
ai.drop_columns([
    "adjacency_matrix",
    "edge_list",
    "number_of_spanning_trees",
    'maximum_degree',
    'minimum_degree',
    'average_degree',
    'number_of_triangles',
    'vertex_connectivity',
    'edge_connectivity',
    'is_simple',
    'clique_number',
])
ai.drop_columns([
    f'number_of_{p}_gons' for p in range(12, 126)
])

# Conjecture lower bounds on a target invariant with a time limit set to 5 minutes
ai.conjecture('number_of_6_gons', bound_type='lower', time_limit_minutes=5)

# Write conjectures to the wall.
ai.write_on_the_wall()
```

## Christine
```python
from graffitiai import Christine

# Initialize Christine
ai =Christine()

# Read in data
ai.read_csv("https://raw.githubusercontent.com/RandyRDavila/GraffitiAI/refs/heads/main/graffitiai/data/data_437.csv")

# Drop unwanted columns
ai.drop_columns([
    "adjacency_matrix",
    "edge_list",
    "number_of_spanning_trees",
    'maximum_degree',
    'minimum_degree',
    'average_degree',
    'number_of_triangles',
    'vertex_connectivity',
    'edge_connectivity',
    'is_simple',
    'clique_number',
])
ai.drop_columns([
    f'number_of_{p}_gons' for p in range(12, 126)
])

# Conjecture on a target invariant with a time limit set to 5 minutes
ai.conjecture('number_of_6_gons', bound_type='lower', time_limit_minutes=5)

# Write conjectures to the wall.
ai.write_on_the_wall()
```

---

## Contributing

Contributions are welcome! If you have suggestions, find bugs, or want to add features, feel free to create an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

GraffitiAI is inspired by the pioneering work of GRAFFITI and built using the ideas of *TxGraffiti* and the *Optimist*.

### Author

Randy R. Davila, PhD

