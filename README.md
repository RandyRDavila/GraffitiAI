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

## TxGraffitiAI
```python
from graffitiai import GraffitiAI

# Point towards the URL hosted on Jillian's GitHub
url = 'https://raw.githubusercontent.com/jeddyhub/Polytope_Database/refs/heads/main/Simple_Polytope_Data/simple_polytope_properties.csv'

# Create an instance of the GraffitiAI class
ai = GraffitiAI()

# Read the data from the URL
ai.read_csv(url)

# Vectorize the p-vector column
ai.vectorize(['p_vector'])

# Define small face counts
ai.knowledge_table["p_3"] = ai.knowledge_table["p_vector"].apply(lambda x: x[0] if len(x) > 2 else 0)
ai.knowledge_table["p_4"] = ai.knowledge_table["p_vector"].apply(lambda x: x[1] if len(x) > 2 else 0)
ai.knowledge_table["p_5"] = ai.knowledge_table["p_vector"].apply(lambda x: x[2] if len(x) > 2 else 0)
ai.knowledge_table["p_6"] = ai.knowledge_table["p_vector"].apply(lambda x: x[3] if len(x) > 3 else 0)
ai.knowledge_table["p_7"] = ai.knowledge_table["p_vector"].apply(lambda x: x[4] if len(x) > 4 else 0)
ai.knowledge_table['sum(p_vector)'] = ai.knowledge_table['p_vector'].apply(sum)
ai.knowledge_table['sum(p_vector not p_6)'] = ai.knowledge_table['p_vector'].apply(lambda x: sum([i for i in x if i != 6]))
ai.knowledge_table['sum(p_vector) with p >= 7'] = ai.knowledge_table['p_vector'].apply(lambda x: sum([i for i in x if i >= 7]))

ai.update_invariant_knowledge()

# Optionally add statistics on the vector valued column
ai.add_statistics(['p_vector'])

# Drop the columns that are not needed
ai.drop_columns([
    'edgelist',
    'adjacency_matrix',
    'p_vector',
])

# Optionally increase the complexity of the types of conjectures applied
ai.set_complexity( max_complexity=1)

# Generate conjectures on a list of target properties (invariants)
ai.conjecture(
    target_invariants=[
        'sum(p_vector)',
        'p_6',
    ],
    hypothesis=[
      'simple_polytope_graph',
      'simple_polytope_graph_with_p6_greater_than_zero',
   ],
   other_invariants=[
        'p_3',
        'p_4',
        'p_5',
        'p_7',
        'order',
        'size',
        'sum(p_vector)',
        'size',
        'sum(p_vector)',
        'p_6',
        'median_absolute_deviation(p_vector)',
        'max(p_vector)',
        'independence_number',

   ],
    complexity_range=(1, 3),
    lower_b_max=2,
    lower_b_min=-2,
    upper_b_max=3,
    upper_b_min=-3,
    W_lower_bound=None,
    W_upper_bound=None,
    min_touch=1,
)

ai.write_on_the_wall(search=True)
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

