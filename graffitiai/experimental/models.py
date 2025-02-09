import numpy as np
import pandas as pd

from graffitiai import TxGraffiti
from graffitiai.experimental.expression_generator import generate_expressions, simplify

__all__ = [
    "GraffitiII",
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
