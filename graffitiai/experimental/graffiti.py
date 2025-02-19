# graffiti.py
import pandas as pd
import re
import warnings
from graffitiai.experimental.legacy_graffiti import LegacyGraffiti
from graffitiai.base import BaseConjecturer, BoundConjecture # assuming BaseConjecturer remains similar
# from graffitiai.utils import convert_conjecture_dicts


def convert_conjecture_dicts(conjecture_reps, target, hypothesis=None, default_bound_type='lower'):
    """
    Convert conjecture representations into a list of BoundConjecture objects.

    Parameters:
        conjecture_reps (dict or list): Either a dictionary whose keys are bound types (e.g., 'lower')
            and values are lists of conjecture dictionaries, or a list of conjecture dictionaries.
        target (str): The target column (e.g., 'radius').
        hypothesis (str, optional): An optional hypothesis (e.g., a boolean column name).
        default_bound_type (str): If conjecture_reps is a list, this bound type will be assigned to all entries.

    Returns:
        List[BoundConjecture]: A list of BoundConjecture objects created from the representations.
    """
    bound_conjectures = []

    if isinstance(conjecture_reps, dict):
        for outer_bound_type, conj_list in conjecture_reps.items():
            for conj in conj_list:
                candidate_expr = conj.get('rhs_str')
                candidate_func = conj.get('func')
                complexity = conj.get('complexity')
                touch = conj.get('touch', None)
                # Use the candidate's own bound_type if available; otherwise fall back to the outer key.
                candidate_bound_type = conj.get('bound_type', outer_bound_type)

                bc = BoundConjecture(
                    target=target,
                    candidate_expr=candidate_expr,
                    candidate_func=candidate_func,
                    bound_type=candidate_bound_type,
                    hypothesis=hypothesis,
                    complexity=complexity
                )
                bc.touch = touch
                bound_conjectures.append(bc)
    elif isinstance(conjecture_reps, list):
        # Assume all entries are of the default bound type unless they provide one.
        for conj in conjecture_reps:
            candidate_expr = conj.get('rhs_str')
            candidate_func = conj.get('func')
            complexity = conj.get('complexity')
            touch = conj.get('touch', None)
            candidate_bound_type = conj.get('bound_type', default_bound_type)

            bc = BoundConjecture(
                target=target,
                candidate_expr=candidate_expr,
                candidate_func=candidate_func,
                bound_type=candidate_bound_type,
                hypothesis=hypothesis,
                complexity=complexity
            )
            bc.touch = touch
            bound_conjectures.append(bc)
    else:
        raise ValueError("conjecture_reps must be a dictionary or a list")

    return bound_conjectures


class Graffiti(BaseConjecturer):
    def __init__(self, knowledge_table=None):
        super().__init__(knowledge_table)

    def conjecture(self, target, hypothesis=None, filter_property=None, time_limit_minutes=1):
        # In your main Graffiti.conjecture method:
        lower_search = LegacyGraffiti(self.knowledge_table, target, bound_type='lower',
                                    filter_property=filter_property, time_limit=time_limit_minutes*60)
        upper_search = LegacyGraffiti(self.knowledge_table, target, bound_type='upper',
                                    filter_property=filter_property, time_limit=time_limit_minutes*60)
        lower_search.search()
        upper_search.search()

        lower_conjectures = convert_conjecture_dicts(lower_search.accepted_conjectures, target, hypothesis=hypothesis)
        upper_conjectures = convert_conjecture_dicts(upper_search.accepted_conjectures, target, hypothesis=hypothesis)

        self.conjectures[target] = {
            'lower': lower_conjectures,
            'upper': upper_conjectures
}

    def write_on_the_wall(self, target=None):
        for bound_type in ['upper', 'lower']:
            if target is not None:
                if target not in self.conjectures:
                    print(f"No conjectures available for target: {target}")
                    return
                conj_list = self.conjectures[target].get(bound_type, [])
                if not conj_list:
                    print(f"No {bound_type} conjectures for target: {target}")
                    return
                # Use attribute access instead of dictionary indexing
                sorted_conjectures = sorted(conj_list, key=lambda c: c.touch, reverse=True)
                print()
                print(f"GRAFFITI conjectures for {target} ({bound_type}):")
                print("------------------------")
                for conj in sorted_conjectures:
                    print(f"Conjecture: {conj.full_expr} (touch: {conj.touch})")
            else:
                for tgt, bound_dict in self.conjectures.items():
                    conj_list = bound_dict.get(bound_type, [])
                    if not conj_list:
                        continue
                    sorted_conjectures = sorted(conj_list, key=lambda c: c.touch, reverse=True)
                    print(f"GRAFFITI conjectures for {tgt} ({bound_type}):")
                    print("------------------------")
                    for conj in sorted_conjectures:
                        print(f"Conjecture: {conj.full_expr} (touch: {conj.touch})")

