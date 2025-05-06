import networkx as nx
import pandas as pd
import random
import os

from graffitiai.experimental.utils import (
    make_upper_linear_conjecture,
    make_lower_linear_conjecture,
    upper_bound_acceptance,
    lower_bound_acceptance,
    hazel_heuristic,
    morgan_heuristic,
    probability_distribution,
)

__all__ = [
    "Optimist",
]

class Optimist:
    def __init__(
            self,
            objects: list,
            invariants: list,
            known_theorems=[],
            knowledge_table=None,
            read_edgelists=None,
        ):
        if read_edgelists:
            self._read_edgelists(read_edgelists)
        else:
            self.objects = objects
        self.invariants = invariants
        self.known_theorems = known_theorems
        if knowledge_table is None:
            self._build_knowledge()
        else:
            self.knowledge_table = knowledge_table

        self._set_all_conjectures()

    def _build_knowledge(self):
        rows = []
        for i, O in enumerate(self.objects):
            row = {"name": f"object{i}"}
            for function in self.invariants:
                function_name = getattr(function, "__name__", repr(function))
                row[function_name] = function(O)
            rows.append(row)
        self.knowledge_table = pd.DataFrame(rows)
        # find all numerical columns in the knowledge_table
        self.numerical_columns = self.knowledge_table.select_dtypes(include=['number']).columns
        # get numerical functions
        self.numerical_functions = {
            getattr(function, "__name__", repr(function)): function for function in self.invariants if function.__name__ in self.numerical_columns
        }

        # find all boolean columns in the knowledge table
        self.boolean_columns = self.knowledge_table.select_dtypes(include=['bool']).columns
        # get boolean functions
        self.boolean_functions = {
            getattr(function, "__name__", repr(function)): function for function in self.invariants if function.__name__ in self.boolean_columns
        }

    def _read_edgelists(self, dirname):
        import os
        self.objects = []
        files = [f for f in os.listdir(dirname) if f.endswith(".txt")]
        for name in files:
            G = nx.read_edgelist(os.path.join(dirname, name), nodetype=int)
            self.objects.append(G)


    def accept_new_object(self, new_object):
        new_row = {"name": f"object{len(self.objects)}"}
        for function in self.invariants:
            function_name = getattr(function, "__name__", repr(function))
            new_row[function_name] = function(new_object)
        self.knowledge_table = pd.concat([self.knowledge_table, pd.DataFrame([new_row])], ignore_index=True)
        self.objects.append(new_object)
        self._set_all_conjectures()

    def _make_singles_conjecture(
            self,
            hypothesis_function,
            target_function,
            b_upper_bound=None,
            b_lower_bound=None,
            W_upper_bound=None,
            W_lower_bound=None,
        ):
        # _sample_other_invariants_on_target may sometimes raise an error,
        # in this case we randomly sample numerical functions. Do not throw and error.
        try:
            other_functions = self._sample_other_invariants_on_target(hypothesis_function, target_function)
            # other_functions = [self.numerical_functions[other] for other in other_invariants]
        except:
            other_functions = list(set([self.numerical_functions[random.choice(list(self.numerical_functions.keys()))] for _ in range(3)]))

        upper_conjecture = make_upper_linear_conjecture(
            self.knowledge_table,
            hypothesis_function,
            target_function,
            other_functions,
            b_upper_bound=b_upper_bound,
            b_lower_bound=b_lower_bound,
            W_upper_bound=W_upper_bound,
            W_lower_bound=W_lower_bound,
        )

        lower_conjecture = make_lower_linear_conjecture(
            self.knowledge_table,
            hypothesis_function,
            target_function,
            other_functions,
            b_upper_bound=b_upper_bound,
            b_lower_bound=b_lower_bound,
            W_upper_bound=W_upper_bound,
            W_lower_bound=W_lower_bound,
        )
        return upper_conjecture, lower_conjecture

    def _sample_other_invariants_on_target(self, hypothesis_function, target_function):
        others = probability_distribution(self, hypothesis_function, target_function,)
        return [self.numerical_functions[other] for other in others]

    def _make_all_conjectures_on_target(self, target_function):
        full_upper_conjectures = []
        full_lower_conjectures = []

        for hypothesis_function in self.boolean_functions:
            lower_conjectures = []
            upper_conjectures = []
            df = self.knowledge_table.copy()
            df = df[df[hypothesis_function] == True]
            hyp = self.boolean_functions[hypothesis_function]
            for _ in range(5):
                upper_conjecture, lower_conjecture = self._make_singles_conjecture(hyp, target_function)
                if upper_conjecture is not None:
                    if all(upper_bound_acceptance(upper_conjecture, conj, df,) for conj in upper_conjectures):
                        upper_conjectures.append(upper_conjecture)
                if lower_conjecture is not None:
                    if all(lower_bound_acceptance(lower_conjecture, conj, df,) for conj in lower_conjectures):
                        lower_conjectures.append(lower_conjecture)
            for _ in range(5):
                upper_conjecture, lower_conjecture = self._make_singles_conjecture(
                    hyp,
                    target_function,
                    b_upper_bound=1,
                    b_lower_bound=-1,
                    W_upper_bound=None,
                    W_lower_bound=None,
                )
                if upper_conjecture is not None:
                    if all(upper_bound_acceptance(upper_conjecture, conj, df,) for conj in upper_conjectures):
                        upper_conjectures.append(upper_conjecture)

                if lower_conjecture is not None:
                    if all(lower_bound_acceptance(lower_conjecture, conj, df,) for conj in lower_conjectures):
                        lower_conjectures.append(lower_conjecture)
            for _ in range(5):
                upper_conjecture, lower_conjecture = self._make_singles_conjecture(
                    hyp,
                    target_function,
                    b_upper_bound=2,
                    b_lower_bound=-2,
                    W_upper_bound=None,
                    W_lower_bound=None,
                )
                if upper_conjecture is not None:
                    if all(upper_bound_acceptance(upper_conjecture, conj, df,) for conj in upper_conjectures):
                        upper_conjectures.append(upper_conjecture)

                if lower_conjecture is not None:
                    if all(lower_bound_acceptance(lower_conjecture, conj, df,) for conj in lower_conjectures):
                        lower_conjectures.append(lower_conjecture)

            for _ in range(5):
                upper_conjecture, lower_conjecture = self._make_singles_conjecture(
                    hyp,
                    target_function,
                    b_upper_bound=3,
                    b_lower_bound=-3,
                    W_upper_bound=None,
                    W_lower_bound=None,
                )
                if upper_conjecture is not None:
                    if all(upper_bound_acceptance(upper_conjecture, conj, df,) for conj in upper_conjectures):
                        upper_conjectures.append(upper_conjecture)

                if lower_conjecture is not None:
                    if all(lower_bound_acceptance(lower_conjecture, conj, df,) for conj in lower_conjectures):
                        lower_conjectures.append(lower_conjecture)

            for conj in upper_conjectures:
                full_upper_conjectures.append(conj)
            for conj in lower_conjectures:
                full_lower_conjectures.append(conj)


        if full_upper_conjectures != []:
            full_upper_conjectures = hazel_heuristic(full_upper_conjectures, self.knowledge_table, min_touch=1)
        if full_upper_conjectures != []:
            full_upper_conjectures = morgan_heuristic(full_upper_conjectures)
        if full_lower_conjectures != []:
            full_lower_conjectures = hazel_heuristic(full_lower_conjectures, self.knowledge_table, min_touch=1)
        if full_upper_conjectures != []:
            full_lower_conjectures = morgan_heuristic(full_lower_conjectures)
        return full_upper_conjectures, full_lower_conjectures

    def _set_all_conjectures(self):
        upper_conjectures_dict = {}
        lower_conjectures_dict = {}
        for target_function in self.numerical_functions:
            target = self.numerical_functions[target_function]
            upper_conjectures, lower_conjectures = self._make_all_conjectures_on_target(target)
            upper_conjectures_dict[target_function] = upper_conjectures
            lower_conjectures_dict[target_function] = lower_conjectures

        self.upper_conjectures = upper_conjectures_dict
        self.lower_conjectures = lower_conjectures_dict
        all_conjectures = []
        for num_property in self.upper_conjectures:
            for conj in self.upper_conjectures[num_property]:
                all_conjectures.append(conj)
            for conj in self.lower_conjectures[num_property]:
                all_conjectures.append(conj)

        self.all_conjectures = all_conjectures
        self.all_conjectures.sort(key=lambda x: x.rank, reverse=True)

        return all_conjectures
