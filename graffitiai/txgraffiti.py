from itertools import combinations
from tqdm import tqdm  # Import tqdm as a function for progress bars

from graffitiai.utils import make_all_linear_conjectures_range, filter_false_conjectures
from graffitiai.utils import hazel_heuristic, morgan_heuristic, weak_smokey, strong_smokey
from graffitiai.base import BaseConjecturer


class TxGraffiti(BaseConjecturer):

    def apply_heuristics(self, conjectures, min_touch=1, use_morgan=True, use_smokey=True):
        """
        Apply heuristics to refine a list of conjectures.

        Args:
            conjectures (list): List of conjecture objects.
            min_touch (int): Minimum number of instances for a conjecture to be kept.
            use_morgan (bool): Whether to apply the Morgan heuristic.
            use_smokey (bool): Whether to apply the Smokey heuristic.

        Returns:
            list: Refined conjectures.
        """
        if not conjectures:
            return []

        with tqdm(total=3, desc="Applying heuristics", unit="step") as progress:
            conjectures = filter_false_conjectures(conjectures, self.knowledge_table)
            conjectures = hazel_heuristic(conjectures, min_touch=min_touch)
            progress.update(1)

            if use_morgan:
                conjectures = morgan_heuristic(conjectures)
                progress.update(1)

            if use_smokey:
                conjectures = strong_smokey(conjectures, self.knowledge_table)
                progress.update(1)

        return conjectures

    def conjecture(
        self,
        target_invariant=None,
        target_invariants=None,
        other_invariants=None,
        hypothesis=None,
        complexity_range=(2, 3),
        show=False,
        min_touch=0,
        use_morgan=True,
        use_smokey=True,
        lower_b_max=None,
        upper_b_max=None,
        lower_b_min=None,
        upper_b_min=None,
        W_lower_bound=-10,
        W_upper_bound=10,
    ):
        if other_invariants is None:
            other_invariants = self.numerical_columns
        if hypothesis is None:
            hypothesis = self.boolean_columns

        # Determine targets: if a single target is provided, wrap it in a list;
        # otherwise, use target_invariants or fallback to all numerical_columns.
        targets = [target_invariant] if target_invariant else (target_invariants or self.numerical_columns)

        # Compute an estimate of total iterations for progress tracking.
        total_iterations = sum(
            sum(
                len(list(combinations([inv for inv in other_invariants if inv != target], complexity))) * len(hypothesis)
                for complexity in range(complexity_range[0], complexity_range[1] + 1)
            )
            for target in targets
        )

        if total_iterations == 0:
            return

        with tqdm(total=total_iterations, desc="Generating Conjectures", leave=True) as pbar:
            for target in targets:
                upper_conjectures, lower_conjectures = make_all_linear_conjectures_range(
                    self.knowledge_table,
                    target,
                    other_invariants,
                    hypothesis,
                    complexity_range=complexity_range,
                    lower_b_max=lower_b_max,
                    upper_b_max=upper_b_max,
                    lower_b_min=lower_b_min,
                    upper_b_min=upper_b_min,
                    W_lower_bound=W_lower_bound,
                    W_upper_bound=W_upper_bound,
                    progress_bar=pbar
                )

                upper_conjectures = self.apply_heuristics(upper_conjectures, min_touch, use_morgan, use_smokey)
                lower_conjectures = self.apply_heuristics(lower_conjectures, min_touch, use_morgan, use_smokey)

                if show:
                    print(f"Upper Conjectures for {target}:")
                    for i, conj in enumerate(upper_conjectures, 1):
                        print(f"  {i}. {conj} (Equality: {conj.touch} times)")
                    print(f"\nLower Conjectures for {target}:")
                    for i, conj in enumerate(lower_conjectures, 1):
                        print(f"  {i}. {conj} (Equality: {conj.touch} times)")

                self.conjectures[target] = {
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
            >>> ai.write_on_the_wall(target_invariants=['independence_number'])
        """
        upper_conjectures = []
        lower_conjectures = []

        # Gather conjectures based on provided targets.
        if target_invariants is not None:
            for target_invariant in target_invariants:
                upper_conjectures += self.conjectures.get(target_invariant, {}).get("upper", [])
                lower_conjectures += self.conjectures.get(target_invariant, {}).get("lower", [])
        else:
            for target_invariant, results in self.conjectures.items():
                upper_conjectures += results.get("upper", [])
                lower_conjectures += results.get("lower", [])

        # If more than one target is displayed, optionally refine the conjectures.
        if (target_invariants is None and len(self.conjectures) > 1) or (target_invariants is not None and len(target_invariants) > 1):
            upper_conjectures = self.apply_heuristics(upper_conjectures)
            lower_conjectures = self.apply_heuristics(lower_conjectures)

        # Format and print the output.
        def format_conjectures(conjectures, title):
            if not conjectures:
                print(f"{title}:\n  None\n")
                return
            print(f"{title}:")
            for i, conj in enumerate(conjectures):
                print(f"  {i+1}. {conj}.")
            print("-" * 50)

        format_conjectures(upper_conjectures, "Upper Conjectures")
        format_conjectures(lower_conjectures, "Lower Conjectures")
