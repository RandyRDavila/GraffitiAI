
def hazel_heuristic(conjectures, min_touch=0):
    """
    Filters and sorts a list of conjectures based on touch number.

    This heuristic:
    - Removes duplicate conjectures.
    - Removes conjectures that never attain equality (touch <= min_touch).
    - Sorts the remaining conjectures in descending order of touch number.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter and sort.
    min_touch : int, optional
        The minimum touch number required for a conjecture to be retained (default is 0).

    Returns
    -------
    list of Conjecture
        The sorted list of conjectures with the highest touch numbers.
    """
    # Remove duplicate conjectures.
    conjectures = list(set(conjectures))

    # Remove conjectures never attaining equality.
    conjectures = [conj for conj in conjectures if conj.touch > min_touch]

    # Sort the conjectures by touch number.
    conjectures.sort(key=lambda x: x.touch, reverse=True)

    # Return the sorted list of conjectures.
    return conjectures


def morgan_heuristic(conjectures):
    """
    Removes redundant conjectures based on generality.

    A conjecture is considered redundant if another conjecture has the same conclusion
    and a more general hypothesis (i.e., its true_object_set is a superset of the redundant one).

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.

    Returns
    -------
    list of Conjecture
        A list with redundant conjectures removed.
    """
    new_conjectures = conjectures.copy()

    for conj_one in conjectures:
        for conj_two in new_conjectures.copy():  # Make a copy for safe removal
            # Avoid comparing the conjecture with itself
            if conj_one != conj_two:
                # Check if conclusions are the same and conj_one's hypothesis is more general
                if conj_one.conclusion == conj_two.conclusion and conj_one.hypothesis > conj_two.hypothesis:
                    new_conjectures.remove(conj_two)  # Remove the less general conjecture (conj_two)

    return new_conjectures


def weak_smokey(conjectures):
    """
    Selects conjectures based on equality and distinct sharp objects.

    This heuristic:
    - Starts with the conjecture having the highest touch number.
    - Retains conjectures that either satisfy equality or introduce new sharp objects.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.

    Returns
    -------
    list of Conjecture
        A list of strong conjectures with distinct or new sharp objects.
    """
    # Start with the conjecture that has the highest touch number (first in the list).
    conj = conjectures[0]

    # Initialize the list of strong conjectures with the first conjecture.
    strong_conjectures = [conj]

    # Get the set of sharp objects (i.e., objects where the conjecture holds as equality) for the first conjecture.
    sharp_objects = conj.sharps

    # Iterate over the remaining conjectures in the list.
    for conj in conjectures[1:]:
        if conj.is_equal():
            strong_conjectures.append(conj)
            sharp_objects = sharp_objects.union(conj.sharps)
        else:
            # Check if the current conjecture shares the same sharp objects as any already selected strong conjecture.
            if any(conj.sharps.issuperset(known.sharps) for known in strong_conjectures):
                # If it does, add the current conjecture to the list of strong conjectures.
                strong_conjectures.append(conj)
                # Update the set of sharp objects to include the newly discovered sharp objects.
                sharp_objects = sharp_objects.union(conj.sharps)
            # Otherwise, check if the current conjecture introduces new sharp objects (objects where the conjecture holds).
            elif conj.sharps - sharp_objects != set():
                # If new sharp objects are found, add the conjecture to the list.
                strong_conjectures.append(conj)
                # Update the set of sharp objects to include the newly discovered sharp objects.
                sharp_objects = sharp_objects.union(conj.sharps)

    # Return the list of strong, non-redundant conjectures.
    return strong_conjectures


def strong_smokey(conjectures):
    """
    Selects conjectures that strongly subsume others based on sharp objects.

    This heuristic:
    - Starts with the conjecture having the highest touch number.
    - Retains conjectures whose sharp objects are supersets of previously selected conjectures.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.

    Returns
    -------
    list of Conjecture
        A list of conjectures with non-redundant, strongly subsuming sharp objects.
    """
    # Start with the conjecture that has the highest touch number (first in the list).
    conj = conjectures[0]

    # Initialize the list of strong conjectures with the first conjecture.
    strong_conjectures = [conj]

    # Get the set of sharp objects (i.e., objects where the conjecture holds as equality) for the first conjecture.
    sharp_objects = conj.sharps

    # Iterate over the remaining conjectures in the list.
    for conj in conjectures[1:]:
        if conj.is_equal():
            strong_conjectures.append(conj)
        else:
            # Check if the current conjecture set of sharp objects is a superset of any already selected strong conjecture.
            if any(conj.sharps.issuperset(known.sharps) for known in strong_conjectures):
                # If it does, add the current conjecture to the list of strong conjectures.
                strong_conjectures.append(conj)
                sharp_objects = sharp_objects.union(conj.sharps)

    # Return the list of strong, non-redundant conjectures.
    return strong_conjectures


def filter_false_conjectures(conjectures, df):
    """
    Filters conjectures to remove those with counterexamples in the provided data.

    Parameters
    ----------
    conjectures : list of Conjecture
        The list of conjectures to filter.
    df : pandas.DataFrame
        The DataFrame containing graph data.

    Returns
    -------
    list of Conjecture
        A list of conjectures with no counterexamples in the DataFrame.
    """
    new_conjectures = []
    for conj in conjectures:
        if conj.false_objects(df).empty:
            new_conjectures.append(conj)
    return new_conjectures
