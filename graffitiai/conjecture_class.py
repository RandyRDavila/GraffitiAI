

class Hypothesis:
    """
    A base class for graph hypotheses.

    Attributes:
    - statement (str): The hypothesis statement, which describes a property or condition on a graph.
    - true_object_set (set, optional): A set of objects (objects) that satisfy the hypothesis.

    Methods:
    - __str__(): Returns the hypothesis statement as a string.
    - __repr__(): Returns the hypothesis statement in a formal representation.
    - __call__(name, df): Evaluates the hypothesis on a specific graph object, using its name and a DataFrame of graph data.
    - _le__(other): Compares the size of `true_object_set` with another Hypothesis instance.
    - __lt__(other): Checks if the `true_object_set` is smaller than that of another Hypothesis.
    - __ge__(other): Checks if the `true_object_set` is larger or equal in size to that of another Hypothesis.
    - __gt__(other): Checks if the `true_object_set` is larger than that of another Hypothesis.
    - __eq__(other): Checks if two Hypothesis instances have the same statement.
    - __hash__(): Returns a hash of the hypothesis based on the statement.
    """
    def __init__(self, statement, true_object_set=None):
        self.statement = statement
        self.true_object_set = true_object_set

    def __str__(self):
        return f"{self.statement}"

    def __repr__(self):
        return f"{self.statement}"

    def __call__(self, name, df):
        return df.loc[df["name"] == f"{name}.txt"][self.statement]

    def _le__(self, other):
        return len(self.true_object_set) <= len(other.true_object_set)

    def __lt__(self, other):
        return len(self.true_object_set) < len(other.true_object_set)

    def __ge__(self, other):
        return len(self.true_object_set) >= len(other.true_object_set)

    def __gt__(self, other):
        return len(self.true_object_set) > len(other.true_object_set)

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash((self.statement))

class Conclusion:
    """
    A base class for graph conclusions.

    Attributes:
    - lhs (str): The left-hand side of the inequality.
    - inequality (str): The inequality operator (e.g., '<=', '>=', '=').
    - rhs (str): The right-hand side of the inequality.
    - intercept (float): An optional intercept for linear expressions.

    Methods:
    - __str__(): Returns a string representation of the conclusion.
    - __repr__(): Returns a formal representation of the conclusion.
    - __call__(name, df): Evaluates the conclusion on a specific graph object, using its name and a DataFrame.
    - __hash__(): Returns a hash of the conclusion based on the string representation.
    """
    def __init__(self, lhs, inequality, rhs, intercept=0):
        self.lhs = lhs
        self.inequality = inequality
        self.rhs = rhs
        self.intercept = intercept

    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__")

    def __repr__(self):
        raise NotImplementedError("Subclasses must implement __repr__")

    def __call__(self, name, df):
        raise NotImplementedError("Subclasses must implement __call__")

    def __hash__(self):
        return hash((str(self)))

class Conjecture:
    """
    A base class for graph conjectures.

    Attributes:
    - hypothesis (Hypothesis): The hypothesis component of the conjecture.
    - conclusion (Conclusion): The conclusion component of the conjecture.
    - symbol (str): A symbol representing the object being conjectured (default is "G").
    - touch (int): The number of objects that satisfy the hypothesis and conclusion.
    - sharps (set): A set of objects that sharply satisfy the conjecture.
    - difficulty (int): The difficulty level associated with the conjecture.

    Methods:
    - __str__(): Returns a string representation of the conjecture.
    - __repr__(): Returns a formal representation of the conjecture.
    - __call__(name, df): Evaluates the conjecture on a specific graph object, using its name and a DataFrame.
    - get_sharp_objects(df): Abstract method to retrieve sharp objects from the DataFrame.
    - __eq__(other): Checks if two Conjecture instances have the same hypothesis and conclusion.
    - __hash__(): Returns a hash of the conjecture based on the string representation.
    """
    def __init__(self, hypothesis, conclusion, touch=0, sharps=None, difficulty=0):
        self.hypothesis = hypothesis
        self.conclusion = conclusion
        self.touch = touch
        self.difficulty = difficulty

        self.sharps = set(sharps)

    def __str__(self):
        hypothesis = f"For any {self.hypothesis},"
        return f"{hypothesis} {self.conclusion}."

    def __repr__(self):
        hypothesis = f"For any {self.hypothesis},"
        return f"{hypothesis} {self.conclusion}."

    def __call__(self, name, df):
        if self.hypothesis(name, df).values[0]:
            return self.conclusion(name, df)
        else:
            return False

    def get_sharp_objects(self, df):
        raise NotImplementedError("Subclasses must implement get_sharp_objects")

    def __eq__(self, other):
        return self.__str__() == other.__str__()


    def __hash__(self):
        return hash((str(self)))


class MultiLinearConclusion(Conclusion):
    """
    A class for multilinear graph conclusions.

    Attributes:
    - lhs (str): The left-hand side of the inequality.
    - inequality (str): The inequality operator.
    - slopes (list): A list of slope coefficients for the multilinear expression.
    - rhs (list): A list of variables or invariants on the right-hand side of the inequality.
    - intercept (float): An optional intercept term.

    Methods:
    - __str__(): Returns a string representation of the multilinear conclusion.
    - __repr__(): Returns a formal representation of the conclusion.
    - __call__(name, df): Evaluates the conclusion on a specific graph object.
    - reversal(): Returns a new MultiLinearConclusion with the opposite inequality.
    - rhs_evaluate(x): Evaluates the right-hand side expression given an input `x`.
    - __hash__(): Returns a hash based on the string representation of the conclusion.
    - __eq__(other): Checks if two conclusions are equivalent.
    """
    def __init__(self, lhs, inequality, slopes, rhs, intercept):
        super().__init__(lhs, inequality, rhs, intercept)
        self.slopes = slopes

    def __str__(self):
        slope_terms = []
        for m, rhs in zip(self.slopes, self.rhs):
            if m == 1:
                slope_terms.append(f"{rhs}")
            elif m == -1:
                slope_terms.append(f"- {rhs}")
            elif m != 0:
                slope_terms.append(f"{m} * {rhs}")

        slope_str = " + ".join(slope_terms)

        if self.intercept > 0:
            result = f"{slope_str} + {self.intercept}"
        elif self.intercept < 0:
            result = f"{slope_str} - {abs(self.intercept)}"
        else:
            result = slope_str

        result = result.replace("+ -", "-").strip()
        return f"{self.lhs} {self.inequality} {result}"

    def __repr__(self):
        return self.__str__()

    def __call__(self, name, df):
        data = df.loc[df["name"] == f"{name}"]
        rhs_value = sum(m * data[r].values[0] for m, r in zip(self.slopes, self.rhs)) + self.intercept
        if self.inequality == "<=":
            return data[self.lhs].values[0] <= rhs_value
        elif self.inequality == ">=":
            return data[self.lhs].values[0] >= rhs_value
        else:
            data[self.lhs].values[0] == rhs_value

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def reversal(self):
        if self.inequality == "<=":
            return MultiLinearConclusion(self.lhs, ">=", self.slopes, self.rhs, self.intercept)
        elif self.inequality == ">=":
            return MultiLinearConclusion(self.lhs, "<=", self.slopes, self.rhs, self.intercept)

    def rhs_evaluate(self, x):
        return sum(m * x for m in self.slopes) + self.intercept

    def __hash__(self):
        return hash((str(self)))


class MultiLinearConjecture(Conjecture):
    """
    A class for multilinear graph conjectures.

    Attributes:
    - hypothesis (Hypothesis): The hypothesis component of the conjecture.
    - conclusion (MultiLinearConclusion): The conclusion component of the conjecture.
    - symbol (str): A symbol representing the object being conjectured.
    - touch (int): The number of objects that satisfy the hypothesis and conclusion.
    - sharps (set): A set of objects that sharply satisfy the conjecture.
    - difficulty (int): The difficulty level associated with the conjecture.

    Methods:
    - __repr__(): Returns a formal representation of the multilinear conjecture.
    - get_sharp_objects(df): Retrieves objects from the DataFrame that sharply satisfy the conjecture.
    - false_objects(df): Retrieves objects from the DataFrame that fail to satisfy the conjecture.
    - is_equal(): Checks if the inequality in the conclusion is equality.
    - get_functions(invariant_dict): Returns functions to compute LHS and RHS of the conjecture for a graph.
    - get_penality_function(penality_dict): Returns a function to compute penalties based on the conjecture.
    - plot(df): Plots the conjecture data if the conclusion is linear.
    - __hash__(): Returns a hash based on the string representation of the conjecture.
    - __eq__(other): Checks if two conjectures have the same hypothesis and conclusion.
    """
    def __repr__(self):
        hypothesis = f"For any {self.hypothesis},"
        return f"{hypothesis} {self.conclusion}."

    def __str__(self):
        hypothesis = f"For any {self.hypothesis},"
        return f"{hypothesis} {self.conclusion}."

    def get_sharp_objects(self, df):
        return df.loc[(df[self.hypothesis.statement] == True) &
                      (df[self.conclusion.lhs] == sum(self.conclusion.slopes[i] * df[self.conclusion.rhs[i]]
                                                      for i in range(len(self.conclusion.rhs))) + self.conclusion.intercept)]

    def __eq__(self, other):
        return self.hypothesis == other.hypothesis and self.conclusion == other.conclusion

    def false_objects(self, df):
        if self.conclusion.inequality == "<=":
            return df.loc[(df[self.hypothesis.statement] == True) &
                          (df[self.conclusion.lhs] > sum(self.conclusion.slopes[i] * df[self.conclusion.rhs[i]]
                                                         for i in range(len(self.conclusion.rhs))) + self.conclusion.intercept)]
        elif self.conclusion.inequality == ">=":
            return df.loc[(df[self.hypothesis.statement] == True) &
                          (df[self.conclusion.lhs] < sum(self.conclusion.slopes[i] * df[self.conclusion.rhs[i]]
                                                         for i in range(len(self.conclusion.rhs))) + self.conclusion.intercept)]
        else:
            return df.loc[(df[self.hypothesis.statement] == True) &
                          (df[self.conclusion.lhs] != sum(self.conclusion.slopes[i] * df[self.conclusion.rhs[i]]
                                                         for i in range(len(self.conclusion.rhs))) + self.conclusion.intercept)]
    def is_equal(self):
        return self.conclusion.inequality == "="

    def __hash__(self):
        return hash((str(self)))

