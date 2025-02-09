import numpy as np
import pandas as pd

__all__ = [
    "Var",
    "Const",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Max",
    "Min",
]

class Expression:
    def __add__(self, other):
        return Add(self, other if isinstance(other, Expression) else Const(other))

    def __radd__(self, other):
        return Add(Const(other), self)

    def __sub__(self, other):
        return Sub(self, other if isinstance(other, Expression) else Const(other))

    def __rsub__(self, other):
        return Sub(Const(other), self)

    def __mul__(self, other):
        return Mul(self, other if isinstance(other, Expression) else Const(other))

    def __rmul__(self, other):
        return Mul(Const(other), self)

    def __truediv__(self, other):
        return Div(self, other if isinstance(other, Expression) else Const(other))

    def __rtruediv__(self, other):
        return Div(Const(other), self)

    def __xor__(self, other):
        return Pow(self, other if isinstance(other, Expression) else Const(other))

    def __rxor__(self, other):
        return Pow(Const(other), self)

    def eval(self, df):
        raise NotImplementedError("eval not implemented.")

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

class Var(Expression):
    def __init__(self, name):
        self.name = name

    def eval(self, df):
        return df[self.name]

    def __repr__(self):
        return self.name

class Const(Expression):
    def __init__(self, value):
        self.value = value

    def eval(self, df):
        return self.value

    def __repr__(self):
        return str(self.value)

class BinaryOp(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Add(BinaryOp):
    def eval(self, df):
        return self.left.eval(df) + self.right.eval(df)

    def __repr__(self):
        return f"({self.left} + {self.right})"

class Sub(BinaryOp):
    def eval(self, df):
        return self.left.eval(df) - self.right.eval(df)

    def __repr__(self):
        return f"({self.left} - {self.right})"

class Mul(BinaryOp):
    def eval(self, df):
        return self.left.eval(df) * self.right.eval(df)

    def __repr__(self):
        return f"({self.left} * {self.right})"

class Div(BinaryOp):
    def eval(self, df):
        return self.left.eval(df) / self.right.eval(df)

    def __repr__(self):
        return f"({self.left} / {self.right})"

class Pow(BinaryOp):
    def eval(self, df):
        base = self.left.eval(df)
        exponent = self.right.eval(df)

        # Convert base to float to avoid integer exponentiation issues
        base = base.astype(float) if isinstance(base, pd.Series) else float(base)

        try:
            return np.power(base, exponent)
        except ValueError:
            print(f"Skipping exponentiation {base} ** {exponent} due to invalid operation.")
            return np.nan  # Return NaN for invalid cases

    def __repr__(self):
        return f"({self.left} ^ {self.right})"

class Min(BinaryOp):
    def eval(self, df):
        return np.minimum(self.left.eval(df), self.right.eval(df))

    def __repr__(self):
        return f"min({self.left}, {self.right})"

class Max(BinaryOp):
    def eval(self, df):
        return np.maximum(self.left.eval(df), self.right.eval(df))

    def __repr__(self):
        return f"max({self.left}, {self.right})"
