
from graffitiai.experimental.expression_objects import Var, Const, Add, Sub, Mul, Div, Pow, Max, Min

__all__ = [
    "simplify",
    "generate_expressions",
    "generate_expressions_exact",
    "generate_simplified_expressions",
]


def simplify(expr):
    """
    Recursively simplify the expression tree by applying basic identities.
    """
    if isinstance(expr, (Var, Const)):
        return expr

    # Simplify the left and right subtrees.
    left = simplify(expr.left)
    right = simplify(expr.right)

    # Constant folding for binary operations.
    if isinstance(left, Const) and isinstance(right, Const):
        if isinstance(expr, Add):
            return Const(left.value + right.value)
        if isinstance(expr, Sub):
            return Const(left.value - right.value)
        if isinstance(expr, Mul):
            return Const(left.value * right.value)
        if isinstance(expr, Div):
            if right.value == 0:
                return Div(left, right)  # Avoid division by zero.
            return Const(left.value / right.value)
        if isinstance(expr, Pow):
            # Check for 0 raised to a negative power.
            if left.value == 0 and right.value < 0:
                # Avoid folding: return the unsimplified power expression.
                return Pow(left, right)
            try:
                return Const(left.value ** right.value)
            except Exception:
                return Const(float(left.value) ** right.value)

        if isinstance(expr, Max):
            return Max(left, right)
        if isinstance(expr, Min):
            return Min(left, right)

    # Apply algebraic identities.
    if isinstance(expr, Add):
        if isinstance(left, Const) and left.value == 0:
            return right
        if isinstance(right, Const) and right.value == 0:
            return left
    elif isinstance(expr, Sub):
        if isinstance(right, Const) and right.value == 0:
            return left
        if left == right:
            return Const(0)
    elif isinstance(expr, Mul):
        if isinstance(left, Const):
            if left.value == 1:
                return right
            if left.value == 0:
                return Const(0)
        if isinstance(right, Const):
            if right.value == 1:
                return left
            if right.value == 0:
                return Const(0)
    elif isinstance(expr, Div):
        if isinstance(left, Const) and left.value == 0:
            return Const(0)
        if isinstance(right, Const) and right.value == 1:
            return left
        if left == right:
            return Const(1)
    elif isinstance(expr, Pow):
        # New rule: 1 raised to any power is 1.
        if isinstance(left, Const) and left.value == 1:
            return Const(1)
        if isinstance(right, Const) and right.value == 1:
            return left
        if isinstance(right, Const) and right.value == 0:
            return Const(1)

    # If no rule applied, reconstruct the node.
    if isinstance(expr, Add):
        return Add(left, right)
    if isinstance(expr, Sub):
        return Sub(left, right)
    if isinstance(expr, Mul):
        return Mul(left, right)
    if isinstance(expr, Div):
        return Div(left, right)
    if isinstance(expr, Pow):
        return Pow(left, right)
    if isinstance(expr, Max):
        return Max(left, right)
    if isinstance(expr, Min):
        return Min(left, right)

    return expr


def generate_expressions_exact(variables, depth, constants=None):
    """
    Generate all expressions with exactly 'depth' binary operations.
    Leaves are drawn from variables and optionally a list of constants.
    """
    if constants is None:
        constants = []
    if depth == 0:
        return [Var(v) for v in variables] + [Const(n) for n in constants]

    exprs = []
    for left_depth in range(0, depth):
        right_depth = depth - 1 - left_depth
        left_exprs = generate_expressions_exact(variables, left_depth, constants)
        right_exprs = generate_expressions_exact(variables, right_depth, constants)
        for left in left_exprs:
            for right in right_exprs:
                for op in [Add, Sub, Mul, Div, Pow, Max, Min]:
                    exprs.append(op(left, right))
    return exprs

def generate_expressions(variables, max_depth, constants=None):
    """
    Generate all expressions that use up to 'max_depth' binary operations.
    """
    all_exprs = []
    for d in range(max_depth + 1):
        all_exprs.extend(generate_expressions_exact(variables, d, constants))
    return all_exprs


def generate_simplified_expressions(variables, max_depth, constants=None):
    """
    Generate all simplified expressions that use up to 'max_depth' binary operations.
    """
    all_exprs = generate_expressions(variables, max_depth, constants)
    return [simplify(expr) for expr in all_exprs]
