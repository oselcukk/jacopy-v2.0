# 01, First steps

This tutorial walks through building a symbolic expression in
`jacopy`, declaring properties, and simplifying. By the end you'll
be able to write small algebraic expressions with `Expr`, `Symbol`,
`Sum`, `Product`, `Neg`, declare relationships between symbols (such
as degree or commutativity) through a `PropertyRegistry`, and
canonicalise them with `simplify`.

## Symbols and basic construction

A symbol is built with `Symbol(name)`; integer literals via
`Integer(n)`; the standard Python operators `+`, `*`, `-` produce
`Sum`, `Product`, `Neg` under the hood, so `x + y` returns the
exact same object as `Sum(x, y)`.

```python
from jacopy.core.expr import Symbol, Integer, Sum, Product, Neg

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

expr = x + y - z
assert expr == Sum(x, y, Neg(z))

mult = 2 * x
assert mult == Product(Integer(2), x)
```

`Expr` is a value object: hashable, comparable for equality,
immutable. Two structurally identical expressions compare equal via
`==`.

## Declaring properties (`PropertyRegistry`)

By default symbols carry no algebraic content. Relationships are
declared **externally** through a `PropertyRegistry`, that's what
lets the same symbol be reused in different contexts (with
different degrees, for example).

`Graded(degree=k)` records a degree; `Scalar()` marks a symbol as a
commuting scalar:

```python
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
reg.declare(x, Scalar())
reg.declare(y, Scalar())
reg.declare(z, Graded(degree=1))  # z behaves like a 1-form
```

Two symbols declared `Scalar` commute under `Product`; graded
symbols pick up the sign ``(−1)^{|a||b|}`` when swapped.

### Role-driven shortcuts

For common patterns (functions, vector fields, forms, bivectors)
the `jacopy.library.declarations` module provides role-driven
helpers. Each collapses `Symbol(...)` plus the matching
`reg.declare(...)` into a single call:

```python
from jacopy import Functions, VectorFields, Forms, Bivector

reg2 = PropertyRegistry()
f, g = Functions("f g", registry=reg2)         # Graded(degree=0)
X, Y = VectorFields("X Y", registry=reg2)      # Graded(degree=0)
alpha, beta = Forms("α β", degree=1, registry=reg2)  # Graded(degree=1)
pi = Bivector("π", registry=reg2)              # Graded(degree=1), SN-grading
```

Even a single name returns a tuple, `(f,) = Functions("f", ...)`;
`Bivector` is the lone exception, returning the symbol directly.

## `simplify`, canonical form

The `simplify(expr, registry)` pipeline runs:

1. `flatten`, merges nested `Sum` / `Product` nodes.
2. `canonicalize`, pushes `Neg` through sums, collects signs.
3. `distribute`, opens `Sum`s inside `Product`s when needed.
4. `sort_product`, orders factors by registered properties.
5. `collect_terms`, combines like terms (`x + x → 2x`, etc.).

```python
from jacopy.algorithms.simplify import simplify

assert simplify(x + x - x) == x
assert simplify(Product(Integer(2), x, Integer(3)), reg) == Product(Integer(6), x)
assert simplify(Product(y, x), reg) == Product(x, y)  # alphabetic order
```

When called with a `registry`, simplify consults the declared
properties; with `registry=None` only the symbol-independent
passes run (flatten + canonicalize + constant arithmetic).

## Display

The `display` layer offers three rendering paths:

- `to_ascii(expr)`, plain text, suitable for monospace terminals.
- `to_latex(expr)`, LaTeX string; renders directly via
  `LatexDisplay` in Jupyter.
- `print_expr(expr)`, coloured tree if `rich` is installed,
  ASCII fallback otherwise.

```python
from jacopy.display import to_ascii, to_latex

to_ascii(x + y - z)       # 'x + y - z'
to_latex(x + y - z)       # 'x + y - z'
```

## Next step

Once a Lie bracket is layered onto the same `PropertyRegistry`,
proof helpers like `prove_jacobi` build directly on `simplify`.
[`02_jacobi_identity.md`](02_jacobi_identity.md) shows how the
Jacobi identity closes in a single call.
