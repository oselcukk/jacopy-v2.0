# 06, Custom bracket

Earlier tutorials worked with the brackets shipped in the package:
`LieBracket`, `sn`, `KoszulBracket`, `CourantBracket`… When the
user wants to insert their own definition rule (an expansion
function) as a bracket, without writing a full `GradedBracket`
subclass, `jacopy.brackets.custom.CustomBracket` is the entry
point. This tutorial covers (a) the minimum data profile of
`CustomBracket`, (b) declaring its axiom profile through flags,
(c) the path `prove_jacobi` follows on this kind of bracket, and
(d) when to graduate from `CustomBracket` to a `GradedBracket`
subclass.

[05, Cartan calculus](05_cartan_calculus.md) covered the
operator-level relations of stock brackets; here we step back and
define a bracket of our own at the symbol level.

## The minimum profile, two arguments

`CustomBracket(name, expand_fn, *, degree, is_graded_antisymmetric,
satisfies_leibniz, satisfies_graded_jacobi)`. Required fields are
just the name and the expand callable, everything else has
sensible defaults (degree 0, antisymmetric, Leibniz, Jacobi).

`expand_fn` has a fixed signature: `(a, b, registry) → Expr`. The
registry is passed on every call even when the rule doesn't need
it, so the call shape stays uniform across brackets.

```python
from jacopy.brackets.custom import CustomBracket
from jacopy.core.expr import Neg, Product, Sum, Symbol


def commutator(a, b, registry):
    return Sum(Product(a, b), Neg(Product(b, a)))


B = CustomBracket("[·,·]", commutator)
B(Symbol("X"), Symbol("Y")).expand()
# ((X * Y) + (-(Y * X)))
```

`B(X, Y)` returns a `BracketApply` node; calling `.expand()` on it
(or `expand_bracket(...)` at the package level) fires the
`commutator` rule.

## Axiom-profile flags

Four flags carry the bracket's axiomatic claims symbolically. They
control which shortcuts the engine takes and what `prove_jacobi`
expects:

| flag | meaning | default |
|------|---------|---------|
| `degree` | the bracket's degree shift (`|[a,b]| = |a|+|b|+degree`) | `0` |
| `is_graded_antisymmetric` | `[a,b] = −(−1)^{|a||b|}[b,a]` | `True` |
| `satisfies_leibniz` | Leibniz on the second slot | `True` |
| `satisfies_graded_jacobi` | graded Jacobi | `True` / `False` / `None` |

`None` is a third option, *conditional Jacobi*. `DerivedBracket`
is the canonical example; if your custom bracket's Jacobi depends
on a separate condition (e.g. `[Q,Q]_base = 0`), set the flag to
`None` and let the proof layer pick the right strategy.

```python
B_asym = CustomBracket(
    "asym",
    lambda a, b, reg: Product(a, b),
    is_graded_antisymmetric=False,
    satisfies_leibniz=False,
    satisfies_graded_jacobi=False,
)
B_asym.is_graded_antisymmetric, B_asym.satisfies_graded_jacobi
# (False, False)
```

## `prove_jacobi`, the generic dispatch path

A `CustomBracket` is not a `DerivedBracket`; `prove_jacobi`
dispatches it onto the generic `GradedBracket` path. That path:

1. `graded_jacobi_obstruction(a, b, c, registry)`, the triple
   cyclic sum `(−1)^{|a||c|}[a,[b,c]] + …`.
2. Expands every bracket node through `expand_fn` (the
   `bracket-expand` step).
3. Runs `ExpandAndSimplify` against `Integer(0)`.

For the commutator rule the whole chain closes to zero:

```python
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.verifier import prove_jacobi

reg = PropertyRegistry()
for s in (Symbol("X"), Symbol("Y"), Symbol("Z")):
    reg.declare(s, Graded(degree=0))

chain = prove_jacobi(B, Symbol("X"), Symbol("Y"), Symbol("Z"), registry=reg)
len(chain)                        # 2
chain.steps[0].rule               # 'bracket-expand'
chain.steps[1].rule               # 'simplify'
chain.steps[1].after              # 0
```

If you supply a wrong rule, the same pipeline leaves a residual
and raises `ProofFailure`:

```python
from jacopy.proof.strategies import ProofFailure

try:
    prove_jacobi(B_asym, Symbol("X"), Symbol("Y"), Symbol("Z"), registry=reg)
except ProofFailure as exc:
    str(exc)
# "ExpandAndSimplify left residual (3 * X * Y * Z) when proving ... == 0"
```

The residual `3 * X * Y * Z` in the message is the direct symbolic
proof that the `asym` rule does *not* satisfy Jacobi.

## Axiom-obstruction helpers

Three helpers inherited from `GradedBracket` give you the explicit
expression of each axiom claim. Useful for probing a rule on a
symbolic triple / pair before going into a full proof:

```python
a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
for s in (a, b, c):
    reg.declare(s, Graded(degree=0))

B.graded_antisymmetry_obstruction(a, b, reg)
# ([·,·](a, b) + [·,·](b, a))

B.graded_jacobi_obstruction(a, b, c, reg)
# ([·,·](a, [·,·](b, c)) + [·,·](b, [·,·](c, a)) + [·,·](c, [·,·](a, b)))

B.leibniz_obstruction(a, b, c, reg)
# ([·,·](a, (b * c)) + (-([·,·](a, b) * c)) + (-(b * [·,·](a, c))))
```

Each is an `Expr`, running `simplify(..., reg)` on it tests
whether the rule satisfies that axiom. If parity is undecidable
(`None`), it raises `ValueError` and you need to narrow the
operand degrees. The early failure at the symbolic level is by
design, catching it here is cheaper than at proof-layer time.

## `_identity_key` and equality

Two `CustomBracket`s compare equal only when they share the
*same* expand callable. Python function identity is used, two
different `lambda`s with identical names and degrees are still
distinct brackets:

```python
rule_a = lambda a, b, reg: Sum(Product(a, b), Neg(Product(b, a)))
rule_b = lambda a, b, reg: Sum(Product(a, b), Product(b, a))
CustomBracket("B", rule_a) == CustomBracket("B", rule_b)   # False
```

The design is intentional: otherwise the hash-table key for
`DerivedBracket(base=B_a, ...)` would collide with another
derived bracket built on a different base.

## When to graduate from `CustomBracket`

`CustomBracket` is the right tool in two situations:

- Tutorial / exploration: quickly write a rule and run Jacobi /
  Leibniz tests on it.
- The workbench for a bracket whose axioms haven't crystallised
  yet.

But once you need `expand_definition`, obstruction hooks, an
anchor lift, a custom identity key (`_identity_key` extension),
or a `Theorem` registered against `theorem_book`, subclass
`GradedBracket` directly. The package's own brackets
(`KoszulBracket`, `CourantBracket`, `DerivedBracket`,
`SchoutenBracket`) are the latter.

## Next step

`CustomBracket` exposes a rule but does not extract a *structure*.
A derived bracket, by contrast, picks a single "generator" `Q`
and builds the upstairs bracket axioms automatically, Poisson,
Koszul, and Courant brackets are all instances of this
construction. [07, Derived bracket](07_derived_bracket.md).
