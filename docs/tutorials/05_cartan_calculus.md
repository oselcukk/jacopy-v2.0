# 05, Cartan calculus

The arithmetic core of differential geometry reduces to five
relations:

1. `d² = 0`, the exterior derivative squares to zero
2. `[d, ι_X] = L_X`, Cartan's magic formula
3. `[d, L_X] = 0`
4. `[L_X, L_Y] = L_{[X, Y]}`
5. `[L_X, ι_Y] = ι_{[X, Y]}`

`jacopy` collects all five into the
`CartanCalculus(d, lie_derivative, interior, vector_bracket)`
bundle, exposing each as an `OperatorEquation`. This tutorial
shows (a) how to read each relation as an *operator equation*,
(b) which relations close under live verification in which mode
(efficient vs foundational), and (c) the `invariant-d` formula as
a derived theorem.

[04, Lie algebroid](04_lie_algebroid.md) runs the same API with
bundle-tagged operators; the live verifications here are on
``TM``.

## The bundle

```python
from jacopy.algebra.derivation import Derivation
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.cartan import CartanCalculus, RELATIONS
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry

cart = CartanCalculus(
    d=d,
    lie_derivative=lie_derivative,
    interior=interior,
    vector_bracket=LieBracket(),
)
RELATIONS
# ('d_squared_zero', 'cartan_magic', 'd_lie', 'lie_lie', 'lie_iota')
```

## Five relations, five `OperatorEquation`s

`cart.relation(name, X=..., Y=..., algebra=...)` returns the
operator-level equation for each relation. Which arguments are
required depends on the relation, `d²` needs none,
`magic` / `d_lie` need a single `X`, `lie_lie` / `lie_iota` need
both:

```python
reg = PropertyRegistry()
f = Symbol("f")
reg.declare(f, Graded(degree=0))
algebra = ExteriorAlgebra((f,))
X = Derivation("X", degree=0)
Y = Derivation("Y", degree=0)

cart.relation("d_squared_zero", algebra=algebra)
# (d * d) = 0

cart.relation("cartan_magic", X=X, algebra=algebra)
# ((d * ι_X) + (ι_X * d)) = L_X

cart.relation("lie_lie", X=X, Y=Y, algebra=algebra)
# ((L_X * L_Y) + (-(L_Y * L_X))) = L_([X, Y])
```

## `d² = 0`, axiom mode vs theorem mode

The `d²` default-engine rewrite carries two classifications:
`axiom` (default, emits the axiom directly) and `theorem`
(opens a generator-level sub-proof in foundational mode). Pick
between them via `default_engine(..., d_squared_mode=...)`:

```python
from jacopy.calculus.exterior_d import apply_d_squared_zero
from jacopy.proof.expansion import default_engine

x = Symbol("x")
reg.declare(x, Graded(degree=0))

# Plain axiom rewrite, calculation helper
apply_d_squared_zero(d(d(x)))       # 0

# Theorem-mode expansion (foundational)
engine = default_engine(registry=reg, mode="foundational", d_squared_mode="theorem")
expanded, steps = engine.expand(d(d(x)))
expanded                            # 0
steps                               # [ProofStep(rule='d² = 0', d(d(x)) → 0)]
```

In the `theorem` classification the sub-proof derives `d(df) = 0`
on 0-form generators from generator-level facts, the standard
`AgreementOnGenerators` strategy that lifts an operator identity
``d ∘ d = 0`` from agreement on generators.

## All five relations, live proof via `verify`

On the default `CartanCalculus` together with an `ExteriorAlgebra`
on a single function generator, all five relations close under
`verify()`:

```python
chain = cart.verify("cartan_magic", algebra=algebra, X=X, registry=reg)
len(chain)                # 1
chain.steps[0].rule       # definition citation

chain_f = cart.verify(
    "cartan_magic",
    algebra=algebra,
    X=X,
    registry=reg,
    mode="foundational",
)
len(chain_f)              # 1, UnrollToFoundations wraps it

# Three need X,Y; d_squared_zero is purely operator-level:
cart.verify("d_squared_zero", algebra=algebra, registry=reg)
cart.verify("d_lie", algebra=algebra, X=X, registry=reg)
cart.verify("lie_lie", algebra=algebra, X=X, Y=Y, registry=reg)
cart.verify("lie_iota", algebra=algebra, X=X, Y=Y, registry=reg)

# Or all at once:
results = cart.verify_all(algebra=algebra, X=X, Y=Y, registry=reg)
set(results) == {"d_squared_zero", "cartan_magic", "d_lie",
                 "lie_lie", "lie_iota"}
```

Magic fires as a single-step theorem because of
`LieDerivativeCartanDefinition`'s default classification;
foundational mode keeps the same sub-proof as a citation. The
relations `d_lie`, `lie_lie`, `lie_iota` close at the
generator-level via `AgreementOnGenerators` + `ExpandAndSimplify`:
the Lie bracket `[X, Y] = X*Y − Y*X` opens up, graded Leibniz
distributes, and `d²=0` plus the definition `ι_V(df) = V(f)`
collapse the residue. For the algebroid variant see
[04_lie_algebroid.md](04_lie_algebroid.md), over there `verify`'s
inability to fire on bundle-tagged operators is a known deferral.

## `invariant_d`, magic + lie_iota → the `d` formula

The classical Koszul-Cartan "invariant d" formula
`dω(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y])` is a *theorem*
derived from magic and lie_iota. `jacopy` exposes it through a
single helper:

```python
from jacopy.calculus.invariant_d import invariant_d_one_form
from jacopy.brackets.lie import lie

omega = Symbol("ω")
reg.declare(omega, Graded(degree=1))
for s in (X, Y):
    pass  # already declared Graded(0) via fixtures above

invariant_d_one_form(omega, X, Y, bracket=lie)
# (X(ι_Y(ω)) + (-Y(ι_X(ω))) + (-ι_((X * Y) + (-(Y * X)))(ω)))
```

The formula is also exposed as a `Definition`,
`InvariantDOneFormDefinition`, with default classification
`"theorem"` (sub-proof citing magic + lie_iota). Note the
exception: the default classification is `"axiom"` for `d²=0` but
`"theorem"` here, because the formula naturally falls out of
those two relations rather than serving as an axiomatic entry
point.

## Twisted Cartan bundle, `d_H = d + H∧`

For a closed 3-form ``H`` the H-twisted exterior derivative
``d_H`` carries the Cartan calculus through the same five
relations. `jacopy` exposes this as a `TwistedCartanBundle(H)`
wrapper: the bundle builds ``d_H`` as a fresh
`ExteriorDerivative` and rewires the Lie-derivative factory so
that ``d_H`` slots into the bundle, a one-to-one twisted
counterpart to the algebroid bundle.

```python
from jacopy.library import TwistedCartanBundle

H = Symbol("H")
reg.declare(H, Graded(degree=3))
bundle = TwistedCartanBundle(H)
bundle.d             # d_H, fresh degree-+1 ExteriorDerivative
bundle.cartan        # CartanCalculus(d=d_H, L_{H,·}, ι_·, [·,·])

algebra_H = ExteriorAlgebra((f,), d=bundle.d)
bundle.cartan.verify_all(algebra=algebra_H, X=X, Y=Y, registry=reg)
# {'d_squared_zero': ProofChain, 'cartan_magic': ProofChain,
#  'd_lie': ProofChain, 'lie_lie': ProofChain, 'lie_iota': ProofChain}
```

The package treats ``d_H`` as a formal degree-+1 derivative, the
``d + H∧`` decomposition is not unfolded inside the engine.
Constructing a `TwistedCartanBundle` is making the assumption
``dH = 0``: closure of ``d_H² = 0`` rests on it. For the twisted
Courant bracket (the `background_H` kwarg) see
[07_derived_bracket.md](07_derived_bracket.md).

## Next step

Writing your own bracket and running the Jacobi test on it,
`CustomBracket`, the flags, and the interaction with
`prove_jacobi`: [06_custom_bracket.md](06_custom_bracket.md)
(Stage C).
