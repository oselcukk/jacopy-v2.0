# 04, Lie algebroid

A Lie algebroid is a three-piece structure: a vector bundle ``E``
on a manifold ``M``, a bracket living on the sections of ``E``,
and an anchor morphism ``ρ: E → TM``. This tutorial shows how
`jacopy` collects ``(E, [·,·]_E, ρ)`` into a single `LieAlgebroid`
object, why the **anchor compatibility** axiom is treated
separately, and what the algebroid Cartan calculus
(`d_E`, `L_{E,X}`, `ι_{E,X}`) looks like.

It's intended to be read after [03, Poisson geometry](03_poisson_geometry.md),
where derived brackets first showed up.

## The triple `(E, [·,·]_E, ρ)`

`LieAlgebroid(bundle, bracket=..., anchor=..., vector_bracket=...)`
holds the four-tuple in one object:

- `bundle`, the symbolic name of ``E`` (display only).
- `bracket`, the `GradedBracket` on sections of ``E``;
  graded antisymmetry, Jacobi, and Leibniz axioms ride on the
  bracket's *own* flags.
- `anchor`, `Anchor(name="ρ")`: an ``E → TM`` linear morphism
  (sits as a degree-0 `Derivation`, with trivial Leibniz).
- `vector_bracket`, the ``TM`` bracket the compatibility targets.
  Defaults to the `jacopy.brackets.lie.lie` singleton.

```python
from jacopy import VectorFields
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.anchor import Anchor
from jacopy.core.expr import Symbol
from jacopy.core.registry import PropertyRegistry
from jacopy.library.lie_algebroid import LieAlgebroid

reg = PropertyRegistry()
E = Symbol("E")
bracket_E = LieBracket(name="[·,·]_E")
rho = Anchor(name="ρ")

A = LieAlgebroid(E, bracket=bracket_E, anchor=rho, name="E-algebroid")
```

## Anchor compatibility, a separate axiom

The bracket's three axioms (antisymmetry, Jacobi, Leibniz) do
**not** entail anchor compatibility. The identity
``ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}`` is *not* derivable from the
classical Lie bracket axioms; it's part of the Lie algebroid
**definition**. `jacopy` exposes it three different ways:

1. **Raw obstruction (Expr):** the difference that should be zero
  , reducing it via simplify is the user's choice.
2. **VanishingCondition:** the raw obstruction plus a named
   condition.
3. **ProofChain:** a single `axiom`-tagged step, accepts the
   axiom and drives the obstruction to zero.

```python
X, Y = VectorFields("X Y", registry=reg)

A.anchor_compatibility_obstruction(X, Y, reg)
# (ρ(((X * Y) + (-(Y * X)))) + (-((ρ(X) * ρ(Y)) + (-(ρ(Y) * ρ(X))))))

A.anchor_compatibility_condition(X, Y, reg)
# VanishingCondition(..., name='anchor compatibility on E-algebroid')

chain = A.prove_anchor_compatibility(X, Y, registry=reg)
chain.steps[0].rule              # 'LieAlgebroidAnchorCompat'
chain.steps[0].provenance_tag    # 'axiom'
```

Asking `simplify` to drive the obstruction to zero will not work,
since the ``TM`` bracket is atomic, no rewrite rule connects the
two sides. That's by design: it surfaces compatibility as a
deliberate **axiom choice** instead of hiding it.

## The algebroid Cartan bundle

A Cartan calculus living on the exterior algebra ``Λ*E*`` over
``E`` is exposed via `A.cartan`:

- `d_E`, the algebroid exterior derivative (degree +1). Built
  inside `LieAlgebroid` as `ExteriorDerivative(name=f"d_{E}")`,
  reachable via `A.d`.
- `L_{E,X}`, algebroid Lie-derivative factory:
  `cart.lie_derivative(X)`. The bundle tag is woven into the
  display name (`L_E,X`) so it stays distinguishable from the
  manifold Lie derivative inside the same expression.
- `ι_{E,X}`, algebroid interior-product factory:
  `cart.interior(X)`. Tagged similarly as `ι_E,X`.

```python
cart = A.cartan
A.d                 # d_E
cart.lie_derivative(X)   # L_E,X
cart.interior(X)         # ι_E,X
```

The five Cartan relations (`d_E² = 0`, magic, `[d_E, L]`,
`[L, L]`, `[L, ι]`) are exposed through the same API,
`cart.relation(name, X=..., Y=...)` returns an `OperatorEquation`:

```python
eq = cart.relation("cartan_magic", X=X)
# OperatorEquation of the form [d_E, ι_E,X] = L_E,X
```

**Note.** On the algebroid Cartan layer, `cart.verify(...)` does
not auto-close because the current expansion engine pattern-matches
the "definition rewrite" of Cartan magic against the default
``d / ι_X`` ``TM`` operators rather than the algebroid-named
``d_E / ι_E,X``. This is known and recorded under
`engine_cartan_definition_deferral`. Live verified examples of the
five relations live on ``TM`` in
[05, Cartan calculus](05_cartan_calculus.md); the structural
symmetry is identical on the algebroid side.

## Seeded theorem

The compatibility axiom is registered in `theorem_book`:

```python
from jacopy.library import theorem_book

thm = theorem_book.get("lie_algebroid_anchor_compat")
thm.statement    # "ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}"
thm.from_axioms  # ('Lie algebroid anchor compatibility axiom',)
```

The registration lets downstream theorems (algebroid Cartan,
Courant–Dorfman bridge) cite this axiom in a single step.

## Next step

A live walk-through of the five Cartan relations, `d² = 0`,
magic, `[d, L]`, `[L, L]`, `[L, ι]`, on ``TM`` in two modes
(efficient vs foundational): [05_cartan_calculus.md](05_cartan_calculus.md).
