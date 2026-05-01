# 16, Phase 13 deep dive: the `[π, π]_SN` obstruction

Tutorial 12 introduced the Schouten–Nijenhuis bracket and noted that
``[·,·]_SN(π, π)`` is the universal Poisson obstruction, the inert
`BracketApply` handle whose vanishing *is* the Jacobi identity for
``{·,·}_π``. That tutorial closed the function-side Jacobi sum in
**one** step via `prove_jacobi_reduction`, citing the seeded
**Derived Bracket Theorem**.

Phase 13 took the harder road: close the same identity *without
citing any seeded theorem*, using only engine-level rewrite axioms.
This tutorial walks the resulting machinery, the eight axioms and
the `LieBracketVF` atom that make it possible, and the asymmetry
between the form-side (2f-deep) and function-side (2g-deep) chains.

The deeper view matters when you want to:

* extend the engine to a bracket the seeded `poisson_jacobi`
  theorem doesn't cover (a custom derived bracket on a non-Poisson
  bivector, the Courant–Dorfman bracket, ...);
* render a transcript that exhibits the cancellation step-by-step
  (publication or pedagogy);
* understand *why* the function-side chain is short and the
  form-side chain needs bookkeeping.

## `LieBracketVF`, Lie bracket of vector fields as an atom

The first design decision: `[X, Y]_VF` is *not* expanded to
``X*Y − Y*X`` inside the engine. It's an opaque
`Derivation` subclass whose identity is structural over `(X, Y)`:

```python
from jacopy.algebra.derivation import Derivation
from jacopy.algebra.lie_bracket_vf import lie_bracket_vf

X = Derivation("X", 0)
Y = Derivation("Y", 0)

bracket = lie_bracket_vf(X, Y)
print(f"name      : {bracket}")
print(f"X         : {bracket.X}")
print(f"Y         : {bracket.Y}")
print(f"degree    : {bracket._degree}")
print(f"same atom : {bracket == lie_bracket_vf(X, Y)}")
```

Why opaque: after the operator-commutator fold collapses
``L_X ∘ L_Y − L_Y ∘ L_X`` to ``L_{[X,Y]_VF}``, the resulting Lie
derivative must be applicable to a form like any ordinary `L_W`,
downstream Cartan rewrites need a single derivation, not a
two-term commutator. Keeping `[X, Y]_VF` opaque preserves that
uniformity. The literal expansion is still available through
`LieBracket.expand` for callers who need `X*Y − Y*X`.

## The two vector-field axioms (Faz 13.C)

`OpCommutatorVfDefinition` folds the operator commutator into a
`LieBracketVF`-flavoured Lie derivative; `LieVfJacobiDefinition`
discharges the cyclic three-bracket triple. Both fire on a `Sum`
and scan its children for the matching pattern, order-permissive,
so the upstream pipeline doesn't have to canonicalise first.

```python
from jacopy.algebra.derivation import Act
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.vf_axioms import OpCommutatorVfDefinition
from jacopy.core.expr import Symbol, Sum, Neg
from jacopy.proof.expansion import ExpansionEngine

omega = Symbol("ω")
LX, LY = lie_derivative(X), lie_derivative(Y)

# L_X(L_Y(ω)) − L_Y(L_X(ω))
expr = Sum(
    Act(LX, Act(LY, omega)),
    Neg(Act(LY, Act(LX, omega))),
)

engine = ExpansionEngine([OpCommutatorVfDefinition()])
out, steps = engine.expand(expr)
print(f"input  : {expr}")
print(f"output : {out}")
print(f"rule   : {steps[0].rule}")
```

The rule fired exactly once, replacing the two-term commutator
with a single `Act(L_{[X,Y]_VF}, ω)`. The match is structural,
the rule looks for a positive `Act(L_X, Act(L_Y, ω))` paired with
its sign-flipped twin `Neg(Act(L_Y, Act(L_X, ω)))` and rejects
anything else.

The Jacobi rule (`LieVfJacobiDefinition`) is the same kind of
matcher one level up: it finds three cyclically-permuted
``Act(L_{[X,[Y,Z]_VF]_VF}, ω)`` terms sharing a single operand
and rewrites the cyclic sum to `Integer(0)`.

## Function-side closure (2g-deep, end-to-end)

Two more axioms, both in `jacopy.calculus.poisson_axioms`, close
the cyclic Poisson Jacobi sum to ``[·,·]_SN(π, π)``:

* `PoissonAsHamiltonianDefinition` rewrites ``{f, g}_π → X_f(g)``
  for a *pinned* `DerivedBracket` instance (object identity, not
  name match, keeps unrelated brackets out of the rewrite scope).
* `HamiltonianCyclicSnFormulaDefinition` collapses the cyclic
  ``Σ_cyc Act(X_a, Act(X_b, c))`` triple to a single
  `BracketApply([·,·]_SN, π, π)` node.

Together they reduce the cyclic Poisson Jacobi sum end-to-end:

```python
from jacopy.brackets.derived import DerivedBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.poisson_axioms import (
    PoissonAsHamiltonianDefinition,
    HamiltonianCyclicSnFormulaDefinition,
)
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
pi = Symbol("π"); reg.declare(pi, Graded(degree=1))
f = Symbol("f"); reg.declare(f, Graded(degree=-1))
g = Symbol("g"); reg.declare(g, Graded(degree=-1))
h = Symbol("h"); reg.declare(h, Graded(degree=-1))

P = DerivedBracket(sn, pi, name="[·,·]_π")
obs = P.graded_jacobi_obstruction(f, g, h, registry=reg)
print(f"input : {obs}")

engine = ExpansionEngine([
    PoissonAsHamiltonianDefinition(bracket=P, bivector=pi),
    HamiltonianCyclicSnFormulaDefinition(bivector=pi),
])

result, steps = engine.expand(obs)
print(f"steps : {len(steps)}")
for i, s in enumerate(steps):
    print(f"  [{i:02d}] {s.rule}")
print(f"final : {result}")
```

Seven steps: six `PoissonAsHamiltonian` rewrites (inner-then-outer
on each cyclic term, `{f, {g, h}}` becomes `X_f(X_g(h))`) plus
one `HamiltonianCyclicSn` collapse to ``−[·,·]_SN(π, π)``. The
final `Neg` is the sign carried in from the
`graded_jacobi_obstruction` shape, not a sign-error.

## Form-side asymmetry (2f-deep)

The form-side chain, the cyclic Koszul Jacobi sum on three 1-forms
, closes through an analogous `SnBivectorFormulaDefinition` (Faz
13.D), but it doesn't reach a clean ``[·,·]_SN(π, π)`` residue
without extra bookkeeping. The reason is structural: the form-side
sum, after expanding ``{α, β}_K = L_{π^♯α}β − L_{π^♯β}α − d⟨π^♯α, β⟩``
and folding operator commutators, leaves three pieces:

1. The named-bracket cyclic ``Σ_cyc L_{[π^♯·, π^♯·]_VF}(·)``,
   what `SnBivectorFormulaDefinition` rewrites to
   ``[·,·]_SN(π, π)``.
2. Iterated Lie-derivative tails ``L_{π^♯·}(L_{π^♯·}(·))`` that
   need a separate cancellation pass.
3. ``d⟨·, ·⟩`` and ``L_·(d⟨·, ·⟩)`` residues from the Koszul
   third-term that don't enter the SN handle directly.

Pieces (2) and (3) are the *bookkeeping* burden. They cancel
algebraically but require either user-driven simplification or
additional rules to reach a literal-zero residue. The function-side
chain doesn't have that burden because the `Act(X_f, X_g(h))` shape
absorbs all the relevant content into a single iterated derivation
, there's nothing left over to cancel separately.

This asymmetry isn't a bug. It's the natural consequence of
1-forms carrying more structure than functions: the Cartan-layer
expansion of the Koszul bracket *is* deeper than the
hamiltonian-action expansion of the Poisson bracket.

## When to use the deep machinery

| If you want… | Reach for… |
|---|---|
| One-step Jacobi reduction citing a seeded theorem | `PoissonBracket.prove_jacobi_reduction` (tutorial 12) |
| Step-by-step transcript through `LieBracketVF` folds | This tutorial's two-axiom engine, function-side |
| The same chain on a custom derived bracket without a seeded theorem | Pin a `DerivedBracket`, layer `PoissonAsHamiltonian` + `HamiltonianCyclicSn` |
| Form-side cancellation showing the 3-form ``a ∧ b ∧ c`` pairing | Faz 13.D `SnBivectorFormulaDefinition` + manual residue work |

The default workflow stays at the seeded-theorem level,
`prove_jacobi_reduction` is shorter and the transcript reads as
"by the Derived Bracket Theorem". This deeper machinery is what
sits *underneath* that one-line theorem citation, ready when the
seed isn't applicable.

## Summary

* `LieBracketVF(X, Y)` is an opaque `Derivation` atom, kept
  unexpanded so the operator-commutator fold yields a single
  derivation that downstream Cartan rules can consume.
* `OpCommutatorVfDefinition` folds ``L_X(L_Y(ω)) − L_Y(L_X(ω))``
  into ``L_{[X,Y]_VF}(ω)``; `LieVfJacobiDefinition` discharges
  the cyclic Lie-Jacobi triple to zero.
* Function-side: `PoissonAsHamiltonianDefinition` +
  `HamiltonianCyclicSnFormulaDefinition` close the cyclic Poisson
  Jacobi sum to ``[·,·]_SN(π, π)`` in seven engine steps, no
  seeded theorem cited.
* Form-side (2f-deep): `SnBivectorFormulaDefinition` produces the
  same SN handle for the named-bracket cyclic, but bookkeeping
  residues from iterated Lie tails and `d⟨·, ·⟩` need additional
  cancellation work.
* The seeded `poisson_jacobi` theorem (cited via
  `prove_jacobi_reduction`) is the one-step shortcut for this
  whole chain, use it as the default workflow; reach for this
  machinery when the seed doesn't apply.
