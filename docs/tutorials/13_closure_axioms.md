# 13, Closure properties & axiom flags

Some facts are too structural to derive, every Hamiltonian-invariance
proof on a symplectic form invokes ``dω = 0``, every Poisson-bracket
calculation invokes ``π(α, β) = −π(β, α)``, every vector-field-equality
proof on a non-degenerate form peels through ``ι_Y ω = ι_Z ω ⇒ Y = Z``.
Reproducing those rewrites inline in every notebook is the noisy
pre-Phase 12.B pattern; the **closure-axiom layer** elevates them to
*declarative* properties on the registry that an engine rule consumes
as a primitive.

This tutorial walks the three closure properties (``Closed``,
``Antisymmetric``, ``NonDegenerate``) end-to-end: how to declare them,
which engine rule each pairs with, and the rewrite shape it produces.
By the end you'll know how to wire a closure axiom into a custom
engine, and when the problem-wrapper layer (tutorial 14) does it for
you automatically.

The three flags follow a single recipe:

| Property | Pairs with | Rewrite shape |
|---|---|---|
| `Closed` | `ClosedFormDefinition` | `Act(d, ω) → 0` |
| `Antisymmetric` | `RegistryAntiSymCanonicalDefinition` | `π(β, α) → −π(α, β)` (canonical sort) |
| `NonDegenerate` | `NonDegenerateInteriorEqualityDefinition` | `ι_Y ω − ι_Z ω → Y − Z` |

## `Closed`, `dω = 0` on demand

Declare the property; layer the rule onto an engine; the engine
rewrites every `Act(d, ω)` to `0` for the registered form. The
property carries no internal data, it's a one-bit fact about ``ω``.

```python
from jacopy.algebra.derivation import Act
from jacopy.calculus.closed_axioms import ClosedFormDefinition
from jacopy.calculus.exterior_d import d
from jacopy.core.expr import Symbol, Integer
from jacopy.core.properties import Graded, Closed
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import prove_equivalence
from jacopy.proof.expansion import ExpansionEngine, default_engine

reg = PropertyRegistry()
omega = Symbol("ω")
reg.declare(omega, Graded(degree=2))
reg.declare(omega, Closed())          # the closure axiom flag

# Layer the rule onto the default engine.
base = default_engine(registry=reg)
engine = ExpansionEngine(
    list(base.definitions) + [ClosedFormDefinition(registry=reg)]
)

chain = prove_equivalence(Act(d, omega), Integer(0),
                          registry=reg, engine=engine)
print(f"d(ω) = 0 in {len(chain)} steps: {chain.steps[0].rule}")
```

The single named step (`Closed: d(ω) = 0 when ω is declared closed`)
is the proof artefact: the chain transcript reads as *"because ω is
closed"*, not as a 50-line ε-δ argument. With `registry=None` the
rule is a no-op, a safety hatch, not a default.

## `Antisymmetric`, bivectors with a sign rule

`Antisymmetric()` flags a binary head whose `MultiEval(head, α, β)`
swap-pair canonicalises to ``−head(β, α)``. Typical use: a Schouten–
Nijenhuis bivector ``π`` whose pairing `π(α, β)` should sort
arguments by `repr` order with a sign.

```python
from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Antisymmetric

reg2 = PropertyRegistry()
pi    = Symbol("π"); reg2.declare(pi, Antisymmetric())
alpha = Symbol("α"); reg2.declare(alpha, Graded(degree=1))
beta  = Symbol("β"); reg2.declare(beta,  Graded(degree=1))

engine2 = ExpansionEngine([RegistryAntiSymCanonicalDefinition(registry=reg2)])

raw = MultiEval(pi, beta, alpha)      # out of canonical order
expanded, steps = engine2.expand(raw)
print(f"input    : {raw}")
print(f"expanded : {expanded}")
print(f"rule     : {steps[0].rule}")
```

The rule fires only once per node (each fire sorts the only
out-of-order pair) and only when `repr(args[0]) > repr(args[1])`
, that termination guarantee is what makes it safe to bundle into
larger engines.

The cancellation pattern follows immediately: `π(α, β) + π(β, α)`
collapses to zero when this rule meets the simplify pipeline.

```python
from jacopy.algorithms.simplify import simplify

eq = MultiEval(pi, alpha, beta) + MultiEval(pi, beta, alpha)
expanded, _ = engine2.expand(eq)
print(f"π(α,β) + π(β,α) → {simplify(expanded, reg2)}")
```

## `NonDegenerate`, peeling `ι_(·) ω` off both sides

`NonDegenerate()` encodes the bundle map ``X ↦ ι_X ω`` being
injective. The paired rule fires on a two-term `Sum` whose children
are interior products of the *same* form against vector fields with
opposite signs, exactly the obstruction shape a vector-field
equality produces.

```python
from jacopy.calculus.interior import interior
from jacopy.calculus.nondegenerate_axioms import (
    NonDegenerateInteriorEqualityDefinition,
)
from jacopy.core.properties import NonDegenerate

reg3 = PropertyRegistry()
omega = Symbol("ω")
reg3.declare(omega, Graded(degree=2))
reg3.declare(omega, NonDegenerate())

Y = Symbol("Y"); reg3.declare(Y, Graded(degree=1))
Z = Symbol("Z"); reg3.declare(Z, Graded(degree=1))

engine3 = ExpansionEngine(
    [NonDegenerateInteriorEqualityDefinition(registry=reg3)]
)

obstruction = Act(interior(Y), omega) - Act(interior(Z), omega)
expanded, steps = engine3.expand(obstruction)
print(f"input    : {obstruction}")
print(f"expanded : {expanded}")
print(f"rule     : {steps[0].rule}")
```

Mathematically: `ω` non-degenerate ⇔ the linear map `X ↦ ι_X ω` is
injective ⇒ `ι_Y ω = ι_Z ω ⇒ Y = Z`. Implementation: the rule peels
the common `ι_(·) ω` shell off the difference, leaving the
vector-field difference for the rest of the engine to reduce.

## Why declarative beats inline

Each closure rule is registry-aware: at construction it stores a
reference to a `PropertyRegistry`, and at `matches` time it queries
that registry for the relevant flag. A few practical consequences:

1. **One rule, every form.** A single `ClosedFormDefinition(registry=reg)`
   handles every form declared `Closed()` in `reg`, no per-form
   `Definition` subclass to spell out.
2. **Pre-declaring opts out.** If you already declared `Graded` /
   `Closed` / `Antisymmetric` on a symbol, the problem wrappers
   (tutorial 14) see those flags and don't re-declare. Pre-declaring
   is the override mechanism.
3. **`registry=None` is a no-op.** Engines built without a registry
   keep these rules dormant, a useful safety default for low-level
   engines that don't want to consume registry context.

## When the wrapper does it for you

`SymplecticProblem(ω, [...])` and `KoszulProblem(π, [...])` already
auto-declare their structural axioms on the registry and bake the
matching rules into their engines:

| Wrapper | Auto-declares | Wires |
|---|---|---|
| `SymplecticProblem` | `Closed(ω)`, `NonDegenerate(ω)` | both rules on `prob.engine` |
| `KoszulProblem` | `Antisymmetric(π)` | the antisym rule + tilde calculus |

Tutorial 14 walks those wrappers end-to-end. The point of *this*
tutorial is the layer underneath: when the wrapper isn't a fit (you
have an almost-symplectic form, a metric, a Riemannian volume…), the
same property-declaration recipe still works, declare the flag,
layer the rule onto your engine, and the rewrite fires.

## Closure axioms beyond the form layer

The same registry-property pattern scales:

* `Poisson` flags ``[π, π]_SN = 0`` and is consumed by the tilde-d²
  axiom (`d̃² V → 0` whenever ``π`` is registered Poisson).
* The `Antisymmetric` flag also drives the SN bivector signs in the
  Koszul-bracket expansion rule.
* Faz 13's `LieBracketVfAntiSymmetryDefinition` and
  `LieBracketVfJacobiDefinition` follow the same template, flagged
  on the bracket head, rule consumes the flag.

Each "axiom flag" is a one-bit declarative truth about an algebraic
object that an engine rule cashes in as a rewrite primitive, the
opposite of the `Definition`-per-instance pattern that dominates
imperative proof code.

## Summary

* Three closure properties: `Closed` (`dω = 0`), `Antisymmetric`
  (`π(β,α) → −π(α,β)`), `NonDegenerate` (`ι_Y ω = ι_Z ω ⇒ Y = Z`).
* Each property is a flag on the registry; each pairs with a single
  registry-aware engine rule that fires on its target shape.
* Rules are constructed with `registry=` keyword; `registry=None` is
  a no-op default.
* Pre-declaring on the registry is the opt-out mechanism for the
  problem-wrapper layer (tutorial 14).
* The same flag-plus-rule recipe scales to `Poisson`, the Lie-bracket
  antisymmetry / Jacobi axioms, and any future structural fact.
