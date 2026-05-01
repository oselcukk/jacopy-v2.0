# 17, The tilde calculus

The standard Cartan operators ``ι_X``, ``L_X``, ``d`` act on
*differential forms* and use *vector fields* as the indexing data.
On a Poisson manifold ``(M, π)`` they have a **dual** picture: a set
of operators ``ι̃_ω``, ``L̃_ω``, ``d̃`` that act on *multivector
fields* and use *forms* (and the bivector ``π``) as the indexing
data. The three are jointly called the **tilde calculus** and they
satisfy six Cartan-style identities that mirror the textbook
form-side relations.

This tutorial covers:

1. The three operator atoms (`tilde_interior`, `tilde_d`,
   `tilde_lie`) and the design choice that keeps them opaque.
2. The three defining-identity rewrites (`TildeIotaSwap`,
   `TildeExteriorDLichnerowicz`, `TildeLieMagic`).
3. Auxiliary axioms: ``ι̃² = 0``, ``d̃² = 0`` (under `Poisson`),
   ``d̃ f = −π^♯(df)``, the iota-as-scalar bridge.
4. `tilde_intrinsic_engine` + `prove_tilde_cartan_relation`,
   the entry point that closes the magic and anti-commute relations.
5. `K̃_η`, the tilde Cartan remainder, the polarity-flipped
   shortcut for the derived identities of §3.1.4.

## The duality

Pair-by-pair, the tilde operators are the Koszul-side mirrors of
the standard ones:

| Standard side (forms) | Tilde side (multivectors) |
|---|---|
| ``ι_X ω`` (contract X into ω) | ``ι̃_ω V := ι_V ω`` (contract V into ω, indexed by ω) |
| ``d ω`` (exterior derivative) | ``d̃ V := [π, V]_SN`` (Lichnerowicz differential) |
| ``L_X ω`` (Lie derivative) | ``L̃_ω V := d̃ ι̃_ω V + ι̃_ω d̃ V`` (Cartan magic) |

The reason this is more than a typographical trick: ``π^♯ : T^*M → TM``
turns a form into a vector field, so a "covector-indexed operator on
multivectors" is the natural object on a Poisson manifold. The six
Cartan identities, anti-commute, ``d̃² = 0``, magic, two
``L̃``-bracket commutators, and the ``[d̃, ι̃]`` bridge, encode
exactly the same content as the form-side ones.

## The three atoms

Like `LieDerivative` on the form side, the three tilde operators are
opaque `Derivation` subclasses. Their identity is structural over
their indexing data (form, bivector), not a literal expansion.

```python
from jacopy.calculus.tilde import tilde_interior, tilde_d, tilde_lie
from jacopy.core.expr import Symbol

omega = Symbol("ω")
pi    = Symbol("π")

i_til = tilde_interior(omega)     # ι̃_ω, degree -1
d_til = tilde_d(pi)               # d̃,    degree +1
L_til = tilde_lie(omega, pi)      # L̃_ω,  degree  0

print(f"{i_til}: degree {i_til._degree}")
print(f"{d_til}: degree {d_til._degree}")
print(f"{L_til}: degree {L_til._degree}")
```

The `_degree` matches the multivector grading shift: ``ι̃_ω`` lowers
multivector degree by 1, ``d̃`` raises it by 1, ``L̃_ω`` preserves
it. `tilde_d(pi)` carries `pi` so two distinct bivectors yield
distinct ``d̃`` atoms, useful when proving an identity that
references both ``π`` and a deformation ``π'``.

## The three defining-identity rewrites

`jacopy.calculus.tilde.axioms` carries the three rules that turn
the opaque atoms into their defining expansions.

### ``TildeIotaSwapDefinition``, the swap

The cleanest of the three: ``ι̃_ω V → ι_V ω``. It's just a notation
swap, same numeric content, different indexing.

```python
from jacopy.algebra.derivation import Act
from jacopy.calculus.tilde import TildeIotaSwapDefinition
from jacopy.proof.expansion import ExpansionEngine

V = Symbol("V")
engine = ExpansionEngine([TildeIotaSwapDefinition()])
expr = Act(i_til, V)
out, steps = engine.expand(expr)
print(f"{expr} → {out}")
print(f"rule  : {steps[0].rule}")
```

Note `Act(ι_V, ω)` on the RHS, the form-side `InteriorProduct(V)`
applied to ``ω``. Downstream form-side rules (Cartan magic,
intrinsic engine) take it from there.

### ``TildeExteriorDLichnerowiczDefinition``, Lichnerowicz

The Lichnerowicz differential ``d̃ V := [π, V]_SN``: this is what
*makes* the tilde calculus dual to the form-side calculus, and it's
why ``π`` enters the picture at all.

```python
from jacopy.calculus.tilde import TildeExteriorDLichnerowiczDefinition

engine = ExpansionEngine([TildeExteriorDLichnerowiczDefinition(pi)])
expr = Act(d_til, V)
out, steps = engine.expand(expr)
print(f"{expr} → {out}")
print(f"rule  : {steps[0].rule}")
```

The RHS is a `BracketApply(sn, π, V)` node, the inert SN-bracket
handle covered in tutorial 12. From there, SN base cases or the
Derived Bracket Theorem can close subsequent steps.

### ``TildeLieMagicDefinition``, Cartan magic

The magic formula on the tilde side: ``L̃_ω V → d̃ ι̃_ω V + ι̃_ω d̃ V``.
This is the *defining* identity, not a derived one, in the engine,
the tilde Lie operator is opaque until this rule fires.

```python
from jacopy.calculus.tilde import TildeLieMagicDefinition

engine = ExpansionEngine([TildeLieMagicDefinition(pi)])
expr = Act(L_til, V)
out, steps = engine.expand(expr)
print(f"{expr} → {out}")
print(f"rule  : {steps[0].rule}")
```

After this rule fires, the right-hand side has two ``d̃`` /
``ι̃`` chains that downstream rules can flatten.

## Auxiliary axioms

`jacopy.calculus.tilde.aux_axioms` carries five auxiliary rules
that handle special cases the three defining rules don't reach by
themselves:

| Rule | Folds |
|---|---|
| `TildeIotaOnZeroVectorDefinition` | ``ι̃_ω f → 0`` for ``f`` of degree 0 |
| `TildeIotaSquaredZeroDefinition` | ``ι̃_ω(ι̃_ω V) → 0`` |
| `TildeLieOnZeroVectorDefinition` | ``L̃_ω f → π^♯(ω)(f)`` |
| `TildeDOfFunctionDefinition` | ``d̃ f → −π^♯(df)`` |
| `TildeDSquaredPoissonDefinition` | ``d̃² V → 0`` when ``π`` is `Poisson` |

The `Poisson` flag matters: ``d̃² = 0`` is the **Jacobi identity for π**
in dual form (``[π, π]_SN = 0``). Without the flag, the engine
keeps ``d̃² V`` opaque.

```python
from jacopy.calculus.tilde import TildeDSquaredPoissonDefinition
from jacopy.core.properties import Poisson
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
reg.declare(pi, Poisson())

engine = ExpansionEngine([TildeDSquaredPoissonDefinition(pi, registry=reg)])
expr = Act(d_til, Act(d_til, V))
out, steps = engine.expand(expr)
print(f"{expr} → {out}")
print(f"rule  : {steps[0].rule}")
```

The same `Poisson(π)` declaration unlocks the SN-bracket Jacobi
chain (tutorial 12) and the d̃² closure here, a single registry
flag, two engine rules consume it.

## `tilde_intrinsic_engine` and the Cartan relations

`tilde_intrinsic_engine(pi, koszul, *, registry=...)` bundles all
the rules above plus standard-side `MultiEval` helpers and
`Sharp` / `Pairing` plumbing into a single engine. Pair it with
`prove_tilde_cartan_relation(lhs, rhs, *, etas, engine, registry)`
and the magic formula closes mechanically.

```python
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.musical import Sharp
from jacopy.calculus.tilde import (
    tilde_intrinsic_engine, prove_tilde_cartan_relation,
)
from jacopy.core.expr import Sum
from jacopy.core.properties import Graded

reg = PropertyRegistry()
reg.declare(pi,    Graded(degree=1)); reg.declare(pi, Poisson())
reg.declare(omega, Graded(degree=1))
reg.declare(V,     Graded(degree=1))
eta = Symbol("η"); reg.declare(eta, Graded(degree=1))

sharp  = Sharp(pi)
koszul = KoszulBracket(sharp)
eng    = tilde_intrinsic_engine(pi, koszul, sharp=sharp, registry=reg)

lhs = Act(L_til, V)
rhs = Sum(Act(d_til, Act(i_til, V)), Act(i_til, Act(d_til, V)))

chain = prove_tilde_cartan_relation(
    lhs, rhs, etas=(eta,), engine=eng, registry=reg,
)
print(f"L̃_ω = d̃ ι̃_ω + ι̃_ω d̃ closes in {len(chain)} steps")
```

Both sides are wrapped in `MultiEval(·, η, slot_kind="covector")`,
that's how the prover routes the engine to the *tilde* intrinsic
rules instead of the form-side ones, so the same engine can carry
both kinds of MultiEval node without aliasing.

The anti-commute relation ``ι̃_ω(ι̃_μ V) + ι̃_μ(ι̃_ω V) = 0`` (relation
1 in §3.1.3) closes through the same machinery:

```python
from jacopy.core.expr import Integer

mu = Symbol("μ");  reg.declare(mu, Graded(degree=1))
W  = Symbol("W");  reg.declare(W,  Graded(degree=2))
i_mu = tilde_interior(mu)

lhs = Sum(Act(i_til, Act(i_mu, W)), Act(i_mu, Act(i_til, W)))
chain = prove_tilde_cartan_relation(
    lhs, Integer(0), etas=(eta,), engine=eng, registry=reg,
)
print(f"ι̃ anti-commute closes in {len(chain)} steps")
```

The two non-trivial relations not closed by the bare bundle,
``[L̃_α, L̃_β] = L̃_{[α,β]_K}`` and ``[L̃_α, ι̃_β] = ι̃_{[α,β]_K}``
, need the Koszul-bracket expansion rule layered on top
(`KoszulProblem.tilde_intrinsic_engine` adds it). They're outside
the scope of this tutorial; reach for the `KoszulProblem` wrapper
when you need them.

## `K̃_η`, the tilde Cartan remainder

`K̃_η := −L̃_η + d̃ ∘ ι̃_η` is the polarity-flipped form of the
magic formula: it's what survives when you commute the ``L̃`` and
``d̃ ι̃`` halves. The atom is inert; the rule
`TildeCartanRemainderDefinition` realises the defining expansion.

```python
from jacopy.calculus.cartan_remainder_axioms import (
    TildeCartanRemainderDefinition,
)
from jacopy.calculus.tilde import K_tilde

K = K_tilde(eta, pi)            # K̃_η, degree 0
print(f"{K}: degree {K._degree}, form {K.form}, bivector {K.bivector}")

engine = ExpansionEngine([TildeCartanRemainderDefinition()])
expr = Act(K, V)
out, steps = engine.expand(expr)
print(f"{expr} → {out}")
print(f"rule  : {steps[0].rule}")
```

When does this matter? The §3.1.4 derived identities involve
``L̃ − d̃ ∘ ι̃`` chains where treating ``K̃`` as a single named
operator collapses the bookkeeping. Two ``K̃`` instances on
different bivectors stay distinct (the atom keys on
``(name, degree, form, bivector)``), so a deformation argument
``π → π + ε σ`` can carry both ``K̃_η`` operators side-by-side.

## When the tilde engine is the right pick

Reach for the tilde calculus when:

* you're on a Poisson manifold and the proof references ``[π, V]_SN``
  / Lichnerowicz / Koszul-bracket-on-1-forms;
* the goal involves multivector evaluation against a tuple of
  forms (covector-slot `MultiEval`);
* a §3.1.3 / §3.1.4 derived identity (the dual Cartan relations)
  is what you're trying to close.

Don't use it for:

* a form-side Cartan calculation, `intrinsic_engine` (tutorial
  15) handles that, and the slot-kind discipline keeps the two
  pictures separate;
* a generic non-Poisson bivector, without the `Poisson` flag,
  ``d̃² V`` doesn't collapse and most §3.1.3 derivations stall;
* SN-bracket-only work, `[π, V]_SN` directly through `sn.expand`
  is shorter than routing it through ``d̃``.

## Summary

* The tilde calculus is the Cartan calculus dualised by ``π^♯``:
  three operator atoms (`tilde_interior`, `tilde_d`, `tilde_lie`)
  acting on multivectors, indexed by forms and ``π``.
* Three defining rewrites (`TildeIotaSwap`,
  `TildeExteriorDLichnerowicz`, `TildeLieMagic`) plus five
  auxiliary rules close the engine-level machinery.
* `tilde_intrinsic_engine(pi, koszul, …)` +
  `prove_tilde_cartan_relation(lhs, rhs, *, etas=…, engine=…)`
  closes the magic formula in 7 steps and the anti-commute
  relation in 12. The slot-kind discipline (``covector``) keeps
  the tilde engine from aliasing the form-side intrinsic engine.
* The `Poisson` registry flag unlocks ``d̃² V → 0`` in 4 engine
  steps, same flag drives the SN-bracket Jacobi chain.
* `K̃_η` and `TildeCartanRemainderDefinition` are the
  polarity-flipped shortcut for §3.1.4 derived identities; the
  remainder atom keys on ``(form, bivector)`` so multiple
  ``K̃`` operators coexist without aliasing.
