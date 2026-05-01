# 19, The Courant family: Dorfman, Courant, Dirac

The Courant algebroid lives on ``TM ⊕ T*M``, vector fields and
1-forms paired together. It carries:

* the **Courant bracket** ``[·, ·]_C``, graded-antisymmetric, Jacobi
  fails up to an exact term;
* the **Dorfman bracket** ``[·, ·]_D``, *not* antisymmetric, but
  Jacobi (Leibniz) holds exactly;
* an optional **H-twist** by a closed 3-form ``H`` modifying the form
  half;
* **Dirac subbundles** ``L ⊂ TM ⊕ T*M``, maximally isotropic,
  involutive, generalising both Poisson and presymplectic structures.

The two brackets are *the same machinery* viewed two ways: same
Cartan operators, same Lie bracket on vectors, different combination
on the form half. The Courant–Dorfman bridge identity is the precise
statement of how they differ.

This tutorial covers:

1. `SectionPair`, operands as ``(X, α)`` pairs.
2. `CourantAlgebroid`, both brackets, the H-twist, and the bridge.
3. `prove_courant_dorfman_bridge` and `prove_jacobi_reduction`, the
   two seeded theorems.
4. `DiracStructure`, pairing, isotropy, involutivity.
5. `poisson_dirac` and `presymplectic_dirac`, the canonical Dirac
   structures of the two source geometries.

## Section pairs as operands

`SectionPair(vector, form)` is the `Expr` wrapper for an element of
``Γ(TM ⊕ T*M)``. Both brackets consume `SectionPair`s and produce a
`SectionPair`; you read off the vector / form halves through the
`.vector` and `.form` accessors.

```python
from jacopy.brackets.dorfman import SectionPair
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
X = Symbol("X");  reg.declare(X, Graded(degree=0))
Y = Symbol("Y");  reg.declare(Y, Graded(degree=0))
alpha = Symbol("α"); reg.declare(alpha, Graded(degree=1))
beta  = Symbol("β"); reg.declare(beta,  Graded(degree=1))

a = SectionPair(X, alpha)
b = SectionPair(Y, beta)
print(f"a = ({a.vector}, {a.form})")
print(f"b = ({b.vector}, {b.form})")
```

## `CourantAlgebroid`, both brackets at once

Construct with no arguments for the standard untwisted algebroid;
pass `background_H=H` to twist by a closed 3-form ``H``. Both
brackets are exposed and use the **same** Cartan operators, that
sharing is what makes the Courant–Dorfman bridge identity exact
rather than approximate.

```python
from jacopy.library.courant_algebroid import CourantAlgebroid

C = CourantAlgebroid()
print(f"name           : {C.name}")
print(f"twisted        : {C.is_twisted}")

dorf = C.expand_dorfman(a, b, registry=reg)
cour = C.expand(a, b,         registry=reg)
print(f"Dorfman form half : {dorf.form}")
print(f"Courant form half : {cour.form}")
```

Both vector halves are ``[X, Y]`` (same `vector_bracket`); the form
halves differ:

* Dorfman: ``L_X β − ι_Y dα`` (Leibniz, no symmetrising correction)
* Courant: ``L_X β − L_Y α − ½ d(ι_X β − ι_Y α)`` (graded-antisymmetric)

## The bridge identity

`prove_courant_dorfman_bridge(a, b)` asserts that the difference
collapses to a single exact correction:

```
[a, b]_D − [a, b]_C = (0, ½ d(ι_X β + ι_Y α))
```

The proof is a single `theorem`-tagged step, the algebraic identity
*is* the theorem. Cartan's magic formula
``L_Y α = d(ι_Y α) + ι_Y(dα)`` is what makes the cancellation work.

```python
chain = C.prove_courant_dorfman_bridge(a, b, registry=reg)
step = chain.steps[0]
print(f"steps  : {len(chain)}")
print(f"rule   : {step.rule}")
print(f"tag    : {step.provenance_tag}")

correction = C.bridge_correction(a, b)
print(f"correction.vector : {correction.vector}")
print(f"correction.form   : {correction.form}")
```

`bridge_correction` builds the canonical RHS as an explicit
`SectionPair(0, ½ d(ι_X β + ι_Y α))` so callers can inspect or
re-cite it without re-running the proof.

## H-twisted Jacobi

The untwisted Courant algebroid satisfies Jacobi exactly, the
obstruction is the literal ``0``. Adding an H-twist puts the
obstruction at ``dH``: Jacobi closes iff ``dH = 0``.

```python
H = Symbol("H"); reg.declare(H, Graded(degree=3))
C_H = CourantAlgebroid(background_H=H)
print(f"C_H twisted    : {C_H.is_twisted}")
print(f"name           : {C_H.name}")

chain = C_H.prove_jacobi_reduction(registry=reg)
step = chain.steps[0]
print(f"rule           : {step.rule}")
print(f"justification  : {step.justification}")
```

Both `prove_jacobi_reduction` variants emit a single
`axiom`-tagged step. The untwisted case maps the literal ``0``
obstruction to itself (vacuous Jacobi); the twisted case lands on
``dH``, leaving the caller to discharge ``dH = 0`` separately
(typically by declaring `Closed(H)` on the registry, see tutorial
13).

## Dirac structures

A `DiracStructure` pins a maximally-isotropic involutive subbundle
``L ⊂ TM ⊕ T*M``. The wrapper carries the ambient
`CourantAlgebroid` and a symbolic name for ``L``; it does **not**
model section membership ``a ∈ Γ(L)`` symbolically, there's no
predicate algebra for that. What it does model is the two defining
properties as **axiom**-tagged proof steps.

```python
from jacopy.library.dirac import DiracStructure

L_sym = Symbol("L")
D     = DiracStructure(C, L_sym)
print(f"Dirac          : {D.name}")
```

### Pairing, isotropy

The canonical pairing on ``TM ⊕ T*M`` is
``⟨a, b⟩ = ½(ι_X β + ι_Y α)``. Its diagonal
``⟨a, a⟩ = ι_X α`` is the single-section isotropy obstruction; full
bilinear isotropy follows by polarisation.

```python
print(f"⟨a, b⟩         : {D.pairing(a, b)}")
print(f"⟨a, a⟩         : {D.isotropy_obstruction(a)}")

chain = D.prove_isotropy(a)
print(f"isotropy proof : {len(chain)} step, rule={chain.steps[0].rule}")
```

### Involutivity

``[a, b]_C ∈ Γ(L)`` is involutivity. The wrapper surfaces a
placeholder symbol for the subbundle-membership obstruction (since
membership isn't an `Expr`-level predicate) and discharges it
through the `DiracInvolutivityAxiom` step.

```python
chain = D.prove_involutivity(a, b)
print(f"involutivity   : {len(chain)} step, rule={chain.steps[0].rule}")
```

Both `prove_isotropy` and `prove_involutivity` emit single
`axiom`-tagged steps, the citation form. Callers that want to
expand the pairing arithmetically should reach for `pairing` and
`isotropy_obstruction` directly and run the engine themselves.

## The two canonical Dirac structures

`poisson_dirac(π)` and `presymplectic_dirac(ω)` are factory
shortcuts for the two source geometries that *every* discussion of
Dirac structures starts with:

| Factory | Subbundle | Source |
|---|---|---|
| `poisson_dirac(π)` | ``L_π = {(π^♯ α, α)}`` | Poisson bivector |
| `presymplectic_dirac(ω)` | ``L_ω = {(X, ω^♭ X)}`` | closed 2-form |

```python
from jacopy.library.dirac import poisson_dirac, presymplectic_dirac

pi = Symbol("π"); reg.declare(pi, Graded(degree=1))
omega = Symbol("ω"); reg.declare(omega, Graded(degree=2))

L_pi    = poisson_dirac(pi,        courant=C)
L_omega = presymplectic_dirac(omega, courant=C)

print(f"poisson_dirac        : {L_pi.name}")
print(f"presymplectic_dirac  : {L_omega.name}")
```

Each factory only records the subbundle name (``L_π`` /
``L_ω``); the isotropy and involutivity axioms remain axioms on
the resulting `DiracStructure`. *Proving* "``dω = 0`` ⇒ ``L_ω`` is
Dirac" or "``[π, π]_SN = 0`` ⇒ ``L_π`` is Dirac" is a separate
theorem, both inherit the axiom-step proofs unchanged from the
parent class.

## When to use what

| If you want… | Reach for… |
|---|---|
| Compare the two brackets on the same operands | `expand` + `expand_dorfman` |
| Cite the bridge identity in a chain | `prove_courant_dorfman_bridge` |
| H-twisted Jacobi reduction | `prove_jacobi_reduction` (with `background_H`) |
| Dirac isotropy / involutivity citations | `DiracStructure.prove_isotropy` / `prove_involutivity` |
| The pairing as an `Expr` | `DiracStructure.pairing` |
| Poisson / presymplectic special cases | `poisson_dirac` / `presymplectic_dirac` |

The Courant family is mostly *citation-shaped* in the engine layer
, the deep algebraic content lives in the seeded theorems
(`courant_jacobi_twist`, `courant_dorfman_bridge`,
`dirac_isotropy`, `dirac_involutivity`), not in step-by-step
rewrites. That's a deliberate choice: spelling these out in the
engine would dwarf the rest of the library, and the textbook
proofs are the proofs you want anyway.

## Summary

* `SectionPair(X, α)` is the operand for both Courant and Dorfman
  brackets; `.vector` / `.form` accessors retrieve the halves.
* `CourantAlgebroid` carries both brackets on shared Cartan
  operators. `expand` is Courant, `expand_dorfman` is Dorfman; both
  vector halves are the same Lie bracket, form halves differ.
* `prove_courant_dorfman_bridge` cites the exact correction
  ``(0, ½ d(ι_X β + ι_Y α))`` as a single `theorem`-tagged step.
* `prove_jacobi_reduction` is vacuous when untwisted; H-twisted it
  lands the obstruction on ``dH`` (closes iff ``dH = 0``).
* `DiracStructure(courant, L)` carries pairing, isotropy obstruction
  ``ι_X α``, and the involutivity placeholder; both defining
  properties are `axiom`-tagged proof steps.
* `poisson_dirac(π)` and `presymplectic_dirac(ω)` are factory
  shortcuts for the two source geometries; they inherit isotropy /
  involutivity as axioms (the conditional "if ``π`` Poisson / ``ω``
  closed" theorems are out of scope here).
