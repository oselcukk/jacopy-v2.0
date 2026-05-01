# 23, Cartan structure equations

The two **Cartan structure equations** are the index-laden,
form-valued companions to the torsion / curvature definitions of
tutorial 20. They live on a connection ``∇`` and a local frame
``F``:

```
T^a   = de^a    + Σ_b ω^a_b ∧ e^b           (Cartan I)
R^a_b = dω^a_b  + Σ_c ω^a_c ∧ ω^c_b         (Cartan II)
```

where ``ω^a_b(∇)`` is the **connection 1-form**, ``T^a(∇)`` the
**torsion 2-form**, and ``R^a_b(∇)`` the **curvature 2-form**,
all bound to the choice of frame.

`CartanStructureProblem(∇, F)` is the wrapper that proves both
identities mechanically. It bundles 24 engine rules across seven
phases, torsion / curvature unfolding, connection axioms, frame
decomposition, indexed-sum machinery, wedge expansion, intrinsic
``d``, and frame duality. This tutorial walks the wrapper and the
two proofs.

## The connection-form atom

`ConnectionForm(∇, F, upper, lower)` is the inert 1-form atom
``ω^a_b(∇)`` parameterised by a connection, a frame, and two
indices. Its definition lives as the engine rule
`ConnectionFormDecompositionDefinition` (introduced in tutorial 22)
which links it to ``∇_V X_b → Σ_c ω^c_b(V) · X_c``.

```python
from jacopy.calculus.connection import AffineConnection
from jacopy.calculus.local_frame import LocalFrame

nabla = AffineConnection("∇")
F     = LocalFrame("F", dim=3)

from jacopy.calculus.cartan_forms import (
    ConnectionForm, TorsionForm, CurvatureForm,
)

omega_ab = ConnectionForm(nabla, F, "a", "b")
T_a      = TorsionForm   (nabla, F, "a")
R_ab     = CurvatureForm (nabla, F, "a", "b")

print(f"connection form : {omega_ab}")
print(f"torsion form    : {T_a}")
print(f"curvature form  : {R_ab}")
```

`TorsionForm` and `CurvatureForm` are the classical
``T^a = ⟨e^a, T(∇)(·, ·)⟩`` and
``R^a_b = ⟨e^a, R(∇)(·, ·) X_b⟩`` packaged as inert atoms. The
defining rules `TorsionFormDefinition` / `CurvatureFormDefinition`
in the engine open them up to the underlying `Torsion` /
`Curvature` from tutorial 20.

## `CartanStructureProblem`

```python
from jacopy.library.cartan_structure import CartanStructureProblem

prob = CartanStructureProblem(nabla, F)
print(f"name           : {prob.name}")
print(f"engine rules   : {len(prob.engine.definitions)}")
```

The 24-rule engine has a per-problem `PropertyRegistry` declaring
`FrameCovector` and `ConnectionForm` as degree 1, required because
the wedge alternating expansion and the arity-1 MultiEval-Pairing
bridge consult the registry before firing.

The wrapper exposes builders for the LHS and RHS of both
equations:

```python
from jacopy.algebra.derivation import Derivation

U, V = Derivation("U", 0), Derivation("V", 0)

lhs1 = prob.first_cartan_lhs(U, V, "a")
rhs1 = prob.first_cartan_rhs(U, V, "a")
print(f"Cartan I LHS : {lhs1}")
print(f"Cartan I RHS : {rhs1}")
```

The bound dummy ``b`` in `rhs1` is freshly minted on each call as a
bound `FrameIndex`, caller-supplied `upper_a` (`"a"`) is the only
free index in the result.

## `prove_first_cartan`, Cartan I

```
T^a(U, V) = (de^a)(U, V) + Σ_b (ω^a_b ∧ e^b)(U, V)
```

```python
res1 = prob.prove_first_cartan(U, V, "a")
print(f"Cartan I  : ok={res1.ok}, steps={len(res1.steps)}")
```

About 49 steps cover: torsion-form opening to ``⟨e^a, ∇_U V −
∇_V U − [U, V]⟩``, frame decomposition of ``V`` and ``U`` into
their basis expansions, Y-Leibniz on each connection-eval,
connection-form decomposition of ``∇_V X_b``, intrinsic-``d``
expansion of ``de^a``, wedge alternating expansion, and finally
Kronecker contractions and pairing duality collapsing the
``⟨e^a, X_b⟩`` matches.

The `_expand_to_canonical` loop runs `engine + simplify` to a
fix-point; `simplify` includes the `sort_product` pass that puts
`Product` factors in a canonical order, Cartan I residues need
this because the rewrites produce factors in different orders that
have to align before `Sum` cancellation can fire.

## `prove_second_cartan`, Cartan II

```
R^a_b(U, V) = (dω^a_b)(U, V) + Σ_c (ω^a_c ∧ ω^c_b)(U, V)
```

```python
res2 = prob.prove_second_cartan(U, V, "a", "b")
print(f"Cartan II : ok={res2.ok}, steps={len(res2.steps)}")
```

~54 steps. Same machinery: curvature-form opening, frame
decomposition, Y-Leibniz, connection-form decomposition,
intrinsic ``d`` on ``ω^a_b``, wedge expansion. The structural
similarity to Cartan I is intentional, both equations are
shadows of the same abstract identity.

## When to use `CartanStructureProblem`

The wrapper is your default entry point for any
**index-laden Cartan-style proof**. Reach for it when:

* you're proving a structure equation for a connection on a
  manifold with a chosen frame;
* you want the proof transcript to read in the textbook order
  (frame decomposition → Y-Leibniz → connection-form
  decomposition → wedge expansion → duality);
* you need both equations on the same `(∇, F)`, the wrapper's
  engine handles both without rebuilding.

Skip it when:

* you only need a coordinate-free identity (no frame in sight),
  reach for `BianchiProblem` (tutorial 20) or
  `prove_intrinsic_equivalence` (tutorial 15).
* the connection has a custom bracket (Q9 Koszul mode): the
  wrapper auto-detects this and swaps in
  `KoszulExteriorDIntrinsicDefinition`, no caller action needed,
  but the proof transcript is *longer* (anchor-pulled Cartan-d
  shapes) than the standard case.

## A note on the engine bundle

The 24-rule engine is **not** something you'd assemble by hand
casually. The order matters (definitions before linearity, frame
decomposition before duality, indexed-sum push-in before
distribution), and several rules' presence is non-obvious until
you see a residue stall without them. The wrapper is calibrated
on the Q7 / Q9 capstone examples; if you find a residue it can't
close, the fix is almost always a more-refined `Definition` for
the specific shape, not a hand-tuned engine.

## Summary

* Cartan structure equations:
  ``T^a = de^a + Σ_b ω^a_b ∧ e^b`` (I) and
  ``R^a_b = dω^a_b + Σ_c ω^a_c ∧ ω^c_b`` (II) on a connection +
  frame.
* `ConnectionForm`, `TorsionForm`, `CurvatureForm`, the three
  inert form atoms; their definitions live in the
  `cartan_forms` engine rules.
* `CartanStructureProblem(∇, F)`, 24-rule engine bundling
  torsion / curvature unfolding, connection axioms, frame
  decomposition, indexed-sum machinery, wedge expansion,
  intrinsic ``d``, frame duality.
* `prove_first_cartan` (~49 steps) and `prove_second_cartan`
  (~54 steps) close the two identities mechanically. The
  wrapper auto-detects custom-bracket connections (Q9 Koszul
  mode) and swaps in the anchor-pulled intrinsic-``d`` rule.
* Skip when the problem is coordinate-free (use
  `BianchiProblem` or `prove_intrinsic_equivalence`); reach for
  it whenever a frame and an index are in the picture.
