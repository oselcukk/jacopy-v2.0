# 18, Derivator identities (§3.1.5)

The six identities of §3.1.5 are the **derivator-shaped** dual of the
three Cartan-derived identities of §3.1.4. Where §3.1.4 asks "what is
``[L_X, ι_Y]``?" and answers with a Cartan-side formula, §3.1.5 asks
"how badly does ``L_X`` *fail* to be a derivation of the Koszul
bracket?" and answers with `K_V` / `K̃_η` Cartan-remainder corrections.

For a degree-``d`` operator ``φ`` and a graded bracket
``[·, ·]_E`` the **derivator** is

```
D^E_φ(u, v) := φ[u, v]_E − [φu, v]_E − (−1)^{d|u|} [u, φv]_E.
```

``φ`` is a derivation of the bracket iff ``D^E_φ ≡ 0``. The §3.1.5
identities make precise *what* that derivator equals when ``φ`` is
``L_V`` or ``L̃_η`` and the bracket is Koszul or SN, and the
right-hand sides are always built from the Cartan remainders ``K`` and
``K̃``.

This tutorial covers:

1. The `derivator(...)` builder and the §3.1.5 form-side / dual
   layout.
2. `KoszulProblem.derivator_form_engine()` /
   `derivator_multivector_engine()`, the two pre-bundled engines.
3. `prove_derivator(lhs, rhs, *, eval_args, side)`, the entry point
   that closes all six identities in one call.
4. The six identities and their step counts.

## The setup, `KoszulProblem` as the entry point

`KoszulProblem` is the workhorse for everything in §3.1.5. It wraps
``π``, the form inventory, and the multivector inventory, and exposes
the two derivator engines pre-bundled with every rule needed to close
the six identities, Cartan remainders, tilde calculus, Koszul
bracket expansion, the lot.

```python
from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.brackets.schouten import sn as default_sn
from jacopy.calculus.cartan_remainder import K
from jacopy.calculus.derivator import derivator
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.tilde import K_tilde, tilde_d, tilde_interior, tilde_lie
from jacopy.core.expr import Neg, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.koszul_problem import KoszulProblem

reg = PropertyRegistry()
pi = Symbol("π")
omega, eta, mu = (Symbol(s) for s in ("ω", "η", "μ"))
U, V, W = (Symbol(s) for s in ("U", "V", "W"))
for f in (omega, eta, mu, U, V, W):
    reg.declare(f, Graded(degree=1))

Y, Z = Derivation("Y", 0), Derivation("Z", 0)
xi = Symbol("ξ"); reg.declare(xi, Graded(degree=1))

prob = KoszulProblem(
    pi, (omega, eta, mu),
    registry=reg,
    multivectors=((U, 1), (V, 1), (W, 1)),
)
prob.assume_poisson()
K_b = prob.koszul_bracket

print(f"form engine        : {len(prob.derivator_form_engine().definitions)} rules")
print(f"multivector engine : {len(prob.derivator_multivector_engine().definitions)} rules")
```

`assume_poisson()` flags ``π`` as `Poisson` on the registry, that's
what unlocks ``d̃² V → 0`` and the SN-bracket Jacobi step inside the
multivector engine.

## Form-side identity (1), the canonical example

The form-side derivator obstruction is

```
D^{T*M}_{L_U}(η, μ) = L_U[η, μ]_K − [L_U η, μ]_K − [η, L_U μ]_K
```

, how badly ``L_U`` fails to be a derivation of ``[·, ·]_K``. The
§3.1.5 (1) identity says this equals the sum of two
Cartan-remainder corrections:

```
D^{T*M}_{L_U}(η, μ) = L_{K̃_η U} μ + K_{K̃_μ U} η
```

```python
lhs = derivator(lie_derivative(U), K_b, eta, mu)

K_tilde_eta_U = Act(K_tilde(eta, pi), U)
K_tilde_mu_U  = Act(K_tilde(mu, pi),  U)
rhs = Sum(
    Act(lie_derivative(K_tilde_eta_U), mu),
    Act(K(K_tilde_mu_U),               eta),
)

chain = prob.prove_derivator(lhs, rhs, eval_args=(Y,), side="form")
print(f"(1) form-side closes in {len(chain)} steps")
```

The 109 steps cover: Koszul-bracket expansion on three nested calls,
operator-commutator folds, intrinsic Cartan formulas through both
``L_X`` and ``d`` on a 1-form, ``K̃_η U`` polarity-flipping into
``−L̃_η U + d̃ ι̃_η U``, and final cancellation. None of those
intermediate residues need user attention, `prove_derivator` runs
the engine to fix-point.

## Dual multivector-side identity (1')

The dual swaps every operator with its tilde and every form with a
multivector:

```
D̃^{SN}_{L̃_η}(U, V) = L̃_{K_U η} V + K̃_{K_V η} U
```

```python
lhs = derivator(tilde_lie(eta, pi), default_sn, U, V)

K_U_eta = Act(K(U), eta)
K_V_eta = Act(K(V), eta)
rhs = Sum(
    Act(tilde_lie(K_U_eta, pi), V),
    Act(K_tilde(K_V_eta, pi),   U),
)

chain = prob.prove_derivator(
    lhs, rhs, eval_args=(xi,), side="multivector",
)
print(f"(1') multivector-side closes in {len(chain)} steps")
```

The `side="multivector"` switch routes through
`derivator_multivector_engine` and uses `slot_kind="covector"` for
the `MultiEval` wrap, the same routing discipline as
`prove_tilde_cartan_relation` (tutorial 17).

## The full table

The six identities and their step counts under
`prob.prove_derivator(...)`:

| # | Identity | side | steps |
|---|---|---|---|
| (1) | ``D^{T*M}_{L_U}(η, μ) = L_{K̃_η U} μ + K_{K̃_μ U} η`` | form | 109 |
| (2) | ``L_{d̃ ι̃_η U} μ = −[d ι_U η, μ]_K`` | form | 21 |
| (3) | ``0 = d ι_{L̃_ω W} η − d ι_{d̃ ι̃_η W} ω + d ι_W [ω, η]_K`` | form | 30 |
| (1') | ``D̃^{SN}_{L̃_η}(U, V) = L̃_{K_U η} V + K̃_{K_V η} U`` | multivec | 117 |
| (2') | ``L̃_{d ι_U η} V = −[d̃ ι̃_η U, V]_SN`` | multivec | 25 |
| (3') | ``0 = d̃ ι̃_{L_U η} V − d̃ ι̃_{d ι_V η} U + d̃ ι̃_η [U, V]_SN`` | multivec | 23 |

The 109/117 step counts on (1)/(1') reflect how much expansion the
Koszul / SN bracket carries, three nested bracket calls each unfold
into a Cartan / Lichnerowicz formula. (2)/(2') and (3)/(3') are
**leaner** because they reach the bracket-of-derivation form
directly, skipping the cyclic structure of (1)/(1').

## Why declarative beats inline

What `KoszulProblem` saves you from:

* picking the right engine for each side (`form` vs `multivector`)
  with the right axioms layered on;
* threading the registry, ``π``, and the Koszul bracket through every
  rule constructor;
* canonicalising operator-atom index slots before the `MultiEval`
  wrap (the `canonicalize_indices` pre-pass is automatic, see the
  *operator-atom index opacity* memo for why a generic engine walk
  doesn't reach into operator indices);
* matching the slot kind (`vector` / `covector`) to the side.

Without the wrapper, each identity is a 30-line setup of imports,
engine assembly, and pre-pass plumbing. With it, the proof body
collapses to "form the LHS, form the RHS, call `prove_derivator`".

## When you'd reach past the wrapper

You'd skip `KoszulProblem.prove_derivator` if:

* you need a custom rule layered on (use
  `KoszulProblem.derivator_form_engine()` to get a *fresh* engine
  and append to its `definitions`, then call
  `prove_derivator_identity` directly with it);
* the bracket isn't Koszul or SN (custom bracket, define your own
  engine factory mirroring the `derivator_form_engine` pattern);
* you're proving an identity *outside* §3.1.5's derivator shape
  (Cartan magic, ``d² = 0``, ``[L_X, L_Y] = L_{[X,Y]_VF}``, those
  are tutorial 15's `intrinsic_engine`, not this one).

`prove_derivator_identity(lhs, rhs, *, engine, eval_args, slot_kind)`
in `jacopy.calculus.derivator` is the engine-level entry point if
you've assembled your own engine, same role as
`prove_intrinsic_equivalence` for the standard intrinsic engine.

## Summary

* **Derivator** ``D^E_φ(u, v) := φ[u, v]_E − [φu, v]_E − [u, φv]_E``
  measures the failure of ``φ`` to be a graded derivation of
  ``[·, ·]_E``. ``φ`` is a derivation iff ``D^E_φ ≡ 0``.
* The six §3.1.5 identities equate Cartan / SN derivators to sums of
  Cartan-remainder corrections (`K_V` / `K̃_η`). Three live on the
  form side, three on the dual multivector side.
* `KoszulProblem.prove_derivator(lhs, rhs, *, eval_args, side)` closes
  all six in one call, `side="form"` (default) for (1)/(2)/(3),
  `side="multivector"` for (1')/(2')/(3').
* Step counts: 109/21/30 (form) and 117/25/23 (multivector). The
  cyclic (1)/(1') are the heaviest because three nested bracket
  expansions feed into them; (2)/(2') and (3)/(3') skip that cycle.
* `prove_derivator_identity` in `jacopy.calculus.derivator` is the
  engine-level entry point when you need a custom rule layered on
  the pre-bundled engine.
