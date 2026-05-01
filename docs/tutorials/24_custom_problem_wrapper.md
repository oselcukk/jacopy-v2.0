# 24, Writing your own Problem wrapper

You've now seen six **Problem wrappers**: `SymplecticProblem`,
`KoszulProblem`, `BianchiProblem`, `KoszulConnectionProblem`,
`CartanStructureProblem`, and the §3.1.5 derivator engine inside
`KoszulProblem.prove_derivator`. Each one carries the same shape:

* **constructor** taking the geometric data + an optional registry;
* **auto-declaration** of structural axioms on the registry;
* **engine assembly**, pre-bundled `ExpansionEngine` carrying every
  rule needed for the wrapper's target proofs;
* **builder methods**, convenience constructors for the operators
  / brackets / forms specific to the wrapper;
* **prover methods**, `prove_*` entry points that close the
  wrapper's defining identities mechanically.

This tutorial walks the recipe. You'll write a wrapper for a
hypothetical *almost-symplectic* manifold (a non-degenerate but
*not* closed 2-form ``ω``), the exercise illustrates every
moving part without overlapping any existing wrapper.

This tutorial covers:

1. The five-step recipe.
2. A worked example, `AlmostSymplecticProblem(ω, X, Y)`.
3. Picking your axioms, when registry flags fit, when a
   `Definition` subclass is the right tool.
4. Assembling the engine, order, idempotency, loop avoidance.
5. Adding seeded theorems, when the proof is one citation step.

## The five-step recipe

| Step | What | Why |
|---|---|---|
| 1 | Pick the geometric data the wrapper carries | Frame the API |
| 2 | Auto-declare structural axioms on the registry | Closure flags fire when needed |
| 3 | Assemble the engine, register every rule | Single dispatch point |
| 4 | Write builder + prover methods | Match the textbook idiom |
| 5 | (optional) Add seeded theorems for citations | One-step proof shortcuts |

The wrappers in `library/` all follow this. Read
`library/symplectic.py` (~200 lines) end to end if you want a
template smaller than `KoszulProblem` (~1100 lines).

## Worked example, almost-symplectic manifold

An **almost-symplectic** form ``ω`` is non-degenerate but not
necessarily closed (``dω ≠ 0``). The natural object on
``(M, ω)`` is the **musical isomorphism** ``ω^♭ : TM → T*M``
``ω^♭(X) := ι_X ω`` and its inverse ``ω^♯``. Without ``dω = 0``
you lose ``L_X ω = 0 ⇔ X`` Hamiltonian, but you keep the
**vector-field equality** ``ι_X ω = ι_Y ω ⇒ X = Y`` (non-
degeneracy) and the bilinear pairing on functions.

### Step 1, pick the data

```python
from typing import Optional, Tuple
from jacopy.core.expr import Expr, Symbol
from jacopy.core.properties import Graded, NonDegenerate
from jacopy.core.registry import PropertyRegistry

class AlmostSymplecticProblem:
    """`(ω, registry)`, non-degenerate (but not closed) 2-form bundle."""

    __slots__ = ("_omega", "_registry", "_engine", "_name")

    def __init__(
        self,
        omega: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("omega must be an Expr")
        self._omega    = omega
        self._registry = registry if registry is not None else PropertyRegistry()
        self._declare_axioms()
        self._engine   = self._build_engine()
        self._name     = name or f"AlmostSymplecticProblem({omega._repr_inner()})"
```

### Step 2, auto-declare structural axioms

`SymplecticProblem` declares both `Closed(ω)` and
`NonDegenerate(ω)`. Our almost-symplectic version drops
`Closed(ω)`, that's the whole point.

```python
    def _declare_axioms(self) -> None:
        # Pre-declared flags (caller declared first) take precedence.
        if not self._registry.has(self._omega, Graded):
            self._registry.declare(self._omega, Graded(degree=2))
        if not self._registry.has(self._omega, NonDegenerate):
            self._registry.declare(self._omega, NonDegenerate())
```

The `has` check is **the override mechanism**: if the caller
pre-declared `Closed(ω)` (lying about `ω` for a different proof),
we don't fight them. Pre-declaring is the documented escape
hatch for every wrapper in `library/`.

### Step 3, assemble the engine

Layer the relevant rules onto `default_engine(registry=…)`,
that's the standard pattern. For almost-symplectic, we want
non-degeneracy interior-equality but **not** the closed-form rule.

```python
from jacopy.calculus.nondegenerate_axioms import (
    NonDegenerateInteriorEqualityDefinition,
)
from jacopy.proof.expansion import ExpansionEngine, default_engine

    def _build_engine(self) -> ExpansionEngine:
        base = default_engine(registry=self._registry)
        return ExpansionEngine(
            list(base.definitions) + [
                NonDegenerateInteriorEqualityDefinition(registry=self._registry),
            ]
        )
```

**Order matters**: definitions before linearity, frame-scoped
rules before generic ones. `default_engine` already gets that
right, adding your new rules to the *end* is almost always safe;
adding to the front (`[your_rule, *base.definitions]`) is the
escape hatch when a generic rule masks your specific one.

### Step 4, builder + prover methods

```python
from jacopy.algebra.derivation import Act
from jacopy.calculus.interior import interior
from jacopy.proof.simplify_chain import prove_equivalence

    @property
    def omega(self) -> Expr: return self._omega
    @property
    def registry(self) -> PropertyRegistry: return self._registry
    @property
    def engine(self) -> ExpansionEngine: return self._engine

    def musical_flat(self, X: Expr) -> Expr:
        r"""``ω^♭(X) = ι_X ω`` as an `Act` node."""
        return Act(interior(X), self._omega)

    def prove_vector_field_equality(self, X: Expr, Y: Expr):
        r"""Discharge ``ι_X ω = ι_Y ω ⇒ X = Y`` via non-degeneracy."""
        lhs = self.musical_flat(X)
        rhs = self.musical_flat(Y)
        return prove_equivalence(
            lhs, rhs, registry=self._registry, engine=self._engine,
        )
```

`prove_vector_field_equality` is the ergonomic counterpart to
`SymplecticProblem.prove_vector_field_equality`, the wrapper hides
the fact that *the proof itself* is just one engine step (the
non-degeneracy rule fires once on the difference).

### Step 5, (optional) seeded theorem

If your wrapper has an identity that should appear as a
**single citation step** in a proof transcript, register a
`Theorem` on the global `theorem_book`:

```python
from jacopy.proof.theorem_book import theorem_book, Theorem

def _build_almost_symplectic_volume_theorem() -> Theorem:
    return Theorem(
        name="almost_symplectic_volume",
        ...,  # statement / claim / metadata
    )

# at module import time:
theorem_book.register(_build_almost_symplectic_volume_theorem())
```

`SymplecticProblem` registers `poisson_jacobi`,
`poisson_koszul_equivalence`, `poisson_koszul_jacobi`;
`CourantAlgebroid` registers `courant_jacobi_twist` and
`courant_dorfman_bridge`. The pattern is consistent, the seeded
theorem becomes the proof artefact when a wrapper-level prover
emits a single citation.

Skip this step entirely if every proof in your wrapper is genuine
engine arithmetic, seeded theorems are for **identities you
don't want to re-derive every time**.

## Picking your axioms, flags vs definitions

Two ways to wire structural facts into the engine:

| Mechanism | When | Cost |
|---|---|---|
| Registry flag (`Closed`, `NonDegenerate`, `Poisson`, `Antisymmetric`) | Property is a **boolean fact** about a single object | A flag declaration + a registry-aware `Definition` instance |
| Custom `Definition` subclass | Property is a **rewrite shape**, non-trivial input/output pattern matching | A new class implementing `matches` / `rewrite` |

**Prefer flags** when you can: declarative, opt-out via
pre-declaration, single rule fires per object. Reach for a custom
`Definition` when the rewrite shape doesn't fit the
"one-bit-fact" pattern, the §3.1.5 derivator-shaped axioms in
`tilde/closure_axioms.py` are good examples.

If your "axiom" is really a **theorem** (a derivable identity),
register it as a seeded `Theorem` (step 5) rather than a
`Definition`, the proof artefact reads as a citation, not a
rewrite.

## Assembling the engine, order, idempotency, loops

Three rules of thumb:

1. **Definition rules before linearity rules.** Bracket / curvature
   definitions need to fire on the inert atom *before* sum / scalar
   pull-out reaches inside them.

2. **Idempotent rules can go anywhere.** A rule that produces output
   that doesn't re-trigger itself (e.g. duality `⟨e^a, X_b⟩ → δ^a_b`)
   is safe to register late.

3. **Avoid loops by construction.** Frame decomposition + duality is
   the canonical example: each rule undoes the other. The fix is
   **opt-in**: register the decomposition rule only inside the
   specific sub-pass that needs it (see `CartanStructureProblem`'s
   `_build_engine`).

When in doubt, run the engine on a small instance and inspect the
trace. `ExpansionEngine.expand(expr, max_steps=…)` returns the
final expression *and* the step transcript; an unexpected loop
shows up immediately.

## Where to look in the codebase

| Source | Read for |
|---|---|
| `library/symplectic.py` | The smallest non-trivial wrapper |
| `library/courant_algebroid.py` | Wrapper with seeded theorems + bridge identity |
| `library/bianchi_problem.py` | Wrapper with custom proof loop (`_expand_to_canonical`) |
| `library/koszul_problem.py` | The largest wrapper, multi-engine + canonicalize_indices pre-pass |
| `library/cartan_structure.py` | Index-laden wrapper with per-problem registry |

When you're stuck, **read the wrapper closest to your shape**.
The pattern repeats; the differences are in which rules they
bundle and which axioms they auto-declare.

## Summary

* Five-step recipe: data → axioms → engine → methods → (theorems).
* Pre-declared registry flags are the **opt-out mechanism**;
  always check `has(...)` before declaring.
* `default_engine(registry=…)` is the right base for most form-
  side problems; layer your rules onto its `definitions` list.
* Prefer registry flags for one-bit facts; custom `Definition`s
  for rewrite shapes; seeded theorems for derivable identities
  you want as citation steps.
* Engine order: definitions before linearity; frame-scoped rules
  before generic; opt-in rules for loop-prone pairs.
* Read `library/symplectic.py` for the smallest template,
  `library/koszul_problem.py` for the largest.
