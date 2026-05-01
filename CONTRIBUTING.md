# Contributing to jacopy

This guide is mathematician-flavoured: how do you **seed a new
Theorem**, **define a new bracket**, **add a new identity rule**,
or **write a Problem wrapper**? The package's surface is a
collection of those four things, every contribution lands in one
of them.

Before writing, glance at:

- The architectural design doc: [`plan.md`](plan.md) (long;
  intended for package authors, not users).
- The most relevant tutorial, `library/symplectic.py` (smallest
  wrapper) and `library/koszul_problem.py` (largest) are the
  reference templates for non-trivial library modules.
- The existing pattern for whatever you're adding, look at how
  `poisson_jacobi` is seeded if you're adding a Theorem,
  `KoszulBracket` if you're adding a bracket subclass, etc.

The rest of this guide walks each recipe with a concrete
illustration.

---

## 1. Seed a new Theorem in `theorem_book`

A `Theorem` record pairs a human-readable statement with a
`ProofChain`, an axiom list, and optional notes. The pattern
across `library/poisson.py`, `library/lie_algebroid.py`,
`library/courant_algebroid.py`, `library/dirac.py`:

```python
# in library/your_module.py

from jacopy.library.theorem_book import Theorem, theorem_book


def _build_my_theorem() -> Theorem:
    """Construct the canonical `my_theorem` record."""
    # 1. Build the operands you'll prove the identity on.
    #    Use generic Symbols + Graded(degree=...) on a fresh
    #    PropertyRegistry, the operands are display witnesses.
    reg = PropertyRegistry()
    pi = Symbol("π"); reg.declare(pi, Graded(degree=1))
    # ... your operands ...

    # 2. Run the proof on those operands. The chain is what gets
    #    seeded; downstream code never re-runs your proof, it
    #    cites the chain.
    chain = my_object.prove_my_identity(..., registry=reg)

    # 3. Wrap as a Theorem with explicit axiom provenance.
    return Theorem(
        name="my_theorem",                      # registry key
        statement="..."                         # one-line claim
        from_axioms=(                           # atomic axioms it rests on
            "Derived Bracket Theorem",
            "[Q, Q]_base = 0 (your hypothesis)",
        ),
        proof=chain,
        notes="Derivation pipeline + any pitfalls.",
    )


# Module-level seeding, runs at import time.
THEOREM_MY_THEOREM = _build_my_theorem()
if "my_theorem" not in theorem_book:
    theorem_book.add(THEOREM_MY_THEOREM)
```

**Three rules of thumb:**

- **`from_axioms` is the contract.** It declares the atomic
  hypotheses your proof depends on. The package does not
  cross-validate this against the chain, accuracy is on you. Be
  honest: a downstream paper citation will quote what you wrote.
- **The chain is generic.** Build it on fixed symbolic witnesses
  (`f, g, h` for functions, `α, β` for forms, etc.), downstream
  callers produce their own chain on their concrete operands by
  calling the same `prove_*` method that produced your seed.
- **Idempotent registration.** Wrap the `theorem_book.add` call
  in `if "name" not in theorem_book`. Modules can be re-imported
  during testing; double-registering raises.

The eight currently seeded theorems
(`poisson_jacobi`, `poisson_koszul_equivalence`,
`poisson_koszul_jacobi`, `lie_algebroid_anchor_compat`,
`courant_jacobi_twist`, `courant_dorfman_bridge`,
`dirac_isotropy`, `dirac_involutivity`) are the references,
each follows this template.

---

## 2. Define a new bracket

There are two paths, depending on how much structure you need.

### Quick path, `CustomBracket`

For a one-off rule (test, exploration, paper draft), wrap an
expand callable directly. Tutorial 06 covers this end-to-end.

```python
from jacopy.brackets.custom import CustomBracket
from jacopy.core.expr import Sum, Product, Neg

def commutator(a, b, registry):
    return Sum(Product(a, b), Neg(Product(b, a)))

B = CustomBracket(
    "[·,·]",
    commutator,
    is_graded_antisymmetric=True,    # the four axiom flags
    satisfies_leibniz=True,
    satisfies_graded_jacobi=True,    # True / False / None (conditional)
    degree=0,
)
```

### Subclass path, `GradedBracket`

For a bracket with structural identity (a shared anchor, a
seeded theorem, custom obstruction hooks, registry-aware
behaviour), subclass `GradedBracket`. The minimal recipe:

```python
from jacopy.brackets.base import GradedBracket
from jacopy.core.expr import Expr

class MyBracket(GradedBracket):
    """`[·,·]_M`, your bracket's display name + structural identity."""

    is_graded_antisymmetric = True
    satisfies_leibniz       = True
    satisfies_graded_jacobi = None      # conditional, picks DerivedBracketStrategy

    def __init__(self, anchor: Expr, *, name: str = "[·,·]_M"):
        super().__init__(name=name, degree=0)
        self._anchor = anchor

    def expand(self, a: Expr, b: Expr, registry=None) -> Expr:
        """The bracket's expansion rule."""
        # ... build the RHS Expr ...
        return ...

    def _identity_key(self):
        """Structural identity. Two MyBrackets compare equal iff this matches."""
        return (self._name, self._anchor)
```

The `KoszulBracket`, `CourantBracket`, `SchoutenBracket`, and
`DerivedBracket` classes are the reference subclasses, read
`brackets/koszul.py` for the smallest non-trivial example.

**When to graduate from `CustomBracket` to a subclass:**

- You need `expand_definition` (the surface form, not the full
  expansion).
- You need an obstruction hook (`anchor_compatibility_obstruction`,
  `jacobi_condition`).
- You want to seed a theorem that cites this bracket by name.
- The bracket has parameters (anchor, twist, generator) that
  enter its `_identity_key`.

---

## 3. Add a new identity rule (axiom or theorem)

The expansion engine consumes `Definition` subclasses. Each rule
declares a `matches(expr)` predicate and a `rewrite(expr)` rule;
optionally a `theorem_proof_builder()` for foundational mode.
Tutorial 09 walks the axiom-vs-theorem split end-to-end.

### Axiom-class rule, single rewrite, no sub-proof

```python
from jacopy.proof.expansion import Definition
from jacopy.core.expr import Expr, Integer

class MyAxiomDefinition(Definition):
    """`some_pattern → 0`, the rewrite this axiom enacts."""

    name = "my axiom: some_pattern → 0"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, ...) and ...     # shape check

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)                         # canonical replacement
```

### Registry-aware rule, fires only when a flag is declared

```python
from jacopy.core.properties import Closed
from jacopy.core.registry import PropertyRegistry

class MyClosedRule(Definition):
    """Some rewrite that only fires when ω is `Closed`."""

    name = "my rule [Closed]"

    def __init__(self, *, registry: PropertyRegistry | None = None):
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, ...):
            return False
        if self._registry is None:
            return False        # safety hatch, no-op without a registry
        return self._registry.has(expr.target_form, Closed)
```

`registry=None` as a no-op is the package convention, see
`ClosedFormDefinition`, `NonDegenerateInteriorEqualityDefinition`,
`RegistryAntiSymCanonicalDefinition` in `calculus/*_axioms.py`
for the reference.

### Theorem-class rule, attaches a sub-proof in foundational mode

```python
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep

class MyTheoremDefinition(Definition):
    name = "my theorem"

    def matches(self, expr: Expr) -> bool: ...
    def rewrite(self, expr: Expr) -> Expr: ...

    def theorem_proof_builder(self):
        """Return a builder that emits the sub-proof in foundational mode."""
        def build(matched: Expr) -> ProofChain:
            step = ProofStep(
                rule="generator-level axiom you're citing",
                before=matched,
                after=self.rewrite(matched),
                justification="...",
                provenance_tag="axiom",
            )
            return ProofChain(steps=[step])
        return build
```

In efficient mode the rule fires atomically; in foundational
mode the sub-proof is attached on `ProofStep.children`. The
package's `DSquaredZeroDefinition` is the canonical theorem-class
template.

### Where to put your rule

- **Generic / cross-cutting** (works on any registry,
  no library coupling): `jacopy/calculus/<topic>_axioms.py`.
- **Library-specific** (only fires inside a particular Problem
  wrapper): keep it next to the wrapper in `jacopy/library/`.

The line is fuzzy, when in doubt, follow the closest existing
rule. `calculus/closed_axioms.py` is the smallest reference for
a registry-aware rule; `library/cartan_structure.py` is the
reference for "this rule only makes sense inside this wrapper".

---

## 4. Write a Problem wrapper

The five-step recipe is in **[Tutorial 24](docs/tutorials/24_custom_problem_wrapper.md)**
, it walks an `AlmostSymplecticProblem` example end-to-end. The
short version:

1. Pick the geometric data the wrapper carries.
2. Auto-declare structural axioms on the registry, guarded with
   `registry.has(...)` to honour pre-declared flags.
3. Assemble the engine, layer your rules onto
   `default_engine(registry=reg)`.
4. Write builder + prover methods that match the textbook idiom.
5. (Optional) Register seeded `Theorem`s for one-step citations.

The reference templates by size:
`library/symplectic.py` (~200 lines, smallest non-trivial),
`library/courant_algebroid.py` (mid-size, with seeded theorems),
`library/koszul_problem.py` (~1100 lines, the largest, has
multi-engine, `canonicalize_indices` pre-pass, three derivator
modes).

---

## Tests

The package uses pytest. Two suites you'll touch:

- `tests/`, ~2700 unit tests. Add tests for any new
  bracket / Definition / wrapper.
- `tests/test_docs/test_notebooks.py`, 24 notebook smoke tests.
  If you add a tutorial, register its `TUTORIAL_NN` cell list in
  `docs/tutorials/_build_notebooks.py` and the test will pick it
  up automatically on the next regeneration.

Quick commands:

```bash
pytest                                    # full suite
pytest tests/test_brackets/               # one directory
pytest -k "poisson"                       # match by name
python3 docs/tutorials/_build_notebooks.py    # regenerate .ipynb files
pytest tests/test_docs/test_notebooks.py -q   # 24/24 in ~18s
```

## Code style

- No required runtime dependencies. Optional `rich` for
  terminal rendering, optional `nbformat` / `nbclient` /
  `ipykernel` for notebook execution. Don't add hard
  dependencies.
- Type hints throughout. Public APIs use full `from typing import
  ...`; internal helpers can be lighter.
- `__slots__` on Expr subclasses where it matters for memory.
- Display: every `Expr` subclass implements `_repr_inner`; the
  three-way render (`to_ascii`, `to_latex`, `print_expr`) is
  driven by dispatch in `display/`. New Expr nodes need a
  matching `display/{ascii,latex}.py` entry.
- Russian-doll error messages: when raising `ProofFailure` /
  `ValueError` from a deep rewrite, include the expression
  causing the failure plus the rule's identity. The
  `simplify_chain` and `verifier` modules both follow this
  pattern.

## Documentation

- New tutorial: paired `.md` + `TUTORIAL_NN` cell list in
  `_build_notebooks.py` + `build_all` registration. Keep both
  in sync, mismatched markdown vs notebook cells is the most
  common drift.
- New library module: a paragraph in `docs/tutorials/README.md`
  if it warrants its own tutorial; otherwise fold mention into
  the closest existing tutorial.
- Don't write README badges or maintainer-flavoured docs unless
  the user asks. The package is pre-alpha; status churn is real.

## Memory & repo state

- Avoid committing `.aux`, `.log`, `.pdf` artefacts in
  `examples/Question N/` directories, those are
  build-by-product. The corresponding `.tex` is the source of
  truth.
- Memory files in `~/.claude/projects/...` belong to the
  contributor's own session, not the repository.

---

## Where to ask

This is a research-grade package; there's no community channel
yet. Open an issue with a minimal reproducer and the relevant
`ProofFailure` text, most "the engine left a residue I can't
explain" reports are diagnosable from the message text alone
(see Tutorial 10).
