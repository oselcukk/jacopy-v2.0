# Documentation

`jacopy` ships with 25 tutorials, paired `.md` (for reading) and
`.ipynb` (for running). The tutorials are the primary documentation
, there is no separate API reference (the tutorials cover every
public class). Pick the path that matches what you came for.

## Three reading paths

### 🧭 Practitioner, "I want to solve a textbook problem"

For someone who wants to *use* the package: prove a Jacobi
identity, close a Bianchi identity, verify a Cartan structure
equation. Skip the deep machinery.

| Step | Tutorial | What you get |
|---|---|---|
| 1 | [01, First steps](tutorials/01_first_steps.md) | `Expr`, registry, simplify |
| 2 | [02, Jacobi identity](tutorials/02_jacobi_identity.md) | One-line `prove_jacobi` |
| 3 | [03, Poisson geometry](tutorials/03_poisson_geometry.md) | `PoissonBracket`, three views |
| 4 | [11, Publication-ready output](tutorials/11_publication_output.md) | LaTeX / TikZ render |
| 5 | [14, Problem wrappers](tutorials/14_problem_wrappers.md) | The four user-facing wrappers |
| 6 | Pick by topic ↓ |

Then jump to the tutorial that matches your geometry:
- Connection + curvature → [20](tutorials/20_connection_curvature.md)
- Cartan structure equations → [23](tutorials/23_cartan_structure_equations.md)
- Courant / Dirac → [19](tutorials/19_courant_family.md)
- Derivator identities → [18](tutorials/18_derivator_identities.md)

### 📚 Depth-first, "I want to understand how the engine works"

For someone who wants to learn what's *underneath* the API:
expansion engines, axiom flags, intrinsic engines, the Phase 13
machinery.

| Step | Tutorial | Layer |
|---|---|---|
| 1–9 | [01](tutorials/01_first_steps.md) → [09](tutorials/09_foundations.md) | Symbol / proof foundations |
| 10 | [Diagnostics](tutorials/10_diagnostics.md) | Reading a failed proof |
| 12 | [Schouten–Nijenhuis](tutorials/12_schouten_nijenhuis.md) | The Poisson backbone |
| 13 | [Closure axioms](tutorials/13_closure_axioms.md) | Registry-property mechanism |
| 15 | [Intrinsic engine](tutorials/15_intrinsic_engine.md) | MultiEval-level rules |
| 16 | [Phase 13 deep dive](tutorials/16_phase_13_deep_dive.md) | `[π,π]_SN` without seeded theorem |
| 17 | [Tilde calculus](tutorials/17_tilde_calculus.md) | Dual Cartan picture |

### 🔧 Topical / reference, "I'm stuck on X"

For someone debugging a residue or extending the engine.

| Topic | Tutorial |
|---|---|
| Reading `ProofFailure`'s residue | [10](tutorials/10_diagnostics.md) |
| `Closed` / `Antisymmetric` / `NonDegenerate` flags | [13](tutorials/13_closure_axioms.md) |
| `IndexedSum` / `Wedge` / `MultiEval` Expr nodes | [21](tutorials/21_indexed_sum_wedge_multi_eval.md) |
| `LocalFrame`, frame decomposition | [22](tutorials/22_frame_decomposition.md) |
| Writing your own bracket | [06](tutorials/06_custom_bracket.md), [07](tutorials/07_derived_bracket.md) |
| Writing your own Problem wrapper | [24](tutorials/24_custom_problem_wrapper.md) |
| Concrete metric calculations (Schwarzschild, Kerr, FRW, RN, dS/AdS, Vaidya, Bianchi, Gödel) | [25](tutorials/25_frame_calc.md) |
| Curvature invariants (Kretschmann, Ricci², Cotton) | [25](tutorials/25_frame_calc.md) |
| Custom connections (torsion, Weyl, projective) | [25](tutorials/25_frame_calc.md) |

## The full tutorial list

See [tutorials/README.md](tutorials/README.md) for one-line
descriptions of all 25 tutorials and their dependency arrows.

## What's *not* here

- **Generated API reference.** The tutorials cover every public
  class in narrative form; an autodoc-generated reference is
  deferred until the surface stabilises.
- **Per-module changelog.** Pre-alpha; track changes via `git log`
  on `jacopy/` for now.

## Running the notebooks

The `.ipynb` files are regenerated from
`tutorials/_build_notebooks.py`:

```bash
python3 docs/tutorials/_build_notebooks.py
python3 -m pytest tests/test_docs/test_notebooks.py   # 25/25
```

Notebook execution dependencies (one-time):

```bash
pip install nbformat nbclient ipykernel
python3 -m ipykernel install --user --name python3
```
