# Tutorials

25 tutorials, paired `.md` (read) and `.ipynb` (run). All in
English. Each `.ipynb` is a smoke test in
`tests/test_docs/test_notebooks.py` and re-generated from
`_build_notebooks.py`.

For reading paths (practitioner / depth-first / topical), see
[`docs/README.md`](../README.md).

## The 25 tutorials

### Foundations (01–09)

| # | Title | What you get |
|---|---|---|
| 01 | [First steps](01_first_steps.md) | `Expr`, `Symbol`, `Sum`, `Product`, `simplify`, `PropertyRegistry` |
| 02 | [The Jacobi identity](02_jacobi_identity.md) | `prove_jacobi(lie, X, Y, Z)` in two steps |
| 03 | [Poisson geometry](03_poisson_geometry.md) | Three equivalent views of `{f, g}_π`, `[π,π]_SN = 0` reduction |
| 04 | [Lie algebroid](04_lie_algebroid.md) | `(E, [·,·]_E, ρ)` triple, anchor compatibility axiom |
| 05 | [Cartan calculus](05_cartan_calculus.md) | Five Cartan relations, axiom vs theorem mode for `d² = 0` |
| 06 | [Custom bracket](06_custom_bracket.md) | `CustomBracket`, axiom flags, generic `prove_jacobi` dispatch |
| 07 | [Derived bracket](07_derived_bracket.md) | `{a,b}_Q := [[a,Q],b]`, the unifying construction |
| 08 | [The unified picture](08_unified_picture.md) | One hypothesis, many consequences; `theorem_book` citations |
| 09 | [Foundations](09_foundations.md) | Axiom vs theorem, efficient vs foundational, custom axiom sets |

### Workflow (10–11, 14)

| # | Title | What you get |
|---|---|---|
| 10 | [Diagnostics & proof debugging](10_diagnostics.md) | `ProofFailure.report`, residual-shape rules |
| 11 | [Publication-ready output](11_publication_output.md) | LaTeX `align*` chains, TikZ trees, full `.tex` documents |
| 14 | [Problem wrappers](14_problem_wrappers.md) | `SymplecticProblem`, `KoszulProblem`, end-to-end Question 2/3 |

### Depth (12–13, 15–18)

| # | Title | What you get |
|---|---|---|
| 12 | [Schouten–Nijenhuis bracket](12_schouten_nijenhuis.md) | SN axioms, atomic vs derived, Poisson bridge |
| 13 | [Closure axioms](13_closure_axioms.md) | `Closed`, `Antisymmetric`, `NonDegenerate` flags + paired rules |
| 15 | [The intrinsic engine](15_intrinsic_engine.md) | `intrinsic_engine`, `prove_intrinsic_equivalence`, recogniser |
| 16 | [Phase 13 deep dive](16_phase_13_deep_dive.md) | Closing `[π,π]_SN` with no seeded theorem |
| 17 | [Tilde calculus](17_tilde_calculus.md) | `ι̃_ω`, `d̃`, `L̃_ω`, `K̃_η` polarity flip |
| 18 | [Derivator identities (§3.1.5)](18_derivator_identities.md) | Six identities through `KoszulProblem.prove_derivator` |

### Geometry (19–20)

| # | Title | What you get |
|---|---|---|
| 19 | [Courant family](19_courant_family.md) | Dorfman, Courant, H-twist, Dirac (`poisson_dirac`, `presymplectic_dirac`) |
| 20 | [Connection, curvature, Bianchi](20_connection_curvature.md) | `AffineConnection`, `Torsion`, `Curvature`, `BianchiProblem` |

### Internals + applied (21–25)

| # | Title | What you get |
|---|---|---|
| 21 | [IndexedSum, Wedge, MultiEval](21_indexed_sum_wedge_multi_eval.md) | Three structural Expr nodes + their axiom rules |
| 22 | [Local frames, frame decomposition](22_frame_decomposition.md) | `LocalFrame`, duality, opt-in decomposition rules |
| 23 | [Cartan structure equations](23_cartan_structure_equations.md) | `CartanStructureProblem` I & II in 49 / 54 steps |
| 24 | [Writing your own Problem wrapper](24_custom_problem_wrapper.md) | The five-step recipe via `AlmostSymplecticProblem` |
| 25 | [Frame-component differential geometry](25_frame_calc.md) | `jacopy.frame_calc`, Christoffel / Ricci / Einstein / Kretschmann on concrete metrics; 10 library fixtures (Schwarzschild, Kerr, RN, dS/AdS, Vaidya, Bianchi I/V/IX, Gödel, FRW, Minkowski); custom-connection helpers |

## Dependency arrows

The tutorials are mostly independent past their phase
prerequisite. The minimal prerequisites:

```
01 → 02 → 03 → 04 → 05
                ↓
                12 ← 13 → 15 → 16 → 17 → 18
                       ↓
                       14 → 24
                       ↓
20 → 22 → 23 → 24      19
↑
06 → 07 → 08 → 09
                ↓
                10 → 11
                ↓
                21 (any time after 01)
```

Lateral links across this graph appear inside individual
tutorials when relevant, the cross-references aren't
load-bearing for the main reading order.

Tutorial **25** sits outside this graph: `jacopy.frame_calc` is an
opt-in submodule (requires SymPy) for component-level differential
geometry on concrete metrics. Familiarity with curvature concepts
from tutorials 05 (Cartan calculus) or 20 (Bianchi) helps but is
not required.

## Building / testing

```bash
python3 docs/tutorials/_build_notebooks.py        # regenerate all .ipynb
python3 -m pytest tests/test_docs/test_notebooks.py -q   # 25/25 in ~18s
```

The `_build_notebooks.py` script is also where the markdown
versions of code blocks are mirrored in cell form. Edit both the
`.md` *and* the matching `TUTORIAL_NN` cell list when revising
content.
