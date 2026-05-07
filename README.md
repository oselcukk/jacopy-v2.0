# jacopy

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-red)](LICENSE)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange)](https://pypi.org/classifiers/)
[![Tests: 3108](https://img.shields.io/badge/tests-3108%20passing-brightgreen)](#testing)

> **Symbolic engine for graded algebra, brackets, and Cartan calculus,
> with step-by-step proofs the engine generates and the user can read.**

The mathematical core: the *Derived Bracket Theorem* unifies
Poisson, Koszul, and Courant brackets under a single hypothesis
(`[Q, Q]_base = 0`). `jacopy` realises the theorem and its
consequences as a working symbolic system: brackets are objects,
identities are `ProofChain`s, axioms are tagged, and every claim
carries provenance back to a primitive.

---

## 📑 Table of contents

- [What you can do](#-what-you-can-do)
- [Quick start](#-quick-start)
- [Installation](#-installation)
- [Library landmarks](#-library-landmarks)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ What you can do

| Capability | Tutorial |
|---|---|
| 🔁 Prove the Jacobi identity for a Lie / SN / Koszul / Courant bracket and read the proof transcript | [02](docs/tutorials/02_jacobi_identity.md) |
| 🌀 Verify Cartan structure equations on a connection + frame | [23](docs/tutorials/23_cartan_structure_equations.md) |
| ⚙️ Close both Bianchi identities mechanically | [20](docs/tutorials/20_connection_curvature.md) |
| 📐 Discharge the §3.1.5 derivator identities (form-side and dual) | [18](docs/tutorials/18_derivator_identities.md) |
| 🪞 Work with the tilde calculus on a Poisson manifold | [17](docs/tutorials/17_tilde_calculus.md) |
| 🧩 Plug your own bracket / connection / Problem wrapper into the engine | [24](docs/tutorials/24_custom_problem_wrapper.md) |
| 📜 Render proofs as LaTeX `align*` blocks or TikZ trees | [11](docs/tutorials/11_publication_output.md) |

## 🚀 Quick start

```python
from jacopy import VectorFields
from jacopy.brackets.lie import lie
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import prove_jacobi

reg = PropertyRegistry()
X, Y, Z = VectorFields("X Y Z", registry=reg)

chain = prove_jacobi(lie, X, Y, Z, registry=reg)
print(f"Jacobi closes in {len(chain)} steps; final = {chain.steps[-1].after}")
# Jacobi closes in 2 steps; final = 0
```

## 📦 Installation

Install directly from GitHub.

### As a user

```bash
pip install git+https://github.com/oselcukk/jacopy-v2.0.git
```

If you want the component-level differential geometry submodule
(`jacopy.frame_calc`, Christoffel / Ricci / Einstein / Kretschmann
on concrete metrics), install with the `components` extra:

```bash
pip install "jacopy[components] @ git+https://github.com/oselcukk/jacopy-v2.0.git"
```

### For development

**Recommended — virtual environment:**

```bash
git clone https://github.com/oselcukk/jacopy-v2.0.git
cd jacopy-v2.0
make setup                     # creates .venv, installs deps, registers Jupyter kernel
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,parallel]"   # editable + dev tools + joblib for parallel Riemann
python -m ipykernel install --user --name=jacopy --display-name="Python (jacopy .venv)"
```

> **zsh note:** the brackets in `".[dev,parallel]"` need quoting; copy-paste
> the line as-is.

**Alternative — single-file requirements:**

```bash
pip install -r requirements.txt    # no editable install, runtime deps only
```

### Optional dependency groups

| Extras | Adds |
|---|---|
| `[rich]` | `rich`, coloured terminal tree rendering |
| `[test]` | `pytest` |
| `[docs]` | `nbformat`, `nbclient`, `ipykernel`, needed for tutorial notebooks |
| `[components]` | `sympy`, required for `jacopy.frame_calc` (component-level differential geometry) |
| `[parallel]` | `joblib`, multi-core Riemann/Ricci/Einstein for heavy metrics (Kerr, Bianchi-IX) |
| `[dev]` | All of the above (single one-liner for contributors) |

**Requirements:** Python ≥ 3.10. **Zero required runtime dependencies**
for the proof / bracket / Cartan core, works with the standard
library alone. The `frame_calc` submodule is the only part that
needs SymPy (via the `[components]` extra).

## 📚 Library landmarks

### Symbolic / proof layer (operator-level)

The four high-level **Problem wrappers** are the user-facing
entry points for textbook calculations:

| Wrapper | What it solves |
|---|---|
| `SymplecticProblem(ω, X, Y)` | Symplectic / Poisson closure, Hamiltonian VF equality |
| `KoszulProblem(π, [α, β, …])` | Koszul / SN / tilde calculus, §3.1.5 derivator identities |
| `BianchiProblem(∇)` | Torsion / curvature, both Bianchi identities |
| `CartanStructureProblem(∇, F)` | Cartan I & II structure equations |

Lower-level primitives, `Expr` algebra, `PropertyRegistry`,
`ExpansionEngine`, `prove_jacobi`, `prove_intrinsic_equivalence`,
`theorem_book`, `ProofChain`, are documented in the tutorials.

### `jacopy.frame_calc`, component-level differential geometry

For concrete metric calculations (Christoffel symbols, Ricci tensor,
Einstein tensor on real metrics like Schwarzschild or Kerr), use the
`jacopy.frame_calc` submodule. It requires SymPy as an opt-in
dependency (`pip install "jacopy[components]"`):

```python
from jacopy.frame_calc import einstein_tensor, levi_civita
from jacopy.frame_calc.library import schwarzschild

F, g = schwarzschild()
G = einstein_tensor(levi_civita(g), g)
assert G.is_vacuum()      # symbolic vacuum verification
```

**Library fixtures (10):** `minkowski`, `schwarzschild`,
`reissner_nordstrom`, `kerr`, `frw`, `de_sitter`, `anti_de_sitter`,
`vaidya`, `bianchi_I/V/IX`, `godel`.

**Curvature invariants:** `kretschmann`, `ricci_squared`, `cotton` (3D).
For Schwarzschild: `kretschmann(R, g)` returns `48 M² / r⁶`,
diagnoses the horizon as a coordinate (not real) singularity.

**Custom connections:** `connection_with_torsion` (Einstein-Cartan),
`weyl_connection`, `projective_connection`, drop in any user-defined
deformation and route the rest of the pipeline through it.

**Helpers:** `analyze_metric(matrix, coords)` runs the full pipeline
in one call; `to_latex_table(tensor)` renders Christoffel/Ricci as
paper-grade LaTeX `align*`.

Full walkthrough in [Tutorial 25](docs/tutorials/25_frame_calc.md).

## 📖 Documentation

`jacopy` ships with **25 paired tutorials** (`.md` for reading,
`.ipynb` for running). All tutorials are smoke-tested in CI.

- **[`docs/README.md`](docs/README.md)**, three reading paths
  (practitioner / depth-first / topical).
- **[`docs/tutorials/`](docs/tutorials/)**, start at
  [`01_first_steps.md`](docs/tutorials/01_first_steps.md).
- **[`docs/tutorials/README.md`](docs/tutorials/README.md)**,
  full index of all 25 tutorials with one-line descriptions
  and dependency arrows.
- **[`examples/`](examples/)**, the textbook problems the package
  was first calibrated against (Math 595 question sheets).

> **Note:** A generated API reference (Sphinx + autodoc) is
> deferred until the surface stabilises. The tutorials cover every
> public class in narrative form.

## 🧪 Testing

```bash
pytest                                          # full suite, 3108 tests
pytest tests/test_docs/test_notebooks.py -q     # 24 notebook smoke tests
```

| Suite | Count | Time |
|---|---|---|
| Unit tests | 3108 | ~60 s |
| Notebook smoke | 24 | ~18 s |

The mathematical surface (bracket families, Cartan calculus,
derived identities, Bianchi, Cartan structure equations) is
**closed** for the calibration set.

## 🤝 Contributing

See **[`CONTRIBUTING.md`](CONTRIBUTING.md)** for math-flavoured
recipes:

- 📜 Seeding a new `Theorem` in `theorem_book`
- 🔗 Defining a new bracket (`CustomBracket` vs `GradedBracket` subclass)
- 📐 Adding a new identity / axiom rule (`Definition`)
- 🎁 Writing your own Problem wrapper

Each recipe links to the closest reference template in `library/`.

## 📑 Citation

If you use `jacopy` in academic work, please cite:

```bibtex
@software{jacopy,
  title  = {jacopy: Symbolic computation for graded algebra,
            brackets, and Cartan calculus with step-by-step proofs},
  author = {Selçuk, Oğuzhan},
  year   = {2026},
  url    = {https://github.com/oselcukk/jacopy-v2.0},
  note   = {Pre-alpha}
}
```

## 📄 License

**Proprietary, Source-Available, Personal Use Only.**

Copyright (c) 2026 Oğuzhan Selçuk. All rights reserved.

This source code is publicly visible for transparency, academic
review, and personal experimentation. You **may** clone the repo,
read the code, and run it locally for personal/educational/research
use, and cite the software in academic work. You **may not** modify,
redistribute, or use this code in any commercial product or service
without prior written permission.

For commercial licensing or extended-rights inquiries, contact
**stabledifgc@gmail.com**. Full terms in [`LICENSE`](LICENSE).

---

> 💡 **Pre-alpha note.** The user-facing API is stable enough to
> write papers against, but not yet pinned by SemVer. Breaking
> changes between `0.0.x` releases are possible, pin a specific
> commit if you depend on it for reproducible work.
