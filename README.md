# jacopy

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange)](https://pypi.org/classifiers/)
[![Tests: 2792](https://img.shields.io/badge/tests-2792%20passing-brightgreen)](#testing)

> **Symbolic engine for graded algebra, brackets, and Cartan calculus —
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

### For development

```bash
git clone https://github.com/oselcukk/jacopy-v2.0.git
cd jacopy-v2.0
pip install -e ".[dev]"        # editable + dev tools (pytest, rich, nbformat, ...)
```

### Optional dependency groups

| Extras | Adds |
|---|---|
| `[rich]` | `rich` — coloured terminal tree rendering |
| `[test]` | `pytest` |
| `[docs]` | `nbformat`, `nbclient`, `ipykernel` — needed for tutorial notebooks |
| `[dev]` | All of the above (single one-liner for contributors) |

**Requirements:** Python ≥ 3.10. **Zero required runtime dependencies** —
the package works out of the box with the standard library alone;
extras only enhance display, testing, and notebook execution.

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

Lower-level primitives — `Expr` algebra, `PropertyRegistry`,
`ExpansionEngine`, `prove_jacobi`, `prove_intrinsic_equivalence`,
`theorem_book`, `ProofChain` — are documented in the tutorials.

### `jacopy.frame_calc` — component-level differential geometry

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

Library fixtures: `minkowski`, `schwarzschild`, `frw`, `kerr`. Full
walkthrough in [Tutorial 25](docs/tutorials/25_frame_calc.md).

## 📖 Documentation

`jacopy` ships with **24 paired tutorials** (`.md` for reading,
`.ipynb` for running). All tutorials are smoke-tested in CI.

- **[`docs/README.md`](docs/README.md)** — three reading paths
  (practitioner / depth-first / topical).
- **[`docs/tutorials/`](docs/tutorials/)** — start at
  [`01_first_steps.md`](docs/tutorials/01_first_steps.md).
- **[`docs/tutorials/README.md`](docs/tutorials/README.md)** —
  full index of all 24 tutorials with one-line descriptions
  and dependency arrows.
- **[`examples/`](examples/)** — the textbook problems the package
  was first calibrated against (Math 595 question sheets).

> **Note:** A generated API reference (Sphinx + autodoc) is
> deferred until the surface stabilises. The tutorials cover every
> public class in narrative form.

## 🧪 Testing

```bash
pytest                                          # full suite — 2792 tests
pytest tests/test_docs/test_notebooks.py -q     # 24 notebook smoke tests
```

| Suite | Count | Time |
|---|---|---|
| Unit tests | 2792 | ~20 s |
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

[MIT](LICENSE).

---

> 💡 **Pre-alpha note.** The user-facing API is stable enough to
> write papers against, but not yet pinned by SemVer. Breaking
> changes between `0.0.x` releases are possible — pin a specific
> commit if you depend on it for reproducible work.
