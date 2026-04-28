# jacopy

> Symbolic engine for graded algebra, brackets, and Cartan calculus —
> with **step-by-step proofs** the engine generates and the user can read.

The mathematical core: the *Derived Bracket Theorem* unifies
Poisson, Koszul, and Courant brackets under a single hypothesis
(`[Q, Q]_base = 0`). `jacopy` realises the theorem and its
consequences as a working symbolic system: brackets are objects,
identities are `ProofChain`s, axioms are tagged, and every claim
carries provenance back to a primitive.

## What you can do with it

- Prove the Jacobi identity for a Lie / SN / Koszul / Courant
  bracket and read the proof transcript step by step.
- Verify Cartan structure equations (`T^a = de^a + ω^a_b ∧ e^b`)
  on a connection + frame — both Bianchi identities close
  mechanically.
- Discharge the §3.1.5 derivator identities (form-side and dual
  multivector-side) on a Poisson manifold via
  `KoszulProblem.prove_derivator`.
- Write your own bracket / connection / Cartan-style problem and
  plug it into the same machinery.

## Five-line "hello"

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

## Install

```bash
pip install -e ".[dev]"
pytest
```

Python ≥ 3.10. No required runtime dependencies; `rich` (optional)
gives coloured tree rendering, `nbformat` + `nbclient` + `ipykernel`
are needed for the notebook tutorial test suite.

## Where to learn

- **[`docs/README.md`](docs/README.md)** — three reading paths
  (practitioner / depth-first / topical) through the 24 tutorials.
- **[`docs/tutorials/`](docs/tutorials/)** — paired `.md` + `.ipynb`
  for each tutorial. Start at
  [01_first_steps.md](docs/tutorials/01_first_steps.md).
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — recipes for adding a
  bracket, a `Theorem` to `theorem_book`, a new axiom rule, or a
  Problem wrapper.
- **[`examples/`](examples/)** — the textbook problems the package
  was first calibrated against (Math 595 Question sheets).

## Library landmarks

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

## Status

Pre-alpha. ~2700 unit tests + 24 notebook smoke tests; the
mathematical surface (bracket families, Cartan calculus, derived
identities, Bianchi, Cartan structure equations) is closed for the
calibration set. The user-facing API is stable enough to write
papers against, but not yet pinned by SemVer.

## License

MIT.
