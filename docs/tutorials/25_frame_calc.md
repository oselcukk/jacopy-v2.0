# 25 — Frame-component differential geometry (`jacopy.frame_calc`)

`jacopy.frame_calc` is jacopy's component-level submodule for
**concrete metric calculations**: given a metric ``g`` on a frame,
compute Christoffel symbols, Riemann curvature, Ricci tensor,
scalar curvature, Einstein tensor — with optional step-by-step
derivation transcripts that bridge to `ProofChain` for paper-grade
LaTeX output.

This tutorial covers:

1. The `jacopy.frame_calc` submodule layout — what it adds on top
   of jacopy's symbolic / proof layer.
2. Frame setup: `CoordinateFrame` (most common) and `Tetrad`
   (orthonormal bases via vielbein).
3. `ComponentMetric` and its `.inverse()`.
4. `levi_civita(g)` — Christoffel via Koszul formula.
5. `curvature`, `ricci`, `ricci_scalar`, `einstein_tensor`.
6. Optimised mode for Kerr-class metrics.
7. Library fixtures — `minkowski`, `schwarzschild`, `frw`, `kerr`.
8. ProofChain bridge: rendering derivations to paper-grade LaTeX.

`frame_calc` requires **SymPy as an optional dependency** under
the `[components]` extras. Install via:

```bash
pip install "jacopy[components]"
```

## A quick taste — Schwarzschild vacuum in five lines

```python
from jacopy.frame_calc import einstein_tensor, levi_civita
from jacopy.frame_calc.library import schwarzschild

F, g = schwarzschild()
G = einstein_tensor(levi_civita(g), g)
assert G.is_vacuum()
```

That's it — `jacopy` symbolically computes that Schwarzschild's
Einstein tensor vanishes identically, the defining property of a
vacuum solution to Einstein's field equations.

## 1. Submodule layout

`jacopy.frame_calc` is **separate from jacopy's symbolic / proof
layer** — it does component-level numerical-symbolic computation,
where the rest of jacopy does abstract operator algebra on
brackets / Cartan calculus / etc. The two layers complement each
other through the `proof_bridge` module (Stage G).

Pieces you'll encounter:

| Type | What it is |
|---|---|
| `Frame` (protocol) | The interface every frame implementation satisfies |
| `CoordinateFrame` | `e_a = ∂/∂x^a` — most common |
| `Tetrad` | `e_a = e_a^μ ∂/∂x^μ` defined by a vielbein |
| `AbstractFrame` | Symbolic frame with opaque structure constants |
| `ComponentMetric` | A `(0, 2)` symmetric tensor, the metric `g_{ab}` |
| `ComponentMetricInverse` | `(2, 0)` inverse `g^{ab}` |
| `ComponentConnection` | `(1, 2)` Christoffel symbols `Γ^a_{bc}` |
| `LeviCivitaConnection` | A connection with Koszul-formula derivation traces |
| `CurvatureTensor` | `(1, 3)` Riemann `R^a_{bcd}` |
| `RicciTensor` | `(0, 2)` Ricci `R_{ab}` |
| `EinsteinTensor` | `(0, 2)` Einstein `G_{ab}` |

## 2. Frame setup

### `CoordinateFrame` — the natural starting point

```python
from jacopy.frame_calc import CoordinateFrame
import sympy as sp

t, r, theta, phi = sp.symbols("t r theta phi")
F = CoordinateFrame([t, r, theta, phi])

print(F)                          # CoordinateFrame(name='coord(t,r,theta,phi)', dim=4)
print(F.dim)                      # 4
print(F.index_names())            # ('t', 'r', 'theta', 'phi')
print(F.derivative(r**2, 1))      # 2*r  (frame derivative on coord index 1)
print(F.gamma(0, 1, 0))           # 0    (coordinate frames are holonomic)
```

### `Tetrad` — orthonormal frames via vielbein

A tetrad sits on top of a coordinate frame; the vielbein matrix
prescribes how each tetrad vector decomposes into coord vectors.

```python
from jacopy.frame_calc import Tetrad

# A Schwarzschild orthonormal tetrad: each row is e_a^μ
# (this is just illustrative — you'd normally build it from the metric)
M, r = sp.symbols("M r", positive=True)
factor = 1 - 2*M/r
vielbein = sp.diag(
    1 / sp.sqrt(factor),
    sp.sqrt(factor),
    1 / r,
    1 / (r * sp.sin(sp.Symbol("theta", positive=True))),
)

coord_frame = CoordinateFrame(list(sp.symbols("t r theta phi")))
T = Tetrad(coord_frame, vielbein)
print(T.gamma(0, 1, 0))           # non-zero — γ from the vielbein
```

Most of the rest of this tutorial uses `CoordinateFrame` because
it's the more common case in physics literature. Everything below
also works on `Tetrad` — the `Frame` protocol is uniform.

## 3. `ComponentMetric` and `inverse()`

```python
from jacopy.frame_calc import ComponentMetric

t, r, theta, phi = sp.symbols("t r theta phi")
M = sp.Symbol("M", positive=True)
F = CoordinateFrame([t, r, theta, phi])

g = ComponentMetric(F, sp.Matrix([
    [-(1 - 2*M/r),   0,                0,    0],
    [0,              1/(1 - 2*M/r),    0,    0],
    [0,              0,                r**2, 0],
    [0,              0,                0,    r**2 * sp.sin(theta)**2],
]))

print(g[0, 0])                  # -(1 - 2*M/r)
print(g.det())                  # determinant
g_inv = g.inverse()
print(g_inv[0, 0])              # r/(2M - r)  — inverse metric component

# Verify g^{ac} g_{cb} = δ^a_b on a few entries
print(sum(g_inv[0, c] * g[c, 0] for c in range(4)))   # → 1
```

`ComponentMetric` checks symmetry at construction (`g_{ab} = g_{ba}`)
and rejects non-symmetric input.

## 4. `levi_civita(g)` — Christoffel symbols via Koszul

Stage D's deliverable: the unique torsion-free metric-compatible
connection.

```python
from jacopy.frame_calc import levi_civita

LC = levi_civita(g)
print(LC[0, 0, 1])              # Γ^t_{tr} = M / (r²(1-2M/r))
print(LC.nonzero_components())  # full table of non-zero Γ
```

For Schwarzschild, 13 non-zero Christoffel components — all
matching textbook values (`Γ^θ_{rθ} = 1/r`, `Γ^φ_{θφ} = cot θ`,
etc.).

### Step-by-step derivation transcript

Each Christoffel computation records its derivation as a list of
`KoszulStep`s — accessible via `derivation_steps` or the
human-readable `format_derivation`:

```python
print(LC.format_derivation(0, 0, 1))
# Γ^t_{tr}  (via Koszul formula)
# ──────────────────────────────
#   [1] Koszul formula
#       2 g(∇_{e_t} e_t, e_r) = e_t(g_{tc}) + e_t(g_{tc}) - e_c(g_{tt}) - …
#   [2] Frame-derivative terms
#       For each c: e_t(g_{tc}) + e_t(g_{tc}) - e_c(g_{tt})
#       = (0, 2*M/r**2, 0, 0)
#   [3] γ correction (coordinate frame)
#       γ^a_{bc} ≡ 0 for a holonomic frame; skipped.
#   [4] Multiply by ½ g^{ec} and contract over c
#       Γ^t_{tr} = ½ Σ_c g^{tc} · (frame-deriv + γ-terms)
#       = M*(2*M - r)/r**3
#   [5] Simplify
#       sympy.simplify
#       = M/(r*(-2*M + r))
```

### Bridge to `ProofChain` for paper-grade LaTeX

```python
from jacopy.display import chain_to_latex_document

chain = LC.derivation_chain(0, 0, 1)
print(chain_to_latex_document(chain))
# Full LaTeX document with:
#   - \begin{document}
#   - align* block with each step rendered via SymPy's latex
#   - \end{document}
```

The output is publication-ready — the same display pipeline used
by jacopy's other proof transcripts (Cartan magic, Bianchi, etc.).
A paper that mixes abstract operator-level proofs with
component-level Christoffel computations renders both in a
unified visual language.

## 5. Curvature, Ricci, Einstein

The full pipeline composes naturally:

```python
from jacopy.frame_calc import (
    curvature, ricci, ricci_scalar, einstein_tensor,
)

R = curvature(LC)               # Riemann (1, 3)
Ric = ricci(LC)                 # Ricci (0, 2)
R_scalar = ricci_scalar(LC, g)  # scalar
G = einstein_tensor(LC, g)      # Einstein (0, 2)

print(Ric.is_zero())            # True (Schwarzschild is Ricci-flat)
print(R_scalar)                 # 0
print(G.is_vacuum())            # True (vacuum field equations)
```

Each tensor's components are accessible via indexing
(`R[a, b, c, d]`, `Ric[a, b]`, `G[a, b]`). All carry derivation
traces in default mode.

### A note on sign convention

The Ricci tensor follows the convention from the operator
definition `R(U, V) W := ∇_U∇_V W − ∇_V∇_U W − ∇_{[U,V]} W`, with
contraction `Ric_{ab} := R^c_{acb}`. This gives Ricci with the
**opposite** sign of the Wald / Carroll physics convention — on a
2-sphere of radius `R₀`,

```python
Ric = -(1 / R₀²) g
ricci_scalar = -2 / R₀²
```

The Einstein-tensor vacuum condition (`G ≡ 0`) is
convention-independent; both terms in `G = Ric − ½ R g` flip sign
together. Schwarzschild's vacuum verification works in either
convention.

## 6. Optimised mode — Kerr-class metrics

For complex metrics like Kerr (off-diagonal `dt dφ` cross-term
plus `Σ = r² + a²cos²θ` denominators), per-entry `sympy.simplify`
calls on the inner Christoffel / curvature loops can blow up
exponentially. **Default mode times out on full Kerr Ricci**.

The fix: `optimized=True` skips per-entry simplify, keeping
expressions in raw form. Components remain mathematically correct
— `sympy.simplify` on access produces the clean form when needed.

```python
from jacopy.frame_calc.library import kerr

F, g = kerr()
LC = levi_civita(g, optimized=True)
G = einstein_tensor(LC, g, optimized=True)

# ~13 seconds for the full pipeline (vs 180s+ default mode)
assert G.is_vacuum()
# Surprise: G entries are *literal zero* in raw form —
# no simplify needed for the vacuum check.
assert G.is_zero(simplify=False)
```

The trade-off — optimised mode skips recording derivation traces.
`derivation_chain()` raises `RuntimeError` in optimised mode:

```python
LC.derivation_chain(0, 0, 1)
# RuntimeError: derivation_steps unavailable: this connection was
# built with optimized=True ...
```

Use **default mode** when you want paper-grade transcripts;
**optimised mode** when you want correct results on heavy metrics
in feasible time.

## 7. Library fixtures

Ready-made factories for the metrics most papers calibrate against.

```python
from jacopy.frame_calc.library import (
    minkowski, schwarzschild, frw, kerr,
)

# Minkowski 4D — flat
F, g = minkowski()                  # signature='-+++' default

# Schwarzschild — static spherical vacuum
F, g = schwarzschild()              # M is a positive Symbol

# FRW — homogeneous isotropic cosmology
F, g = frw()                        # k=0 (flat) default; a(t) is sp.Function
F, g = frw(k=1)                     # closed universe
F, g = frw(a_func=t**sp.Rational(2, 3))   # matter-dominated explicit

# Kerr — rotating vacuum (Boyer-Lindquist)
F, g = kerr()                       # M, a both positive Symbols
```

Each factory accepts optional `Symbol` / `Function` overrides so
the metric composes with user-supplied parameters.

## 8. ProofChain bridge — paper-grade rendering

The `proof_bridge` module wraps frame-calc derivation traces into
the same `ProofChain` data type the rest of jacopy uses, with
`provenance_tag="computation"`:

```python
from jacopy.frame_calc import steps_to_proof_chain
from jacopy.display import chain_to_latex_document

# A particular Christoffel's derivation as a ProofChain
chain = LC.derivation_chain(0, 0, 1)

# Same pipeline as jacopy's other proof transcripts:
print(chain_to_latex_document(chain))
```

The `SymPyAtom` opaque atom is the type bridge — it wraps SymPy
expressions inside jacopy `Expr` so they can sit in
`ProofStep.before / after` slots. The display layer registers a
LaTeX dispatcher for `SymPyAtom` that delegates to `sympy.latex()`
for clean rendering.

## Drop-in template — paste your metric, get everything

For paper work, the most common need is: "I have a metric on
some chart, give me Christoffels / Ricci / Einstein". Copy the
template below, **replace only the metric-matrix block**, and the
rest of the pipeline runs as-is on whatever metric you provided.

The example uses the **Reissner-Nordström** (charged Schwarzschild)
metric — not in the library because it's a different *family*
(non-vacuum, electromagnetic source). The point is to show that
you don't need a library factory: any metric matrix works.

```python
import sympy as sp
from jacopy.frame_calc import (
    CoordinateFrame, ComponentMetric,
    levi_civita, ricci, ricci_scalar, einstein_tensor,
)

# ─────────────────────────────────────────────────────────────
# 1. Coordinates — adjust to your metric's chart
# ─────────────────────────────────────────────────────────────
t, r, theta, phi = sp.symbols("t r theta phi")
coords = [t, r, theta, phi]

# Any extra parameters (mass, charge, cosmological constant, …):
M = sp.Symbol("M", positive=True)
Q = sp.Symbol("Q", positive=True)

# ─────────────────────────────────────────────────────────────
# 2. METRIC MATRIX — REPLACE THIS BLOCK WITH YOUR OWN
# ─────────────────────────────────────────────────────────────
# Reissner-Nordström: charged static spherical black hole
factor = 1 - 2*M/r + Q**2 / r**2
metric_matrix = sp.Matrix([
    [-factor,         0,        0,                          0],
    [0,        1/factor,        0,                          0],
    [0,               0,     r**2,                          0],
    [0,               0,        0,    r**2 * sp.sin(theta)**2],
])

# ─────────────────────────────────────────────────────────────
# 3. Pipeline — runs as-is on whatever metric is above
# ─────────────────────────────────────────────────────────────
F = CoordinateFrame(coords)
g = ComponentMetric(F, metric_matrix)
LC = levi_civita(g)
Ric = ricci(LC)
R = ricci_scalar(LC, g)
G = einstein_tensor(LC, g)

# ─────────────────────────────────────────────────────────────
# 4. Output — summary + all non-zero entries
# ─────────────────────────────────────────────────────────────
names = F.index_names()
print(f"# non-zero Christoffel: {len(LC.nonzero_components())}")
print(f"Ricci scalar R   = {sp.simplify(R)}")
print(f"Ric.is_zero()    = {Ric.is_zero()}")
print(f"G.is_vacuum()    = {G.is_vacuum()}")

print("\nChristoffel symbols (non-zero):")
for (e, a, b), val in LC.nonzero_components().items():
    print(f"  Γ^{names[e]}_{{{names[a]}{names[b]}}} = {val}")

print("\nEinstein tensor entries (non-zero):")
for a in range(F.dim):
    for b in range(a, F.dim):
        val = sp.simplify(sp.trigsimp(G[a, b]))
        if val != 0:
            print(f"  G_{{{names[a]}{names[b]}}} = {val}")
```

Output for Reissner-Nordström: 13 non-zero Christoffels,
``R_scalar = 0`` (a known property), Ricci non-zero
(non-vacuum), Einstein tensor with the four diagonal entries
``G_{tt}, G_{rr}, G_{θθ}, G_{φφ}`` carrying the electromagnetic
stress-energy form.

**To compute on a different metric**, change *only* block 2.
Examples you can drop in:

```python
# Schwarzschild-de Sitter (Λ ≠ 0): cosmological constant added
Lambda = sp.Symbol("Lambda")
factor = 1 - 2*M/r - Lambda*r**2/3
metric_matrix = sp.Matrix([
    [-factor,         0,        0,                          0],
    [0,        1/factor,        0,                          0],
    [0,               0,     r**2,                          0],
    [0,               0,        0,    r**2 * sp.sin(theta)**2],
])

# Anti-de Sitter in static coordinates: Λ < 0
# (just flip the sign of the Λr²/3 term)

# Vaidya (radiating): r → r and t → u (advanced time), m = m(u)
u = sp.Symbol("u")
m = sp.Function("m")(u)
metric_matrix = sp.Matrix([
    [-(1 - 2*m/r),   1,       0,                       0],
    [1,              0,       0,                       0],
    [0,              0,    r**2,                       0],
    [0,              0,       0,    r**2 * sp.sin(theta)**2],
])
# (use coords = [u, r, theta, phi])
```

Each of these runs through blocks 3-4 unchanged.

For Kerr-class metrics where the default-mode pipeline times out,
add ``optimized=True`` to every call:

```python
LC = levi_civita(g, optimized=True)
Ric = ricci(LC, optimized=True)
R = ricci_scalar(LC, g, optimized=True)
G = einstein_tensor(LC, g, optimized=True)
```

The output entries become raw (unsimplified) but mathematically
correct; ``G.is_vacuum()`` and other zero-checks still work via
SymPy's basic arithmetic. Use ``sp.simplify(LC[a, b, c])`` on the
specific entries you want to inspect.

## When to use `frame_calc` vs the rest of jacopy

| If you want… | Use… |
|---|---|
| Concrete metric → Christoffel / Ricci / Einstein components | `jacopy.frame_calc` |
| Abstract operator algebra (Cartan magic, Bianchi, derived bracket theorems) | the rest of `jacopy` |
| Both, in the same proof transcript | both — `proof_bridge` unifies them |

`frame_calc` is the answer to "I have a metric for my paper, give
me the tensors". The rest of jacopy is the answer to "I have an
identity I want to prove from axioms". Different problems,
different tools, but the same paper-grade output pipeline.

## Summary

* `jacopy.frame_calc` is the component-level submodule for
  concrete metric calculations.
* Three frame types (`CoordinateFrame`, `Tetrad`, `AbstractFrame`)
  share a common `Frame` protocol; higher-level operations are
  frame-agnostic.
* Pipeline: `g → LC → R → Ric → R_scalar → G`. Default mode
  records full derivation traces; optimised mode skips them for
  Kerr-class performance.
* Library fixtures (`minkowski`, `schwarzschild`, `frw`, `kerr`)
  cover the standard metrics; users can build custom metrics on
  any frame.
* `derivation_chain(...)` lifts any per-entry trace to a
  `ProofChain` for paper-grade LaTeX rendering through the
  existing display pipeline.
* SymPy is an opt-in dependency under the `[components]` extras.
