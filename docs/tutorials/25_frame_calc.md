# 25, Frame-component differential geometry (`jacopy.frame_calc`)

`jacopy.frame_calc` is jacopy's component-level submodule for
**concrete metric calculations**: given a metric ``g`` on a frame,
compute Christoffel symbols, Riemann curvature, Ricci tensor,
scalar curvature, Einstein tensor, with optional step-by-step
derivation transcripts that bridge to `ProofChain` for paper-grade
LaTeX output.

This tutorial covers:

1. The `jacopy.frame_calc` submodule layout, what it adds on top
   of jacopy's symbolic / proof layer.
2. Frame setup: `CoordinateFrame` (most common) and `Tetrad`
   (orthonormal bases via vielbein).
3. `ComponentMetric` and its `.inverse()`.
4. `levi_civita(g)`, Christoffel via Koszul formula.
5. `curvature`, `ricci`, `ricci_scalar`, `einstein_tensor`.
6. Optimised mode for Kerr-class metrics.
7. Library fixtures, `minkowski`, `schwarzschild`, `frw`, `kerr`.
8. ProofChain bridge: rendering derivations to paper-grade LaTeX.

`frame_calc` requires **SymPy as an optional dependency** under
the `[components]` extras. Install via:

```bash
pip install "jacopy[components]"
```

## A quick taste, Schwarzschild vacuum in five lines

```python
from jacopy.frame_calc import einstein_tensor, levi_civita
from jacopy.frame_calc.library import schwarzschild

F, g = schwarzschild()
G = einstein_tensor(levi_civita(g), g)
assert G.is_vacuum()
```

That's it, `jacopy` symbolically computes that Schwarzschild's
Einstein tensor vanishes identically, the defining property of a
vacuum solution to Einstein's field equations.

## 1. Submodule layout

`jacopy.frame_calc` is **separate from jacopy's symbolic / proof
layer**, it does component-level numerical-symbolic computation,
where the rest of jacopy does abstract operator algebra on
brackets / Cartan calculus / etc. The two layers complement each
other through the `proof_bridge` module (Stage G).

Pieces you'll encounter:

| Type | What it is |
|---|---|
| `Frame` (protocol) | The interface every frame implementation satisfies |
| `CoordinateFrame` | `e_a = ∂/∂x^a`, most common |
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

### `CoordinateFrame`, the natural starting point

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

### `Tetrad`, orthonormal frames via vielbein

A tetrad sits on top of a coordinate frame; the vielbein matrix
prescribes how each tetrad vector decomposes into coord vectors.

```python
from jacopy.frame_calc import Tetrad

# A Schwarzschild orthonormal tetrad: each row is e_a^μ
# (this is just illustrative, you'd normally build it from the metric)
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
print(T.gamma(0, 1, 0))           # non-zero, γ from the vielbein
```

Most of the rest of this tutorial uses `CoordinateFrame` because
it's the more common case in physics literature. Everything below
also works on `Tetrad`, the `Frame` protocol is uniform.

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
print(g_inv[0, 0])              # r/(2M - r) , inverse metric component

# Verify g^{ac} g_{cb} = δ^a_b on a few entries
print(sum(g_inv[0, c] * g[c, 0] for c in range(4)))   # → 1
```

`ComponentMetric` checks symmetry at construction (`g_{ab} = g_{ba}`)
and rejects non-symmetric input.

## 4. `levi_civita(g)`, Christoffel symbols via Koszul

Stage D's deliverable: the unique torsion-free metric-compatible
connection.

```python
from jacopy.frame_calc import levi_civita

LC = levi_civita(g)
print(LC[0, 0, 1])              # Γ^t_{tr} = M / (r²(1-2M/r))
print(LC.nonzero_components())  # full table of non-zero Γ
```

For Schwarzschild, 13 non-zero Christoffel components, all
matching textbook values (`Γ^θ_{rθ} = 1/r`, `Γ^φ_{θφ} = cot θ`,
etc.).

### Step-by-step derivation transcript

Each Christoffel computation records its derivation as a list of
`KoszulStep`s, accessible via `derivation_steps` or the
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

The output is publication-ready, the same display pipeline used
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
**opposite** sign of the Wald / Carroll physics convention, on a
2-sphere of radius `R₀`,

```python
Ric = -(1 / R₀²) g
ricci_scalar = -2 / R₀²
```

The Einstein-tensor vacuum condition (`G ≡ 0`) is
convention-independent; both terms in `G = Ric − ½ R g` flip sign
together. Schwarzschild's vacuum verification works in either
convention.

## 6. Optimised mode, Kerr-class metrics

For complex metrics like Kerr (off-diagonal `dt dφ` cross-term
plus `Σ = r² + a²cos²θ` denominators), per-entry `sympy.simplify`
calls on the inner Christoffel / curvature loops can blow up
exponentially. **Default mode times out on full Kerr Ricci**.

The fix: `optimized=True` skips per-entry simplify, keeping
expressions in raw form. Components remain mathematically correct
, `sympy.simplify` on access produces the clean form when needed.

```python
from jacopy.frame_calc.library import kerr

F, g = kerr()
LC = levi_civita(g, optimized=True)
G = einstein_tensor(LC, g, optimized=True)

# ~13 seconds for the full pipeline (vs 180s+ default mode)
assert G.is_vacuum()
# Surprise: G entries are *literal zero* in raw form,
# no simplify needed for the vacuum check.
assert G.is_zero(simplify=False)
```

The trade-off, optimised mode skips recording derivation traces.
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

# Minkowski 4D, flat
F, g = minkowski()                  # signature='-+++' default

# Schwarzschild, static spherical vacuum
F, g = schwarzschild()              # M is a positive Symbol

# FRW, homogeneous isotropic cosmology
F, g = frw()                        # k=0 (flat) default; a(t) is sp.Function
F, g = frw(k=1)                     # closed universe
F, g = frw(a_func=t**sp.Rational(2, 3))   # matter-dominated explicit

# Kerr, rotating vacuum (Boyer-Lindquist)
F, g = kerr()                       # M, a both positive Symbols
```

Each factory accepts optional `Symbol` / `Function` overrides so
the metric composes with user-supplied parameters.

## 8. ProofChain bridge, paper-grade rendering

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

The `SymPyAtom` opaque atom is the type bridge, it wraps SymPy
expressions inside jacopy `Expr` so they can sit in
`ProofStep.before / after` slots. The display layer registers a
LaTeX dispatcher for `SymPyAtom` that delegates to `sympy.latex()`
for clean rendering.

## Drop-in template, paste your metric, get everything

For paper work, the most common need is: "I have a metric on
some chart, give me Christoffels / Ricci / Einstein". Copy the
template below, **replace only the metric-matrix block**, and the
rest of the pipeline runs as-is on whatever metric you provided.

The example uses the **Reissner-Nordström** (charged Schwarzschild)
metric, not in the library because it's a different *family*
(non-vacuum, electromagnetic source). The point is to show that
you don't need a library factory: any metric matrix works.

```python
import sympy as sp
from jacopy.frame_calc import (
    CoordinateFrame, ComponentMetric,
    levi_civita, ricci, ricci_scalar, einstein_tensor,
)

# ─────────────────────────────────────────────────────────────
# 1. Coordinates, adjust to your metric's chart
# ─────────────────────────────────────────────────────────────
t, r, theta, phi = sp.symbols("t r theta phi")
coords = [t, r, theta, phi]

# Any extra parameters (mass, charge, cosmological constant, …):
M = sp.Symbol("M", positive=True)
Q = sp.Symbol("Q", positive=True)

# ─────────────────────────────────────────────────────────────
# 2. METRIC MATRIX, REPLACE THIS BLOCK WITH YOUR OWN
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
# 3. Pipeline, runs as-is on whatever metric is above
# ─────────────────────────────────────────────────────────────
F = CoordinateFrame(coords)
g = ComponentMetric(F, metric_matrix)
LC = levi_civita(g)
Ric = ricci(LC)
R = ricci_scalar(LC, g)
G = einstein_tensor(LC, g)

# ─────────────────────────────────────────────────────────────
# 4. Output, summary + all non-zero entries
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

## Custom connection, independent of the metric

A **connection** and a **metric** are two independent geometric
objects. The Levi-Civita connection is the *unique* connection
that's both **torsion-free** and **metric-compatible** for a
given metric, but it's just one of many possible connections on
the same manifold. In Einstein-Cartan theory, teleparallel
gravity, Palatini formulations, and other modified gravity
frameworks, the connection is **not** Levi-Civita.

`einstein_tensor(connection, g)` accepts **any**
`ComponentConnection`, not just `LeviCivitaConnection`. So you
can:

- Build a connection with **arbitrary Christoffel symbols** via
  `ComponentConnection(F, christoffel_table)`.
- Compute the resulting Einstein tensor under that connection.
- Compare against Levi-Civita to see how torsion / non-metricity
  changes the answer.

### Symbol-domain matching (important pitfall)

When you supply Christoffel symbols by hand, **use the symbols
the frame already carries**, not freshly-created ones. Library
factories like `schwarzschild()` create symbols with specific
assumptions (`r > 0`, `M > 0`); your hand-written `sp.symbols('r')`
is a *different* symbol object even though it shares the name.

```python
F, g = schwarzschild()
t, r, theta, phi = F.coords          # ← use these
M = sp.Symbol("M", positive=True)    # ← assumption must match factory's
```

If you skip this, your Christoffel formulas will reference
"phantom" symbols and `einstein_tensor` will produce nonsense.

### Sanity check, manual Schwarzschild matches Levi-Civita

Build the textbook Schwarzschild Christoffels by hand, wrap them
in `ComponentConnection`, and verify the result matches
`levi_civita(g)` exactly:

```python
import sympy as sp
from jacopy.frame_calc import (
    ComponentConnection, einstein_tensor, levi_civita,
)
from jacopy.frame_calc.library import schwarzschild

F, g = schwarzschild()
t, r, theta, phi = F.coords
M = sp.Symbol("M", positive=True)

# Pre-allocated zero array for the (1, 2)-tensor
manual = sp.MutableDenseNDimArray.zeros(F.dim, F.dim, F.dim)

# Set the 13 non-zero entries, textbook Schwarzschild
factor = 1 - 2*M/r
val = M / (r**2 * factor)
manual[0, 0, 1] = val          # Γ^t_tr
manual[0, 1, 0] = val          # Γ^t_rt
manual[1, 0, 0] = M*factor/r**2   # Γ^r_tt
manual[1, 1, 1] = -M / (r**2 * factor)   # Γ^r_rr
manual[1, 2, 2] = -(r - 2*M)               # Γ^r_θθ
manual[1, 3, 3] = -(r - 2*M)*sp.sin(theta)**2  # Γ^r_φφ
manual[2, 1, 2] = manual[2, 2, 1] = 1/r    # Γ^θ_rθ, Γ^θ_θr
manual[2, 3, 3] = -sp.sin(theta)*sp.cos(theta)  # Γ^θ_φφ
manual[3, 1, 3] = manual[3, 3, 1] = 1/r    # Γ^φ_rφ, Γ^φ_φr
manual[3, 2, 3] = manual[3, 3, 2] = (
    sp.cos(theta) / sp.sin(theta)
)   # Γ^φ_θφ, Γ^φ_φθ

manual_conn = ComponentConnection(F, manual)
LC = levi_civita(g)

# Entry-by-entry comparison
for a in range(F.dim):
    for b in range(F.dim):
        for c in range(F.dim):
            assert sp.simplify(
                sp.trigsimp(LC[a, b, c] - manual_conn[a, b, c])
            ) == 0

# Both produce the same vacuum Einstein tensor
assert einstein_tensor(LC, g).is_vacuum()
assert einstein_tensor(manual_conn, g).is_vacuum()
```

This is the smoke test, your hand-written Christoffels match
the package's Levi-Civita output, so the API is consistent.

### Non-trivial use: same metric, different connection

For modified-gravity work, you'd add a torsion correction or use
a fully independent connection. Here's a connection that's the
Schwarzschild Levi-Civita **plus a torsion term**, the same
metric, but the Einstein tensor is no longer vacuum because the
connection is no longer torsion-free:

```python
# Levi-Civita as the baseline
LC = levi_civita(g)

# Add an antisymmetric correction: T^t_{rθ} = sin θ
new_christoffel = sp.MutableDenseNDimArray.zeros(F.dim, F.dim, F.dim)
for a in range(F.dim):
    for b in range(F.dim):
        for c in range(F.dim):
            new_christoffel[a, b, c] = LC[a, b, c]

# Antisymmetric torsion perturbation
new_christoffel[0, 1, 2] += sp.Rational(1, 2) * sp.sin(theta)
new_christoffel[0, 2, 1] -= sp.Rational(1, 2) * sp.sin(theta)

torsion_conn = ComponentConnection(F, new_christoffel)
G_with_torsion = einstein_tensor(torsion_conn, g)

print("Levi-Civita G.is_vacuum():     ", einstein_tensor(LC, g).is_vacuum())
print("Custom-torsion G.is_vacuum():  ", G_with_torsion.is_vacuum())
# → True / False, adding torsion breaks vacuum
```

The `einstein_tensor(connection, g)` call doesn't care where the
Christoffel symbols came from, it just computes
`G_{ab} = Ric_{ab} - ½ R g_{ab}` from whatever connection you
supply.

### When you'd actually use this

| Scenario | Why custom connection |
|---|---|
| Standard GR (vacuum, Einstein-Maxwell, Schwarzschild family) | Use Levi-Civita, `levi_civita(g)` |
| Einstein-Cartan theory | Connection has torsion, supply Christoffels with `T ≠ 0` |
| Teleparallel gravity | Connection is flat (`R = 0`) but has torsion |
| Palatini formulation | Vary `g` and `Γ` independently |
| Affine theory (no metric) | Connection alone determines the geometry |

For the standard-GR cases the metric → Levi-Civita → tensors
chain is all you need. The custom-connection path opens up when
the physics requires it.

### API stress test, arbitrary symbols

Before showing physically-motivated patterns, here is what
happens with **completely arbitrary** symbol parameters. Useful
to verify the API accepts whatever you throw at it; not useful
for paper work.

**Schwarzschild metric, made-up A, B:**

```python
F_sw, g_sw = schwarzschild()
t_sw, r_sw, theta_sw, phi_sw = F_sw.coords
A, B = sp.symbols("A B")

manual = sp.MutableDenseNDimArray.zeros(F_sw.dim, F_sw.dim, F_sw.dim)
manual[0, 0, 1] = manual[0, 1, 0] = A
manual[1, 0, 0] = B
manual[1, 1, 1] = -B
manual[2, 1, 2] = manual[2, 2, 1] = 1 / r_sw
manual[3, 1, 3] = manual[3, 3, 1] = 1 / r_sw

manual_conn = ComponentConnection(F_sw, manual)
G_manual = einstein_tensor(manual_conn, g_sw)
# G's entries are non-zero functions of A, B (4 diagonal)
```

The result is `G_{tt}, G_{rr}, G_{θθ}, G_{φφ}` non-zero with
arbitrary A,B-dependence. Plug `A`, `B` into actual Levi-Civita
values to recover vacuum, but **only the angular block is
already filled** (1/r); the diagonal Schwarzschild Christoffels
that A, B replace need *full* Levi-Civita formulas, otherwise
the result stays non-vacuum.

**2D polar metric, made-up A, B:**

```python
x, y = sp.symbols("x y")
A, B = sp.symbols("A B")
F = CoordinateFrame([x, y])
g = ComponentMetric(F, sp.Matrix([[1, 0], [0, x**2]]))

manual = sp.MutableDenseNDimArray.zeros(F.dim, F.dim, F.dim)
manual[0, 0, 0] = A                 # Γ^x_{xx}
manual[0, 1, 1] = B*x               # Γ^x_{yy}
manual[1, 0, 1] = 1/x + A           # Γ^y_{xy}
manual[1, 1, 0] = 1/x - A           # Γ^y_{yx}

manual_conn = ComponentConnection(F, manual)
G = einstein_tensor(manual_conn, g)
# G_{xx} = -A/(2x), G_{yy} = A·x/2  ← B doesn't appear!
```

**The interesting fact**: `B` doesn't appear in `G` at all. The
2D Lovelock theorem guarantees `G ≡ 0` for any **torsion-free**
connection. Here `T^y_{xy} = Γ^y_{xy} − Γ^y_{yx} = 2A`, so
`α=0` is the torsion-free condition. When you set `A=0`, `G`
collapses to zero regardless of `B`. The arbitrary-parameter
API test accidentally **verifies Lovelock's 2D theorem**
symbolically.

This is informative but not paper-grade input. For paper work,
use one of the physically-motivated patterns below.

### Physically-motivated deformation patterns

These are the patterns you'd actually find in modified-gravity
papers. Each is a Levi-Civita connection plus a specific
deformation tensor parameterised by a small number of physical
quantities.

#### Pattern 1: Levi-Civita + antisymmetric torsion (2D)

A single scalar `α` controls a torsion-violating perturbation:

```python
x, y = sp.symbols("x y")
alpha = sp.Symbol("alpha", real=True)
F = CoordinateFrame([x, y])
g = ComponentMetric(F, sp.Matrix([[1, 0], [0, x**2]]))
LC = levi_civita(g)

Gamma = sp.MutableDenseNDimArray(LC.components)
Gamma[0, 0, 1] += alpha             # Γ^x_{xy}
Gamma[0, 1, 0] -= alpha             # Γ^x_{yx} (antisymmetric → torsion)

torsion_conn = ComponentConnection(F, Gamma)
G_torsion = einstein_tensor(torsion_conn, g)
```

**Output**: `G_{xx} = α²/(2x²)`, `G_{xy} = G_{yx} = -α/x`,
`G_{yy} = -α²/2`. At `α = 0` recovers Levi-Civita (vacuum, by
Lovelock 2D).

#### Pattern 2: Levi-Civita + Weyl non-metricity (2D)

Single scalar `W_x` controls a Weyl-type deformation:

```python
W_x = sp.Symbol("W_x", real=True)
g_mat = g.matrix()
g_inv = g_mat.inv()
W = [W_x, 0]
W_up = [sum(g_inv[a, b] * W[b] for b in range(F.dim)) for a in range(F.dim)]

Gamma = sp.MutableDenseNDimArray(LC.components)
for a in range(F.dim):
    for b in range(F.dim):
        for c in range(F.dim):
            δ_ab = 1 if a == b else 0
            δ_ac = 1 if a == c else 0
            Gamma[a, b, c] += sp.Rational(1, 2) * (
                δ_ab * W[c] + δ_ac * W[b] - g_mat[b, c] * W_up[a]
            )

weyl_conn = ComponentConnection(F, Gamma)
G_weyl = einstein_tensor(weyl_conn, g)
```

**Output**: `G ≡ 0` even though the connection is non-metric.
The Weyl deformation **preserves** torsion-freeness, so
Lovelock still applies in 2D. Pedagogically: torsion is what
breaks Lovelock, not non-metricity.

#### Pattern 3: Schwarzschild + antisymmetric torsion (4D)

Single scalar `ε` adds a `t-φ` cross-term torsion:

```python
F_sw, g_sw = schwarzschild()
epsilon = sp.Symbol("epsilon", real=True)
LC = levi_civita(g_sw)

Gamma = sp.MutableDenseNDimArray(LC.components)
Gamma[3, 1, 0] += epsilon           # Γ^φ_{rt}
Gamma[3, 0, 1] -= epsilon           # Γ^φ_{tr}

torsion_conn = ComponentConnection(F_sw, Gamma)
G = einstein_tensor(torsion_conn, g_sw)
```

**Output**: only **two** non-zero entries:
``G_{tφ} = ε(-2M+r) sin²θ`` and its mirror `G_{φt}`. Compact
torsion-driven correction to vacuum Schwarzschild.

#### Pattern 4: Schwarzschild + Weyl non-metricity (4D)

Same Weyl construction, on Schwarzschild. **Use `optimized=True`
because the 4D pipeline is slower:**

```python
W_r = sp.Symbol("W_r", real=True)
LC = levi_civita(g_sw, optimized=True)
g_mat = g_sw.matrix()
g_inv = g_mat.inv()
W = [0, W_r, 0, 0]
W_up = [sum(g_inv[μ, ν] * W[ν] for ν in range(F_sw.dim))
        for μ in range(F_sw.dim)]

Gamma = sp.MutableDenseNDimArray(LC.components)
for μ in range(F_sw.dim):
    for ν in range(F_sw.dim):
        for ρ in range(F_sw.dim):
            δ_μν = 1 if μ == ν else 0
            δ_μρ = 1 if μ == ρ else 0
            Gamma[μ, ν, ρ] += sp.Rational(1, 2) * (
                δ_μν * W[ρ] + δ_μρ * W[ν] - g_mat[ν, ρ] * W_up[μ]
            )

weyl_conn = ComponentConnection(F_sw, Gamma)
G = einstein_tensor(weyl_conn, g_sw, optimized=True)
```

**Output**: four diagonal `G` entries, polynomial in `W_r`,
`M`, `r`, plus `sin²θ` for `G_{φφ}`. At `W_r = 0` recovers
vacuum.

#### Pattern 5: FLRW + scalar-gradient projective deformation

Connection deformed by a scalar field's gradient, a typical
scalar-tensor gravity setup. Coupling form `Γ + δ A_a + δ A_a`:

```python
t, r, theta, phi = sp.symbols("t r theta phi")
a_func = sp.Function("a")(t)
varphi = sp.Function("varphi")(t)
F = CoordinateFrame([t, r, theta, phi])

g = ComponentMetric(F, sp.Matrix([
    [-1, 0, 0, 0],
    [0, a_func**2, 0, 0],
    [0, 0, a_func**2 * r**2, 0],
    [0, 0, 0, a_func**2 * r**2 * sp.sin(theta)**2],
]))
LC = levi_civita(g)

A = [sp.diff(varphi, c) for c in F.coords]    # gradient ∂_μ φ
Gamma = sp.MutableDenseNDimArray(LC.components)
for μ in range(F.dim):
    for ν in range(F.dim):
        for ρ in range(F.dim):
            δ_μν = 1 if μ == ν else 0
            δ_μρ = 1 if μ == ρ else 0
            Gamma[μ, ν, ρ] += δ_μν * A[ρ] + δ_μρ * A[ν]

scalar_conn = ComponentConnection(F, Gamma)
G_def = einstein_tensor(scalar_conn, g)
```

**Output**: full `G` with `Derivative(a, t)`, `Derivative(varphi, t)`,
and second derivatives. Modified Friedmann equations form. At
`varphi'(t) = 0` recovers FLRW Levi-Civita.

### Insight summary

Five patterns, three structural lessons:

| Pattern | Recovers Levi-Civita at | Lovelock 2D collapse? |
|---|---|---|
| 2D + α torsion | `α = 0` | No (torsion breaks it) |
| 2D + W_x Weyl | `W_x = 0` | **Yes** (`G ≡ 0` always) |
| Schwarzschild + ε torsion | `ε = 0` | n/a (4D) |
| Schwarzschild + W_r Weyl | `W_r = 0` | n/a (4D) |
| FLRW + scalar-grad | `varphi'(t) = 0` | n/a (4D, non-vacuum baseline) |

Two concrete takeaways:

1. **API stress-tests with arbitrary symbols** are valid and
   accidentally surface deep theorems (the A, B examples
   verifying Lovelock 2D).
2. **Physically-meaningful examples** parameterise the
   deformation by a small number of fields/constants, with the
   `parameter → 0` limit recovering Levi-Civita. This is the
   pattern you'll find in modified-gravity papers.

## When to use `frame_calc` vs the rest of jacopy

| If you want… | Use… |
|---|---|
| Concrete metric → Christoffel / Ricci / Einstein components | `jacopy.frame_calc` |
| Abstract operator algebra (Cartan magic, Bianchi, derived bracket theorems) | the rest of `jacopy` |
| Both, in the same proof transcript | both, `proof_bridge` unifies them |

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
