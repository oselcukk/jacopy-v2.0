# Faz 18, `jacopy.frame_calc`

> Frame-component differential geometry calculations: given a frame
> + metric (and optionally a connection), compute Christoffel
> symbols / torsion / curvature / Ricci tensor / Ricci scalar /
> Einstein tensor, with **step-by-step derivation transcripts**
> bridged through `ProofChain`.

## Goal

The professor's notes (`Frame expressions` document) define every
geometric quantity through frame-component formulas:

| Quantity | Formula |
|---|---|
| `T^a_{bc}` | `Γ^a_{bc} − Γ^a_{cb} − γ^a_{bc}` |
| `R^a_{bcd}` | `e_b(Γ^a_{cd}) − e_c(Γ^a_{bd}) + Γ^e_{cd}Γ^a_{be} − Γ^e_{bd}Γ^a_{ce} − γ^e_{bc}Γ^a_{ed}` |
| `Γ(^g∇)^e_{ab}` | `½ g^{ec}(e_a(g_{bc}) + e_b(g_{ac}) − e_c(g_{ab}) − γ^d_{bc}g_{da} − γ^d_{ac}g_{db} + γ^d_{ab}g_{dc})` |
| `Ric_{ab}` | `R^c_{acb}` |
| `R` | `Ric_{ab} g^{ab}` |
| `G_{ab}` | `Ric_{ab} − ½ R g_{ab}` |

`jacopy.frame_calc` exposes these as **mechanical computations**
that the user can run, inspect step by step, and embed into
larger proofs. The module is **not** about re-proving the
underlying identities, those are taken as given.

## Architecture

### Two-tier evaluation

The module supports the same API at two levels:

- **Concrete (SymPy-backed):** when `g_{ab}` is given as explicit
  coordinate functions (e.g. `−(1−2M/r)`), SymPy handles matrix
  inverse, partial derivatives, and simplification. Components
  return as SymPy expressions.
- **Symbolic (opaque):** when `g_{ab}` is left as abstract symbols
  (`g_00`, `g_01`, …) or when the frame is `AbstractFrame`,
  components remain symbolic. The inverse metric and frame
  derivatives surface as opaque `Expr` nodes that downstream
  formulas keep symbolic.

Both modes share the same `Frame`, `ComponentMetric`,
`LeviCivita`, etc. classes. The mode is determined by the frame
type and metric input, the user never explicitly picks "concrete
vs symbolic".

### SymPy as opt-in dependency

`pyproject.toml` will gain:

```toml
[project.optional-dependencies]
components = ["sympy>=1.12"]
```

Importing `jacopy.frame_calc` raises a friendly `ImportError` if
SymPy is missing **and** the user tries to use a concrete-frame
feature. Pure abstract-frame work doesn't need SymPy.

### Bridge to `ProofChain`

Every component-level computation also produces a structured
derivation chain:

```python
LC[a, b, c]                        # → SymPy expression (or opaque Expr)
LC.derivation_chain(a, b, c)       # → ProofChain compatible with chain_to_latex_document
```

The chain steps cite the formula being applied
(`"Koszul formula on (e_a, e_b, e_c)"`), the substitutions
performed (`"Substitute g_{ab} = ..."`), and the simplifications.
Same visual language as Cartan magic / Bianchi proofs in the
rest of the package.

### Frame compatibility, the key invariant

All three frame types (`CoordinateFrame`, `Tetrad`,
`AbstractFrame`) implement the same `Frame` protocol. Higher-level
code (`ComponentMetric.inverse()`, `LeviCivita`, `Curvature`,
`Ricci`, …) is **frame-agnostic**, it goes through the protocol
and never branches on frame type.

This invariant is what lets us start with `CoordinateFrame` (Stage
A focus) without painting ourselves into a corner for `Tetrad` /
`AbstractFrame` (Stages B+).

The protocol:

```python
class Frame(Protocol):
    dim: int
    name: str

    def index_names(self) -> tuple[str, ...]:
        """Display names for the frame indices, e.g. ('t', 'r', 'θ', 'φ')."""

    def derivative(self, expr, a: int) -> Expr:
        """The action `e_a(expr)`, a function-on-manifold derivation.

        - CoordinateFrame: returns SymPy ∂expr/∂x^a.
        - Tetrad: returns e_a^μ ∂expr/∂x^μ via vielbein.
        - AbstractFrame: returns an opaque FrameDerivativeExpr(self, a, expr).
        """

    def gamma(self, a: int, b: int, c: int) -> Expr:
        """Lie bracket structure constant γ^a_{bc}.

        - CoordinateFrame: always 0.
        - Tetrad: computed from vielbein and SymPy bracket [e_b, e_c].
        - AbstractFrame: opaque GammaExpr(self, a, b, c) by default;
          user-supplied table if provided.
        """
```

This is the **only** interface that `LeviCivita`, `Curvature`,
etc. depend on. As long as a frame implements `derivative` and
`gamma`, all formulas work.

## Stages

### Stage A, Frame infrastructure (`CoordinateFrame`-first)

**Files:**
- `jacopy/frame_calc/__init__.py`
- `jacopy/frame_calc/frame.py`

**Deliverables:**

```python
# Protocol (informal, duck-typed, not strict ABC)
class Frame:
    dim: int
    name: str
    def index_names(self) -> tuple[str, ...]: ...
    def derivative(self, expr, a: int) -> Expr: ...
    def gamma(self, a: int, b: int, c: int) -> Expr: ...

class CoordinateFrame(Frame):
    """Frame `e_a = ∂/∂x^a` for a coordinate chart."""
    def __init__(self, coords: list[sp.Symbol]):
        self.coords = coords
        self.dim = len(coords)
        self.name = f"coord({','.join(str(c) for c in coords)})"

    def derivative(self, expr, a: int) -> sp.Expr:
        return sp.diff(expr, self.coords[a])

    def gamma(self, a: int, b: int, c: int) -> sp.Expr:
        return sp.Integer(0)   # coordinate frames are torsion-free

class AbstractFrame(Frame):
    """Stage A stub. Full implementation in Stage A.2, see below."""
    # at Stage A: defined as a placeholder so type checks pass and
    # downstream modules can import it without breaking.
    def derivative(self, expr, a: int):
        raise NotImplementedError("AbstractFrame populated at Stage A.2")
    def gamma(self, a: int, b: int, c: int):
        raise NotImplementedError("AbstractFrame populated at Stage A.2")

class Tetrad(Frame):
    """Stage B, left as a placeholder import-ably in Stage A."""
    def derivative(self, *_):
        raise NotImplementedError("Tetrad populated at Stage B")
    def gamma(self, *_):
        raise NotImplementedError("Tetrad populated at Stage B")
```

**Tests:**
- `tests/test_frame_calc/test_coordinate_frame.py`
  - `CoordinateFrame([t, r])` builds; `dim == 2`
  - `frame.derivative(r**2, 1) == 2*r`
  - `frame.gamma(0, 1, 0) == 0`
- Stub tests for `AbstractFrame` / `Tetrad` raise `NotImplementedError`.

**Compatibility note:** `CoordinateFrame` returns SymPy types from
`derivative` / `gamma`. `AbstractFrame` (Stage A.2) will return
`jacopy.core.expr.Expr` types instead, wrapped in a thin
`FrameDerivativeExpr` / `GammaExpr` if needed. Higher-level code
needs to handle both, so we'll route everything through a uniform
`as_sympy(expr) → sp.Expr` and `as_jacopy(expr) → Expr` adapter
once Stage A.2 lands.

### Stage A.2, `AbstractFrame` implementation

**Files:**
- `jacopy/frame_calc/abstract_frame.py`
- `jacopy/frame_calc/symbolic_atoms.py`

**Deliverables:**

```python
class FrameDerivativeExpr(Expr):
    """Opaque atom for `e_a(f)` when `f` is symbolic and frame abstract."""
    def __init__(self, frame: AbstractFrame, a: int, body: Expr):
        ...

class GammaExpr(Expr):
    """Opaque atom for γ^a_{bc} when frame is abstract and no table given."""
    def __init__(self, frame: AbstractFrame, a: int, b: int, c: int):
        ...

class AbstractFrame(Frame):
    """User-supplied Lie bracket structure constants; frame derivative opaque."""
    def __init__(
        self,
        dim: int,
        *,
        gamma_table: dict[tuple[int, int, int], Expr] | None = None,
        index_names: tuple[str, ...] | None = None,
    ):
        ...

    def derivative(self, expr, a):
        return FrameDerivativeExpr(self, a, expr)

    def gamma(self, a, b, c):
        if self._table is None:
            return GammaExpr(self, a, b, c)
        return self._table.get((a, b, c), Integer(0))
```

**Cross-frame compatibility:** `CoordinateFrame.derivative` returns
SymPy; `AbstractFrame.derivative` returns `Expr`. Stage C
(`ComponentMetric`) introduces an adapter that lifts SymPy ↔ Expr
as needed so downstream formulas stay frame-agnostic.

**Tests:**
- `frame.derivative(g_00, 1) == FrameDerivativeExpr(frame, 1, g_00)`
- `frame.gamma(0, 1, 2)` returns the table value or a `GammaExpr`
  if absent.

### Stage B, `Tetrad` implementation

**Files:**
- `jacopy/frame_calc/tetrad.py`

**Deliverables:**

```python
class Tetrad(Frame):
    """Tetrad `e_a = e_a^μ ∂/∂x^μ` defined by a vielbein matrix."""
    def __init__(
        self,
        coord_frame: CoordinateFrame,
        vielbein: sp.Matrix,    # shape (dim, dim): row a is e_a^μ
    ):
        ...

    def derivative(self, expr, a):
        # e_a(f) = e_a^μ ∂f/∂x^μ
        return sum(
            self.vielbein[a, mu] * self.coord_frame.derivative(expr, mu)
            for mu in range(self.dim)
        )

    def gamma(self, a, b, c):
        # γ^a_{bc} = e^a([e_b, e_c]) computed via vielbein
        # [e_b, e_c]^μ = e_b^ν ∂_ν e_c^μ − e_c^ν ∂_ν e_b^μ
        # γ^a_{bc} = (e^{-1})^a_μ [e_b, e_c]^μ
        ...
```

**Tests:**
- Schwarzschild orthonormal tetrad → diagonal Minkowski `g_{ab}`.
- `γ` reduces to 0 when vielbein is identity.

### Stage C, Component tensors

**Files:**
- `jacopy/frame_calc/component_tensor.py`

**Deliverables:**

```python
class ComponentTensor:
    """A (q, r)-tensor by frame components.

    Stored as a SymPy `Array` of shape `(dim,) * (q + r)`.
    Use ComponentMetric / ComponentConnection / etc. for typed wrappers.
    """
    def __init__(self, frame: Frame, components: sp.Array, signature: tuple[int, int]):
        ...

    def __getitem__(self, idx) -> sp.Expr | Expr: ...

    def matrix(self) -> sp.Matrix: ...    # only for (0,2) and (2,0)


class ComponentMetric(ComponentTensor):
    """A (0, 2) symmetric metric `g_{ab}`."""
    def __init__(self, frame: Frame, matrix: sp.Matrix):
        # Verify shape, symmetry (best-effort, abstract entries skip check)
        super().__init__(frame, sp.Array(matrix), signature=(0, 2))

    def inverse(self) -> "ComponentMetricInverse":
        """Compute g^{ab}.

        Concrete frame: SymPy `matrix.inv()`.
        Abstract entries: returns ComponentMetricInverse with opaque g_inv[a, b]
        symbols whose product with self satisfies g^{ac} g_{cb} = δ^a_b axiomatically.
        """
        ...


class ComponentMetricInverse(ComponentTensor):
    """A (2, 0) tensor: the inverse metric `g^{ab}`."""


class ComponentConnection(ComponentTensor):
    """A connection given by Christoffel symbols `Γ^a_{bc}`.

    Stored as a (1, 2) tensor of shape (dim, dim, dim), index ordering: [a, b, c].
    """
```

**Tests:**
- `ComponentMetric(F, Matrix(...)).inverse().inverse() == original`
- `ComponentMetric` accepts symmetric matrices; raises on
  non-symmetric (concrete entries only).

### Stage D, Levi-Civita / Koszul

**Files:**
- `jacopy/frame_calc/levi_civita.py`

**Deliverables:**

```python
def levi_civita(g: ComponentMetric) -> ComponentConnection:
    """Compute Christoffel symbols via Koszul formula.

    For each (e, a, b), apply
        Γ^e_{ab} = ½ g^{ec}(
            e_a(g_{bc}) + e_b(g_{ac}) - e_c(g_{ab})
            - γ^d_{bc} g_{da} - γ^d_{ac} g_{db} + γ^d_{ab} g_{dc}
        ).

    The implementation:
      1. Computes g_inv = g.inverse() once.
      2. For each (e, a, b), enumerates the contraction sums over c, d.
      3. Returns a ComponentConnection whose [e, a, b] entry is the simplified
         result, and whose .derivation_chain(e, a, b) returns the corresponding
         ProofChain.
    """


class LeviCivitaConnection(ComponentConnection):
    """A ComponentConnection whose entries trace back to Koszul-formula chains."""
    def derivation_chain(self, e: int, a: int, b: int) -> ProofChain:
        """Return the step-by-step derivation of Γ^e_{ab}."""
```

**Derivation-chain steps (per `(e, a, b)`):**

| Step | Rule (display) | Before → After |
|---|---|---|
| 1 | "Koszul formula" | abstract `2g(∇_{e_a} e_b, e_c)` ↦ frame-component RHS |
| 2 | "Frame derivative `e_a(g_{bc})`" | unevaluated → SymPy ∂ result (or opaque) |
| 3 | "Frame derivative `e_b(g_{ac})`" | similar |
| 4 | "Frame derivative `−e_c(g_{ab})`" | similar |
| 5 | "γ^d_{bc} g_{da} term" | sum expansion |
| 6 | "γ^d_{ac} g_{db} term" | sum expansion |
| 7 | "γ^d_{ab} g_{dc} term" | sum expansion |
| 8 | "Multiply by ½ g^{ec}" | sum contraction |
| 9 | "Simplify" (concrete frames) | SymPy `simplify(result)` |

For coordinate frames, steps 5-7 vanish (γ ≡ 0); the chain
shortens to 5 steps.

**Tests:**
- Minkowski metric → all Christoffel symbols = 0.
- Schwarzschild → matches textbook table:
  - `Γ^t_{tr} = M / (r² (1 − 2M/r))`
  - `Γ^r_{tt} = M (1 − 2M/r) / r²`
  - `Γ^r_{rr} = −M / (r² (1 − 2M/r))`
  - `Γ^θ_{rθ} = 1/r`
  - `Γ^φ_{rφ} = 1/r`
  - `Γ^φ_{θφ} = cot θ`
  - `Γ^r_{θθ} = −(r − 2M)`
  - `Γ^r_{φφ} = −(r − 2M) sin² θ`
  - `Γ^θ_{φφ} = −sin θ cos θ`
- `derivation_chain(0, 0, 1).steps[-1].after` == `Γ^t_{tr}` value.
- AbstractFrame: `Γ^e_{ab}` returned as symbolic combinator over
  `e_a(g_{bc})` and `γ^d_{bc}` opaque atoms.

### Stage E, Torsion + curvature

**Files:**
- `jacopy/frame_calc/torsion.py`
- `jacopy/frame_calc/curvature.py`

**Deliverables:**

```python
def torsion(connection: ComponentConnection) -> ComponentTensor:
    """T^a_{bc} = Γ^a_{bc} − Γ^a_{cb} − γ^a_{bc}.

    Returns a (1, 2) ComponentTensor.  Antisymmetric in (b, c).
    """


def curvature(connection: ComponentConnection) -> ComponentTensor:
    """R^a_{bcd} = e_b(Γ^a_{cd}) − e_c(Γ^a_{bd})
                  + Γ^e_{cd} Γ^a_{be} − Γ^e_{bd} Γ^a_{ce}
                  − γ^e_{bc} Γ^a_{ed}.

    Returns a (1, 3) ComponentTensor.  Antisymmetric in (b, c).
    Each entry has a derivation_chain.
    """


class CurvatureTensor(ComponentTensor):
    def derivation_chain(self, a, b, c, d) -> ProofChain: ...
```

**Tests:**
- Levi-Civita on Minkowski → all torsion / curvature = 0.
- Levi-Civita on Schwarzschild → matches textbook curvature table
  (or at least the diagonal Ricci entries computed from it; full
  curvature has too many entries to assert exhaustively).
- Torsion-free check: `torsion(levi_civita(g)).is_zero()` is True.

### Stage F, Ricci / scalar / Einstein

**Files:**
- `jacopy/frame_calc/ricci.py`
- `jacopy/frame_calc/einstein.py`

**Deliverables:**

```python
def ricci(connection: ComponentConnection) -> ComponentTensor:
    """Ric_{ab} = R^c_{acb}, contracted from curvature.

    Returns a (0, 2) ComponentTensor.
    """


def ricci_scalar(connection: ComponentConnection, g: ComponentMetric) -> sp.Expr:
    """R = Ric_{ab} g^{ab}."""


def einstein_tensor(connection: ComponentConnection, g: ComponentMetric) -> ComponentTensor:
    """G_{ab} = Ric_{ab} − ½ R g_{ab}.

    Returns a (0, 2) ComponentTensor.  Vanishing G means vacuum solution.
    """
```

**Tests:**
- Schwarzschild → vacuum: every component of `einstein_tensor` is 0.
- FRW `ds² = −dt² + a(t)² (dx² + dy² + dz²)` with cosmological
  constant → `G_{tt}` matches Friedmann equation form.
- Minkowski → `Ric = 0`, `R = 0`, `G = 0`.

### Stage G, `ProofChain` bridge

**Files:**
- `jacopy/frame_calc/proof_bridge.py`

**Deliverables:**

```python
def derivation_to_proof_chain(steps: list[FrameCalcStep]) -> ProofChain:
    """Lift a list of frame-calc steps to a jacopy ProofChain.

    Each FrameCalcStep carries (rule, before_text, after_expr, justification).
    The lift wraps each into a ProofStep with provenance_tag='computation'
    so chain_to_latex_document renders it side-by-side with operator-level
    proofs from the rest of the package.
    """


# Add a fresh provenance tag for this layer:
#   provenance_tag='frame-calc' or 'computation'
```

**Tests:**
- `chain_to_latex_document` round-trips through a Christoffel chain.
- `display_chain(LC.derivation_chain(...))` works in Jupyter.

### Stage H, Concrete metric fixtures

**Files:**
- `jacopy/frame_calc/library/__init__.py`
- `jacopy/frame_calc/library/minkowski.py`
- `jacopy/frame_calc/library/schwarzschild.py`
- `jacopy/frame_calc/library/frw.py`

**Deliverables:**

```python
def minkowski(signature: str = "−+++") -> tuple[CoordinateFrame, ComponentMetric]:
    """Standard 4D Minkowski metric in (t, x, y, z) coordinates."""


def schwarzschild(M_sym: sp.Symbol | None = None) -> tuple[CoordinateFrame, ComponentMetric]:
    """Schwarzschild metric in (t, r, θ, φ) coordinates.

    Returns the frame and the metric, the user composes
    `levi_civita(g)`, `einstein_tensor(...)`, etc. on top.
    """


def frw(a_func: sp.Function, k: int = 0) -> tuple[CoordinateFrame, ComponentMetric]:
    """FRW metric `ds² = −dt² + a(t)² (dr² + r² dΩ²)` (k=0 flat case shown).

    `a_func` is a SymPy Function (e.g. `sp.Function('a')(t)`).
    """
```

**Tests:**
- Each fixture works end-to-end: `g → LC → curvature → ricci → einstein`.
- Vacuum / non-vacuum assertions match textbook expectations.

### Stage I, Tutorial 25

**Files:**
- `docs/tutorials/25_frame_calc.md`
- `docs/tutorials/_build_notebooks.py`, append `TUTORIAL_25`

**Tutorial outline:**

1. **Setting up a coordinate frame.** Symbols, `CoordinateFrame`.
2. **Defining a metric.** `ComponentMetric` from a SymPy matrix.
3. **Christoffel symbols via Koszul.** `levi_civita(g)`, table view, single-entry view.
4. **Step-by-step derivation.** `LC.derivation_chain(e, a, b)`,
   render via `chain_to_latex_document`.
5. **Curvature, Ricci, Einstein.** Pipeline `g → LC → R → Ric → R → G`.
6. **Vacuum check.** Schwarzschild `G == 0`.
7. **Going abstract.** Same pipeline on `AbstractFrame` with
   user-supplied `γ` table, outputs symbolic results.
8. **Going to a tetrad.** `Tetrad(coord_frame, vielbein)`,
   orthonormal Schwarzschild example.
9. **What's next.** Bridge to existing `LeviCivitaProblem` (Faz 19?
   future seeded theorem on operator-level Levi-Civita uniqueness).

### Stage J, Test suite

**Files:**
- `tests/test_frame_calc/test_coordinate_frame.py`
- `tests/test_frame_calc/test_abstract_frame.py`
- `tests/test_frame_calc/test_tetrad.py`
- `tests/test_frame_calc/test_component_tensor.py`
- `tests/test_frame_calc/test_levi_civita.py`
- `tests/test_frame_calc/test_torsion.py`
- `tests/test_frame_calc/test_curvature.py`
- `tests/test_frame_calc/test_ricci.py`
- `tests/test_frame_calc/test_einstein.py`
- `tests/test_frame_calc/test_proof_bridge.py`
- `tests/test_frame_calc/test_library/test_schwarzschild.py`
- `tests/test_frame_calc/test_library/test_frw.py`

**Coverage target:** every formula in the professor's document
verified against at least one concrete fixture (Minkowski for
zero cases, Schwarzschild for non-zero, FRW for time-dependent).

## Dependencies on the rest of jacopy

- `jacopy.core.expr.Expr`, for opaque atoms in abstract mode
  (`FrameDerivativeExpr`, `GammaExpr`, opaque `g_inv` symbols).
- `jacopy.proof.chain.ProofChain` and
  `jacopy.proof.step.ProofStep`, for the bridge layer.
- `jacopy.display.chain_to_latex_document`, for paper output.
- No dependency on the bracket / Cartan / Bianchi machinery.
  `frame_calc` is a sibling, not a dependent.

## Naming + import surface

Top-level:
```python
from jacopy.frame_calc import (
    CoordinateFrame, AbstractFrame, Tetrad,
    ComponentMetric, ComponentConnection,
    levi_civita, torsion, curvature,
    ricci, ricci_scalar, einstein_tensor,
)

from jacopy.frame_calc.library import (
    minkowski, schwarzschild, frw,
)
```

`jacopy/__init__.py` does **not** re-export these, keeps the
top-level namespace small. Users opt in by importing
`jacopy.frame_calc`.

## Order of work (suggested)

1. **Stage A**, `CoordinateFrame` + `Frame` protocol stubs.
2. **Stage A.2**, `AbstractFrame` + opaque atoms.
3. **Stage C**, `ComponentMetric` + inverse.
4. **Stage D**, `LeviCivita` (largest single deliverable).
5. **Stage G**, bridge to `ProofChain` (so we have rendering ready).
6. **Stage E**, torsion + curvature.
7. **Stage F**, Ricci / scalar / Einstein.
8. **Stage H**, concrete fixtures (uses everything; final integration test).
9. **Stage B**, `Tetrad` (after Stages D-F land, needs robust
   abstract-frame plumbing first).
10. **Stage I + J**, tutorial + final test pass.

Stage B (Tetrad) deliberately late, the math is tractable but
the design pressure on the `Frame` protocol is highest. Once
`AbstractFrame` and `CoordinateFrame` both work end-to-end through
Stages C-F, `Tetrad` lands as a third frame type that follows the
same protocol.

## Compatibility checklist for `CoordinateFrame`-first

When implementing Stage A with `CoordinateFrame` only, take care
that the design **doesn't accidentally depend on SymPy semantics**
in ways that break for `AbstractFrame`:

- ✅ `Frame.derivative` returns "something Expr-like", can be SymPy
  Expr or jacopy `Expr`. Use `as_jacopy(x)` / `as_sympy(x)` adapters
  at boundaries.
- ✅ `Frame.gamma` returns 0 for coordinate frames; opaque atom for
  abstract. Both are valid downstream, formulas just contract
  symbolically.
- ✅ `ComponentMetric.inverse()` for coordinate frame uses
  `sp.Matrix.inv()`. For abstract frame, returns wrapper with
  opaque `g_inv` symbols. Higher-level code only uses
  `metric.inverse()[a, b]`, never inspects the underlying type.
- ✅ Simplification: `LeviCivita` calls `sp.simplify(...)` only when
  every contributing term is a SymPy expression (concrete frame).
  For mixed / abstract, leaves the result un-simplified.
- ❌ Avoid: any `if isinstance(frame, CoordinateFrame): ...` branches
  in `LeviCivita` / `Curvature` / `Ricci`. All branching goes
  through the `Frame` protocol.

## Open design questions (to revisit per stage)

1. **Inverse-metric for abstract frame:** how opaque?
   - Option A: pure symbol `g_inv[a, b]` per entry.
   - Option B: a `ComponentMetricInverse` that *promises* `g · g_inv = I`
     but doesn't actually compute it (lazy).
   - Lean toward (A) for simplicity; reconsider if proof bridges need (B).

2. **Index gymnastics:** raise / lower indices via `g`?
   - Stage C deliberately skips this, the boxed formulas don't
     require general index gymnastics. Add at the end if a tutorial
     wants `g_{ab} V^b` operations.

3. **Display:** how to render a `ComponentTensor` in Jupyter?
   - Stage H punts: `repr` shows the SymPy array. Tutorial 25
     can introduce a richer renderer if needed.

4. **Tetrad → AbstractFrame conversion:** if user gives a Tetrad
   with abstract `e_a^μ`, should it auto-promote to AbstractFrame?
   - Stage B decides. Likely not, keep them separate types.

## Out of scope for Faz 18

- **Variational principles / Lagrangian.** Different style, not
  formulaic.
- **Numerical evaluation of `G = T`.** That's a physics-solver task,
  not a formula evaluator.
- **Coordinate transformations.** User can compute a metric in two
  charts manually if needed.
- **General relativity examples requiring Killing vectors,
  causal structure analysis, etc.** All separate phases if ever.

## Estimated effort

| Stage | Description | Effort |
|---|---|---|
| A | CoordinateFrame + Frame protocol | 1-2 days |
| A.2 | AbstractFrame + opaque atoms | 2 days |
| C | ComponentMetric + inverse | 1 day |
| D | LeviCivita / Koszul | 3 days |
| G | ProofChain bridge | 2 days |
| E | Torsion + curvature | 2 days |
| F | Ricci + scalar + Einstein | 1 day |
| H | Library fixtures (Minkowski, Schwarzschild, FRW) | 1 day |
| B | Tetrad | 2 days |
| I | Tutorial 25 | 2 days |
| J | Test suite | 2 days |
| **Total** | | **~19 days** |

Practical schedule: 4-5 weeks of focused work.

## Success criteria

- `from jacopy.frame_calc import schwarzschild, levi_civita,
  einstein_tensor` runs.
- `g, F = schwarzschild(); LC = levi_civita(g); G =
  einstein_tensor(LC, g)` produces an exactly-zero
  `ComponentTensor` (vacuum solution verified).
- `chain_to_latex_document(LC.derivation_chain(0, 0, 1))` produces
  a publication-ready LaTeX block showing each step of the Koszul
  formula as it specialises to `Γ^t_{tr}`.
- The same pipeline works on `AbstractFrame` with symbolic `g_{ab}`
  and `γ^a_{bc}`, every formula closes symbolically with opaque
  atoms.
- 25th tutorial in the test suite (`tests/test_docs/test_notebooks.py`)
  passes.
- 2792 + (~200 frame_calc) unit tests, all green.
