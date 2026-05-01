# 22, Local frames and frame decomposition

A **local frame** is a basis ``(X_1, …, X_n)`` of vector fields on
an open set, together with its dual coframe ``(e^1, …, e^n)`` of
1-forms. Frames are the bridge between coordinate-free expressions
``∇_X Y`` and the index-laden objects ``ω^a_b``, ``Γ^c_{ab}``,
``T^a{}_{bc}`` that classical differential geometry uses.

In jacopy a frame is a **library wrapper** rather than an `Expr`,
it carries name, dimension, and display symbols, and it produces the
`Expr` instances `X_a`, `e^a` on demand. The companion engine rules
realise frame duality and frame decomposition.

This tutorial covers:

1. `LocalFrame(name, dim=…)`, the wrapper, identity, and display
   conventions.
2. `FrameIndex`, `FrameVectorField`, `FrameCovector`,
   `KroneckerDelta`, the four `Expr` shapes a frame produces.
3. `FramePairingDualityDefinition`, ``⟨e^a, X_b⟩ → δ^a_b``.
4. `FrameDecompositionDefinition` and friends, opt-in rules that
   re-expand a vector field over a frame's basis.

## `LocalFrame`, the wrapper

```python
from jacopy.calculus.local_frame import LocalFrame, FrameIndex

F = LocalFrame("F", dim=3)
print(f"frame    : {F}")
print(f"name     : {F.name}")
print(f"dim      : {F.dim}")
print(f"vf sym   : {F.vf_symbol}")
print(f"coframe  : {F.coframe_symbol}")
```

`dim=None` is allowed, that's the **symbolic-dimension** mode Faz
17 proofs use, where the frame stands for any ``n``-dimensional
basis. Frames with the same `(name, dim, vf_symbol, coframe_symbol)`
compare equal, so two notebook cells can construct "the same"
frame without ending up with two distinct objects.

## Indices and basis elements

A `FrameIndex` is the dummy / free index (a plain `Atom`); the
frame produces frame VFs `X_a` and dual covectors `e^a`:

```python
a, b = F.index("a"), F.index("b")
X_a, X_b = F.X(a),       F.X(b)
e_a, e_b = F.coframe(a), F.coframe(b)

print(f"X_a : {X_a}")
print(f"e_a : {e_a}")
print(f"X_a class : {X_a.__class__.__name__}")
print(f"e_a class : {e_a.__class__.__name__}")
```

`FrameVectorField` subclasses `Derivation` (degree 0), every
existing pass that walks Derivation shapes (`∇`, `ι`, `L`, `Pairing`)
picks frame VFs up automatically. `FrameCovector` is an `Atom`
that interacts with vectors only through `Pairing`, its
algebraic content is exactly the duality rule below.

Equality on both classes includes the **frame name**, so two
frames sharing only their VF symbol stay distinguishable. That's
what lets two coexisting frames in a single proof not cross-fire.

## `KroneckerDelta`, the contraction unit

`KroneckerDelta(i, j)` is the ``δ^i_j`` symbol with built-in
collapse: when the two indices are *structurally equal*, the
constructor returns `One` directly.

```python
from jacopy.calculus.local_frame import KroneckerDelta

print(f"δ^a_a = {KroneckerDelta(a, a)}")  # → 1
print(f"δ^a_b = {KroneckerDelta(a, b)}")  # → δ^a_b (opaque)
```

That's enough to make duality cancel inside a sum: once
`IndexedSumKroneckerContractDefinition` (tutorial 21) recognises
``Σ_i δ_i^j A_i``, it collapses to ``A_j``.

## `FramePairingDualityDefinition`, `⟨e^a, X_b⟩ → δ^a_b`

The duality rule is *frame-scoped*: it fires only when both halves
of the pairing belong to the same `LocalFrame`. Two frames in one
proof never cross-fire.

```python
from jacopy.calculus.pairing import Pairing
from jacopy.proof.expansion import ExpansionEngine

p = Pairing(e_b, X_a)
print(f"raw pairing : {p}")

engine = ExpansionEngine([F.duality_definition()])
out, steps = engine.expand(p)
print(f"after rule  : {out}")
print(f"rule        : {steps[0].rule}")
```

Building the rule via `F.duality_definition()` ensures the
frame-scoping is wired correctly. The right-hand side is a
`KroneckerDelta`, collapses to `1` when indices match (see
above), stays opaque otherwise.

## Frame decomposition, opt-in rules

`FrameDecompositionDefinition(F)` rewrites a vector field as
``W → Σ_a e^a(W) · X_a``. **It's opt-in for a reason**: pairing
it with the duality rule creates a loop (the rule re-fires on the
``X_a`` it just produced). The Cartan-structure proof
(`CartanStructureProblem`, tutorial 23) turns it on for a specific
sub-pass and turns it off again.

```python
from jacopy.algebra.derivation import Derivation
from jacopy.calculus.frame_decomposition import FrameDecompositionDefinition

W = Derivation("W", 0)
rule = FrameDecompositionDefinition(F)

# Apply the rule once directly (avoid running the engine to fix-point, loops).
print(f"W → {rule.rewrite(W)}")
```

The output is an `IndexedSum` over the frame: a fresh bound
`FrameIndex` (the hat-decorated dummy in the display) avoids
shadowing existing free indices.

The two companion rules in `frame_decomposition`:

| Rule | Folds |
|---|---|
| `FrameDecompositionDefinition(F)` | ``W → Σ_a e^a(W) · X_a`` for any non-frame VF ``W`` |
| `ConnectionEvalYFrameDecompositionDefinition(F, ∇)` | ``∇_X Y → ∇_X (Σ_a e^a(Y) · X_a)`` (positional) |
| `ConnectionFormDecompositionDefinition(F, ∇, ω)` | ``∇_V X_b → Σ_c ω^c{}_b(V) · X_c`` |

The third rule is the one that introduces the **connection form**
``ω^c{}_b(∇)``. It's the keystone of the Cartan structure equation
proofs, every Christoffel-symbol-flavoured calculation funnels
through it.

## Why three frame-decomposition rules

Each handles a distinct shape that Cartan I/II reductions produce:

1. ``e^a(∇_U V)``, frame-expand ``V`` first, then pull
   ``e^a`` past the connection.
2. ``∇_U(e^c(V) · X_c)``, Y-Leibniz produces ``∇_U X_c``,
   which collapses through `ConnectionFormDecompositionDefinition`.
3. ``∇_U V`` directly, the positional rule
   `ConnectionEvalYFrameDecompositionDefinition` decomposes the
   ``Y`` slot only, avoiding the cross-talk that the global
   `FrameDecompositionDefinition` would cause when paired with
   `FramePairingDualityDefinition`.

The split exists because none of these three is the right tool for
*every* residue shape, picking the right one is what
`CartanStructureProblem` does for you (tutorial 23).

## When you'd touch frames directly

Most calls to a frame happen inside higher-level wrappers
(`CartanStructureProblem`, Q7 / Q9 capstones). You'd reach for
`LocalFrame` directly when:

* writing a custom proof that needs **both** an index-free and
  index-laden representation in the same chain;
* defining a problem-specific rewrite rule that consumes
  `FrameVectorField` / `FrameCovector` shapes (e.g. a structure
  equation on an exotic algebroid);
* debugging a residue containing `FrameIndex` / `KroneckerDelta`
  / `IndexedSum` and needing to know which frame each shape
  belongs to.

## Summary

* `LocalFrame(name, *, dim=None, vf_symbol="X", coframe_symbol="e")`
 , library wrapper; `dim=None` is the symbolic-dimension mode.
  Frames with matching tuples compare equal.
* Four `Expr` shapes from a frame: `FrameIndex`, `FrameVectorField`
  (`Derivation` subclass), `FrameCovector` (`Atom`), `KroneckerDelta`
  (collapses to `One` on matching indices).
* `FramePairingDualityDefinition`, `⟨e^a, X_b⟩ → δ^a_b`,
  frame-scoped (no cross-firing between frames).
* Three opt-in frame-decomposition rules in `frame_decomposition`:
  `FrameDecompositionDefinition` (global), `ConnectionEvalYFrameDecompositionDefinition`
  (positional), `ConnectionFormDecompositionDefinition` (introduces
  `ω^c{}_b`). They form loops with duality unless turned on
  selectively, `CartanStructureProblem` (tutorial 23) handles the
  bookkeeping.
