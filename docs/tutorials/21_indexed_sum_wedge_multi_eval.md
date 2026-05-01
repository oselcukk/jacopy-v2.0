# 21, IndexedSum, Wedge, MultiEval

Three structural `Expr` nodes underpin everything that came after Faz
17: **`MultiEval`** for textbook ``ω(Y_1, …, Y_p)`` evaluation,
**`Wedge`** for the graded-antisymmetric ``α ∧ β`` distinct from
non-commutative `Product`, and **`IndexedSum`** for bound-index
summations ``Σ_i …`` over a frame. None of them carry algebraic
content on their own, antisymmetry, distribution, and Kronecker
contraction live as engine `Definition`s in the companion
`*_axioms` modules.

This tutorial covers:

1. `MultiEval(head, *args)`, the multilinear evaluation atom +
   `swapped`, `with_args`, `has_repeated_arg`, `validate_arity`
   helpers.
2. `Wedge.make(α, β, …)`, the graded-antisymmetric product node
   with smart-constructor identities.
3. `IndexedSum(dummy, range_, body)`, α-equivalence as `==`,
   shadowing semantics, `with_dummy`.
4. The companion axiom rules, `WedgeMultiEvalAlternating`,
   `IndexedSumSumDistribute`, etc.

## `MultiEval`, multilinear evaluation

`multi_eval(head, *args, alternating=True, slot_kind="vector")`
builds the `MultiEval` node. The head is a form (or multivector),
the args are vector fields (or 1-forms when `slot_kind="covector"`).
The slot kind is *declarative*, it documents intent but doesn't
change structural algebra; it's the routing discipline that keeps
form-side and tilde-side intrinsic engines from aliasing (tutorials
15 and 17).

```python
from jacopy.core.expr import Symbol
from jacopy.core.multi_eval import multi_eval, has_repeated_arg

omega = Symbol("ω")
X, Y  = Symbol("X"), Symbol("Y")

me = multi_eval(omega, X, Y)
print(f"ω(X, Y)        : {me}")
print(f"arity          : {me.arity}")
print(f"alternating    : {me.alternating}")
print(f"slot_kind      : {me.slot_kind}")
```

`swapped(i, j)` returns `(new_node, sign)`, the sign is `-1` when
`alternating=True` and `i != j`, `+1` otherwise. `has_repeated_arg`
checks whether any arg appears more than once (the input shape that
forces the whole node to `0` under alternating semantics).

```python
swapped, sign = me.swapped(0, 1)
print(f"swap → {swapped}, sign={sign}")

me_repeat = multi_eval(omega, X, X)
print(f"has_repeated_arg : {has_repeated_arg(me_repeat)}")
```

**Arity is not validated at construction.** A user assembling
`MultiEval(omega, X, Y, Z)` must know that `omega` is a 3-form. The
reason: `head` may be a compound expression like `Act(d, omega)`
whose degree depends on a registry lookup, and the structural Expr
layer deliberately avoids registry dependencies. Use
`validate_arity(expr, registry)` if you want the explicit check.

## `Wedge`, graded-antisymmetric product

In differential geometry ``α ∧ β`` is **not** the same as the plain
`Product(α, β)`: it carries graded antisymmetry
``α ∧ β = (−1)^{|α||β|} β ∧ α`` and the degree law
``|α ∧ β| = |α| + |β|``. `Product` is reserved for the
non-commutative scalar / operator product (used for both ``f · g``
and ``D_1 ∘ D_2``), so wedges live in their own node.

```python
from jacopy.core.expr import Integer
from jacopy.core.wedge import Wedge

alpha, beta, gamma = Symbol("α"), Symbol("β"), Symbol("γ")

w = Wedge.make(alpha, beta, gamma)
print(f"α ∧ β ∧ γ : {w}")

# Smart-constructor identities
print(f"absorb 0  : {Wedge.make(alpha, Integer(0), beta)}")
print(f"drop 1    : {Wedge.make(alpha, Integer(1), beta)}")
```

`Wedge.make` flattens nested wedges (associativity), absorbs any
`0` factor, drops integer `1` factors (the wedge unit), and
collapses one-factor wedges to that factor. **Degree-aware**
identities like ``α ∧ α = 0`` for degree-1 ``α`` are *not* applied
here, they need degree information from a registry, and live in
the algorithms layer.

## `WedgeMultiEvalAlternatingDefinition`, the wedge action

The textbook expansion
``(α_1 ∧ … ∧ α_p)(X_1, …, X_p) = Σ_σ sign(σ) ∏_i α_i(X_{σ(i)})``
fires through this engine rule:

```python
from jacopy.calculus.wedge_axioms import WedgeMultiEvalAlternatingDefinition
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine

reg = PropertyRegistry()
for f in (alpha, beta):
    reg.declare(f, Graded(degree=1))
for f in (X, Y):
    reg.declare(f, Graded(degree=0))

w_eval = multi_eval(Wedge.make(alpha, beta), X, Y)
engine = ExpansionEngine([WedgeMultiEvalAlternatingDefinition(registry=reg)])
out, steps = engine.expand(w_eval)

print(f"(α ∧ β)(X, Y) → {out}")
print(f"rule          : {steps[0].rule}")
```

The rule needs degree information (it only fires on degree-1
factors against the right number of args), which is why it takes a
`registry=` kwarg.

## `IndexedSum`, bound-index summation

`indexed_sum(dummy, range_, body)` builds the bound-index
summation. The dummy is an `Atom` (typically a `FrameIndex`); the
range is opaque metadata (typically a `LocalFrame`); the body is
any `Expr` referring to the dummy.

```python
from jacopy.calculus.local_frame import FrameIndex, LocalFrame
from jacopy.core.indexed_sum import indexed_sum

i = FrameIndex("i")
frame = LocalFrame("e", dim=3)
T_i = Symbol("T_i")

S = indexed_sum(i, frame, T_i)
print(f"Σ_i T_i        : {S}")
print(f"dummy          : {S.dummy}")
print(f"range          : {S.range_}")
print(f"body           : {S.body}")
```

### α-equivalence as `==`

Two `IndexedSum`s are equal when their ranges match and their
bodies become structurally identical after a dummy renaming.
`with_dummy(new_dummy)` produces an α-equivalent copy:

```python
j = FrameIndex("j")
S_j = S.with_dummy(j)
print(f"S (i)          : {S}")
print(f"S (j)          : {S_j}")
print(f"S == S_j       : {S == S_j}")
```

α-equivalence is implemented through a depth-aware sentinel
(``$bound_<depth>``), so nested binders whose original dummies happen
to share a name still distinguish correctly.

### Shadowing

`substitute_atom` honours shadowing: if the inner binder uses the
same dummy as the outer, the outer name is hidden inside.

## `IndexedSumSumDistributeDefinition` and friends

Engine rules that operate on `IndexedSum` nodes:

| Rule | Folds |
|---|---|
| `IndexedSumSumDistributeDefinition` | ``Σ_i (A + B) → Σ_i A + Σ_i B`` |
| `IndexedSumNegPullDefinition` | ``Σ_i Neg(X) → Neg(Σ_i X)`` |
| `IndexedSumScalarPullDefinition` | ``Σ_i (c · X) → c · Σ_i X`` (when ``c`` is dummy-free) |
| `IndexedSumKroneckerContractDefinition` | ``Σ_i δ_i^j A_i → A_j`` |
| `IndexedSumPairingPushIn{Right,Left}Definition` | pull `Pairing` past a sum on either side |
| `MultiEvalIndexedSumPushInDefinition` | pull `MultiEval` past a sum |
| `ConnectionEvalIndexedSumPushInDefinition` | pull `∇_X` past a sum |

Quick look at the distributor:

```python
from jacopy.calculus.indexed_sum_axioms import (
    IndexedSumSumDistributeDefinition,
)
from jacopy.core.expr import Sum

A, B = Symbol("A_i"), Symbol("B_i")
S = indexed_sum(i, frame, Sum(A, B))
engine = ExpansionEngine([IndexedSumSumDistributeDefinition()])
out, steps = engine.expand(S)

print(f"Σ_i (A + B) → {out}")
print(f"rule        : {steps[0].rule}")
```

The push-in family (`Pairing`, `MultiEval`, `ConnectionEval`)
exists because all three of those nodes contract over an external
slot, and the engine needs explicit rules to shuffle them past a
binder. Without them, ``∇_X (Σ_i A_i)`` would stay opaque to the
connection axioms, they don't reach into the body of an
`IndexedSum`.

## Where these nodes get used

| Node | Used in |
|---|---|
| `MultiEval` | `prove_intrinsic_equivalence` (tutorial 15), `prove_tilde_cartan_relation` (tutorial 17), §3.1.5 derivator drivers (tutorial 18) |
| `Wedge` | Cartan structure equations (tutorial 23), `WedgeMultiEvalAlternatingDefinition` |
| `IndexedSum` | Frame decomposition (tutorial 22), Cartan structure equations (tutorial 23), Q7 / Q9 capstones |

Most users never touch these nodes directly, they get built and
consumed inside higher-level wrappers. The reason this tutorial
exists at all is that **debugging** an unclosed proof on the Cartan
or frame side eventually drops you into a residue containing one of
these three, and you want to know what the rules and conventions
are.

## Summary

* `MultiEval(head, *args)`, multilinear evaluation. Slot kind is
  declarative; arity not validated at construction. `swapped(i, j)`
  carries the alternating sign.
* `Wedge.make(α, β, …)`, graded-antisymmetric product, distinct
  from `Product`. Smart constructor handles 0/1/associativity;
  degree-aware identities (`α ∧ α = 0`) live in algorithms layer.
* `IndexedSum(dummy, range_, body)`, bound-index sum with
  α-equivalence as `==`, depth-aware sentinel for nested binders,
  `with_dummy(new)` for explicit renaming.
* Engine rules in `wedge_axioms` / `indexed_sum_axioms` realise the
  algebraic content: alternating wedge expansion, sum-distribute,
  scalar-pull, Kronecker contract, push-in past Pairing /
  MultiEval / ConnectionEval.
* Direct use is rare; debugging Cartan / frame proofs is the usual
  reason to know these nodes' shapes and conventions.
