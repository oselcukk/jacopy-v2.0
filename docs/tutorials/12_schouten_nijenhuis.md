# 12 — The Schouten–Nijenhuis bracket

The Lie bracket of vector fields tells you how two flows fail to
commute. `[·,·]_SN` — the **Schouten–Nijenhuis** bracket — is the
unique extension of that idea to the full algebra of multivector
fields ``⊕_k Γ(Λ^k TM)``: a graded Lie bracket of degree 0 (in the
shifted grading ``|X| = k − 1``) that acts as a graded derivation in
each slot with respect to the wedge product. It is also the algebraic
backbone of Poisson geometry — a bivector ``π`` is Poisson exactly
when ``[π, π]_SN = 0``.

This tutorial walks the API:

1. the four atomic base cases on 1-vectors and functions,
2. the wedge-Leibniz recursion that climbs into higher multivectors,
3. the opaque return for atomic higher-order multivectors (e.g.
   a bare bivector symbol ``π``), which is what makes ``[π, π]_SN``
   a usable handle in proofs,
4. the bridge to `PoissonBracket` — `jacobi_obstruction`,
   `prove_jacobi_reduction` — that consumes that handle.

## Shifted grading

In `jacopy` a multivector ``X ∈ Λ^k TM`` carries SN-degree ``|X| = k − 1``:

| Object | Tensor degree | SN degree |
|---|---|---|
| function ``f`` (0-vector) | 0 | −1 |
| vector field ``X`` (1-vector) | 1 | 0 |
| bivector ``π`` (2-vector) | 2 | 1 |
| trivector | 3 | 2 |

Declare this with `Graded(degree=...)`. **Do not use `Scalar()` for
functions when working with SN** — `Scalar()` declares "tensor degree
0", which the SN engine reads as 1-vector. Use `Graded(degree=-1)`
for a function in SN contexts.

```python
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
X = Symbol("X"); reg.declare(X, Graded(degree=0))   # 1-vector
Y = Symbol("Y"); reg.declare(Y, Graded(degree=0))
f = Symbol("f"); reg.declare(f, Graded(degree=-1))  # function
g = Symbol("g"); reg.declare(g, Graded(degree=-1))
```

## The four base cases

`sn.expand(a, b, registry)` returns the closed form when both operands
are atomic and one of the four base shapes applies. These are the
characterising rules of the bracket:

```python
from jacopy.brackets.schouten import sn

print("[X, Y]_SN =", sn.expand(X, Y, reg))   # X*Y − Y*X  (Lie bracket)
print("[f, g]_SN =", sn.expand(f, g, reg))   # 0
print("[f, X]_SN =", sn.expand(f, X, reg))   # −X(f)
print("[X, f]_SN =", sn.expand(X, f, reg))   # X(f)
```

The 1-vector / 1-vector case reproduces the Lie bracket as
``X * Y − Y * X`` (the engine treats vector fields as derivations on
the function algebra). The two function-vector cases give
``±X(f)`` — the action of ``X`` on ``f`` — and two functions bracket
to zero, since multiplication on ``C^∞`` is commutative.

## Wedge Leibniz

For wedge products, SN climbs the recursion

```
[X ∧ Y, Z]_SN = X ∧ [Y, Z]_SN + (−1)^{|Y||Z|} [X, Z]_SN ∧ Y
[Z, X ∧ Y]_SN = [Z, X]_SN ∧ Y + (−1)^{|X||Z|} X ∧ [Z, Y]_SN
```

Wedges reuse `Product` (a dedicated `Wedge` Expr is deferred polish
that isn't needed at this layer). Feeding a `Product` of two
1-vectors makes the bracket descend through both base cases:

```python
from jacopy.core.expr import Product

bivec = Product(X, Y)              # X ∧ Y, SN-degree 1
print("[X ∧ Y, f]_SN =", sn.expand(bivec, f, reg))
# (X * Y(f)) + (X(f) * Y)
```

The result reads: ``X ∧ [Y, f] + (−1)^{|Y||f|}[X, f] ∧ Y``
``= X ∧ Y(f) + [X, f] ∧ Y`` (since ``|Y| · |f| = 0 · −1 = 0``,
the sign is ``+1``).

## Atomic higher-order multivectors stay opaque

A bare `Symbol` declared `Graded(degree=1)` stands in for an atomic
bivector — there's no wedge to peel, so SN can't descend into it.
Rather than raising, `expand` returns the inert `BracketApply` node:

```python
pi = Symbol("π"); reg.declare(pi, Graded(degree=1))
print("[π, π]_SN =", sn.self_bracket(pi, reg))
# [·,·]_SN(π, π)
```

That opaque return *is the point*. It gives you a typed handle to
the obstruction that you can compare against `0`, render to LaTeX,
or feed into a proof closure as a hypothesis. The shape
``[·,·]_SN(π, π) = 0`` is exactly the Poisson condition.

`self_bracket(Q)` is a thin wrapper around `expand(Q, Q, ...)` —
useful for the universal obstruction pattern:

| Object ``Q`` | ``[Q, Q]_SN = 0`` is | Phase |
|---|---|---|
| bivector ``π`` | Poisson condition | Faz 9 Stage B |
| Courant generator ``Θ`` | Courant compatibility | (higher algebras) |

## Bridge to `PoissonBracket`

`PoissonBracket(π)` is the user-facing wrapper; behind the scenes it
delegates the Jacobi question to SN. Two methods make the bridge
visible:

```python
from jacopy.library.poisson import PoissonBracket

P = PoissonBracket(pi)
print("obstruction:", P.jacobi_obstruction())
print("condition  :", P.jacobi_condition())
```

`jacobi_obstruction()` returns ``[·,·]_SN(π, π)`` — the same opaque
handle SN gives you directly. `jacobi_condition()` wraps that into
the textbook statement ``[·,·]_SN(π, π) = 0``.

The killer move is `prove_jacobi_reduction(f, g, h)` — the
**Derived Bracket Theorem** mechanised as a single proof step:

```python
h = Symbol("h"); reg.declare(h, Graded(degree=-1))

chain = P.prove_jacobi_reduction(f, g, h, registry=reg)
print(f"initial: {chain.initial}")
print(f"final  : {chain.final}")
print(f"steps  : {len(chain)}  rule: {chain.steps[0].rule}")
```

The cyclic Jacobi sum
``{f,{g,h}} + {g,{h,f}} + {h,{f,g}}`` collapses in one step (rule
`DerivedBracketTheorem`) to ``[·,·]_SN(π, π)``. Once you cite the
seeded `poisson_jacobi` axiom (or assume it) the obstruction
vanishes and the Jacobi identity is proved.

## When SN stays inert

Three situations keep the bracket opaque rather than producing a
closed form:

1. **Atomic higher-order multivector** (``[π, π]_SN`` for atomic
   ``π``) — already covered above. This is *useful* opacity: the
   handle drives the proof.
2. **Symbolic SN-degree** — if any operand's `Graded(degree=...)`
   is symbolic (e.g. ``Degree.var("k")``), the wedge-Leibniz parity
   ``(−1)^{|Y||Z|}`` can't be decided and `expand` falls back to the
   `BracketApply`. Declare concrete integer degrees to push past this.
3. **Forms** — `sn.expand(α, π)` for a 1-form ``α`` and a multivector
   ``π`` is **not defined**. SN is the multivector-only bracket; the
   form-level operation is the Koszul bracket, which lives behind
   `DerivedBracket(sn, π, acting_on=Sharp(π))` (see tutorial 7 on
   derived brackets).

## Summary

* `sn = SchoutenBracket()` is a graded Lie bracket of degree 0 in the
  shifted grading ``|X| = k − 1``.
* Four base cases on 1-vectors / functions: ``[X,Y] = XY − YX``,
  ``[f,g] = 0``, ``[f,X] = −X(f)``, ``[X,f] = X(f)``.
* Wedge Leibniz pushes the bracket through `Product(X, Y)` factors
  with a graded sign.
* Atomic higher multivectors return an opaque `BracketApply` —
  ``sn.self_bracket(π) = [·,·]_SN(π, π)`` is the universal Poisson
  obstruction, and `PoissonBracket.prove_jacobi_reduction` collapses
  the cyclic Jacobi sum to it in one step via the Derived Bracket
  Theorem.
* For form-level operations use the Koszul bracket (tutorial 7) —
  SN deliberately doesn't lift onto forms.
