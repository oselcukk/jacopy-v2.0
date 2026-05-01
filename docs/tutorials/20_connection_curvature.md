# 20, Connection, curvature, and Bianchi identities

An **affine connection** ``∇`` on ``TM`` is the linear-algebraic
backbone of differential geometry: it's what lets you take directional
derivatives of vector fields. Once you have ``∇`` you get **torsion**
``T(X, Y)`` measuring its asymmetry, **curvature** ``R(X, Y) Z``
measuring its non-flatness, and the **two Bianchi identities**,
algebraic identities relating ``T``, ``R``, and their covariant
derivatives.

This tutorial covers:

1. `AffineConnection`, the atom, ``∇_X Y`` evaluation, the four
   connection axioms.
2. `Torsion` and `Curvature`, the symbolic builders.
3. `cov_deriv_torsion` / `cov_deriv_curvature`, covariant derivatives.
4. `BianchiProblem`, the wrapper that bundles every rule needed.
5. `prove_first_bianchi` / `prove_second_bianchi`, the two
   identities, mechanically.
6. `koszul_connection`, the cotangent-bundle Koszul variant for
   problems on ``T*M``.

## `AffineConnection` and `∇_X Y`

`AffineConnection(name)` is an atom, opaque to the engine until a
rule fires. Its core operation is `∇.eval(X, Y)` building the
`ConnectionEvalExpr` node ``∇_X Y``.

```python
from jacopy.algebra.derivation import Derivation
from jacopy.calculus.connection import AffineConnection

nabla = AffineConnection("∇")
X, Y = Derivation("X", 0), Derivation("Y", 0)

print(f"connection : {nabla}")
print(f"∇_X Y      : {nabla.eval(X, Y)}")
```

The four core engine rules realise the **defining axioms** of an
affine connection:

| Rule | Identity |
|---|---|
| `ConnectionXLinearityDefinition` | ``∇_{X+Y} Z = ∇_X Z + ∇_Y Z`` |
| `ConnectionXScalarPullDefinition` | ``∇_{f X} Y = f ∇_X Y`` |
| `ConnectionYAdditivityDefinition` | ``∇_X (Y + Z) = ∇_X Y + ∇_X Z`` |
| `ConnectionYLeibnizDefinition` | ``∇_X (f Y) = X(f) Y + f ∇_X Y`` |

The Leibniz rule's ``X(f)`` term routes through
`AffineConnection.function_action(X, f)`, that's the hook for
algebroid connections to swap ``X(f)`` for ``ρ(X)(f)``.

## `Torsion` and `Curvature`

`Torsion(∇)(X, Y)` and `Curvature(∇)(X, Y) Z` are the two structural
obstructions of a connection:

```
T(X, Y) := ∇_X Y − ∇_Y X − [X, Y]
R(X, Y) Z := ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X,Y]} Z
```

Both are inert until the engine's `TorsionDefinitionDefinition` /
`CurvatureDefinitionDefinition` fire.

```python
from jacopy.calculus.torsion_curvature import Torsion, Curvature

W, Z = Derivation("W", 0), Derivation("Z", 0)
T = Torsion(nabla, X, Y)
R = Curvature(nabla, X, Y, W)

print(f"T(X, Y)        : {T}")
print(f"R(X, Y) W      : {R}")
```

The bracket inside ``T`` and ``R`` is the
**vector-field bracket** of the connection, `LieBracketVF` by
default, or a custom bracket if the connection was built with one.
That's what makes the same `BianchiProblem` machinery work for both
plain affine connections and the algebroid Koszul connection (see
the bottom section).

## `BianchiProblem`, the wrapper

`BianchiProblem(connection, *, registry)` bundles every rule needed
to close the two Bianchi identities: the four connection axioms, the
torsion / curvature definitions, the covariant-derivative
definitions, and the closure family for the bracket flavour the
connection carries (LBVF or BracketApply).

```python
from jacopy.core.registry import PropertyRegistry
from jacopy.library.bianchi_problem import BianchiProblem

reg  = PropertyRegistry()
prob = BianchiProblem(nabla, registry=reg)
print(f"engine rules : {len(prob.engine.definitions)}")
print(f"connection   : {prob.connection}")
```

The `registry` is consulted by the Y-Leibniz rule for the
`Graded(degree=0)` test on a function factor, pass `None` if you
only have pure vector-field arguments.

## First Bianchi identity

```
cycl_{X,Y,Z} R(X, Y) Z = cycl_{X,Y,Z} [(∇_X T)(Y, Z) + T(T(X, Y), Z)]
```

`prove_first_bianchi(X, Y, Z)` builds both sides, takes the
difference, and runs the engine + canonicalize + collect_terms loop
until the residue lands on `0`. The result carries `ok=True` on
success, the per-side initial / final expressions, and the full
proof transcript.

```python
res1 = prob.prove_first_bianchi(X, Y, W)
print(f"Bianchi I  : ok={res1.ok}, steps={len(res1.lhs_steps)}")
```

Roughly 60 steps cover: torsion / curvature unfolding (3 cyclic
copies × 2 sides), connection-axiom Leibniz / linearity rewrites,
and LBVF closure (Sum / Neg linearity, antisymmetry, Jacobi).

## Second Bianchi identity

```
cycl_{X,Y,Z} (∇_X R)(Y, Z) W = cycl_{X,Y,Z} R(X, T(Y, Z)) W
```

Same protocol, one extra fixed argument:

```python
res2 = prob.prove_second_bianchi(X, Y, W, Z)
print(f"Bianchi II : ok={res2.ok}, steps={len(res2.lhs_steps)}")
```

The second identity is structurally similar, three cyclic terms on
each side, comparable engine workload, ~63 steps in total.

## The Koszul connection, same identities on `T*M`

`koszul_connection(name, *, anchor, bracket)` produces an algebroid
connection ``∇̃`` on ``T*M`` for Poisson problems. The same
`BianchiProblem` wrapper works on it, the engine swaps in the
`BracketApply` closure family (instead of LBVF) but otherwise the
proof is identical.

```python
from jacopy.calculus.connection import koszul_connection
from jacopy.calculus.anchor import Anchor

pi_sharp = Anchor("π^♯")
nabla_tilde = koszul_connection("∇̃", anchor=pi_sharp)
print(f"connection : {nabla_tilde}")
print(f"anchor     : {nabla_tilde.anchor}")
print(f"bracket    : {nabla_tilde.bracket}")
```

The torsion and curvature definitions on `nabla_tilde` emit
`BracketApply([·,·]_K, X, Y)` instead of `LieBracketVF(X, Y)`, and
the function-action ``X(f)`` becomes ``π^♯(X)(f)`` through the
anchor. None of that is your concern at the call site:
`BianchiProblem(nabla_tilde)` does the right thing.

## Where this sits in the bigger picture

`BianchiProblem` is the **structural backbone** of the connection
machinery, every other higher-level wrapper (`KoszulConnectionProblem`,
`CartanStructureProblem`) layers on top of it. The Koszul facet of
`KoszulConnectionProblem` is just a `BianchiProblem` with the right
connection plugged in; the Cartan structure equations of tutorial 23
use the same connection axioms one level up.

If you only need ``∇_X Y`` evaluation without going through Bianchi
, e.g. you're proving an identity that involves a ``∇_X Y`` term but
no torsion / curvature, reach for the **four connection axioms
directly** rather than the full `BianchiProblem` engine. Their
constructors take `(connection, *, registry)` and slot into any
custom engine.

## Summary

* `AffineConnection(name)` is the atom; `∇.eval(X, Y)` builds the
  `ConnectionEvalExpr` node.
* Four defining-axiom engine rules:
  `ConnectionXLinearity`, `ConnectionXScalarPull`,
  `ConnectionYAdditivity`, `ConnectionYLeibniz`.
* `Torsion(∇, X, Y)` and `Curvature(∇, X, Y, Z)`, inert until the
  engine's `TorsionDefinition` / `CurvatureDefinition` fire,
  unfolding to ``∇``-commutators + bracket terms.
* `BianchiProblem(connection, *, registry)` bundles every rule into
  one engine; `prove_first_bianchi` / `prove_second_bianchi` close
  in ~60-63 steps each.
* `koszul_connection(name, *, anchor, bracket)` builds the algebroid
  variant on ``T*M``; same `BianchiProblem` wrapper handles it,
  swapping in `BracketApply` closure rules.
