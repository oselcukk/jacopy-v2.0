# 15, The intrinsic engine

The default Cartan engine (`default_engine`) reasons about
``L_X``, ``ι_X``, ``d`` *as operators on forms*, it knows
``[L_X, ι_Y] = ι_{[X,Y]}``, ``d² = 0``, and friends as bare operator
identities. That's enough for many proofs, but it gets stuck the
moment you ask "what does ``L_X ω`` *evaluate to* on a tuple of
vector fields?". For that you need the **intrinsic** picture: the
textbook formulas

```
(ι_X ω)(Y_1, …) = ω(X, Y_1, …)
(L_X ω)(Y_1, …) = X(ω(Y_1, …)) − Σ_i ω(…, [X, Y_i], …)
(d ω)(Y_0, …, Y_p) = Σ_i (−1)^i Y_i(ω(…)) + Σ_{i<j} (−1)^{i+j} ω([Y_i,Y_j], …)
```

, each "explodes" the operator into a function of the form's values
on vector-field tuples. The intrinsic engine bundles those three
expansions plus the multilinear-`MultiEval` book-keeping needed to
keep the result canonical.

This tutorial covers:

1. `intrinsic_engine()` and the seven base rules (12.A.5);
2. `intrinsic_engine_with_closure()` and the four extra rules that
   close ``d² = 0``, ``[L_X, ι_Y] = ι_{[X,Y]}``, and friends (12.A.6);
3. `prove_intrinsic_equivalence`, the dedicated equivalence prover
   that drives those rules to fix-point;
4. `IntrinsicFormulaRecognizer`, pure-shape inspection of
   `MultiEval(Act(op, ω), Y_1, …)` for higher-level tactics.

## Why a separate engine

The package-wide `default_engine()` doesn't carry multi-eval
axioms, it leaves textbook Cartan obstructions like
``ω(X, [Y,Z]_VF) + ω([Y,Z]_VF, X)`` unreduced. The intrinsic engine
adds the four `MultiEval`-level helpers (arg-linearity,
head-linearity, repeat-arg-zero, alternating-normal) that handle
*signed antisymmetric multilinear* arithmetic on those objects.
That's why `prove_intrinsic_equivalence` exists rather than
overloading `prove_equivalence`, they target different residue
shapes.

## The base bundle

`intrinsic_engine()` returns a fresh `ExpansionEngine` carrying
seven Faz 12.A.5 rules. The order matters, intrinsic operator
rules come before multi-eval helpers so a
`MultiEval(Act(op, ω), …)` is matched by its operator-specific
rule before head-linearity has a chance to scan it.

```python
from jacopy.calculus.intrinsic_engine import intrinsic_engine

eng = intrinsic_engine()
print(f"rules : {len(eng.definitions)}")
for r in eng.definitions:
    print(f"  - {r.name}")
```

The three intrinsic rules (`InteriorProductIntrinsicDefinition`,
`LieDerivativeIntrinsicDefinition`, `ExteriorDIntrinsicDefinition`)
fire on `MultiEval(Act(op, ω), Y_1, …)` and replace the wrapped
operator with the textbook formula. The four `MultiEval` helpers
canonicalise the result.

## `prove_intrinsic_equivalence` on `ι² = 0`

The prover wraps the standard `Sum(lhs, Neg(rhs)) → simplify` cycle
into a single call. The transcript is a `ProofChain` you can render,
diagnose, or feed to publication helpers.

```python
from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.interior import interior
from jacopy.calculus.intrinsic_engine import prove_intrinsic_equivalence
from jacopy.core.expr import Symbol, Integer
from jacopy.core.multi_eval import multi_eval

omega = Symbol("ω")
X, Y = Derivation("X", 0), Derivation("Y", 0)

lhs = multi_eval(Act(interior(X), Act(interior(X), omega)), Y)
chain = prove_intrinsic_equivalence(lhs, Integer(0))
print(f"ι_X ι_X ω = 0 closes in {len(chain)} steps")
for s in chain.steps:
    print(f"  - {s.rule}")
```

Note `Derivation("X", 0)`, at this layer vector fields are
constructed as derivations of degree 0 directly, rather than as
generic symbols with a `Graded` declaration. The intrinsic-engine
rules look for the `Derivation` shape, so this matters.

## Cartan magic on a 2-form

The flagship 12.A test: prove
``(ι_X d + d ι_X) ω = L_X ω`` on a 2-form, evaluated on `(Y, Z)`.
Twelve steps without any closure axioms, the base bundle alone
handles it.

```python
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Sum

omega = Symbol("ω")
X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))

lhs = Sum(
    multi_eval(Act(interior(X), Act(default_d, omega)), Y, Z),
    multi_eval(Act(default_d, Act(interior(X), omega)), Y, Z),
)
rhs = multi_eval(Act(lie_derivative(X), omega), Y, Z)

chain = prove_intrinsic_equivalence(lhs, rhs)
print(f"Cartan magic closes in {len(chain)} steps")
```

Each of those steps is a named, citeable rewrite. Pipe the chain
through `chain_to_latex` (tutorial 11) and you have a publication-
ready transcript of the textbook proof.

## Closure-complete bundle

`intrinsic_engine_with_closure()` adds four Faz 12.A.6 rules that
operate on the **post-expansion residue**, they fold compound
shapes that the base bundle leaves unreduced. The four:

| Rule | Folds |
|---|---|
| `VfActCommutatorDefinition` | `Act(X, Act(Y, f)) − Act(Y, Act(X, f)) → Act([X, Y]_VF, f)` |
| `LieBracketVfAntiSymmetryDefinition` | `[X,Y]_VF(…) + [Y,X]_VF(…) → 0` |
| `LieBracketVfJacobiDefinition` | sign-permuted three-bracket cyclic → `0` |
| `IotaActAsScalarDefinition` | `Act(D, Act(ι_X, ω)) → Act(D, ω(X))` (1-form bridge) |

Together they close ``[L_X, ι_Y] ω = ι_{[X,Y]_VF} ω``, ``d² = 0``,
and ``[L_X, L_Y] ω = L_{[X,Y]_VF} ω`` on 1- and 2-forms. The
example everyone wants:

```python
from jacopy.algebra.lie_bracket_vf import lie_bracket_vf
from jacopy.calculus.intrinsic_engine import intrinsic_engine_with_closure
from jacopy.core.expr import Neg

omega = Symbol("ω")
X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
XY = lie_bracket_vf(X, Y)

lhs = Sum(
    multi_eval(Act(lie_derivative(X), Act(interior(Y), omega)), Z),
    Neg(multi_eval(Act(interior(Y), Act(lie_derivative(X), omega)), Z)),
)
rhs = multi_eval(Act(interior(XY), omega), Z)

chain = prove_intrinsic_equivalence(
    lhs, rhs, engine=intrinsic_engine_with_closure(),
)
print(f"[L_X, ι_Y] ω = ι_{{[X,Y]}} ω closes in {len(chain)} steps")
```

## `IntrinsicFormulaRecognizer`, shape inspection

Sometimes you don't want to *prove* anything, you just want to ask
"is this `MultiEval` actually an intrinsic operator applied to a
form?". The recognizer answers that without running an engine:

```python
from jacopy.calculus.intrinsic_engine import IntrinsicFormulaRecognizer

expr = multi_eval(Act(lie_derivative(X), omega), Y, Z)
match = IntrinsicFormulaRecognizer().recognize(expr)
print(f"operator     : {match.operator}")
print(f"vector_field : {match.vector_field}")
print(f"omega        : {match.omega}")
print(f"args         : {match.args}")
```

`recognize()` returns an `IntrinsicFormulaMatch` (or `None`); the
fields carry exactly enough to *rebuild* an equivalent expression,
so a higher-level tactic that wants to manipulate the form directly
doesn't have to reach back into the original node.

`classify(expr)` is the convenience shortcut that returns just the
operator label (``"interior"`` / ``"lie"`` / ``"exterior_d"``) when
you only need to dispatch.

The recognizer only inspects the *outer* head, nested operators
aren't unwrapped, so `Act(L_X, Act(ι_Y, ω))` reports the inner
`Act(ι_Y, ω)` verbatim as `omega`. That's deliberate: the recognizer
is composable, not greedy.

## When the intrinsic engine is the wrong choice

The intrinsic engine is the right pick when:

* the equality involves `MultiEval(Act(op, ω), Y_1, …)` shapes;
* you want the closure transcript to read like the textbook
  intrinsic-formula computation;
* the residue would otherwise contain unreduced `MultiEval` /
  `LieBracketVF` arithmetic.

Don't use it for:

* generic operator-equation work (``L²``, raw Jacobi on a custom
  bracket), that belongs to `prove_equivalence` with the package
  default engine.
* problem-specific axioms (closure / non-degeneracy / SN bivector
  signs), wrap them via the closure-axiom layer (tutorial 13) or
  reach for a problem wrapper (tutorial 14).

Bonus failure mode you should know about: ``d²`` and ``[L_X, L_Y]``
on a **3-form or higher** still don't close, the closure axioms
are calibrated for 1- and 2-forms. The prover will raise
`ProofFailure` with the surviving residue in the message; the fix
isn't a deeper engine, it's a more refined `Definition` for the
specific residue shape. (See tutorial 10 for diagnostics.)

## Summary

* `intrinsic_engine()` carries seven Faz 12.A.5 rules covering
  multi-eval expansion of `ι_X`, `L_X`, `d` plus the four
  `MultiEval` book-keeping helpers.
* `intrinsic_engine_with_closure()` adds four Faz 12.A.6 closure
  rules, VF-commutator folding, bracket antisymmetry / Jacobi,
  iota-as-scalar bridge, to close ``d² = 0``, ``[L_X, ι_Y] = ι_{[X,Y]}``,
  and ``[L_X, L_Y] = L_{[X,Y]}`` on 1- and 2-forms.
* `prove_intrinsic_equivalence` runs the engine to fix-point and
  returns a `ProofChain`, the entry point for textbook intrinsic
  proofs.
* `IntrinsicFormulaRecognizer` is the pure-shape inspector, answers
  "is this an intrinsic-operator wrapped multi-eval, and if so
  what's inside?" without running any rewriting.
