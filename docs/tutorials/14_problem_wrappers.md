# 14, Solving textbook problems with Problem wrappers

The lower-level pieces of `jacopy` (`Symbol`, `BracketApply`,
`prove_equivalence`, `default_engine`) give you everything you need,
*if* you're willing to wire each problem from scratch: declare every
grading, register every defining relation, layer every closure axiom
on the engine. For one-off experiments that's fine. For a textbook's
worth of related questions on the same `(M, ω)` it's noisy and
error-prone.

The **Problem wrapper** layer collapses that boilerplate. A
`SymplecticProblem(ω, [f, g, …], registry=…)` instance bundles:

* the manifold structure (form, optional bivector, anchors / musical
  maps);
* one `HamiltonianVectorField` per designated function, each registered
  with its defining relation `ι_{X_f} ω = ±df` on the engine;
* the closure axioms (`Closed(ω)`, `NonDegenerate(ω)`, `Antisymmetric(π)`)
  auto-declared on the registry;
* a pre-wired `ExpansionEngine` carrying the symplectic-specific rules
  (closed form, non-degeneracy peel, C∞-linearity in pairings, musical
  compatibility when `π` is supplied).

You then ask the wrapper for proofs through `prove_*` methods. Each
returns a `ProofChain` you can inspect, render, or feed into the
diagnostic / publication helpers from tutorials 10 and 11.

This tutorial walks through the two main wrappers, `SymplecticProblem`
(form-side) and `KoszulProblem` (Poisson form-bracket side), on
problems lifted from §2 of the question bank.

## Setting up a `SymplecticProblem`

Three things to feed in: the symplectic 2-form `ω`, the functions you
care about (each becomes a Hamiltonian), and a registry. Everything
else is auto-wired.

```python
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.library.symplectic_problem import SymplecticProblem

reg = PropertyRegistry()
omega = Symbol("ω"); reg.declare(omega, Graded(degree=2))
f = Symbol("f");  reg.declare(f, Scalar())
g = Symbol("g");  reg.declare(g, Scalar())

prob = SymplecticProblem(omega, [f, g], registry=reg)
print(prob)
```

The wrapper auto-declares `Closed(ω)` and `NonDegenerate(ω)` on the
registry, this is the symplectic problem statement, not a separate
hypothesis you need to remember to assert. It also creates one
`HamiltonianVectorField` per function (`X_f`, `X_g`) with the defining
relation `ι_{X_f} ω = −df` baked into the engine (`sign="-"` is the
default; pass `sign="+"` for the textbook convention).

## Question 2a, Hamiltonian invariance

The first canonical problem on a symplectic form: the symplectic form
is preserved under the Hamiltonian flow, `L_{X_f} ω = 0`. With the
wrapper, this is one method call:

```python
chain = prob.prove_hamiltonian_invariance(f)
print(f"closure: {chain.initial} → {chain.final}")
print(f"steps  : {len(chain)}")
for i, step in enumerate(chain.steps):
    print(f"  [{i:02d}] {step.rule:35s} {step.before} → {step.after}")
```

The chain is the textbook computation, mechanised:

```
L_{X_f}(ω)
  → d(ι_{X_f} ω) + ι_{X_f}(d ω)        # Cartan magic
  → d(−df) + ι_{X_f}(0)                # defining relation + Closed(ω)
  → −d(d(f)) + 0
  → 0                                   # d² = 0
```

Every step has a named rule (`L_X := d∘ι_X + ι_X∘d`,
`Closed: d(ω) = 0`, `d² = 0`, …) so you can read the proof end-to-end
and spot exactly where each axiom enters.

## Question 2c, Hamiltonian equality

A more involved problem: prove the bracket `{f,g}_ω` defines a
Hamiltonian via `[X_f, X_g] = X_{\{f,g\}}`. The wrapper exposes two
helpers that close this kind of question, `prove_vector_field_equality`
(reduces to `Y = Z` via non-degeneracy) and `prove_hamiltonian_equality`
(closes `ι_Y ω = ±dh` directly).

The simplest demonstration is the trivial reflexive case, prove that
each Hamiltonian equals itself as a vector field:

```python
Xf = prob.hamiltonian(f)
chain = prob.prove_vector_field_equality(Xf, Xf)
print(f"steps: {len(chain)}, final: {chain.final}")
```

That's a 3-step closure: the obstruction `ι_{X_f} ω − ι_{X_f} ω` peels
through the `NonDegenerate` rule to `X_f − X_f`, then `simplify`
collapses it to `0`. The non-trivial use is when `Y` is e.g. a Lie
bracket of two Hamiltonians and you want to recognise it as a third
Hamiltonian, same call, different operands.

## Setting up a `KoszulProblem`

The form-side counterpart: a Poisson bivector `π` with a designated
list of forms, the Koszul bracket on 1-forms, and the tilde calculus
machinery (L̃ / ι̃ / d̃) pre-wired.

```python
from jacopy.library.koszul_problem import KoszulProblem

reg2 = PropertyRegistry()
pi    = Symbol("π"); reg2.declare(pi,    Graded(degree=2))
alpha = Symbol("α"); reg2.declare(alpha, Graded(degree=1))
beta  = Symbol("β"); reg2.declare(beta,  Graded(degree=1))

kprob = KoszulProblem(pi, [alpha, beta], registry=reg2)
print(kprob)
print(f"bracket: {kprob.koszul_bracket}")
print(f"sharp  : {kprob.sharp}")
```

The wrapper auto-declares `Antisymmetric(π)` and exposes the bracket-
expansion rule directly:

```python
from jacopy.brackets.base import BracketApply

raw = BracketApply(kprob.koszul_bracket, alpha, beta)
print("input :", raw)
print("output:", kprob.bracket_expansion_rule.rewrite(raw))
```

The rewrite recovers the classical Koszul formula
`L_{π^♯α} β − L_{π^♯β} α − d⟨π^♯α, β⟩` term by term, useful when you
want to reduce a `[α,β]_K` expression by hand without committing to a
full proof closure.

## When to step outside the wrapper

The wrapper is a convenience, not a wall. Three escape hatches:

1. **The pre-wired engine is exposed as `prob.engine`**, feed it to
   `prove_equivalence(..., engine=prob.engine)` to drive arbitrary
   equalities under the wrapper's axiom set. Useful when the
   problem you want to close isn't one of the named `prove_*`
   helpers.
2. **The hamiltonians are accessible via `prob.hamiltonian(f)`**, you
   can build expressions with them (e.g. `Act(X_f, omega)`) and pass
   them to a strategy directly.
3. **The wrapper never overrides existing registry declarations**,
   if you've already declared `Graded` / `Scalar` / `Closed` on the
   relevant operands, the wrapper sees them and stays out of the
   way. Pre-declaring is the way to override the wrapper's default
   conventions.

## Anatomy of a wrapper

The naming convention is consistent across the library:

| Wrapper | Bundles | Lives in |
|---|---|---|
| `SymplecticProblem` | `(M, ω, π?, {f_i}, registry, engine)` | `library/symplectic_problem.py` |
| `KoszulProblem` | `(π, ρ = π^♯, K, {α_i}, …)` + tilde calculus | `library/koszul_problem.py` |
| `BianchiProblem` | `(connection, registry)` for T̃-Bianchi I/II | `library/bianchi_problem.py` |
| `CartanFormPropertyProblem` | `(connection, frame)` for §3.1.6 props | (Q9 stage 9.D) |
| `CartanStructureProblem` | `(connection, frame)` for Cartan I/II | (Q7 / Q9) |
| `KoszulConnectionProblem` | facade over the three above for Koszul connections | `library/koszul_connection_problem.py` |

Each follows the same pattern, `__init__` validates inputs and
declares structural axioms on the registry, properties expose the
underlying objects, and `prove_*` methods drive closures via a
pre-built engine. Tutorial 24 (in the backlog) walks through writing
your own following the same recipe.

## Summary

* Problem wrappers bundle `(structure, designated operands, registry,
  engine)` so you don't re-wire the same axioms across every question.
* `SymplecticProblem` covers form-side problems on `(M, ω)`,
  Hamiltonian invariance, vector-field equality, Hamiltonian equality.
* `KoszulProblem` covers Poisson form-bracket problems,
  `[α, β]_K` expansion, tilde calculus, derivator engines.
* Wrappers don't override existing declarations; they fill in the
  *structural* axioms (Closed / NonDegenerate / Antisymmetric) when
  you haven't.
* The pre-built engine is exposed via `prob.engine` for closures
  outside the named helpers.
