# 10, Diagnostics & proof debugging

This tutorial covers what to do when a `prove_equivalence` call **fails**.
By the end you'll know how to read a `ProofFailure`, attach a
`DiagnosticReport` to surviving residuals, and extend the diagnostic
catalogue with your own rules.

The diagnostic layer was added in Phase 11.A, before it, a stalled
proof gave you a bare residual `Expr` and an exception message; you
were on your own to spot which rewrite never fired. The layer
mechanises the first pass of that detective work: `diagnose(residual)`
walks the surviving expression, recognises *stalled shapes* (a
`d(d(ω))` that never collapsed under d² = 0, an `Act(op, 0)` that
never got annihilated, an unclassified factor with no grading
evidence), and emits a hint per shape, a hypothesis list, not a
proof, that points you at the specific rewrite to check.

## A failing proof

We'll start by building an `ExpansionEngine` that's deliberately
missing the `d² = 0` rule so we can watch a real failure surface.
Forming the obstruction `d(d(ω)) == 0` should close trivially in the
default engine, but our hand-built one doesn't know `d² = 0`, so it
stalls.

```python
from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import d
from jacopy.core.expr import Symbol, Integer
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import prove_equivalence, ProofFailure
from jacopy.proof.expansion import (
    ExpansionEngine,
    LieDerivativeCartanDefinition,
    ActOverSumOpDefinition,
    IotaSquaredZeroDefinition,
    IotaOnZeroFormDefinition,
    IotaOnExactOneFormDefinition,
)

reg = PropertyRegistry()
omega = Symbol("ω"); reg.declare(omega, Graded(degree=2))

# Engine without DSquaredZeroDefinition.
engine_no_d2 = ExpansionEngine([
    LieDerivativeCartanDefinition(),
    ActOverSumOpDefinition(),
    IotaSquaredZeroDefinition(),
    IotaOnZeroFormDefinition(),
    IotaOnExactOneFormDefinition(d=d),
])

try:
    chain = prove_equivalence(
        Act(d, Act(d, omega)),
        Integer(0),
        registry=reg,
        engine=engine_no_d2,
    )
except ProofFailure as exc:
    print(exc)
```

The exception message is two-tier: a one-line summary
(`ExpandAndSimplify left residual d(d(ω)) when proving d(d(ω)) == 0`)
followed by a structured `DiagnosticReport` block. The summary tells
you *that* the proof stalled; the report tells you *why* the engine
couldn't reduce the residual further.

## Reading the report

`ProofFailure.report` is a `DiagnosticReport` (or `None` if the
strategy didn't attach one). It carries:

* `report.residual`, the surviving `Expr`, exactly the term that
  `simplify` couldn't drive to `0`.
* `report.hints`, a list of `DiagnosticHint`, one per recognised
  stalled shape.

Each hint exposes four fields:

| Field | Type | What it says |
|---|---|---|
| `category` | `str` | Machine key (`stalled-d-squared`, `unclassified-factor`, …), filter on this in code. |
| `message` | `str` | Human-readable description of the stall. |
| `location` | `Optional[Expr]` | The offending sub-expression. |
| `suggestion` | `Optional[str]` | Concrete fix (often "enable mode X" or "register definition Y"). |

```python
try:
    prove_equivalence(
        Act(d, Act(d, omega)), Integer(0),
        registry=reg, engine=engine_no_d2,
    )
except ProofFailure as exc:
    report = exc.report
    print(f"residual : {report.residual}")
    print(f"hints    : {len(report)}")
    for hint in report:
        print(f"  [{hint.category}] {hint.message}")
        if hint.suggestion is not None:
            print(f"    fix: {hint.suggestion}")
```

## Calling `diagnose()` directly

You don't need a `ProofFailure` to use the diagnostic layer.
`diagnose(expr, registry=…)` runs the full rule catalogue against any
expression. Useful when you're poking at an intermediate term and want
to know what the engine *would* hint about it.

```python
from jacopy.core.expr import Product
from jacopy.proof import diagnose

mystery = Symbol("M")  # never registered
report = diagnose(Product(mystery, omega), registry=reg)
print(report.format())
```

`format()` is the same renderer used by `ProofFailure.__str__`, so
the output is one-to-one with what you'd see if the residual showed
up in a real failure.

## The built-in rule catalogue

The rules ship in `jacopy.proof.diagnostic_rules` and register
themselves on import. The five seeded rules cover the modelling gaps
that historically appeared when porting Cartan-calculus proofs:

| Category | When it fires | Suggested fix |
|---|---|---|
| `stalled-d-squared` | `Act(d, Act(d, x))` survived | enable `d_squared_mode="axiom"` or register `DSquaredZeroDefinition` |
| `stalled-iota-squared` | `Act(ι_X, Act(ι_X, x))` survived | register `IotaSquaredZeroDefinition` |
| `stalled-act-over-zero` | `Act(op, 0)` reached the residual | extend `_expand_act` to recognise `Integer(0)` |
| `stalled-act-over-neg-op` | `Act(Neg(op), x)` failed to peel the sign | extend `_expand_act` to peel `Neg` |
| `unreduced-iota-on-df` | `ι_V(d(f))` where `V` is a sum/product of derivations | upgrade `IotaOnExactOneFormDefinition` for compound vector fields |
| `unclassified-factor` | a `Product` factor has no grading evidence | declare the factor as `Scalar()` or `Graded(degree=…)` |
| `symbol-vector-field` | a bare `Symbol` plays a vector-field role without a `Derivation` wrapper | wrap the symbol in `VectorField` or register it as `Graded` |

Filter to a single category with `report.by_category("stalled-d-squared")`
when you only care about one class of hint.

## Adding your own rule

A diagnostic rule is just a function
`(expr, registry, engine) → Iterable[DiagnosticHint]`. Register it
with the `register_rule` decorator and it joins the catalogue
immediately, no engine wiring required, because diagnostics are
read-only on the residual tree.

```python
from jacopy.proof import DiagnosticHint, register_rule

@register_rule
def warn_on_unregistered_omega(expr, registry, engine):
    """Toy rule: flag bare 'ω' that lost its Graded declaration."""
    from jacopy.core.expr import Symbol
    if registry is None:
        return
    stack = [expr]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Symbol) and cur.name == "ω":
            if not registry.has(cur, Graded):
                yield DiagnosticHint(
                    category="omega-unregistered",
                    message="ω appears without a registered grading",
                    location=cur,
                    suggestion="reg.declare(ω, Graded(degree=...))",
                )
        stack.extend(getattr(cur, "children", ()))
```

Rules are pure functions on the residual tree, so they compose: the
dispatcher de-duplicates hints by `(category, location)` if two rules
catch the same stall.

## When the report is empty

`bool(report)` is `False` when no rule fired, the residual is in a
shape the catalogue doesn't recognise. That's a signal to either (a)
inspect the term by hand to find a missing rewrite the existing rules
should know about, then add a rule, or (b) accept that the stall is
genuine, the equality you tried to prove may simply not hold.

## Summary

* `prove_equivalence` raises `ProofFailure` on stall; `exc.report`
  carries a `DiagnosticReport`.
* The report bundles `residual` + a list of `DiagnosticHint`s, each
  with `category` / `message` / `location` / `suggestion`.
* `diagnose(expr, registry=…)` runs the same pipeline against any
  expression, handy for inspecting intermediate terms.
* The built-in catalogue covers d²/ι² stalls, `Act` linearity gaps,
  unreduced iota-on-df, and unclassified `Product` factors.
* Add a rule with `@register_rule`, it's a one-file change with no
  engine plumbing.
