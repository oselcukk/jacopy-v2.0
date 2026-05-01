# 07, Derived bracket

The mathematical heart of the package: starting from a single
bracket and a chosen *generator*, an entire bracket family arises
automatically. This construction, the **derived bracket**,
defines `{a, b}_Q := [[a, Q]_base, b]_base`. The Leibniz axiom
holds *regardless* of what `Q` is; antisymmetry and Jacobi reduce
to a single equation, `[Q, Q]_base = 0`. The Poisson, Koszul, and
Courant brackets are all instances of this construction, the
Derived Bracket Theorem is proved once and applied to each by a
single citation.

[06, Custom bracket](06_custom_bracket.md) showed how to plug
your own rule into the generic `GradedBracket` skeleton. Here the
rule is built automatically, you just pick `(base, Q)`.

## The construction, `{a, b}_Q := [[a, Q]_base, b]_base`

```python
from jacopy.brackets.derived import DerivedBracket, derived_bracket
from jacopy.brackets.lie import LieBracket
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
Q = Symbol("Q")
reg.declare(Q, Graded(degree=1))
lie = LieBracket()

d = DerivedBracket(lie, Q, degree_Q=1)
d.name           # '{·,·}_Q'
d.degree         # Degree.const(-1), formula |Q| − 2
d.satisfies_leibniz          # True  (universal)
d.is_graded_antisymmetric    # True  (recorded as a theorem result)
d.satisfies_graded_jacobi    # None  (conditional, depends on [Q,Q]_base = 0)
```

`d.satisfies_graded_jacobi` reports `None`: not unconditionally
`True`/`False`, but **conditional**. The proof layer handles that
condition by reducing `[Q, Q]_base` to zero.

The degree formula: `|{·,·}_Q| = |Q| − 2`. Without
`degree_Q`, the default is 0, which makes the derived bracket
shift by `−2`, generally not meaningful. Live examples cluster
around `degree_Q=1` (Poisson, Koszul, Courant).

## The two-faced expansion

To *see* the two-stage expansion you have two faces:

- `expand(a, b, registry)`, the full expansion; both inner and
  outer `[·, ·]_base` nodes are resolved.
- `expand_definition(a, b, registry)`, the definition's *surface*:
  the outer and inner `BracketApply` are kept inert as two
  layers.

```python
a, b = Symbol("a"), Symbol("b")
for s in (a, b):
    reg.declare(s, Graded(degree=0))

d.expand(a, b, reg)
# ((((a * Q) + (-(Q * a))) * b) + (-(b * ((a * Q) + (-(Q * a))))))

d.expand_definition(a, b, reg)
# [·,·]([·,·](a, Q), b)
```

`expand_definition` is meant for proof presentation, "here is
the canonical form of the derived bracket"; `expand` is what
feeds into the simplify pipeline.

## Jacobi obstruction, the universal reduction

The Derived Bracket Theorem: graded Jacobi for `{·,·}_Q` holds
on **every** triple if and only if a *single* expression
vanishes: `[Q, Q]_base`. This is the same mathematical argument
for every derived bracket in the package, the theorem is proved
once and flows as a citation to every instantiation.

The API exposes three faces:

```python
d.jacobi_obstruction(reg)
# ((Q * Q) + (-(Q * Q))) , full expansion through the base bracket

d.jacobi_obstruction_raw()
# [·,·](Q, Q)             , base bracket inert, for display

d.jacobi_condition(reg)
# VanishingCondition(obstruction=..., name='Jacobi condition on {·,·}_Q')
```

`VanishingCondition.holds(registry)` runs the obstruction through
`simplify` and checks whether it collapses to `Integer(0)`. For a
Lie base, `[Q, Q] = Q*Q − Q*Q → 0` falls out immediately:

```python
d.jacobi_condition(reg).holds(reg)   # True
```

## `prove_jacobi`, `DerivedBracketStrategy` dispatch

`jacopy.proof.verifier.prove_jacobi` picks a path based on the
bracket type. For a `DerivedBracket` it dispatches automatically
to `DerivedBracketStrategy`, a three-step chain:

1. `DerivedBracketTheorem` (theorem step): the triple cyclic
   Jacobi sum reduces to the universal obstruction `[Q, Q]_base`.
2. `base-bracket-expand`: the base bracket's own rule opens
   `[Q, Q]`.
3. `simplify`: collapses to zero in canonical form.

```python
from jacopy.proof.verifier import prove_jacobi

a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
for s in (a, b, c):
    reg.declare(s, Graded(degree=0))

chain = prove_jacobi(d, a, b, c, registry=reg)
len(chain)                    # 3
[s.rule for s in chain.steps]
# ['DerivedBracketTheorem', 'base-bracket-expand', 'simplify']
chain.steps[-1].after         # 0
```

If the obstruction can't be reduced to zero (e.g. the base
bracket has no rewrite that resolves `[Q, Q]`),
`DerivedBracketStrategy` returns the residual through
`ProofFailure`, the theorem's condition is not met.

## `acting_on`, Koszul equivalence

On a Poisson manifold the classical Koszul bracket lives on
1-forms and produces the three-term formula. The same structure
falls out of a derived bracket built on the Schouten–Nijenhuis
base, provided you supply `acting_on=π^♯` (or a generic anchor
`ρ`).

When `acting_on` is supplied, `expand` automatically emits the
Koszul form:

```python
from jacopy.brackets.derived import DerivedBracket
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.anchor import Anchor

reg = PropertyRegistry()
pi = Symbol("π")
reg.declare(pi, Graded(degree=1))
alpha, beta = Symbol("α"), Symbol("β")
for s in (alpha, beta):
    reg.declare(s, Graded(degree=1))

rho = Anchor("ρ")
koszul_derived = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho)

koszul_derived.expand(alpha, beta)
# (L_ρ(α)(β) + (-L_ρ(β)(α)) + (-d(⟨ρ(α), β⟩)))

KoszulBracket(rho).expand(alpha, beta)
# (L_ρ(α)(β) + (-L_ρ(β)(α)) + (-d(⟨ρ(α), β⟩)))
```

The two outputs are *structurally* equal. That's the symbolic
verification of "the classical Koszul bracket is the
`(π, π)`-derived bracket on the SN base, anchored by Sharp
(`π^♯`)".

If `acting_on=None` (default), the canonical
`{a,b}_Q = [[a,Q],b]` path is preserved, no silent rewriting.
The anchor also enters the identity key:
`DerivedBracket(sn, π, degree_Q=1, acting_on=Anchor("ρ1"))` is a
*different* bracket from the same construction with
`acting_on=Anchor("ρ2")`.

## Poisson-as-derived

`DerivedBracket(sn, π, degree_Q=1)` *is* the derived construction
of the Poisson bracket. Calling `prove_jacobi` through the
generic dispatcher reduces the obstruction to `[·,·]_SN(π, π)`,
which surfaces as a `ProofFailure`, the honest mathematical
diagnosis: the Poisson hypothesis `[π, π]_SN = 0` is *not* a
generic simplify rule, it must be carried as an explicit
assumption. For production use, prefer the
`jacopy.library.poisson.PoissonBracket` wrapper, it surfaces
the seeded theorem `poisson_jacobi` as a single-step citation:

```python
from jacopy.library import theorem_book
from jacopy.library.declarations import Bivector, Functions
from jacopy.library.poisson import PoissonBracket

reg = PropertyRegistry()
pi = Bivector("π", registry=reg)
f, g, h = Functions("f g h", degree=-1, registry=reg)

poisson = PoissonBracket.from_bivector(pi)
chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
len(chain)                    # 1
chain.steps[0].rule           # 'DerivedBracketTheorem'
chain.steps[0].after          # [·,·]_SN(π, π)

theorem_book.get("poisson_jacobi").from_axioms
# ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')
```

[03, Poisson geometry](03_poisson_geometry.md) lays out the
three-view presentation of the same path; here we just flag
that the derived construction sits underneath Poisson.

## H-twist, the Courant bracket's conditional Jacobi

The same reduction lands on a different equation for another
bracket family. The H-twisted Courant bracket
`[(X,α),(Y,β)]_C = ([X,Y], L_X β − L_Y α − ½ d(ι_X β − ι_Y α) +
ι_Y ι_X H)` satisfies graded Jacobi if and only if `dH = 0`.
`CourantBracket(background_H=H)` exposes that condition directly
through `jacobi_condition`:

```python
from jacopy.brackets.courant import CourantBracket

reg = PropertyRegistry()
H = Symbol("H")
reg.declare(H, Graded(degree=3))

C = CourantBracket(background_H=H)
C.is_twisted                  # True
cond = C.jacobi_condition(reg)
cond.name                     # 'Courant Jacobi condition (H-twisted by H)'
cond.obstruction              # d(H)

CourantBracket().jacobi_condition(reg).name
# 'Courant Jacobi (untwisted, vacuous)'
```

`CourantBracket` itself is *not* a `DerivedBracket` subclass,
operands are section pairs (TM ⊕ T*M) and the half-and-half
expansion (with the Dorfman bridge) is treated separately. But
its presentation of conditional Jacobi is identical to the
derived-bracket machinery: a single equation (`dH = 0`), a single
`VanishingCondition`.

## Summary

The derived bracket's single-equation reduction stitches the
package around one theorem:

| bracket | base | Q | condition |
|---------|------|---|-----------|
| Poisson (`{f,g}_π`) | `sn` | `π` | `[π, π]_SN = 0` |
| Koszul classical | `sn` + `acting_on=π^♯` | `π` | (same) |
| Courant (H-twisted) |, |, | `dH = 0` |
| generic | any | chosen | `[Q, Q]_base = 0` |

## Next step

The closing piece that turns the table above into a single
mathematical picture: how `DerivedBracket` + `CartanCalculus` +
`TheoremBook` + `ProofChain` are used together,
[08, Unified picture](08_unified_picture.md) (Stage D).
