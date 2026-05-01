# 09, Foundations

This final tutorial drops to one question: "*why* is `d² = 0` an
axiom?" The answer reveals the package's pedagogical backbone,
property provenance, efficient vs foundational mode, the
axiom-vs-theorem classification, and working with custom axiom
sets. Earlier tutorials gave the brackets and theorems; here we
look at what ground sits underneath them and how the package
keeps that ground explicit.

[08, The unified picture](08_unified_picture.md) tied the
theorems together; here we drop into the axiom layer that sits
*beneath* those theorems.

## Two layers of claim

The package carries assertions on two layers:

- **Axiom.** A primitive equation. Example: `d(df) = 0` on
  0-forms, a generic axiom imposed on the generators of
  Ω*(M).
- **Theorem.** A consequence of axioms. Example: `d² = 0` as
  an operator identity, derived for general-degree forms from
  `d(df) = 0` + Leibniz via the agreement-on-generators
  argument.

`ExpansionEngine` carries this distinction through
`Definition.is_theorem`. Every rule is either an axiom
(`is_theorem=False`) or a theorem (`is_theorem=True`); theorems
can produce a sub-proof through `theorem_proof_builder()`.

```python
from jacopy.proof.expansion import default_engine

eng = default_engine()
for d in eng.definitions:
    label = "theorem" if d.is_theorem else "axiom"
    print(f"{label:<8} | {d.name}")
# axiom    | L_X := d∘ι_X + ι_X∘d (Cartan definition)
# axiom    | Act linearity: (A + B)(x) = A(x) + B(x)
# axiom    | d² = 0
# axiom    | ι_X ∘ ι_X = 0
# axiom    | ι_X(f) = 0 on 0-forms
# axiom    | ι_X(df) = X(f)
```

The default engine is conservative, it treats everything as an
axiom. No rule is derived from anything deeper. That's the
"efficient" mode: short, fast, no proof-substrate behind it.

## `d_squared_mode="theorem"`, derive `d² = 0`

Mark `d² = 0` as a *theorem* by configuring the engine
explicitly:

```python
eng_th = default_engine(d_squared_mode="theorem")
[d for d in eng_th.definitions if d.name == "d² = 0"][0].is_theorem
# True
```

The flag alone doesn't change the proof layer; it just says
"this rule is not an axiom, it's a deeper derivation". To
*see* the derivation you need foundational mode.

## Efficient vs foundational mode

`mode="efficient"` (default): fast tag-only proofs. Each rule
fires in a single step with no sub-proof attached.

`mode="foundational"`: when a theorem-class rule fires, a
sub-proof is attached on `ProofStep.children`, showing which
more primitive axiom(s) the rule rests on.

```python
from jacopy.calculus.invariant_d import default_d
from jacopy.core.registry import PropertyRegistry
from jacopy.core.expr import Symbol, Integer
from jacopy.core.properties import Graded
from jacopy.proof.verifier import prove_equivalence

reg = PropertyRegistry()
omega = Symbol("ω")
reg.declare(omega, Graded(degree=2))
expr = default_d(default_d(omega))     # d(d(ω))

eng_eff = default_engine(registry=reg, mode="efficient",
                         d_squared_mode="theorem")
eng_fnd = default_engine(registry=reg, mode="foundational",
                         d_squared_mode="theorem")

eff = prove_equivalence(expr, Integer(0), registry=reg, engine=eng_eff)
fnd = prove_equivalence(expr, Integer(0), registry=reg, engine=eng_fnd)

[(s.rule, len(s.children)) for s in eff.steps]
# [('d² = 0', 0), ('simplify', 0)]

[(s.rule, len(s.children)) for s in fnd.steps]
# [('d² = 0', 1), ('simplify', 0)]

fnd.steps[0].children[0].rule
# 'd(df) = 0 on 0-forms (generator axiom)'
fnd.steps[0].children[0].justification
# 'operator identity d ∘ d = 0 extends from the generator-level axiom
#  d(df) = 0 by agreement on the generators of Ω*(M) ...'
```

The same `d² = 0` step lands on the same `after` value (`0`) in
both modes, but foundational mode also carries the answer to
"so where does it come from?". The generator-level axiom
(`d(df) = 0`) is the *only* primitive input to the argument;
everything else is the "if it agrees on generators it agrees on
all of Ω*(M)" extension principle.

## Custom axiom sets

`default_engine` is a convenience, the actual data is the
`ExpansionEngine.definitions` list. By assembling the list
yourself you can change the package's axiomatic basis:

- Drop `DSquaredZeroDefinition` entirely and keep only the
  generator-level `d(df) = 0`.
- Use `LieDerivativeCartanDefinition` as a *definition* rather
  than an axiom and assume it derived from Cartan's magic
  formula.
- Inject your own algebraic theory's axioms by writing a
  `Definition` subclass.

The `Definition` API is minimal, `matches(expr)`,
`rewrite(expr)`, plus the optional `theorem_proof_builder()`
that supplies the sub-proof in foundational mode.

### Axiom class, the shortest path

A rule that reduces a `c_zero` symbol to zero. You build your
own engine with `ExpansionEngine([...])`, disabling default
rules and running just this one:

```python
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.core.expr import Symbol, Integer, Sum

class ZeroConstAxiom(Definition):
    name = "c_zero := 0 (axiom)"

    def matches(self, expr):
        return isinstance(expr, Symbol) and expr.name == "c_zero"

    def rewrite(self, expr):
        return Integer(0)

engine = ExpansionEngine([ZeroConstAxiom()])
expanded, steps = engine.expand(Sum(Symbol("c_zero"), Symbol("x")))
# expanded:  (0 + x)
# steps:     [ProofStep(rule='c_zero := 0 (axiom)', provenance_tag='axiom')]
```

Without overriding `theorem_proof_builder`, `is_theorem=False`,
no child step even in foundational mode.

### Theorem class, attach a sub-proof

To present the same rule as a theorem, `theorem_proof_builder`
returns a `ProofChain` builder:

```python
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep

class ZeroConstTheorem(Definition):
    name = "c_zero := 0 (theorem)"

    def matches(self, expr):
        return isinstance(expr, Symbol) and expr.name == "c_zero"

    def rewrite(self, expr):
        return Integer(0)

    def theorem_proof_builder(self):
        def build(matched):
            step = ProofStep(
                rule="c_zero = c_zero − c_zero (axiom)",
                before=matched,
                after=Integer(0),
                justification="self-annihilation axiom on c_zero",
                provenance_tag="axiom",
            )
            return ProofChain(steps=[step])
        return build

eff = ExpansionEngine([ZeroConstTheorem()], mode="efficient")
fnd = ExpansionEngine([ZeroConstTheorem()], mode="foundational")

c = Symbol("c_zero")
eff_exp, eff_steps = eff.expand(c)
fnd_exp, fnd_steps = fnd.expand(c)

len(eff_steps[0].children)   # 0
len(fnd_steps[0].children)   # 1
fnd_steps[0].children[0].rule
# 'c_zero = c_zero − c_zero (axiom)'
```

Same `Definition`; in efficient mode it fires atomically, in
foundational mode it attaches a one-step sub-proof. The
package's own `DSquaredZeroDefinition` works the same way,
its sub-proof builder cites the `d(df) = 0` generator axiom.

## Theorem Book structure

Expansion rules carry operator-level provenance; theorem-level
provenance lives in
[`jacopy.library.theorem_book`](../../jacopy/library/theorem_book.py).
The data structure:

```python
from jacopy.library.theorem_book import Theorem
import dataclasses

[f.name for f in dataclasses.fields(Theorem)]
# ['name', 'statement', 'from_axioms', 'proof', 'notes']
```

Five fields:

- `name`, the registry key (e.g. `"poisson_jacobi"`).
- `statement`, a human-readable claim.
- `from_axioms`, `Tuple[str, ...]` of atomic axioms it depends on.
- `proof`, the theorem's canonical `ProofChain`.
- `notes`, extra context (optional).

Querying the singleton registry `theorem_book`:

```python
from jacopy.library import theorem_book

theorem_book.names()
# ('poisson_jacobi', 'poisson_koszul_equivalence',
#  'poisson_koszul_jacobi', 'lie_algebroid_anchor_compat',
#  'courant_jacobi_twist', 'courant_dorfman_bridge',
#  'dirac_isotropy', 'dirac_involutivity')

thm = theorem_book.get("poisson_jacobi")
thm.statement
# '{f, g, h}_π cyclic sum = 0 when [π, π]_SN = 0'
thm.from_axioms
# ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')
thm.proof.steps[0].rule
# 'DerivedBracketTheorem'
```

Seeded theorems are registered at package init time
(`jacopy/library/__init__.py` submodule loads). Downstream code
does *not* re-prove a theorem; it pulls
`theorem_book.get(name).proof` and embeds that chain inside a
larger `ProofChain`. That's the spine of the "single citation,
many uses" strategy, every new library module registers its
theorems and the Theorem Book grows.

## Property provenance, once more

Property tags label themselves as `axiom` or `theorem`. The
same distinction shows up here on `Definition.is_theorem`,
they're really one backbone: "is this claim primitive, or
derived from other primitives?", tracked at both the symbol
level (properties) and the operator level (expansion rules).

Putting it together:

| layer | carrier | primitive | derived |
|-------|---------|-----------|---------|
| symbol | `Property.provenance` | `"axiom"` | `"theorem"` |
| expansion | `Definition.is_theorem` | `False` | `True` |
| theorem | `Theorem.from_axioms` | atomic strings |, |

Same philosophy on all three layers: **every claim's source is
tracked**. When a user asks "which axiom does this rest on?"
the package can answer mechanically.

## Pedagogical close

The whole package is built around one architectural decision:
*never lose provenance*. The decision shows up as:

1. **Property-level.** Every `Property` (graded antisymmetry,
   Leibniz, Jacobi, …) carries the axiom it derives from.
2. **Expansion-level.** Every `Definition` declares whether it
   is an axiom or a theorem; foundational mode answers "what
   sits under this proof?" mechanically.
3. **Theorem-level.** Every `Theorem.from_axioms` declares the
   atomic axioms it depends on; every `theorem_book.get(name)`
   call generates a citation chain.
4. **Bracket-level.** Structural theorems like the Derived
   Bracket Theorem share an obstruction inside a "single
   hypothesis, many consequences" frame, and the package
   detects that automatically.

The package was built so that *how do you know?* stays
answerable from symbol all the way up to theorem. Every
`ProofChain` is an argument tree, roots in axioms, leaves at
`Integer(0)`. The pedagogical value follows: a user grasps a
theorem together with the structure beneath it, not as a
black box.

## End of the tutorial series

Nine chapters:

1. [01, First steps](01_first_steps.md)
2. [02, The Jacobi identity](02_jacobi_identity.md)
3. [03, Poisson geometry](03_poisson_geometry.md)
4. [04, Lie algebroid](04_lie_algebroid.md)
5. [05, Cartan calculus](05_cartan_calculus.md)
6. [06, Custom bracket](06_custom_bracket.md)
7. [07, Derived bracket](07_derived_bracket.md)
8. [08, The unified picture](08_unified_picture.md)
9. **09, Foundations** ← you are here

From symbol to theorem, axiom to unified picture, at every
step we saw what the package can actually deliver, with live
API examples. From here on the package is a tool: add your own
brackets, your own theorems, your own axiom sets, and run on
top of it.
