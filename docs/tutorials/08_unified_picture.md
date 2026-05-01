# 08, The unified picture

The pedagogical claim of this package: Poisson geometry + Lie
algebroid + Cartan calculus + Courant geometry are all faces of a
single mathematical mechanism. At the centre stands the *Derived
Bracket Theorem*, a bracket's two structural axioms (antisymmetry
+ graded Jacobi) reduce to a single equation, `[Q, Q]_base = 0`.
This tutorial shows how the same hypothesis closes both the
function-side and form-side Jacobi identities, how `theorem_book`
serves the hierarchy through a single citation chain, and why
"one assumption, many consequences" falls out naturally from the
architecture.

[07, Derived bracket](07_derived_bracket.md) introduced the
mechanism; [05, Cartan calculus](05_cartan_calculus.md) opened
operator-level proofs. Here we tie those pieces together.

## A single hypothesis: `[π, π]_SN = 0`

Calling a bivector `π` a "Poisson bivector" is the same as
asserting it anti-commutes with itself under the
Schouten–Nijenhuis bracket: `[π, π]_SN = 0`. `jacopy` produces
this equation as a universal `VanishingCondition`:

```python
from jacopy.library.declarations import Bivector, Forms, Functions
from jacopy.library.poisson import PoissonBracket
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
pi = Bivector("π", registry=reg)
poisson = PoissonBracket.from_bivector(pi)

poisson.jacobi_condition(reg).obstruction     # [·,·]_SN(π, π)
poisson.koszul_jacobi_condition(reg).obstruction  # the same
```

The two conditions point to the *same* `Expr`, only the display
name differs. Not a coincidence: the derived bracket's
obstruction depends only on `(base, Q)`; layering
`acting_on=π^♯` on top doesn't change the Jacobi obstruction.

## The same condition, two proof faces

**Function level.** The classical Poisson Jacobi identity
vanishes the cyclic sum on three functions.
`prove_jacobi_reduction` reduces that to the obstruction in a
single *DerivedBracketTheorem* step:

```python
f, g, h = Functions("f g h", degree=-1, registry=reg)
func_chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
func_chain.steps[0].rule    # 'DerivedBracketTheorem'
func_chain.steps[0].after   # [·,·]_SN(π, π)
```

**Form level.** The same mechanism gives the Koszul Jacobi
identity on 1-forms. `prove_koszul_jacobi_reduction` produces
the same theorem citation and lands on the *same*
`[π, π]_SN` obstruction, the `π^♯` anchor is in play only for
operand lifting:

```python
alpha, beta, gamma = Forms("α β γ", degree=1, registry=reg)
form_chain = poisson.prove_koszul_jacobi_reduction(
    alpha, beta, gamma, registry=reg
)
form_chain.steps[0].rule    # 'DerivedBracketTheorem'
form_chain.steps[0].after   # [·,·]_SN(π, π)

# They intersect at the same obstruction:
func_chain.steps[0].after == form_chain.steps[0].after   # True
```

The mathematical content: "classical Poisson Jacobi" and
"classical Koszul Jacobi" are *not* two separate theorems,
both are the Derived Bracket Theorem reducing to the same
universal obstruction.

## The classical–derived bridge

The same hypothesis additionally gives the theorem "classical
Koszul bracket = SN-derived bracket (with `π^♯`)".
`PoissonBracket` verifies it in a single step:

```python
chain = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)
len(chain)                # 1
chain.steps[0].rule       # 'reflexive', both sides land on the same canonical form
```

The meaning of the `reflexive` step: the package's expand rules
brought both sides to the same `Expr` tree; equality holds
*structurally*. That structural identity is *exactly why* Koszul
Jacobi reduces to the same obstruction as function Jacobi.

## Citation chain via seeded theorems

Three seeded theorems wait inside `theorem_book`. Each declares
its dependence on atomic axioms via `from_axioms`, the
theorem-level counterpart of the package's *property provenance*
philosophy:

```python
from jacopy.library import theorem_book

theorem_book.get("poisson_jacobi").from_axioms
# ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')

theorem_book.get("poisson_koszul_equivalence").from_axioms
# ('derived bracket definition', 'classical Koszul bracket definition',
#  'π^♯ = Sharp(π) as common anchor')

theorem_book.get("poisson_koszul_jacobi").from_axioms
# ('Derived Bracket Theorem', 'π^♯ = Sharp(π) as form-lift anchor',
#  '[π, π]_SN = 0 (Poisson hypothesis)')
```

Each citation has a `ProofChain` behind it, the theorem's
canonical proof. Downstream code can take that chain via
`theorem_book.get(...)` and embed it directly into a larger proof:

```python
thm = theorem_book.get("poisson_jacobi")
thm.proof                 # ProofChain(1 steps)
thm.proof.steps[0].rule   # 'DerivedBracketTheorem'
```

The same pattern shows up for the Lie algebroid
(`lie_algebroid_anchor_compat`) and Courant geometry
(`courant_jacobi_twist`, `courant_dorfman_bridge`,
`dirac_isotropy`, `dirac_involutivity`), the package currently
carries 8 seeded theorems.

## A parallel example: `dH = 0`

The same "single equation, many consequences" pattern repeats on
the Courant side with a different hypothesis. The H-twisted
Courant bracket's graded Jacobi holds if and only if the twist
3-form is closed (`dH = 0`):

```python
from jacopy.brackets.courant import CourantBracket
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded

reg_h = PropertyRegistry()
H = Symbol("H")
reg_h.declare(H, Graded(degree=3))

C = CourantBracket(background_H=H)
cond = C.jacobi_condition(reg_h)
cond.obstruction          # d(H)

theorem_book.get("courant_jacobi_twist").from_axioms
# ('Courant algebroid Jacobi axiom', 'dH = 0 (closed-3-form hypothesis)')
```

The structural analogy: a single equation `[π, π]_SN = 0` on the
Poisson side, a single equation `dH = 0` on the Courant side,
both reduce a bracket's two structural axioms (antisymmetry +
Jacobi) to one condition, and the theorem book serves the
reduction as a single citation.

## Pedagogical takeaways

Observable consequences of the package design:

1. **One theorem, many consequences.** The Derived Bracket
   Theorem is proved once; Poisson function Jacobi, Poisson
   form Jacobi, Koszul Jacobi, Courant Jacobi are all
   instantiations.
2. **Shared obstruction.** Two distinct proof paths (function
   vs form) that land on the same `Expr` are backed by the same
   theorem. The package catches this mechanically, even when
   display names differ.
3. **Traceable citation chain.**
   `theorem_book.get(name).from_axioms` exposes which atomic
   axioms a theorem rests on; one can trace a paper's proof
   flow alongside the corresponding `from_axioms` lists.
4. **New bracket, old theorem.** A user defining a fresh
   derived bracket doesn't redo the Jacobi proof from scratch,
   `prove_jacobi` auto-dispatches to `DerivedBracketStrategy`
   and reuses the same theorem citation.

## Next step

This tutorial answered "which theorem connects to where?". The
final tutorial, [09, Foundations](09_foundations.md), drops
to "where does an axiom come from?". Why is `d² = 0` an axiom;
what sub-proof generates it from the generator level in
foundational mode; how the proof layer reconfigures itself when
working with a custom axiom set.
