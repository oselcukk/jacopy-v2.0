# 03, Poisson geometry

This tutorial puts the Poisson bracket on a symplectic manifold
and walks the three equivalent views of `{f, g}_π`, derived,
Hamiltonian, Koszul, through a single `PoissonBracket` object.
The goal is to set up both element-level computations and the
Jacobi proof reduced to the universal condition `[π, π]_SN = 0`.

Familiarity with [first steps](01_first_steps.md) and the [Jacobi
identity](02_jacobi_identity.md) tutorials is assumed.

## The symplectic manifold, `(ω, π, ♭, ♯)` bundle

`SymplecticManifold(ω, bivector=π)` keeps four pieces of data
together: the form `ω`, the inverse bivector `π`, the musical
maps `ω^♭` / `π^♯`, and the `MusicalCompatibility` axiom between
them, `ω^♭ ∘ π^♯ = id`. The wrapper adds nothing mathematically
new; its value is keeping the `(ω, π, ♭, ♯, compat)` tuple
consistent at construction time.

```python
from jacopy import Bivector, Forms, Functions
from jacopy.core.registry import PropertyRegistry
from jacopy.library.symplectic import SymplecticManifold

reg = PropertyRegistry()
(omega,) = Forms("ω", degree=2, registry=reg)
pi = Bivector("π", registry=reg)

M = SymplecticManifold(omega, bivector=pi, name="(M, ω, π)")
M.flat          # ω♭
M.sharp         # π♯
M.compatibility # MusicalCompatibility(ω, π, ...)
```

Note: the `Bivector` helper declares `Graded(degree=1)` for `π`,
that's the SN-shifted degree of a 2-vector (`k − 1 = 1`).
`SymplecticManifold` builds on top of the same registry.

## `PoissonBracket`, three equivalent views

`PoissonBracket.from_bivector(π)` ties three presentations of the
bracket to methods on the same object. Functions in this working
registry must carry SN-shifted degree `−1` (a 0-form sits at `−1`
in SN), the `degree=-1` kwarg of the `Functions` helper exists
for precisely this context.

```python
from jacopy.library.poisson import PoissonBracket

f, g, h = Functions("f g h", degree=-1, registry=reg)
poisson = PoissonBracket.from_bivector(pi)
```

### View 1, the derived bracket

`{f, g}_π = [[f, π]_SN, g]_SN`, the derived-bracket form built
on the Schouten–Nijenhuis lift:

```python
poisson.expand(f, g, reg)   # [·,·]_SN(f, π)(g)
```

### View 2, the Hamiltonian vector field

`{f, g}_π = X_f(g)`, the action of `f`'s Hamiltonian vector
field on `g`. `via_hamiltonian` returns the symbolic form,
exposing the structure behind the bracket.

```python
poisson.via_hamiltonian(f, g)  # X_f(g)
poisson.hamiltonian_vf(f)      # X_f
```

To prove the two views name the *same* bracket, the symplectic
identity `ι_{X_f} ω + df = 0` is enough,
`prove_hamiltonian_equivalence` closes it step-by-step using the
`MusicalCompatibility` axiom:

```python
from jacopy.display import chain_to_ascii

chain = M.prove_hamiltonian_equivalence(f, registry=reg)
print(chain_to_ascii(chain))
```

The 5-step chain follows this path: `ι_X ω` rewrites to
`ω^♭(X)` via the musical equality → substitution
`X_f = −π^♯(df)` → the leading `−` is pulled out →
`ω^♭ ∘ π^♯ = id` compatibility frees `df` → simplify reduces
the remaining `−df + df` to zero.

### View 3, the Koszul three-term formula

At the form level the same bracket acts on 1-forms;
`{α, β}_π = L_{π^♯(α)} β − L_{π^♯(β)} α − d⟨π^♯(α), β⟩`:

```python
alpha, beta = Forms("α β", degree=1, registry=reg)
poisson.koszul_expand(alpha, beta, reg)
# (L_π♯(α)(β) + (-L_π♯(β)(α)) + (-d(⟨π♯(α), β⟩)))
```

The classical Koszul bracket (`KoszulBracket(Sharp(π))`) and the
derived bracket are *structurally* equal on this 1-form input,
`prove_koszul_equivalence` discharges that with a single
reflexive step:

```python
chain = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)
len(chain)                           # 1
chain.steps[0].rule                  # 'reflexive'
```

## `[π, π]_SN = 0`, the single condition

The Derived Bracket Theorem states that Jacobi for `{·, ·}_π`
reduces to a single condition: `[π, π]_SN = 0`. `PoissonBracket`
exposes that obstruction as a raw `Expr`, as a
`VanishingCondition`, and as a reduction chain on `(f, g, h)`.

```python
poisson.jacobi_obstruction(reg)   # [·,·]_SN(π, π)
poisson.jacobi_condition(reg)     # VanishingCondition(..., name='Poisson Jacobi condition on {·,·}_π')
```

The three-input reduction chain has a single step:

```python
chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
len(chain)                        # 1
chain.steps[0].rule               # 'DerivedBracketTheorem'
chain.steps[0].after              # [·,·]_SN(π, π)
```

The chain does *not* discharge the obstruction, for atomic `π`
the `[π, π]_SN` handle stays opaque. Once the Poisson hypothesis
kicks in (i.e. the user declares `π` to be a Poisson bivector)
the same step closes the Jacobi identity.

## Theorem Book, seeded theorem

The library carries this reduction as a ready `Theorem` entry
under `theorem_book.get("poisson_jacobi")`, `prove_jacobi_reduction`
doesn't rebuild the chain on every call; downstream code can wire
the result via a single citation.

```python
from jacopy.library import theorem_book

thm = theorem_book.get("poisson_jacobi")
thm.statement     # "{f, g, h}_π cyclic sum = 0 when [π, π]_SN = 0"
thm.from_axioms   # ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')
```

`thm.proof` carries the theorem's canonical `ProofChain`, i.e.
the one-step reduction we just saw.

## Next step

The Lie algebroid framework applies the same derivation strategy
to a bracket living on a vector bundle (rather than on a manifold
directly). The anchor compatibility axiom, the algebroid Cartan
bundle, and how this structure is registered with `theorem_book`
are covered in [04_lie_algebroid.md](04_lie_algebroid.md).
