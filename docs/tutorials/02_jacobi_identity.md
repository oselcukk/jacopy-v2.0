# 02, The Jacobi identity

This tutorial shows how the Jacobi identity for a Lie bracket
closes as a `ProofChain` in `jacopy`. Familiarity with the
[first-steps tutorial](01_first_steps.md) is assumed.

## The Lie bracket

`jacopy.brackets.lie` exposes two things:

- `LieBracket`, a `GradedBracket` subclass with an optional name.
- `lie`, the process-wide singleton for the standard ("TM") Lie
  bracket.

Tutorials use the singleton throughout. On a specific manifold or
algebroid, build your own `LieBracket(name="[·,·]_E")` instance
instead.

```python
from jacopy.brackets.lie import lie
```

## Three vector fields

Jacobi needs three symbols. The `VectorFields` helper declares
each as `Graded(degree=0)`, the `PropertyRegistry` derives the
sign rules in the Jacobi expansion from that degree.

```python
from jacopy import VectorFields
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
X, Y, Z = VectorFields("X Y Z", registry=reg)
```

For the primitive path, see the *Declaring properties* section of
[01_first_steps.md](01_first_steps.md), the helper is just a
wrapper around `Symbol(...) + reg.declare(sym, Graded(degree=0))`.

## `prove_jacobi`

The graded Jacobi identity:

$$[X,[Y,Z]] + (-1)^{|X||Y|+|X||Z|}\,[Y,[Z,X]] + (-1)^{|Y||Z|+|X||Z|}\,[Z,[X,Y]] = 0$$

In degree 0 every sign is `+1`, recovering the standard Jacobi:

$$[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0.$$

`prove_jacobi` takes a bracket + three operands + a registry and
returns a `ProofChain`:

```python
from jacopy.proof import prove_jacobi

chain = prove_jacobi(lie, X, Y, Z, registry=reg)
assert chain.steps[-1].after  # final state, should be 0
```

The chain length depends on the bracket's internal rewrite rules.
On `lie` the chain has two steps: the first opens the Jacobi
obstruction's bracket into canonical form, the second collapses
the residue to zero.

## Display

ASCII:

```python
from jacopy.display import chain_to_ascii
print(chain_to_ascii(chain))
```

LaTeX (`align*` blocks) inside Jupyter:

```python
from jacopy.display import display_chain
display_chain(chain)
```

## Conditional Jacobi

Not every bracket satisfies Jacobi unconditionally. The
`CourantBracket`, for example, sets
`satisfies_graded_jacobi=None`: with an H-twist the obstruction
equals `dH`, without it the obstruction is `0`. `prove_jacobi`
does **not** carry that signal, instead, reach for
`bracket.jacobi_condition()` to obtain a `VanishingCondition`. A
`ProofChain` is then built through library-level helpers such as
`CourantAlgebroid.prove_jacobi_reduction`.

The next tutorial ([03_poisson_geometry.md](03_poisson_geometry.md))
shows how the Poisson bracket derives Jacobi via the derived-bracket
theorem.
