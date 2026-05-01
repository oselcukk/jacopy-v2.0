# 11, Publication-ready output

`jacopy` keeps proofs as data: a `ProofChain` is a list of
`ProofStep`s, each a triple `(before, after, rule)`. That structure is
useful in a notebook, you can iterate, inspect, render, but the
moment you want to *paste a proof into a paper*, you need LaTeX. This
tutorial covers the four publication helpers:

| Function | Output | Use case |
|---|---|---|
| `chain_to_latex` | `gather*` block | Splice into an existing `.tex` source |
| `chain_to_latex_document` | full `\documentclass{article}` | One-shot pdflatex compile |
| `chain_to_tikz` | `tikzpicture` block | Splice diagram into existing slides/notes |
| `chain_to_tikz_document` | full document with `tikz` preamble | One-shot pdflatex compile |

The two `_document` variants are **not** wrappers around the inline
ones, they assemble a complete preamble (`amsmath`, `amssymb`,
`tikz`, `positioning`) so the output compiles standalone. Use them
when you want a quick PDF; use the inline variants when you're
splicing into an existing source.

## A worked chain

For demonstration we use a tiny but real proof: `d(d(ω)) == 0`. The
default engine knows `d² = 0` as an axiom, so the chain is two steps,
just enough to exercise both renderers without flooding the output.

```python
from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import d
from jacopy.core.expr import Symbol, Integer
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import prove_equivalence

reg = PropertyRegistry()
omega = Symbol("ω"); reg.declare(omega, Graded(degree=2))

chain = prove_equivalence(
    Act(d, Act(d, omega)),
    Integer(0),
    registry=reg,
)
print(f"steps: {len(chain)}")
for s in chain.steps:
    print(f"  [{s.rule}] {s.before} → {s.after}")
```

## Inline LaTeX: `chain_to_latex`

`chain_to_latex(chain)` returns a `gather*` block, one math row per
step. The result wraps the body in `\allowdisplaybreaks\scriptsize`
so long chains page-break properly and rule annotations don't crowd
out the expressions.

```python
from jacopy.display import chain_to_latex

body = chain_to_latex(chain)
print(body)
```

Each row uses `\to` (not `=`) to make the rewriting direction explicit;
the `\quad \text{[rule] (provenance)}` annotation reads as the
justification of the row. The block is copy-pasteable into any LaTeX
source that already loads `amsmath`.

## Standalone document: `chain_to_latex_document`

For a one-shot PDF, wrap the chain in a full document via
`chain_to_latex_document`. The function assembles the
`\documentclass{article}` preamble, splices a `\title` / `\author` /
`\maketitle` block when those kwargs are non-empty, and ends with
`\end{document}`, `pdflatex output.tex` compiles directly.

```python
from jacopy.display import chain_to_latex_document

doc = chain_to_latex_document(
    chain,
    title="d² = 0",
    author="jacopy",
)
# Write it out and compile with `pdflatex out.tex`.
print(doc[:400])
```

`preamble_extras=` lets you splice your project's macros between the
default `amsmath`/`amssymb` block and `\begin{document}`, for
instance, redefining `\renewcommand{\arraystretch}{1.3}` or pulling
in your group's `\input{macros.tex}`.

## TikZ diagrams: `chain_to_tikz`

The same chain renders as a vertical TikZ diagram via `chain_to_tikz`.
Each `before` / `after` becomes a boxed node (`n + 1` nodes for `n`
steps), and arrows are labelled with the rule name plus the
provenance tag in parentheses when present.

```python
from jacopy.display import chain_to_tikz

diagram = chain_to_tikz(chain)
print(diagram)
```

Use this when prose isn't enough, when you want a reader to *see*
the rewriting tree at a glance. For deeply nested chains the diagram
flattens to its top-level steps; nested sub-proofs aren't exploded
inline (deliberately, the `tikzpicture` would explode in size).

## Standalone TikZ document

`chain_to_tikz_document` wraps the diagram in a full document with the
`tikz` and `positioning` preamble. `pdflatex` compiles it directly:

```python
from jacopy.display import chain_to_tikz_document

doc = chain_to_tikz_document(
    chain,
    title="d² = 0 (diagram)",
    node_distance="1.4cm",
)
# Write to disk, then `pdflatex out.tex`.
print(doc[:300])
```

`node_distance` controls the vertical spacing, bump it up when your
expression labels are long enough to crowd each other.

## Round-trip to PDF

The pattern is the same for both `_document` variants: write the
string to disk, run `pdflatex`. `subprocess.run` from a notebook
exercises the full round-trip:

```python
# (Skip executing this cell unless pdflatex is on your PATH.)
import subprocess, tempfile
from pathlib import Path
from jacopy.display import chain_to_latex_document

doc = chain_to_latex_document(chain, title="d² = 0")
with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "proof.tex"
    src.write_text(doc, encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", str(src)],
                   cwd=tmp, check=True)
    print(sorted(Path(tmp).iterdir()))
```

If `pdflatex` isn't installed, the call raises `FileNotFoundError`,
that's purely an environmental issue, not something `jacopy` can
fix. The `.tex` content stands on its own; any LaTeX compiler
(`pdflatex`, `lualatex`, `xelatex`) handles it.

## Choosing between the four

| Goal | Use |
|---|---|
| Inline equations in a paper | `chain_to_latex` |
| Standalone PDF of a single proof | `chain_to_latex_document` |
| Diagram in an existing slide deck | `chain_to_tikz` |
| Standalone PDF showing the rewriting tree | `chain_to_tikz_document` |

In a notebook the LaTeX block also renders inline if you wrap it in
`Markdown` / `IPython.display.Math`, but for that workflow the richer
`display_chain` (`jacopy.display.display_chain`) is the better entry
point, it knows about `_repr_latex_` and renders without a manual
`Markdown(...)` wrap.

## Summary

* Four helpers; two inline (`chain_to_latex`, `chain_to_tikz`), two
  standalone (`chain_to_latex_document`, `chain_to_tikz_document`).
* Inline: paste into existing `.tex`. Standalone: write + `pdflatex`.
* Title / author / preamble extras are kwargs on the `_document`
  variants, empty strings produce a body-only document.
* Nested sub-proofs flatten in both renderers, pick the inline
  helpers and compose by hand if you want a tree.
