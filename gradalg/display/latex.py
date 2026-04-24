r"""
LaTeX renderer for :class:`~gradalg.core.expr.Expr` trees and
:class:`~gradalg.proof.step.ProofStep` / :class:`~gradalg.proof.chain.ProofChain`.

The output is raw LaTeX math — no ``$…$`` delimiters — so the caller
chooses the surrounding environment. :func:`chain_to_latex` wraps a
:class:`ProofChain` in an ``align*`` body; individual expressions come
out as atomic math snippets ready to splice into equations, tables, or
``\text{…}`` arguments.

Name sanitising handles the Unicode glyphs that appear naturally in
the package (``ι_X``, ``ω``, ``α``, ``♭``, ``Θ``) by translating them
to standard LaTeX commands (``\iota_X``, ``\omega``, ``\alpha``,
``\flat``, ``\Theta``). Multi-character subscripts are automatically
braced — ``X_ab`` becomes ``X_{ab}`` — so the sanitiser's output is
pdfLaTeX-safe without the caller having to pre-format names.

Dispatch is MRO-based: the most specific registered class for
``type(expr)`` wins, which lets :class:`Derivation` subclasses fall
through to the generic handler without each subclass registering
independently.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Type

from gradalg.algebra.commutator import Commutator
from gradalg.algebra.derivation import Act, Derivation
from gradalg.brackets.base import BracketApply
from gradalg.brackets.dorfman import SectionPair
from gradalg.calculus.pairing import Pairing
from gradalg.core.expr import (
    Expr,
    Integer,
    Neg,
    Power,
    Product,
    Rational,
    Sum,
    Symbol,
)
from gradalg.proof.chain import ProofChain
from gradalg.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Name sanitising                                                        #
# --------------------------------------------------------------------- #


_UNICODE_TO_LATEX: Dict[str, str] = {
    # lowercase Greek
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # uppercase Greek
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    # musical / algebraic
    "♭": r"\flat", "♯": r"\sharp",
    "∧": r"\wedge", "∨": r"\vee",
    "∘": r"\circ", "⊕": r"\oplus", "⊗": r"\otimes",
    "·": r"\cdot",
    "⟨": r"\langle", "⟩": r"\rangle",
    "∞": r"\infty",
    # hat / bars / primes occasionally show up in names
    "∂": r"\partial",
}


_MULTICHAR_SUB = re.compile(r"_(\w{2,})")


def latex_name(name: str) -> str:
    """Translate a Derivation/Symbol name into a LaTeX math snippet.

    Replaces Unicode mathematical glyphs with their standard LaTeX
    commands, then braces multi-character subscripts so ``X_ab``
    renders as ``X_{ab}``. Single-character subscripts (``X_f``) are
    left alone — LaTeX handles them without braces.
    """
    if not isinstance(name, str):
        raise TypeError("latex_name: expected a str")
    for glyph, replacement in _UNICODE_TO_LATEX.items():
        name = name.replace(glyph, replacement)
    name = _MULTICHAR_SUB.sub(lambda m: "_{" + m.group(1) + "}", name)
    return name


# --------------------------------------------------------------------- #
# Dispatch                                                              #
# --------------------------------------------------------------------- #


# Same precedence rungs as the ASCII renderer; a child renders with
# parens when its own precedence is below the surrounding context's.
_P_ATOM = 100
_P_CALL = 90
_P_POWER = 80
_P_PRODUCT = 60
_P_NEG = 50
_P_SUM = 40


Handler = Callable[[Expr, int], str]
_HANDLERS: Dict[Type[Expr], Handler] = {}


def _register(cls: Type[Expr]):
    def decorator(fn: Handler) -> Handler:
        _HANDLERS[cls] = fn
        return fn

    return decorator


def to_latex(expr: Expr, ctx_precedence: int = 0) -> str:
    """Render ``expr`` as a LaTeX math snippet (no ``$`` delimiters)."""
    if not isinstance(expr, Expr):
        raise TypeError("to_latex: expected an Expr")
    for cls in type(expr).__mro__:
        h = _HANDLERS.get(cls)
        if h is not None:
            return h(expr, ctx_precedence)
    # Last-resort fallback: sanitise the ``__repr__`` output so at least
    # Greek glyphs still come through as LaTeX commands.
    return latex_name(repr(expr))


def _wrap(text: str, own_prec: int, ctx_prec: int) -> str:
    if own_prec < ctx_prec:
        return f"\\left({text}\\right)"
    return text


# --------------------------------------------------------------------- #
# Core expression types                                                 #
# --------------------------------------------------------------------- #


@_register(Symbol)
def _sym(expr: Symbol, _ctx: int) -> str:
    return latex_name(expr.name)


@_register(Integer)
def _int(expr: Integer, ctx: int) -> str:
    v = expr.value
    if v < 0:
        return _wrap(str(v), _P_NEG, ctx)
    return str(v)


@_register(Rational)
def _rat(expr: Rational, ctx: int) -> str:
    p, q = expr.p, expr.q
    if p < 0:
        text = f"-\\frac{{{-p}}}{{{q}}}"
        return _wrap(text, _P_NEG, ctx)
    return f"\\frac{{{p}}}{{{q}}}"


@_register(Neg)
def _neg(expr: Neg, ctx: int) -> str:
    inner = to_latex(expr.arg, _P_NEG + 1)
    return _wrap(f"-{inner}", _P_NEG, ctx)


@_register(Sum)
def _sum(expr: Sum, ctx: int) -> str:
    parts: list[str] = []
    for i, child in enumerate(expr.children):
        if isinstance(child, Neg):
            inner = to_latex(child.arg, _P_NEG + 1)
            parts.append(("- " if i > 0 else "-") + inner)
        else:
            rendered = to_latex(child, _P_SUM + 1)
            parts.append(("+ " if i > 0 else "") + rendered)
    text = " ".join(parts) if len(parts) > 1 else parts[0] if parts else "0"
    return _wrap(text, _P_SUM, ctx)


@_register(Product)
def _prod(expr: Product, ctx: int) -> str:
    parts = [to_latex(c, _P_PRODUCT + 1) for c in expr.children]
    text = " \\, ".join(parts) if parts else "1"
    return _wrap(text, _P_PRODUCT, ctx)


@_register(Power)
def _pow(expr: Power, ctx: int) -> str:
    base = to_latex(expr.base, _P_POWER + 1)
    exp = to_latex(expr.exp, 0)
    return _wrap(f"{{{base}}}^{{{exp}}}", _P_POWER, ctx)


# --------------------------------------------------------------------- #
# Algebra                                                               #
# --------------------------------------------------------------------- #


@_register(Derivation)
def _deriv(expr: Derivation, _ctx: int) -> str:
    return latex_name(expr.name)


@_register(Act)
def _act(expr: Act, ctx: int) -> str:
    op = to_latex(expr.op, _P_CALL + 1)
    arg = to_latex(expr.arg, 0)
    return _wrap(f"{op}\\!\\left({arg}\\right)", _P_CALL, ctx)


@_register(Commutator)
def _comm(expr: Commutator, _ctx: int) -> str:
    a = to_latex(expr.a, 0)
    b = to_latex(expr.b, 0)
    return f"\\left[{a},\\, {b}\\right]"


# --------------------------------------------------------------------- #
# Bracket / section / pairing nodes                                     #
# --------------------------------------------------------------------- #


@_register(BracketApply)
def _bracket_apply(expr: BracketApply, _ctx: int) -> str:
    a = to_latex(expr.a, 0)
    b = to_latex(expr.b, 0)
    tag = latex_name(expr.bracket.name)
    return f"\\left[{a},\\, {b}\\right]_{{{tag}}}"


@_register(SectionPair)
def _section(expr: SectionPair, _ctx: int) -> str:
    v = to_latex(expr.vector, 0)
    f = to_latex(expr.form, 0)
    return f"\\left({v},\\, {f}\\right)"


@_register(Pairing)
def _pairing(expr: Pairing, _ctx: int) -> str:
    a = to_latex(expr.alpha, 0)
    X = to_latex(expr.X, 0)
    return f"\\langle {a},\\, {X} \\rangle"


# --------------------------------------------------------------------- #
# Proof transcript                                                      #
# --------------------------------------------------------------------- #


def _escape_text(text: str) -> str:
    """Escape special LaTeX chars and lift Unicode math glyphs into math mode.

    ``\\text{…}`` arguments sit in textmode, but rule/justification
    strings routinely carry Unicode math glyphs (``ι``, ``ω``, ``∘``)
    copied from operator names. pdfLaTeX with the default input encoding
    chokes on them. Wrapping each translated glyph in ``\\ensuremath{…}``
    flips to math mode locally — the surrounding text stays in textmode,
    and the output is UTF-8-free for the LaTeX kernel.

    Order matters: escape the LaTeX-special ASCII chars first (so
    underscores / ampersands from the rule name don't derail
    ``align*``), then translate Unicode. The backslashes introduced by
    the Unicode pass are post-escape, so they are emitted verbatim.
    """
    text = (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("$", r"\$")
    )
    for glyph, cmd in _UNICODE_TO_LATEX.items():
        if glyph in text:
            text = text.replace(glyph, f"\\ensuremath{{{cmd}}}")
    return text


def step_to_latex(step: ProofStep) -> str:
    r"""Render a single :class:`ProofStep` as an ``align*``-ready line.

    Produces ``before &\to after &&\text{[rule]}``. The rule and
    justification are passed through :func:`_escape_text` so stray
    underscores or ampersands don't derail the surrounding
    ``align*`` body.
    """
    if not isinstance(step, ProofStep):
        raise TypeError("step_to_latex: expected a ProofStep")
    before = to_latex(step.before)
    after = to_latex(step.after)
    rule = _escape_text(step.rule)
    tag = f"\\,({step.provenance_tag})" if step.provenance_tag else ""
    annotation = f"\\text{{[{rule}]{tag}}}"
    if step.justification:
        annotation = (
            annotation + f"\\;\\text{{--- {_escape_text(step.justification)}}}"
        )
    return f"{before} &\\to {after} && {annotation}"


def chain_to_latex(chain: ProofChain) -> str:
    r"""Render a :class:`ProofChain` as an ``\begin{align*}…\end{align*}`` block.

    Nested sub-proofs are not expanded inline — strategies that want a
    rich tree rendering should iterate the steps themselves and compose
    the output. This keeps the ``align*`` body flat and copy-pasteable
    into a paper or notes document.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_latex: expected a ProofChain")
    if len(chain) == 0:
        return "\\begin{align*}\n\\text{(empty proof chain)}\n\\end{align*}"
    body = " \\\\\n".join(step_to_latex(s) for s in chain.steps)
    return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"


# --------------------------------------------------------------------- #
# Standalone document export                                             #
# --------------------------------------------------------------------- #


_DEFAULT_PREAMBLE = (
    r"\usepackage{amsmath}" "\n"
    r"\usepackage{amssymb}" "\n"
    r"\usepackage[utf8]{inputenc}" "\n"
)


def chain_to_latex_document(
    chain: ProofChain,
    *,
    title: str = "",
    author: str = "",
    preamble_extras: str = "",
) -> str:
    r"""Wrap :func:`chain_to_latex` in a full ``\documentclass`` document.

    The result is a pdfLaTeX-ready ``article`` document — the caller can
    write it to disk and run ``pdflatex file.tex`` without further
    massaging. ``preamble_extras`` is spliced between the default
    ``amsmath``/``amssymb`` block and ``\begin{document}`` so
    projects with their own macros can inject them verbatim.

    ``title`` / ``author``, when non-empty, trigger a ``\title``/
    ``\author`` / ``\maketitle`` block. Empty strings are treated as
    "no title" — the chain just renders on a blank page.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_latex_document: expected a ProofChain")
    body = chain_to_latex(chain)
    lines = [
        r"\documentclass{article}",
        _DEFAULT_PREAMBLE.rstrip(),
    ]
    if preamble_extras:
        lines.append(preamble_extras.rstrip())
    if title:
        lines.append(f"\\title{{{_escape_text(title)}}}")
    if author:
        lines.append(f"\\author{{{_escape_text(author)}}}")
    lines.append(r"\begin{document}")
    if title or author:
        lines.append(r"\maketitle")
    lines.append(body)
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# TikZ chain diagram                                                     #
# --------------------------------------------------------------------- #


def _tikz_escape(text: str) -> str:
    """Escape a math-mode label for embedding in a TikZ ``node`` body.

    TikZ nodes inside ``$…$`` inherit math mode, which is what we want
    for expressions. The label text comes straight from
    :func:`to_latex` — math-mode safe already. Rule labels are text,
    run through :func:`_escape_text`.
    """
    return _escape_text(text)


def chain_to_tikz(
    chain: ProofChain,
    *,
    node_distance: str = "1.2cm",
) -> str:
    r"""Render a :class:`ProofChain` as a vertical TikZ diagram.

    Each step's ``before`` / ``after`` become boxed nodes; consecutive
    ``after`` and next ``before`` coincide, so the chain produces
    ``n + 1`` nodes for ``n`` steps. Arrows are labelled with the
    rule name (provenance tag in parentheses when present).

    The output is a ``tikzpicture`` environment — paste it into any
    LaTeX document that loads the ``tikz`` package. For a standalone
    file wrapping this, see :func:`chain_to_tikz_document`.

    Nested sub-proofs flatten: this renderer only walks ``chain.steps``.
    Callers that want a tree diagram should compose multiple calls.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_tikz: expected a ProofChain")
    if len(chain) == 0:
        return (
            f"\\begin{{tikzpicture}}[node distance={node_distance}]\n"
            "\\node {(empty proof chain)};\n"
            "\\end{tikzpicture}"
        )
    lines = [f"\\begin{{tikzpicture}}[node distance={node_distance}]"]
    # Emit n+1 nodes: e0, e1, ..., en. Each step's ``before`` is the
    # prior node's expression, which we already emitted — so only emit
    # the first ``before`` plus every ``after``.
    first = chain.steps[0].before
    lines.append(f"\\node[draw, rectangle] (e0) {{${to_latex(first)}$}};")
    for i, step in enumerate(chain.steps):
        label = to_latex(step.after)
        lines.append(
            f"\\node[draw, rectangle, below=of e{i}] "
            f"(e{i + 1}) {{${label}$}};"
        )
    for i, step in enumerate(chain.steps):
        tag = f" ({step.provenance_tag})" if step.provenance_tag else ""
        rule = _tikz_escape(f"{step.rule}{tag}")
        lines.append(
            f"\\draw[->] (e{i}) -- node[right] {{\\small {rule}}} "
            f"(e{i + 1});"
        )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def chain_to_tikz_document(
    chain: ProofChain,
    *,
    title: str = "",
    author: str = "",
    node_distance: str = "1.2cm",
) -> str:
    r"""Standalone ``\documentclass`` wrapper around :func:`chain_to_tikz`.

    Loads ``tikz`` and ``positioning`` (for ``below=of`` placement) in
    the preamble so the output is directly ``pdflatex``-able.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_tikz_document: expected a ProofChain")
    body = chain_to_tikz(chain, node_distance=node_distance)
    lines = [
        r"\documentclass{article}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{positioning}",
    ]
    if title:
        lines.append(f"\\title{{{_escape_text(title)}}}")
    if author:
        lines.append(f"\\author{{{_escape_text(author)}}}")
    lines.append(r"\begin{document}")
    if title or author:
        lines.append(r"\maketitle")
    lines.append(r"\begin{center}")
    lines.append(body)
    lines.append(r"\end{center}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"
