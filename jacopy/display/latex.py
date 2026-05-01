r"""
LaTeX renderer for :class:`~jacopy.core.expr.Expr` trees and
:class:`~jacopy.proof.step.ProofStep` / :class:`~jacopy.proof.chain.ProofChain`.

The output is raw LaTeX math, no ``$…$`` delimiters, so the caller
chooses the surrounding environment. :func:`chain_to_latex` wraps a
:class:`ProofChain` in an ``align*`` body; individual expressions come
out as atomic math snippets ready to splice into equations, tables, or
``\text{…}`` arguments.

Name sanitising handles the Unicode glyphs that appear naturally in
the package (``ι_X``, ``ω``, ``α``, ``♭``, ``Θ``) by translating them
to standard LaTeX commands (``\iota_X``, ``\omega``, ``\alpha``,
``\flat``, ``\Theta``). Multi-character subscripts are automatically
braced, ``X_ab`` becomes ``X_{ab}``, so the sanitiser's output is
pdfLaTeX-safe without the caller having to pre-format names.

Dispatch is MRO-based: the most specific registered class for
``type(expr)`` wins, which lets :class:`Derivation` subclasses fall
through to the generic handler without each subclass registering
independently.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Type

from jacopy.algebra.commutator import Commutator
from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.brackets.dorfman import SectionPair
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.multi_eval import MultiEval
from jacopy.core.wedge import Wedge
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Power,
    Product,
    Rational,
    Sum,
    Symbol,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


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
    # arrows / ellipsis / Unicode minus that surface in rule names &
    # justifications. pdfLaTeX's default OT1 font has no glyph for
    # U+2212, U+2192, U+2026, etc.; map them to standard math commands.
    "−": "-",          # U+2212 minus sign → ASCII hyphen
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "↦": r"\mapsto",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "…": r"\dots",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "±": r"\pm", "×": r"\times", "÷": r"\div",
}


_MULTICHAR_SUB = re.compile(r"_(\w{2,})")


_COMBINING_TILDE_CHAR = "̃"
_COMBINING_DOT_CHAR = "̇"
_MATH_COMBINING_TILDE_RE = re.compile(rf"(.)({_COMBINING_TILDE_CHAR})")
_MATH_COMBINING_DOT_RE = re.compile(rf"(.)({_COMBINING_DOT_CHAR})")


def _brace_nested_subscripts(name: str) -> str:
    """Wrap nested subscripts so pdfLaTeX doesn't see ``Base_a_b``.

    Cartan-remainder / Hamiltonian names like ``K_X_ι_U(μ)`` carry an
    inner operator subscript inside an outer subscript slot. Without
    bracing, pdfLaTeX reports "Double subscript". When two or more
    ``_`` appear at depth 0 (outside any ``{}`` group), wrap everything
    after the first in braces and recurse on the wrapped content so
    every layer of nesting gets explicit grouping:
    ``K_X_ι_U(μ)`` → ``K_{X_{ι_U(μ)}}``.
    """
    depth = 0
    positions: list[int] = []
    for i, c in enumerate(name):
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        elif c == "_" and depth == 0:
            positions.append(i)
    if len(positions) < 2:
        return name
    first = positions[0]
    inner = _brace_nested_subscripts(name[first + 1 :])
    return name[: first + 1] + "{" + inner + "}"


def latex_name(name: str) -> str:
    """Translate a Derivation/Symbol name into a LaTeX math snippet.

    Collapses combining diacritics (``K̃`` → ``\\tilde{K}``), braces
    nested subscripts so ``X_ι_U(η)`` renders as ``X_{ι_U(η)}`` (avoids
    pdfLaTeX's "Double subscript" error), replaces Unicode mathematical
    glyphs with their standard LaTeX commands, then braces multi-
    character subscripts so ``X_ab`` renders as ``X_{ab}``. Single-
    character subscripts (``X_f``) are left alone, LaTeX handles them
    without braces.
    """
    if not isinstance(name, str):
        raise TypeError("latex_name: expected a str")
    name = _MATH_COMBINING_TILDE_RE.sub(
        lambda m: "\\tilde{" + m.group(1) + "}", name
    )
    name = _MATH_COMBINING_DOT_RE.sub(
        lambda m: "\\dot{" + m.group(1) + "}", name
    )
    name = _brace_nested_subscripts(name)
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
_P_WEDGE = 55
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


@_register(Wedge)
def _wedge(expr: Wedge, ctx: int) -> str:
    parts = [to_latex(c, _P_WEDGE + 1) for c in expr.children]
    text = " \\wedge ".join(parts)
    return _wrap(text, _P_WEDGE, ctx)


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


@_register(MultiEval)
def _multi_eval(expr: MultiEval, ctx: int) -> str:
    head = to_latex(expr.head, _P_CALL + 1)
    arglist = ",\\, ".join(to_latex(a, 0) for a in expr.args)
    text = f"{head}\\!\\left({arglist}\\right)"
    return _wrap(text, _P_CALL, ctx)


# --------------------------------------------------------------------- #
# Tilde-calculus operators                                              #
# --------------------------------------------------------------------- #


@_register(TildeInteriorProduct)
def _tilde_iota(expr: TildeInteriorProduct, _ctx: int) -> str:
    form = to_latex(expr.form, 0)
    return f"\\tilde{{\\iota}}_{{{form}}}"


@_register(TildeExteriorDerivative)
def _tilde_d(expr: TildeExteriorDerivative, _ctx: int) -> str:
    return r"\tilde{d}"


@_register(TildeLieDerivative)
def _tilde_lie(expr: TildeLieDerivative, _ctx: int) -> str:
    form = to_latex(expr.form, 0)
    return f"\\tilde{{\\mathcal{{L}}}}_{{{form}}}"


# --------------------------------------------------------------------- #
# Proof transcript                                                      #
# --------------------------------------------------------------------- #


_COMBINING_TILDE = "̃"
_COMBINING_DOT_ABOVE = "̇"
_COMBINING_TILDE_RE = re.compile(rf"(.)({_COMBINING_TILDE})")
_COMBINING_DOT_RE = re.compile(rf"(.)({_COMBINING_DOT_ABOVE})")


def _escape_text(text: str) -> str:
    """Escape special LaTeX chars and lift Unicode math glyphs into math mode.

    ``\\text{…}`` arguments sit in textmode, but rule/justification
    strings routinely carry Unicode math glyphs (``ι``, ``ω``, ``∘``)
    copied from operator names. pdfLaTeX with the default input encoding
    chokes on them. Wrapping each translated glyph in ``\\ensuremath{…}``
    flips to math mode locally, the surrounding text stays in textmode,
    and the output is UTF-8-free for the LaTeX kernel.

    Order matters: escape the LaTeX-special ASCII chars first (so
    underscores / ampersands from the rule name don't derail
    ``align*``), then collapse combining diacritics (``L̃`` →
    ``\\tilde{L}``), and finally translate the standalone Unicode
    glyphs. The backslashes introduced by the diacritic and Unicode
    passes are post-escape, so they are emitted verbatim.
    """
    text = (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("$", r"\$")
    )
    text = _COMBINING_TILDE_RE.sub(
        lambda m: f"\\ensuremath{{\\tilde{{{m.group(1)}}}}}", text
    )
    text = _COMBINING_DOT_RE.sub(
        lambda m: f"\\ensuremath{{\\dot{{{m.group(1)}}}}}", text
    )
    for glyph, cmd in _UNICODE_TO_LATEX.items():
        if glyph in text:
            text = text.replace(glyph, f"\\ensuremath{{{cmd}}}")
    return text


def step_to_latex(step: ProofStep) -> str:
    r"""Render a single :class:`ProofStep` as a one-line math snippet.

    Produces ``before \to after \quad \text{[rule]}``. No ``&``
    alignment markers are emitted: chain rendering uses ``gather*``
    rather than ``align*`` because ``align*``'s shared alignment column
    is set by the widest row in the chain and would push every row off
    the right margin once an intermediate step's expression balloons.
    The rule / justification are passed through :func:`_escape_text` so
    stray underscores or ampersands don't break the surrounding
    environment. The justification is omitted when it merely duplicates
    the rule name (engine-emitted axioms often set the justification to
    ``"apply axiom: <rule>"``, which would double the width of every
    row in the rendered output).
    """
    if not isinstance(step, ProofStep):
        raise TypeError("step_to_latex: expected a ProofStep")
    before = to_latex(step.before)
    after = to_latex(step.after)
    rule = _escape_text(step.rule)
    tag = f"\\,({step.provenance_tag})" if step.provenance_tag else ""
    annotation = f"\\text{{[{rule}]{tag}}}"
    just = step.justification or ""
    just_strip = just.removeprefix("apply axiom: ").removeprefix("apply ")
    if just and just_strip != step.rule:
        annotation = (
            annotation + f"\\;\\text{{--- {_escape_text(just)}}}"
        )
    return f"{before} \\to {after} \\quad {annotation}"


def chain_to_latex(chain: ProofChain) -> str:
    r"""Render a :class:`ProofChain` as an ``\begin{align*}…\end{align*}`` block.

    Nested sub-proofs are not expanded inline, strategies that want a
    rich tree rendering should iterate the steps themselves and compose
    the output. This keeps the ``align*`` body flat and copy-pasteable
    into a paper or notes document.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_latex: expected a ProofChain")
    if len(chain) == 0:
        return "\\begin{gather*}\n\\text{(empty proof chain)}\n\\end{gather*}"
    body = " \\\\\n".join(step_to_latex(s) for s in chain.steps)
    # ``gather*`` (not ``align*``) so each row is independent: long
    # intermediate-step expressions don't push every other row off the
    # right margin via a shared alignment column. ``\allowdisplaybreaks``
    # lets the chain page-break instead of LaTeX cramming 100+ rows
    # into a single vbox (which then silently overflows). ``\scriptsize``
    # shrinks the rule-annotation text. The braces scope both
    # directives. MathJax ignores the size/break hints and renders the
    # same ``gather*`` block in-notebook.
    return (
        "{\\allowdisplaybreaks\\scriptsize\n"
        f"\\begin{{gather*}}\n{body}\n\\end{{gather*}}\n"
        "}"
    )


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

    The result is a pdfLaTeX-ready ``article`` document, the caller can
    write it to disk and run ``pdflatex file.tex`` without further
    massaging. ``preamble_extras`` is spliced between the default
    ``amsmath``/``amssymb`` block and ``\begin{document}`` so
    projects with their own macros can inject them verbatim.

    ``title`` / ``author``, when non-empty, trigger a ``\title``/
    ``\author`` / ``\maketitle`` block. Empty strings are treated as
    "no title", the chain just renders on a blank page.
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
    :func:`to_latex`, math-mode safe already. Rule labels are text,
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

    The output is a ``tikzpicture`` environment, paste it into any
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
    # prior node's expression, which we already emitted, so only emit
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
