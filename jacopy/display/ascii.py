"""
Plain-text renderer for :class:`~jacopy.core.expr.Expr` trees and
:class:`~jacopy.proof.step.ProofStep` / :class:`~jacopy.proof.chain.ProofChain`.

The goal is a mathematically legible ASCII form, closer to how the
objects are written in prose than the raw ``__repr__`` output. Sign
normalisation folds ``Sum(a, Neg(b), c)`` into ``a - b + c`` instead of
``(a + (-b) + c)``, and precedence-aware parentheses keep nested
compositions readable without over-parenthesising atoms.

Dispatch is MRO-based: the handler registered for the most specific
superclass of ``type(expr)`` wins. That way a :class:`Derivation`
subclass like :class:`~jacopy.calculus.exterior_d.ExteriorDerivative`
falls through to the generic :class:`Derivation` handler without every
subclass having to register one.

The ``to_ascii`` entry point treats the Unicode glyphs that appear in
operator names (``ι``, ``ω``, ``♭``, ``⟨⟩``) as plain text, the
terminal typically prints them fine, and anything that cannot handle
Unicode also cannot handle the rest of the mathematical symbols the
package uses. The LaTeX renderer ships a dedicated sanitiser for the
``\\iota``-style translation.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

from jacopy.algebra.commutator import Commutator
from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.brackets.dorfman import SectionPair
from jacopy.calculus.pairing import Pairing
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


# Precedence rungs. Higher binds tighter; a child whose precedence is
# strictly below the context's gets parenthesised.
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


def to_ascii(expr: Expr, ctx_precedence: int = 0) -> str:
    """Render ``expr`` as plain text.

    ``ctx_precedence`` is the binding tightness of the surrounding
    context; the rendered string is parenthesised if its own precedence
    falls below it. Callers normally leave this at ``0``.
    """
    if not isinstance(expr, Expr):
        raise TypeError("to_ascii: expected an Expr")
    for cls in type(expr).__mro__:
        h = _HANDLERS.get(cls)
        if h is not None:
            return h(expr, ctx_precedence)
    return repr(expr)


def _wrap(text: str, own_prec: int, ctx_prec: int) -> str:
    if own_prec < ctx_prec:
        return f"({text})"
    return text


# --------------------------------------------------------------------- #
# Core expression types                                                 #
# --------------------------------------------------------------------- #


@_register(Symbol)
def _sym(expr: Symbol, _ctx: int) -> str:
    return expr.name


@_register(Integer)
def _int(expr: Integer, ctx: int) -> str:
    v = expr.value
    if v < 0:
        return _wrap(str(v), _P_NEG, ctx)
    return str(v)


@_register(Rational)
def _rat(expr: Rational, ctx: int) -> str:
    text = f"{expr.p}/{expr.q}"
    # Rationals with negative numerators read as negated quotients;
    # treat the whole thing as an atom for bracketing purposes.
    return _wrap(text, _P_ATOM, ctx) if expr.p >= 0 else _wrap(text, _P_NEG, ctx)


@_register(Neg)
def _neg(expr: Neg, ctx: int) -> str:
    inner = to_ascii(expr.arg, _P_NEG + 1)
    text = f"-{inner}"
    return _wrap(text, _P_NEG, ctx)


@_register(Sum)
def _sum(expr: Sum, ctx: int) -> str:
    parts: list[str] = []
    for i, child in enumerate(expr.children):
        if isinstance(child, Neg):
            inner = to_ascii(child.arg, _P_NEG + 1)
            parts.append(("- " if i > 0 else "-") + inner)
        else:
            rendered = to_ascii(child, _P_SUM + 1)
            parts.append(("+ " if i > 0 else "") + rendered)
    text = " ".join(parts) if len(parts) > 1 else parts[0] if parts else "0"
    return _wrap(text, _P_SUM, ctx)


@_register(Product)
def _prod(expr: Product, ctx: int) -> str:
    parts = [to_ascii(c, _P_PRODUCT + 1) for c in expr.children]
    text = " * ".join(parts) if parts else "1"
    return _wrap(text, _P_PRODUCT, ctx)


@_register(Wedge)
def _wedge(expr: Wedge, ctx: int) -> str:
    parts = [to_ascii(c, _P_WEDGE + 1) for c in expr.children]
    text = " ∧ ".join(parts)
    return _wrap(text, _P_WEDGE, ctx)


@_register(Power)
def _pow(expr: Power, ctx: int) -> str:
    base = to_ascii(expr.base, _P_POWER + 1)
    exp = to_ascii(expr.exp, _P_POWER + 1)
    text = f"{base}**{exp}"
    return _wrap(text, _P_POWER, ctx)


# --------------------------------------------------------------------- #
# Algebra                                                               #
# --------------------------------------------------------------------- #


@_register(Derivation)
def _deriv(expr: Derivation, _ctx: int) -> str:
    return expr.name


@_register(Act)
def _act(expr: Act, ctx: int) -> str:
    # Render composite operators (sums of Ds, products, etc.) inside the
    # left position with parens so "A+B applied to x" reads "(A + B)(x)".
    op = to_ascii(expr.op, _P_CALL + 1)
    arg = to_ascii(expr.arg, 0)
    text = f"{op}({arg})"
    return _wrap(text, _P_CALL, ctx)


@_register(Commutator)
def _comm(expr: Commutator, _ctx: int) -> str:
    a = to_ascii(expr.a, 0)
    b = to_ascii(expr.b, 0)
    return f"[{a}, {b}]"


# --------------------------------------------------------------------- #
# Bracket / section / pairing nodes                                     #
# --------------------------------------------------------------------- #


@_register(BracketApply)
def _bracket_apply(expr: BracketApply, _ctx: int) -> str:
    a = to_ascii(expr.a, 0)
    b = to_ascii(expr.b, 0)
    return f"[{a}, {b}]_{{{expr.bracket.name}}}"


@_register(SectionPair)
def _section(expr: SectionPair, _ctx: int) -> str:
    v = to_ascii(expr.vector, 0)
    f = to_ascii(expr.form, 0)
    return f"({v}, {f})"


@_register(Pairing)
def _pairing(expr: Pairing, _ctx: int) -> str:
    a = to_ascii(expr.alpha, 0)
    X = to_ascii(expr.X, 0)
    return f"<{a}, {X}>"


@_register(MultiEval)
def _multi_eval(expr: MultiEval, ctx: int) -> str:
    head = to_ascii(expr.head, _P_CALL + 1)
    arglist = ", ".join(to_ascii(a, 0) for a in expr.args)
    text = f"{head}({arglist})"
    return _wrap(text, _P_CALL, ctx)


# --------------------------------------------------------------------- #
# Proof transcript                                                      #
# --------------------------------------------------------------------- #


#: Recognised verbosity levels for proof-transcript rendering.
#:
#: * ``"full"``  , rule + tag + ``before → after`` + justification + children.
#: * ``"summary"``, rule + tag + ``before → after`` + children (no justification).
#: * ``"compact"``, rule + tag only (flat list; no before/after, no children).
VERBOSITY_MODES = ("full", "summary", "compact")


def _check_verbosity(verbosity: str) -> None:
    if verbosity not in VERBOSITY_MODES:
        raise ValueError(
            f"verbosity must be one of {VERBOSITY_MODES}, got {verbosity!r}"
        )


def step_to_ascii(
    step: ProofStep,
    indent: int = 0,
    max_depth: int = 64,
    *,
    verbosity: str = "full",
) -> str:
    """Render a single :class:`ProofStep` including nested children.

    The format mirrors :meth:`ProofStep.format` but routes the ``before``
    / ``after`` expressions through :func:`to_ascii` so sign
    normalisation and precedence-aware parens kick in.

    ``verbosity`` selects how much of each step is shown, see
    :data:`VERBOSITY_MODES`.
    """
    if not isinstance(step, ProofStep):
        raise TypeError("step_to_ascii: expected a ProofStep")
    _check_verbosity(verbosity)
    pad = "  " * indent
    tag = f" ({step.provenance_tag})" if step.provenance_tag else ""
    if verbosity == "compact":
        # Rule + tag only, a one-line table-of-contents entry.
        return f"{pad}[{step.rule}]{tag}"
    before = to_ascii(step.before)
    after = to_ascii(step.after)
    head = f"{pad}[{step.rule}]{tag} {before} -> {after}"
    if verbosity == "full" and step.justification:
        head = f"{head}  -- {step.justification}"
    lines = [head]
    if max_depth > 0 and step.children:
        for ch in step.children:
            lines.append(
                step_to_ascii(
                    ch,
                    indent + 1,
                    max_depth - 1,
                    verbosity=verbosity,
                )
            )
    return "\n".join(lines)


def chain_to_ascii(
    chain: ProofChain,
    max_depth: int = 64,
    *,
    verbosity: str = "full",
) -> str:
    """Render an entire :class:`ProofChain` as an ordered step list.

    In ``"compact"`` mode children are suppressed, so the result reads
    as a flat rule-sequence regardless of nesting.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_ascii: expected a ProofChain")
    _check_verbosity(verbosity)
    if len(chain) == 0:
        return "(empty proof chain)"
    return "\n".join(
        step_to_ascii(s, indent=0, max_depth=max_depth, verbosity=verbosity)
        for s in chain.steps
    )
