"""
Rich-terminal renderer, coloured, tree-style output for Exprs and
ProofChains.

This module leans on the optional ``rich`` package to produce ANSI
coloured, hierarchical renderings of proof transcripts. When ``rich``
is not installed, every render / print function transparently falls
back to the plain-ASCII output of :mod:`jacopy.display.ascii`. No
``ImportError`` is surfaced to the caller: the return type is always
``str`` and the functions always succeed.

The module-level flag :data:`HAS_RICH` exposes which code path is
active, so tests (and users who want to branch on availability) can
check without re-importing ``rich`` themselves.

Styling is deliberately conservative:

- ``[rule]`` names, bold cyan, since that's the primary scanning
  anchor when eyeballing a transcript.
- ``(axiom)`` / ``(theorem)`` provenance tags, yellow / green, so the
  two classification modes stand apart at a glance.
- Arrow ``→``, dim, decorative only.
- Trailing ``, justification``, dim italic, meta-information.

The tree wraps an optional header ``Proof (N steps)`` around the
chain's top-level steps, with each step's ``children`` becoming nested
tree branches. That mirrors the ASCII renderer's indentation semantics
but uses rich's box characters so the hierarchy reads at a glance.
"""

from __future__ import annotations

import io
from typing import Optional

from jacopy.core.expr import Expr
from jacopy.display.ascii import (
    VERBOSITY_MODES,
    chain_to_ascii,
    step_to_ascii,
    to_ascii,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


def _check_verbosity(verbosity: str) -> None:
    if verbosity not in VERBOSITY_MODES:
        raise ValueError(
            f"verbosity must be one of {VERBOSITY_MODES}, got {verbosity!r}"
        )

try:  # pragma: no cover - exercised only when rich is installed
    from rich.console import Console
    from rich.text import Text
    from rich.tree import Tree

    HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only when rich is missing
    HAS_RICH = False
    Console = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    Tree = None  # type: ignore[assignment]


# --------------------------------------------------------------------- #
# Styles                                                                #
# --------------------------------------------------------------------- #


_STYLE_RULE = "bold cyan"
_STYLE_TAG_AXIOM = "yellow"
_STYLE_TAG_THEOREM = "green"
_STYLE_ARROW = "dim"
_STYLE_JUSTIFY = "italic dim"
_STYLE_TITLE = "bold"


# --------------------------------------------------------------------- #
# Internal builders (only callable when HAS_RICH)                       #
# --------------------------------------------------------------------- #


def _step_head(step: ProofStep, verbosity: str = "full"):
    # pragma: no cover - rich-only path
    """Compose a rich ``Text`` for a single step's head line.

    ``"compact"`` drops the ``before → after`` fragment entirely; the
    step renders as rule + tag only.  ``"summary"`` keeps the arrow but
    omits the trailing justification.  ``"full"`` is the default.
    """
    t = Text()
    t.append(f"[{step.rule}]", style=_STYLE_RULE)
    if step.provenance_tag == "axiom":
        t.append(f" ({step.provenance_tag})", style=_STYLE_TAG_AXIOM)
    elif step.provenance_tag == "theorem":
        t.append(f" ({step.provenance_tag})", style=_STYLE_TAG_THEOREM)
    if verbosity == "compact":
        return t
    t.append(" ")
    t.append(to_ascii(step.before))
    t.append(" → ", style=_STYLE_ARROW)
    t.append(to_ascii(step.after))
    if verbosity == "full" and step.justification:
        t.append(f" , {step.justification}", style=_STYLE_JUSTIFY)
    return t


def _build_step_tree(
    step: ProofStep, max_depth: int, verbosity: str = "full"
):
    # pragma: no cover - rich-only path
    tree = Tree(_step_head(step, verbosity))
    # Compact mode is a flat table-of-contents, suppress children so
    # the output stays one-line-per-step regardless of nesting.
    if max_depth > 0 and verbosity != "compact":
        for ch in step.children:
            tree.add(_build_step_tree(ch, max_depth - 1, verbosity))
    return tree


def _build_chain_tree(
    chain: ProofChain,
    max_depth: int,
    title: bool,
    verbosity: str = "full",
):
    # pragma: no cover - rich-only path
    heading = (
        Text(f"Proof ({len(chain)} steps)", style=_STYLE_TITLE)
        if title
        else Text("")
    )
    root = Tree(heading)
    for step in chain.steps:
        root.add(_build_step_tree(step, max_depth, verbosity))
    return root


def _render_to_text(renderable, *, styles: bool = True) -> str:
    # pragma: no cover - rich-only path
    """Render a rich object through a recording Console and return the text.

    The recording Console needs a sink to write to even though only the
    captured buffer matters; pointing ``file`` at an in-memory
    ``StringIO`` keeps the render from duplicating on stdout when the
    caller later ``print()``s the returned string.
    """
    console = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=True,
        color_system="truecolor" if styles else None,
        width=120,
    )
    console.print(renderable)
    return console.export_text(styles=styles).rstrip("\n")


# --------------------------------------------------------------------- #
# Public render API                                                     #
# --------------------------------------------------------------------- #


def render_expr(expr: Expr) -> str:
    """Render an :class:`Expr` for terminal display.

    Single-expression rendering has no rule/tag structure to colour, so
    the output is the same as :func:`to_ascii` in both rich-present and
    fallback modes. The function is provided for API symmetry.
    """
    if not isinstance(expr, Expr):
        raise TypeError("render_expr: expected an Expr")
    return to_ascii(expr)


def render_step(
    step: ProofStep,
    *,
    max_depth: int = 64,
    verbosity: str = "full",
) -> str:
    """Render a :class:`ProofStep` as a coloured tree (or ASCII fallback).

    ``verbosity`` is one of :data:`VERBOSITY_MODES`, ``"full"`` shows
    rule + tag + before→after + justification + children; ``"summary"``
    drops the justification; ``"compact"`` collapses each step to a
    single ``[rule] (tag)`` line with no children.
    """
    if not isinstance(step, ProofStep):
        raise TypeError("render_step: expected a ProofStep")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        return step_to_ascii(step, max_depth=max_depth, verbosity=verbosity)
    return _render_to_text(_build_step_tree(step, max_depth, verbosity))


def render_chain(
    chain: ProofChain,
    *,
    max_depth: int = 64,
    title: bool = True,
    verbosity: str = "full",
) -> str:
    """Render a :class:`ProofChain` as a coloured tree (or ASCII fallback).

    When ``title`` is true and rich is present, the output is headed by
    a ``Proof (N steps)`` banner before the tree; when ``title`` is
    false the banner is suppressed. The flag is ignored in the ASCII
    fallback, whose layout does not accommodate a header line.

    ``verbosity`` selects how much of each step is shown,
    see :data:`VERBOSITY_MODES`.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("render_chain: expected a ProofChain")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        return chain_to_ascii(chain, max_depth=max_depth, verbosity=verbosity)
    if len(chain) == 0:
        return "(empty proof chain)"
    return _render_to_text(_build_chain_tree(chain, max_depth, title, verbosity))


# --------------------------------------------------------------------- #
# Convenience printers                                                  #
# --------------------------------------------------------------------- #


def print_expr(expr: Expr, *, console: Optional["Console"] = None) -> None:
    """Pretty-print an :class:`Expr` to the terminal."""
    if not isinstance(expr, Expr):
        raise TypeError("print_expr: expected an Expr")
    if not HAS_RICH:
        print(to_ascii(expr))
        return
    (console or Console()).print(to_ascii(expr))  # pragma: no cover


def print_step(
    step: ProofStep,
    *,
    console: Optional["Console"] = None,
    max_depth: int = 64,
    verbosity: str = "full",
) -> None:
    """Pretty-print a :class:`ProofStep` to the terminal."""
    if not isinstance(step, ProofStep):
        raise TypeError("print_step: expected a ProofStep")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        print(step_to_ascii(step, max_depth=max_depth, verbosity=verbosity))
        return
    # pragma: no cover - rich-only path
    (console or Console()).print(_build_step_tree(step, max_depth, verbosity))


def print_chain(
    chain: ProofChain,
    *,
    console: Optional["Console"] = None,
    max_depth: int = 64,
    title: bool = True,
    verbosity: str = "full",
) -> None:
    """Pretty-print a :class:`ProofChain` to the terminal."""
    if not isinstance(chain, ProofChain):
        raise TypeError("print_chain: expected a ProofChain")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        print(chain_to_ascii(chain, max_depth=max_depth, verbosity=verbosity))
        return
    # pragma: no cover - rich-only path
    if len(chain) == 0:
        print("(empty proof chain)")
        return
    cons = console or Console()
    cons.print(_build_chain_tree(chain, max_depth, title, verbosity))
