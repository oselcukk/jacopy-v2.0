r"""
Jupyter / IPython display adapters.

Wraps the LaTeX output of :mod:`jacopy.display.latex` in a lightweight
displayable that Jupyter's rich-display machinery picks up through the
duck-typed ``_repr_latex_`` / ``_repr_html_`` / ``_repr_mimebundle_``
protocol. No runtime dependency on ``IPython`` is required: the
returned objects expose the required methods themselves, and Jupyter
calls them directly when the object is the last expression in a cell.

Three helpers cover the common cases:

* :func:`display_expr`, a single :class:`~jacopy.core.expr.Expr`,
  wrapped as inline math (``$…$``) so it flows inside a prose cell.
* :func:`display_step`, a single :class:`~jacopy.proof.step.ProofStep`,
  wrapped in an ``align*`` body so arrows and annotations align.
* :func:`display_chain` (aliased :func:`display_proof`), an entire
  :class:`~jacopy.proof.chain.ProofChain` rendered as one ``align*``
  block, ready to paste into a paper draft.

The Expr classes themselves are intentionally not monkey-patched with
``_repr_latex_``. Patching the core class hierarchy from a display
module entangles concerns and makes the rich-display contract invisible
from the model code. Explicit wrappers keep the opt-in boundary clean
and let tests assert the display output without booting a notebook.
"""

from __future__ import annotations

import html as _html

from jacopy.core.expr import Expr
from jacopy.display.ascii import VERBOSITY_MODES
from jacopy.display.latex import chain_to_latex, step_to_latex, to_latex
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


def _check_verbosity(verbosity: str) -> None:
    if verbosity not in VERBOSITY_MODES:
        raise ValueError(
            f"verbosity must be one of {VERBOSITY_MODES}, got {verbosity!r}"
        )


class LatexDisplay:
    """Jupyter-friendly wrapper around a LaTeX math string.

    Carries the raw LaTeX and a hint (:attr:`environment`) telling the
    renderer whether the payload is a self-contained environment like
    ``\\begin{align*}…\\end{align*}`` (display as-is) or a math snippet
    that should be wrapped in ``$…$`` for inline rendering.

    Outside Jupyter, ``str(obj)`` and ``repr(obj)`` both return the raw
    LaTeX so the object still behaves like a plain string when printed.
    """

    __slots__ = ("_latex", "_environment")

    def __init__(self, latex: str, *, environment: bool = False) -> None:
        if not isinstance(latex, str):
            raise TypeError("LatexDisplay expects a str")
        self._latex = latex
        self._environment = bool(environment)

    @property
    def latex(self) -> str:
        return self._latex

    @property
    def environment(self) -> bool:
        return self._environment

    # ---- Jupyter rich-display protocol ----------------------------- #

    def _repr_latex_(self) -> str:
        """Math payload consumed by Jupyter's ``text/latex`` mimetype."""
        if self._environment:
            return self._latex
        return f"${self._latex}$"

    def _repr_html_(self) -> str:
        """HTML fallback so non-LaTeX frontends still render via MathJax."""
        if self._environment:
            # MathJax in HTML picks up `\begin{align*}` at top level.
            body = self._latex
        else:
            body = f"\\({self._latex}\\)"
        return f'<div class="jacopy-latex">{body}</div>'

    def _repr_mimebundle_(
        self, include=None, exclude=None
    ) -> dict:
        """Combined bundle, Jupyter prefers this when available."""
        bundle = {
            "text/latex": self._repr_latex_(),
            "text/html": self._repr_html_(),
            "text/plain": self._latex,
        }
        if include is not None:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude is not None:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle

    # ---- Plain-string behaviour ------------------------------------ #

    def __str__(self) -> str:
        return self._latex

    def __repr__(self) -> str:
        return self._latex

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LatexDisplay):
            return (
                self._latex == other._latex
                and self._environment == other._environment
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._latex, self._environment))


# --------------------------------------------------------------------- #
# Public helpers                                                        #
# --------------------------------------------------------------------- #


def display_expr(expr: Expr) -> LatexDisplay:
    """Wrap an :class:`Expr` as inline Jupyter math."""
    if not isinstance(expr, Expr):
        raise TypeError("display_expr: expected an Expr")
    return LatexDisplay(to_latex(expr))


def display_step(step: ProofStep) -> LatexDisplay:
    """Wrap a single :class:`ProofStep` in a ``gather*`` body."""
    if not isinstance(step, ProofStep):
        raise TypeError("display_step: expected a ProofStep")
    body = step_to_latex(step)
    return LatexDisplay(
        f"\\begin{{gather*}}\n{body}\n\\end{{gather*}}",
        environment=True,
    )


def display_chain(chain: ProofChain) -> LatexDisplay:
    """Wrap a :class:`ProofChain` as a full ``align*`` block."""
    if not isinstance(chain, ProofChain):
        raise TypeError("display_chain: expected a ProofChain")
    return LatexDisplay(chain_to_latex(chain), environment=True)


# Backwards-friendly alias matching the roadmap naming.
display_proof = display_chain


# --------------------------------------------------------------------- #
# Collapsible HTML rendering                                            #
# --------------------------------------------------------------------- #
#
# Jupyter's ``align*`` renders a transcript as a flat table of arrows
# that is perfect on paper but loses the nesting of foundational proofs
# (a single Cartan-identity derivation carries a sub-chain for every
# axiom fire). This alternative renderer emits HTML with ``<details>``
# elements so the nesting collapses and expands in-notebook, and each
# step's ``before → after`` still renders through MathJax via ``\(...\)``
# delimiters.


class HtmlProofDisplay:
    """Jupyter-friendly wrapper around a raw HTML proof-tree payload.

    Unlike :class:`LatexDisplay`, this carries HTML rather than LaTeX
    math, so the rich-display protocol only advertises ``text/html``.
    ``str`` / ``repr`` return the raw HTML so the object behaves like a
    plain string when printed outside a notebook.
    """

    __slots__ = ("_html",)

    def __init__(self, html: str) -> None:
        if not isinstance(html, str):
            raise TypeError("HtmlProofDisplay expects a str")
        self._html = html

    @property
    def html(self) -> str:
        return self._html

    def _repr_html_(self) -> str:
        return self._html

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        bundle = {"text/html": self._html, "text/plain": self._html}
        if include is not None:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude is not None:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle

    def __str__(self) -> str:
        return self._html

    def __repr__(self) -> str:
        return self._html

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HtmlProofDisplay):
            return self._html == other._html
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._html)


def _step_summary_html(step: ProofStep, verbosity: str) -> str:
    """Compose the ``<summary>`` (or leaf body) HTML for a single step.

    Rule names / justifications are HTML-escaped because they can contain
    characters like ``<``, ``&`` that must not appear raw in the DOM. The
    ``before → after`` fragment is emitted as MathJax-delimited LaTeX so
    the notebook typesets the expressions properly instead of printing
    their source.
    """
    rule = _html.escape(step.rule)
    pieces = [f'<span class="jacopy-rule">[{rule}]</span>']
    if step.provenance_tag:
        tag = _html.escape(step.provenance_tag)
        pieces.append(f'<span class="jacopy-tag">({tag})</span>')
    if verbosity != "compact":
        before = to_latex(step.before)
        after = to_latex(step.after)
        pieces.append(f'<span class="jacopy-math">\\({before} \\to {after}\\)</span>')
        if verbosity == "full" and step.justification:
            just = _html.escape(step.justification)
            pieces.append(f'<span class="jacopy-just">, {just}</span>')
    return " ".join(pieces)


def _step_to_html(step: ProofStep, verbosity: str, max_depth: int) -> str:
    """Render a :class:`ProofStep` as a single HTML fragment.

    Leaf steps (or compact-mode / depth-capped steps) become a plain
    ``<div>``; steps with visible children become ``<details open>`` so
    the reader sees the nesting expanded by default but can fold it
    away. ``max_depth`` mirrors the terminal renderer's semantics,
    ``0`` suppresses descent.
    """
    summary = _step_summary_html(step, verbosity)
    show_children = (
        verbosity != "compact" and max_depth > 0 and bool(step.children)
    )
    if not show_children:
        return f'<div class="jacopy-step">{summary}</div>'
    child_html = "".join(
        _step_to_html(ch, verbosity, max_depth - 1) for ch in step.children
    )
    return (
        '<details open class="jacopy-step">'
        f"<summary>{summary}</summary>"
        f'<div class="jacopy-children">{child_html}</div>'
        "</details>"
    )


def display_step_collapsible(
    step: ProofStep,
    *,
    max_depth: int = 64,
    verbosity: str = "full",
) -> HtmlProofDisplay:
    """Wrap a :class:`ProofStep` as a collapsible HTML tree for Jupyter.

    ``verbosity`` is one of :data:`VERBOSITY_MODES`, see
    :mod:`jacopy.display.ascii` for the mode semantics.
    """
    if not isinstance(step, ProofStep):
        raise TypeError("display_step_collapsible: expected a ProofStep")
    _check_verbosity(verbosity)
    return HtmlProofDisplay(_step_to_html(step, verbosity, max_depth))


def display_chain_collapsible(
    chain: ProofChain,
    *,
    max_depth: int = 64,
    title: bool = True,
    verbosity: str = "full",
) -> HtmlProofDisplay:
    """Wrap a :class:`ProofChain` as a collapsible HTML tree for Jupyter.

    A ``Proof (N steps)`` heading precedes the tree when ``title`` is
    true. Empty chains render as a placeholder div so downstream code
    doesn't have to special-case the empty shape.
    """
    if not isinstance(chain, ProofChain):
        raise TypeError("display_chain_collapsible: expected a ProofChain")
    _check_verbosity(verbosity)
    if len(chain) == 0:
        return HtmlProofDisplay(
            '<div class="jacopy-proof">(empty proof chain)</div>'
        )
    steps_html = "".join(
        _step_to_html(s, verbosity, max_depth) for s in chain.steps
    )
    if title:
        header = (
            f'<div class="jacopy-proof-header">Proof ({len(chain)} steps)</div>'
        )
        body = header + steps_html
    else:
        body = steps_html
    return HtmlProofDisplay(f'<div class="jacopy-proof">{body}</div>')
