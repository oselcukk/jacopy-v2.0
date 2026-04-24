r"""
Jupyter / IPython display adapters.

Wraps the LaTeX output of :mod:`gradalg.display.latex` in a lightweight
displayable that Jupyter's rich-display machinery picks up through the
duck-typed ``_repr_latex_`` / ``_repr_html_`` / ``_repr_mimebundle_``
protocol. No runtime dependency on ``IPython`` is required: the
returned objects expose the required methods themselves, and Jupyter
calls them directly when the object is the last expression in a cell.

Three helpers cover the common cases:

* :func:`display_expr` — a single :class:`~gradalg.core.expr.Expr`,
  wrapped as inline math (``$…$``) so it flows inside a prose cell.
* :func:`display_step` — a single :class:`~gradalg.proof.step.ProofStep`,
  wrapped in an ``align*`` body so arrows and annotations align.
* :func:`display_chain` (aliased :func:`display_proof`) — an entire
  :class:`~gradalg.proof.chain.ProofChain` rendered as one ``align*``
  block, ready to paste into a paper draft.

The Expr classes themselves are intentionally not monkey-patched with
``_repr_latex_``. Patching the core class hierarchy from a display
module entangles concerns and makes the rich-display contract invisible
from the model code. Explicit wrappers keep the opt-in boundary clean
and let tests assert the display output without booting a notebook.
"""

from __future__ import annotations

from gradalg.core.expr import Expr
from gradalg.display.latex import chain_to_latex, step_to_latex, to_latex
from gradalg.proof.chain import ProofChain
from gradalg.proof.step import ProofStep


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
        return f'<div class="gradalg-latex">{body}</div>'

    def _repr_mimebundle_(
        self, include=None, exclude=None
    ) -> dict:
        """Combined bundle — Jupyter prefers this when available."""
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
    """Wrap a single :class:`ProofStep` in an ``align*`` body."""
    if not isinstance(step, ProofStep):
        raise TypeError("display_step: expected a ProofStep")
    body = step_to_latex(step)
    return LatexDisplay(
        f"\\begin{{align*}}\n{body}\n\\end{{align*}}",
        environment=True,
    )


def display_chain(chain: ProofChain) -> LatexDisplay:
    """Wrap a :class:`ProofChain` as a full ``align*`` block."""
    if not isinstance(chain, ProofChain):
        raise TypeError("display_chain: expected a ProofChain")
    return LatexDisplay(chain_to_latex(chain), environment=True)


# Backwards-friendly alias matching the roadmap naming.
display_proof = display_chain
