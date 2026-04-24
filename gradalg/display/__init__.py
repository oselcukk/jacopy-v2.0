"""Rendering helpers for :class:`~gradalg.core.expr.Expr` trees and
:class:`~gradalg.proof.chain.ProofChain` / :class:`~gradalg.proof.step.ProofStep`.

Stage A — pure-stdlib renderers:

- :mod:`gradalg.display.ascii` — terminal-friendly plain text.
- :mod:`gradalg.display.latex` — paper-quality LaTeX math snippets and
  ``align*`` bodies for proof transcripts.
"""

from gradalg.display.ascii import (
    chain_to_ascii,
    step_to_ascii,
    to_ascii,
)
from gradalg.display.jupyter import (
    LatexDisplay,
    display_chain,
    display_expr,
    display_proof,
    display_step,
)
from gradalg.display.latex import (
    chain_to_latex,
    latex_name,
    step_to_latex,
    to_latex,
)
from gradalg.display.terminal import (
    HAS_RICH,
    print_chain,
    print_expr,
    print_step,
    render_chain,
    render_expr,
    render_step,
)

__all__ = [
    "HAS_RICH",
    "LatexDisplay",
    "chain_to_ascii",
    "chain_to_latex",
    "display_chain",
    "display_expr",
    "display_proof",
    "display_step",
    "latex_name",
    "print_chain",
    "print_expr",
    "print_step",
    "render_chain",
    "render_expr",
    "render_step",
    "step_to_ascii",
    "step_to_latex",
    "to_ascii",
    "to_latex",
]
