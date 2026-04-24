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
from gradalg.display.latex import (
    chain_to_latex,
    latex_name,
    step_to_latex,
    to_latex,
)

__all__ = [
    "chain_to_ascii",
    "chain_to_latex",
    "latex_name",
    "step_to_ascii",
    "step_to_latex",
    "to_ascii",
    "to_latex",
]
