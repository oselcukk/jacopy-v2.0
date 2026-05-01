r"""
Bridge from frame-calc step records to ProofChain, Stage G.

The Faz 18 stages D / E / F record per-entry derivation traces as
lists of step records (:class:`KoszulStep`, :class:`CurvatureStep`,
:class:`RicciStep`). Stage G lifts those lists into the existing
:class:`~jacopy.proof.chain.ProofChain` data type so the same
display layer (:func:`~jacopy.display.chain_to_latex.chain_to_latex`,
:func:`~jacopy.display.chain_to_latex.chain_to_latex_document`,
:func:`~jacopy.display.jupyter.display_chain`) handles paper-grade
rendering for both abstract operator-level proofs (the rest of
jacopy) and component-level frame-calc derivations.

Mechanics
---------

Each frame-calc step's ``.expression`` field is a SymPy expression
(or ``None`` for narration-only steps). To embed those into
:class:`~jacopy.proof.step.ProofStep`'s ``before`` / ``after`` slots
, which require jacopy :class:`~jacopy.core.expr.Expr`, we wrap
them in :class:`~jacopy.frame_calc.symbolic_atoms.SymPyAtom` opaque
atoms.

The resulting :class:`ProofStep`s carry ``provenance_tag="computation"``
, a Stage-G addition to :data:`ProofStep._VALID_TAGS` that signals
"this is a calculational step, not an axiom or theorem citation".
The display layer renders it identically to axiom/theorem steps.
"""

from __future__ import annotations

from typing import Any, Iterable

import sympy as sp

from jacopy.core.expr import Expr, Integer
from jacopy.frame_calc.symbolic_atoms import SymPyAtom
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Conversion helpers                                                    #
# --------------------------------------------------------------------- #


def _wrap_sympy(value: Any) -> Expr:
    """Coerce a value to a jacopy :class:`Expr` for ProofStep slots.

    SymPy expressions and bare Python numbers wrap into
    :class:`SymPyAtom`; jacopy :class:`Expr`s pass through; tuples
    (frame-calc step intermediates) wrap as a single
    :class:`SymPyAtom` of the SymPy ``Tuple`` form.
    """
    if isinstance(value, Expr):
        return value
    if value is None:
        return SymPyAtom(sp.S.Zero)
    if isinstance(value, tuple):
        # tuple of SymPy expressions → SymPy Tuple
        return SymPyAtom(sp.Tuple(*value))
    return SymPyAtom(value)


def steps_to_proof_chain(
    steps: Iterable[Any], *, head_label: str = ""
) -> ProofChain:
    """Lift any list of frame-calc step records to a ProofChain.

    Accepts :class:`KoszulStep`, :class:`CurvatureStep`,
    :class:`RicciStep`, or any object with ``rule``,
    ``description``, ``expression`` attributes.

    Each step becomes one :class:`ProofStep` with:

    * ``rule``     ← step's ``rule``
    * ``before``   ← previous step's ``after`` (or
      :class:`Integer(0)` for the very first step)
    * ``after``    ← :class:`SymPyAtom` of step's ``expression``,
      or previous ``after`` if expression is ``None`` (narration step)
    * ``justification`` ← step's ``description``
    * ``provenance_tag`` ← ``"computation"``

    Parameters
    ----------
    steps
        Iterable of step records.
    head_label
        Optional context string appended to the first step's
        justification (e.g. ``"Γ^t_{tr} via Koszul formula"``).

    Returns
    -------
    ProofChain
        Ready to feed into :func:`~jacopy.display.chain_to_latex.chain_to_latex_document`
        or :func:`~jacopy.display.jupyter.display_chain`.
    """
    chain = ProofChain()
    prev_after: Expr = Integer(0)
    for i, step in enumerate(steps):
        rule = getattr(step, "rule", "")
        description = getattr(step, "description", "")
        expression = getattr(step, "expression", None)

        if i == 0 and head_label:
            description = (
                f"{head_label}: {description}" if description else head_label
            )

        if expression is None:
            after = prev_after
        else:
            after = _wrap_sympy(expression)

        chain.append(
            ProofStep(
                before=prev_after,
                after=after,
                rule=rule,
                justification=description,
                provenance_tag="computation",
            )
        )
        prev_after = after
    return chain


# --------------------------------------------------------------------- #
# Display registration, runs at import time                            #
# --------------------------------------------------------------------- #


def _register_latex_handler() -> None:
    """Register a LaTeX handler for :class:`SymPyAtom`.

    The dispatcher in :mod:`jacopy.display.latex` is keyed by class;
    we register here at import time so any
    :func:`~jacopy.display.latex.to_latex` call on a SymPyAtom
    delegates to :func:`sympy.latex` for clean output.
    """
    from jacopy.display import latex as _latex_module

    def _sympy_atom_handler(expr: SymPyAtom, _ctx: int) -> str:
        return sp.latex(expr.sympy)

    _latex_module._HANDLERS[SymPyAtom] = _sympy_atom_handler


_register_latex_handler()
