"""
Operator-equation helper, thin object wrapper around agreement-on-generators.

The :class:`OperatorEquation` class bundles a left-hand operator, a
right-hand operator, and the :class:`ExteriorAlgebra` (or any
``generators``-exposing algebra) they act on, so callers can write

    OperatorEquation(lhs=L_X, rhs=Sum(d∘ι_X, ι_X∘d), algebra=Ω).prove()

rather than threading the same arguments through
:func:`~jacopy.proof.verifier.prove_operator_equation` at every call
site.

The class does not introduce a new tactic. It reuses
:class:`~jacopy.proof.strategies.AgreementOnGenerators` as the default
:class:`Strategy`; any other strategy can be plugged in via the
``strategy`` argument to :meth:`prove`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from jacopy.core.expr import Expr
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.strategies import Strategy
from jacopy.proof.verifier import prove_operator_equation


@dataclass(frozen=True)
class OperatorEquation:
    """Deferred statement ``lhs == rhs`` on an algebra.

    Holding the triple ``(lhs, rhs, algebra)`` as data lets callers
    compose operator equations before attempting any proof, e.g. store
    a list of Cartan relations and discharge them in a batch. Instances
    are frozen and hashable so they can be keyed in registries of
    "known" / "to-prove" equations.

    Parameters
    ----------
    lhs
        Left operator (typically an :class:`~jacopy.algebra.derivation.Derivation`,
        a :class:`~jacopy.core.expr.Sum` of derivations, or a composition).
    rhs
        Right operator, of the same degree as ``lhs``.
    algebra
        Any object exposing a ``generators`` property, the canonical
        example is :class:`~jacopy.calculus.exterior_algebra.ExteriorAlgebra`.
    """

    lhs: Expr
    rhs: Expr
    algebra: Any

    def prove(
        self,
        *,
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
        strategy: Optional[Strategy] = None,
    ) -> ProofChain:
        """Discharge the equation and return the transcript.

        Delegates to
        :func:`~jacopy.proof.verifier.prove_operator_equation`, which
        defaults to :class:`~jacopy.proof.strategies.AgreementOnGenerators`
        over the algebra's generators. Raises
        :class:`~jacopy.proof.strategies.ProofFailure` when degrees
        disagree or any per-generator sub-proof fails; the exception
        names the offending generator so the missing grading
        declaration or definition is easy to find.
        """
        return prove_operator_equation(
            self.lhs,
            self.rhs,
            self.algebra,
            registry=registry,
            engine=engine,
            sub_strategy=strategy,
        )
