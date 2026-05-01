"""
Cartan calculus framework.

:class:`CartanCalculus` bundles the four ingredients of a concrete
Cartan calculus, exterior derivative ``d``, Lie-derivative factory
``L``, interior-product factory ``ι``, and vector-field bracket, and
exposes the five canonical relations as reusable
:class:`OperatorEquation` objects:

* ``d² = 0``
* ``[d, ι_X] = L_X``    (Cartan's magic formula)
* ``[d, L_X] = 0``
* ``[L_X, L_Y] = L_{[X, Y]}``
* ``[L_X, ι_Y] = ι_{[X, Y]}``

Relations parametric in one or two vector fields are built on demand;
:meth:`verify` closes a named relation on a supplied algebra via the
standard :class:`~jacopy.proof.strategies.AgreementOnGenerators`
strategy, and :meth:`verify_all` runs every relation in one sweep so
a test can assert the whole Cartan package in a single call.

The class does not introduce a new tactic. ``mode`` selects between
the fast default (axiomatic theorem-mode closures) and the
foundational unroll (via
:class:`~jacopy.proof.strategies.UnrollToFoundations`) so the
generated transcripts can be either concise or self-contained.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from jacopy.algebra.commutator import commutator
from jacopy.algebra.derivation import Derivation, compose
from jacopy.brackets.base import GradedBracket
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.operator_equation import OperatorEquation
from jacopy.core.expr import Expr, Integer
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine, default_engine
from jacopy.proof.strategies import (
    ExpandAndSimplify,
    ProofFailure,
    Strategy,
    UnrollToFoundations,
)


RELATIONS: Tuple[str, ...] = (
    "d_squared_zero",
    "cartan_magic",
    "d_lie",
    "lie_lie",
    "lie_iota",
)

MODES: Tuple[str, ...] = ("efficient", "foundational")


LieDerivativeFactory = Callable[[Expr], Derivation]
InteriorFactory = Callable[[Expr], Derivation]


class CartanCalculus:
    """Cartan calculus bundle.

    Parameters
    ----------
    d
        Exterior derivative :class:`Derivation` (degree +1). Typically
        :data:`jacopy.calculus.exterior_d.d`.
    lie_derivative
        Factory ``X → L_X``.
    interior
        Factory ``X → ι_X``.
    vector_bracket
        Bracket on vector fields, ``L_{[X, Y]}`` uses
        ``vector_bracket(X, Y)`` to name the bracket of two fields.

    The factories are kept as callables (rather than pre-instantiated
    operators) because the parametric relations need fresh ``L_X /
    ι_X / L_{[X,Y]} / ι_{[X,Y]}`` for each application. Callers with a
    single fixed vector field can still bind the factory once at the
    call site.
    """

    def __init__(
        self,
        d: Derivation,
        lie_derivative: LieDerivativeFactory,
        interior: InteriorFactory,
        vector_bracket: GradedBracket,
    ) -> None:
        if not isinstance(d, Derivation):
            raise TypeError("CartanCalculus 'd' must be a Derivation")
        if not callable(lie_derivative):
            raise TypeError("CartanCalculus 'lie_derivative' must be callable")
        if not callable(interior):
            raise TypeError("CartanCalculus 'interior' must be callable")
        if not isinstance(vector_bracket, GradedBracket):
            raise TypeError(
                "CartanCalculus 'vector_bracket' must be a GradedBracket"
            )
        self._d = d
        self._lie_derivative = lie_derivative
        self._interior = interior
        self._vector_bracket = vector_bracket

    # ---- accessors -------------------------------------------------- #

    @property
    def d(self) -> Derivation:
        return self._d

    @property
    def lie_derivative(self) -> LieDerivativeFactory:
        return self._lie_derivative

    @property
    def interior(self) -> InteriorFactory:
        return self._interior

    @property
    def vector_bracket(self) -> GradedBracket:
        return self._vector_bracket

    # ---- relations as OperatorEquations ---------------------------- #

    def relation(
        self,
        name: str,
        *,
        X: Optional[Expr] = None,
        Y: Optional[Expr] = None,
        algebra: Any = None,
    ) -> OperatorEquation:
        """Return the :class:`OperatorEquation` for a named relation.

        Required parameters depend on the relation: ``d_squared_zero``
        is unary, ``cartan_magic`` and ``d_lie`` need ``X``, and
        ``lie_lie`` / ``lie_iota`` need both ``X`` and ``Y``. ``algebra``
        is carried through to the returned :class:`OperatorEquation`
        so ``.prove()`` can fire immediately.
        """
        if name not in RELATIONS:
            raise ValueError(
                f"Unknown Cartan relation {name!r}; "
                f"expected one of {RELATIONS}"
            )

        if name == "d_squared_zero":
            lhs = compose(self._d, self._d)
            rhs = Integer(0)
            return OperatorEquation(lhs=lhs, rhs=rhs, algebra=algebra)

        if name == "cartan_magic":
            if X is None:
                raise ValueError("cartan_magic requires X")
            iota_X = self._interior(X)
            L_X = self._lie_derivative(X)
            lhs = commutator(self._d, iota_X).expand()
            return OperatorEquation(lhs=lhs, rhs=L_X, algebra=algebra)

        if name == "d_lie":
            if X is None:
                raise ValueError("d_lie requires X")
            L_X = self._lie_derivative(X)
            lhs = commutator(self._d, L_X).expand()
            return OperatorEquation(lhs=lhs, rhs=Integer(0), algebra=algebra)

        if name == "lie_lie":
            if X is None or Y is None:
                raise ValueError("lie_lie requires X and Y")
            L_X = self._lie_derivative(X)
            L_Y = self._lie_derivative(Y)
            XY = self._vector_bracket.expand(X, Y)
            L_XY = self._lie_derivative(XY)
            lhs = commutator(L_X, L_Y).expand()
            return OperatorEquation(lhs=lhs, rhs=L_XY, algebra=algebra)

        # lie_iota
        if X is None or Y is None:
            raise ValueError("lie_iota requires X and Y")
        L_X = self._lie_derivative(X)
        iota_Y = self._interior(Y)
        XY = self._vector_bracket.expand(X, Y)
        iota_XY = self._interior(XY)
        lhs = commutator(L_X, iota_Y).expand()
        return OperatorEquation(lhs=lhs, rhs=iota_XY, algebra=algebra)

    # ---- verification ---------------------------------------------- #

    def verify(
        self,
        name: str,
        *,
        algebra: Any,
        X: Optional[Expr] = None,
        Y: Optional[Expr] = None,
        mode: str = "efficient",
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> ProofChain:
        """Prove a single Cartan relation on ``algebra``.

        ``mode`` picks the engine flavour:

        * ``"efficient"``, standard :func:`default_engine` in efficient
          mode. Theorem-classified definitions fire as single-step
          rewrites.
        * ``"foundational"``, wraps the sub-strategy in
          :class:`UnrollToFoundations`, so every theorem unrolls to its
          axiomatic proof in the transcript.

        Raises :class:`ProofFailure` if the underlying proof fails.
        """
        if mode not in MODES:
            raise ValueError(
                f"CartanCalculus.verify mode must be one of {MODES}, "
                f"got {mode!r}"
            )
        eq = self.relation(name, X=X, Y=Y, algebra=algebra)

        sub_strategy: Optional[Strategy]
        if mode == "foundational":
            sub_strategy = UnrollToFoundations(ExpandAndSimplify())
        else:
            sub_strategy = None

        eff_engine = engine
        if eff_engine is None and registry is not None:
            # Pass this calculus' own ``d`` so definitions pinned to an
            # :class:`ExteriorDerivative` target (``DSquaredZero`` and
            # ``IotaOnExactOneForm``) fire on bundle-specific operators
            # like a Lie-algebroid ``d_E``, not just the TM default.
            eff_engine = default_engine(
                registry=registry, d=self._d, mode=mode
            )

        return eq.prove(
            registry=registry,
            engine=eff_engine,
            strategy=sub_strategy,
        )

    def verify_all(
        self,
        *,
        algebra: Any,
        X: Expr,
        Y: Expr,
        mode: str = "efficient",
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> Dict[str, ProofChain]:
        """Prove every Cartan relation in :data:`RELATIONS` in turn.

        Returns a ``{name: ProofChain}`` mapping. If any relation
        fails, the :class:`ProofFailure` is raised immediately with
        the offending relation name annotated, the mapping is partial
        on failure, not merged with error objects, so downstream
        callers don't have to disambiguate.
        """
        results: Dict[str, ProofChain] = {}
        for name in RELATIONS:
            try:
                results[name] = self.verify(
                    name,
                    algebra=algebra,
                    X=X,
                    Y=Y,
                    mode=mode,
                    registry=registry,
                    engine=engine,
                )
            except ProofFailure as exc:
                raise ProofFailure(
                    f"CartanCalculus.verify_all failed on {name!r}: {exc}"
                )
        return results

    # ---- identity --------------------------------------------------- #

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, CartanCalculus):
            return NotImplemented
        return (
            self._d == other._d
            and self._lie_derivative is other._lie_derivative
            and self._interior is other._interior
            and self._vector_bracket == other._vector_bracket
        )

    def __hash__(self) -> int:
        # Factories compare by identity (``is``) in __eq__; use their
        # ``id`` here so equal objects hash the same.
        return hash(
            (
                type(self).__name__,
                self._d,
                id(self._lie_derivative),
                id(self._interior),
                self._vector_bracket,
            )
        )

    def __repr__(self) -> str:
        return (
            f"CartanCalculus(d={self._d._repr_inner()}, "
            f"bracket={self._vector_bracket.name})"
        )
