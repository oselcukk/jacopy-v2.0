r"""
Koszul-connection capstone wrapper, Q9 Stage 9.F.

A :class:`KoszulConnectionProblem` bundles a single
:class:`~jacopy.calculus.connection.AffineConnection` whose vector
bracket is a :class:`~jacopy.brackets.base.GradedBracket`
(canonically a :class:`~jacopy.brackets.koszul.KoszulBracket` on
``T*M`` for the Q9 Poisson setting) together with a local frame and
forwards the three families of mechanised proofs the Q9 chapter cites:

* Bianchi I / II, via :class:`~jacopy.library.bianchi_problem.BianchiProblem`
  with its ``BracketApply`` closure family swapped in for the LBVF
  rules (Stage 9.C).
* Cartan form-property props (``ω̃``, ``T̃``, ``R̃`` C∞-bilinearity +
  antisymmetry), via :class:`~jacopy.library.cartan_form_property.CartanFormPropertyProblem`,
  inherited verbatim because the property axioms don't open the
  bracket (Stage 9.D certification).
* Cartan structure equations I / II on T*M, via
  :class:`~jacopy.library.cartan_structure.CartanStructureProblem`,
  whose engine swaps in the anchor-pulled
  :class:`~jacopy.calculus.intrinsic_axioms.KoszulExteriorDIntrinsicDefinition`
  whenever the connection carries a bracket (Stage 9.E).

The wrapper is a thin facade: each sub-problem is built lazily on
first access so callers that only need (say) Bianchi I don't pay the
construction cost of the Cartan engine. The top-level constructor
takes an already-built ``koszul_connection`` so that the user's choice
of anchor / bracket / function-action wiring stays explicit at the
call site.
"""

from __future__ import annotations

from typing import Optional

from jacopy.calculus.connection import AffineConnection
from jacopy.calculus.local_frame import LocalFrame
from jacopy.calculus.metric import MetricTensor
from jacopy.library.bianchi_problem import BianchiProblem
from jacopy.library.cartan_form_property import CartanFormPropertyProblem
from jacopy.library.cartan_structure import CartanStructureProblem


class KoszulConnectionProblem:
    r"""``(∇̃, F)``, Koszul-connection capstone bundle.

    Parameters
    ----------
    connection
        :class:`AffineConnection` whose ``bracket`` slot carries a
        :class:`~jacopy.brackets.base.GradedBracket` (e.g. a
        :class:`~jacopy.brackets.koszul.KoszulBracket`). The
        :func:`~jacopy.calculus.connection.koszul_connection` factory
        is the canonical way to build one.
    frame
        :class:`LocalFrame` used by the Cartan structure / form-property
        sub-problems. The Bianchi facet does not consult a frame.
    metric
        Optional :class:`MetricTensor` on ``T*M``. Required only for the
        $\widetilde{Q}_{ab}$ non-metricity property proofs; forwarded to
        the form-property facet on construction. Bianchi and Cartan
        structure facets do not consult a metric.
    name
        Display name; defaults to a ``(∇̃, F)`` summary.
    """

    __slots__ = (
        "_conn",
        "_frame",
        "_metric",
        "_name",
        "_bianchi",
        "_form_property",
        "_cartan_structure",
    )

    def __init__(
        self,
        connection: AffineConnection,
        frame: LocalFrame,
        *,
        metric: Optional[MetricTensor] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "KoszulConnectionProblem requires an AffineConnection"
            )
        if connection.bracket is None:
            raise ValueError(
                "KoszulConnectionProblem requires the connection to "
                "carry a bracket, pass a koszul_connection(...) or "
                "another bracket-equipped AffineConnection. For a "
                "bracket-free connection use the BianchiProblem / "
                "CartanStructureProblem wrappers directly."
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "KoszulConnectionProblem requires a LocalFrame"
            )
        if metric is not None and not isinstance(metric, MetricTensor):
            raise TypeError(
                "KoszulConnectionProblem metric must be a "
                "MetricTensor or None"
            )
        self._conn = connection
        self._frame = frame
        self._metric = metric
        self._name = (
            name
            if name is not None
            else (
                f"KoszulConnectionProblem({connection._repr_inner()},"
                f"{frame.name})"
            )
        )
        self._bianchi: Optional[BianchiProblem] = None
        self._form_property: Optional[CartanFormPropertyProblem] = None
        self._cartan_structure: Optional[CartanStructureProblem] = None

    # ---- accessors ------------------------------------------------- #

    @property
    def connection(self) -> AffineConnection:
        return self._conn

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    @property
    def metric(self) -> Optional[MetricTensor]:
        return self._metric

    @property
    def name(self) -> str:
        return self._name

    # ---- sub-problem facets ---------------------------------------- #

    @property
    def bianchi(self) -> BianchiProblem:
        """Stage 9.C facet: Bianchi I / II with BracketApply closure."""
        if self._bianchi is None:
            self._bianchi = BianchiProblem(self._conn)
        return self._bianchi

    @property
    def form_property(self) -> CartanFormPropertyProblem:
        r"""Stage 9.D facet: Cartan form-property propagators.

        Forwards the optional metric so $\widetilde{Q}_{ab}$
        non-metricity proofs are available when a metric was supplied.
        """
        if self._form_property is None:
            self._form_property = CartanFormPropertyProblem(
                self._conn, self._frame, metric=self._metric
            )
        return self._form_property

    @property
    def cartan_structure(self) -> CartanStructureProblem:
        """Stage 9.E facet: Cartan I / II with anchor-pulled d̃."""
        if self._cartan_structure is None:
            self._cartan_structure = CartanStructureProblem(
                self._conn, self._frame
            )
        return self._cartan_structure

    def __repr__(self) -> str:
        return (
            f"KoszulConnectionProblem({self._conn._repr_inner()},"
            f"{self._frame.name})"
        )
