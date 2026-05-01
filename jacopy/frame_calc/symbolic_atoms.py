r"""
Opaque ``Expr`` atoms for the abstract-frame mode, Stage A.2.

When a frame is :class:`~jacopy.frame_calc.frame.AbstractFrame`, the
frame derivative ``e_a(f)`` and the Lie bracket structure constant
``γ^a_{bc}`` cannot be evaluated to concrete SymPy expressions. They
remain symbolic, and this module supplies the two opaque
:class:`~jacopy.core.expr.Expr` atoms that carry that symbolism
through downstream formulas:

* :class:`FrameDerivativeExpr`, ``e_a(body)`` with ``body`` a
  arbitrary :class:`Expr`.
* :class:`GammaExpr`, ``γ^a_{bc}`` indexed atom, used when
  :class:`AbstractFrame`'s user-supplied ``gamma_table`` doesn't
  cover ``(a, b, c)``.

Both atoms key on ``(frame, indices, body)`` for structural identity
so that two abstract frames in the same expression don't cross-fire,
and so simplification passes can recognise repeated occurrences.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Atom, Expr


# --------------------------------------------------------------------- #
# FrameDerivativeExpr, ``e_a(body)``                                   #
# --------------------------------------------------------------------- #


class FrameDerivativeExpr(Expr):
    r"""Opaque ``e_a(body)`` for an abstract frame.

    The frame and index ``a`` sit as parametric slots; ``body`` is a
    child :class:`Expr` so the engine walks freely (substitutions
    inside the body still fire).

    Two :class:`FrameDerivativeExpr` instances compare equal iff
    their frames, indices, and (structurally) their bodies match.
    """

    __slots__ = ("_frame", "_index", "_body")

    def __init__(
        self,
        frame: Any,  # AbstractFrame; typed weakly to avoid import cycle
        index: int,
        body: Expr,
    ) -> None:
        if not isinstance(index, int):
            raise TypeError(
                "FrameDerivativeExpr index must be int, "
                f"got {type(index).__name__}"
            )
        if not isinstance(body, Expr):
            raise TypeError(
                "FrameDerivativeExpr body must be an Expr, "
                f"got {type(body).__name__}"
            )
        self._frame = frame
        self._index = index
        self._body = body

    @property
    def frame(self) -> Any:
        return self._frame

    @property
    def index(self) -> int:
        return self._index

    @property
    def body(self) -> Expr:
        return self._body

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._body,)

    def _rebuild(
        self, new_children: Tuple[Expr, ...]
    ) -> "FrameDerivativeExpr":
        if len(new_children) != 1:
            raise ValueError(
                "FrameDerivativeExpr._rebuild expects exactly one child"
            )
        return FrameDerivativeExpr(self._frame, self._index, new_children[0])

    def _key(self) -> Any:
        # Frame is keyed by its id, structural equality across
        # AbstractFrame instances is intentional only when the same
        # object is used. Two distinct abstract frames stay distinct.
        return ("FrameDerivativeExpr", id(self._frame), self._index, self._body)

    def _repr_inner(self) -> str:
        names = self._frame.index_names()
        ix = names[self._index]
        return f"e_{ix}({self._body._repr_inner()})"


# --------------------------------------------------------------------- #
# GammaExpr, ``γ^a_{bc}``                                              #
# --------------------------------------------------------------------- #


class GammaExpr(Atom):
    r"""Opaque structure constant ``γ^a_{bc}`` for an abstract frame.

    A pure :class:`Atom`, no children, no rebuild. Two
    :class:`GammaExpr` instances compare equal iff their frames and
    all three indices match.
    """

    __slots__ = ("_frame", "_a", "_b", "_c")

    def __init__(
        self,
        frame: Any,
        a: int,
        b: int,
        c: int,
    ) -> None:
        for label, value in (("a", a), ("b", b), ("c", c)):
            if not isinstance(value, int):
                raise TypeError(
                    f"GammaExpr index {label} must be int, "
                    f"got {type(value).__name__}"
                )
        self._frame = frame
        self._a = a
        self._b = b
        self._c = c

    @property
    def frame(self) -> Any:
        return self._frame

    @property
    def upper(self) -> int:
        return self._a

    @property
    def lower(self) -> Tuple[int, int]:
        return (self._b, self._c)

    def _key(self) -> Any:
        return ("GammaExpr", id(self._frame), self._a, self._b, self._c)

    def _repr_inner(self) -> str:
        names = self._frame.index_names()
        return f"γ^{names[self._a]}_{{{names[self._b]}{names[self._c]}}}"


# --------------------------------------------------------------------- #
# SymPyAtom, jacopy `Expr` wrapping a SymPy expression                 #
# --------------------------------------------------------------------- #


class InverseMetricEntryExpr(Atom):
    r"""Opaque atom for ``g^{ab}`` when the metric is abstract.

    On an :class:`AbstractFrame`, the inverse metric components
    cannot be computed via SymPy's :meth:`Matrix.inv` because the
    underlying metric entries are jacopy :class:`Expr` atoms (not
    SymPy expressions). The Levi-Civita / Koszul formula needs to
    refer to ``g^{ab}`` symbolically, so we expose each inverse
    entry as a fresh :class:`Atom` keyed on ``(metric_id, a, b)``.

    The atom carries no algebraic relation to the metric, its
    only contract is structural identity. Higher-level proofs that
    need the relation ``g^{ac} g_{cb} = δ^a_b`` would need explicit
    rewrite rules (out of scope for Stage A.2 / Stage D follow-up).
    """

    __slots__ = ("_frame", "_metric_id", "_a", "_b")

    def __init__(
        self,
        frame: Any,
        metric_id: int,
        a: int,
        b: int,
    ) -> None:
        for label, value in (("a", a), ("b", b)):
            if not isinstance(value, int):
                raise TypeError(
                    f"InverseMetricEntryExpr index {label} must be int"
                )
        self._frame = frame
        self._metric_id = metric_id
        self._a = a
        self._b = b

    @property
    def frame(self) -> Any:
        return self._frame

    @property
    def upper(self) -> Tuple[int, int]:
        return (self._a, self._b)

    def _key(self) -> Any:
        return (
            "InverseMetricEntryExpr",
            self._metric_id,
            self._a,
            self._b,
        )

    def _repr_inner(self) -> str:
        names = self._frame.index_names()
        return f"g^{{{names[self._a]}{names[self._b]}}}"


class SymPyAtom(Atom):
    r"""Opaque jacopy :class:`Atom` wrapping a SymPy expression.

    The Stage G :class:`~jacopy.proof.chain.ProofChain` bridge needs
    to embed SymPy expressions (component values like ``Γ^t_{tr} =
    M/(r²(1-2M/r))``) inside :class:`~jacopy.proof.step.ProofStep`'s
    ``before`` / ``after`` slots, which require jacopy ``Expr``
    instances.

    :class:`SymPyAtom` is the bridge: a jacopy :class:`Atom` whose
    only payload is a SymPy expression. Equality / hashing key on
    the structural form of the underlying SymPy expression.

    Display: :meth:`_repr_inner` calls ``str(sp_expr)``; the LaTeX
    layer (:mod:`jacopy.display.latex`) registers a dedicated
    handler that uses :func:`sympy.latex`, so paper-grade
    :func:`~jacopy.display.chain_to_latex_document` output renders
    SymPy entries correctly.

    The wrapping is **opaque from jacopy's algebra perspective**,
    no jacopy engine rule fires on it, no substitution descends
    into it. That's intentional: SymPy and jacopy operate on
    different algebras, and the bridge only mediates *display*,
    not arithmetic.
    """

    __slots__ = ("_sympy",)

    def __init__(self, sympy_expr: Any) -> None:
        # Coerce ints / floats to sympy if needed
        try:
            import sympy as _sp
            if not isinstance(sympy_expr, _sp.Basic):
                sympy_expr = _sp.sympify(sympy_expr)
        except Exception:  # noqa: BLE001
            pass
        self._sympy = sympy_expr

    @property
    def sympy(self) -> Any:
        """The wrapped SymPy expression."""
        return self._sympy

    def _key(self) -> Any:
        # SymPy expressions are hashable; key directly on them
        return ("SymPyAtom", self._sympy)

    def _repr_inner(self) -> str:
        return str(self._sympy)
