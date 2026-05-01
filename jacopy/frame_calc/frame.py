r"""
Frame protocol + the three frame implementations, Stage A.

A *frame* is a basis ``(e_1, …, e_n)`` of vector fields on (an open
set of) a manifold, together with the dual coframe ``(e^1, …, e^n)``
of 1-forms. In `frame_calc` a :class:`Frame` is *not* an :class:`Expr`
, it is a wrapper exposing two operations downstream code consumes:

* **Frame derivative** ``e_a(f)``, the action of the frame vector
  field ``e_a`` on a function ``f``.
* **Lie bracket structure constants** ``γ^a_{bc}`` from
  ``[e_b, e_c] = γ^a_{bc} e_a``.

Three concrete frames implement the protocol:

* :class:`CoordinateFrame`, ``e_a = ∂/∂x^a``. Frame derivative
  is SymPy partial differentiation; structure constants are zero.
  This is the Stage A focus.
* :class:`AbstractFrame`, ``e_a`` opaque, ``γ^a_{bc}`` either
  user-supplied or left as opaque atoms. Stage A.2.
* :class:`Tetrad`, ``e_a = e_a^μ ∂/∂x^μ`` defined by a vielbein
  matrix on top of a coordinate frame. Stage B.

Stage A defines the protocol and a fully-working
:class:`CoordinateFrame`; the other two are stubs that raise
:class:`NotImplementedError` until their stages land.

The protocol is **duck-typed** rather than a strict
:class:`abc.ABC`, Python's structural typing keeps the interface
informal, and downstream code only ever calls the two methods.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc


# --------------------------------------------------------------------- #
# Protocol                                                              #
# --------------------------------------------------------------------- #


class Frame:
    """Protocol shared by every frame implementation.

    Subclasses **must** override :meth:`derivative` and :meth:`gamma`.
    They **should** set :attr:`dim`, :attr:`name`, and provide
    :meth:`index_names` for display.

    The base class is a runtime-checkable interface: instantiating it
    directly raises :class:`TypeError`. Use :class:`CoordinateFrame`,
    :class:`AbstractFrame`, or :class:`Tetrad`.
    """

    dim: int
    name: str

    def __init__(self) -> None:
        if type(self) is Frame:
            raise TypeError(
                "Frame is an abstract protocol; instantiate "
                "CoordinateFrame, AbstractFrame, or Tetrad instead."
            )

    # ---- protocol methods ------------------------------------------ #

    def index_names(self) -> Tuple[str, ...]:
        """Display names for the frame indices.

        Returns a tuple of length :attr:`dim`. Default implementation
        returns ``('0', '1', …, str(dim - 1))``; override for prettier
        output (e.g. ``('t', 'r', 'θ', 'φ')``).
        """
        return tuple(str(i) for i in range(self.dim))

    def derivative(self, expr: Any, a: int) -> Any:
        """The frame-derivative ``e_a(expr)``.

        Returns a SymPy expression for concrete frames
        (:class:`CoordinateFrame`, :class:`Tetrad`) and an opaque
        :class:`~jacopy.core.expr.Expr` for :class:`AbstractFrame`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override derivative"
        )

    def gamma(self, a: int, b: int, c: int) -> Any:
        """Lie bracket structure constant ``γ^a_{bc}``.

        Concrete frames return SymPy expressions (zero for
        :class:`CoordinateFrame`); :class:`AbstractFrame` returns
        either a user-supplied entry or an opaque atom.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override gamma"
        )

    # ---- introspection helpers ------------------------------------- #

    def _check_index(self, label: str, value: int) -> None:
        """Bounds-check a frame index; raise :class:`IndexError` otherwise."""
        if not isinstance(value, int):
            raise TypeError(
                f"{type(self).__name__}.{label}: index must be int, "
                f"got {type(value).__name__}"
            )
        if not 0 <= value < self.dim:
            raise IndexError(
                f"{type(self).__name__}.{label}: index {value} out of "
                f"range for dim={self.dim}"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"


# --------------------------------------------------------------------- #
# CoordinateFrame, Stage A                                             #
# --------------------------------------------------------------------- #


class CoordinateFrame(Frame):
    r"""Coordinate frame ``e_a = ∂/∂x^a`` on a chart.

    Parameters
    ----------
    coords
        A sequence of SymPy symbols (typically created via
        ``sp.symbols('t r θ φ')``). The frame's dimension equals
        ``len(coords)``; the symbols themselves serve as the
        differentiation variables for :meth:`derivative`.
    name
        Optional display name; defaults to a comma-joined list of
        coordinate symbols.

    Notes
    -----
    Coordinate frames are **holonomic**: the structure constants
    ``γ^a_{bc} = 0`` because partial derivatives commute. This is
    the simplest case of the Faz 18 protocol and the one most
    physics computations reach for.

    The :meth:`derivative` of a SymPy expression is just
    :func:`sympy.diff` on the corresponding coordinate symbol.
    Higher-level callers (``levi_civita``, ``curvature``, …) treat
    the result as a SymPy expression, they don't need to know
    the frame is coordinate.
    """

    __slots__ = ("_coords", "name", "dim")

    def __init__(
        self,
        coords: Sequence[sp.Symbol],
        *,
        name: str | None = None,
    ) -> None:
        coords_t = tuple(coords)
        if not coords_t:
            raise ValueError(
                "CoordinateFrame requires at least one coordinate symbol"
            )
        for c in coords_t:
            if not isinstance(c, sp.Symbol):
                raise TypeError(
                    "CoordinateFrame coords must be SymPy Symbols, "
                    f"got {type(c).__name__}"
                )
        self._coords: Tuple[sp.Symbol, ...] = coords_t
        self.dim: int = len(coords_t)
        self.name: str = (
            name
            if name is not None
            else f"coord({','.join(str(c) for c in coords_t)})"
        )

    # ---- coordinate-specific accessors ----------------------------- #

    @property
    def coords(self) -> Tuple[sp.Symbol, ...]:
        """The coordinate symbols, in order."""
        return self._coords

    def coord(self, a: int) -> sp.Symbol:
        """The coordinate symbol at index ``a``."""
        self._check_index("coord", a)
        return self._coords[a]

    # ---- protocol methods ------------------------------------------ #

    def index_names(self) -> Tuple[str, ...]:
        return tuple(str(c) for c in self._coords)

    def derivative(self, expr: Any, a: int) -> sp.Expr:
        r"""Return ``e_a(expr) = ∂expr/∂x^a`` as a SymPy expression.

        Accepts SymPy expressions, ints, floats, or anything
        :func:`sympy.sympify` recognises; returns a SymPy ``Expr``.
        """
        self._check_index("derivative", a)
        return sp.diff(sp.sympify(expr), self._coords[a])

    def gamma(self, a: int, b: int, c: int) -> sp.Integer:
        r"""``γ^a_{bc} = 0`` for coordinate frames (always)."""
        for label, value in (("a", a), ("b", b), ("c", c)):
            self._check_index(f"gamma {label}", value)
        return sp.Integer(0)

    # ---- equality / hashing ---------------------------------------- #

    def _key(self) -> Any:
        return ("CoordinateFrame", self._coords, self.name)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CoordinateFrame)
            and self._key() == other._key()
        )

    def __hash__(self) -> int:
        return hash(self._key())


# --------------------------------------------------------------------- #
# AbstractFrame, Stage A.2 stub                                        #
# --------------------------------------------------------------------- #


class AbstractFrame(Frame):
    r"""Abstract frame with user-supplied (or opaque) ``γ^a_{bc}``.

    Stage A.2 deliverable. Used when the metric components and frame
    structure are kept symbolic, every derivative becomes an opaque
    :class:`~jacopy.frame_calc.symbolic_atoms.FrameDerivativeExpr` and
    every uncovered structure constant becomes a
    :class:`~jacopy.frame_calc.symbolic_atoms.GammaExpr`. Higher-level
    formulas (Koszul, curvature, …) compose these atoms structurally.

    Parameters
    ----------
    dim
        Frame dimension; must be a positive int.
    gamma_table
        Optional ``{(a, b, c): expr}`` mapping. Entries returned by
        :meth:`gamma` come from this table when present; otherwise an
        opaque :class:`GammaExpr` is returned. Useful when you have
        partial information about the structure constants (some
        identically zero, some closed-form).
    index_names
        Optional tuple of length ``dim`` for display. Defaults to
        ``("0", "1", …)``.
    name
        Optional display name; defaults to ``f"abstract({dim})"``.

    Notes
    -----
    Two distinct :class:`AbstractFrame` instances are *not* equal,
    each ``derivative`` / ``gamma`` call keys on ``id(self)``, so two
    abstract frames inside the same expression stay distinct. This
    matches the way Cartan calculus's :class:`LocalFrame` distinguishes
    frames by name.
    """

    __slots__ = ("dim", "name", "_index_names", "_gamma_table")

    def __init__(
        self,
        dim: int,
        *,
        gamma_table: dict[Tuple[int, int, int], Any] | None = None,
        index_names: Sequence[str] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                "AbstractFrame dim must be a positive int, "
                f"got {dim!r}"
            )
        self.dim = dim
        self.name = name if name is not None else f"abstract({dim})"
        if index_names is not None:
            ix = tuple(index_names)
            if len(ix) != dim:
                raise ValueError(
                    f"AbstractFrame index_names length {len(ix)} "
                    f"does not match dim {dim}"
                )
            self._index_names = ix
        else:
            self._index_names = tuple(str(i) for i in range(dim))
        if gamma_table is not None:
            for key in gamma_table:
                if (
                    not isinstance(key, tuple)
                    or len(key) != 3
                    or not all(isinstance(k, int) for k in key)
                ):
                    raise TypeError(
                        "AbstractFrame gamma_table keys must be "
                        f"(int, int, int) tuples, got {key!r}"
                    )
                for k in key:
                    if not 0 <= k < dim:
                        raise IndexError(
                            f"AbstractFrame gamma_table key {key} has "
                            f"index out of range for dim={dim}"
                        )
            self._gamma_table: dict[
                Tuple[int, int, int], Any
            ] | None = dict(gamma_table)
        else:
            self._gamma_table = None

    def index_names(self) -> Tuple[str, ...]:
        return self._index_names

    def derivative(self, expr: Any, a: int) -> Any:
        r"""Return ``e_a(expr)`` as an opaque
        :class:`~jacopy.frame_calc.symbolic_atoms.FrameDerivativeExpr`.

        ``expr`` must be a :class:`~jacopy.core.expr.Expr`. The body
        is preserved verbatim, substitutions inside it still fire
        through the standard :meth:`Expr.substitute_atom` traversal.
        """
        self._check_index("derivative", a)
        # Local import: symbolic_atoms imports Expr but not Frame, no cycle.
        from jacopy.frame_calc.symbolic_atoms import FrameDerivativeExpr
        from jacopy.core.expr import Expr as _Expr

        if not isinstance(expr, _Expr):
            raise TypeError(
                "AbstractFrame.derivative requires an Expr body, "
                f"got {type(expr).__name__}"
            )
        return FrameDerivativeExpr(self, a, expr)

    def gamma(self, a: int, b: int, c: int) -> Any:
        r"""Return ``γ^a_{bc}``.

        If the constructor's ``gamma_table`` covers ``(a, b, c)``,
        return that entry. Otherwise return a fresh
        :class:`~jacopy.frame_calc.symbolic_atoms.GammaExpr` opaque
        atom keyed on ``(self, a, b, c)``.
        """
        for label, value in (("a", a), ("b", b), ("c", c)):
            self._check_index(f"gamma {label}", value)
        if (
            self._gamma_table is not None
            and (a, b, c) in self._gamma_table
        ):
            return self._gamma_table[(a, b, c)]
        from jacopy.frame_calc.symbolic_atoms import GammaExpr

        return GammaExpr(self, a, b, c)


# --------------------------------------------------------------------- #
# Tetrad, Stage B stub                                                  #
# --------------------------------------------------------------------- #


class Tetrad(Frame):
    r"""Tetrad ``e_a = e_a^μ ∂/∂x^μ`` defined by a vielbein matrix.

    A tetrad sits on top of a coordinate frame; the **vielbein** is
    a square matrix whose row ``a`` gives the components of the
    tetrad vector ``e_a`` against the coordinate basis vectors:

    .. math::

        e_a = e_a{}^\mu \, \frac{\partial}{\partial x^\mu}.

    The frame derivative is the vielbein-weighted coordinate
    derivative

    .. math::

        e_a(f) = e_a{}^\mu \, \partial_\mu f.

    The Lie bracket structure constants ``γ^a_{bc}`` come from
    expanding ``[e_b, e_c] = γ^a_{bc} e_a`` in the coordinate
    basis; explicitly,

    .. math::

        [e_b, e_c]^\mu
        = e_b{}^\nu \, \partial_\nu e_c{}^\mu
        - e_c{}^\nu \, \partial_\nu e_b{}^\mu,
        \qquad
        \gamma^a{}_{bc} = (e^{-1})^a{}_\mu \, [e_b, e_c]^\mu,

    where ``e^{-1}`` is the inverse vielbein (dual coframe) matrix.

    Parameters
    ----------
    coord_frame
        The underlying :class:`CoordinateFrame` on whose chart the
        tetrad sits.
    vielbein
        ``dim × dim`` SymPy matrix; row ``a`` is ``e_a^μ``. Must
        match ``coord_frame.dim``.
    name
        Optional display name; defaults to
        ``f"tetrad({coord_frame.name})"``.

    Notes
    -----
    The vielbein must be **invertible** (so ``γ`` can be computed
    from the inverse). Singularity is detected lazily on the first
    :meth:`gamma` call. An identity vielbein reduces the tetrad to
    the underlying coordinate frame (γ ≡ 0).
    """

    __slots__ = ("dim", "name", "_coord_frame", "_vielbein", "_vielbein_inv")

    def __init__(
        self,
        coord_frame: CoordinateFrame,
        vielbein: Any,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(coord_frame, CoordinateFrame):
            raise TypeError(
                "Tetrad coord_frame must be a CoordinateFrame"
            )
        if not isinstance(vielbein, sp.Matrix):
            try:
                vielbein = sp.Matrix(vielbein)
            except Exception as exc:  # noqa: BLE001
                raise TypeError(
                    "Tetrad vielbein must be a SymPy Matrix or matrix-like"
                ) from exc
        if vielbein.shape != (coord_frame.dim, coord_frame.dim):
            raise ValueError(
                f"Tetrad vielbein shape {vielbein.shape} must match "
                f"coord_frame.dim={coord_frame.dim}"
            )
        self._coord_frame = coord_frame
        self._vielbein = vielbein
        self._vielbein_inv: sp.Matrix | None = None
        self.dim = coord_frame.dim
        self.name = (
            name
            if name is not None
            else f"tetrad({coord_frame.name})"
        )

    # ---- accessors ------------------------------------------------- #

    @property
    def coord_frame(self) -> CoordinateFrame:
        return self._coord_frame

    @property
    def vielbein(self) -> sp.Matrix:
        """Vielbein matrix; row ``a`` is ``e_a^μ``."""
        return self._vielbein

    @property
    def vielbein_inverse(self) -> sp.Matrix:
        """Inverse vielbein (dual coframe); cached after first compute."""
        if self._vielbein_inv is None:
            try:
                self._vielbein_inv = self._vielbein.inv()
            except (sp.matrices.exceptions.NonInvertibleMatrixError, ValueError) as exc:
                raise ValueError(
                    "Tetrad vielbein is singular; γ structure constants "
                    "and frame-component algebra require an invertible "
                    "vielbein."
                ) from exc
        return self._vielbein_inv

    def index_names(self) -> Tuple[str, ...]:
        # Tetrad indices are abstract (not coordinates); name them
        # numerically unless the coord frame's names happen to fit
        # (e.g. an identity vielbein where the tetrad coincides with
        # coordinates).
        return tuple(f"^{i}" for i in range(self.dim))

    # ---- protocol methods ----------------------------------------- #

    def derivative(self, expr: Any, a: int) -> sp.Expr:
        r"""Frame derivative ``e_a(expr) = e_a^μ ∂_μ expr``."""
        self._check_index("derivative", a)
        # Coerce to SymPy if needed
        try:
            sp_expr = sp.sympify(expr)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(
                "Tetrad.derivative requires a SymPy-coercible expression"
            ) from exc
        result: sp.Expr = sp.S.Zero
        for mu in range(self.dim):
            comp = self._vielbein[a, mu]
            if comp == 0:
                continue
            result += comp * self._coord_frame.derivative(sp_expr, mu)
        return result

    def gamma(self, a: int, b: int, c: int) -> sp.Expr:
        r"""Structure constant ``γ^a_{bc}`` from the vielbein."""
        for label, value in (("a", a), ("b", b), ("c", c)):
            self._check_index(f"gamma {label}", value)

        # Antisymmetry in (b, c)
        if b == c:
            return sp.Integer(0)
        if b > c:
            return -self.gamma(a, c, b)

        # Compute [e_b, e_c]^μ
        bracket_mu: list[sp.Expr] = []
        for mu in range(self.dim):
            term: sp.Expr = sp.S.Zero
            for nu in range(self.dim):
                # e_b^ν ∂_ν e_c^μ
                e_b_nu = self._vielbein[b, nu]
                if e_b_nu != 0:
                    term += e_b_nu * self._coord_frame.derivative(
                        self._vielbein[c, mu], nu
                    )
                # − e_c^ν ∂_ν e_b^μ
                e_c_nu = self._vielbein[c, nu]
                if e_c_nu != 0:
                    term -= e_c_nu * self._coord_frame.derivative(
                        self._vielbein[b, mu], nu
                    )
            bracket_mu.append(sp.simplify(term))

        # γ^a_{bc} = (e^{-1})^a_μ [e_b, e_c]^μ
        # The inverse vielbein has shape (dim, dim) where the row
        # is the upper index a and the column is the coord index μ.
        e_inv = self.vielbein_inverse
        result: sp.Expr = sp.S.Zero
        for mu in range(self.dim):
            result += e_inv[a, mu] * bracket_mu[mu]
        return sp.simplify(result)
