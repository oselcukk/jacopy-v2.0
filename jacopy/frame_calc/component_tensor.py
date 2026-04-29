r"""
Component-tensor wrappers — Stage C.

A *component tensor* stores the frame components of a
``(q, r)``-tensor as a SymPy ``MutableDenseNDimArray`` of shape
``(dim,) * (q + r)``. The wrapper carries the frame, signature, and
ergonomic helpers (``__getitem__``, ``is_zero``, ``simplify``,
``matrix`` view for rank-2). It is **not** an
:class:`~jacopy.core.expr.Expr` — it sits one level above the
expression layer, holding multiple expressions in an indexed
container.

Three typed subclasses cover the layer's needs at this stage:

* :class:`ComponentMetric` — symmetric ``(0, 2)`` tensor ``g_{ab}``.
  Carries :meth:`inverse` (SymPy ``Matrix.inv()``) and :meth:`det`.
* :class:`ComponentMetricInverse` — ``(2, 0)`` tensor ``g^{ab}``.
  Same shape as the metric; the type itself signals which slot
  position is upper.
* :class:`ComponentConnection` — ``(1, 2)`` tensor ``Γ^a_{bc}``.
  Used for Christoffel symbols and any user-supplied connection.

Stage-C scope is **concrete** entries (SymPy expressions). Abstract
entries — produced by :class:`AbstractFrame` upstream — slot in
naturally for storage and indexing, but :meth:`ComponentMetric.inverse`
needs SymPy semantics it cannot apply to jacopy ``Expr`` atoms; the
abstract path is deferred to Stage D, where the Koszul-formula
implementation will fold any opaque ``g^{ab}`` symbols it needs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc

from jacopy.core.expr import Expr
from jacopy.frame_calc.frame import (
    AbstractFrame,
    CoordinateFrame,
    Frame,
)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _as_array(
    components: Any, expected_shape: Tuple[int, ...]
) -> sp.MutableDenseNDimArray:
    """Coerce ``components`` to a SymPy ``MutableDenseNDimArray`` of
    shape ``expected_shape``.

    Accepts a ``sp.Matrix``, ``sp.ImmutableDenseNDimArray``,
    ``sp.MutableDenseNDimArray``, or a nested Python list/tuple.
    """
    if isinstance(
        components,
        (sp.MutableDenseNDimArray, sp.ImmutableDenseNDimArray),
    ):
        arr = sp.MutableDenseNDimArray(components)
    elif isinstance(components, sp.Matrix):
        if len(expected_shape) != 2:
            raise ValueError(
                f"Cannot wrap an sp.Matrix as a rank-{len(expected_shape)} "
                "tensor — Matrix is rank-2 only"
            )
        arr = sp.MutableDenseNDimArray(components.tolist())
    else:
        # Nested list / tuple
        arr = sp.MutableDenseNDimArray(components)
    if tuple(arr.shape) != expected_shape:
        raise ValueError(
            f"component tensor shape mismatch: got {tuple(arr.shape)}, "
            f"expected {expected_shape}"
        )
    return arr


def _all_indices(shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    """Yield every multi-index (i, j, ...) for a tensor of given shape."""
    if not shape:
        yield ()
        return
    head, *tail = shape
    for i in range(head):
        for rest in _all_indices(tuple(tail)):
            yield (i,) + rest


# --------------------------------------------------------------------- #
# ComponentTensor — base class                                          #
# --------------------------------------------------------------------- #


class ComponentTensor:
    """Frame-component representation of a ``(q, r)``-tensor.

    Parameters
    ----------
    frame
        The :class:`Frame` whose dimension the tensor's shape matches.
    components
        Initial entries — accepts ``sp.Matrix`` (rank 2 only),
        ``sp.MutableDenseNDimArray`` /
        ``sp.ImmutableDenseNDimArray``, or nested list/tuple.
    signature
        ``(q, r)`` — number of upper and lower indices. ``q + r``
        must equal the rank of ``components``.

    Notes
    -----
    The tensor itself does not know which index slots are
    contravariant vs covariant beyond the ``(q, r)`` count. Subclasses
    (:class:`ComponentMetric`, :class:`ComponentConnection`) encode
    that meaning structurally.
    """

    __slots__ = ("_frame", "_components", "_signature")

    def __init__(
        self,
        frame: Frame,
        components: Any,
        *,
        signature: Tuple[int, int],
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError(
                "ComponentTensor frame must be a Frame instance"
            )
        if (
            not isinstance(signature, tuple)
            or len(signature) != 2
            or not all(isinstance(s, int) and s >= 0 for s in signature)
        ):
            raise ValueError(
                "ComponentTensor signature must be (q, r) with q, r ≥ 0 ints"
            )
        rank = signature[0] + signature[1]
        expected_shape = (frame.dim,) * rank
        self._frame = frame
        self._components = _as_array(components, expected_shape)
        self._signature = signature

    # ---- accessors ------------------------------------------------- #

    @property
    def frame(self) -> Frame:
        return self._frame

    @property
    def signature(self) -> Tuple[int, int]:
        return self._signature

    @property
    def rank(self) -> int:
        return self._signature[0] + self._signature[1]

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._components.shape)

    @property
    def components(self) -> sp.MutableDenseNDimArray:
        """Underlying SymPy ``Array``. Treat as immutable (copy on edit)."""
        return self._components

    # ---- indexing -------------------------------------------------- #

    def __getitem__(self, idx: Any) -> Any:
        """Component access.

        ``T[a, b, c]`` returns the entry at multi-index ``(a, b, c)``.
        Out-of-range indices raise :class:`IndexError`.
        """
        if isinstance(idx, int):
            idx = (idx,)
        idx = tuple(idx)
        if len(idx) != self.rank:
            raise IndexError(
                f"{type(self).__name__}: expected {self.rank} indices, "
                f"got {len(idx)}"
            )
        for label, value in enumerate(idx):
            if not isinstance(value, int):
                raise TypeError(
                    f"{type(self).__name__}: index {label} must be int"
                )
            if not 0 <= value < self._frame.dim:
                raise IndexError(
                    f"{type(self).__name__}: index {label} = {value} "
                    f"out of range for dim={self._frame.dim}"
                )
        return self._components[idx]

    # ---- iteration / inspection ----------------------------------- #

    def all_indices(self) -> Iterator[Tuple[int, ...]]:
        """Yield every multi-index of this tensor in row-major order."""
        return _all_indices(self.shape)

    def nonzero_components(
        self, *, simplify: bool = True
    ) -> Dict[Tuple[int, ...], Any]:
        """Return a dict of nonzero entries keyed by multi-index.

        Each value is run through :func:`sympy.simplify` (when
        ``simplify=True``, the default) before the zero check, so an
        algebraically-zero entry that hasn't been simplified yet is
        recognised.
        """
        out: Dict[Tuple[int, ...], Any] = {}
        for idx in self.all_indices():
            val = self._components[idx]
            if simplify:
                try:
                    val_s = sp.simplify(val)
                except (TypeError, AttributeError):
                    val_s = val
            else:
                val_s = val
            if val_s != 0:
                out[idx] = val_s
        return out

    def is_zero(self, *, simplify: bool = True) -> bool:
        """Check whether every component is mathematically zero.

        With ``simplify=True`` (default), each entry is run through
        :func:`sympy.simplify` and (for trigonometric expressions)
        :func:`sympy.trigsimp` before the zero check; this catches
        the common ``sin²+cos²−1`` pattern that survives plain
        evaluation. Set ``simplify=False`` for a quick literal check.
        """
        for idx in self.all_indices():
            val = self._components[idx]
            if simplify:
                try:
                    val = sp.simplify(val)
                    if val != 0:
                        val = sp.trigsimp(val)
                except (TypeError, AttributeError):
                    pass
            if val != 0:
                return False
        return True

    def simplify(self) -> "ComponentTensor":
        """Return a copy with every component simplified."""
        new_arr = sp.MutableDenseNDimArray(self._components)
        for idx in self.all_indices():
            try:
                new_arr[idx] = sp.simplify(self._components[idx])
            except (TypeError, AttributeError):
                pass  # abstract entries (jacopy Expr) — leave alone
        return self._rebuild_from_array(new_arr)

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "ComponentTensor":
        """Hook for subclasses to preserve their typed identity."""
        return ComponentTensor(
            self._frame, new_arr, signature=self._signature
        )

    # ---- equality + repr ------------------------------------------ #

    def equals(self, other: object, *, simplify: bool = True) -> bool:
        """Structural equality between two component tensors.

        Compares frame, signature, and shape; then each component
        through :func:`sympy.simplify` (when ``simplify=True``).
        """
        if not isinstance(other, ComponentTensor):
            return False
        if self._frame != other._frame:
            return False
        if self._signature != other._signature:
            return False
        if self.shape != other.shape:
            return False
        for idx in self.all_indices():
            a, b = self._components[idx], other._components[idx]
            if simplify:
                try:
                    if sp.simplify(a - b) != 0:
                        return False
                except (TypeError, AttributeError):
                    if a != b:
                        return False
            else:
                if a != b:
                    return False
        return True

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"frame={self._frame.name!r}, "
            f"signature={self._signature}, "
            f"shape={self.shape})"
        )


# --------------------------------------------------------------------- #
# ComponentMetric — symmetric (0, 2) tensor                             #
# --------------------------------------------------------------------- #


class ComponentMetric(ComponentTensor):
    r"""Symmetric ``(0, 2)`` tensor ``g_{ab} = g(e_a, e_b)``.

    Parameters
    ----------
    frame
        The :class:`Frame` against which the metric is given.
    matrix
        SymPy ``Matrix`` (or anything :func:`_as_array` accepts) of
        shape ``(frame.dim, frame.dim)``. Symmetry is checked at
        construction (best-effort: when an entry is a SymPy
        expression, ``sp.simplify(g[a,b] − g[b,a]) == 0`` is
        required).

    Notes
    -----
    On a concrete frame (:class:`CoordinateFrame`,
    :class:`Tetrad`) the entries are SymPy expressions — full
    arithmetic available. On an :class:`AbstractFrame` the entries
    may be jacopy :class:`~jacopy.core.expr.Expr` atoms; storage and
    indexing work, but :meth:`inverse` requires SymPy semantics and
    raises ``NotImplementedError`` until Stage D fills in the
    abstract path.
    """

    __slots__ = ()

    def __init__(self, frame: Frame, matrix: Any) -> None:
        super().__init__(frame, matrix, signature=(0, 2))
        # Symmetry check (best-effort)
        for a in range(frame.dim):
            for b in range(a + 1, frame.dim):
                diff = self._components[a, b] - self._components[b, a]
                try:
                    diff = sp.simplify(diff)
                except (TypeError, AttributeError):
                    pass
                if diff != 0:
                    raise ValueError(
                        f"ComponentMetric is not symmetric: "
                        f"g[{a},{b}] = {self._components[a, b]} ≠ "
                        f"g[{b},{a}] = {self._components[b, a]}"
                    )

    # ---- typed accessors ------------------------------------------ #

    def matrix(self) -> sp.Matrix:
        """Return the underlying components as a SymPy ``Matrix``."""
        return sp.Matrix(self._components.tolist())

    def det(self) -> Any:
        """Compute ``det g``.

        Concrete (SymPy) entries: returns a SymPy expression.
        Abstract entries: raises :class:`NotImplementedError`
        (deferred to Stage D).
        """
        if isinstance(self._frame, AbstractFrame):
            # Could synthesise an opaque Det atom here; defer to Stage D.
            raise NotImplementedError(
                "ComponentMetric.det on AbstractFrame is deferred to "
                "Stage D, where the abstract Koszul path needs it."
            )
        return self.matrix().det()

    def inverse(self) -> "ComponentMetricInverse":
        r"""Compute ``g^{ab}`` — the (2, 0) inverse metric.

        Concrete (SymPy) entries: returns a :class:`ComponentMetricInverse`
        whose matrix is :meth:`sympy.Matrix.inv` of this metric. Each
        entry is run through :func:`sympy.simplify` for readability.

        Abstract entries: raises :class:`NotImplementedError`. The
        Levi-Civita stage (D) will introduce opaque ``g^{ab}`` atoms
        for the abstract path; until then, abstract-mode inverse is
        unsupported.
        """
        if isinstance(self._frame, AbstractFrame):
            raise NotImplementedError(
                "ComponentMetric.inverse on AbstractFrame is deferred "
                "to Stage D, where opaque g^{ab} atoms will be "
                "introduced for the symbolic Koszul-formula path."
            )
        inv = self.matrix().inv()
        # Apply simplify component-wise for cleaner output
        inv_simplified = sp.Matrix(inv.shape[0], inv.shape[1], lambda i, j: sp.simplify(inv[i, j]))
        return ComponentMetricInverse(self._frame, inv_simplified)

    # ---- structure preservation ----------------------------------- #

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "ComponentMetric":
        # Bypass symmetry re-check on rebuild — the source is a
        # ComponentMetric so symmetry was verified at construction.
        out = object.__new__(ComponentMetric)
        ComponentTensor.__init__(
            out, self._frame, new_arr, signature=(0, 2)
        )
        return out


# --------------------------------------------------------------------- #
# ComponentMetricInverse — (2, 0) tensor                                #
# --------------------------------------------------------------------- #


class ComponentMetricInverse(ComponentTensor):
    r"""``(2, 0)`` tensor ``g^{ab}``.

    Functionally identical to a generic ``(2, 0)`` tensor; the
    type signals "this is the inverse metric". Constructed by
    :meth:`ComponentMetric.inverse`; rarely instantiated directly.
    """

    __slots__ = ()

    def __init__(self, frame: Frame, matrix: Any) -> None:
        super().__init__(frame, matrix, signature=(2, 0))

    def matrix(self) -> sp.Matrix:
        """Return the underlying components as a SymPy ``Matrix``."""
        return sp.Matrix(self._components.tolist())

    def inverse(self) -> ComponentMetric:
        r"""``(g^{ab})^{-1} = g_{ab}`` — round-trip back to the metric."""
        if isinstance(self._frame, AbstractFrame):
            raise NotImplementedError(
                "ComponentMetricInverse.inverse on AbstractFrame is "
                "deferred to Stage D."
            )
        inv = self.matrix().inv()
        inv_simplified = sp.Matrix(inv.shape[0], inv.shape[1], lambda i, j: sp.simplify(inv[i, j]))
        return ComponentMetric(self._frame, inv_simplified)

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "ComponentMetricInverse":
        out = object.__new__(ComponentMetricInverse)
        ComponentTensor.__init__(
            out, self._frame, new_arr, signature=(2, 0)
        )
        return out


# --------------------------------------------------------------------- #
# ComponentConnection — (1, 2) tensor of Christoffel symbols            #
# --------------------------------------------------------------------- #


class ComponentConnection(ComponentTensor):
    r"""``(1, 2)`` tensor ``Γ^a_{bc}`` — connection coefficients.

    Index convention: ``connection[a, b, c]`` is ``Γ^a_{bc}``, with
    ``a`` upper and ``(b, c)`` lower. The plan's Stage D will produce
    a :class:`LeviCivitaConnection` subclass that also carries
    derivation chains; user-supplied connections (any tabulated
    Christoffel) instantiate this class directly.
    """

    __slots__ = ()

    def __init__(self, frame: Frame, christoffel: Any) -> None:
        super().__init__(frame, christoffel, signature=(1, 2))

    def upper(self, a: int, b: int, c: int) -> Any:
        """Alias for ``self[a, b, c]`` matching the index notation."""
        return self[a, b, c]

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "ComponentConnection":
        out = object.__new__(ComponentConnection)
        ComponentTensor.__init__(
            out, self._frame, new_arr, signature=(1, 2)
        )
        return out
