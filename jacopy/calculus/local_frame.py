r"""
Local frames + dual coframes, Faz 17.A.

A *local frame* :math:`(X_a)` is an ordered tuple of vector fields whose
values span the tangent space at every point in the chart's domain.
Each frame comes with a *dual coframe* :math:`(e^a)` of 1-forms,
characterised by the duality

.. math::

    e^a(X_b) = \delta^a_b.

The Cartan structure equations are mechanised on top of this data
(see :class:`~jacopy.library.cartan_structure.CartanStructureProblem`).
This module ships the smallest piece of it:

* :class:`FrameIndex`, an index atom, ``free`` (an outer hypothesis
  variable) or ``bound`` (an :class:`IndexedSum` dummy in 17.E).
* :class:`KroneckerDelta`, the index-pair node ``δ^a_b``. Auto-collapses
  to :data:`~jacopy.core.expr.One` when both indices are *free* and
  carry the same name; bound-index reductions are left for the
  :class:`IndexedSum` contraction rule (17.E).
* :class:`LocalFrame`, library wrapper bundling a frame name, an
  optional dimension, a vector-field display symbol, and a coframe
  display symbol. Used as a factory for :class:`FrameVectorField` and
  :class:`FrameCovector` instances.
* :class:`FrameVectorField`, a :class:`~jacopy.algebra.derivation.Derivation`
  subclass tagged with ``(frame_name, idx)`` so it interoperates with
  every existing engine pass that already speaks ``Derivation``
  (∇, Lie, interior, Pairing, …).
* :class:`FrameCovector`, an :class:`Atom` for the dual basis 1-form.
* :class:`FramePairingDualityDefinition`, the engine rule
  ``⟨e^a, X_b⟩ → δ^a_b`` scoped to a specific frame so two frames in
  the same proof never cross-fire.

The duality axiom is the only engine rule introduced here, all other
machinery (Pairing C∞-linearity, Sum/Neg distribution, …) is reused
unchanged from Faz 12-13.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Atom, Expr, One
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# FrameIndex atom                                                       #
# --------------------------------------------------------------------- #


class FrameIndex(Atom):
    """A frame index ``a``, ``b``, ``c``, … carrying a free/bound flag.

    *Free* indices are outer hypothesis variables: they appear in the
    statement of an identity (``ω^a_b``, ``T^a``) and remain visible in
    the result. *Bound* indices are :class:`IndexedSum` dummies; they
    are alpha-fresh on each construction and only meaningful inside the
    sum that binds them. The two kinds compare unequal even when their
    names coincide, so a serialised proof transcript never confuses a
    hypothesis index with a dummy.
    """

    __slots__ = ("_name", "_kind")

    KINDS: Tuple[str, ...] = ("free", "bound")

    def __init__(self, name: str, kind: str = "free") -> None:
        if not isinstance(name, str):
            raise TypeError("FrameIndex name must be a str")
        if not name:
            raise ValueError("FrameIndex name must be non-empty")
        if kind not in self.KINDS:
            raise ValueError(
                f"FrameIndex kind must be one of {self.KINDS}, got {kind!r}"
            )
        self._name = name
        self._kind = kind

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def is_free(self) -> bool:
        return self._kind == "free"

    @property
    def is_bound(self) -> bool:
        return self._kind == "bound"

    def _key(self) -> Any:
        return (self._name, self._kind)

    def _repr_inner(self) -> str:
        # Bound indices print with a hat to mark the dummy at the
        # transcript level; free indices print as-is.
        if self._kind == "bound":
            return f"{self._name}̂"
        return self._name


def _coerce_index(idx: Union[FrameIndex, str]) -> FrameIndex:
    """Accept a :class:`FrameIndex` or a string (treated as free)."""
    if isinstance(idx, FrameIndex):
        return idx
    if isinstance(idx, str):
        return FrameIndex(idx, "free")
    raise TypeError(
        f"frame index must be a FrameIndex or str, got "
        f"{type(idx).__name__}"
    )


# --------------------------------------------------------------------- #
# KroneckerDelta                                                        #
# --------------------------------------------------------------------- #


class KroneckerDelta(Expr):
    r"""The index-pair node :math:`\delta^i_j` over :class:`FrameIndex`.

    Auto-simplifies to :data:`~jacopy.core.expr.One` in the constructor
    when both indices are *free* and carry the same name, this
    captures the ``δ^a_a = 1`` reduction at a free contraction without
    needing an extra engine pass. Bound-index δ's stay inert; their
    reduction (``Σ_b δ^a_b · f(b) → f(a)``) is the
    :class:`IndexedSumKroneckerContract` rule introduced in 17.E and
    only fires inside an :class:`IndexedSum`.
    """

    __slots__ = ("_i", "_j")

    def __new__(cls, i: FrameIndex, j: FrameIndex):
        if not isinstance(i, FrameIndex):
            raise TypeError(
                "KroneckerDelta first argument must be a FrameIndex"
            )
        if not isinstance(j, FrameIndex):
            raise TypeError(
                "KroneckerDelta second argument must be a FrameIndex"
            )
        if i.is_free and j.is_free and i.name == j.name:
            return One
        inst = super().__new__(cls)
        inst._i = i
        inst._j = j
        return inst

    def __init__(self, i: FrameIndex, j: FrameIndex) -> None:
        # All state assigned in __new__; nothing further to do here.
        pass

    @property
    def i(self) -> FrameIndex:
        return self._i

    @property
    def j(self) -> FrameIndex:
        return self._j

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._i, self._j)

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> Expr:
        if len(new_children) != 2:
            raise ValueError(
                "KroneckerDelta._rebuild expects exactly two children"
            )
        i, j = new_children
        if not isinstance(i, FrameIndex) or not isinstance(j, FrameIndex):
            raise TypeError(
                "KroneckerDelta._rebuild children must be FrameIndex"
            )
        return KroneckerDelta(i, j)

    def _key(self) -> Any:
        return (self._i, self._j)

    def _repr_inner(self) -> str:
        return (
            f"δ^{self._i._repr_inner()}_{self._j._repr_inner()}"
        )


# --------------------------------------------------------------------- #
# Frame VF and covector                                                 #
# --------------------------------------------------------------------- #


class FrameVectorField(Derivation):
    """Frame basis vector field ``X_a`` from a :class:`LocalFrame`.

    Subclasses :class:`Derivation` (degree 0) so every existing pass
    that walks ``Derivation`` shapes, ∇, ι, L, Pairing, picks the
    frame VFs up automatically. Equality includes the *frame name* so
    that two frames sharing only their VF symbol stay distinguishable.
    """

    __slots__ = ("_frame", "_idx")

    def __init__(self, frame: "LocalFrame", idx: FrameIndex) -> None:
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "FrameVectorField first argument must be a LocalFrame"
            )
        if not isinstance(idx, FrameIndex):
            raise TypeError(
                "FrameVectorField second argument must be a FrameIndex"
            )
        super().__init__(f"{frame.vf_symbol}_{idx.name}", 0)
        self._frame = frame
        self._idx = idx

    @property
    def frame(self) -> "LocalFrame":
        return self._frame

    @property
    def frame_name(self) -> str:
        return self._frame.name

    @property
    def idx(self) -> FrameIndex:
        return self._idx

    def _key(self) -> Any:
        return (self._frame.name, self._idx)

    def _repr_inner(self) -> str:
        return f"{self._frame.vf_symbol}_{self._idx._repr_inner()}"

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        if (
            isinstance(dummy, FrameIndex)
            and isinstance(target, FrameIndex)
            and self._idx == dummy
        ):
            return FrameVectorField(self._frame, target)
        return self


class FrameCovector(Atom):
    """Frame basis 1-form ``e^a`` from a :class:`LocalFrame`.

    Stored as an opaque atom with ``(frame, idx)`` payload; its
    interaction with vector fields is mediated solely by
    :class:`FramePairingDualityDefinition` (Pairing-level rewrite).
    Faz 17.C will introduce :class:`ConnectionForm` etc., which are
    1-forms in their own right and carry coframe symbols implicitly.
    """

    __slots__ = ("_frame", "_idx")

    def __init__(self, frame: "LocalFrame", idx: FrameIndex) -> None:
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "FrameCovector first argument must be a LocalFrame"
            )
        if not isinstance(idx, FrameIndex):
            raise TypeError(
                "FrameCovector second argument must be a FrameIndex"
            )
        self._frame = frame
        self._idx = idx

    @property
    def frame(self) -> "LocalFrame":
        return self._frame

    @property
    def frame_name(self) -> str:
        return self._frame.name

    @property
    def idx(self) -> FrameIndex:
        return self._idx

    def _key(self) -> Any:
        return (self._frame.name, self._idx)

    def _repr_inner(self) -> str:
        return f"{self._frame.coframe_symbol}^{self._idx._repr_inner()}"

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        if (
            isinstance(dummy, FrameIndex)
            and isinstance(target, FrameIndex)
            and self._idx == dummy
        ):
            return FrameCovector(self._frame, target)
        return self


# --------------------------------------------------------------------- #
# LocalFrame wrapper                                                    #
# --------------------------------------------------------------------- #


class LocalFrame:
    """Library wrapper bundling a frame's identity and display symbols.

    The frame is uniquely identified by ``name``; ``dim`` is optional,
    when ``None``, the frame is treated as having a *symbolic*
    dimension, which is the mode Faz 17 uses for its proofs. Two
    :class:`LocalFrame` instances with the same ``(name, dim,
    vf_symbol, coframe_symbol)`` tuple compare equal so that callers
    constructing a frame in two notebook cells don't accidentally end
    up with two distinct objects.

    The wrapper itself is **not** an :class:`Expr`; it is a factory
    plus an identity carrier. The :class:`Expr` instances live behind
    :meth:`X` and :meth:`coframe`.
    """

    __slots__ = ("_name", "_dim", "_vf_symbol", "_coframe_symbol")

    def __init__(
        self,
        name: str,
        *,
        dim: Optional[int] = None,
        vf_symbol: str = "X",
        coframe_symbol: str = "e",
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("LocalFrame name must be a str")
        if not name:
            raise ValueError("LocalFrame name must be non-empty")
        if dim is not None:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise TypeError(
                    "LocalFrame dim must be a positive int or None"
                )
            if dim <= 0:
                raise ValueError(
                    "LocalFrame dim must be a positive int or None"
                )
        if not isinstance(vf_symbol, str) or not vf_symbol:
            raise ValueError("LocalFrame vf_symbol must be a non-empty str")
        if not isinstance(coframe_symbol, str) or not coframe_symbol:
            raise ValueError(
                "LocalFrame coframe_symbol must be a non-empty str"
            )
        self._name = name
        self._dim = dim
        self._vf_symbol = vf_symbol
        self._coframe_symbol = coframe_symbol

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def vf_symbol(self) -> str:
        return self._vf_symbol

    @property
    def coframe_symbol(self) -> str:
        return self._coframe_symbol

    def index(self, name: str, *, bound: bool = False) -> FrameIndex:
        """Build a :class:`FrameIndex` against this frame's index domain."""
        return FrameIndex(name, "bound" if bound else "free")

    def X(self, idx: Union[FrameIndex, str]) -> FrameVectorField:
        """Build the frame VF ``X_idx``."""
        return FrameVectorField(self, _coerce_index(idx))

    def coframe(self, idx: Union[FrameIndex, str]) -> FrameCovector:
        """Build the dual basis 1-form ``e^idx``."""
        return FrameCovector(self, _coerce_index(idx))

    def duality_definition(self) -> "FramePairingDualityDefinition":
        """Build the engine rule ``⟨e^a, X_b⟩ → δ^a_b`` for this frame."""
        return FramePairingDualityDefinition(self)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        return (
            isinstance(other, LocalFrame)
            and self._name == other._name
            and self._dim == other._dim
            and self._vf_symbol == other._vf_symbol
            and self._coframe_symbol == other._coframe_symbol
        )

    def __hash__(self) -> int:
        return hash(
            (self._name, self._dim, self._vf_symbol, self._coframe_symbol)
        )

    def __repr__(self) -> str:
        bits = [repr(self._name)]
        if self._dim is not None:
            bits.append(f"dim={self._dim}")
        if self._vf_symbol != "X":
            bits.append(f"vf_symbol={self._vf_symbol!r}")
        if self._coframe_symbol != "e":
            bits.append(f"coframe_symbol={self._coframe_symbol!r}")
        return f"LocalFrame({', '.join(bits)})"


def local_frame(
    name: str = "X",
    *,
    dim: Optional[int] = None,
    vf_symbol: str = "X",
    coframe_symbol: str = "e",
) -> LocalFrame:
    """Functional constructor for :class:`LocalFrame`."""
    return LocalFrame(
        name,
        dim=dim,
        vf_symbol=vf_symbol,
        coframe_symbol=coframe_symbol,
    )


# --------------------------------------------------------------------- #
# Duality axiom                                                         #
# --------------------------------------------------------------------- #


class FramePairingDualityDefinition(Definition):
    r"""Engine rule ``⟨e^a, X_b⟩ → δ^a_b`` scoped to one frame.

    Fires only when both sides of the pairing belong to the *same*
    :class:`LocalFrame` (compared by name); two frames in the same proof
    never cross-fire. The right-hand side is :class:`KroneckerDelta`,
    which itself collapses to :data:`~jacopy.core.expr.One` when the
    indices match (see :class:`KroneckerDelta.__new__`). Pairings whose
    slots wrap :class:`Sum` / :class:`Neg` are first unwrapped by
    :class:`PairingLinearityDefinition` (Faz 13.B); this rule then
    applies on each leaf pairing.
    """

    def __init__(self, frame: LocalFrame) -> None:
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "FramePairingDualityDefinition requires a LocalFrame"
            )
        self._frame = frame
        self.name = f"Frame duality [{frame.name}]"

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        if not isinstance(expr.alpha, FrameCovector):
            return False
        if not isinstance(expr.X, FrameVectorField):
            return False
        return (
            expr.alpha.frame_name == self._frame.name
            and expr.X.frame_name == self._frame.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, Pairing)
        alpha = expr.alpha
        X = expr.X
        assert isinstance(alpha, FrameCovector)
        assert isinstance(X, FrameVectorField)
        return KroneckerDelta(alpha.idx, X.idx)
