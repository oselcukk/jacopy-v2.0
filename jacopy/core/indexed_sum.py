r"""
Indexed sum :math:`\Sigma_d \, \mathrm{body}`, Faz 17.E.

A :class:`IndexedSum` is a bound-index summation over a discrete range
, the structural shape behind every "sum over frame indices" appearing
in Faz 17 (and beyond). The dummy is an :class:`Atom` (typically a
:class:`~jacopy.calculus.local_frame.FrameIndex` with kind ``"bound"``);
the range is an opaque domain object (typically a
:class:`~jacopy.calculus.local_frame.LocalFrame`); the body is any
:class:`Expr` referring to the dummy via the standard
``substitute_atom`` protocol.

Two principles drive the design:

1. **α-equivalence as ``==``**. Two :class:`IndexedSum`'s are equal
   when their ranges match and their bodies become structurally
   identical after a dummy renaming. Hash/equality use a depth-aware
   sentinel so nested binders whose original dummies happen to share
   a name still distinguish correctly. (The sentinel name lives in
   ``$bound_<depth>``.)
2. **No engine semantics here.** Σ-distribute, Wedge/Pairing pull-out,
   Kronecker contraction, and frame decomposition all live in the
   companion :mod:`jacopy.calculus.indexed_sum_axioms` module. This
   class is the inert structural container only.

The type carried in the ``range`` slot is opaque: this module never
dispatches on it. Callers may pass a :class:`LocalFrame`, a tuple of
basis indices, or anything else hashable. The default
:meth:`_repr_inner` consults a ``name`` attribute when present.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Atom, Expr


# --------------------------------------------------------------------- #
# IndexedSum                                                            #
# --------------------------------------------------------------------- #


class IndexedSum(Expr):
    r"""Bound-index summation :math:`\Sigma_d \, \mathrm{body}`."""

    __slots__ = ("_dummy", "_range", "_body")

    def __init__(self, dummy: Atom, range_: Any, body: Expr) -> None:
        if not isinstance(dummy, Atom):
            raise TypeError(
                "IndexedSum dummy must be an Atom (typically a FrameIndex)"
            )
        if not isinstance(body, Expr):
            raise TypeError("IndexedSum body must be an Expr")
        self._dummy = dummy
        self._range = range_
        self._body = body

    # ---- accessors -------------------------------------------------- #

    @property
    def dummy(self) -> Atom:
        return self._dummy

    @property
    def range_(self) -> Any:
        return self._range

    @property
    def body(self) -> Expr:
        return self._body

    # ---- Expr protocol --------------------------------------------- #

    @property
    def children(self) -> Tuple[Expr, ...]:
        # Body is the only Expr child. The dummy is structural metadata
        #, exposing it as a child would leak the binder into every
        # generic walk and break shadowing semantics.
        return (self._body,)

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "IndexedSum":
        if len(new_children) != 1:
            raise ValueError(
                "IndexedSum._rebuild expects exactly one child (the body)"
            )
        return IndexedSum(self._dummy, self._range, new_children[0])

    def _key(self) -> Any:
        return _alpha_canonical_key(self, 0)

    def _repr_inner(self) -> str:
        return (
            f"Σ_{self._dummy._repr_inner()}{self._range_repr_suffix()} "
            f"{self._body._repr_inner()}"
        )

    def _range_repr_suffix(self) -> str:
        if self._range is None:
            return ""
        name = getattr(self._range, "name", None)
        if name is not None:
            return f"∈{name}"
        return ""

    # ---- substitution + α-renaming --------------------------------- #

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        if self._dummy == dummy:
            # Shadowing: the inner binder hides the outer name.
            return self
        new_body = self._body.substitute_atom(dummy, target)
        if new_body is self._body:
            return self
        return IndexedSum(self._dummy, self._range, new_body)

    def with_dummy(self, new_dummy: Atom) -> "IndexedSum":
        """Return an α-equivalent :class:`IndexedSum` using ``new_dummy``."""
        if not isinstance(new_dummy, Atom):
            raise TypeError("new_dummy must be an Atom")
        if new_dummy == self._dummy:
            return self
        new_body = self._body.substitute_atom(self._dummy, new_dummy)
        return IndexedSum(new_dummy, self._range, new_body)

    def substitute_dummy_with(self, target: Expr) -> Expr:
        """Return ``body[dummy ↦ target]``.

        Used by the Kronecker-contraction rewrite, once the engine
        decides ``Σ_b δ(a, b) · f(b) → f(a)``, it asks for the body
        with the dummy replaced by the contracting free index.
        """
        if not isinstance(target, Expr):
            raise TypeError("target must be an Expr")
        return self._body.substitute_atom(self._dummy, target)


# --------------------------------------------------------------------- #
# α-canonical hashing / equality                                        #
# --------------------------------------------------------------------- #


_SENTINEL_PREFIX = "$bound"


def _make_sentinel(prototype: Atom, depth: int) -> Atom:
    """Mint a fresh bound-style dummy of the same class as ``prototype``.

    Uses depth-keyed naming so nested binders whose originals shared a
    name still get distinct sentinels.
    """
    name = f"{_SENTINEL_PREFIX}_{depth}"
    cls = type(prototype)
    # FrameIndex-style atoms accept (name, "bound"); plain Symbol-style
    # atoms accept just (name,). Try the richer form first.
    try:
        return cls(name, "bound")
    except TypeError:
        return cls(name)


def _alpha_canonical_key(node: IndexedSum, depth: int) -> Any:
    sentinel = _make_sentinel(node._dummy, depth)
    canon_body = node._body.substitute_atom(node._dummy, sentinel)
    return (
        "IndexedSum",
        _range_key(node._range),
        _walk_key(canon_body, depth + 1),
    )


def _range_key(range_: Any) -> Any:
    if range_ is None:
        return ("None",)
    name = getattr(range_, "name", None)
    if name is not None:
        return ("named", type(range_).__name__, name)
    # Fall back to repr, keeps the key hashable for anything Python
    # callers throw at us, at the cost of distinguishing two equal
    # ranges only by their repr() text.
    return ("opaque", type(range_).__name__, repr(range_))


def _walk_key(expr: Expr, depth: int) -> Any:
    """Hashable structural key, descending into nested IndexedSums with
    depth-aware sentinels.
    """
    if isinstance(expr, IndexedSum):
        return _alpha_canonical_key(expr, depth)
    if expr.is_atom:
        return ("Atom", type(expr).__name__, expr._key())
    return (
        "Compound",
        type(expr).__name__,
        tuple(_walk_key(c, depth) for c in expr.children),
    )


# --------------------------------------------------------------------- #
# Functional constructor                                                #
# --------------------------------------------------------------------- #


def indexed_sum(dummy: Atom, range_: Any, body: Expr) -> IndexedSum:
    """Functional constructor mirroring ``multi_eval`` / ``pairing``."""
    return IndexedSum(dummy, range_, body)


# --------------------------------------------------------------------- #
# dummy-in predicate                                                     #
# --------------------------------------------------------------------- #


def dummy_in(expr: Expr, dummy: Atom) -> bool:
    """True when ``dummy`` occurs (free) in ``expr``.

    Respects :class:`IndexedSum` shadowing: an inner binder using the
    same dummy hides the outer occurrence. Detects hidden occurrences
    inside frame-aware atoms (e.g. a :class:`~jacopy.calculus.local_frame.FrameVectorField`
    whose private ``_idx`` slot equals ``dummy``) by probing with a
    distinct sentinel atom of the same class via the
    :meth:`Expr.substitute_atom` protocol.
    """
    if not isinstance(dummy, Atom):
        raise TypeError("dummy must be an Atom")
    return _dummy_appears(expr, dummy)


def _dummy_appears(expr: Expr, dummy: Atom) -> bool:
    if expr == dummy:
        return True
    if isinstance(expr, IndexedSum):
        if expr._dummy == dummy:
            return False  # shadowed
        return _dummy_appears(expr._body, dummy)
    if expr.is_atom:
        probe = _make_distinct_probe(dummy)
        if probe is None:
            return False
        return expr.substitute_atom(dummy, probe) is not expr
    return any(_dummy_appears(c, dummy) for c in expr.children)


def _make_distinct_probe(dummy: Atom) -> Atom:
    """Build an Atom of the same class as ``dummy`` but with a fresh name."""
    cls = type(dummy)
    for name in ("$_probe_", "$_probe2_", "$_probe3_"):
        try:
            probe = cls(name, "bound")
        except TypeError:
            try:
                probe = cls(name)
            except TypeError:
                return None
        if probe != dummy:
            return probe
    return None
