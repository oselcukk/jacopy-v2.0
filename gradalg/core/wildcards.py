"""
Pattern matching via wildcards.

Wildcards are Expr atoms that stand in for arbitrary subtrees. They
let rewrite rules be written as patterns: ``d(?A + ?B) -> d(?A) + d(?B)``
for linearity, ``d(?A * ?B) -> d(?A)*?B + ?A*d(?B)`` for Leibniz,
``[?X, [?Y, ?Z]] + cyclic -> 0`` for Jacobi, and so on.

Two wildcard kinds:

* :class:`Wildcard` — a single hole that matches one Expr. It may
  carry a *type filter*: a :class:`Property` class that the candidate
  must be registered with. Type-filtered matching requires a
  :class:`PropertyRegistry` to be passed to :func:`match`.

* :class:`SeqWildcard` — a sequence hole that matches zero or more
  consecutive children of a Sum or Product. Only one is permitted
  per Sum/Product level; it binds to a tuple. Sequence wildcards are
  what let Leibniz generalise to n factors without the rule author
  having to enumerate arities.

Matching is structural and order-preserving — it does not reorder
Sum or Product children. Commutative matching is the job of the
algorithms layer, which canonicalizes expressions before invoking
:func:`match`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, Union

from gradalg.core.expr import Atom, Expr
from gradalg.core.properties import Property
from gradalg.core.registry import PropertyRegistry


# --------------------------------------------------------------------- #
# Wildcard nodes                                                         #
# --------------------------------------------------------------------- #


class Wildcard(Atom):
    """Single-hole pattern atom.

    A wildcard with ``name`` binds that name to whatever subtree it
    matches. If the same name appears twice in a pattern, both
    occurrences must match *structurally equal* targets.
    """

    __slots__ = ("_name", "_type_filter")

    def __init__(
        self,
        name: str,
        type_filter: Optional[Type[Property]] = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("Wildcard name must be a str")
        if not name:
            raise ValueError("Wildcard name must be non-empty")
        if type_filter is not None:
            if not (isinstance(type_filter, type) and issubclass(type_filter, Property)):
                raise TypeError("type_filter must be a Property subclass")
        self._name = name
        self._type_filter = type_filter

    @property
    def name(self) -> str:
        return self._name

    @property
    def type_filter(self) -> Optional[Type[Property]]:
        return self._type_filter

    def _key(self) -> Any:
        return (self._name, self._type_filter)

    def _repr_inner(self) -> str:
        if self._type_filter is None:
            return f"?{self._name}"
        return f"?{self._name}:{self._type_filter.__name__}"


class SeqWildcard(Expr):
    """Sequence wildcard matching a run of Sum/Product children.

    Usage is restricted to being a direct child of :class:`Sum` or
    :class:`Product`. At most one per level: two sequence wildcards in
    the same Sum would make the match ambiguous without a disambiguation
    strategy, and that complexity doesn't pay for itself yet.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("SeqWildcard name must be a str")
        if not name:
            raise ValueError("SeqWildcard name must be non-empty")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def children(self) -> Tuple[Expr, ...]:
        return ()

    def _key(self) -> Any:
        return self._name

    def _repr_inner(self) -> str:
        return f"?*{self._name}"


# --------------------------------------------------------------------- #
# Matching                                                              #
# --------------------------------------------------------------------- #

# Bindings map names to either a single Expr (Wildcard) or a tuple of
# Exprs (SeqWildcard). Public callers treat this as opaque.
Binding = Union[Expr, Tuple[Expr, ...]]
Bindings = Dict[str, Binding]


def match(
    pattern: Expr,
    target: Expr,
    registry: Optional[PropertyRegistry] = None,
) -> Optional[Bindings]:
    """Attempt to match ``target`` against ``pattern``.

    Returns a fresh bindings dict on success, or ``None`` on failure.

    If ``pattern`` contains type-filtered wildcards, ``registry`` must
    be supplied; otherwise the filter is treated as a failure. This is
    deliberate — silently ignoring a type filter would mask bugs.
    """
    bindings: Bindings = {}
    if _match(pattern, target, bindings, registry):
        return bindings
    return None


def _match(
    pattern: Expr,
    target: Expr,
    bindings: Bindings,
    registry: Optional[PropertyRegistry],
) -> bool:
    if isinstance(pattern, Wildcard):
        return _bind_wildcard(pattern, target, bindings, registry)
    if isinstance(pattern, SeqWildcard):
        # Outside a Sum/Product context, bind a single target.
        return _bind_single(pattern.name, target, bindings)
    if type(pattern) is not type(target):
        return False
    if pattern.is_atom:
        return pattern == target

    return _match_children(pattern.children, target.children, bindings, registry)


def _match_children(
    pchildren: Tuple[Expr, ...],
    tchildren: Tuple[Expr, ...],
    bindings: Bindings,
    registry: Optional[PropertyRegistry],
) -> bool:
    seq_idx: Optional[int] = None
    for i, p in enumerate(pchildren):
        if isinstance(p, SeqWildcard):
            if seq_idx is not None:
                raise ValueError(
                    "More than one SeqWildcard at the same level is not allowed"
                )
            seq_idx = i

    if seq_idx is None:
        if len(pchildren) != len(tchildren):
            return False
        for pc, tc in zip(pchildren, tchildren):
            if not _match(pc, tc, bindings, registry):
                return False
        return True

    prefix = pchildren[:seq_idx]
    seq_wild: SeqWildcard = pchildren[seq_idx]  # type: ignore[assignment]
    suffix = pchildren[seq_idx + 1:]

    if len(prefix) + len(suffix) > len(tchildren):
        return False

    for i, p in enumerate(prefix):
        if not _match(p, tchildren[i], bindings, registry):
            return False

    suffix_start = len(tchildren) - len(suffix)
    for j, p in enumerate(suffix):
        if not _match(p, tchildren[suffix_start + j], bindings, registry):
            return False

    middle = tuple(tchildren[len(prefix):suffix_start])
    return _bind_sequence(seq_wild.name, middle, bindings)


def _bind_wildcard(
    wild: Wildcard,
    target: Expr,
    bindings: Bindings,
    registry: Optional[PropertyRegistry],
) -> bool:
    if wild.type_filter is not None:
        if registry is None:
            return False
        if not registry.has(target, wild.type_filter):
            return False
    return _bind_single(wild.name, target, bindings)


def _bind_single(name: str, target: Expr, bindings: Bindings) -> bool:
    prior = bindings.get(name)
    if prior is None:
        bindings[name] = target
        return True
    if isinstance(prior, tuple):
        return False
    return prior == target


def _bind_sequence(
    name: str, seq: Tuple[Expr, ...], bindings: Bindings
) -> bool:
    prior = bindings.get(name)
    if prior is None:
        bindings[name] = seq
        return True
    if not isinstance(prior, tuple):
        return False
    return prior == seq


# --------------------------------------------------------------------- #
# Substitution back into a pattern                                      #
# --------------------------------------------------------------------- #


def substitute(pattern: Expr, bindings: Bindings) -> Expr:
    """Replace every wildcard in ``pattern`` with its binding.

    Unbound wildcards are left in place (they're not errors — a rule's
    right-hand side may intentionally contain wildcards that don't
    appear on the left). SeqWildcards must be inside a Sum/Product;
    using one elsewhere is rejected at the Expr-building level.
    """
    if isinstance(pattern, Wildcard):
        bound = bindings.get(pattern.name)
        if bound is None or isinstance(bound, tuple):
            return pattern
        return bound
    if isinstance(pattern, SeqWildcard):
        return pattern
    if pattern.is_atom:
        return pattern

    new_children: list[Expr] = []
    for child in pattern.children:
        if isinstance(child, SeqWildcard):
            bound = bindings.get(child.name)
            if bound is None:
                new_children.append(child)
            elif isinstance(bound, tuple):
                new_children.extend(bound)
            else:
                new_children.append(bound)
        else:
            new_children.append(substitute(child, bindings))

    # Rebuild via the same constructor. Smart constructors live on
    # Sum/Product via `.make()` — but we want structural preservation
    # here, not simplification. Direct constructor call.
    return type(pattern)(*new_children)
