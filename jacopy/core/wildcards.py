"""
Pattern matching via wildcards.

Wildcards are Expr atoms that stand in for arbitrary subtrees. They
let rewrite rules be written as patterns: ``d(?A + ?B) -> d(?A) + d(?B)``
for linearity, ``d(?A * ?B) -> d(?A)*?B + ?A*d(?B)`` for Leibniz,
``[?X, [?Y, ?Z]] + cyclic -> 0`` for Jacobi, and so on.

Two wildcard kinds:

* :class:`Wildcard`, a single hole that matches one Expr. It may
  carry constraints:

  - ``type_filter``: a :class:`Property` class the candidate must be
    registered with. Requires a :class:`PropertyRegistry` to be passed
    to :func:`match`.
  - ``expr_type``: an :class:`Expr` subclass (or tuple of them) the
    candidate must be an instance of. Lets a rule say "match only a
    Symbol here", "match any Sum here", and so on.

* :class:`SeqWildcard`, a sequence hole that matches zero or more
  consecutive children of a Sum or Product. Multiple per level are
  allowed; matching backtracks over possible splits.

Matching is structural and order-preserving, it does not reorder
Sum or Product children. Commutative matching is the job of the
algorithms layer, which canonicalizes expressions before invoking
:func:`match`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from jacopy.core.expr import Atom, Expr
from jacopy.core.properties import Property
from jacopy.core.registry import PropertyRegistry


# --------------------------------------------------------------------- #
# Wildcard nodes                                                         #
# --------------------------------------------------------------------- #


ExprTypeFilter = Union[Type[Expr], Tuple[Type[Expr], ...]]


def _normalize_expr_type(
    expr_type: Optional[ExprTypeFilter],
) -> Optional[Tuple[Type[Expr], ...]]:
    if expr_type is None:
        return None
    if isinstance(expr_type, type):
        candidates: Tuple[Type[Expr], ...] = (expr_type,)
    elif isinstance(expr_type, tuple):
        candidates = expr_type
    else:
        raise TypeError(
            "expr_type must be an Expr subclass or a tuple of them"
        )
    for cls in candidates:
        if not (isinstance(cls, type) and issubclass(cls, Expr)):
            raise TypeError(
                "expr_type must be an Expr subclass or a tuple of them"
            )
    return candidates


class Wildcard(Atom):
    """Single-hole pattern atom.

    A wildcard with ``name`` binds that name to whatever subtree it
    matches. If the same name appears twice in a pattern, both
    occurrences must match *structurally equal* targets.

    Optional constraints:

    * ``type_filter``, a :class:`Property` class. The target must be
      registered as that property in the :class:`PropertyRegistry`
      passed to :func:`match`.
    * ``expr_type``, an :class:`Expr` subclass, or a tuple of them.
      The target must be an instance. Evaluated without the registry.
    """

    __slots__ = ("_name", "_type_filter", "_expr_type")

    def __init__(
        self,
        name: str,
        type_filter: Optional[Type[Property]] = None,
        expr_type: Optional[ExprTypeFilter] = None,
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
        self._expr_type = _normalize_expr_type(expr_type)

    @property
    def name(self) -> str:
        return self._name

    @property
    def type_filter(self) -> Optional[Type[Property]]:
        return self._type_filter

    @property
    def expr_type(self) -> Optional[Tuple[Type[Expr], ...]]:
        return self._expr_type

    def _key(self) -> Any:
        return (self._name, self._type_filter, self._expr_type)

    def _repr_inner(self) -> str:
        parts = [f"?{self._name}"]
        if self._type_filter is not None:
            parts.append(f":{self._type_filter.__name__}")
        if self._expr_type is not None:
            names = "|".join(c.__name__ for c in self._expr_type)
            parts.append(f"<{names}>")
        return "".join(parts)


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
    deliberate, silently ignoring a type filter would mask bugs.
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
    """Backtracking match over a run of pattern children against targets.

    Non-seq pattern children consume exactly one target each; SeqWildcards
    can consume any number (0..remaining). With multiple SeqWildcards in
    the same run we try all splits, left-to-right, restoring bindings on
    failure so each candidate gets a clean slate.
    """
    return _match_children_rec(
        list(pchildren), 0, list(tchildren), 0, bindings, registry
    )


def _match_children_rec(
    ps: List[Expr],
    pi: int,
    ts: List[Expr],
    ti: int,
    bindings: Bindings,
    registry: Optional[PropertyRegistry],
) -> bool:
    if pi == len(ps):
        return ti == len(ts)

    p = ps[pi]
    if isinstance(p, SeqWildcard):
        remaining = len(ts) - ti
        # Try every length k in 0..remaining; leftmost-shortest first so
        # the match order is predictable and stable.
        for k in range(0, remaining + 1):
            snapshot = dict(bindings)
            seq = tuple(ts[ti:ti + k])
            if _bind_sequence(p.name, seq, bindings):
                if _match_children_rec(
                    ps, pi + 1, ts, ti + k, bindings, registry
                ):
                    return True
            bindings.clear()
            bindings.update(snapshot)
        return False

    if ti == len(ts):
        return False

    snapshot = dict(bindings)
    if _match(p, ts[ti], bindings, registry):
        if _match_children_rec(
            ps, pi + 1, ts, ti + 1, bindings, registry
        ):
            return True
    bindings.clear()
    bindings.update(snapshot)
    return False


def _bind_wildcard(
    wild: Wildcard,
    target: Expr,
    bindings: Bindings,
    registry: Optional[PropertyRegistry],
) -> bool:
    if wild.expr_type is not None:
        if not isinstance(target, wild.expr_type):
            return False
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

    Unbound wildcards are left in place (they're not errors, a rule's
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
    # Sum/Product via `.make()`, but we want structural preservation
    # here, not simplification. Direct constructor call via _rebuild
    # so types whose __init__ diverges from children (BracketApply
    # carries its bracket outside children) reconstruct correctly.
    return pattern._rebuild(tuple(new_children))
