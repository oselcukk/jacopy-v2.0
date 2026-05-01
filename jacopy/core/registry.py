"""
Property registry, the binding between expressions and their properties.

The registry answers two questions:

1. *Does this expression have property X?* Asked constantly by
   algorithms: ``is this a scalar?``, ``what's this operator's
   degree?``, ``is this bracket graded-antisymmetric?``.
2. *Why does it have property X?* Asked when writing proofs: was it
   an axiom the user declared, or derived earlier? If derived, by
   what rule?

Three ways to attach a property:

* :meth:`declare`, exact expression → property. Highest priority.
* :meth:`declare_for_class`, every instance of an :class:`Expr`
  subclass carries the property. Subclasses inherit via MRO.
* :meth:`declare_for_predicate`, any expression matching a
  user-supplied predicate carries the property. Checked last, in
  registration order.

Two scoping mechanisms support the foundational / custom proof modes:

* :meth:`scope`, rolls back every mutation inside the block on exit.
  Useful for hypothetical experiments and bracketing a proof inside a
  temporary declaration.
* :meth:`axioms`, narrows the visible property types to a whitelist.
  Within the block, any property whose type is not in the whitelist
  looks as if it had never been declared. This is how the ``custom``
  axiom mode from plan.md is implemented.

``strict_axioms_only`` is orthogonal to :meth:`axioms`: the former
filters by provenance (axiom vs derived), the latter by property
type.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple, Type, TypeVar

from jacopy.core.expr import Expr
from jacopy.core.properties import Property, Provenance

P = TypeVar("P", bound=Property)

Predicate = Callable[[Expr], bool]


class PropertyRegistry:
    """Stores properties keyed by expression, class, or predicate.

    Lookup order in :meth:`get`:

    1. Exact expression match.
    2. Class-based: walks :attr:`type.__mro__` from most specific to
       :class:`Expr`.
    3. Predicate-based: first registered predicate whose property type
       matches wins.

    An expression or class may hold multiple *distinct* property types,
    but at most one of any given property-type from the same source
    (exact / class / predicate). Duplicate declarations at the same
    source raise :class:`ValueError`, the intended fix is an explicit
    :meth:`retract`.
    """

    def __init__(self, *, strict_axioms_only: bool = False) -> None:
        self._props: dict[Expr, dict[type, Property]] = defaultdict(dict)
        self._by_class: dict[type, dict[type, Property]] = defaultdict(dict)
        self._by_predicate: List[Tuple[Predicate, Property]] = []
        self._strict = strict_axioms_only
        self._axiom_whitelist: Optional[Set[type]] = None

    # ---- mode ------------------------------------------------------ #

    @property
    def strict_axioms_only(self) -> bool:
        return self._strict

    def set_strict(self, strict: bool) -> None:
        self._strict = bool(strict)

    # ---- mutation: exact --------------------------------------------- #

    def declare(self, expr: Expr, prop: Property) -> None:
        """Attach ``prop`` to the exact expression ``expr``."""
        if not isinstance(expr, Expr):
            raise TypeError("Registry key must be an Expr")
        if not isinstance(prop, Property):
            raise TypeError("Can only declare Property instances")
        bucket = self._props[expr]
        if type(prop) in bucket:
            raise ValueError(
                f"{type(prop).__name__} already declared on {expr!r}"
            )
        bucket[type(prop)] = prop

    def retract(
        self, expr: Expr, prop_cls: Type[Property]
    ) -> Optional[Property]:
        """Remove the exact-expression property of type ``prop_cls``."""
        bucket = self._props.get(expr)
        if bucket is None:
            return None
        return bucket.pop(prop_cls, None)

    # ---- mutation: class-based -------------------------------------- #

    def declare_for_class(
        self, cls: Type[Expr], prop: Property
    ) -> None:
        """Attach ``prop`` to every instance of ``cls`` (subclasses inherit).

        Exact-expression declarations still take priority over class-based
        ones. An explicit :meth:`retract` on a particular expression does
        NOT remove the class-based fallback, it only clears an exact
        binding.
        """
        if not (isinstance(cls, type) and issubclass(cls, Expr)):
            raise TypeError("declare_for_class: `cls` must be an Expr subclass")
        if not isinstance(prop, Property):
            raise TypeError("declare_for_class: `prop` must be a Property")
        bucket = self._by_class[cls]
        if type(prop) in bucket:
            raise ValueError(
                f"{type(prop).__name__} already declared for class "
                f"{cls.__name__}"
            )
        bucket[type(prop)] = prop

    def retract_for_class(
        self, cls: Type[Expr], prop_cls: Type[Property]
    ) -> Optional[Property]:
        bucket = self._by_class.get(cls)
        if bucket is None:
            return None
        return bucket.pop(prop_cls, None)

    # ---- mutation: predicate-based ---------------------------------- #

    def declare_for_predicate(
        self, predicate: Predicate, prop: Property
    ) -> None:
        """Attach ``prop`` to every expression satisfying ``predicate``.

        Predicates are matched in registration order; the first matching
        entry whose property type is being queried wins. Use sparingly,
        a badly-written predicate is the usual way to silently poison a
        registry.
        """
        if not callable(predicate):
            raise TypeError(
                "declare_for_predicate: `predicate` must be callable"
            )
        if not isinstance(prop, Property):
            raise TypeError(
                "declare_for_predicate: `prop` must be a Property"
            )
        self._by_predicate.append((predicate, prop))

    # ---- query ----------------------------------------------------- #

    def _filter(self, prop: Optional[Property]) -> Optional[Property]:
        """Apply strict-mode and axiom-whitelist visibility filters."""
        if prop is None:
            return None
        if self._strict and prop.provenance is Provenance.DERIVED:
            return None
        if (
            self._axiom_whitelist is not None
            and type(prop) not in self._axiom_whitelist
        ):
            return None
        return prop

    def get(
        self, expr: Expr, prop_cls: Type[P]
    ) -> Optional[P]:
        """Return the property of type ``prop_cls`` on ``expr``, or ``None``.

        Lookup cascades: exact → class (MRO) → predicate. The first
        hit, after applying visibility filters, is returned.
        """
        # 1. Exact.
        prop: Optional[Property] = self._props.get(expr, {}).get(prop_cls)
        if prop is None:
            # 2. Class-based.
            for cls in type(expr).__mro__:
                prop = self._by_class.get(cls, {}).get(prop_cls)
                if prop is not None:
                    break
        if prop is None:
            # 3. Predicate.
            for pred, p in self._by_predicate:
                if isinstance(p, prop_cls) and pred(expr):
                    prop = p
                    break
        return self._filter(prop)  # type: ignore[return-value]

    def has(self, expr: Expr, prop_cls: Type[Property]) -> bool:
        return self.get(expr, prop_cls) is not None

    def all_for(self, expr: Expr) -> Iterator[Property]:
        """Yield every property visible for ``expr`` (after filters).

        Collects candidate property types across all three sources and
        resolves each via :meth:`get` so priority and filters apply
        uniformly.
        """
        candidate_types: Set[type] = set()
        candidate_types.update(self._props.get(expr, {}).keys())
        for cls in type(expr).__mro__:
            candidate_types.update(self._by_class.get(cls, {}).keys())
        for pred, p in self._by_predicate:
            if pred(expr):
                candidate_types.add(type(p))
        for prop_cls in candidate_types:
            prop = self.get(expr, prop_cls)
            if prop is not None:
                yield prop

    # ---- scope / axiom context managers ----------------------------- #

    def _snapshot(self):
        return (
            {k: dict(v) for k, v in self._props.items()},
            {k: dict(v) for k, v in self._by_class.items()},
            list(self._by_predicate),
            self._strict,
            self._axiom_whitelist,
        )

    def _restore(self, snap) -> None:
        props, by_class, preds, strict, whitelist = snap
        self._props = defaultdict(dict, props)
        self._by_class = defaultdict(dict, by_class)
        self._by_predicate = preds
        self._strict = strict
        self._axiom_whitelist = whitelist

    @contextmanager
    def scope(self):
        """Roll back every registry mutation made inside the block.

        Designed for hypothetical edits: declare a property, run a proof,
        and have the declaration vanish afterwards even if the proof
        raised. Safe to nest.
        """
        snap = self._snapshot()
        try:
            yield self
        finally:
            self._restore(snap)

    @contextmanager
    def axioms(self, allowed: Iterable[Type[Property]]):
        """Narrow the visible property types to ``allowed`` during the block.

        Within the block, :meth:`get` returns ``None`` for any property
        whose type is not in ``allowed``, as if it had not been
        declared. Complements :attr:`strict_axioms_only`, which filters
        by provenance rather than type.

        Nests correctly: an inner ``axioms()`` overrides the outer
        whitelist for its scope and restores on exit.
        """
        allowed_set: Set[type] = set(allowed)
        for c in allowed_set:
            if not (isinstance(c, type) and issubclass(c, Property)):
                raise TypeError(
                    "axioms: every entry must be a Property subclass"
                )
        prev = self._axiom_whitelist
        self._axiom_whitelist = allowed_set
        try:
            yield self
        finally:
            self._axiom_whitelist = prev

    # ---- dunder ---------------------------------------------------- #

    def __len__(self) -> int:
        return (
            sum(len(b) for b in self._props.values())
            + sum(len(b) for b in self._by_class.values())
            + len(self._by_predicate)
        )

    def __contains__(self, expr: object) -> bool:
        if not isinstance(expr, Expr):
            return False
        if self._props.get(expr):
            return True
        for cls in type(expr).__mro__:
            if self._by_class.get(cls):
                return True
        for pred, _ in self._by_predicate:
            if pred(expr):
                return True
        return False


# --------------------------------------------------------------------- #
# Module-level default                                                  #
# --------------------------------------------------------------------- #

_default_registry: PropertyRegistry = PropertyRegistry()


def default_registry() -> PropertyRegistry:
    """Return the process-wide default registry.

    Convenient for quick interactive use. For serious work, construct
    an explicit :class:`PropertyRegistry` so contexts don't bleed
    across computations.
    """
    return _default_registry


def reset_default_registry() -> None:
    """Replace the default registry with a fresh, empty one."""
    global _default_registry
    _default_registry = PropertyRegistry()
