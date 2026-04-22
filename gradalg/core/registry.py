"""
Property registry — the binding between expressions and their properties.

The registry answers two questions:

1. *Does this expression have property X?* Asked constantly by
   algorithms: ``is this a scalar?``, ``what's this operator's
   degree?``, ``is this bracket graded-antisymmetric?``.
2. *Why does it have property X?* Asked when writing proofs: was it
   an axiom the user declared, or derived earlier? If derived, by
   what rule?

Strict mode (``strict_axioms_only=True``) hides derived properties
at query time. This is how the unroll / foundational proof mode works:
flip the switch and the registry behaves as if nothing beyond the
initial axioms has been established yet, forcing algorithms to
re-derive everything from first principles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Optional, Type, TypeVar

from gradalg.core.expr import Expr
from gradalg.core.properties import Property, Provenance

P = TypeVar("P", bound=Property)


class PropertyRegistry:
    """Stores properties keyed by expression and property-type.

    An expression may hold multiple properties, but at most one of any
    given property-type — declaring the same type twice raises
    :class:`ValueError`. Use :meth:`retract` to remove a property
    before replacing it; making replacement explicit prevents silent
    overwrites of axioms.
    """

    def __init__(self, *, strict_axioms_only: bool = False) -> None:
        self._props: dict[Expr, dict[type, Property]] = defaultdict(dict)
        self._strict = strict_axioms_only

    # ---- mode ------------------------------------------------------ #

    @property
    def strict_axioms_only(self) -> bool:
        return self._strict

    def set_strict(self, strict: bool) -> None:
        self._strict = bool(strict)

    # ---- mutation -------------------------------------------------- #

    def declare(self, expr: Expr, prop: Property) -> None:
        """Attach ``prop`` to ``expr``.

        Raises :class:`ValueError` if a property of the same type is
        already declared on this expression.
        """
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
        """Remove the property of type ``prop_cls`` from ``expr``.

        Returns the removed property, or ``None`` if there was none.
        """
        bucket = self._props.get(expr)
        if bucket is None:
            return None
        return bucket.pop(prop_cls, None)

    # ---- query ----------------------------------------------------- #

    def get(self, expr: Expr, prop_cls: Type[P]) -> Optional[P]:
        """Return the property of type ``prop_cls`` on ``expr``.

        In strict mode, derived properties are hidden and return
        ``None`` as if they were never established.
        """
        prop = self._props.get(expr, {}).get(prop_cls)
        if prop is None:
            return None
        if self._strict and prop.provenance is Provenance.DERIVED:
            return None
        return prop  # type: ignore[return-value]

    def has(self, expr: Expr, prop_cls: Type[Property]) -> bool:
        return self.get(expr, prop_cls) is not None

    def all_for(self, expr: Expr) -> Iterator[Property]:
        """Yield every property registered on ``expr``.

        Filtered by strict mode, just like :meth:`get`.
        """
        for prop in self._props.get(expr, {}).values():
            if self._strict and prop.provenance is Provenance.DERIVED:
                continue
            yield prop

    # ---- dunder ---------------------------------------------------- #

    def __len__(self) -> int:
        return sum(len(b) for b in self._props.values())

    def __contains__(self, expr: object) -> bool:
        if not isinstance(expr, Expr):
            return False
        return bool(self._props.get(expr))


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
