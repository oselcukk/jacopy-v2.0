"""
Symbolic degrees, integer-valued polynomials used as gradings.

Every graded object in the package carries a degree: ``|d| = 1``,
``|ι_X| = -1``, ``|L_X| = 0``, or for a generic p-form simply ``|α| = p``
where ``p`` is left *symbolic*. This module provides the algebra needed
to manipulate such degrees uniformly: a single polynomial class over
integer coefficients, with named variables standing for unknown
integer degrees.

The central operation is :meth:`Degree.parity`. Koszul signs have the
form ``(-1)^{d}`` where ``d`` is a degree, and the sign depends only
on ``d mod 2``. Even when the degree is symbolic, parity can often be
decided, ``2|α||β|`` is always even regardless of how ``|α|`` and
``|β|`` resolve, because its integer coefficient is even. This is
what lets generic-degree proofs of Cartan-type identities close.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple, Union


# A monomial is a sorted tuple of ``(var_name, exponent)`` pairs with
# ``exponent > 0``. The empty tuple represents the constant monomial 1.
Monomial = Tuple[Tuple[str, int], ...]

_CONST_MONO: Monomial = ()


class Degree:
    """Commutative polynomial in degree variables, over ℤ.

    Instances are immutable. Internal representation is a canonical
    tuple of ``(monomial, coefficient)`` pairs sorted by monomial with
    zero coefficients stripped, so equality and hashing are
    structural and ``Degree(...) == Degree(...)`` means the two
    polynomials are *equal as polynomials*, not merely numerically
    coincident at some point.

    Construct via the classmethods :meth:`const` and :meth:`var`;
    direct construction from a term map is supported but primarily
    intended for internal use.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Mapping[Monomial, int]) -> None:
        items = tuple(
            sorted(
                ((m, c) for m, c in terms.items() if c != 0),
                key=lambda mc: mc[0],
            )
        )
        object.__setattr__(self, "_terms", items)

    # ---- constructors --------------------------------------------- #

    @classmethod
    def const(cls, n: int) -> "Degree":
        """Constant degree ``n``."""
        if isinstance(n, bool):
            raise TypeError("Cannot coerce bool to Degree")
        n = int(n)
        if n == 0:
            return cls({})
        return cls({_CONST_MONO: n})

    @classmethod
    def var(cls, name: str) -> "Degree":
        """A single degree variable, e.g. ``Degree.var("|α|")``."""
        if not isinstance(name, str):
            raise TypeError("Degree variable name must be a str")
        if not name:
            raise ValueError("Degree variable name must be non-empty")
        return cls({((name, 1),): 1})

    @classmethod
    def zero(cls) -> "Degree":
        return cls({})

    # ---- introspection -------------------------------------------- #

    @property
    def terms(self) -> Tuple[Tuple[Monomial, int], ...]:
        """Canonical ``((monomial, coefficient), …)`` tuple."""
        return self._terms

    @property
    def is_zero(self) -> bool:
        return not self._terms

    def as_int(self) -> Optional[int]:
        """Return the integer value if fully concrete, else ``None``."""
        if not self._terms:
            return 0
        if len(self._terms) == 1:
            mono, c = self._terms[0]
            if mono == _CONST_MONO:
                return c
        return None

    def variables(self) -> frozenset:
        """Set of variable names appearing anywhere in the polynomial."""
        vs: set[str] = set()
        for mono, _ in self._terms:
            for name, _ in mono:
                vs.add(name)
        return frozenset(vs)

    def parity(self) -> Optional[int]:
        """Parity of the degree, 0 (even), 1 (odd), or ``None``.

        Returns ``None`` when parity depends on the unknown values of
        the free variables. A symbolic term contributes 0 mod 2 iff
        its coefficient is even, regardless of variable values, so
        the parity is *decidable* whenever every symbolic term has
        an even coefficient. This is exactly the case that matters
        for Koszul-sign cancellation.
        """
        const_parity = 0
        for mono, c in self._terms:
            if mono == _CONST_MONO:
                const_parity = (const_parity + c) % 2
            else:
                if c % 2 != 0:
                    return None
        return const_parity

    # ---- arithmetic ------------------------------------------------ #

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        merged: Dict[Monomial, int] = {}
        for m, c in self._terms:
            merged[m] = merged.get(m, 0) + c
        for m, c in other._terms:
            merged[m] = merged.get(m, 0) + c
        return Degree(merged)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return Degree({m: -c for m, c in self._terms})

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other - self

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        merged: Dict[Monomial, int] = {}
        for m1, c1 in self._terms:
            for m2, c2 in other._terms:
                m = _mono_mul(m1, m2)
                merged[m] = merged.get(m, 0) + c1 * c2
        return Degree(merged)

    def __rmul__(self, other):
        return self.__mul__(other)

    @classmethod
    def _coerce(cls, value: object) -> "Degree | type(NotImplemented)":
        if isinstance(value, Degree):
            return value
        if isinstance(value, bool):
            return NotImplemented
        if isinstance(value, int):
            return cls.const(value)
        return NotImplemented

    # ---- equality / hash ------------------------------------------ #

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Degree):
            return self._terms == other._terms
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return self.as_int() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._terms)

    # ---- repr ------------------------------------------------------ #

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts: list[str] = []
        for mono, c in self._terms:
            parts.append(_fmt_term(mono, c, first=not parts))
        return "".join(parts)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _mono_mul(m1: Monomial, m2: Monomial) -> Monomial:
    """Multiply two monomials, combining exponents by variable name."""
    d: Dict[str, int] = {}
    for name, e in m1:
        d[name] = d.get(name, 0) + e
    for name, e in m2:
        d[name] = d.get(name, 0) + e
    return tuple(sorted(d.items()))


def _fmt_term(mono: Monomial, c: int, first: bool) -> str:
    if first:
        if c < 0:
            sign, c = "-", -c
        else:
            sign = ""
    else:
        if c < 0:
            sign, c = " - ", -c
        else:
            sign = " + "

    if mono == _CONST_MONO:
        body = str(c)
    else:
        factors = [name if e == 1 else f"{name}^{e}" for name, e in mono]
        mono_str = "*".join(factors)
        body = mono_str if c == 1 else f"{c}*{mono_str}"

    return sign + body


# --------------------------------------------------------------------- #
# Coercion used by Graded                                               #
# --------------------------------------------------------------------- #


DegreeLike = Union[int, Degree]


def as_degree(value: DegreeLike) -> Degree:
    """Coerce an ``int`` or :class:`Degree` into a :class:`Degree`.

    Rejects ``bool`` deliberately, ``Graded(degree=True)`` is almost
    always a bug.
    """
    if isinstance(value, Degree):
        return value
    if isinstance(value, bool):
        raise TypeError("Cannot coerce bool to Degree")
    if isinstance(value, int):
        return Degree.const(value)
    raise TypeError(
        f"Cannot coerce object of type {type(value).__name__!r} to Degree"
    )
