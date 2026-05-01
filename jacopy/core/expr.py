"""
Expression tree, the foundation of jacopy.

Every symbolic object in the package lives in this tree. Nodes are
immutable; operators (`+`, `-`, `*`, `**`) build new trees.

This module deliberately stays minimal: structural representation,
equality, hashing, traversal, and a few trivial normalisations (drop
zeros in sums, drop ones in products, collapse zero-factor products).
Non-trivial rewriting, Koszul signs, Leibniz rule, bracket expansion,
lives in `jacopy.algorithms` and later layers.
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from math import gcd as _gcd
from typing import Any, Iterator, Tuple


# --------------------------------------------------------------------- #
# Base class                                                            #
# --------------------------------------------------------------------- #


class Expr(ABC):
    """Abstract base class for all expressions.

    Subclasses are immutable. Structural equality and hashing are derived
    from the node's type together with a canonical ``_key()`` payload
    provided by each subclass.
    """

    __slots__ = ()

    # ---- structural protocol ---------------------------------------- #

    @property
    @abstractmethod
    def children(self) -> Tuple["Expr", ...]:
        """Tuple of immediate children. Atoms return ``()``."""

    @property
    def is_atom(self) -> bool:
        return len(self.children) == 0

    def _rebuild(self, new_children: Tuple["Expr", ...]) -> "Expr":
        """Return a new node of the same type with ``new_children``.

        Default: ``type(self)(*new_children)``. Subclasses whose
        constructor signature doesn't match the children tuple (e.g.
        :class:`BracketApply`, which carries a non-Expr bracket
        reference outside its children) override this hook.
        """
        return type(self)(*new_children)

    @abstractmethod
    def _key(self) -> Any:
        """Canonical payload for equality / hashing.

        Must be hashable and structurally unique: two expressions of the
        same concrete class are equal iff their ``_key()``s are equal.
        """

    # ---- equality and hashing --------------------------------------- #

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Expr):
            return NotImplemented
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))

    # ---- traversal -------------------------------------------------- #

    def walk(self) -> Iterator["Expr"]:
        """Pre-order traversal yielding ``self`` followed by descendants."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, predicate) -> Iterator["Expr"]:
        """Yield descendants (including self) satisfying ``predicate``."""
        for node in self.walk():
            if predicate(node):
                yield node

    def replace_at(self, path: Tuple[int, ...], new: "Expr") -> "Expr":
        """Return a new tree with ``new`` spliced at ``path``.

        ``path`` is a tuple of child indices, ``(1, 0)`` means "the
        first child of the second child". An empty path replaces the
        root. Rebuilding uses the raw constructor (no smart-ctor
        flattening), so the surrounding shape is preserved exactly.
        """
        if not isinstance(new, Expr):
            raise TypeError("replace_at: `new` must be an Expr")
        if not path:
            return new
        if self.is_atom:
            raise IndexError("replace_at: cannot descend into an atom")
        idx = path[0]
        children_list = list(self.children)
        if idx < 0 or idx >= len(children_list):
            raise IndexError(
                f"replace_at: index {idx} out of range for "
                f"{type(self).__name__} with {len(children_list)} children"
            )
        children_list[idx] = children_list[idx].replace_at(path[1:], new)
        return self._rebuild(tuple(children_list))

    def clone(self) -> "Expr":
        """Return ``self``.

        :class:`Expr` instances are immutable, so structural sharing is
        always safe. ``clone()`` exists for API symmetry with mutable
        tree libraries; it does not copy.
        """
        return self

    # ---- bound-variable substitution -------------------------------- #

    def substitute_atom(self, dummy: "Expr", target: "Expr") -> "Expr":
        """Return ``self`` with every occurrence of ``dummy`` replaced by ``target``.

        ``dummy`` must be an :class:`Atom`. The default implementation
        does the obvious top-level match, then recurses through
        :meth:`children`. Atoms that hide a sub-atom inside a private
        slot (e.g. a frame vector field carrying a ``FrameIndex``)
        override this method to expose that hidden state for
        substitution. Binders that introduce their own dummy (notably
        :class:`~jacopy.core.indexed_sum.IndexedSum`) override to
        respect shadowing.
        """
        if self == dummy:
            return target
        if self.is_atom:
            return self
        new_children = tuple(
            c.substitute_atom(dummy, target) for c in self.children
        )
        if all(a is b for a, b in zip(new_children, self.children)):
            return self
        return self._rebuild(new_children)

    # ---- operator overloading --------------------------------------- #

    def __add__(self, other):
        return Sum.make(self, _wrap(other))

    def __radd__(self, other):
        return Sum.make(_wrap(other), self)

    def __sub__(self, other):
        return Sum.make(self, Neg(_wrap(other)))

    def __rsub__(self, other):
        return Sum.make(_wrap(other), Neg(self))

    def __neg__(self):
        return Neg(self)

    def __pos__(self):
        return self

    def __mul__(self, other):
        return Product.make(self, _wrap(other))

    def __rmul__(self, other):
        return Product.make(_wrap(other), self)

    def __pow__(self, other):
        return Power(self, _wrap(other))

    # ---- repr -------------------------------------------------------- #

    def __repr__(self) -> str:
        return self._repr_inner()

    @abstractmethod
    def _repr_inner(self) -> str: ...


# --------------------------------------------------------------------- #
# Coercion helper                                                       #
# --------------------------------------------------------------------- #


def _wrap(value: object) -> "Expr":
    """Coerce a Python number into an :class:`Expr`.

    ``bool`` is rejected deliberately, ``True + Symbol("x")`` almost
    always reflects a bug, not a genuine arithmetic operation.
    """
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        raise TypeError("Cannot coerce bool to Expr")
    if isinstance(value, int):
        return Integer(value)
    if isinstance(value, numbers.Rational):
        return Rational(value.numerator, value.denominator)
    raise TypeError(
        f"Cannot coerce object of type {type(value).__name__!r} to Expr"
    )


def _is_integer_value(expr: "Expr", value: int) -> bool:
    return isinstance(expr, Integer) and expr.value == value


# --------------------------------------------------------------------- #
# Atoms                                                                 #
# --------------------------------------------------------------------- #


class Atom(Expr):
    """An expression with no children."""

    __slots__ = ()

    @property
    def children(self) -> Tuple["Expr", ...]:
        return ()


class Symbol(Atom):
    """A named symbolic variable, e.g. ``Symbol("X")``."""

    __slots__ = ("_name",)

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Symbol name must be a str")
        if not name:
            raise ValueError("Symbol name must be non-empty")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _key(self) -> Any:
        return self._name

    def _repr_inner(self) -> str:
        return self._name


class Integer(Atom):
    """Integer literal.

    Values ``-1``, ``0``, ``1`` are cached as singletons so that
    structural equality with common sentinels is cheap and references
    like :data:`Zero` and :data:`One` compare identical under ``is``.
    """

    __slots__ = ("_value",)

    _cache: "dict[int, Integer]" = {}

    def __new__(cls, value: int):
        v = int(value)
        cached = cls._cache.get(v)
        if cached is not None:
            return cached
        inst = super().__new__(cls)
        inst._value = v
        if v in (-1, 0, 1):
            cls._cache[v] = inst
        return inst

    def __init__(self, value: int):
        # All state assigned in __new__; avoid clobbering cached instances.
        pass

    @property
    def value(self) -> int:
        return self._value

    def _key(self) -> Any:
        return self._value

    def _repr_inner(self) -> str:
        return str(self._value)


class Rational(Atom):
    """A rational number ``p/q`` in lowest terms with ``q > 0``.

    Automatically collapses to :class:`Integer` when the denominator is
    one after reduction.
    """

    __slots__ = ("_p", "_q")

    def __new__(cls, p: int, q: int = 1):
        p_i, q_i = int(p), int(q)
        if q_i == 0:
            raise ZeroDivisionError("Rational denominator cannot be zero")
        if q_i < 0:
            p_i, q_i = -p_i, -q_i
        g = _gcd(abs(p_i), q_i)
        if g:
            p_i //= g
            q_i //= g
        if q_i == 1:
            return Integer(p_i)
        inst = super().__new__(cls)
        inst._p = p_i
        inst._q = q_i
        return inst

    def __init__(self, p: int, q: int = 1):
        pass

    @property
    def p(self) -> int:
        return self._p

    @property
    def q(self) -> int:
        return self._q

    def _key(self) -> Any:
        return (self._p, self._q)

    def _repr_inner(self) -> str:
        return f"{self._p}/{self._q}"


# --------------------------------------------------------------------- #
# Compound expressions                                                  #
# --------------------------------------------------------------------- #


class Neg(Expr):
    """Unary negation ``-x``.

    Kept as a first-class node rather than ``-1 * x`` so that the sign
    is easy to pattern-match against later, many rewrite rules and
    proof-display routines care about whether a term is negated.
    """

    __slots__ = ("_arg",)

    def __init__(self, arg: Expr):
        if not isinstance(arg, Expr):
            raise TypeError("Neg argument must be an Expr")
        self._arg = arg

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._arg,)

    def _key(self) -> Any:
        return self._arg

    def _repr_inner(self) -> str:
        inner = self._arg._repr_inner()
        # Avoid "--x" surfacing in output; the algorithm layer will
        # ultimately normalise double negations.
        return f"(-{inner})"


class Sum(Expr):
    """n-ary sum. Order of children is preserved as given."""

    __slots__ = ("_children",)

    def __init__(self, *children: Expr):
        for c in children:
            if not isinstance(c, Expr):
                raise TypeError("Sum children must be Expr")
        self._children = tuple(children)

    @property
    def children(self) -> Tuple[Expr, ...]:
        return self._children

    @classmethod
    def make(cls, *args: Expr) -> Expr:
        """Smart constructor.

        Flattens nested sums (associativity), drops integer zeros, and
        collapses trivial cases (no terms -> ``0``; one term -> that
        term). No commutative reordering or like-term collection, that
        belongs to the algorithms layer.
        """
        flat: list[Expr] = []
        for a in args:
            if isinstance(a, Sum):
                flat.extend(a._children)
            else:
                flat.append(a)
        flat = [x for x in flat if not _is_integer_value(x, 0)]
        if not flat:
            return Zero
        if len(flat) == 1:
            return flat[0]
        return cls(*flat)

    def _key(self) -> Any:
        return self._children

    def _repr_inner(self) -> str:
        return "(" + " + ".join(c._repr_inner() for c in self._children) + ")"


class Product(Expr):
    """n-ary product. Non-commutative by default.

    Commutativity, anti-commutativity and graded (Koszul) signs live
    in :mod:`jacopy.algorithms.sort_product`; this class only stores
    the ordered tuple of factors.
    """

    __slots__ = ("_children",)

    def __init__(self, *children: Expr):
        for c in children:
            if not isinstance(c, Expr):
                raise TypeError("Product children must be Expr")
        self._children = tuple(children)

    @property
    def children(self) -> Tuple[Expr, ...]:
        return self._children

    @classmethod
    def make(cls, *args: Expr) -> Expr:
        """Smart constructor.

        Flattens nested products, absorbs zero (any zero factor yields
        ``0``), drops integer ones, and collapses trivial cases.
        Preserves factor order.
        """
        flat: list[Expr] = []
        for a in args:
            if isinstance(a, Product):
                flat.extend(a._children)
            else:
                flat.append(a)
        for x in flat:
            if _is_integer_value(x, 0):
                return Zero
        flat = [x for x in flat if not _is_integer_value(x, 1)]
        if not flat:
            return One
        if len(flat) == 1:
            return flat[0]
        return cls(*flat)

    def _key(self) -> Any:
        return self._children

    def _repr_inner(self) -> str:
        return "(" + " * ".join(c._repr_inner() for c in self._children) + ")"


class Power(Expr):
    """Exponentiation ``base ** exp``.

    No evaluation is performed; even ``Power(One, anything)`` is
    preserved. Later passes may rewrite trivial powers.
    """

    __slots__ = ("_base", "_exp")

    def __init__(self, base: Expr, exp: Expr):
        if not isinstance(base, Expr) or not isinstance(exp, Expr):
            raise TypeError("Power operands must be Expr")
        self._base = base
        self._exp = exp

    @property
    def base(self) -> Expr:
        return self._base

    @property
    def exp(self) -> Expr:
        return self._exp

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._base, self._exp)

    def _key(self) -> Any:
        return (self._base, self._exp)

    def _repr_inner(self) -> str:
        return f"({self._base._repr_inner()}**{self._exp._repr_inner()})"


# --------------------------------------------------------------------- #
# Convenience singletons                                                #
# --------------------------------------------------------------------- #

Zero: Integer = Integer(0)
One: Integer = Integer(1)
NegOne: Integer = Integer(-1)
