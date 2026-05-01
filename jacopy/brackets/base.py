"""
Graded bracket framework.

A graded bracket is a bilinear operation ``[·, ·]`` on a graded module.
Each bracket carries:

* a ``degree``, shifts total grade by this amount: ``|[a, b]| = |a| + |b| + degree``.
* ``is_graded_antisymmetric``, whether ``[a, b] = −(−1)^{|a||b|} [b, a]`` holds.
* ``satisfies_leibniz``, whether ``[a, b*c] = [a, b]*c + (−1)^{|a||b|} b*[a, c]`` holds.
* ``satisfies_graded_jacobi``, whether the graded Jacobi identity holds
  (``True``, ``False``, or ``None`` for *conditional* brackets like the
  derived bracket, whose Jacobi is controlled by a separate condition).

Brackets are *not* Exprs: they are Python-level operators. Applying a
bracket to two Exprs builds an inert :class:`BracketApply` node in the
expression tree, analogous to :class:`jacopy.algebra.derivation.Act`.
The expansion (turning the opaque ``B(a, b)`` node into its defining
formula) is done by :meth:`GradedBracket.expand`, which subclasses
implement.

The module also exposes *axiom obstruction* helpers. These return the
Expr that the corresponding axiom claims is zero, feed it to
``simplify`` / ``canonicalize`` to check the axiom on concrete inputs.
Mutating the tree to ``0`` is the certificate that the axiom holds;
leaving a non-trivial remainder exposes the counterexample.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree


# --------------------------------------------------------------------- #
# Application node                                                       #
# --------------------------------------------------------------------- #


class BracketApply(Expr):
    """Inert application of a bracket: ``B(a, b)``.

    Carries a reference to the :class:`GradedBracket` instance so the
    :func:`expand_bracket` algorithm (and :meth:`expand`) can delegate
    back to the bracket's expansion rule. The bracket itself is not an
    Expr, treating it as such would invite accidental structural
    comparisons between two different brackets that happen to share a
    name or degree.
    """

    __slots__ = ("_bracket", "_a", "_b")

    def __init__(
        self, bracket: "GradedBracket", a: Expr, b: Expr
    ) -> None:
        if not isinstance(bracket, GradedBracket):
            raise TypeError("BracketApply requires a GradedBracket instance")
        if not isinstance(a, Expr):
            raise TypeError("BracketApply left operand must be an Expr")
        if not isinstance(b, Expr):
            raise TypeError("BracketApply right operand must be an Expr")
        self._bracket = bracket
        self._a = a
        self._b = b

    @property
    def bracket(self) -> "GradedBracket":
        return self._bracket

    @property
    def a(self) -> Expr:
        return self._a

    @property
    def b(self) -> Expr:
        return self._b

    @property
    def children(self) -> Tuple[Expr, ...]:
        # The bracket reference is not a child, children are Expr
        # operands only. Tree walks (walk, find, pattern match) see a
        # binary node.
        return (self._a, self._b)

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> Expr:
        # Default ``type(self)(*children)`` would drop the bracket
        # reference, since it lives outside the children tuple.
        if len(new_children) != 2:
            raise ValueError(
                "BracketApply._rebuild expects exactly 2 children "
                f"(a, b), got {len(new_children)}"
            )
        a, b = new_children
        return BracketApply(self._bracket, a, b)

    def _key(self) -> Any:
        return (self._bracket, self._a, self._b)

    def _repr_inner(self) -> str:
        return (
            f"{self._bracket.name}"
            f"({self._a._repr_inner()}, {self._b._repr_inner()})"
        )

    def expand(self, registry: Optional[PropertyRegistry] = None) -> Expr:
        """Defer to the bracket's expansion rule."""
        return self._bracket.expand(self._a, self._b, registry)


# --------------------------------------------------------------------- #
# Base class                                                             #
# --------------------------------------------------------------------- #


class GradedBracket(ABC):
    """Abstract graded bracket on a graded module.

    Subclasses declare their degree and axiom flags via ``__init__`` and
    implement :meth:`expand`. The default axiom flags reflect the most
    common case (degree 0, antisymmetric, Leibniz, Jacobi holds); override
    in subclasses for anything exotic.
    """

    def __init__(
        self,
        name: str,
        *,
        degree: DegreeLike = 0,
        is_graded_antisymmetric: bool = True,
        satisfies_leibniz: bool = True,
        satisfies_graded_jacobi: Optional[bool] = True,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Bracket name must be a non-empty string")
        self._name = name
        self._degree = as_degree(degree)
        self._is_graded_antisymmetric = bool(is_graded_antisymmetric)
        self._satisfies_leibniz = bool(satisfies_leibniz)
        # None is allowed: conditional Jacobi (e.g. derived bracket).
        if satisfies_graded_jacobi is not None:
            satisfies_graded_jacobi = bool(satisfies_graded_jacobi)
        self._satisfies_graded_jacobi = satisfies_graded_jacobi

    # ---- read-only accessors --------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def degree(self) -> Degree:
        return self._degree

    @property
    def is_graded_antisymmetric(self) -> bool:
        return self._is_graded_antisymmetric

    @property
    def satisfies_leibniz(self) -> bool:
        return self._satisfies_leibniz

    @property
    def satisfies_graded_jacobi(self) -> Optional[bool]:
        """True / False / None (conditional)."""
        return self._satisfies_graded_jacobi

    # ---- application ----------------------------------------------- #

    def __call__(self, a: Expr, b: Expr) -> BracketApply:
        return BracketApply(self, a, b)

    # ---- sign-convention hooks ------------------------------------- #
    #
    # The closure-axiom rules in :mod:`jacopy.calculus.bracket_apply_axioms`
    # consult these hooks instead of hardcoding signs, so subclasses with
    # genuine graded-sign behavior on their operands (e.g. a future
    # Schouten-Nijenhuis rule layer) can override without rewriting the
    # rules. Defaults preserve the *literal-antisym* convention used by
    # the existing degree-0 brackets (Koszul, Lie bracket on TM, …):
    # ``[a, b] = −[b, a]`` and the cyclic Jacobi ``Σ_cyclic [A, [B, C]] = 0``
    # carries no per-term sign factor.

    def pair_swap_sign(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Optional[int]:
        """Sign such that ``[a, b] = pair_swap_sign · [b, a]``.

        Returns ``±1`` for a decidable swap, ``None`` when undecidable
        (the rule should then decline). Default: ``−1`` for a
        graded-antisymmetric bracket (literal antisym), ``None``
        otherwise. Subclasses implementing the full graded convention
        ``[a, b] = −(−1)^{|a||b|}[b, a]`` should override to compute
        ``-1 if (degree_of(a) * degree_of(b)).parity() == 0 else +1``.
        """
        if not self._is_graded_antisymmetric:
            return None
        return -1

    def jacobi_term_sign(
        self,
        A: Expr,
        B: Expr,
        C: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Optional[int]:
        """Per-term sign for the ``[A, [B, C]]`` summand in cyclic Jacobi.

        Cyclic Jacobi reads
        ``Σ_cyclic jacobi_term_sign(A, B, C) · [A, [B, C]] = 0``.
        Default: ``+1`` (literal Jacobi, no per-term Koszul factor).
        Subclasses with full graded behavior override to return
        ``+1 if (degree_of(A) * degree_of(C)).parity() == 0 else −1``.
        Returns ``None`` for undecidable parity.
        """
        return 1

    @abstractmethod
    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """Return the definitional expansion of ``[a, b]``.

        Subclasses implement this with their concrete formula. The
        expansion is purely syntactic, it does not invoke simplify /
        canonicalize; the caller pipes the result through whichever
        reductions they need.
        """

    # ---- axiom obstruction helpers --------------------------------- #
    #
    # Each helper returns the Expr the axiom claims is zero. Feed it to
    # simplify() / canonicalize() and collect_terms() to get a
    # verdict on concrete inputs.

    def graded_antisymmetry_obstruction(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``[a, b] + (−1)^{|a||b|} [b, a]``.

        Claims to be zero iff the bracket is graded-antisymmetric on the
        pair ``(a, b)``. The sign parity is computed from the declared
        degrees of ``a`` and ``b``; undecidable parity raises
        :class:`ValueError` so the caller can narrow the degrees.
        """
        parity = (degree_of(a, registry) * degree_of(b, registry)).parity()
        if parity is None:
            raise ValueError(
                f"antisymmetry obstruction parity is symbolic for "
                f"({a!r}, {b!r}); narrow the operand degrees"
            )
        ab = BracketApply(self, a, b)
        ba = BracketApply(self, b, a)
        # (−1)^{|a||b|} = +1 when parity==0, −1 when parity==1.
        second = ba if parity == 0 else Neg(ba)
        return Sum(ab, second)

    def graded_jacobi_obstruction(
        self,
        a: Expr,
        b: Expr,
        c: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """Cyclic Jacobi with Koszul signs.

        Returns

            (−1)^{|a||c|} [a, [b, c]]
          + (−1)^{|b||a|} [b, [c, a]]
          + (−1)^{|c||b|} [c, [a, b]]

        which the graded Jacobi identity claims is zero. Undecidable
        parity raises :class:`ValueError`.
        """
        deg_a = degree_of(a, registry)
        deg_b = degree_of(b, registry)
        deg_c = degree_of(c, registry)
        terms = []
        for x, y, z, p in (
            (a, b, c, (deg_a * deg_c).parity()),
            (b, c, a, (deg_b * deg_a).parity()),
            (c, a, b, (deg_c * deg_b).parity()),
        ):
            if p is None:
                raise ValueError(
                    "Jacobi obstruction parity is symbolic; narrow "
                    "operand degrees"
                )
            inner = BracketApply(self, y, z)
            outer = BracketApply(self, x, inner)
            terms.append(outer if p == 0 else Neg(outer))
        return Sum(*terms)

    def leibniz_obstruction(
        self,
        a: Expr,
        b: Expr,
        c: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``[a, b*c] − [a, b]*c − (−1)^{|a||b|} b*[a, c]``.

        Claims to be zero when the bracket acts as a graded derivation
        in its second slot. Undecidable sign parity raises.
        """
        parity = (degree_of(a, registry) * degree_of(b, registry)).parity()
        if parity is None:
            raise ValueError(
                "Leibniz obstruction parity is symbolic; narrow "
                "operand degrees"
            )
        left = BracketApply(self, a, Product(b, c))
        t1 = Product(BracketApply(self, a, b), c)
        t2 = Product(b, BracketApply(self, a, c))
        # Axiom: left − t1 − (−1)^{|a||b|} t2. Fold the sign directly
        # rather than stacking Neg(Neg(…)).
        third = Neg(t2) if parity == 0 else t2
        return Sum(left, Neg(t1), third)

    # ---- identity / hashing ---------------------------------------- #

    def _identity_key(self) -> Any:
        """Override in subclasses carrying parametric data.

        Two brackets are equal iff they share concrete type, name,
        degree, and axiom profile. Parametric brackets (e.g. derived
        bracket with a specific generator) extend this key with their
        own payload.
        """
        return (
            self._name,
            self._degree,
            self._is_graded_antisymmetric,
            self._satisfies_leibniz,
            self._satisfies_graded_jacobi,
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return self._identity_key() == other._identity_key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._identity_key()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r})"


# --------------------------------------------------------------------- #
# Standalone expand                                                      #
# --------------------------------------------------------------------- #


def expand_bracket(
    node: BracketApply, registry: Optional[PropertyRegistry] = None
) -> Expr:
    """Expand a single :class:`BracketApply` via its bracket's rule.

    Matches the :func:`jacopy.algebra.commutator.expand_commutator`
    pattern, a standalone function that the rewrite layer can plug in
    as a rule, plus a :meth:`BracketApply.expand` convenience method.
    """
    return node.expand(registry)
