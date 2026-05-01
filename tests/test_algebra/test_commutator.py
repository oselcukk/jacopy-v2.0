"""Tests for jacopy.algebra.commutator."""

import pytest

from jacopy.algebra.commutator import (
    Commutator,
    commutator,
    expand_commutator,
)
from jacopy.algebra.derivation import Derivation
from jacopy.algorithms.collect_terms import collect_terms
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestCommutatorConstruction:
    def test_children(self):
        a, b = Symbol("a"), Symbol("b")
        c = Commutator(a, b)
        assert c.a is a
        assert c.b is b
        assert c.children == (a, b)

    def test_factory(self):
        a, b = Symbol("a"), Symbol("b")
        assert commutator(a, b) == Commutator(a, b)

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            Commutator("a", Symbol("b"))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            Commutator(Symbol("a"), "b")  # type: ignore[arg-type]

    def test_equality(self):
        a, b = Symbol("a"), Symbol("b")
        assert Commutator(a, b) == Commutator(a, b)
        assert Commutator(a, b) != Commutator(b, a)

    def test_repr(self):
        a, b = Symbol("a"), Symbol("b")
        assert repr(Commutator(a, b)) == "[a, b]"


# --------------------------------------------------------------------- #
# expand, even parity                                                   #
# --------------------------------------------------------------------- #


class TestExpandEvenParity:
    def test_two_degree_zero_derivations(self):
        """|A|*|B| = 0 even → A*B − B*A."""
        A = Derivation("A", degree=0)
        B = Derivation("B", degree=0)
        c = Commutator(A, B)
        # Expand: A*B − B*A
        assert c.expand() == Sum(Product(A, B), Neg(Product(B, A)))

    def test_odd_and_even(self):
        """|A|=1, |B|=2 → 1*2=2 even → A*B − B*A."""
        A = Derivation("A", degree=1)
        B = Derivation("B", degree=2)
        c = Commutator(A, B)
        assert c.expand() == Sum(Product(A, B), Neg(Product(B, A)))

    def test_via_registry(self):
        reg = PropertyRegistry()
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Graded(degree=2))
        reg.declare(b, Graded(degree=2))
        # 2*2 = 4, even
        assert commutator(a, b).expand(reg) == Sum(
            Product(a, b), Neg(Product(b, a))
        )


# --------------------------------------------------------------------- #
# expand, odd parity                                                    #
# --------------------------------------------------------------------- #


class TestExpandOddParity:
    def test_two_odd_derivations(self):
        """|A|*|B| = 1 odd → A*B + B*A (anticommutator)."""
        A = Derivation("A", degree=1)
        B = Derivation("B", degree=1)
        c = Commutator(A, B)
        assert c.expand() == Sum(Product(A, B), Product(B, A))

    def test_d_d_anticommutator(self):
        """[d, d] = 2 d² on the nose, just the syntactic expansion."""
        d = Derivation("d", degree=1)
        c = Commutator(d, d)
        # |d|*|d| = 1 odd → d*d + d*d
        assert c.expand() == Sum(Product(d, d), Product(d, d))


# --------------------------------------------------------------------- #
# expand, symbolic parity                                               #
# --------------------------------------------------------------------- #


class TestExpandSymbolic:
    def test_symbolic_raises(self):
        A = Derivation("A", degree=Degree.var("|A|"))
        B = Derivation("B", degree=Degree.var("|B|"))
        with pytest.raises(ValueError, match="symbolic"):
            Commutator(A, B).expand()

    def test_symbolic_but_decidable_even(self):
        """2*|A|*|B| has known-even parity and expands cleanly."""
        A = Derivation(
            "A", degree=Degree.const(2) * Degree.var("|A|"),
        )
        B = Derivation("B", degree=Degree.var("|B|"))
        # |A|*|B| = 2*|A|*|B|, coefficient 2 → even parity.
        assert Commutator(A, B).expand() == Sum(
            Product(A, B), Neg(Product(B, A))
        )


# --------------------------------------------------------------------- #
# Unresolvable operands                                                  #
# --------------------------------------------------------------------- #


class TestExpandRequiresDegrees:
    def test_unknown_operand_raises(self):
        A = Derivation("A", degree=1)
        x = Symbol("x")  # no degree declared
        c = Commutator(A, x)
        with pytest.raises(ValueError, match="not determined"):
            c.expand()


# --------------------------------------------------------------------- #
# Antisymmetry: [A,B] = -(-1)^{|A||B|} [B,A]                            #
# --------------------------------------------------------------------- #


class TestAntisymmetry:
    def test_even_parity_flip(self):
        """Even parity: [A,B] = −[B,A]. Sum collapses to zero."""
        A = Derivation("A", degree=0)
        B = Derivation("B", degree=2)
        lhs = Commutator(A, B).expand()
        rhs = Commutator(B, A).expand()
        # lhs = A*B − B*A, rhs = B*A − A*B. Sum collect → 0.
        collected = collect_terms(Sum(lhs, rhs))
        assert collected == Integer(0)

    def test_odd_parity_flip(self):
        """Odd parity: [A,B] = +[B,A]. Both expansions are the same Sum
        up to term ordering."""
        A = Derivation("A", degree=1)
        B = Derivation("B", degree=1)
        lhs = Commutator(A, B).expand()       # A*B + B*A
        rhs = Commutator(B, A).expand()       # B*A + A*B
        # Compare as multisets of summands, Sum preserves order, but
        # the identity is order-independent.
        assert isinstance(lhs, Sum) and isinstance(rhs, Sum)
        assert sorted(lhs.children, key=repr) == sorted(
            rhs.children, key=repr
        )


# --------------------------------------------------------------------- #
# [d, d] = 2 d²                                                          #
# --------------------------------------------------------------------- #


class TestDDIdentity:
    def test_odd_d_anticommutator_collects_to_two_dd(self):
        """|d|=1 (odd) → [d,d] expands to d*d + d*d → 2*d*d after
        collecting like terms."""
        d = Derivation("d", degree=1)
        expanded = Commutator(d, d).expand()
        collected = collect_terms(expanded)
        assert collected == Product(Integer(2), d, d)
