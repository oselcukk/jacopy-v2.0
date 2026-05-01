"""Tests for jacopy.core.symbolic_degree."""

import pytest

from jacopy.core.symbolic_degree import Degree, as_degree


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


class TestConst:
    def test_zero(self):
        d = Degree.const(0)
        assert d.is_zero
        assert d.as_int() == 0

    def test_positive(self):
        d = Degree.const(5)
        assert d.as_int() == 5

    def test_negative(self):
        d = Degree.const(-3)
        assert d.as_int() == -3

    def test_rejects_bool(self):
        with pytest.raises(TypeError):
            Degree.const(True)


class TestVar:
    def test_basic(self):
        a = Degree.var("|α|")
        assert a.as_int() is None
        assert "|α|" in a.variables()

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            Degree.var("")

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            Degree.var(42)

    def test_distinct_vars_unequal(self):
        assert Degree.var("p") != Degree.var("q")

    def test_same_name_equal(self):
        assert Degree.var("p") == Degree.var("p")


# --------------------------------------------------------------------- #
# Arithmetic                                                            #
# --------------------------------------------------------------------- #


class TestArithmetic:
    def test_add_collects_like_terms(self):
        a = Degree.var("p")
        s = a + a
        # 2p, one term with coefficient 2
        assert len(s.terms) == 1
        assert s.terms[0][1] == 2

    def test_add_commutative_canonical(self):
        p, q = Degree.var("p"), Degree.var("q")
        assert (p + q) == (q + p)

    def test_add_int(self):
        p = Degree.var("p")
        assert (p + 3) == (3 + p)
        assert (p + 3).as_int() is None

    def test_subtract_cancels(self):
        p = Degree.var("p")
        assert (p - p).is_zero

    def test_neg(self):
        p = Degree.var("p")
        assert -p == Degree({((("p", 1),)): -1})

    def test_mul_commutative(self):
        p, q = Degree.var("p"), Degree.var("q")
        assert p * q == q * p

    def test_mul_int_scales_coefficient(self):
        p = Degree.var("p")
        assert 2 * p == p + p

    def test_mul_distributes(self):
        p, q = Degree.var("p"), Degree.var("q")
        lhs = (p + q) * (p + q)
        # p^2 + 2pq + q^2
        rhs = p * p + 2 * p * q + q * q
        assert lhs == rhs

    def test_zero_times_anything_is_zero(self):
        p = Degree.var("p")
        assert (Degree.zero() * p).is_zero
        assert (p * 0).is_zero


# --------------------------------------------------------------------- #
# Parity, the Koszul-sign-critical query                               #
# --------------------------------------------------------------------- #


class TestParity:
    def test_concrete_even(self):
        assert Degree.const(4).parity() == 0

    def test_concrete_odd(self):
        assert Degree.const(7).parity() == 1

    def test_zero(self):
        assert Degree.zero().parity() == 0

    def test_symbolic_unknown(self):
        assert Degree.var("p").parity() is None

    def test_symbolic_times_two_is_even(self):
        """The key fact: 2*p is always even, regardless of p's value."""
        p = Degree.var("p")
        assert (2 * p).parity() == 0

    def test_p_plus_p_is_even(self):
        p = Degree.var("p")
        assert (p + p).parity() == 0

    def test_p_minus_p_is_zero_even(self):
        p = Degree.var("p")
        assert (p - p).parity() == 0

    def test_symbolic_product_with_even_coeff(self):
        """Koszul cancellation case: 2|α||β| is even."""
        a = Degree.var("|α|")
        b = Degree.var("|β|")
        assert (2 * a * b).parity() == 0

    def test_mixed_even_symbolic_odd_constant(self):
        p = Degree.var("p")
        assert (2 * p + 1).parity() == 1

    def test_mixed_odd_symbolic_unknown(self):
        p = Degree.var("p")
        assert (p + 1).parity() is None


# --------------------------------------------------------------------- #
# Equality / hashing                                                    #
# --------------------------------------------------------------------- #


class TestEqualityHashing:
    def test_polynomial_equality_not_pointwise(self):
        p = Degree.var("p")
        assert (p + 0) == p
        assert (p * 1) == p

    def test_equal_to_int(self):
        assert Degree.const(5) == 5
        assert Degree.const(0) == 0
        assert Degree.var("p") != 0

    def test_hashable_in_set(self):
        p = Degree.var("p")
        s = {p, p + 0, Degree.var("p")}
        assert len(s) == 1

    def test_canonical_order(self):
        """Different construction orders give the same Degree."""
        a, b, c = Degree.var("a"), Degree.var("b"), Degree.var("c")
        assert (a + b + c) == (c + b + a) == (b + a + c)


# --------------------------------------------------------------------- #
# as_degree                                                             #
# --------------------------------------------------------------------- #


class TestAsDegree:
    def test_passes_through(self):
        d = Degree.const(3)
        assert as_degree(d) is d

    def test_wraps_int(self):
        assert as_degree(5) == Degree.const(5)

    def test_rejects_bool(self):
        with pytest.raises(TypeError):
            as_degree(True)

    def test_rejects_str(self):
        with pytest.raises(TypeError):
            as_degree("p")


# --------------------------------------------------------------------- #
# Repr sanity                                                           #
# --------------------------------------------------------------------- #


class TestRepr:
    def test_zero(self):
        assert repr(Degree.zero()) == "0"

    def test_const(self):
        assert repr(Degree.const(7)) == "7"
        assert repr(Degree.const(-3)) == "-3"

    def test_var(self):
        assert repr(Degree.var("p")) == "p"

    def test_sum(self):
        p, q = Degree.var("p"), Degree.var("q")
        # Canonical sorted order is alphabetical.
        assert repr(p + q) == "p + q"

    def test_coeff(self):
        p = Degree.var("p")
        assert repr(3 * p) == "3*p"

    def test_power(self):
        p = Degree.var("p")
        assert repr(p * p) == "p^2"
