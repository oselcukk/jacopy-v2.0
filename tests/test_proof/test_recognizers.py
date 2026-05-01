"""Tests for jacopy.proof.recognizers."""

from jacopy.algebra.commutator import Commutator
from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Product, Sum, Symbol
from jacopy.proof.recognizers import (
    AntisymmetryMatch,
    AntisymmetryRecognizer,
    CommutatorMatch,
    CommutatorRecognizer,
    LeibnizMatch,
    LeibnizRecognizer,
)


# --------------------------------------------------------------------- #
# CommutatorRecognizer                                                   #
# --------------------------------------------------------------------- #


class TestCommutatorRecognizer:
    def test_recognizes_bare_commutator(self):
        A, B = Symbol("A"), Symbol("B")
        match = CommutatorRecognizer().recognize(Commutator(A, B))
        assert isinstance(match, CommutatorMatch)
        assert match.a == A
        assert match.b == B

    def test_rejects_non_commutator(self):
        A = Symbol("A")
        assert CommutatorRecognizer().recognize(A) is None

    def test_rejects_sum_of_commutators(self):
        # A Sum containing a Commutator still isn't a Commutator at the
        # top level, the recognizer only inspects the root node.
        A, B = Symbol("A"), Symbol("B")
        expr = Sum(Commutator(A, B), Commutator(B, A))
        assert CommutatorRecognizer().recognize(expr) is None


# --------------------------------------------------------------------- #
# LeibnizRecognizer                                                      #
# --------------------------------------------------------------------- #


class TestLeibnizRecognizer:
    def test_recognizes_act_on_product(self):
        D = Derivation("D", degree=1)
        a, b = Symbol("a"), Symbol("b")
        match = LeibnizRecognizer().recognize(Act(D, Product(a, b)))
        assert isinstance(match, LeibnizMatch)
        assert match.op == D
        assert match.factors == (a, b)

    def test_preserves_factor_order(self):
        D = Derivation("D", degree=1)
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        match = LeibnizRecognizer().recognize(Act(D, Product(a, b, c)))
        assert match is not None
        assert match.factors == (a, b, c)

    def test_rejects_non_act(self):
        a, b = Symbol("a"), Symbol("b")
        assert LeibnizRecognizer().recognize(Product(a, b)) is None

    def test_rejects_act_on_non_product(self):
        D = Derivation("D", degree=1)
        a = Symbol("a")
        assert LeibnizRecognizer().recognize(Act(D, a)) is None


# --------------------------------------------------------------------- #
# AntisymmetryRecognizer                                                 #
# --------------------------------------------------------------------- #


class TestAntisymmetryRecognizer:
    def test_recognizes_commutator(self):
        A, B = Symbol("A"), Symbol("B")
        match = AntisymmetryRecognizer().recognize(Commutator(A, B))
        assert isinstance(match, AntisymmetryMatch)
        assert match.a == A
        assert match.b == B

    def test_rejects_non_commutator(self):
        A = Symbol("A")
        assert AntisymmetryRecognizer().recognize(A) is None
