"""Tests for the Schouten-Nijenhuis bracket."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.base import BracketApply
from jacopy.brackets.derived import DerivedBracket, VanishingCondition
from jacopy.brackets.lie import LieBracket
from jacopy.brackets.schouten import SchoutenBracket, _sn_degree, sn
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol, Zero
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    """Registry with a few functions (SN deg −1) and vector fields (SN deg 0)."""
    r = PropertyRegistry()
    for name in ("f", "g", "h"):
        r.declare(Symbol(name), Graded(degree=-1))
    for name in ("X", "Y", "Z"):
        r.declare(Symbol(name), Graded(degree=0))
    return r


# --------------------------------------------------------------------- #
# Axiom profile                                                          #
# --------------------------------------------------------------------- #


class TestAxiomProfile:
    def test_degree_zero(self):
        assert SchoutenBracket().degree == Degree.const(0)

    def test_graded_antisymmetric(self):
        assert SchoutenBracket().is_graded_antisymmetric is True

    def test_leibniz_and_jacobi(self):
        b = SchoutenBracket()
        assert b.satisfies_leibniz is True
        assert b.satisfies_graded_jacobi is True

    def test_default_name(self):
        assert "SN" in SchoutenBracket().name

    def test_singleton_is_instance(self):
        assert isinstance(sn, SchoutenBracket)


# --------------------------------------------------------------------- #
# Base cases                                                             #
# --------------------------------------------------------------------- #


class TestBaseCases:
    def test_two_vectors_reduce_to_lie(self, reg):
        X, Y = Symbol("X"), Symbol("Y")
        out = SchoutenBracket().expand(X, Y, reg)
        assert out == LieBracket().expand(X, Y)

    def test_two_functions_vanish(self, reg):
        f, g = Symbol("f"), Symbol("g")
        out = SchoutenBracket().expand(f, g, reg)
        assert out == Zero

    def test_function_then_vector_gives_neg_action(self, reg):
        """``[f, X]_SN = −X(f)``."""
        f, X = Symbol("f"), Symbol("X")
        out = SchoutenBracket().expand(f, X, reg)
        assert out == Neg(Act(X, f))

    def test_vector_then_function_gives_action(self, reg):
        """``[X, f]_SN = X(f)``, antisymmetric partner of the above."""
        X, f = Symbol("X"), Symbol("f")
        out = SchoutenBracket().expand(X, f, reg)
        assert out == Act(X, f)

    def test_antisymmetry_between_cases_2_and_3(self, reg):
        """``[X, f] + [f, X] = 0`` on atomic operands (signs ≡ even ⇒ straight)."""
        X, f = Symbol("X"), Symbol("f")
        br = SchoutenBracket()
        # |X||f| = 0*(-1) = 0 → (-1)^0 = +1 → [X,f] = −[f,X].
        # Both expansions are concrete Acts / Neg(Acts), so their sum
        # is structurally zero after Neg cancellation.
        left = br.expand(X, f, reg)
        right = br.expand(f, X, reg)
        assert left == Act(X, f)
        assert right == Neg(Act(X, f))


# --------------------------------------------------------------------- #
# SN-aware degree                                                        #
# --------------------------------------------------------------------- #


class TestSNDegree:
    def test_atomic_function(self, reg):
        assert _sn_degree(Symbol("f"), reg) == Degree.const(-1)

    def test_atomic_vector(self, reg):
        assert _sn_degree(Symbol("X"), reg) == Degree.const(0)

    def test_wedge_of_two_vectors_is_bivector(self, reg):
        """Two 1-vectors wedged give a 2-vector of SN degree 1."""
        X, Y = Symbol("X"), Symbol("Y")
        assert _sn_degree(Product(X, Y), reg) == Degree.const(1)

    def test_wedge_of_three_vectors_is_trivector(self, reg):
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        assert _sn_degree(Product(X, Y, Z), reg) == Degree.const(2)

    def test_wedge_of_vector_and_function(self, reg):
        """1-vector + function: 0 + (−1) + 1 = 0. A 1-vector times a
        function lives in the same SN grade as a 1-vector."""
        X, f = Symbol("X"), Symbol("f")
        assert _sn_degree(Product(X, f), reg) == Degree.const(0)


# --------------------------------------------------------------------- #
# Wedge Leibniz                                                          #
# --------------------------------------------------------------------- #


class TestWedgeLeibnizSlot1:
    def test_vec_wedge_vec_against_function(self, reg):
        """``[X ∧ Y, f]_SN = X ∧ [Y, f] + (−1)^{|Y||f|} [X, f] ∧ Y``
        = ``X * Y(f) + X(f) * Y`` with ``|Y||f| = 0 * (−1) = 0``."""
        X, Y, f = Symbol("X"), Symbol("Y"), Symbol("f")
        out = SchoutenBracket().expand(Product(X, Y), f, reg)
        expected = Sum(
            Product(X, Act(Y, f)),
            Product(Act(X, f), Y),
        )
        assert out == expected

    def test_recursion_descends_into_three_factor_wedge(self, reg):
        """A 3-factor wedge peels one factor per level."""
        X, Y, Z, f = Symbol("X"), Symbol("Y"), Symbol("Z"), Symbol("f")
        # [X∧Y∧Z, f] should be Sum(X ∧ [Y∧Z, f], [X, f] ∧ (Y∧Z)).
        # Where |Y∧Z| = 1 and |f| = −1, product parity = 1 → sign = −1.
        out = SchoutenBracket().expand(Product(X, Y, Z), f, reg)
        assert isinstance(out, Sum)
        # Leading factor is the un-peeled X ∧ [tail, f].
        first = out.children[0]
        assert isinstance(first, Product)
        assert first.children[0] is X

    def test_symbolic_operand_returns_opaque(self):
        """When operand degrees can't be resolved the expand falls back
        to an opaque :class:`BracketApply` rather than raising."""
        br = SchoutenBracket()
        a, b = Symbol("a"), Symbol("b")
        # No registry → no degrees → no base case, no Leibniz to trigger.
        out = br.expand(a, b)
        assert isinstance(out, BracketApply)
        assert out.bracket is br
        assert out.a is a and out.b is b


class TestWedgeLeibnizSlot2:
    def test_function_against_vec_wedge_vec(self, reg):
        """``[f, X ∧ Y]_SN = [f, X] ∧ Y + (−1)^{|X||f|} X ∧ [f, Y]``
        = ``−X(f) * Y + X * (−Y(f))`` with ``|X||f| = 0``."""
        X, Y, f = Symbol("X"), Symbol("Y"), Symbol("f")
        out = SchoutenBracket().expand(f, Product(X, Y), reg)
        expected = Sum(
            Product(Neg(Act(X, f)), Y),
            Product(X, Neg(Act(Y, f))),
        )
        assert out == expected


# --------------------------------------------------------------------- #
# Bivector self-bracket, the Poisson obstruction shape                 #
# --------------------------------------------------------------------- #


class TestBivectorSelfBracket:
    def test_atomic_bivector_self_bracket_is_opaque(self):
        """A bare ``π`` declared ``Graded(degree=1)`` has no wedge to
        decompose, so ``[π, π]_SN`` stays as an inert BracketApply,
        exactly the shape the derived-bracket machinery consumes as
        the Poisson condition's obstruction."""
        r = PropertyRegistry()
        pi = Symbol("π")
        r.declare(pi, Graded(degree=1))
        br = SchoutenBracket()
        out = br.self_bracket(pi, r)
        assert isinstance(out, BracketApply)
        assert out.bracket is br
        assert out.a is pi and out.b is pi

    def test_decomposable_bivector_expands(self, reg):
        """When the bivector is written as ``X ∧ Y`` explicitly, the
        self-bracket recurses through wedge Leibniz into six vector-
        pair Lie brackets (three inner 1-vector pairs, each feeding
        two outer terms). We don't pin the exact algebraic form here,
        the shape test is that the result is a (non-trivial) Sum whose
        leaves reach the 1-vector operands."""
        X, Y = Symbol("X"), Symbol("Y")
        pi = Product(X, Y)
        out = SchoutenBracket().self_bracket(pi, reg)
        descendants = list(out.walk())
        assert any(node is X for node in descendants)
        assert any(node is Y for node in descendants)
        # No opaque BracketApplies left, everything decomposed.
        assert not any(isinstance(n, BracketApply) for n in descendants)


# --------------------------------------------------------------------- #
# Poisson-as-DerivedBracket, Stage 1 end-to-end                        #
# --------------------------------------------------------------------- #


class TestPoissonFromDerivedBracket:
    """``PoissonBracket(π) = DerivedBracket(SN, π, degree_Q=1)``, with
    the obstruction ``[π, π]_SN`` as its Jacobi condition.

    Per the plan, no dedicated ``PoissonBracket`` class is needed: the
    existing derived-bracket machinery closes Jacobi automatically once
    ``[π, π]_SN = 0`` is assumed. These tests verify the wiring holds.
    """

    def test_poisson_jacobi_condition_is_sn_self_bracket(self):
        """The derived-bracket Jacobi condition unpacks to
        ``[π, π]_SN``, the canonical Poisson obstruction."""
        r = PropertyRegistry()
        pi = Symbol("π")
        r.declare(pi, Graded(degree=1))
        poisson = DerivedBracket(sn, pi, degree_Q=1)
        cond = poisson.jacobi_condition(r)
        assert isinstance(cond, VanishingCondition)
        # Obstruction is the SN self-bracket of π, which stays opaque
        # for atomic π, still the right typed handle for a proof.
        assert cond.obstruction == sn.self_bracket(pi, r)

    def test_poisson_bracket_degree_is_minus_one(self):
        """Derived bracket formula: ``|{·,·}_π| = |π| − 2 = −1``."""
        r = PropertyRegistry()
        pi = Symbol("π")
        r.declare(pi, Graded(degree=1))
        poisson = DerivedBracket(sn, pi, degree_Q=1)
        assert poisson.degree == Degree.const(-1)

    def test_poisson_bracket_of_functions_type_checks(self):
        """``{f, g}_π`` should construct without error, contents are
        deferred to downstream simplification but the construction path
        (through SN.expand on functions / bivectors) must not blow up."""
        r = PropertyRegistry()
        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        r.declare(pi, Graded(degree=1))
        r.declare(f, Graded(degree=-1))
        r.declare(g, Graded(degree=-1))
        poisson = DerivedBracket(sn, pi, degree_Q=1)
        # Should not raise. We don't pin the result, concrete π is
        # opaque so the inner bracket stays symbolic, which is the
        # right behaviour for the Stage-1 contract.
        result = poisson.expand(f, g, r)
        assert result is not None
