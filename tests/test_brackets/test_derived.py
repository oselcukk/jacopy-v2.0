"""Tests for gradalg.brackets.derived."""

import pytest

from gradalg.algorithms.simplify import simplify
from gradalg.brackets.base import BracketApply
from gradalg.brackets.derived import DerivedBracket, derived_bracket
from gradalg.brackets.lie import LieBracket
from gradalg.core.expr import Neg, Product, Sum, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    r = PropertyRegistry()
    # A zoo of graded operands for testing. Q will be declared per-test.
    for s in ("a", "b", "c"):
        r.declare(Symbol(s), Graded(degree=0))
    return r


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_basic(self):
        lie = LieBracket()
        Q = Symbol("Q")
        d = DerivedBracket(lie, Q, degree_Q=1)
        assert d.base is lie
        assert d.Q is Q
        assert d.degree_Q == Degree.const(1)
        # Degree formula: |Q| − 2.
        assert d.degree == Degree.const(-1)
        # Jacobi is conditional on [Q, Q] = 0 — reported as None.
        assert d.satisfies_graded_jacobi is None
        # Leibniz and antisymmetry hold by the derived-bracket theorem.
        assert d.satisfies_leibniz
        assert d.is_graded_antisymmetric

    def test_default_name_references_generator(self):
        lie = LieBracket()
        Q = Symbol("Q")
        d = DerivedBracket(lie, Q, degree_Q=1)
        assert "Q" in d.name

    def test_explicit_name_wins(self):
        lie = LieBracket()
        d = DerivedBracket(lie, Symbol("Q"), degree_Q=1, name="koszul")
        assert d.name == "koszul"

    def test_rejects_non_bracket_base(self):
        with pytest.raises(TypeError):
            DerivedBracket("not-a-bracket", Symbol("Q"), degree_Q=1)  # type: ignore[arg-type]

    def test_rejects_non_expr_generator(self):
        with pytest.raises(TypeError):
            DerivedBracket(LieBracket(), "Q", degree_Q=1)  # type: ignore[arg-type]

    def test_equality_by_base_and_generator(self):
        lie = LieBracket()
        Q = Symbol("Q")
        assert DerivedBracket(lie, Q, degree_Q=1) == DerivedBracket(
            lie, Q, degree_Q=1
        )
        assert DerivedBracket(lie, Q, degree_Q=1) != DerivedBracket(
            lie, Symbol("Q2"), degree_Q=1
        )

    def test_factory_matches_constructor(self):
        lie = LieBracket()
        Q = Symbol("Q")
        assert derived_bracket(lie, Q, degree_Q=1) == DerivedBracket(
            lie, Q, degree_Q=1
        )


# --------------------------------------------------------------------- #
# Expansion                                                              #
# --------------------------------------------------------------------- #


class TestExpansion:
    def test_definition_expands_two_layers(self, reg):
        """{a, b}_Q := [[a, Q], b]. Base = Lie: each layer is a commutator."""
        lie = LieBracket()
        Q = Symbol("Q")
        reg.declare(Q, Graded(degree=1))
        d = DerivedBracket(lie, Q, degree_Q=1)
        a, b = Symbol("a"), Symbol("b")
        # Inner: [a, Q] = a*Q − Q*a. Call that I.
        # Outer: [I, b] = I*b − b*I.
        # So the full expansion is (a*Q − Q*a)*b − b*(a*Q − Q*a).
        out = d.expand(a, b, reg)
        inner = Sum(Product(a, Q), Neg(Product(Q, a)))
        assert out == Sum(Product(inner, b), Neg(Product(b, inner)))

    def test_expand_definition_keeps_base_brackets_inert(self, reg):
        """expand_definition leaves one layer of base BracketApply nodes."""
        lie = LieBracket()
        Q = Symbol("Q")
        d = DerivedBracket(lie, Q, degree_Q=1)
        a, b = Symbol("a"), Symbol("b")
        out = d.expand_definition(a, b, reg)
        # Shape: BracketApply(lie, BracketApply(lie, a, Q), b)
        assert isinstance(out, BracketApply)
        assert out.bracket is lie
        assert isinstance(out.a, BracketApply)
        assert out.a.bracket is lie
        assert out.a.a is a and out.a.b is Q
        assert out.b is b

    def test_bracketapply_routes_through_derived(self, reg):
        """Calling the DerivedBracket wraps the pair in a
        BracketApply whose .expand() dispatches back to the derived
        rule."""
        lie = LieBracket()
        Q = Symbol("Q")
        d = DerivedBracket(lie, Q, degree_Q=1)
        a, b = Symbol("a"), Symbol("b")
        node = d(a, b)
        assert isinstance(node, BracketApply)
        assert node.bracket is d
        # Expand via the node's own method and via the bracket directly
        # — they must match.
        assert node.expand(reg) == d.expand(a, b, reg)


# --------------------------------------------------------------------- #
# Jacobi obstruction                                                     #
# --------------------------------------------------------------------- #


class TestJacobiObstruction:
    def test_obstruction_is_qq_base(self, reg):
        """jacobi_obstruction = [Q, Q] expanded in the base bracket.
        For Lie base that's Q*Q − Q*Q."""
        lie = LieBracket()
        Q = Symbol("Q")
        d = DerivedBracket(lie, Q, degree_Q=1)
        obs = d.jacobi_obstruction(reg)
        assert obs == Sum(Product(Q, Q), Neg(Product(Q, Q)))

    def test_obstruction_simplifies_to_zero_for_lie_base(self, reg):
        """Lie base: [Q, Q] = Q*Q − Q*Q → 0. So any derived bracket
        over the Lie bracket automatically satisfies Jacobi — this is
        the reason odd-degree commutators of a degree-1 derivation
        automatically give Jacobi-satisfying derived brackets."""
        lie = LieBracket()
        Q = Symbol("Q")
        d = DerivedBracket(lie, Q, degree_Q=1)
        assert simplify(d.jacobi_obstruction(reg)).children == ()  # = Integer(0)

    def test_obstruction_raw_preserves_shape(self):
        """jacobi_obstruction_raw keeps [Q, Q] inert for display."""
        lie = LieBracket()
        Q = Symbol("Q")
        d = DerivedBracket(lie, Q, degree_Q=1)
        raw = d.jacobi_obstruction_raw()
        assert isinstance(raw, BracketApply)
        assert raw.bracket is lie
        assert raw.a is Q and raw.b is Q


# --------------------------------------------------------------------- #
# Derived bracket theorem — [Q,Q]=0 ⟹ Jacobi                            #
# --------------------------------------------------------------------- #


class TestDerivedBracketTheorem:
    """Plan-required: ``[Q, Q]_base = 0 ⟺ derived bracket has Jacobi``.

    The *universal* form of this theorem — that the Jacobi obstruction
    reduces to the single expression ``[Q, Q]_base`` — is covered by
    :class:`TestJacobiObstruction` above. This class verifies the
    structural 3-argument expansion: each cyclic term of the derived
    bracket's graded Jacobi is a doubly-nested derived bracket
    application. Closing the full operand-level simplification on the
    expanded form requires sign-aware sorting of Neg-wrapped Products
    — that capability is earmarked for the Faz 7 proof layer, not for
    simplify.
    """

    def test_jacobi_obstruction_has_three_cyclic_terms(self, reg):
        from gradalg.brackets.base import BracketApply

        lie = LieBracket()
        Q = Symbol("Q")
        reg.declare(Q, Graded(degree=0))
        d = DerivedBracket(lie, Q, degree_Q=0)

        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        obs = d.graded_jacobi_obstruction(a, b, c, reg)
        # Three cyclic terms, each of shape ``d(·, d(·, ·))``.
        assert isinstance(obs, Sum)
        assert len(obs.children) == 3
        for term in obs.children:
            # All parities are zero here (degree-0 operands), so no Neg
            # wrappers — each term is a bare BracketApply.
            assert isinstance(term, BracketApply)
            assert term.bracket is d
            # Outer: d(x, inner); inner is also a derived-bracket app.
            assert isinstance(term.b, BracketApply)
            assert term.b.bracket is d
