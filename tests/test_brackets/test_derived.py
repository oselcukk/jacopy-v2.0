"""Tests for jacopy.brackets.derived."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.base import BracketApply
from jacopy.brackets.derived import DerivedBracket, derived_bracket
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.lie import LieBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.anchor import Anchor
from jacopy.calculus.exterior_d import d as d_op
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.pairing import Pairing, pairing
from jacopy.core.expr import Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


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
        # Jacobi is conditional on [Q, Q] = 0, reported as None.
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
        #, they must match.
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
        over the Lie bracket automatically satisfies Jacobi, this is
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
# Derived bracket theorem, [Q,Q]=0 ⟹ Jacobi                            #
# --------------------------------------------------------------------- #


class TestDerivedBracketTheorem:
    """Plan-required: ``[Q, Q]_base = 0 ⟺ derived bracket has Jacobi``.

    The *universal* form of this theorem, that the Jacobi obstruction
    reduces to the single expression ``[Q, Q]_base``, is covered by
    :class:`TestJacobiObstruction` above. This class verifies the
    structural 3-argument expansion: each cyclic term of the derived
    bracket's graded Jacobi is a doubly-nested derived bracket
    application. Closing the full operand-level simplification on the
    expanded form requires sign-aware sorting of Neg-wrapped Products
   , that capability is earmarked for the Faz 7 proof layer, not for
    simplify.
    """

    def test_jacobi_obstruction_has_three_cyclic_terms(self, reg):
        from jacopy.brackets.base import BracketApply

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
            # wrappers, each term is a bare BracketApply.
            assert isinstance(term, BracketApply)
            assert term.bracket is d
            # Outer: d(x, inner); inner is also a derived-bracket app.
            assert isinstance(term.b, BracketApply)
            assert term.b.bracket is d


# --------------------------------------------------------------------- #
# VanishingCondition / jacobi_condition                                  #
# --------------------------------------------------------------------- #


class TestJacobiCondition:
    def test_returns_vanishing_condition(self, reg):
        from jacopy.brackets.derived import VanishingCondition
        lie = LieBracket()
        d = DerivedBracket(lie, Symbol("Q"), degree_Q=1)
        cond = d.jacobi_condition(reg)
        assert isinstance(cond, VanishingCondition)

    def test_condition_obstruction_matches_expanded_form(self, reg):
        lie = LieBracket()
        d = DerivedBracket(lie, Symbol("Q"), degree_Q=1)
        cond = d.jacobi_condition(reg)
        assert cond.obstruction == d.jacobi_obstruction(reg)

    def test_name_references_bracket(self, reg):
        lie = LieBracket()
        d = DerivedBracket(lie, Symbol("Q"), degree_Q=1, name="koszul")
        cond = d.jacobi_condition(reg)
        assert "koszul" in cond.name

    def test_holds_true_for_lie_base(self, reg):
        """LieBracket gives [Q, Q] = Q*Q - Q*Q = 0 → Jacobi holds."""
        reg.declare(Symbol("Q"), Graded(degree=1))
        lie = LieBracket()
        d = DerivedBracket(lie, Symbol("Q"), degree_Q=1)
        assert d.jacobi_condition(reg).holds(reg) is True

    def test_is_hashable_and_equatable(self, reg):
        """Frozen dataclass → usable as dict key."""
        lie = LieBracket()
        d = DerivedBracket(lie, Symbol("Q"), degree_Q=1)
        c1 = d.jacobi_condition(reg)
        c2 = d.jacobi_condition(reg)
        assert c1 == c2
        assert hash(c1) == hash(c2)


# --------------------------------------------------------------------- #
# acting_on: anchor-lifted derived bracket (Koszul case)                 #
# --------------------------------------------------------------------- #


class TestActingOn:
    def test_default_is_none(self):
        d = DerivedBracket(LieBracket(), Symbol("Q"), degree_Q=1)
        assert d.acting_on is None

    def test_stores_anchor(self):
        rho = Anchor("ρ")
        d = DerivedBracket(sn, Symbol("π"), degree_Q=1, acting_on=rho)
        assert d.acting_on is rho

    def test_rejects_non_derivation_acting_on(self):
        with pytest.raises(TypeError):
            DerivedBracket(
                sn, Symbol("π"), degree_Q=1, acting_on="ρ"  # type: ignore[arg-type]
            )

    def test_accepts_musical_sharp_as_anchor(self):
        """``Sharp`` is a ``Derivation`` too, any ``Derivation`` works."""
        from jacopy.calculus.musical import sharp
        pi = Symbol("π")
        sh = sharp(pi)
        d = DerivedBracket(sn, pi, degree_Q=1, acting_on=sh)
        assert d.acting_on is sh

    def test_expand_emits_koszul_three_term_sum(self):
        rho = Anchor("ρ")
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        d = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho)
        out = d.expand(alpha, beta)
        assert isinstance(out, Sum)
        assert len(out.children) == 3

    def test_expand_structurally_equals_classical_koszul(self):
        """The reason for the ``acting_on`` kwarg: derived-bracket form
        on 1-forms reproduces the classical Koszul bracket exactly."""
        rho = Anchor("ρ")
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        k_out = KoszulBracket(rho).expand(alpha, beta)
        d_out = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho).expand(alpha, beta)
        assert k_out == d_out

    def test_expand_first_term_is_L_rho_alpha_beta(self):
        rho = Anchor("ρ")
        alpha, beta = Symbol("α"), Symbol("β")
        d = DerivedBracket(sn, Symbol("π"), degree_Q=1, acting_on=rho)
        out = d.expand(alpha, beta)
        assert out.children[0] == Act(lie_derivative(Act(rho, alpha)), beta)

    def test_expand_third_term_wraps_pairing_in_d(self):
        rho = Anchor("ρ")
        alpha, beta = Symbol("α"), Symbol("β")
        d = DerivedBracket(sn, Symbol("π"), degree_Q=1, acting_on=rho)
        out = d.expand(alpha, beta)
        assert out.children[2] == Neg(Act(d_op, pairing(Act(rho, alpha), beta)))

    def test_acting_on_none_preserves_default_path(self, reg):
        """Classic {a,b}_Q = [[a,Q],b] path is not disturbed."""
        lie = LieBracket()
        Q = Symbol("Q")
        reg.declare(Q, Graded(degree=1))
        d = DerivedBracket(lie, Q, degree_Q=1)
        a, b = Symbol("a"), Symbol("b")
        default_expanded = d.expand(a, b, reg)
        manual = lie.expand(lie.expand(a, Q, reg), b, reg)
        assert default_expanded == manual

    def test_different_acting_on_distinguishes_brackets(self):
        """Anchor participates in identity, two derived brackets with
        different anchors are distinct."""
        pi = Symbol("π")
        d1 = DerivedBracket(sn, pi, degree_Q=1, acting_on=Anchor("ρ1"))
        d2 = DerivedBracket(sn, pi, degree_Q=1, acting_on=Anchor("ρ2"))
        assert d1 != d2

    def test_jacobi_obstruction_unchanged_by_acting_on(self, reg):
        """``acting_on`` only rewires ``expand``. The Jacobi obstruction
        is still ``[Q, Q]_base``."""
        rho = Anchor("ρ")
        Q = Symbol("Q")
        reg.declare(Q, Graded(degree=1))
        lie = LieBracket()
        d_plain = DerivedBracket(lie, Q, degree_Q=1)
        d_lift = DerivedBracket(lie, Q, degree_Q=1, acting_on=rho)
        assert d_lift.jacobi_obstruction(reg) == d_plain.jacobi_obstruction(reg)

    def test_degree_formula_unchanged_by_acting_on(self):
        """``|{·,·}_Q| = |Q| − 2`` regardless of ``acting_on``; the
        interpretation on forms is the caller's responsibility."""
        rho = Anchor("ρ")
        d = DerivedBracket(sn, Symbol("π"), degree_Q=1, acting_on=rho)
        assert d.degree == Degree.const(-1)
