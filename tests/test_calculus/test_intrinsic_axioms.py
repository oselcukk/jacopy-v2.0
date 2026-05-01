"""Tests for intrinsic Cartan operator axioms (Faz 12.A.1+)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algebra.lie_bracket_vf import lie_bracket_vf
from jacopy.algorithms.simplify import simplify
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.intrinsic_axioms import (
    ExteriorDIntrinsicDefinition,
    InteriorProductIntrinsicDefinition,
    LieDerivativeIntrinsicDefinition,
)
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.multi_eval_axioms import (
    MultiEvalArgLinearityDefinition,
    MultiEvalRepeatArgZeroDefinition,
)
from jacopy.core.expr import Neg, Sum, Symbol
from jacopy.core.multi_eval import multi_eval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# InteriorProductIntrinsicDefinition                                    #
# --------------------------------------------------------------------- #


class TestInteriorProductIntrinsicMatches:
    def test_matches_iota_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = InteriorProductIntrinsicDefinition()
        assert rule.matches(multi_eval(interior(X)(omega), Y))

    def test_no_match_plain_head(self):
        omega = Symbol("ω")
        X = Derivation("X", 0)
        rule = InteriorProductIntrinsicDefinition()
        assert not rule.matches(multi_eval(omega, X))

    def test_no_match_d_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = InteriorProductIntrinsicDefinition()
        # d ω as head, different operator, shouldn't fire.
        assert not rule.matches(multi_eval(default_d(omega), X, Y))

    def test_no_match_covector_slot_kind(self):
        # Bivector-on-covectors evaluation, interior product doesn't
        # apply, the rule must stay inert.
        omega = Symbol("ω")
        alpha = Symbol("α")
        X = Derivation("X", 0)
        rule = InteriorProductIntrinsicDefinition()
        node = multi_eval(
            interior(X)(omega),
            alpha,
            slot_kind="covector",
        )
        assert not rule.matches(node)


class TestInteriorProductIntrinsicRewrite:
    def test_two_form_one_remaining_arg(self):
        # (ι_X ω)(Y) = ω(X, Y)
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = InteriorProductIntrinsicDefinition()
        out = rule.rewrite(multi_eval(interior(X)(omega), Y))
        assert out == multi_eval(omega, X, Y)

    def test_three_form_two_remaining_args(self):
        # (ι_X ω)(Y, Z) = ω(X, Y, Z)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        rule = InteriorProductIntrinsicDefinition()
        out = rule.rewrite(multi_eval(interior(X)(omega), Y, Z))
        assert out == multi_eval(omega, X, Y, Z)

    def test_preserves_alternating_flag(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = InteriorProductIntrinsicDefinition()
        out = rule.rewrite(
            multi_eval(interior(X)(omega), Y, alternating=False)
        )
        assert isinstance(out, type(multi_eval(omega, X, Y)))
        assert out.alternating is False

    def test_preserves_slot_kind(self):
        # vector slot stays vector after rewrite (covector is rejected
        # at match time, so this only confirms the non-pathological path).
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = InteriorProductIntrinsicDefinition()
        out = rule.rewrite(multi_eval(interior(X)(omega), Y))
        assert out.slot_kind == "vector"

    def test_works_with_compound_form(self):
        # Form is itself an Act, e.g. d β at the head, the rewrite
        # treats it structurally, no special unwrapping.
        beta = Symbol("β")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        rule = InteriorProductIntrinsicDefinition()
        head = interior(X)(default_d(beta))
        out = rule.rewrite(multi_eval(head, Y, Z))
        assert out == multi_eval(default_d(beta), X, Y, Z)


# --------------------------------------------------------------------- #
# Engine integration                                                    #
# --------------------------------------------------------------------- #


class TestEngineIntegration:
    def test_engine_unfolds_single_iota(self):
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        engine = ExpansionEngine([InteriorProductIntrinsicDefinition()])
        out, steps = engine.expand(
            multi_eval(interior(X)(omega), Y, Z)
        )
        assert out == multi_eval(omega, X, Y, Z)
        assert len(steps) == 1

    def test_engine_unfolds_nested_iota(self):
        # (ι_X (ι_Y ω))(Z) → (ι_Y ω)(X, Z) → ω(Y, X, Z)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        engine = ExpansionEngine([InteriorProductIntrinsicDefinition()])
        out, steps = engine.expand(
            multi_eval(interior(X)(interior(Y)(omega)), Z)
        )
        assert out == multi_eval(omega, Y, X, Z)
        assert len(steps) == 2

    def test_repeat_kills_after_intrinsic(self):
        # (ι_X ω)(X) → ω(X, X) → 0   when alternating
        omega = Symbol("ω")
        X = Derivation("X", 0)
        engine = ExpansionEngine(
            [
                InteriorProductIntrinsicDefinition(),
                MultiEvalRepeatArgZeroDefinition(),
            ]
        )
        out, _ = engine.expand(multi_eval(interior(X)(omega), X))
        # Engine fix-points to 0.
        from jacopy.core.expr import Zero
        assert out is Zero

    def test_combines_with_arg_linearity(self):
        # (ι_X ω)(Y + Z) → (ι_X ω)(Y) + (ι_X ω)(Z) → ω(X,Y) + ω(X,Z)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        engine = ExpansionEngine(
            [
                InteriorProductIntrinsicDefinition(),
                MultiEvalArgLinearityDefinition(),
            ]
        )
        out, _ = engine.expand(
            multi_eval(interior(X)(omega), Sum(Y, Z))
        )
        result = simplify(out, PropertyRegistry())
        assert result == Sum(
            multi_eval(omega, X, Y), multi_eval(omega, X, Z)
        )


# --------------------------------------------------------------------- #
# LieDerivativeIntrinsicDefinition                                      #
# --------------------------------------------------------------------- #


class TestLieDerivativeIntrinsicMatches:
    def test_matches_lie_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = LieDerivativeIntrinsicDefinition()
        assert rule.matches(multi_eval(lie_derivative(X)(omega), Y))

    def test_no_match_plain_head(self):
        omega = Symbol("ω")
        X = Derivation("X", 0)
        rule = LieDerivativeIntrinsicDefinition()
        assert not rule.matches(multi_eval(omega, X))

    def test_no_match_iota_head(self):
        # ι_X is not L_X, different operator family.
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = LieDerivativeIntrinsicDefinition()
        assert not rule.matches(multi_eval(interior(X)(omega), Y))

    def test_no_match_d_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = LieDerivativeIntrinsicDefinition()
        assert not rule.matches(multi_eval(default_d(omega), X, Y))

    def test_no_match_covector_slot_kind(self):
        omega = Symbol("ω")
        alpha = Symbol("α")
        X = Derivation("X", 0)
        rule = LieDerivativeIntrinsicDefinition()
        node = multi_eval(
            lie_derivative(X)(omega),
            alpha,
            slot_kind="covector",
        )
        assert not rule.matches(node)


class TestLieDerivativeIntrinsicRewrite:
    def test_one_form_one_arg(self):
        # (L_X ω)(Y) = X(ω(Y)) − ω([X, Y]_VF)
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = LieDerivativeIntrinsicDefinition()
        out = rule.rewrite(multi_eval(lie_derivative(X)(omega), Y))
        expected = Sum.make(
            Act(X, multi_eval(omega, Y)),
            Neg(multi_eval(omega, lie_bracket_vf(X, Y))),
        )
        assert out == expected

    def test_two_form_two_args(self):
        # (L_X ω)(Y, Z) = X(ω(Y, Z)) − ω([X,Y]_VF, Z) − ω(Y, [X,Z]_VF)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        rule = LieDerivativeIntrinsicDefinition()
        out = rule.rewrite(multi_eval(lie_derivative(X)(omega), Y, Z))
        expected = Sum.make(
            Act(X, multi_eval(omega, Y, Z)),
            Neg(multi_eval(omega, lie_bracket_vf(X, Y), Z)),
            Neg(multi_eval(omega, Y, lie_bracket_vf(X, Z))),
        )
        assert out == expected

    def test_three_form_three_bracket_terms(self):
        omega = Symbol("ω")
        X, Y, Z, W = (Derivation(s, 0) for s in ("X", "Y", "Z", "W"))
        rule = LieDerivativeIntrinsicDefinition()
        out = rule.rewrite(multi_eval(lie_derivative(X)(omega), Y, Z, W))
        # 1 vector-action term + 3 bracket corrections = 4-term Sum.
        assert isinstance(out, Sum)
        assert len(out.children) == 4

    def test_preserves_alternating_flag(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = LieDerivativeIntrinsicDefinition()
        out = rule.rewrite(
            multi_eval(lie_derivative(X)(omega), Y, alternating=False)
        )
        # Both inner MultiEvals, the X-action one and the bracket one,
        # carry the non-alternating flag forward.
        assert isinstance(out, Sum)
        first, neg = out.children
        assert first.arg.alternating is False
        assert neg.arg.alternating is False

    def test_preserves_slot_kind(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = LieDerivativeIntrinsicDefinition()
        out = rule.rewrite(multi_eval(lie_derivative(X)(omega), Y))
        first, neg = out.children
        assert first.arg.slot_kind == "vector"
        assert neg.arg.slot_kind == "vector"

    def test_works_with_compound_form(self):
        # Form is itself an Act, e.g. d β at the head.
        beta = Symbol("β")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = LieDerivativeIntrinsicDefinition()
        head = lie_derivative(X)(default_d(beta))
        out = rule.rewrite(multi_eval(head, Y))
        expected = Sum.make(
            Act(X, multi_eval(default_d(beta), Y)),
            Neg(multi_eval(default_d(beta), lie_bracket_vf(X, Y))),
        )
        assert out == expected


class TestLieDerivativeEngineIntegration:
    def test_engine_unfolds_single_lie(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        engine = ExpansionEngine([LieDerivativeIntrinsicDefinition()])
        out, steps = engine.expand(multi_eval(lie_derivative(X)(omega), Y))
        expected = Sum.make(
            Act(X, multi_eval(omega, Y)),
            Neg(multi_eval(omega, lie_bracket_vf(X, Y))),
        )
        assert out == expected
        assert len(steps) == 1

    def test_engine_combines_with_arg_linearity(self):
        # (L_X ω)(Y + Z) →
        #   X(ω(Y+Z)) − ω([X, Y+Z]_VF)
        # Bracket term keeps [X, Y+Z]_VF opaque (no bracket linearity
        # wired in here); the action term distributes through arg-linearity:
        #   X(ω(Y) + ω(Z)) is left as-is by Sum-inside-Act (no rule); but
        #   the inner MultiEval is what gets distributed.
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        engine = ExpansionEngine(
            [
                LieDerivativeIntrinsicDefinition(),
                MultiEvalArgLinearityDefinition(),
            ]
        )
        out, _ = engine.expand(
            multi_eval(lie_derivative(X)(omega), Sum(Y, Z))
        )
        # Inner ω(Y+Z) splits to ω(Y)+ω(Z) inside the Act.
        # The bracket term ω([X, Y+Z]_VF) stays as-is (LieBracketVF is
        # an opaque atom; its argument distribution is a separate axiom).
        result = simplify(out, PropertyRegistry())
        expected = Sum.make(
            Act(X, Sum(multi_eval(omega, Y), multi_eval(omega, Z))),
            Neg(multi_eval(omega, lie_bracket_vf(X, Sum(Y, Z)))),
        )
        assert result == expected

    def test_engine_combines_with_iota_intrinsic(self):
        # (L_X (ι_Y ω))(Z) →
        #   X((ι_Y ω)(Z)) − (ι_Y ω)([X, Z]_VF)
        # Then iota intrinsic absorbs Y into each remaining slot:
        #   X(ω(Y, Z)) − ω(Y, [X, Z]_VF)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        engine = ExpansionEngine(
            [
                LieDerivativeIntrinsicDefinition(),
                InteriorProductIntrinsicDefinition(),
            ]
        )
        head = lie_derivative(X)(interior(Y)(omega))
        out, _ = engine.expand(multi_eval(head, Z))
        expected = Sum.make(
            Act(X, multi_eval(omega, Y, Z)),
            Neg(multi_eval(omega, Y, lie_bracket_vf(X, Z))),
        )
        assert out == expected


# --------------------------------------------------------------------- #
# ExteriorDIntrinsicDefinition                                          #
# --------------------------------------------------------------------- #


class TestExteriorDIntrinsicMatches:
    def test_matches_d_head_two_args(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = ExteriorDIntrinsicDefinition()
        assert rule.matches(multi_eval(default_d(omega), X, Y))

    def test_matches_arity_one(self):
        # (df)(X) = X(f); arity-1 case is handled (no inner MultiEval wrap).
        f = Symbol("f")
        X = Derivation("X", 0)
        rule = ExteriorDIntrinsicDefinition()
        assert rule.matches(multi_eval(default_d(f), X))

    def test_no_match_plain_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = ExteriorDIntrinsicDefinition()
        assert not rule.matches(multi_eval(omega, X, Y))

    def test_no_match_iota_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = ExteriorDIntrinsicDefinition()
        assert not rule.matches(multi_eval(interior(X)(omega), Y))

    def test_no_match_covector_slot_kind(self):
        omega = Symbol("ω")
        alpha, beta = Symbol("α"), Symbol("β")
        rule = ExteriorDIntrinsicDefinition()
        node = multi_eval(
            default_d(omega), alpha, beta, slot_kind="covector"
        )
        assert not rule.matches(node)


class TestExteriorDIntrinsicRewrite:
    def test_two_args_one_form(self):
        # ω is a 1-form; (dω)(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y]_VF)
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = ExteriorDIntrinsicDefinition()
        out = rule.rewrite(multi_eval(default_d(omega), X, Y))
        expected = Sum.make(
            Act(X, multi_eval(omega, Y)),
            Neg(Act(Y, multi_eval(omega, X))),
            Neg(multi_eval(omega, lie_bracket_vf(X, Y))),
        )
        assert out == expected

    def test_three_args_two_form(self):
        # ω is a 2-form; (dω)(X, Y, Z) = three vector terms + three bracket terms
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        rule = ExteriorDIntrinsicDefinition()
        out = rule.rewrite(multi_eval(default_d(omega), X, Y, Z))
        expected = Sum.make(
            # Vector-action sum: +X(ω(Y,Z)) − Y(ω(X,Z)) + Z(ω(X,Y))
            Act(X, multi_eval(omega, Y, Z)),
            Neg(Act(Y, multi_eval(omega, X, Z))),
            Act(Z, multi_eval(omega, X, Y)),
            # Bracket sum:
            #   (i,j)=(0,1) sign=−: −ω([X,Y]_VF, Z)
            #   (i,j)=(0,2) sign=+: +ω([X,Z]_VF, Y)
            #   (i,j)=(1,2) sign=−: −ω([Y,Z]_VF, X)
            Neg(multi_eval(omega, lie_bracket_vf(X, Y), Z)),
            multi_eval(omega, lie_bracket_vf(X, Z), Y),
            Neg(multi_eval(omega, lie_bracket_vf(Y, Z), X)),
        )
        assert out == expected

    def test_term_count_grows_with_arity(self):
        # n+1 vector-action terms + C(n+1, 2) bracket terms.
        omega = Symbol("ω")
        Xs = [Derivation(f"X{i}", 0) for i in range(4)]  # X_0..X_3
        rule = ExteriorDIntrinsicDefinition()
        out = rule.rewrite(multi_eval(default_d(omega), *Xs))
        # arity 4 → 4 vector-action + C(4,2)=6 bracket = 10 terms.
        assert isinstance(out, Sum)
        assert len(out.children) == 10

    def test_one_arg_zero_form(self):
        # (df)(X) = X(f); single Act, no inner MultiEval wrap, no bracket.
        f = Symbol("f")
        X = Derivation("X", 0)
        rule = ExteriorDIntrinsicDefinition()
        out = rule.rewrite(multi_eval(default_d(f), X))
        assert out == Sum.make(Act(X, f))

    def test_one_arg_inner_iota(self):
        # (d(ι_X ω))(Y) = Y(ι_X ω); the inner 0-form is preserved verbatim.
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        iota_omega = Act(interior(X), omega)
        rule = ExteriorDIntrinsicDefinition()
        out = rule.rewrite(multi_eval(default_d(iota_omega), Y))
        assert out == Sum.make(Act(Y, iota_omega))

    def test_preserves_alternating_flag(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = ExteriorDIntrinsicDefinition()
        out = rule.rewrite(
            multi_eval(default_d(omega), X, Y, alternating=False)
        )
        # Inner MultiEvals all carry the non-alternating flag.
        assert isinstance(out, Sum)
        for child in out.children:
            inner = child.arg if isinstance(child, Neg) else child
            # inner is either Act(X_i, MultiEval) or MultiEval.
            multi = inner.arg if isinstance(inner, Act) else inner
            assert multi.alternating is False

    def test_preserves_slot_kind(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = ExteriorDIntrinsicDefinition()
        out = rule.rewrite(multi_eval(default_d(omega), X, Y))
        for child in out.children:
            inner = child.arg if isinstance(child, Neg) else child
            multi = inner.arg if isinstance(inner, Act) else inner
            assert multi.slot_kind == "vector"


class TestExteriorDEngineIntegration:
    def test_engine_unfolds_single_d_two_form(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        engine = ExpansionEngine([ExteriorDIntrinsicDefinition()])
        out, steps = engine.expand(multi_eval(default_d(omega), X, Y))
        expected = Sum.make(
            Act(X, multi_eval(omega, Y)),
            Neg(Act(Y, multi_eval(omega, X))),
            Neg(multi_eval(omega, lie_bracket_vf(X, Y))),
        )
        assert out == expected
        assert len(steps) == 1

    def test_engine_combines_with_iota_intrinsic(self):
        # (d (ι_Z ω))(X, Y), ω is a 2-form, so ι_Z ω is a 1-form, and
        # d(ι_Z ω) is a 2-form. The d intrinsic fires first:
        #   X((ι_Z ω)(Y)) − Y((ι_Z ω)(X)) − (ι_Z ω)([X, Y]_VF)
        # then the ι intrinsic absorbs Z into each remaining slot:
        #   X(ω(Z, Y)) − Y(ω(Z, X)) − ω(Z, [X, Y]_VF)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        engine = ExpansionEngine(
            [
                ExteriorDIntrinsicDefinition(),
                InteriorProductIntrinsicDefinition(),
            ]
        )
        head = default_d(interior(Z)(omega))
        out, _ = engine.expand(multi_eval(head, X, Y))
        expected = Sum.make(
            Act(X, multi_eval(omega, Z, Y)),
            Neg(Act(Y, multi_eval(omega, Z, X))),
            Neg(multi_eval(omega, Z, lie_bracket_vf(X, Y))),
        )
        assert out == expected

    def test_engine_combines_with_arg_linearity(self):
        # (dω)(X + Y, Z) for a 1-form ω →
        #   (dω)(X, Z) + (dω)(Y, Z)
        # then each expands by Koszul:
        #   X(ω(Z)) − Z(ω(X)) − ω([X, Z]_VF)
        # + Y(ω(Z)) − Z(ω(Y)) − ω([Y, Z]_VF)
        # ArgLinearity must run *first* (top-down), but the engine is
        # bottom-up, so the d-intrinsic on the compound first arg fires
        # and gets [X+Y, Z]_VF as an opaque atom; ArgLinearity then
        # distributes only inside the inner MultiEvals it can see.
        # The simpler invariant we assert: simplify gives a flat Sum
        # whose terms include the correct bracket-with-Sum.
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        engine = ExpansionEngine(
            [
                ExteriorDIntrinsicDefinition(),
                MultiEvalArgLinearityDefinition(),
            ]
        )
        out, _ = engine.expand(
            multi_eval(default_d(omega), Sum(X, Y), Z)
        )
        result = simplify(out, PropertyRegistry())
        # Bracket term keeps [X+Y, Z]_VF opaque (no bracket linearity
        # axiom wired in here); the inner ω(Z) and ω(X+Y) terms split
        # by arg-linearity where applicable.
        expected = Sum.make(
            Act(Sum(X, Y), multi_eval(omega, Z)),
            Neg(
                Act(
                    Z,
                    Sum(multi_eval(omega, X), multi_eval(omega, Y)),
                )
            ),
            Neg(multi_eval(omega, lie_bracket_vf(Sum(X, Y), Z))),
        )
        assert result == expected


# --------------------------------------------------------------------- #
# Q9 Stage 9.E, Koszul intrinsic d̃                                     #
# --------------------------------------------------------------------- #


class TestKoszulExteriorDIntrinsic:
    """Connection-parametric d̃ rule routes the function action through
    ``connection.function_action`` and emits a ``BracketApply`` of the
    connection's bracket, both essential to closing Cartan I/II on a
    Koszul connection.
    """

    def _bracketed_connection(self):
        from jacopy.calculus.anchor import Anchor
        from jacopy.calculus.connection import koszul_connection
        from jacopy.brackets.koszul import KoszulBracket

        anchor = Anchor(name="ρ")
        bracket = KoszulBracket(anchor)
        return koszul_connection("∇̃", anchor=anchor, bracket=bracket), bracket

    def test_arity_one_emits_anchored_act(self):
        from jacopy.algebra.derivation import Derivation, Act
        from jacopy.calculus.anchor import AnchoredVectorField
        from jacopy.calculus.exterior_d import d
        from jacopy.calculus.intrinsic_axioms import (
            KoszulExteriorDIntrinsicDefinition,
        )
        from jacopy.core.expr import Symbol
        from jacopy.core.multi_eval import MultiEval

        conn, _ = self._bracketed_connection()
        rule = KoszulExteriorDIntrinsicDefinition(conn)
        omega = Symbol("ω")
        X = Derivation("X", 0)
        expr = MultiEval(Act(d, omega), X, slot_kind="vector")
        out = rule.rewrite(expr)
        # Σ over a single arg: term = function_action(X, ω) = Act(ρ(X), ω).
        assert out == Act(AnchoredVectorField(conn.anchor, X), omega)

    def test_arity_two_uses_connection_bracket(self):
        from jacopy.algebra.derivation import Derivation, Act
        from jacopy.brackets.base import BracketApply
        from jacopy.calculus.exterior_d import d
        from jacopy.calculus.intrinsic_axioms import (
            KoszulExteriorDIntrinsicDefinition,
        )
        from jacopy.core.expr import Symbol
        from jacopy.core.multi_eval import MultiEval

        conn, bracket = self._bracketed_connection()
        rule = KoszulExteriorDIntrinsicDefinition(conn)
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        expr = MultiEval(Act(d, omega), X, Y, slot_kind="vector")
        out = rule.rewrite(expr)
        # Last summand carries ω([X, Y]_K), the BracketApply uses the
        # connection's bracket, not LieBracketVF.
        bracket_summand = out.children[-1]
        # bracket summand is Neg(MultiEval(ω, [X,Y]_K)), sign (i+j)%2=1
        # for (0, 1).
        from jacopy.core.expr import Neg
        assert isinstance(bracket_summand, Neg)
        inner = bracket_summand.arg
        assert isinstance(inner, MultiEval)
        assert isinstance(inner.args[0], BracketApply)
        assert inner.args[0].bracket is bracket

    def test_rejects_connection_without_bracket(self):
        from jacopy.calculus.connection import connection
        from jacopy.calculus.intrinsic_axioms import (
            KoszulExteriorDIntrinsicDefinition,
        )
        import pytest

        nabla = connection("∇")  # no bracket
        with pytest.raises(ValueError):
            KoszulExteriorDIntrinsicDefinition(nabla)
