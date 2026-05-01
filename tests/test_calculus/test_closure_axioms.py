"""Tests for Faz 12.A.6 closure axioms + intrinsic_engine_with_closure."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algebra.lie_bracket_vf import LieBracketVF, lie_bracket_vf
from jacopy.calculus.closure_axioms import (
    IotaActAsScalarDefinition,
    LieBracketVfAntiSymmetryDefinition,
    LieBracketVfJacobiDefinition,
    VfActCommutatorDefinition,
    _extract_bracket_with_wrapper,
    _peel_lie_bracket_jacobi,
)
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.intrinsic_engine import (
    intrinsic_engine_with_closure,
    prove_intrinsic_equivalence,
)
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Integer, Neg, Sum, Symbol
from jacopy.core.multi_eval import multi_eval
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# VfActCommutatorDefinition                                              #
# --------------------------------------------------------------------- #


class TestVfActCommutator:
    def test_matches_simple_pair(self):
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        f = Symbol("f")
        rule = VfActCommutatorDefinition()
        expr = Sum(Act(X, Act(Y, f)), Neg(Act(Y, Act(X, f))))
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Act(LieBracketVF(X, Y), f)

    def test_matches_swapped_order(self):
        # Negated child first, positive second, still folds.
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        f = Symbol("f")
        rule = VfActCommutatorDefinition()
        expr = Sum(Neg(Act(Y, Act(X, f))), Act(X, Act(Y, f)))
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Act(LieBracketVF(X, Y), f)

    def test_keeps_extra_children(self):
        # A pair sits among unrelated terms; only the pair is consumed.
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        f, g = Symbol("f"), Symbol("g")
        rule = VfActCommutatorDefinition()
        expr = Sum(g, Act(X, Act(Y, f)), Neg(Act(Y, Act(X, f))))
        result = rule.rewrite(expr)
        # Result should have the bracket fold and the unchanged g.
        assert isinstance(result, Sum)
        assert Act(LieBracketVF(X, Y), f) in result.children
        assert g in result.children

    def test_skips_same_vf(self):
        # Act(X, Act(X, f)) − Act(X, Act(X, f)) is trivially zero;
        # the rule wouldn't gain anything from folding into [X,X]_VF.
        X = Derivation("X", 0)
        f = Symbol("f")
        rule = VfActCommutatorDefinition()
        expr = Sum(Act(X, Act(X, f)), Neg(Act(X, Act(X, f))))
        assert not rule.matches(expr)

    def test_skips_cartan_operators(self):
        # Generic VF-commutator deliberately doesn't fire on L_X / ι_X
        # / d, those have their own intrinsic axioms.
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        omega = Symbol("ω")
        rule = VfActCommutatorDefinition()
        expr = Sum(
            Act(lie_derivative(X), Act(lie_derivative(Y), omega)),
            Neg(Act(lie_derivative(Y), Act(lie_derivative(X), omega))),
        )
        assert not rule.matches(expr)


# --------------------------------------------------------------------- #
# LieBracketVfAntiSymmetryDefinition                                     #
# --------------------------------------------------------------------- #


class TestLieBracketAntiSymmetry:
    def test_cancels_bare_pair(self):
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        f = Symbol("f")
        rule = LieBracketVfAntiSymmetryDefinition()
        expr = Sum(
            Act(LieBracketVF(X, Y), f),
            Act(LieBracketVF(Y, X), f),
        )
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Integer(0)

    def test_cancels_inside_multi_eval_slot(self):
        # ω(W, [X,Y]) + ω(W, [Y,X]) = 0, same wrapper, opposite
        # bracket orientations.
        X, Y, W = (Derivation(s, 0) for s in ("X", "Y", "W"))
        omega = Symbol("ω")
        rule = LieBracketVfAntiSymmetryDefinition()
        expr = Sum(
            multi_eval(omega, W, LieBracketVF(X, Y)),
            multi_eval(omega, W, LieBracketVF(Y, X)),
        )
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Integer(0)

    def test_skips_opposite_signs(self):
        # +[X,Y](f) − [Y,X](f) = +[X,Y] + [X,Y] = 2[X,Y], not zero.
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        f = Symbol("f")
        rule = LieBracketVfAntiSymmetryDefinition()
        expr = Sum(
            Act(LieBracketVF(X, Y), f),
            Neg(Act(LieBracketVF(Y, X), f)),
        )
        assert not rule.matches(expr)

    def test_skips_different_wrappers(self):
        X, Y, W, Z = (Derivation(s, 0) for s in ("X", "Y", "W", "Z"))
        omega = Symbol("ω")
        rule = LieBracketVfAntiSymmetryDefinition()
        expr = Sum(
            multi_eval(omega, W, LieBracketVF(X, Y)),
            multi_eval(omega, Z, LieBracketVF(Y, X)),
        )
        assert not rule.matches(expr)


# --------------------------------------------------------------------- #
# LieBracketVfJacobiDefinition                                           #
# --------------------------------------------------------------------- #


class TestLieBracketJacobi:
    def test_cyclic_bare_form(self):
        # [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        f = Symbol("f")
        rule = LieBracketVfJacobiDefinition()
        expr = Sum(
            Act(LieBracketVF(X, LieBracketVF(Y, Z)), f),
            Act(LieBracketVF(Y, LieBracketVF(Z, X)), f),
            Act(LieBracketVF(Z, LieBracketVF(X, Y)), f),
        )
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Integer(0)

    def test_leibniz_form(self):
        # [X,[Y,Z]] − [Y,[X,Z]] − [[X,Y],Z] = 0
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        f = Symbol("f")
        rule = LieBracketVfJacobiDefinition()
        expr = Sum(
            Act(LieBracketVF(X, LieBracketVF(Y, Z)), f),
            Neg(Act(LieBracketVF(Y, LieBracketVF(X, Z)), f)),
            Neg(Act(LieBracketVF(LieBracketVF(X, Y), Z), f)),
        )
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Integer(0)

    def test_outer_anti_symmetric_form(self):
        # [[X,Y],Z] − [[X,Z],Y] + [[Y,Z],X] = 0, the d²=0 residue
        # pattern after outer-anti-symmetrising.
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        f = Symbol("f")
        rule = LieBracketVfJacobiDefinition()
        expr = Sum(
            Act(LieBracketVF(LieBracketVF(X, Y), Z), f),
            Neg(Act(LieBracketVF(LieBracketVF(X, Z), Y), f)),
            Act(LieBracketVF(LieBracketVF(Y, Z), X), f),
        )
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Integer(0)

    def test_inside_multi_eval_slot(self):
        # ω(W, [X,[Y,Z]]) + ω(W, [Y,[Z,X]]) + ω(W, [Z,[X,Y]]) = 0
        X, Y, Z, W = (Derivation(s, 0) for s in ("X", "Y", "Z", "W"))
        omega = Symbol("ω")
        rule = LieBracketVfJacobiDefinition()
        expr = Sum(
            multi_eval(omega, W, LieBracketVF(X, LieBracketVF(Y, Z))),
            multi_eval(omega, W, LieBracketVF(Y, LieBracketVF(Z, X))),
            multi_eval(omega, W, LieBracketVF(Z, LieBracketVF(X, Y))),
        )
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Integer(0)

    def test_keeps_unrelated_terms(self):
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        f, g = Symbol("f"), Symbol("g")
        rule = LieBracketVfJacobiDefinition()
        expr = Sum(
            g,
            Act(LieBracketVF(X, LieBracketVF(Y, Z)), f),
            Act(LieBracketVF(Y, LieBracketVF(Z, X)), f),
            Act(LieBracketVF(Z, LieBracketVF(X, Y)), f),
        )
        assert rule.rewrite(expr) == g

    def test_skips_two_term_partial(self):
        # Only two cyclic terms, not enough for Jacobi.
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        f = Symbol("f")
        rule = LieBracketVfJacobiDefinition()
        expr = Sum(
            Act(LieBracketVF(X, LieBracketVF(Y, Z)), f),
            Act(LieBracketVF(Y, LieBracketVF(Z, X)), f),
        )
        assert not rule.matches(expr)


class TestPeelHelpers:
    def test_peel_inner_right(self):
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        variants = _peel_lie_bracket_jacobi(LieBracketVF(X, LieBracketVF(Y, Z)))
        # Two variants: (+1, X, Y, Z) and (−1, X, Z, Y).
        assert (+1, X, Y, Z) in variants
        assert (-1, X, Z, Y) in variants

    def test_peel_inner_left(self):
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        variants = _peel_lie_bracket_jacobi(LieBracketVF(LieBracketVF(X, Y), Z))
        # [[X,Y],Z] = -[Z,[X,Y]] → (-1, Z, X, Y); inner anti-sym
        # gives (+1, Z, Y, X).
        assert (-1, Z, X, Y) in variants
        assert (+1, Z, Y, X) in variants

    def test_peel_unnested(self):
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        # [X, Y] is depth-1, not depth-2, no Jacobi variants.
        assert _peel_lie_bracket_jacobi(LieBracketVF(X, Y)) == ()

    def test_extract_bare(self):
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        bracket, _ = _extract_bracket_with_wrapper(LieBracketVF(X, Y))
        assert bracket == LieBracketVF(X, Y)

    def test_extract_act_op(self):
        # Act(LieBracketVF(X, Y), f), bracket is the operator.
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        f = Symbol("f")
        bracket, _ = _extract_bracket_with_wrapper(Act(LieBracketVF(X, Y), f))
        assert bracket == LieBracketVF(X, Y)

    def test_extract_skips_two_brackets(self):
        # If two args of a MultiEval each contain a bracket, the
        # wrapper is ambiguous, return None rather than picking one.
        X, Y, Z, W = (Derivation(s, 0) for s in ("X", "Y", "Z", "W"))
        omega = Symbol("ω")
        expr = multi_eval(omega, LieBracketVF(X, Y), LieBracketVF(Z, W))
        bracket, _ = _extract_bracket_with_wrapper(expr)
        assert bracket is None


# --------------------------------------------------------------------- #
# IotaActAsScalarDefinition                                              #
# --------------------------------------------------------------------- #


class TestIotaActAsScalar:
    def test_matches_plain_vf_outer(self):
        # Act(Y, Act(ι_X, ω)), Y plain VF, inner is ι_X(ω).
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = IotaActAsScalarDefinition()
        assert rule.matches(Act(Y, Act(interior(X), omega)))

    def test_no_match_outer_d(self):
        # Act(d, Act(ι_X, ω)), outer is exterior derivative; d treats
        # ι_X(ω) as a form, not a scalar. Don't bridge here.
        omega = Symbol("ω")
        X = Derivation("X", 0)
        rule = IotaActAsScalarDefinition()
        assert not rule.matches(Act(default_d, Act(interior(X), omega)))

    def test_no_match_outer_lie(self):
        # Act(L_Y, Act(ι_X, ω)), outer Lie derivative is a Cartan op.
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = IotaActAsScalarDefinition()
        assert not rule.matches(Act(lie_derivative(Y), Act(interior(X), omega)))

    def test_no_match_outer_iota(self):
        # Act(ι_Y, Act(ι_X, ω)), both layers are interior products.
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = IotaActAsScalarDefinition()
        assert not rule.matches(Act(interior(Y), Act(interior(X), omega)))

    def test_no_match_inner_not_iota(self):
        # Act(Y, Act(Z, f)), inner Act has a plain VF, not ι_X.
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        f = Symbol("f")
        rule = IotaActAsScalarDefinition()
        assert not rule.matches(Act(Y, Act(Z, f)))

    def test_no_match_bare_iota_act(self):
        # Just Act(ι_X, ω), no outer Act. Rule needs the outer scalar
        # context to fire (the in-MultiEval case is the iota-intrinsic's
        # responsibility).
        omega = Symbol("ω")
        X = Derivation("X", 0)
        rule = IotaActAsScalarDefinition()
        assert not rule.matches(Act(interior(X), omega))

    def test_rewrites_to_act_multieval(self):
        # Act(Y, Act(ι_X, ω)) → Act(Y, MultiEval(ω, X)).
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rule = IotaActAsScalarDefinition()
        out = rule.rewrite(Act(Y, Act(interior(X), omega)))
        assert out == Act(Y, multi_eval(omega, X))

    def test_rewrites_with_lie_bracket_outer(self):
        # LieBracketVF acts as a plain VF on a scalar; bridge should fire.
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        rule = IotaActAsScalarDefinition()
        bracket = LieBracketVF(Y, Z)
        out = rule.rewrite(Act(bracket, Act(interior(X), omega)))
        assert out == Act(bracket, multi_eval(omega, X))


# --------------------------------------------------------------------- #
# intrinsic_engine_with_closure factory                                  #
# --------------------------------------------------------------------- #


class TestClosureEngineFactory:
    def test_returns_engine(self):
        eng = intrinsic_engine_with_closure()
        assert isinstance(eng, ExpansionEngine)

    def test_bundles_twelve_rules(self):
        # 8 base + 3 closure + 1 iota-bridge = 12
        eng = intrinsic_engine_with_closure()
        assert len(eng.definitions) == 12

    def test_includes_closure_rules(self):
        eng = intrinsic_engine_with_closure()
        names = [type(d).__name__ for d in eng.definitions]
        assert "VfActCommutatorDefinition" in names
        assert "LieBracketVfAntiSymmetryDefinition" in names
        assert "LieBracketVfJacobiDefinition" in names
        assert "IotaActAsScalarDefinition" in names

    def test_each_call_returns_fresh_engine(self):
        eng1 = intrinsic_engine_with_closure()
        eng2 = intrinsic_engine_with_closure()
        assert eng1 is not eng2


# --------------------------------------------------------------------- #
# End-to-end Cartan-relation closures (the headline of 12.A.6)            #
# --------------------------------------------------------------------- #


class TestCartanRelationsClose:
    """Each open Faz 12.A.4 relation closes with the closure engine."""

    def test_d_squared_one_form(self):
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        lhs = multi_eval(Act(default_d, Act(default_d, omega)), X, Y, Z)
        chain = prove_intrinsic_equivalence(
            lhs, Integer(0), engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_d_squared_two_form(self):
        omega = Symbol("ω")
        X, Y, Z, W = (Derivation(s, 0) for s in ("X", "Y", "Z", "W"))
        lhs = multi_eval(Act(default_d, Act(default_d, omega)), X, Y, Z, W)
        chain = prove_intrinsic_equivalence(
            lhs, Integer(0), engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_d_lie_commutator_one_form(self):
        # [d, L_X] ω = 0 on a 1-form, eval (Y, Z)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        lhs = Sum(
            multi_eval(Act(default_d, Act(lie_derivative(X), omega)), Y, Z),
            Neg(multi_eval(Act(lie_derivative(X), Act(default_d, omega)), Y, Z)),
        )
        chain = prove_intrinsic_equivalence(
            lhs, Integer(0), engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_d_lie_commutator_two_form(self):
        omega = Symbol("ω")
        X, Y, Z, W = (Derivation(s, 0) for s in ("X", "Y", "Z", "W"))
        lhs = Sum(
            multi_eval(Act(default_d, Act(lie_derivative(X), omega)), Y, Z, W),
            Neg(
                multi_eval(
                    Act(lie_derivative(X), Act(default_d, omega)), Y, Z, W
                )
            ),
        )
        chain = prove_intrinsic_equivalence(
            lhs, Integer(0), engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_lie_lie_commutator_one_form(self):
        # [L_X, L_Y] ω = L_{[X,Y]_VF} ω, eval Z
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        XY = lie_bracket_vf(X, Y)
        lhs = Sum(
            multi_eval(Act(lie_derivative(X), Act(lie_derivative(Y), omega)), Z),
            Neg(
                multi_eval(
                    Act(lie_derivative(Y), Act(lie_derivative(X), omega)), Z
                )
            ),
        )
        rhs = multi_eval(Act(lie_derivative(XY), omega), Z)
        chain = prove_intrinsic_equivalence(
            lhs, rhs, engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_lie_lie_commutator_two_form(self):
        omega = Symbol("ω")
        X, Y, Z, W = (Derivation(s, 0) for s in ("X", "Y", "Z", "W"))
        XY = lie_bracket_vf(X, Y)
        lhs = Sum(
            multi_eval(
                Act(lie_derivative(X), Act(lie_derivative(Y), omega)), Z, W
            ),
            Neg(
                multi_eval(
                    Act(lie_derivative(Y), Act(lie_derivative(X), omega)), Z, W
                )
            ),
        )
        rhs = multi_eval(Act(lie_derivative(XY), omega), Z, W)
        chain = prove_intrinsic_equivalence(
            lhs, rhs, engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1


class TestPreviouslyClosingRelationsStillClose:
    """The four 12.A.4 relations don't regress under the richer engine."""

    def test_cartan_magic_one_form(self):
        # 1-form ω, eval at single Y: (ι_X d + d ι_X)(ω)(Y) = L_X(ω)(Y).
        # Closes via the arity-1 d-intrinsic branch + IotaActAsScalar bridge.
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        lhs = Sum(
            multi_eval(Act(interior(X), Act(default_d, omega)), Y),
            multi_eval(Act(default_d, Act(interior(X), omega)), Y),
        )
        rhs = multi_eval(Act(lie_derivative(X), omega), Y)
        chain = prove_intrinsic_equivalence(
            lhs, rhs, engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_cartan_magic_two_form(self):
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        lhs = Sum(
            multi_eval(Act(interior(X), Act(default_d, omega)), Y, Z),
            multi_eval(Act(default_d, Act(interior(X), omega)), Y, Z),
        )
        rhs = multi_eval(Act(lie_derivative(X), omega), Y, Z)
        chain = prove_intrinsic_equivalence(
            lhs, rhs, engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_iota_squared_zero(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        lhs = multi_eval(Act(interior(X), Act(interior(X), omega)), Y)
        chain = prove_intrinsic_equivalence(
            lhs, Integer(0), engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_iota_iota_anticommute_two_form(self):
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        lhs = Sum(
            multi_eval(interior(X)(interior(Y)(omega)), Z),
            multi_eval(interior(Y)(interior(X)(omega)), Z),
        )
        chain = prove_intrinsic_equivalence(
            lhs, Integer(0), engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1

    def test_l_iota_commutator_two_form(self):
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        XY = lie_bracket_vf(X, Y)
        lhs = Sum(
            multi_eval(Act(lie_derivative(X), Act(interior(Y), omega)), Z),
            Neg(multi_eval(Act(interior(Y), Act(lie_derivative(X), omega)), Z)),
        )
        rhs = multi_eval(Act(interior(XY), omega), Z)
        chain = prove_intrinsic_equivalence(
            lhs, rhs, engine=intrinsic_engine_with_closure()
        )
        assert len(chain) >= 1
