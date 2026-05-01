"""Tests for the Faz 12.A.5 intrinsic-formula proof surface."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algebra.lie_bracket_vf import lie_bracket_vf
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.intrinsic_engine import (
    IntrinsicFormulaMatch,
    IntrinsicFormulaRecognizer,
    intrinsic_engine,
    prove_intrinsic_equivalence,
)
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Neg, Sum, Symbol
from jacopy.core.multi_eval import multi_eval
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.strategies import ProofFailure


# --------------------------------------------------------------------- #
# intrinsic_engine factory                                               #
# --------------------------------------------------------------------- #


class TestIntrinsicEngineFactory:
    def test_returns_expansion_engine(self):
        eng = intrinsic_engine()
        assert isinstance(eng, ExpansionEngine)

    def test_bundles_eight_rules(self):
        eng = intrinsic_engine()
        assert len(eng.definitions) == 8

    def test_rule_order_intrinsics_first(self):
        # Intrinsic rules come before multi-eval helpers so a
        # MultiEval(Act(op, ω), …) gets matched by the operator-specific
        # rule before head-linearity has a chance to scan it.
        eng = intrinsic_engine()
        names = [type(d).__name__ for d in eng.definitions]
        assert names.index("InteriorProductIntrinsicDefinition") < names.index(
            "MultiEvalHeadLinearityDefinition"
        )
        assert names.index("LieDerivativeIntrinsicDefinition") < names.index(
            "MultiEvalHeadLinearityDefinition"
        )
        assert names.index("ExteriorDIntrinsicDefinition") < names.index(
            "MultiEvalHeadLinearityDefinition"
        )

    def test_each_call_returns_fresh_engine(self):
        # Engines are stateful (carry their own step buffers in some
        # configurations); the factory should hand out independent
        # instances so concurrent / nested proofs don't share state.
        eng1 = intrinsic_engine()
        eng2 = intrinsic_engine()
        assert eng1 is not eng2

    def test_engine_expands_iota(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        eng = intrinsic_engine()
        out, steps = eng.expand(multi_eval(Act(interior(X), omega), Y))
        assert out == multi_eval(omega, X, Y)
        assert len(steps) == 1


# --------------------------------------------------------------------- #
# IntrinsicFormulaRecognizer                                             #
# --------------------------------------------------------------------- #


class TestRecognizerInterior:
    def test_recognizes_iota_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rec = IntrinsicFormulaRecognizer()
        m = rec.recognize(multi_eval(Act(interior(X), omega), Y))
        assert m is not None
        assert m.operator == "interior"
        assert m.vector_field == X
        assert m.omega == omega
        assert m.args == (Y,)

    def test_classify_iota(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rec = IntrinsicFormulaRecognizer()
        assert rec.classify(multi_eval(Act(interior(X), omega), Y)) == "interior"


class TestRecognizerLie:
    def test_recognizes_lie_head(self):
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        rec = IntrinsicFormulaRecognizer()
        m = rec.recognize(multi_eval(Act(lie_derivative(X), omega), Y, Z))
        assert m is not None
        assert m.operator == "lie"
        assert m.vector_field == X
        assert m.args == (Y, Z)

    def test_classify_lie(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rec = IntrinsicFormulaRecognizer()
        assert rec.classify(
            multi_eval(Act(lie_derivative(X), omega), Y)
        ) == "lie"


class TestRecognizerExteriorD:
    def test_recognizes_d_head(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rec = IntrinsicFormulaRecognizer()
        m = rec.recognize(multi_eval(Act(default_d, omega), X, Y))
        assert m is not None
        assert m.operator == "exterior_d"
        assert m.vector_field is None
        assert m.omega == omega
        assert m.args == (X, Y)

    def test_classify_d(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rec = IntrinsicFormulaRecognizer()
        assert rec.classify(
            multi_eval(Act(default_d, omega), X, Y)
        ) == "exterior_d"


class TestRecognizerNonMatch:
    def test_plain_head_returns_none(self):
        omega = Symbol("ω")
        X = Derivation("X", 0)
        rec = IntrinsicFormulaRecognizer()
        assert rec.recognize(multi_eval(omega, X)) is None
        assert rec.classify(multi_eval(omega, X)) is None

    def test_non_multieval_returns_none(self):
        omega = Symbol("ω")
        rec = IntrinsicFormulaRecognizer()
        assert rec.recognize(omega) is None
        assert rec.classify(omega) is None

    def test_act_with_non_intrinsic_op_returns_none(self):
        # Act(SomeDerivation, ω) with a plain Derivation op (not ι/L/d)
        #, the recognizer should walk away cleanly.
        omega = Symbol("ω")
        X = Derivation("X", 0)
        rec = IntrinsicFormulaRecognizer()
        assert rec.recognize(multi_eval(Act(X, omega), X)) is None

    def test_nested_intrinsic_only_matches_outer(self):
        # MultiEval(Act(L_X, Act(ι_Y, ω)), Z), outer head is L_X.
        # The recognizer reports L_X with omega = Act(ι_Y, ω); it does
        # NOT unwrap further. Re-recognition on the inner is the
        # caller's job.
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        rec = IntrinsicFormulaRecognizer()
        inner = Act(interior(Y), omega)
        m = rec.recognize(multi_eval(Act(lie_derivative(X), inner), Z))
        assert m is not None
        assert m.operator == "lie"
        assert m.vector_field == X
        assert m.omega == inner  # not unwrapped


class TestRecognizerMatchDataclass:
    def test_match_carries_alternating_and_slot_kind(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        rec = IntrinsicFormulaRecognizer()
        m = rec.recognize(
            multi_eval(
                Act(interior(X), omega),
                Y,
                alternating=False,
                slot_kind="vector",
            )
        )
        assert m.alternating is False
        assert m.slot_kind == "vector"

    def test_match_is_frozen_dataclass(self):
        m = IntrinsicFormulaMatch(
            operator="interior",
            vector_field=Symbol("X"),
            omega=Symbol("ω"),
            args=(),
            alternating=True,
            slot_kind="vector",
        )
        with pytest.raises(Exception):
            m.operator = "lie"  # frozen dataclass rejects mutation


# --------------------------------------------------------------------- #
# prove_intrinsic_equivalence                                            #
# --------------------------------------------------------------------- #


class TestProveEquivalenceBasics:
    def test_reflexive_returns_single_step_chain(self):
        omega = Symbol("ω")
        chain = prove_intrinsic_equivalence(omega, omega)
        assert isinstance(chain, ProofChain)
        assert len(chain.steps) == 1
        assert chain.steps[0].rule == "reflexive"

    def test_iota_intrinsic_one_step(self):
        # (ι_X ω)(Y) = ω(X, Y)
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        lhs = multi_eval(Act(interior(X), omega), Y)
        rhs = multi_eval(omega, X, Y)
        chain = prove_intrinsic_equivalence(lhs, rhs)
        assert chain.steps[-1].rule in {"simplify", "ι_X intrinsic: (ι_X ω)(Y_1, …) = ω(X, Y_1, …)"}

    def test_lie_intrinsic_on_one_form(self):
        # (L_X ω)(Y) = X(ω(Y)) − ω([X,Y]_VF)
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        lhs = multi_eval(Act(lie_derivative(X), omega), Y)
        rhs = Sum(
            Act(X, multi_eval(omega, Y)),
            Neg(multi_eval(omega, lie_bracket_vf(X, Y))),
        )
        chain = prove_intrinsic_equivalence(lhs, rhs)
        assert len(chain.steps) >= 1


class TestProveEquivalenceCartanRelations:
    def test_cartan_magic_two_form(self):
        # (ι_X d + d ι_X)ω = L_X ω on a 2-form, evaluated on (Y, Z)
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        lhs = Sum(
            multi_eval(Act(interior(X), Act(default_d, omega)), Y, Z),
            multi_eval(Act(default_d, Act(interior(X), omega)), Y, Z),
        )
        rhs = multi_eval(Act(lie_derivative(X), omega), Y, Z)
        chain = prove_intrinsic_equivalence(lhs, rhs)
        # Magic closed in 11 engine steps + 1 simplify (12.A.4 sanity check)
        assert len(chain.steps) >= 1
        assert isinstance(chain, ProofChain)

    def test_iota_iota_anticommute_two_form(self):
        # (ι_X ι_Y + ι_Y ι_X) ω = 0 on a 2-form, evaluated on Z
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        lhs = Sum(
            multi_eval(interior(X)(interior(Y)(omega)), Z),
            multi_eval(interior(Y)(interior(X)(omega)), Z),
        )
        from jacopy.core.expr import Integer
        chain = prove_intrinsic_equivalence(lhs, Integer(0))
        assert len(chain.steps) >= 1

    def test_iota_squared_zero(self):
        # ι_X ι_X ω = 0
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        from jacopy.core.expr import Integer
        lhs = multi_eval(Act(interior(X), Act(interior(X), omega)), Y)
        chain = prove_intrinsic_equivalence(lhs, Integer(0))
        assert len(chain.steps) >= 1

    def test_l_iota_commutator_two_form(self):
        # [L_X, ι_Y] ω = ι_{[X,Y]_VF} ω on a 2-form, evaluated on Z
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        XY = lie_bracket_vf(X, Y)
        lhs = Sum(
            multi_eval(Act(lie_derivative(X), Act(interior(Y), omega)), Z),
            Neg(multi_eval(Act(interior(Y), Act(lie_derivative(X), omega)), Z)),
        )
        rhs = multi_eval(Act(interior(XY), omega), Z)
        chain = prove_intrinsic_equivalence(lhs, rhs)
        assert len(chain.steps) >= 1


class TestProveEquivalenceFailure:
    def test_open_d_squared_raises(self):
        # d² = 0 on a 1-form, evaluated on (X, Y, Z), does NOT close
        # without 12.A.6 axioms; prove_intrinsic_equivalence should
        # raise ProofFailure with the residue in the message.
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        from jacopy.core.expr import Integer
        lhs = multi_eval(Act(default_d, Act(default_d, omega)), X, Y, Z)
        with pytest.raises(ProofFailure) as ei:
            prove_intrinsic_equivalence(lhs, Integer(0))
        msg = str(ei.value)
        assert "residual" in msg

    def test_bogus_rhs_raises(self):
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        bogus = Symbol("not-a-cartan-thing")
        with pytest.raises(ProofFailure):
            prove_intrinsic_equivalence(
                multi_eval(Act(interior(X), omega), Y),
                bogus,
            )


class TestProveEquivalenceCustomEngine:
    def test_passing_engine_overrides_default(self):
        # If a caller passes a stripped-down engine (only the iota rule),
        # Cartan magic won't close, confirms the engine kwarg is honoured.
        from jacopy.calculus.intrinsic_axioms import (
            InteriorProductIntrinsicDefinition,
        )
        omega = Symbol("ω")
        X, Y, Z = (Derivation(s, 0) for s in ("X", "Y", "Z"))
        stripped = ExpansionEngine(
            [InteriorProductIntrinsicDefinition()]
        )
        lhs = Sum(
            multi_eval(Act(interior(X), Act(default_d, omega)), Y, Z),
            multi_eval(Act(default_d, Act(interior(X), omega)), Y, Z),
        )
        rhs = multi_eval(Act(lie_derivative(X), omega), Y, Z)
        with pytest.raises(ProofFailure):
            prove_intrinsic_equivalence(lhs, rhs, engine=stripped)

    def test_chain_steps_have_proofstep_shape(self):
        # Each step in the chain should expose .before/.after/.rule.
        omega = Symbol("ω")
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        chain = prove_intrinsic_equivalence(
            multi_eval(Act(interior(X), omega), Y),
            multi_eval(omega, X, Y),
        )
        for step in chain.steps:
            assert hasattr(step, "before")
            assert hasattr(step, "after")
            assert hasattr(step, "rule")
