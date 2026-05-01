"""Tests for jacopy.proof.strategies.UnrollToFoundations."""

import pytest

from jacopy.core.expr import Integer, Symbol
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import (
    Definition,
    ExpansionEngine,
    default_engine,
)
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import (
    ExpandAndSimplify,
    ProofFailure,
    Strategy,
    UnrollToFoundations,
)


class _StubTheorem(Definition):
    """Theorem rewriting Symbol('t') → Integer(0) with a stand-in sub-proof."""

    name = "stub-theorem"

    def matches(self, expr):
        return isinstance(expr, Symbol) and expr.name == "t"

    def rewrite(self, expr):
        return Integer(0)

    def theorem_proof_builder(self):
        def _build(matched):
            chain = ProofChain()
            chain.append(
                ProofStep(
                    matched,
                    Integer(0),
                    rule="stub-sub-proof",
                    justification="stand-in sub-proof",
                    provenance_tag="axiom",
                )
            )
            return chain
        return _build


class _CaptureEngineStrategy(Strategy):
    """Strategy that records the engine it receives, then closes trivially."""

    name = "capture-engine"

    def __init__(self) -> None:
        self.seen_engine = None

    def prove(self, lhs, rhs, *, registry=None, engine=None):
        self.seen_engine = engine
        chain = ProofChain()
        chain.append(
            ProofStep(lhs, rhs, rule=self.name, justification="noop"),
        )
        return chain


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_requires_inner_strategy(self):
        with pytest.raises(TypeError, match="Strategy"):
            UnrollToFoundations("not a strategy")  # type: ignore[arg-type]

    def test_inner_accessible(self):
        inner = ExpandAndSimplify()
        strat = UnrollToFoundations(inner)
        assert strat.inner is inner


# --------------------------------------------------------------------- #
# Engine-mode plumbing                                                   #
# --------------------------------------------------------------------- #


class TestEngineMode:
    def test_none_engine_defaults_to_foundational(self):
        capture = _CaptureEngineStrategy()
        UnrollToFoundations(capture).prove(Symbol("a"), Symbol("a"))
        assert capture.seen_engine is not None
        assert capture.seen_engine.mode == "foundational"

    def test_efficient_engine_is_switched(self):
        capture = _CaptureEngineStrategy()
        eff = default_engine(mode="efficient")
        UnrollToFoundations(capture).prove(
            Symbol("a"), Symbol("a"), engine=eff,
        )
        assert capture.seen_engine.mode == "foundational"
        # The switch should yield a new engine, not mutate the original.
        assert eff.mode == "efficient"

    def test_foundational_engine_passed_through(self):
        capture = _CaptureEngineStrategy()
        foundational = default_engine(mode="foundational")
        UnrollToFoundations(capture).prove(
            Symbol("a"), Symbol("a"), engine=foundational,
        )
        # Same instance, no wrapping when already foundational.
        assert capture.seen_engine is foundational


# --------------------------------------------------------------------- #
# End-to-end: theorem rewrite produces sub-proof under unroll            #
# --------------------------------------------------------------------- #


class TestEndToEnd:
    def test_theorem_sub_proof_appears_in_chain(self):
        # Engine with our stub theorem registered; inner ExpandAndSimplify
        # will form Sum(t, -t) and expand `t` → 0 via the theorem. Under
        # UnrollToFoundations, that step carries the sub-proof.
        eng = ExpansionEngine([_StubTheorem()], mode="efficient")
        strat = UnrollToFoundations(ExpandAndSimplify())
        chain = strat.prove(Symbol("t"), Integer(0), engine=eng)

        # Some step in the chain should be the theorem firing with its
        # sub-proof attached as children.
        theorem_steps = [
            s for s in chain
            if s.provenance_tag == "theorem" and s.children
        ]
        assert theorem_steps, "expected unrolled theorem step with children"
        assert theorem_steps[0].children[0].rule == "stub-sub-proof"

    def test_efficient_mode_strips_sub_proof(self):
        # Without UnrollToFoundations the same engine should leave the
        # theorem step childless.
        eng = ExpansionEngine([_StubTheorem()], mode="efficient")
        chain = ExpandAndSimplify().prove(Symbol("t"), Integer(0), engine=eng)
        theorem_steps = [s for s in chain if s.provenance_tag == "theorem"]
        assert theorem_steps
        for s in theorem_steps:
            assert s.children == ()


# --------------------------------------------------------------------- #
# Failure propagation                                                    #
# --------------------------------------------------------------------- #


class TestFailurePropagation:
    def test_inner_failure_reraised(self):
        with pytest.raises(ProofFailure):
            UnrollToFoundations(ExpandAndSimplify()).prove(
                Symbol("a"), Symbol("b"),
            )
