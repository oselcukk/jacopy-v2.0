"""Tests for LocalFrame + KroneckerDelta + duality, Faz 17.A."""

from __future__ import annotations

import pytest

from jacopy.calculus.local_frame import (
    FrameCovector,
    FrameIndex,
    FramePairingDualityDefinition,
    FrameVectorField,
    KroneckerDelta,
    LocalFrame,
    local_frame,
)
from jacopy.calculus.pairing import Pairing, pairing
from jacopy.calculus.pairing_axioms import PairingLinearityDefinition
from jacopy.core.expr import One, Sum
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# FrameIndex                                                            #
# --------------------------------------------------------------------- #


def test_frame_index_default_is_free():
    a = FrameIndex("a")
    assert a.name == "a"
    assert a.kind == "free"
    assert a.is_free is True
    assert a.is_bound is False


def test_frame_index_bound_kind():
    b = FrameIndex("b", "bound")
    assert b.kind == "bound"
    assert b.is_bound is True
    assert b.is_free is False


def test_frame_index_equality_includes_kind():
    a_free = FrameIndex("a")
    a_dup = FrameIndex("a")
    a_bound = FrameIndex("a", "bound")
    b_free = FrameIndex("b")
    assert a_free == a_dup
    assert hash(a_free) == hash(a_dup)
    assert a_free != a_bound  # kind differs
    assert a_free != b_free   # name differs


def test_frame_index_repr_shows_hat_for_bound():
    a = FrameIndex("a")
    a_bound = FrameIndex("a", "bound")
    assert a._repr_inner() == "a"
    assert "a" in a_bound._repr_inner()
    assert a_bound._repr_inner() != a._repr_inner()


def test_frame_index_rejects_invalid():
    with pytest.raises(TypeError):
        FrameIndex(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FrameIndex("")
    with pytest.raises(ValueError):
        FrameIndex("a", "magic")


def test_frame_index_is_an_atom():
    a = FrameIndex("a")
    assert a.is_atom
    assert a.children == ()


# --------------------------------------------------------------------- #
# KroneckerDelta                                                        #
# --------------------------------------------------------------------- #


def test_kronecker_free_equal_collapses_to_one():
    a = FrameIndex("a")
    delta = KroneckerDelta(a, a)
    assert delta is One


def test_kronecker_free_distinct_stays_inert():
    a = FrameIndex("a")
    b = FrameIndex("b")
    delta = KroneckerDelta(a, b)
    assert isinstance(delta, KroneckerDelta)
    assert delta.i == a
    assert delta.j == b
    assert delta.children == (a, b)


def test_kronecker_bound_bound_stays_inert_even_if_same_name():
    b1 = FrameIndex("b", "bound")
    b2 = FrameIndex("b", "bound")
    delta = KroneckerDelta(b1, b2)
    assert isinstance(delta, KroneckerDelta)


def test_kronecker_mixed_kind_stays_inert():
    a_free = FrameIndex("a", "free")
    a_bound = FrameIndex("a", "bound")
    delta = KroneckerDelta(a_free, a_bound)
    assert isinstance(delta, KroneckerDelta)


def test_kronecker_repr_contains_delta():
    a = FrameIndex("a")
    b = FrameIndex("b")
    text = KroneckerDelta(a, b)._repr_inner()
    assert "δ" in text
    assert "a" in text
    assert "b" in text


def test_kronecker_rejects_non_index():
    with pytest.raises(TypeError):
        KroneckerDelta("a", "b")  # type: ignore[arg-type]
    a = FrameIndex("a")
    with pytest.raises(TypeError):
        KroneckerDelta(a, "b")  # type: ignore[arg-type]


def test_kronecker_equality_and_hash():
    a = FrameIndex("a")
    b = FrameIndex("b")
    c = FrameIndex("c")
    assert KroneckerDelta(a, b) == KroneckerDelta(a, b)
    assert hash(KroneckerDelta(a, b)) == hash(KroneckerDelta(a, b))
    assert KroneckerDelta(a, b) != KroneckerDelta(a, c)
    # Order matters for asymmetric δ^i_j; structurally they differ.
    assert KroneckerDelta(a, b) != KroneckerDelta(b, a)


def test_kronecker_rebuild_roundtrips():
    a = FrameIndex("a")
    b = FrameIndex("b")
    delta = KroneckerDelta(a, b)
    rebuilt = delta._rebuild((a, b))
    assert rebuilt == delta


# --------------------------------------------------------------------- #
# LocalFrame wrapper                                                    #
# --------------------------------------------------------------------- #


def test_local_frame_basic_attrs():
    F = LocalFrame("F", dim=4)
    assert F.name == "F"
    assert F.dim == 4
    assert F.vf_symbol == "X"
    assert F.coframe_symbol == "e"


def test_local_frame_default_factory():
    F = local_frame()
    assert F.name == "X"
    assert F.dim is None


def test_local_frame_custom_symbols():
    F = LocalFrame("G", vf_symbol="Y", coframe_symbol="f")
    assert F.X("a")._repr_inner() == "Y_a"
    assert F.coframe("a")._repr_inner() == "f^a"


def test_local_frame_equality_by_payload():
    F1 = LocalFrame("F", dim=3)
    F2 = LocalFrame("F", dim=3)
    F3 = LocalFrame("F", dim=4)
    F4 = LocalFrame("F'", dim=3)
    assert F1 == F2
    assert hash(F1) == hash(F2)
    assert F1 != F3
    assert F1 != F4


def test_local_frame_rejects_invalid_name():
    with pytest.raises(TypeError):
        LocalFrame(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LocalFrame("")


def test_local_frame_rejects_invalid_dim():
    with pytest.raises(ValueError):
        LocalFrame("F", dim=0)
    with pytest.raises(ValueError):
        LocalFrame("F", dim=-2)
    with pytest.raises(TypeError):
        LocalFrame("F", dim="four")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LocalFrame("F", dim=True)  # type: ignore[arg-type]


def test_local_frame_index_helper():
    F = LocalFrame("F")
    a = F.index("a")
    assert isinstance(a, FrameIndex)
    assert a.is_free
    b = F.index("b", bound=True)
    assert b.is_bound


def test_local_frame_repr_shows_relevant_kwargs():
    F = LocalFrame("F", dim=3, vf_symbol="Y")
    text = repr(F)
    assert "F" in text and "dim=3" in text and "Y" in text


# --------------------------------------------------------------------- #
# FrameVectorField                                                      #
# --------------------------------------------------------------------- #


def test_frame_vf_creation():
    F = LocalFrame("F")
    Xa = F.X("a")
    assert isinstance(Xa, FrameVectorField)
    assert Xa.frame == F
    assert Xa.idx.name == "a"
    assert Xa.idx.is_free


def test_frame_vf_is_a_derivation():
    from jacopy.algebra.derivation import Derivation
    F = LocalFrame("F")
    Xa = F.X("a")
    assert isinstance(Xa, Derivation)


def test_frame_vf_repr_uses_frame_symbol():
    F = LocalFrame("F", vf_symbol="Y")
    assert F.X("a")._repr_inner() == "Y_a"


def test_frame_vf_equality_by_frame_name_and_idx():
    F = LocalFrame("F")
    F_dup = LocalFrame("F")
    G = LocalFrame("G")
    assert F.X("a") == F_dup.X("a")
    assert F.X("a") != F.X("b")
    assert F.X("a") != G.X("a")


def test_frame_vf_accepts_explicit_index():
    F = LocalFrame("F")
    a = FrameIndex("a", "bound")
    Xa = F.X(a)
    assert Xa.idx.is_bound


def test_frame_vf_rejects_bad_args():
    F = LocalFrame("F")
    with pytest.raises(TypeError):
        F.X(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# FrameCovector                                                         #
# --------------------------------------------------------------------- #


def test_frame_covector_creation():
    F = LocalFrame("F")
    ea = F.coframe("a")
    assert isinstance(ea, FrameCovector)
    assert ea.frame == F
    assert ea.idx.name == "a"
    assert ea.children == ()


def test_frame_covector_repr_uses_frame_symbol():
    F = LocalFrame("F", coframe_symbol="θ")
    assert F.coframe("a")._repr_inner() == "θ^a"


def test_frame_covector_equality_by_frame_name_and_idx():
    F = LocalFrame("F")
    F_dup = LocalFrame("F")
    G = LocalFrame("G")
    assert F.coframe("a") == F_dup.coframe("a")
    assert F.coframe("a") != F.coframe("b")
    assert F.coframe("a") != G.coframe("a")


def test_frame_covector_rejects_bad_args():
    F = LocalFrame("F")
    with pytest.raises(TypeError):
        F.coframe(99)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Duality axiom, matches                                               #
# --------------------------------------------------------------------- #


def test_duality_matches_well_formed_pairing():
    F = LocalFrame("F")
    rule = F.duality_definition()
    pair = pairing(F.coframe("a"), F.X("b"))
    assert rule.matches(pair)


def test_duality_rejects_non_pairing_inputs():
    F = LocalFrame("F")
    rule = F.duality_definition()
    assert not rule.matches(F.X("b"))
    assert not rule.matches(F.coframe("a"))


def test_duality_rejects_swapped_slot_order():
    F = LocalFrame("F")
    rule = F.duality_definition()
    # Pairing with VF in alpha-slot is structurally legal but not the
    # duality shape, rule should not fire.
    bad = Pairing(F.X("a"), F.coframe("b"))
    assert not rule.matches(bad)


def test_duality_rejects_cross_frame_pairing():
    F = LocalFrame("F")
    G = LocalFrame("G")
    rule = F.duality_definition()
    assert not rule.matches(pairing(F.coframe("a"), G.X("b")))
    assert not rule.matches(pairing(G.coframe("a"), F.X("b")))


# --------------------------------------------------------------------- #
# Duality axiom, rewrite                                               #
# --------------------------------------------------------------------- #


def test_duality_rewrites_to_kronecker():
    F = LocalFrame("F")
    rule = F.duality_definition()
    pair = pairing(F.coframe("a"), F.X("b"))
    res = rule.rewrite(pair)
    assert isinstance(res, KroneckerDelta)
    assert res.i.name == "a"
    assert res.j.name == "b"


def test_duality_collapses_when_indices_match():
    F = LocalFrame("F")
    rule = F.duality_definition()
    pair = pairing(F.coframe("a"), F.X("a"))
    res = rule.rewrite(pair)
    assert res is One


def test_duality_constructor_rejects_non_local_frame():
    with pytest.raises(TypeError):
        FramePairingDualityDefinition("not-a-frame")  # type: ignore[arg-type]


def test_duality_name_carries_frame_id():
    F = LocalFrame("F")
    rule = F.duality_definition()
    assert "F" in rule.name


# --------------------------------------------------------------------- #
# Engine integration                                                    #
# --------------------------------------------------------------------- #


def test_engine_reduces_pairing_to_kronecker():
    F = LocalFrame("F")
    engine = ExpansionEngine([F.duality_definition()])
    expr = pairing(F.coframe("a"), F.X("b"))
    final, steps = engine.expand(expr)
    assert isinstance(final, KroneckerDelta)
    assert len(steps) == 1


def test_engine_reduces_to_one_when_indices_match():
    F = LocalFrame("F")
    engine = ExpansionEngine([F.duality_definition()])
    expr = pairing(F.coframe("a"), F.X("a"))
    final, steps = engine.expand(expr)
    assert final is One
    assert len(steps) == 1


def test_engine_two_frames_no_cross_fire():
    F = LocalFrame("F")
    G = LocalFrame("G")
    engine = ExpansionEngine([F.duality_definition()])
    expr = pairing(F.coframe("a"), G.X("a"))
    final, steps = engine.expand(expr)
    assert final == expr
    assert len(steps) == 0


def test_engine_distributes_through_pairing_sum_then_collapses():
    F = LocalFrame("F")
    engine = ExpansionEngine(
        [
            PairingLinearityDefinition(),
            F.duality_definition(),
        ]
    )
    expr = pairing(
        F.coframe("a"),
        Sum.make(F.X("b"), F.X("c")),
    )
    final, steps = engine.expand(expr)
    assert isinstance(final, Sum)
    # Both summand pairings reduce to KroneckerDelta nodes.
    assert all(
        isinstance(c, KroneckerDelta) for c in final.children
    )
    indices = sorted(c.j.name for c in final.children)
    assert indices == ["b", "c"]


def test_engine_distributes_alpha_sum_then_collapses():
    F = LocalFrame("F")
    engine = ExpansionEngine(
        [
            PairingLinearityDefinition(),
            F.duality_definition(),
        ]
    )
    expr = pairing(
        Sum.make(F.coframe("a"), F.coframe("b")),
        F.X("c"),
    )
    final, _ = engine.expand(expr)
    assert isinstance(final, Sum)
    assert all(isinstance(c, KroneckerDelta) for c in final.children)


def test_engine_collapses_diagonal_inside_sum_to_one():
    F = LocalFrame("F")
    engine = ExpansionEngine(
        [
            PairingLinearityDefinition(),
            F.duality_definition(),
        ]
    )
    # ⟨e^a, X_a + X_b⟩ → δ^a_a + δ^a_b → 1 + δ^a_b
    expr = pairing(
        F.coframe("a"),
        Sum.make(F.X("a"), F.X("b")),
    )
    final, _ = engine.expand(expr)
    assert isinstance(final, Sum)
    assert One in final.children
    assert any(isinstance(c, KroneckerDelta) for c in final.children)
