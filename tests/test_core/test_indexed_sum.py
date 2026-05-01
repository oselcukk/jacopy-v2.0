"""Tests for the :class:`IndexedSum` Expr node, Faz 17.E.1 + 17.E.2."""

from __future__ import annotations

import pytest

from jacopy.calculus.cartan_forms import ConnectionForm
from jacopy.calculus.connection import connection
from jacopy.calculus.local_frame import (
    FrameIndex,
    KroneckerDelta,
    LocalFrame,
)
from jacopy.core.expr import Neg, Product, Sum, Symbol, Zero
from jacopy.core.indexed_sum import IndexedSum, dummy_in, indexed_sum


# --------------------------------------------------------------------- #
# construction / type checking                                          #
# --------------------------------------------------------------------- #


def test_construct_with_frame_index_and_local_frame():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = F.X(b)
    s = IndexedSum(b, F, body)
    assert s.dummy is b
    assert s.range_ is F
    assert s.body is body


def test_construct_rejects_non_atom_dummy():
    F = LocalFrame("F")
    body = F.X(F.index("b", bound=True))
    with pytest.raises(TypeError):
        IndexedSum(Sum.make(Symbol("a"), Symbol("b")), F, body)  # type: ignore[arg-type]


def test_construct_rejects_non_expr_body():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    with pytest.raises(TypeError):
        IndexedSum(b, F, "not an expr")  # type: ignore[arg-type]


def test_indexed_sum_factory_function_matches_class():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = F.X(b)
    a = indexed_sum(b, F, body)
    c = IndexedSum(b, F, body)
    assert a == c


# --------------------------------------------------------------------- #
# children + _rebuild contract                                           #
# --------------------------------------------------------------------- #


def test_children_is_body_only():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = F.X(b)
    s = IndexedSum(b, F, body)
    assert s.children == (body,)


def test_rebuild_replaces_body_only():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = F.X(b)
    s = IndexedSum(b, F, body)
    new_body = F.coframe(b)
    s2 = s._rebuild((new_body,))
    assert s2.dummy is b
    assert s2.range_ is F
    assert s2.body is new_body


def test_rebuild_rejects_wrong_arity():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.X(b))
    with pytest.raises(ValueError):
        s._rebuild(())


# --------------------------------------------------------------------- #
# α-equivalence equality + hashing                                       #
# --------------------------------------------------------------------- #


def test_alpha_equal_with_different_dummy_names():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    c = F.index("c", bound=True)
    s_b = IndexedSum(b, F, F.X(b))
    s_c = IndexedSum(c, F, F.X(c))
    assert s_b == s_c
    assert hash(s_b) == hash(s_c)


def test_alpha_unequal_when_body_uses_free_index():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    s_dummy = IndexedSum(b, F, F.X(b))
    s_free = IndexedSum(b, F, F.X(a))
    assert s_dummy != s_free


def test_alpha_unequal_across_different_frames():
    F1 = LocalFrame("F1")
    F2 = LocalFrame("F2")
    b1 = F1.index("b", bound=True)
    b2 = F2.index("b", bound=True)
    s1 = IndexedSum(b1, F1, F1.X(b1))
    s2 = IndexedSum(b2, F2, F2.X(b2))
    assert s1 != s2


def test_alpha_canonical_handles_kronecker_delta():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    c = F.index("c", bound=True)
    s_b = IndexedSum(b, F, KroneckerDelta(a, b))
    s_c = IndexedSum(c, F, KroneckerDelta(a, c))
    assert s_b == s_c


def test_alpha_canonical_handles_connection_form():
    nabla = connection("∇")
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    c = F.index("c", bound=True)
    s_b = IndexedSum(b, F, ConnectionForm(nabla, F, a, b))
    s_c = IndexedSum(c, F, ConnectionForm(nabla, F, a, c))
    assert s_b == s_c


# --------------------------------------------------------------------- #
# nested IndexedSums (depth-aware sentinels)                             #
# --------------------------------------------------------------------- #


def test_nested_indexed_sums_alpha_equal():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    c = F.index("c", bound=True)
    p = F.index("p", bound=True)
    q = F.index("q", bound=True)
    inner_bc = IndexedSum(c, F, KroneckerDelta(b, c))
    inner_pq = IndexedSum(q, F, KroneckerDelta(p, q))
    outer_bc = IndexedSum(b, F, inner_bc)
    outer_pq = IndexedSum(p, F, inner_pq)
    assert outer_bc == outer_pq


def test_nested_indexed_sums_distinguish_correlated_dummies():
    """Σ_b Σ_c δ(b, c) and Σ_b Σ_c δ(b, b) must hash unequally."""
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    c = F.index("c", bound=True)
    cross = IndexedSum(b, F, IndexedSum(c, F, KroneckerDelta(b, c)))
    self_only = IndexedSum(b, F, IndexedSum(c, F, KroneckerDelta(b, b)))
    assert cross != self_only


# --------------------------------------------------------------------- #
# substitute_atom + with_dummy + substitute_dummy_with                  #
# --------------------------------------------------------------------- #


def test_substitute_atom_skips_into_body():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    free_target = F.index("x")
    s = IndexedSum(b, F, F.X(a))
    s2 = s.substitute_atom(a, free_target)
    assert s2.body == F.X(free_target)


def test_substitute_atom_respects_shadowing():
    F = LocalFrame("F")
    b1 = F.index("b", bound=True)
    b2 = F.index("b", bound=True)
    a = F.index("a")
    # Outer body references "b" via a name that the inner IS also binds.
    s = IndexedSum(b1, F, IndexedSum(b2, F, F.X(b1)))
    # Substituting `b` should NOT touch inside the inner IS (shadowed).
    s2 = s.substitute_atom(b1, a)
    # The structure of s2 should be: outer body unchanged.
    assert s2 == s


def test_with_dummy_returns_alpha_equivalent_form():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    new_dummy = F.index("d", bound=True)
    s = IndexedSum(b, F, F.X(b))
    s2 = s.with_dummy(new_dummy)
    assert s == s2
    assert s2.dummy == new_dummy
    assert s2.body == F.X(new_dummy)


def test_with_dummy_rejects_non_atom():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.X(b))
    with pytest.raises(TypeError):
        s.with_dummy(Sum.make(Symbol("x"), Symbol("y")))  # type: ignore[arg-type]


def test_substitute_dummy_with_returns_body_at_target():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.X(b))
    body_at_a = s.substitute_dummy_with(a)
    assert body_at_a == F.X(a)


def test_substitute_dummy_with_into_kronecker_collapses_when_free():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    # body = δ(a, b)
    s = IndexedSum(b, F, KroneckerDelta(a, b))
    # δ(a, a) collapses to One via KroneckerDelta.__new__
    body_at_a = s.substitute_dummy_with(a)
    from jacopy.core.expr import One
    assert body_at_a == One


# --------------------------------------------------------------------- #
# repr                                                                   #
# --------------------------------------------------------------------- #


def test_repr_contains_sigma_and_dummy_name():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.X(b))
    text = repr(s)
    assert "Σ" in text
    assert "b" in text
    assert "F" in text


# --------------------------------------------------------------------- #
# Sum/Neg in body don't collapse the IndexedSum                          #
# --------------------------------------------------------------------- #


def test_indexed_sum_with_sum_body():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = Sum.make(F.X(b), Neg(F.X(b)))
    s = IndexedSum(b, F, body)
    assert isinstance(s, IndexedSum)
    assert s.body == body


# --------------------------------------------------------------------- #
# dummy_in predicate                                                     #
# --------------------------------------------------------------------- #


def test_dummy_in_finds_direct_occurrence():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    assert dummy_in(b, b)


def test_dummy_in_finds_hidden_in_frame_vf():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    assert dummy_in(F.X(b), b)


def test_dummy_in_finds_hidden_in_frame_covector():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    assert dummy_in(F.coframe(b), b)


def test_dummy_in_finds_in_connection_form_lower():
    nabla = connection("∇")
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    om = ConnectionForm(nabla, F, a, b)
    assert dummy_in(om, b)
    assert not dummy_in(om, F.index("c", bound=True))


def test_dummy_in_misses_when_absent():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    assert not dummy_in(F.X(a), b)


def test_dummy_in_respects_shadowing():
    F = LocalFrame("F")
    b1 = F.index("b", bound=True)
    b2 = F.index("b", bound=True)
    inner = IndexedSum(b2, F, F.X(b1))
    # b1 is shadowed inside inner because b2 == b1 (name + kind).
    assert not dummy_in(inner, b1)


def test_dummy_in_walks_into_product_and_sum():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    e = Product.make(F.X(a), F.X(b))
    assert dummy_in(e, b)
    s = Sum.make(F.X(a), F.X(b))
    assert dummy_in(s, b)


def test_dummy_in_rejects_non_atom_dummy():
    F = LocalFrame("F")
    body = F.X(F.index("b", bound=True))
    with pytest.raises(TypeError):
        dummy_in(body, Sum.make(Symbol("x"), Symbol("y")))
