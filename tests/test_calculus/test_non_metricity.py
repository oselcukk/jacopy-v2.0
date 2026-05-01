"""Tests for MetricTensor + NonMetricityEvalExpr, Faz 17.B."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.connection import AffineConnection, connection
from jacopy.algebra.derivation import Act
from jacopy.calculus.connection import ConnectionEvalExpr
from jacopy.calculus.metric import (
    MetricEvalExpr,
    MetricEvalLinearityDefinition,
    MetricEvalScalarPullDefinition,
    MetricEvalSymmetryDefinition,
    MetricTensor,
    metric,
)
from jacopy.calculus.non_metricity import (
    NonMetricityCompatibilityDefinition,
    NonMetricityEvalExpr,
    NonMetricityVLinearityDefinition,
    NonMetricityVScalarPullDefinition,
    NonMetricityXYSymmetryDefinition,
)
from jacopy.core.expr import Neg, Product, Sum, Symbol
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# MetricTensor                                                          #
# --------------------------------------------------------------------- #


def test_metric_tensor_carries_name():
    g = MetricTensor("g")
    assert g.name == "g"
    assert g._repr_inner() == "g"


def test_metric_factory_default_name():
    g = metric()
    assert g.name == "g"


def test_metric_tensor_equality_and_hash():
    g1 = MetricTensor("g")
    g2 = metric("g")
    h = MetricTensor("h")
    assert g1 == g2
    assert hash(g1) == hash(g2)
    assert g1 != h


def test_metric_tensor_rejects_invalid_name():
    with pytest.raises(TypeError):
        MetricTensor(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MetricTensor("")


def test_metric_tensor_is_atom():
    g = metric()
    assert g.is_atom
    assert g.children == ()


# --------------------------------------------------------------------- #
# NonMetricityEvalExpr, construction                                   #
# --------------------------------------------------------------------- #


def test_non_metricity_eval_children_and_key():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Q = NonMetricityEvalExpr(nabla, g, V, X, Y)
    assert Q.children == (V, X, Y)
    assert Q.V is V
    assert Q.X is X
    assert Q.Y is Y
    assert Q.connection == nabla
    assert Q.metric == g


def test_non_metricity_eval_equality():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    a = NonMetricityEvalExpr(nabla, g, V, X, Y)
    b = NonMetricityEvalExpr(nabla, g, V, X, Y)
    assert a == b
    assert hash(a) == hash(b)


def test_non_metricity_eval_distinguishes_metrics():
    nabla = connection()
    g1 = metric("g")
    g2 = metric("g'")
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    a = NonMetricityEvalExpr(nabla, g1, V, X, Y)
    b = NonMetricityEvalExpr(nabla, g2, V, X, Y)
    assert a != b


def test_non_metricity_eval_distinguishes_connections():
    nabla1 = connection("∇")
    nabla2 = connection("∇'")
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    a = NonMetricityEvalExpr(nabla1, g, V, X, Y)
    b = NonMetricityEvalExpr(nabla2, g, V, X, Y)
    assert a != b


def test_non_metricity_eval_rebuild_preserves_params():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Q = NonMetricityEvalExpr(nabla, g, V, X, Y)
    V2 = Derivation("V'", 0)
    rebuilt = Q._rebuild((V2, X, Y))
    assert isinstance(rebuilt, NonMetricityEvalExpr)
    assert rebuilt.connection == nabla
    assert rebuilt.metric == g
    assert rebuilt.V == V2


def test_non_metricity_eval_repr_includes_all_args():
    nabla = connection("∇")
    g = metric("g")
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    text = NonMetricityEvalExpr(nabla, g, V, X, Y)._repr_inner()
    assert "Q" in text
    assert "∇" in text
    assert "g" in text
    assert "V" in text and "X" in text and "Y" in text


def test_non_metricity_eval_rejects_invalid_args():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    with pytest.raises(TypeError):
        NonMetricityEvalExpr("not-conn", g, V, X, Y)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityEvalExpr(nabla, "not-metric", V, X, Y)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityEvalExpr(nabla, g, "V", X, Y)  # type: ignore[arg-type]


def test_non_metricity_eval_rebuild_arity_check():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Q = NonMetricityEvalExpr(nabla, g, V, X, Y)
    with pytest.raises(ValueError):
        Q._rebuild((V, X))


# --------------------------------------------------------------------- #
# V-linearity (Sum / Neg distribution)                                  #
# --------------------------------------------------------------------- #


def test_v_linearity_distributes_sum_in_v_slot():
    nabla = connection()
    g = metric()
    A = Derivation("A", 0)
    B = Derivation("B", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVLinearityDefinition(nabla, g)
    expr = NonMetricityEvalExpr(nabla, g, Sum.make(A, B), X, Y)
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Sum)
    assert res.children == (
        NonMetricityEvalExpr(nabla, g, A, X, Y),
        NonMetricityEvalExpr(nabla, g, B, X, Y),
    )


def test_v_linearity_distributes_neg_in_v_slot():
    nabla = connection()
    g = metric()
    A = Derivation("A", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVLinearityDefinition(nabla, g)
    expr = NonMetricityEvalExpr(nabla, g, Neg(A), X, Y)
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Neg)
    assert res.arg == NonMetricityEvalExpr(nabla, g, A, X, Y)


def test_v_linearity_distributes_sum_with_neg_terms():
    nabla = connection()
    g = metric()
    A = Derivation("A", 0)
    B = Derivation("B", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVLinearityDefinition(nabla, g)
    expr = NonMetricityEvalExpr(nabla, g, Sum.make(A, Neg(B)), X, Y)
    res = rule.rewrite(expr)
    assert isinstance(res, Sum)
    assert NonMetricityEvalExpr(nabla, g, A, X, Y) in res.children
    assert any(
        isinstance(c, Neg)
        and c.arg == NonMetricityEvalExpr(nabla, g, B, X, Y)
        for c in res.children
    )


def test_v_linearity_does_not_match_when_v_is_atomic():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVLinearityDefinition(nabla, g)
    assert not rule.matches(NonMetricityEvalExpr(nabla, g, V, X, Y))


def test_v_linearity_does_not_match_other_metric():
    nabla = connection()
    g1 = metric("g")
    g2 = metric("g'")
    A = Derivation("A", 0)
    B = Derivation("B", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVLinearityDefinition(nabla, g1)
    expr = NonMetricityEvalExpr(nabla, g2, Sum.make(A, B), X, Y)
    assert not rule.matches(expr)


def test_v_linearity_does_not_match_other_connection():
    nabla1 = connection("∇")
    nabla2 = connection("∇'")
    g = metric()
    A = Derivation("A", 0)
    B = Derivation("B", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVLinearityDefinition(nabla1, g)
    expr = NonMetricityEvalExpr(nabla2, g, Sum.make(A, B), X, Y)
    assert not rule.matches(expr)


def test_v_linearity_constructor_rejects_bad_args():
    g = metric()
    nabla = connection()
    with pytest.raises(TypeError):
        NonMetricityVLinearityDefinition("not-conn", g)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityVLinearityDefinition(nabla, "not-metric")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# V-scalar-pull                                                         #
# --------------------------------------------------------------------- #


def test_v_scalar_pull_extracts_leading_scalar():
    nabla = connection()
    g = metric()
    f = Symbol("f")
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVScalarPullDefinition(nabla, g)
    expr = NonMetricityEvalExpr(nabla, g, Product(f, V), X, Y)
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Product)
    assert res.children[0] == f
    inner = res.children[1]
    assert isinstance(inner, NonMetricityEvalExpr)
    assert inner.V == V


def test_v_scalar_pull_handles_multiple_leading_scalars():
    nabla = connection()
    g = metric()
    f = Symbol("f")
    h = Symbol("h")
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVScalarPullDefinition(nabla, g)
    expr = NonMetricityEvalExpr(nabla, g, Product.make(f, h, V), X, Y)
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Product)
    inner = res.children[1]
    assert isinstance(inner, NonMetricityEvalExpr)
    assert inner.V == V


def test_v_scalar_pull_does_not_match_non_product():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVScalarPullDefinition(nabla, g)
    assert not rule.matches(NonMetricityEvalExpr(nabla, g, V, X, Y))


def test_v_scalar_pull_does_not_match_singleton_product():
    nabla = connection()
    g = metric()
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVScalarPullDefinition(nabla, g)
    # A singleton Product would have collapsed to its sole child by
    # Product.make; constructing one directly is a degenerate case
    # the rule deliberately ignores.
    expr = NonMetricityEvalExpr(nabla, g, Product(V), X, Y)
    assert not rule.matches(expr)


def test_v_scalar_pull_other_metric_no_match():
    nabla = connection()
    g1 = metric("g")
    g2 = metric("g'")
    f = Symbol("f")
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = NonMetricityVScalarPullDefinition(nabla, g1)
    expr = NonMetricityEvalExpr(nabla, g2, Product(f, V), X, Y)
    assert not rule.matches(expr)


# --------------------------------------------------------------------- #
# Engine integration                                                    #
# --------------------------------------------------------------------- #


def test_engine_v_linearity_distribute_then_collapse():
    nabla = connection()
    g = metric()
    A = Derivation("A", 0)
    B = Derivation("B", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    engine = ExpansionEngine(
        [NonMetricityVLinearityDefinition(nabla, g)]
    )
    expr = NonMetricityEvalExpr(nabla, g, Sum.make(A, B), X, Y)
    final, steps = engine.expand(expr)
    assert isinstance(final, Sum)
    assert len(final.children) == 2
    assert len(steps) == 1


def test_engine_v_scalar_pull_extracts_factor():
    nabla = connection()
    g = metric()
    f = Symbol("f")
    V = Derivation("V", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    engine = ExpansionEngine(
        [NonMetricityVScalarPullDefinition(nabla, g)]
    )
    expr = NonMetricityEvalExpr(nabla, g, Product(f, V), X, Y)
    final, steps = engine.expand(expr)
    assert isinstance(final, Product)
    assert final.children[0] == f
    inner = final.children[1]
    assert isinstance(inner, NonMetricityEvalExpr)
    assert inner.V == V
    assert len(steps) == 1


def test_engine_two_metrics_no_cross_fire():
    nabla = connection()
    g1 = metric("g")
    g2 = metric("g'")
    A = Derivation("A", 0)
    B = Derivation("B", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    engine = ExpansionEngine(
        [
            NonMetricityVLinearityDefinition(nabla, g1),
            NonMetricityVScalarPullDefinition(nabla, g1),
        ]
    )
    expr = NonMetricityEvalExpr(nabla, g2, Sum.make(A, B), X, Y)
    final, steps = engine.expand(expr)
    assert final == expr
    assert len(steps) == 0


def test_engine_combined_sum_then_scalar_pull_within_terms():
    """``Q(fA + B, X, Y) → Q(fA,X,Y) + Q(B,X,Y) → f·Q(A,X,Y) + Q(B,X,Y)``."""
    nabla = connection()
    g = metric()
    f = Symbol("f")
    A = Derivation("A", 0)
    B = Derivation("B", 0)
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    engine = ExpansionEngine(
        [
            NonMetricityVLinearityDefinition(nabla, g),
            NonMetricityVScalarPullDefinition(nabla, g),
        ]
    )
    expr = NonMetricityEvalExpr(
        nabla, g, Sum.make(Product(f, A), B), X, Y
    )
    final, steps = engine.expand(expr)
    assert isinstance(final, Sum)
    # The scalar f should have been pulled out of the first term.
    assert any(
        isinstance(c, Product)
        and any(
            isinstance(g_c, NonMetricityEvalExpr) and g_c.V == A
            for g_c in c.children
        )
        for c in final.children
    )
    # The B term stays bare (no Product around it).
    assert any(
        isinstance(c, NonMetricityEvalExpr) and c.V == B
        for c in final.children
    )
    assert len(steps) >= 2


# --------------------------------------------------------------------- #
# MetricEvalExpr                                                        #
# --------------------------------------------------------------------- #


def test_metric_eval_expr_carries_metric_and_args():
    g = metric()
    X = Derivation("X")
    Y = Derivation("Y")
    e = MetricEvalExpr(g, X, Y)
    assert e.metric == g
    assert e.X == X
    assert e.Y == Y
    assert e.children == (X, Y)


def test_metric_eval_expr_equality():
    g = metric()
    X = Derivation("X")
    Y = Derivation("Y")
    a = MetricEvalExpr(g, X, Y)
    b = MetricEvalExpr(metric("g"), X, Y)
    assert a == b
    assert hash(a) == hash(b)


def test_metric_eval_expr_distinguishes_metrics():
    g1 = metric("g1")
    g2 = metric("g2")
    X = Derivation("X")
    Y = Derivation("Y")
    assert MetricEvalExpr(g1, X, Y) != MetricEvalExpr(g2, X, Y)


def test_metric_eval_expr_rebuild_preserves_metric():
    g = metric()
    X = Derivation("X")
    Y = Derivation("Y")
    Z = Derivation("Z")
    e = MetricEvalExpr(g, X, Y)
    e2 = e._rebuild((X, Z))
    assert isinstance(e2, MetricEvalExpr)
    assert e2.metric == g
    assert e2.children == (X, Z)


def test_metric_eval_expr_repr():
    g = metric()
    X = Derivation("X")
    Y = Derivation("Y")
    s = repr(MetricEvalExpr(g, X, Y))
    assert "g(X,Y)" in s


def test_metric_eval_expr_validation():
    g = metric()
    X = Derivation("X")
    with pytest.raises(TypeError):
        MetricEvalExpr("not a metric", X, X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MetricEvalExpr(g, "X", X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MetricEvalExpr(g, X, "Y")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# MetricEvalSymmetryDefinition                                          #
# --------------------------------------------------------------------- #


def _ordered_pair():
    """Return (lo, hi) Derivations with repr(lo) < repr(hi)."""
    a = Derivation("A")
    b = Derivation("B")
    if repr(a) < repr(b):
        return a, b
    return b, a


def test_metric_symmetry_canonicalises_out_of_order():
    g = metric()
    rule = MetricEvalSymmetryDefinition(g)
    lo, hi = _ordered_pair()
    expr = MetricEvalExpr(g, hi, lo)
    assert rule.matches(expr)
    out = rule.rewrite(expr)
    assert isinstance(out, MetricEvalExpr)
    assert out.X == lo and out.Y == hi


def test_metric_symmetry_no_match_when_already_canonical():
    g = metric()
    rule = MetricEvalSymmetryDefinition(g)
    lo, hi = _ordered_pair()
    assert not rule.matches(MetricEvalExpr(g, lo, hi))


def test_metric_symmetry_no_match_on_equal_args():
    g = metric()
    rule = MetricEvalSymmetryDefinition(g)
    X = Derivation("X")
    assert not rule.matches(MetricEvalExpr(g, X, X))


def test_metric_symmetry_scoped_to_metric():
    g1 = metric("g1")
    g2 = metric("g2")
    rule = MetricEvalSymmetryDefinition(g1)
    lo, hi = _ordered_pair()
    assert not rule.matches(MetricEvalExpr(g2, hi, lo))


def test_metric_symmetry_rejects_non_metric():
    with pytest.raises(TypeError):
        MetricEvalSymmetryDefinition("not a metric")  # type: ignore[arg-type]


def test_metric_symmetry_engine_canonicalises():
    g = metric()
    engine = ExpansionEngine([MetricEvalSymmetryDefinition(g)])
    lo, hi = _ordered_pair()
    expr = MetricEvalExpr(g, hi, lo)
    final, steps = engine.expand(expr)
    assert isinstance(final, MetricEvalExpr)
    assert final.X == lo and final.Y == hi
    # idempotent: a second pass does nothing
    final2, steps2 = engine.expand(final)
    assert final2 == final
    assert len(steps2) == 0


# --------------------------------------------------------------------- #
# NonMetricityXYSymmetryDefinition                                      #
# --------------------------------------------------------------------- #


def test_non_metricity_xy_symmetry_canonicalises():
    nabla = connection()
    g = metric()
    rule = NonMetricityXYSymmetryDefinition(nabla, g)
    V = Derivation("V")
    lo, hi = _ordered_pair()
    expr = NonMetricityEvalExpr(nabla, g, V, hi, lo)
    assert rule.matches(expr)
    out = rule.rewrite(expr)
    assert isinstance(out, NonMetricityEvalExpr)
    assert out.V == V
    assert out.X == lo and out.Y == hi


def test_non_metricity_xy_symmetry_no_match_when_canonical():
    nabla = connection()
    g = metric()
    rule = NonMetricityXYSymmetryDefinition(nabla, g)
    V = Derivation("V")
    lo, hi = _ordered_pair()
    assert not rule.matches(NonMetricityEvalExpr(nabla, g, V, lo, hi))


def test_non_metricity_xy_symmetry_no_match_on_equal_xy():
    nabla = connection()
    g = metric()
    rule = NonMetricityXYSymmetryDefinition(nabla, g)
    V = Derivation("V")
    X = Derivation("X")
    assert not rule.matches(NonMetricityEvalExpr(nabla, g, V, X, X))


def test_non_metricity_xy_symmetry_scoped_to_pair():
    nabla1 = connection("nabla1")
    nabla2 = connection("nabla2")
    g1 = metric("g1")
    g2 = metric("g2")
    rule = NonMetricityXYSymmetryDefinition(nabla1, g1)
    V = Derivation("V")
    lo, hi = _ordered_pair()
    assert not rule.matches(
        NonMetricityEvalExpr(nabla2, g1, V, hi, lo)
    )
    assert not rule.matches(
        NonMetricityEvalExpr(nabla1, g2, V, hi, lo)
    )


def test_non_metricity_xy_symmetry_validation():
    nabla = connection()
    g = metric()
    with pytest.raises(TypeError):
        NonMetricityXYSymmetryDefinition("not conn", g)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityXYSymmetryDefinition(nabla, "not metric")  # type: ignore[arg-type]


def test_non_metricity_xy_symmetry_engine_canonicalises():
    nabla = connection()
    g = metric()
    engine = ExpansionEngine(
        [NonMetricityXYSymmetryDefinition(nabla, g)]
    )
    V = Derivation("V")
    lo, hi = _ordered_pair()
    expr = NonMetricityEvalExpr(nabla, g, V, hi, lo)
    final, _ = engine.expand(expr)
    assert isinstance(final, NonMetricityEvalExpr)
    assert final.X == lo and final.Y == hi
    final2, steps2 = engine.expand(final)
    assert final2 == final
    assert len(steps2) == 0


def test_non_metricity_xy_symmetry_combines_with_v_linearity():
    nabla = connection()
    g = metric()
    engine = ExpansionEngine(
        [
            NonMetricityVLinearityDefinition(nabla, g),
            NonMetricityXYSymmetryDefinition(nabla, g),
        ]
    )
    A = Derivation("A")
    B = Derivation("B")
    lo, hi = _ordered_pair()
    expr = NonMetricityEvalExpr(nabla, g, Sum.make(A, B), hi, lo)
    final, _ = engine.expand(expr)
    assert isinstance(final, Sum)
    for term in final.children:
        assert isinstance(term, NonMetricityEvalExpr)
        assert term.X == lo and term.Y == hi


# --------------------------------------------------------------------- #
# MetricEvalLinearityDefinition                                         #
# --------------------------------------------------------------------- #


def test_metric_linearity_distributes_sum_left():
    g = metric()
    A = Derivation("A")
    B = Derivation("B")
    Y = Derivation("Y")
    rule = MetricEvalLinearityDefinition(g)
    expr = MetricEvalExpr(g, Sum.make(A, B), Y)
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Sum)
    assert MetricEvalExpr(g, A, Y) in res.children
    assert MetricEvalExpr(g, B, Y) in res.children


def test_metric_linearity_distributes_sum_right():
    g = metric()
    X = Derivation("X")
    A = Derivation("A")
    B = Derivation("B")
    rule = MetricEvalLinearityDefinition(g)
    expr = MetricEvalExpr(g, X, Sum.make(A, B))
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Sum)
    assert MetricEvalExpr(g, X, A) in res.children
    assert MetricEvalExpr(g, X, B) in res.children


def test_metric_linearity_distributes_neg_left():
    g = metric()
    A = Derivation("A")
    Y = Derivation("Y")
    rule = MetricEvalLinearityDefinition(g)
    expr = MetricEvalExpr(g, Neg(A), Y)
    res = rule.rewrite(expr)
    assert isinstance(res, Neg)
    assert res.arg == MetricEvalExpr(g, A, Y)


def test_metric_linearity_distributes_neg_right():
    g = metric()
    X = Derivation("X")
    A = Derivation("A")
    rule = MetricEvalLinearityDefinition(g)
    expr = MetricEvalExpr(g, X, Neg(A))
    res = rule.rewrite(expr)
    assert isinstance(res, Neg)
    assert res.arg == MetricEvalExpr(g, X, A)


def test_metric_linearity_no_match_when_atomic():
    g = metric()
    X = Derivation("X")
    Y = Derivation("Y")
    rule = MetricEvalLinearityDefinition(g)
    assert not rule.matches(MetricEvalExpr(g, X, Y))


def test_metric_linearity_scoped_to_metric():
    g1 = metric("g1")
    g2 = metric("g2")
    A = Derivation("A")
    B = Derivation("B")
    Y = Derivation("Y")
    rule = MetricEvalLinearityDefinition(g1)
    assert not rule.matches(MetricEvalExpr(g2, Sum.make(A, B), Y))


def test_metric_linearity_validation():
    with pytest.raises(TypeError):
        MetricEvalLinearityDefinition("not metric")  # type: ignore[arg-type]


def test_metric_linearity_engine_distributes_both_slots():
    g = metric()
    A = Derivation("A")
    B = Derivation("B")
    C = Derivation("C")
    D = Derivation("D")
    engine = ExpansionEngine([MetricEvalLinearityDefinition(g)])
    expr = MetricEvalExpr(g, Sum.make(A, B), Sum.make(C, D))
    final, _ = engine.expand(expr)
    assert isinstance(final, Sum)
    # 4 cross terms expected (nested Sum-of-Sum is fine; walk recursively)
    def collect_metrics(e):
        out = []
        if isinstance(e, MetricEvalExpr):
            out.append(e)
        for c in e.children:
            out.extend(collect_metrics(c))
        return out

    metric_terms = collect_metrics(final)
    assert len(metric_terms) == 4
    pairs = {(m.X, m.Y) for m in metric_terms}
    assert pairs == {(A, C), (A, D), (B, C), (B, D)}


# --------------------------------------------------------------------- #
# MetricEvalScalarPullDefinition                                        #
# --------------------------------------------------------------------- #


def test_metric_scalar_pull_left():
    g = metric()
    f = Symbol("f")
    X = Derivation("X")
    Y = Derivation("Y")
    rule = MetricEvalScalarPullDefinition(g)
    expr = MetricEvalExpr(g, Product(f, X), Y)
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Product)
    assert res.children[0] == f
    assert isinstance(res.children[1], MetricEvalExpr)
    assert res.children[1].X == X
    assert res.children[1].Y == Y


def test_metric_scalar_pull_right():
    g = metric()
    f = Symbol("f")
    X = Derivation("X")
    Y = Derivation("Y")
    rule = MetricEvalScalarPullDefinition(g)
    expr = MetricEvalExpr(g, X, Product(f, Y))
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Product)
    assert res.children[0] == f
    assert isinstance(res.children[1], MetricEvalExpr)
    assert res.children[1].X == X
    assert res.children[1].Y == Y


def test_metric_scalar_pull_no_match_when_no_product():
    g = metric()
    X = Derivation("X")
    Y = Derivation("Y")
    rule = MetricEvalScalarPullDefinition(g)
    assert not rule.matches(MetricEvalExpr(g, X, Y))


def test_metric_scalar_pull_scoped_to_metric():
    g1 = metric("g1")
    g2 = metric("g2")
    f = Symbol("f")
    X = Derivation("X")
    Y = Derivation("Y")
    rule = MetricEvalScalarPullDefinition(g1)
    assert not rule.matches(MetricEvalExpr(g2, Product(f, X), Y))


def test_metric_scalar_pull_validation():
    with pytest.raises(TypeError):
        MetricEvalScalarPullDefinition("not metric")  # type: ignore[arg-type]


def test_metric_scalar_pull_engine_pulls_both_slots():
    g = metric()
    f = Symbol("f")
    h = Symbol("h")
    X = Derivation("X")
    Y = Derivation("Y")
    engine = ExpansionEngine([MetricEvalScalarPullDefinition(g)])
    expr = MetricEvalExpr(g, Product(f, X), Product(h, Y))
    final, steps = engine.expand(expr)
    # both factors pulled; final should be f·h·g(X, Y) (some Product nesting)
    assert len(steps) >= 2
    # contains an inner MetricEvalExpr with bare X, Y
    def contains_bare(e):
        if isinstance(e, MetricEvalExpr):
            return e.X == X and e.Y == Y
        if hasattr(e, "children"):
            return any(contains_bare(c) for c in e.children)
        return False

    assert contains_bare(final)


# --------------------------------------------------------------------- #
# NonMetricityCompatibilityDefinition                                   #
# --------------------------------------------------------------------- #


def test_compatibility_opens_to_three_terms():
    nabla = connection()
    g = metric()
    V = Derivation("V")
    X = Derivation("X")
    Y = Derivation("Y")
    rule = NonMetricityCompatibilityDefinition(nabla, g)
    expr = NonMetricityEvalExpr(nabla, g, V, X, Y)
    assert rule.matches(expr)
    res = rule.rewrite(expr)
    assert isinstance(res, Sum)
    # Three terms: V(g(X,Y)), -g(∇_V X, Y), -g(X, ∇_V Y).
    children = res.children
    assert len(children) == 3
    # First: Act(V, g(X,Y))
    assert any(
        isinstance(c, Act)
        and c.op == V
        and isinstance(c.arg, MetricEvalExpr)
        and c.arg.X == X
        and c.arg.Y == Y
        for c in children
    )
    # Second/third: Neg of MetricEvalExpr containing ConnectionEvalExpr.
    neg_metric_terms = [
        c
        for c in children
        if isinstance(c, Neg) and isinstance(c.arg, MetricEvalExpr)
    ]
    assert len(neg_metric_terms) == 2
    # One has ∇_V X in left slot, the other ∇_V Y in right slot.
    left_form = next(
        (
            t
            for t in neg_metric_terms
            if isinstance(t.arg.X, ConnectionEvalExpr)
        ),
        None,
    )
    right_form = next(
        (
            t
            for t in neg_metric_terms
            if isinstance(t.arg.Y, ConnectionEvalExpr)
        ),
        None,
    )
    assert left_form is not None and right_form is not None
    # ConnectionEvalExpr(∇, V, X) stores V in .X and X in .Y (∇_X Y convention).
    assert left_form.arg.X.X == V and left_form.arg.X.Y == X
    assert left_form.arg.Y == Y
    assert right_form.arg.X == X
    assert right_form.arg.Y.X == V and right_form.arg.Y.Y == Y


def test_compatibility_no_match_when_v_not_derivation():
    nabla = connection()
    g = metric()
    A = Derivation("A")
    B = Derivation("B")
    X = Derivation("X")
    Y = Derivation("Y")
    rule = NonMetricityCompatibilityDefinition(nabla, g)
    # Sum-V, not a Derivation, must defer to V-linearity first.
    assert not rule.matches(
        NonMetricityEvalExpr(nabla, g, Sum.make(A, B), X, Y)
    )


def test_compatibility_scoped_to_pair():
    nabla1 = connection("nabla1")
    nabla2 = connection("nabla2")
    g1 = metric("g1")
    g2 = metric("g2")
    V = Derivation("V")
    X = Derivation("X")
    Y = Derivation("Y")
    rule = NonMetricityCompatibilityDefinition(nabla1, g1)
    assert not rule.matches(NonMetricityEvalExpr(nabla2, g1, V, X, Y))
    assert not rule.matches(NonMetricityEvalExpr(nabla1, g2, V, X, Y))


def test_compatibility_validation():
    nabla = connection()
    g = metric()
    with pytest.raises(TypeError):
        NonMetricityCompatibilityDefinition("not conn", g)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityCompatibilityDefinition(nabla, "not metric")  # type: ignore[arg-type]


def test_compatibility_engine_chains_with_v_linearity():
    nabla = connection()
    g = metric()
    A = Derivation("A")
    B = Derivation("B")
    X = Derivation("X")
    Y = Derivation("Y")
    engine = ExpansionEngine(
        [
            NonMetricityVLinearityDefinition(nabla, g),
            NonMetricityCompatibilityDefinition(nabla, g),
        ]
    )
    expr = NonMetricityEvalExpr(nabla, g, Sum.make(A, B), X, Y)
    final, _ = engine.expand(expr)
    # No NonMetricityEvalExpr should remain, V-linearity splits then
    # compatibility opens both.
    def has_nme(e):
        if isinstance(e, NonMetricityEvalExpr):
            return True
        return any(has_nme(c) for c in e.children)

    assert not has_nme(final)


def test_compatibility_does_not_loop_with_closure_axioms():
    """Closure + opener + symmetry all fire; result still terminates and has no Q."""
    nabla = connection()
    g = metric()
    V = Derivation("V")
    A = Derivation("A")
    B = Derivation("B")
    engine = ExpansionEngine(
        [
            NonMetricityVLinearityDefinition(nabla, g),
            NonMetricityXYSymmetryDefinition(nabla, g),
            NonMetricityCompatibilityDefinition(nabla, g),
        ]
    )
    # Pick an out-of-order pair so symmetry fires too.
    if repr(A) < repr(B):
        lo, hi = A, B
    else:
        lo, hi = B, A
    expr = NonMetricityEvalExpr(nabla, g, V, hi, lo)
    final, _ = engine.expand(expr)

    def has_nme(e):
        if isinstance(e, NonMetricityEvalExpr):
            return True
        return any(has_nme(c) for c in e.children)

    assert not has_nme(final)
