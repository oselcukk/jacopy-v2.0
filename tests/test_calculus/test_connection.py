"""Tests for the affine-connection module, Faz 16.A."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.anchor import Anchor, AnchoredVectorField
from jacopy.calculus.connection import (
    AffineConnection,
    ConnectionEvalExpr,
    ConnectionXLinearityDefinition,
    ConnectionXScalarPullDefinition,
    ConnectionYAdditivityDefinition,
    ConnectionYLeibnizDefinition,
    connection,
    koszul_connection,
)
from jacopy.core.expr import Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry


# --------------------------------------------------------------------- #
# AffineConnection atom                                                  #
# --------------------------------------------------------------------- #


def test_affine_connection_carries_name_and_eq():
    a = AffineConnection("∇")
    b = connection("∇")
    c = AffineConnection("∇'")
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a.name == "∇"


def test_affine_connection_rejects_invalid_name():
    with pytest.raises(TypeError):
        AffineConnection(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AffineConnection("")


def test_affine_connection_repr():
    nabla = connection("∇")
    assert nabla._repr_inner() == "∇"


# --------------------------------------------------------------------- #
# ConnectionEvalExpr                                                     #
# --------------------------------------------------------------------- #


def test_connection_eval_children_and_key():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    e = nabla.eval(X, Y)
    assert isinstance(e, ConnectionEvalExpr)
    assert e.children == (X, Y)
    assert e.X is X
    assert e.Y is Y
    assert e.connection == nabla
    assert e._repr_inner() == "∇_X(Y)"


def test_connection_eval_structural_equality():
    nabla = connection()
    other = connection("∇'")
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    a = nabla.eval(X, Y)
    b = nabla.eval(X, Y)
    c = nabla.eval(Y, X)
    d = other.eval(X, Y)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d


def test_connection_eval_rebuild_preserves_connection():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    e = nabla.eval(X, Y)
    rebuilt = e._rebuild((X, Z))
    assert isinstance(rebuilt, ConnectionEvalExpr)
    assert rebuilt.connection == nabla
    assert rebuilt.children == (X, Z)


def test_connection_eval_walks_into_slots():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    e = nabla.eval(X, Y)
    walked = list(e.walk())
    # Walk: self, X, Y
    assert e in walked
    assert X in walked
    assert Y in walked


def test_connection_eval_type_errors():
    nabla = connection()
    X = Derivation("X", 0)
    with pytest.raises(TypeError):
        ConnectionEvalExpr("not-a-connection", X, X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConnectionEvalExpr(nabla, "not-an-expr", X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConnectionEvalExpr(nabla, X, "not-an-expr")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# X-linearity axiom                                                      #
# --------------------------------------------------------------------- #


def test_x_linearity_distributes_sum():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    rule = ConnectionXLinearityDefinition(nabla)
    e = nabla.eval(Sum(X, Y), Z)
    assert rule.matches(e)
    rhs = rule.rewrite(e)
    assert rhs == Sum.make(nabla.eval(X, Z), nabla.eval(Y, Z))


def test_x_linearity_distributes_neg():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = ConnectionXLinearityDefinition(nabla)
    e = nabla.eval(Neg(X), Y)
    assert rule.matches(e)
    rhs = rule.rewrite(e)
    assert rhs == Neg(nabla.eval(X, Y))


def test_x_linearity_does_not_fire_on_atom_x():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = ConnectionXLinearityDefinition(nabla)
    e = nabla.eval(X, Y)
    assert not rule.matches(e)


def test_x_linearity_scoped_to_connection():
    a = connection("∇1")
    b = connection("∇2")
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    rule_a = ConnectionXLinearityDefinition(a)
    e = b.eval(Sum(X, Y), Z)
    assert not rule_a.matches(e)


def test_x_linearity_rejects_non_connection():
    with pytest.raises(TypeError):
        ConnectionXLinearityDefinition("not-a-connection")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Y-additivity axiom                                                     #
# --------------------------------------------------------------------- #


def test_y_additivity_distributes_sum():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    rule = ConnectionYAdditivityDefinition(nabla)
    e = nabla.eval(X, Sum(Y, Z))
    assert rule.matches(e)
    rhs = rule.rewrite(e)
    assert rhs == Sum.make(nabla.eval(X, Y), nabla.eval(X, Z))


def test_y_additivity_distributes_neg():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = ConnectionYAdditivityDefinition(nabla)
    e = nabla.eval(X, Neg(Y))
    assert rule.matches(e)
    rhs = rule.rewrite(e)
    assert rhs == Neg(nabla.eval(X, Y))


def test_y_additivity_does_not_fire_on_atom_y():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = ConnectionYAdditivityDefinition(nabla)
    e = nabla.eval(X, Y)
    assert not rule.matches(e)


# --------------------------------------------------------------------- #
# Y-Leibniz axiom                                                        #
# --------------------------------------------------------------------- #


def test_y_leibniz_distributes_function_factor():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    f = Symbol("f")
    reg = PropertyRegistry()
    reg.declare(f, Graded(degree=0))
    rule = ConnectionYLeibnizDefinition(nabla, registry=reg)
    e = nabla.eval(X, Product(f, Y))
    assert rule.matches(e)
    rhs = rule.rewrite(e)
    expected = Sum.make(
        Product.make(Act(X, f), Y),
        Product.make(f, nabla.eval(X, Y)),
    )
    assert rhs == expected


def test_y_leibniz_does_not_fire_when_first_factor_is_one_form():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    alpha = Symbol("α")
    reg = PropertyRegistry()
    reg.declare(alpha, Graded(degree=1))
    rule = ConnectionYLeibnizDefinition(nabla, registry=reg)
    e = nabla.eval(X, Product(alpha, Y))
    assert not rule.matches(e)


def test_y_leibniz_does_not_fire_on_plain_y():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = ConnectionYLeibnizDefinition(nabla)
    e = nabla.eval(X, Y)
    assert not rule.matches(e)


def test_y_leibniz_scoped_to_connection():
    a = connection("∇1")
    b = connection("∇2")
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    f = Symbol("f")
    reg = PropertyRegistry()
    reg.declare(f, Graded(degree=0))
    rule_a = ConnectionYLeibnizDefinition(a, registry=reg)
    e = b.eval(X, Product(f, Y))
    assert not rule_a.matches(e)


# --------------------------------------------------------------------- #
# Three-axiom integration via engine                                     #
# --------------------------------------------------------------------- #


def test_engine_drives_full_expansion():
    """``∇_{X+Y}(f·Z + W) → all 4 textbook terms``."""
    from jacopy.proof.expansion import ExpansionEngine

    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    W = Derivation("W", 0)
    f = Symbol("f")
    reg = PropertyRegistry()
    reg.declare(f, Graded(degree=0))

    engine = ExpansionEngine(
        [
            ConnectionXLinearityDefinition(nabla),
            ConnectionYAdditivityDefinition(nabla),
            ConnectionYLeibnizDefinition(nabla, registry=reg),
        ]
    )
    e = nabla.eval(Sum(X, Y), Sum(Product(f, Z), W))
    final, steps = engine.expand(e)
    # Should contain four ConnectionEvalExpr's worth of terms after
    # full distribution: ∇_X(f·Z), ∇_X(W), ∇_Y(f·Z), ∇_Y(W), and
    # each f·Z further distributes to X(f)·Z + f·∇_X Z. So six
    # leaves total.
    leaves = list(final.find(lambda n: isinstance(n, ConnectionEvalExpr)))
    assert len(leaves) == 4
    assert len(steps) >= 3


# --------------------------------------------------------------------- #
# ConnectionXScalarPullDefinition, Faz 17.D                             #
# --------------------------------------------------------------------- #


def test_x_scalar_pull_rejects_non_connection():
    with pytest.raises(TypeError):
        ConnectionXScalarPullDefinition("nabla")  # type: ignore[arg-type]


def test_x_scalar_pull_pulls_two_factor_product():
    nabla = connection("∇")
    rule = ConnectionXScalarPullDefinition(nabla)
    f, X, Y = Symbol("f"), Symbol("X"), Symbol("Y")
    e = nabla.eval(Product(f, X), Y)
    assert rule.matches(e)
    out = rule.rewrite(e)
    assert isinstance(out, Product)
    inner = out.children[1]
    assert isinstance(inner, ConnectionEvalExpr)
    assert inner.X == X and inner.Y == Y


def test_x_scalar_pull_pulls_three_factor_product():
    nabla = connection("∇")
    rule = ConnectionXScalarPullDefinition(nabla)
    f, g, X, Y = Symbol("f"), Symbol("g"), Symbol("X"), Symbol("Y")
    e = nabla.eval(Product(f, g, X), Y)
    out = rule.rewrite(e)
    assert isinstance(out, Product)
    # leading factors should fold into a Product(f, g)
    inner = out.children[1]
    assert isinstance(inner, ConnectionEvalExpr)
    assert inner.X == X


def test_x_scalar_pull_does_not_match_singleton_product():
    nabla = connection("∇")
    rule = ConnectionXScalarPullDefinition(nabla)
    X, Y = Symbol("X"), Symbol("Y")
    # Single-factor product, there's no scalar to pull.
    e = nabla.eval(X, Y)
    assert not rule.matches(e)


def test_x_scalar_pull_scoped_to_specific_connection():
    n1 = connection("∇1")
    n2 = connection("∇2")
    rule = ConnectionXScalarPullDefinition(n1)
    f, X, Y = Symbol("f"), Symbol("X"), Symbol("Y")
    # Wrong connection.
    e = n2.eval(Product(f, X), Y)
    assert not rule.matches(e)


class TestAnchoredConnection:
    """The Q9 (Math 595) algebroid generalisation: ``∇̃`` on ``T*M``."""

    def test_default_anchor_is_none(self):
        nabla = connection()
        assert nabla.anchor is None

    def test_carries_anchor(self):
        sharp = Anchor("π^♯")
        nabla = connection("∇̃", anchor=sharp)
        assert nabla.anchor is sharp
        assert nabla.name == "∇̃"

    def test_anchor_participates_in_equality(self):
        a = connection("∇", anchor=Anchor("π^♯"))
        b = connection("∇", anchor=Anchor("π^♯"))
        c = connection("∇")
        d = connection("∇", anchor=Anchor("ρ"))
        assert a == b
        assert a != c
        assert a != d

    def test_function_action_default_is_act(self):
        nabla = connection()
        X = Symbol("X")
        f = Symbol("f")
        assert nabla.function_action(X, f) == Act(X, f)

    def test_function_action_anchored_wraps_through_anchor(self):
        sharp = Anchor("π^♯")
        nabla = connection("∇̃", anchor=sharp)
        omega = Symbol("ω")
        f = Symbol("f")
        out = nabla.function_action(omega, f)
        assert isinstance(out, Act)
        avf = out.op
        assert isinstance(avf, AnchoredVectorField)
        assert avf.anchor is sharp
        assert avf.section is omega
        assert out.arg is f

    def test_y_leibniz_emits_anchored_directional_term(self):
        sharp = Anchor("π^♯")
        nabla = connection("∇̃", anchor=sharp)
        omega = Symbol("ω")
        eta = Symbol("η")
        f = Symbol("f")
        reg = PropertyRegistry()
        reg.declare(f, Graded(degree=0))
        rule = ConnectionYLeibnizDefinition(nabla, registry=reg)
        e = nabla.eval(omega, Product(f, eta))
        assert rule.matches(e)
        rhs = rule.rewrite(e)
        expected = Sum.make(
            Product.make(
                Act(AnchoredVectorField(sharp, omega), f), eta
            ),
            Product.make(f, nabla.eval(omega, eta)),
        )
        assert rhs == expected

    def test_rejects_non_anchor(self):
        with pytest.raises(TypeError):
            AffineConnection("∇", anchor="not-an-anchor")  # type: ignore[arg-type]

    def test_koszul_connection_factory_default(self):
        nabla = koszul_connection()
        assert nabla.name == "∇̃"
        assert isinstance(nabla.anchor, Anchor)
        assert nabla.anchor.name == "π^♯"

    def test_koszul_connection_factory_custom_names(self):
        nabla = koszul_connection("∇'", anchor_name="ρ")
        assert nabla.name == "∇'"
        assert nabla.anchor is not None
        assert nabla.anchor.name == "ρ"

    def test_koszul_connection_y_leibniz_full_engine_pass(self):
        """``∇̃_ω(f·η) → π^♯(ω)(f)·η + f·∇̃_ω η`` via engine."""
        from jacopy.proof.expansion import ExpansionEngine

        nabla = koszul_connection()
        omega = Symbol("ω")
        eta = Symbol("η")
        f = Symbol("f")
        reg = PropertyRegistry()
        reg.declare(f, Graded(degree=0))
        reg.declare(omega, Graded(degree=1))
        reg.declare(eta, Graded(degree=1))

        engine = ExpansionEngine(
            [
                ConnectionYAdditivityDefinition(nabla),
                ConnectionYLeibnizDefinition(nabla, registry=reg),
            ]
        )
        e = nabla.eval(omega, Product(f, eta))
        final, steps = engine.expand(e)
        assert len(steps) >= 1
        # The first term should carry an AnchoredVectorField wrapping ω.
        avf_nodes = list(
            final.find(lambda n: isinstance(n, AnchoredVectorField))
        )
        assert len(avf_nodes) == 1
        assert avf_nodes[0].section == omega


def test_x_scalar_pull_in_engine_carries_omega_proof_step():
    from jacopy.proof.expansion import ExpansionEngine

    nabla = connection("∇")
    engine = ExpansionEngine(
        [ConnectionXScalarPullDefinition(nabla)]
    )
    f, V, Xb = Symbol("f"), Symbol("V"), Symbol("X_b")
    e = nabla.eval(Product(f, V), Xb)
    final, steps = engine.expand(e)
    assert isinstance(final, Product)
    # one step should have fired
    assert len(steps) >= 1
