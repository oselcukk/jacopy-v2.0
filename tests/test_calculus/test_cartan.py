"""Tests for CartanCalculus — relation access, verification, modes."""

import pytest

from gradalg.algebra.derivation import Derivation
from gradalg.brackets.lie import LieBracket
from gradalg.calculus.cartan import MODES, RELATIONS, CartanCalculus
from gradalg.calculus.exterior_algebra import ExteriorAlgebra
from gradalg.calculus.exterior_d import d
from gradalg.calculus.interior import interior
from gradalg.calculus.lie_derivative import lie_derivative
from gradalg.calculus.operator_equation import OperatorEquation
from gradalg.core.expr import Integer, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.proof import ProofChain, ProofFailure


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def calc():
    return CartanCalculus(
        d=d,
        lie_derivative=lie_derivative,
        interior=interior,
        vector_bracket=LieBracket(),
    )


@pytest.fixture
def context():
    reg = PropertyRegistry()
    f = Symbol("f")
    reg.declare(f, Graded(degree=0))
    algebra = ExteriorAlgebra((f,))
    return reg, algebra, f


@pytest.fixture
def XY():
    X = Derivation("X", degree=0)
    Y = Derivation("Y", degree=0)
    return X, Y


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_defaults_instantiate(self, calc):
        assert calc.d is d

    def test_rejects_non_derivation_d(self):
        with pytest.raises(TypeError, match="Derivation"):
            CartanCalculus(
                d="d",  # type: ignore[arg-type]
                lie_derivative=lie_derivative,
                interior=interior,
                vector_bracket=LieBracket(),
            )

    def test_rejects_non_callable_factory(self):
        with pytest.raises(TypeError, match="callable"):
            CartanCalculus(
                d=d,
                lie_derivative="not callable",  # type: ignore[arg-type]
                interior=interior,
                vector_bracket=LieBracket(),
            )

    def test_rejects_non_bracket(self):
        with pytest.raises(TypeError, match="GradedBracket"):
            CartanCalculus(
                d=d,
                lie_derivative=lie_derivative,
                interior=interior,
                vector_bracket="not a bracket",  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------- #
# relation()                                                             #
# --------------------------------------------------------------------- #


class TestRelationBuilder:
    def test_all_relations_known(self):
        assert set(RELATIONS) == {
            "d_squared_zero",
            "cartan_magic",
            "d_lie",
            "lie_lie",
            "lie_iota",
        }

    def test_d_squared_zero_needs_no_fields(self, calc, context):
        _, algebra, _ = context
        eq = calc.relation("d_squared_zero", algebra=algebra)
        assert isinstance(eq, OperatorEquation)
        assert eq.rhs == Integer(0)

    def test_cartan_magic_requires_X(self, calc):
        with pytest.raises(ValueError, match="cartan_magic requires X"):
            calc.relation("cartan_magic")

    def test_lie_lie_requires_Y(self, calc, XY):
        X, _ = XY
        with pytest.raises(ValueError, match="lie_lie requires X and Y"):
            calc.relation("lie_lie", X=X)

    def test_lie_iota_builds_operator_equation(self, calc, XY, context):
        X, Y = XY
        _, algebra, _ = context
        eq = calc.relation("lie_iota", X=X, Y=Y, algebra=algebra)
        assert isinstance(eq, OperatorEquation)

    def test_unknown_relation_raises(self, calc):
        with pytest.raises(ValueError, match="Unknown Cartan relation"):
            calc.relation("not_a_real_relation")


# --------------------------------------------------------------------- #
# verify() — the one reliably-closing relation on a function algebra     #
# --------------------------------------------------------------------- #


class TestVerifyCartanMagic:
    def test_cartan_magic_closes_on_function_algebra(
        self, calc, context, XY
    ):
        """[d, ι_X] = L_X holds by definition when L_X uses the cartan form."""
        reg, algebra, _ = context
        X, _ = XY
        chain = calc.verify(
            "cartan_magic",
            algebra=algebra,
            X=X,
            registry=reg,
        )
        assert isinstance(chain, ProofChain)

    def test_foundational_mode_uses_unroll_wrapper(self, calc, context, XY):
        reg, algebra, _ = context
        X, _ = XY
        chain = calc.verify(
            "cartan_magic",
            algebra=algebra,
            X=X,
            registry=reg,
            mode="foundational",
        )
        assert isinstance(chain, ProofChain)

    def test_rejects_unknown_mode(self, calc, context, XY):
        reg, algebra, _ = context
        X, _ = XY
        with pytest.raises(ValueError, match="mode must be one of"):
            calc.verify(
                "cartan_magic",
                algebra=algebra,
                X=X,
                registry=reg,
                mode="lightning",
            )

    def test_modes_tuple_is_canonical(self):
        assert MODES == ("efficient", "foundational")


# --------------------------------------------------------------------- #
# verify() — the other four relations on a function algebra              #
# --------------------------------------------------------------------- #


class TestVerifyOtherRelations:
    """Regression tests for d_squared_zero, d_lie, lie_lie, lie_iota.

    Closing these required five engine/strategy fixes (Derivation-degree
    fallback in sort_product, zero-polymorphic degree in
    AgreementOnGenerators, Neg/Zero handling in product_rule, looped
    expand+product-rule fix-point in ExpandAndSimplify, and iota on
    Derivation-combinations). Guarding all four here so regressions
    surface immediately.
    """

    def test_d_squared_zero_closes_on_function_algebra(
        self, calc, context
    ):
        reg, algebra, _ = context
        chain = calc.verify(
            "d_squared_zero", algebra=algebra, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_d_lie_closes_on_function_algebra(self, calc, context, XY):
        reg, algebra, _ = context
        X, _ = XY
        chain = calc.verify(
            "d_lie", algebra=algebra, X=X, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_lie_lie_closes_on_function_algebra(self, calc, context, XY):
        reg, algebra, _ = context
        X, Y = XY
        chain = calc.verify(
            "lie_lie", algebra=algebra, X=X, Y=Y, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_lie_iota_closes_on_function_algebra(self, calc, context, XY):
        reg, algebra, _ = context
        X, Y = XY
        chain = calc.verify(
            "lie_iota", algebra=algebra, X=X, Y=Y, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_verify_all_closes_every_relation(self, calc, context, XY):
        reg, algebra, _ = context
        X, Y = XY
        results = calc.verify_all(
            algebra=algebra, X=X, Y=Y, registry=reg
        )
        assert set(results) == set(RELATIONS)
        for chain in results.values():
            assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# Identity                                                               #
# --------------------------------------------------------------------- #


class TestIdentity:
    def test_equal_calculi_compare_equal(self):
        lb = LieBracket()
        c1 = CartanCalculus(d, lie_derivative, interior, lb)
        c2 = CartanCalculus(d, lie_derivative, interior, lb)
        assert c1 == c2

    def test_different_brackets_compare_unequal(self):
        c1 = CartanCalculus(d, lie_derivative, interior, LieBracket("[·,·]"))
        c2 = CartanCalculus(
            d, lie_derivative, interior, LieBracket("[·,·]_2")
        )
        assert c1 != c2

    def test_repr_mentions_bracket(self, calc):
        assert "[·,·]" in repr(calc) or "bracket" in repr(calc)
