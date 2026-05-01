"""Tests for :class:`jacopy.core.multi_eval.MultiEval` (Faz 12.A.0)."""

import pytest

from jacopy.algebra.derivation import Derivation, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Sum, Symbol
from jacopy.core.multi_eval import (
    MultiEval,
    has_repeated_arg,
    multi_eval,
    validate_arity,
)
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.display.ascii import to_ascii
from jacopy.display.latex import to_latex


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_factory_matches_class(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        a = multi_eval(omega, X, Y)
        b = MultiEval(omega, X, Y)
        assert a == b
        assert isinstance(a, MultiEval)

    def test_accessors(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        m = multi_eval(omega, X, Y, Z)
        assert m.head is omega
        assert m.args == (X, Y, Z)
        assert m.arity == 3
        assert m.alternating is True
        assert m.slot_kind == "vector"

    def test_children_include_head_and_args(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y)
        assert m.children == (omega, X, Y)
        assert not m.is_atom

    def test_repr_shape(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y)
        assert repr(m) == "ω(X, Y)"

    def test_zero_args_rejected(self):
        with pytest.raises(ValueError, match="at least one argument"):
            MultiEval(Symbol("ω"))

    def test_non_expr_head_rejected(self):
        with pytest.raises(TypeError, match="head must be an Expr"):
            MultiEval(7, Symbol("X"))  # type: ignore[arg-type]

    def test_non_expr_arg_rejected(self):
        with pytest.raises(TypeError, match="arguments must be Expr"):
            MultiEval(Symbol("ω"), Symbol("X"), 3)  # type: ignore[arg-type]

    def test_alternating_must_be_bool(self):
        with pytest.raises(TypeError, match="alternating must be a bool"):
            MultiEval(Symbol("ω"), Symbol("X"), alternating="yes")  # type: ignore[arg-type]

    def test_slot_kind_validated(self):
        with pytest.raises(ValueError, match="slot_kind must be one of"):
            MultiEval(Symbol("ω"), Symbol("X"), slot_kind="weird")


# --------------------------------------------------------------------- #
# Equality / hashing                                                    #
# --------------------------------------------------------------------- #


class TestEqualityAndHash:
    def test_structural_equality(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        a = multi_eval(omega, X, Y)
        b = multi_eval(omega, X, Y)
        assert a == b
        assert hash(a) == hash(b)

    def test_arg_order_distinguishes(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        assert multi_eval(omega, X, Y) != multi_eval(omega, Y, X)

    def test_alternating_flag_distinguishes(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        a = multi_eval(omega, X, Y, alternating=True)
        b = multi_eval(omega, X, Y, alternating=False)
        assert a != b

    def test_slot_kind_distinguishes(self):
        head = Symbol("π")
        arg = Symbol("α")
        a = multi_eval(head, arg, slot_kind="vector")
        b = multi_eval(head, arg, slot_kind="covector")
        assert a != b

    def test_unequal_to_other_expr_types(self):
        omega = Symbol("ω")
        X = Symbol("X")
        m = multi_eval(omega, X)
        assert m != Sum(omega, X)
        assert m != omega


# --------------------------------------------------------------------- #
# Tree manipulation                                                     #
# --------------------------------------------------------------------- #


class TestTreeOps:
    def test_rebuild_preserves_flags(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        m = multi_eval(omega, X, Y, alternating=False, slot_kind="covector")
        m2 = m._rebuild((omega, X, Z))
        assert m2.head is omega
        assert m2.args == (X, Z)
        assert m2.alternating is False
        assert m2.slot_kind == "covector"

    def test_rebuild_arity_check(self):
        omega = Symbol("ω")
        X = Symbol("X")
        m = multi_eval(omega, X)
        with pytest.raises(ValueError, match="head plus at least one arg"):
            m._rebuild((omega,))

    def test_replace_at_descends_into_args(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        m = multi_eval(omega, X, Y)
        # children index 2 == second arg
        m2 = m.replace_at((2,), Z)
        assert m2.args == (X, Z)

    def test_with_args_preserves_head_and_flags(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        m = multi_eval(omega, X, Y, slot_kind="covector")
        m2 = m.with_args(Z, X)
        assert m2.head is omega
        assert m2.args == (Z, X)
        assert m2.slot_kind == "covector"


# --------------------------------------------------------------------- #
# Antisymmetry helpers                                                  #
# --------------------------------------------------------------------- #


class TestAntisymmetryHelpers:
    def test_swapped_alternating_yields_minus_one(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y)
        m2, sign = m.swapped(0, 1)
        assert m2.args == (Y, X)
        assert sign == -1

    def test_swapped_same_index_is_identity(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y)
        m2, sign = m.swapped(1, 1)
        assert m2 == m
        assert sign == 1

    def test_swapped_non_alternating_keeps_sign(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y, alternating=False)
        m2, sign = m.swapped(0, 1)
        assert m2.args == (Y, X)
        assert sign == 1

    def test_swapped_index_out_of_range(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y)
        with pytest.raises(IndexError):
            m.swapped(0, 7)

    def test_has_repeated_arg_detects_duplicate(self):
        omega = Symbol("ω")
        X = Symbol("X")
        Y = Symbol("Y")
        assert has_repeated_arg(multi_eval(omega, X, X))
        assert has_repeated_arg(multi_eval(omega, X, Y, X))
        assert not has_repeated_arg(multi_eval(omega, X, Y))

    def test_has_repeated_arg_rejects_non_multi_eval(self):
        with pytest.raises(TypeError, match="expected a MultiEval"):
            has_repeated_arg(Symbol("X"))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Grading                                                                #
# --------------------------------------------------------------------- #


class TestGrading:
    def test_degree_is_zero_independent_of_inputs(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        reg = PropertyRegistry()
        reg.declare(omega, Graded(degree=2))
        m = multi_eval(omega, X, Y)
        assert degree_of(m, reg) == Degree.const(0)

    def test_degree_does_not_require_registry(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        # No registry, head's degree is undefined, but MultiEval still
        # reports degree 0 (an evaluation always produces a scalar).
        m = multi_eval(omega, X, Y)
        assert degree_of(m) == Degree.const(0)


# --------------------------------------------------------------------- #
# Arity validation                                                      #
# --------------------------------------------------------------------- #


class TestValidateArity:
    def test_returns_head_degree_on_match(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        reg = PropertyRegistry()
        reg.declare(omega, Graded(degree=2))
        m = multi_eval(omega, X, Y)
        assert validate_arity(m, registry=reg) == 2

    def test_raises_on_mismatch(self):
        omega = Symbol("ω")
        X = Symbol("X")
        reg = PropertyRegistry()
        reg.declare(omega, Graded(degree=2))
        m = multi_eval(omega, X)
        with pytest.raises(ValueError, match="arity mismatch"):
            validate_arity(m, registry=reg)

    def test_returns_none_when_head_degree_unknown(self):
        omega = Symbol("ω")
        X = Symbol("X")
        m = multi_eval(omega, X)
        # No registry → degree_of(omega) raises → returns None.
        assert validate_arity(m) is None


# --------------------------------------------------------------------- #
# Render dispatch                                                       #
# --------------------------------------------------------------------- #


class TestRenderers:
    def test_to_ascii_basic(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y)
        assert to_ascii(m) == "ω(X, Y)"

    def test_to_ascii_with_compound_arg(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        m = multi_eval(omega, X, Sum(Y, Z))
        # The Sum arg renders without outer parens because the call's
        # internal context is 0.
        assert to_ascii(m) == "ω(X, Y + Z)"

    def test_to_latex_basic(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        m = multi_eval(omega, X, Y)
        out = to_latex(m)
        assert "\\omega" in out
        assert "X" in out and "Y" in out
        assert "\\left(" in out and "\\right)" in out

    def test_to_latex_arg_separator(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        m = multi_eval(omega, X, Y, Z)
        out = to_latex(m)
        # Three args produce two LaTeX-style separators ",\,".
        assert out.count(",\\,") == 2
