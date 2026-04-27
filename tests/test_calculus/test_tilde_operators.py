"""Tests for tilde-calculus operator atoms (Faz 14.A)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.calculus.tilde import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
    tilde_d,
    tilde_interior,
    tilde_lie,
)
from jacopy.core.expr import Symbol
from jacopy.core.symbolic_degree import Degree
from jacopy.display.ascii import to_ascii
from jacopy.display.latex import to_latex


# --------------------------------------------------------------------- #
# TildeInteriorProduct                                                  #
# --------------------------------------------------------------------- #


class TestTildeInteriorProduct:
    def test_is_degree_minus_one_derivation(self):
        omega = Symbol("ω")
        ti = tilde_interior(omega)
        assert isinstance(ti, Derivation)
        assert ti.degree == Degree.const(-1)

    def test_operator_degree_via_degree_of(self):
        omega = Symbol("ω")
        assert degree_of(tilde_interior(omega)) == Degree.const(-1)

    def test_default_name_carries_form(self):
        omega = Symbol("ω")
        assert tilde_interior(omega).name == "ι̃_ω"

    def test_custom_name(self):
        omega = Symbol("ω")
        assert tilde_interior(omega, name="ι̃_E_ω").name == "ι̃_E_ω"

    def test_carries_form_attribute(self):
        omega = Symbol("ω")
        assert tilde_interior(omega).form is omega

    def test_equality_on_form(self):
        omega = Symbol("ω")
        assert tilde_interior(omega) == tilde_interior(omega)

    def test_distinct_forms_give_distinct_operators(self):
        assert tilde_interior(Symbol("ω")) != tilde_interior(Symbol("η"))

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            TildeInteriorProduct("ω")  # type: ignore[arg-type]

    def test_hash_matches_equality(self):
        omega = Symbol("ω")
        assert hash(tilde_interior(omega)) == hash(tilde_interior(omega))

    def test_latex_render(self):
        omega = Symbol("ω")
        V = Symbol("V")
        rendered = to_latex(Act(tilde_interior(omega), V))
        assert r"\tilde{\iota}_{\omega}" in rendered

    def test_ascii_render_uses_name(self):
        omega = Symbol("ω")
        V = Symbol("V")
        assert to_ascii(Act(tilde_interior(omega), V)) == "ι̃_ω(V)"


# --------------------------------------------------------------------- #
# TildeExteriorDerivative                                               #
# --------------------------------------------------------------------- #


class TestTildeExteriorDerivative:
    def test_is_degree_one_derivation(self):
        pi = Symbol("π")
        td = tilde_d(pi)
        assert isinstance(td, Derivation)
        assert td.degree == Degree.const(1)

    def test_default_name_carries_bivector(self):
        pi = Symbol("π")
        assert tilde_d(pi).name == "d̃_π"

    def test_custom_name(self):
        pi = Symbol("π")
        assert tilde_d(pi, name="d̃_E").name == "d̃_E"

    def test_carries_bivector_attribute(self):
        pi = Symbol("π")
        assert tilde_d(pi).bivector is pi

    def test_equality_on_bivector(self):
        pi = Symbol("π")
        assert tilde_d(pi) == tilde_d(pi)

    def test_distinct_bivectors_give_distinct_operators(self):
        assert tilde_d(Symbol("π1")) != tilde_d(Symbol("π2"))

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            TildeExteriorDerivative("π")  # type: ignore[arg-type]

    def test_latex_render(self):
        pi = Symbol("π")
        V = Symbol("V")
        rendered = to_latex(Act(tilde_d(pi), V))
        assert r"\tilde{d}" in rendered


# --------------------------------------------------------------------- #
# TildeLieDerivative                                                    #
# --------------------------------------------------------------------- #


class TestTildeLieDerivative:
    def test_is_degree_zero_derivation(self):
        omega, pi = Symbol("ω"), Symbol("π")
        tl = tilde_lie(omega, pi)
        assert isinstance(tl, Derivation)
        assert tl.degree == Degree.const(0)

    def test_default_name_carries_form(self):
        omega, pi = Symbol("ω"), Symbol("π")
        assert tilde_lie(omega, pi).name == "L̃_ω"

    def test_carries_form_and_bivector(self):
        omega, pi = Symbol("ω"), Symbol("π")
        tl = tilde_lie(omega, pi)
        assert tl.form is omega
        assert tl.bivector is pi

    def test_equality_on_form_and_bivector(self):
        omega, pi = Symbol("ω"), Symbol("π")
        assert tilde_lie(omega, pi) == tilde_lie(omega, pi)

    def test_distinct_forms_give_distinct_operators(self):
        pi = Symbol("π")
        assert tilde_lie(Symbol("ω"), pi) != tilde_lie(Symbol("η"), pi)

    def test_distinct_bivectors_give_distinct_operators(self):
        omega = Symbol("ω")
        assert tilde_lie(omega, Symbol("π1")) != tilde_lie(omega, Symbol("π2"))

    def test_rejects_non_expr_form(self):
        with pytest.raises(TypeError):
            TildeLieDerivative("ω", Symbol("π"))  # type: ignore[arg-type]

    def test_rejects_non_expr_bivector(self):
        with pytest.raises(TypeError):
            TildeLieDerivative(Symbol("ω"), "π")  # type: ignore[arg-type]

    def test_latex_render(self):
        omega, pi = Symbol("ω"), Symbol("π")
        V = Symbol("V")
        rendered = to_latex(Act(tilde_lie(omega, pi), V))
        assert r"\tilde{\mathcal{L}}_{\omega}" in rendered


# --------------------------------------------------------------------- #
# Cross-operator interaction                                            #
# --------------------------------------------------------------------- #


class TestCrossOperator:
    def test_three_operators_are_pairwise_distinct(self):
        omega, pi = Symbol("ω"), Symbol("π")
        ops = [tilde_interior(omega), tilde_d(pi), tilde_lie(omega, pi)]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            assert ops[a] != ops[b]

    def test_module_exports_via_calculus_namespace(self):
        from jacopy.calculus import (
            TildeExteriorDerivative as _TED,
            TildeInteriorProduct as _TIP,
            TildeLieDerivative as _TLD,
            tilde_d as _td,
            tilde_interior as _ti,
            tilde_lie as _tl,
        )

        assert _TED is TildeExteriorDerivative
        assert _TIP is TildeInteriorProduct
        assert _TLD is TildeLieDerivative
        assert _td is tilde_d
        assert _ti is tilde_interior
        assert _tl is tilde_lie
