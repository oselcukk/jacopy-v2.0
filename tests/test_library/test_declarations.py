"""Tests for :mod:`jacopy.library.declarations` tutorial helpers."""

from __future__ import annotations

import pytest

from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.declarations import (
    Bivector,
    Forms,
    Functions,
    VectorFields,
)


class TestFunctions:
    def test_three_functions_return_tuple_of_graded_zero_symbols(self) -> None:
        reg = PropertyRegistry()
        f, g, h = Functions("f g h", registry=reg)
        assert f.name == "f"
        for s in (f, g, h):
            prop = reg.get(s, Graded)
            assert prop is not None
            assert prop.degree.as_int() == 0

    def test_single_name_returns_one_tuple(self) -> None:
        reg = PropertyRegistry()
        (f,) = Functions("f", registry=reg)
        assert f.name == "f"

    def test_empty_name_raises(self) -> None:
        reg = PropertyRegistry()
        with pytest.raises(ValueError):
            Functions("   ", registry=reg)

    def test_non_string_names_rejected(self) -> None:
        reg = PropertyRegistry()
        with pytest.raises(TypeError):
            Functions(["f", "g"], registry=reg)  # type: ignore[arg-type]

    def test_sn_shifted_degree(self) -> None:
        """Functions(degree=-1) for Schouten-Nijenhuis / Poisson context."""
        reg = PropertyRegistry()
        f, g = Functions("f g", degree=-1, registry=reg)
        for s in (f, g):
            prop = reg.get(s, Graded)
            assert prop is not None
            assert prop.degree.as_int() == -1

    def test_functions_degree_must_be_int(self) -> None:
        reg = PropertyRegistry()
        with pytest.raises(TypeError):
            Functions("f", degree=0.5, registry=reg)  # type: ignore[arg-type]


class TestVectorFields:
    def test_three_vector_fields(self) -> None:
        reg = PropertyRegistry()
        X, Y, Z = VectorFields("X Y Z", registry=reg)
        for s in (X, Y, Z):
            prop = reg.get(s, Graded)
            assert prop is not None
            assert prop.degree.as_int() == 0

    def test_integrates_with_prove_jacobi(self) -> None:
        """Jacobi proof should close on symbols declared via the helper."""
        from jacopy.brackets.lie import lie
        from jacopy.core.expr import Integer
        from jacopy.proof import prove_jacobi

        reg = PropertyRegistry()
        X, Y, Z = VectorFields("X Y Z", registry=reg)
        chain = prove_jacobi(lie, X, Y, Z, registry=reg)
        assert chain.steps[-1].after == Integer(0)


class TestForms:
    def test_one_form_has_degree_one(self) -> None:
        reg = PropertyRegistry()
        alpha, beta = Forms("alpha beta", degree=1, registry=reg)
        for s in (alpha, beta):
            prop = reg.get(s, Graded)
            assert prop is not None
            assert prop.degree.as_int() == 1

    def test_two_form_has_degree_two(self) -> None:
        reg = PropertyRegistry()
        (omega,) = Forms("omega", degree=2, registry=reg)
        prop = reg.get(omega, Graded)
        assert prop is not None
        assert prop.degree.as_int() == 2

    def test_degree_must_be_int(self) -> None:
        reg = PropertyRegistry()
        with pytest.raises(TypeError):
            Forms("alpha", degree=1.0, registry=reg)  # type: ignore[arg-type]


class TestBivector:
    def test_returns_bare_symbol_not_tuple(self) -> None:
        reg = PropertyRegistry()
        pi = Bivector("pi", registry=reg)
        assert pi.name == "pi"
        prop = reg.get(pi, Graded)
        assert prop is not None
        assert prop.degree.as_int() == 1

    def test_multiple_names_rejected(self) -> None:
        reg = PropertyRegistry()
        with pytest.raises(ValueError):
            Bivector("pi sigma", registry=reg)


class TestRegistryIsolation:
    def test_declarations_live_only_on_the_passed_registry(self) -> None:
        reg_a = PropertyRegistry()
        reg_b = PropertyRegistry()
        (f,) = Functions("f", registry=reg_a)
        assert reg_a.get(f, Graded) is not None
        assert reg_b.get(f, Graded) is None

    def test_re_declaring_same_name_on_fresh_symbol_works(self) -> None:
        """Each call creates a new Symbol instance, no conflict."""
        reg = PropertyRegistry()
        (f1,) = Functions("f", registry=reg)
        # second call: fresh Symbol("f"); but Symbol() is value-equal,
        # so the registry will already have a Graded declaration on
        # that key, expect ValueError, mirroring PropertyRegistry's
        # duplicate-declaration contract.
        with pytest.raises(ValueError):
            Functions("f", registry=reg)
