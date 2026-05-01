"""Tests for jacopy.calculus.exterior_algebra."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.exterior_d import ExteriorDerivative, d
from jacopy.core.expr import Symbol


class TestConstruction:
    def test_empty_is_fine(self):
        omega = ExteriorAlgebra([])
        assert omega.functions == ()
        assert omega.generators == ()
        assert omega.d is d

    def test_functions_stored(self):
        f, g = Symbol("f"), Symbol("g")
        omega = ExteriorAlgebra([f, g])
        assert omega.functions == (f, g)

    def test_custom_d_respected(self):
        d_E = ExteriorDerivative("d_E")
        omega = ExteriorAlgebra([Symbol("f")], d=d_E)
        assert omega.d is d_E

    def test_rejects_non_expr_function(self):
        with pytest.raises(TypeError):
            ExteriorAlgebra(["f"])  # type: ignore[list-item]


class TestGenerators:
    def test_generators_are_functions_and_their_differentials(self):
        f, g = Symbol("f"), Symbol("g")
        omega = ExteriorAlgebra([f, g])
        gens = omega.generators
        assert gens[:2] == (f, g)
        assert gens[2] == Act(d, f)
        assert gens[3] == Act(d, g)

    def test_generators_use_algebra_d(self):
        d_E = ExteriorDerivative("d_E")
        f = Symbol("f")
        omega = ExteriorAlgebra([f], d=d_E)
        assert omega.generators[1] == Act(d_E, f)


class TestIsGeneratedBy:
    def test_identical_generator_set_covers(self):
        f = Symbol("f")
        omega = ExteriorAlgebra([f])
        assert omega.is_generated_by(omega.generators)

    def test_superset_covers(self):
        f = Symbol("f")
        omega = ExteriorAlgebra([f])
        # Extra elements beyond the declared generators are fine.
        assert omega.is_generated_by(list(omega.generators) + [Symbol("extra")])

    def test_missing_one_form_fails(self):
        f = Symbol("f")
        omega = ExteriorAlgebra([f])
        # Only the function, not its differential, insufficient.
        assert not omega.is_generated_by([f])

    def test_empty_trivially_covered(self):
        omega = ExteriorAlgebra([])
        assert omega.is_generated_by([])
