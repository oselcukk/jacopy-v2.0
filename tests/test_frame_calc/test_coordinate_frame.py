"""Tests for `jacopy.frame_calc.frame.CoordinateFrame` (Faz 18 Stage A)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    AbstractFrame,
    CoordinateFrame,
    Frame,
    Tetrad,
)


# --------------------------------------------------------------------- #
# Frame protocol                                                        #
# --------------------------------------------------------------------- #


class TestFrameProtocol:
    """The base ``Frame`` is an abstract protocol, direct instantiation is
    rejected; subclasses inherit the ``index_names`` default and the index
    bounds-check helper."""

    def test_direct_instantiation_rejected(self) -> None:
        with pytest.raises(TypeError, match="abstract protocol"):
            Frame()

    def test_subclasses_pass_isinstance_check(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        assert isinstance(F, Frame)


# --------------------------------------------------------------------- #
# CoordinateFrame construction                                          #
# --------------------------------------------------------------------- #


class TestCoordinateFrameConstruction:
    def test_basic_construction(self) -> None:
        t, r, theta, phi = sp.symbols("t r theta phi")
        F = CoordinateFrame([t, r, theta, phi])
        assert F.dim == 4
        assert F.coords == (t, r, theta, phi)

    def test_default_name(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        assert F.name == "coord(t,r)"

    def test_explicit_name_override(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r], name="planar")
        assert F.name == "planar"

    def test_empty_coords_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one coordinate"):
            CoordinateFrame([])

    def test_non_symbol_rejected(self) -> None:
        t = sp.Symbol("t")
        with pytest.raises(TypeError, match="SymPy Symbols"):
            CoordinateFrame([t, "r"])  # type: ignore[list-item]

    def test_index_names_returns_coord_strings(self) -> None:
        t, r, theta = sp.symbols("t r theta")
        F = CoordinateFrame([t, r, theta])
        assert F.index_names() == ("t", "r", "theta")

    def test_coord_accessor(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        assert F.coord(0) is t
        assert F.coord(1) is r

    def test_coord_index_out_of_range(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        with pytest.raises(IndexError, match="out of range"):
            F.coord(1)

    def test_repr_contains_name_and_dim(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        assert "coord(t,r)" in repr(F)
        assert "dim=2" in repr(F)


# --------------------------------------------------------------------- #
# CoordinateFrame.derivative                                            #
# --------------------------------------------------------------------- #


class TestCoordinateFrameDerivative:
    def test_derivative_of_coordinate_returns_one(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        assert F.derivative(t, 0) == 1
        assert F.derivative(r, 1) == 1

    def test_derivative_of_other_coordinate_returns_zero(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        assert F.derivative(t, 1) == 0
        assert F.derivative(r, 0) == 0

    def test_derivative_of_polynomial(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        assert sp.simplify(F.derivative(r**2, 1) - 2 * r) == 0

    def test_derivative_of_product(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        result = F.derivative(t * r**2, 1)
        assert sp.simplify(result - 2 * t * r) == 0

    def test_derivative_of_special_function(self) -> None:
        t, r, theta, phi = sp.symbols("t r theta phi")
        F = CoordinateFrame([t, r, theta, phi])
        result = F.derivative(sp.sin(theta), 2)
        assert sp.simplify(result - sp.cos(theta)) == 0

    def test_derivative_accepts_python_int(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        assert F.derivative(5, 0) == 0

    def test_derivative_returns_sympy_expr(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        assert isinstance(F.derivative(t**2, 0), sp.Expr)

    def test_derivative_index_out_of_range(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        with pytest.raises(IndexError):
            F.derivative(t, 2)

    def test_derivative_negative_index_rejected(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        with pytest.raises(IndexError):
            F.derivative(t, -1)

    def test_derivative_non_int_index_rejected(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        with pytest.raises(TypeError, match="index must be int"):
            F.derivative(t, "0")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# CoordinateFrame.gamma                                                 #
# --------------------------------------------------------------------- #


class TestCoordinateFrameGamma:
    def test_gamma_is_zero_everywhere(self) -> None:
        t, r, theta, phi = sp.symbols("t r theta phi")
        F = CoordinateFrame([t, r, theta, phi])
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    assert F.gamma(a, b, c) == 0

    def test_gamma_returns_sympy_zero(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        assert isinstance(F.gamma(0, 0, 0), sp.Integer)

    def test_gamma_index_out_of_range(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        with pytest.raises(IndexError):
            F.gamma(1, 0, 0)
        with pytest.raises(IndexError):
            F.gamma(0, 1, 0)
        with pytest.raises(IndexError):
            F.gamma(0, 0, 1)


# --------------------------------------------------------------------- #
# CoordinateFrame equality + hashing                                    #
# --------------------------------------------------------------------- #


class TestCoordinateFrameEquality:
    def test_same_coords_compare_equal(self) -> None:
        t, r = sp.symbols("t r")
        F1 = CoordinateFrame([t, r])
        F2 = CoordinateFrame([t, r])
        assert F1 == F2
        assert hash(F1) == hash(F2)

    def test_different_coords_compare_unequal(self) -> None:
        t, r, x = sp.symbols("t r x")
        F1 = CoordinateFrame([t, r])
        F2 = CoordinateFrame([t, x])
        assert F1 != F2

    def test_different_names_compare_unequal(self) -> None:
        t, r = sp.symbols("t r")
        F1 = CoordinateFrame([t, r], name="A")
        F2 = CoordinateFrame([t, r], name="B")
        assert F1 != F2

    def test_unrelated_object_not_equal(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        assert F != "coord(t)"
        assert F != 0


# --------------------------------------------------------------------- #
# AbstractFrame stub (Stage A.2 placeholder)                            #
# --------------------------------------------------------------------- #


class TestAbstractFrameConstruction:
    """Stage A.2, `AbstractFrame` is fully populated."""

    def test_construction_with_dim(self) -> None:
        F = AbstractFrame(dim=4)
        assert F.dim == 4
        assert F.name == "abstract(4)"
        assert F.index_names() == ("0", "1", "2", "3")

    def test_construction_with_index_names(self) -> None:
        F = AbstractFrame(dim=2, index_names=("u", "v"))
        assert F.index_names() == ("u", "v")

    def test_index_names_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="index_names length"):
            AbstractFrame(dim=3, index_names=("u", "v"))

    def test_zero_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            AbstractFrame(dim=0)

    def test_negative_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            AbstractFrame(dim=-1)

    def test_explicit_name_override(self) -> None:
        F = AbstractFrame(dim=2, name="my-frame")
        assert F.name == "my-frame"

    def test_gamma_table_invalid_key_shape_rejected(self) -> None:
        with pytest.raises(TypeError, match="tuples"):
            AbstractFrame(dim=2, gamma_table={(0, 1): "x"})  # type: ignore[dict-item]

    def test_gamma_table_index_out_of_range_rejected(self) -> None:
        with pytest.raises(IndexError, match="out of range"):
            AbstractFrame(dim=2, gamma_table={(0, 1, 5): "x"})


class TestAbstractFrameDerivative:
    def test_derivative_returns_frame_derivative_expr(self) -> None:
        from jacopy.core.expr import Symbol
        from jacopy.frame_calc import FrameDerivativeExpr

        F = AbstractFrame(dim=2, index_names=("u", "v"))
        body = Symbol("g_00")
        result = F.derivative(body, 0)
        assert isinstance(result, FrameDerivativeExpr)
        assert result.frame is F
        assert result.index == 0
        assert result.body is body

    def test_derivative_repr(self) -> None:
        from jacopy.core.expr import Symbol

        F = AbstractFrame(dim=2, index_names=("u", "v"))
        result = F.derivative(Symbol("g_00"), 1)
        assert "e_v" in repr(result)
        assert "g_00" in repr(result)

    def test_derivative_index_out_of_range(self) -> None:
        from jacopy.core.expr import Symbol

        F = AbstractFrame(dim=2)
        with pytest.raises(IndexError):
            F.derivative(Symbol("x"), 5)

    def test_derivative_non_expr_body_rejected(self) -> None:
        F = AbstractFrame(dim=2)
        with pytest.raises(TypeError, match="Expr body"):
            F.derivative("a string", 0)

    def test_derivative_equality_structural(self) -> None:
        from jacopy.core.expr import Symbol

        F = AbstractFrame(dim=2)
        body = Symbol("g_00")
        d1 = F.derivative(body, 0)
        d2 = F.derivative(body, 0)
        assert d1 == d2
        assert hash(d1) == hash(d2)

    def test_derivative_different_frames_distinguish(self) -> None:
        from jacopy.core.expr import Symbol

        F1 = AbstractFrame(dim=2)
        F2 = AbstractFrame(dim=2)
        body = Symbol("g_00")
        d1 = F1.derivative(body, 0)
        d2 = F2.derivative(body, 0)
        # Two distinct AbstractFrames stay distinct: keying on id().
        assert d1 != d2


class TestAbstractFrameGamma:
    def test_gamma_returns_gamma_expr_when_no_table(self) -> None:
        from jacopy.frame_calc import GammaExpr

        F = AbstractFrame(dim=3)
        g = F.gamma(0, 1, 2)
        assert isinstance(g, GammaExpr)
        assert g.upper == 0
        assert g.lower == (1, 2)
        assert g.frame is F

    def test_gamma_repr(self) -> None:
        F = AbstractFrame(dim=3, index_names=("a", "b", "c"))
        g = F.gamma(0, 1, 2)
        assert repr(g) == "γ^a_{bc}"

    def test_gamma_table_lookup(self) -> None:
        from jacopy.core.expr import Integer, Symbol

        sym = Symbol("γ012")
        F = AbstractFrame(
            dim=3,
            gamma_table={(0, 1, 2): sym, (0, 0, 0): Integer(0)},
        )
        assert F.gamma(0, 1, 2) is sym
        assert F.gamma(0, 0, 0) == Integer(0)

    def test_gamma_table_miss_returns_opaque(self) -> None:
        from jacopy.core.expr import Symbol
        from jacopy.frame_calc import GammaExpr

        F = AbstractFrame(dim=3, gamma_table={(0, 1, 2): Symbol("g")})
        miss = F.gamma(1, 0, 2)
        assert isinstance(miss, GammaExpr)

    def test_gamma_index_out_of_range(self) -> None:
        F = AbstractFrame(dim=2)
        with pytest.raises(IndexError):
            F.gamma(2, 0, 0)
        with pytest.raises(IndexError):
            F.gamma(0, 2, 0)
        with pytest.raises(IndexError):
            F.gamma(0, 0, 2)

    def test_gamma_equality_structural(self) -> None:
        F = AbstractFrame(dim=3)
        g1 = F.gamma(0, 1, 2)
        g2 = F.gamma(0, 1, 2)
        assert g1 == g2
        assert hash(g1) == hash(g2)

    def test_gamma_different_indices_distinguish(self) -> None:
        F = AbstractFrame(dim=3)
        assert F.gamma(0, 1, 2) != F.gamma(0, 2, 1)
        assert F.gamma(0, 1, 2) != F.gamma(1, 1, 2)

    def test_gamma_different_frames_distinguish(self) -> None:
        F1 = AbstractFrame(dim=3)
        F2 = AbstractFrame(dim=3)
        assert F1.gamma(0, 1, 2) != F2.gamma(0, 1, 2)


class TestAbstractFrameAtomComposition:
    """Symbolic atoms compose with the rest of the Expr algebra."""

    def test_derivative_inside_sum(self) -> None:
        from jacopy.core.expr import Symbol, Sum

        F = AbstractFrame(dim=2)
        d1 = F.derivative(Symbol("a"), 0)
        d2 = F.derivative(Symbol("b"), 0)
        s = Sum(d1, d2)
        # Sum walks its children; derivative atoms participate cleanly.
        assert d1 in tuple(s.walk())
        assert d2 in tuple(s.walk())

    def test_gamma_inside_product(self) -> None:
        from jacopy.core.expr import Product

        F = AbstractFrame(dim=2)
        p = Product(F.gamma(0, 0, 1), F.gamma(1, 1, 0))
        atoms = [n for n in p.walk() if n.is_atom]
        # Both gamma atoms appear inside the product.
        assert len(atoms) == 2


# --------------------------------------------------------------------- #
# Tetrad stub (Stage B placeholder)                                     #
# --------------------------------------------------------------------- #


class TestTetradConstruction:
    """Stage B: Tetrad fully populated with vielbein-based computation."""

    def test_construction_succeeds(self) -> None:
        t, r = sp.symbols("t r")
        coord_frame = CoordinateFrame([t, r])
        T = Tetrad(coord_frame, vielbein=sp.eye(2))
        assert T.dim == 2
        assert T.name == "tetrad(coord(t,r))"
        assert isinstance(T, Tetrad)

    def test_non_coord_frame_rejected(self) -> None:
        with pytest.raises(TypeError, match="CoordinateFrame"):
            Tetrad("not a frame", vielbein=None)  # type: ignore[arg-type]

    def test_vielbein_shape_mismatch_rejected(self) -> None:
        t, r = sp.symbols("t r")
        coord_frame = CoordinateFrame([t, r])
        with pytest.raises(ValueError, match="shape"):
            Tetrad(coord_frame, vielbein=sp.eye(3))

    def test_invalid_vielbein_type(self) -> None:
        t = sp.Symbol("t")
        with pytest.raises(TypeError, match="Matrix"):
            Tetrad(CoordinateFrame([t]), vielbein="not a matrix")

    def test_explicit_name(self) -> None:
        t = sp.Symbol("t")
        T = Tetrad(
            CoordinateFrame([t]), vielbein=sp.eye(1), name="my-tetrad"
        )
        assert T.name == "my-tetrad"


class TestTetradIdentityCase:
    """Identity vielbein → tetrad coincides with the coord frame."""

    def test_derivative_matches_coord_frame(self) -> None:
        t, r = sp.symbols("t r")
        coord_frame = CoordinateFrame([t, r])
        T = Tetrad(coord_frame, vielbein=sp.eye(2))
        # e_a(f) should equal coord_frame.derivative(f, a)
        assert sp.simplify(T.derivative(r**2, 1) - 2 * r) == 0
        assert sp.simplify(T.derivative(t * r, 0) - r) == 0

    def test_gamma_zero_for_identity(self) -> None:
        t, r = sp.symbols("t r")
        coord_frame = CoordinateFrame([t, r])
        T = Tetrad(coord_frame, vielbein=sp.eye(2))
        # Identity vielbein → constant frame components → bracket = 0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    assert T.gamma(a, b, c) == 0


class TestTetradGamma:
    """Non-trivial vielbein → genuine structure constants."""

    def test_gamma_antisymmetric(self) -> None:
        """γ^a_{bc} = -γ^a_{cb} for any tetrad."""
        x, y = sp.symbols("x y", positive=True)
        coord_frame = CoordinateFrame([x, y])
        # A non-trivial vielbein with x-dependence
        vielbein = sp.Matrix([
            [1, 0],
            [0, 1 / x],   # e_1 = (1/x) ∂/∂y
        ])
        T = Tetrad(coord_frame, vielbein=vielbein)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    assert sp.simplify(
                        T.gamma(a, b, c) + T.gamma(a, c, b)
                    ) == 0

    def test_gamma_b_equals_c_zero(self) -> None:
        x, y = sp.symbols("x y", positive=True)
        coord_frame = CoordinateFrame([x, y])
        vielbein = sp.Matrix([[1, 0], [0, 1 / x]])
        T = Tetrad(coord_frame, vielbein=vielbein)
        for a in range(2):
            for b in range(2):
                assert T.gamma(a, b, b) == 0

    def test_singular_vielbein_raises(self) -> None:
        x, y = sp.symbols("x y")
        coord_frame = CoordinateFrame([x, y])
        # Singular vielbein
        T = Tetrad(coord_frame, vielbein=sp.Matrix([[1, 0], [0, 0]]))
        with pytest.raises(ValueError, match="singular"):
            T.gamma(0, 0, 1)


class TestTetradWithLeviCivita:
    """The frame protocol should now let levi_civita work on a Tetrad."""

    def test_identity_tetrad_minkowski_christoffel_zero(self) -> None:
        from jacopy.frame_calc import ComponentMetric, levi_civita

        t, x, y, z = sp.symbols("t x y z", real=True)
        coord_frame = CoordinateFrame([t, x, y, z])
        T = Tetrad(coord_frame, vielbein=sp.eye(4))
        # In an identity-tetrad with Minkowski metric in tetrad indices,
        # the Christoffel symbols should be zero (flat space, flat tetrad).
        g = ComponentMetric(T, sp.diag(-1, 1, 1, 1))
        LC = levi_civita(g)
        assert LC.is_zero()
