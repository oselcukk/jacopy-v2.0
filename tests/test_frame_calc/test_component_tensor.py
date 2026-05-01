"""Tests for `jacopy.frame_calc.component_tensor` (Faz 18 Stage C)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    AbstractFrame,
    ComponentConnection,
    ComponentMetric,
    ComponentMetricInverse,
    ComponentTensor,
    CoordinateFrame,
)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def polar() -> tuple[CoordinateFrame, sp.Matrix]:
    """Polar 2D coordinate frame + flat metric `dr² + r² dθ²`."""
    r, theta = sp.symbols("r theta", positive=True)
    F = CoordinateFrame([r, theta])
    g = sp.Matrix([[1, 0], [0, r**2]])
    return F, g


@pytest.fixture
def schwarzschild_2d() -> tuple[CoordinateFrame, sp.Matrix]:
    """A 2D slice of Schwarzschild, for non-trivial inversion tests."""
    t, r = sp.symbols("t r", positive=True)
    M = sp.Symbol("M", positive=True)
    F = CoordinateFrame([t, r])
    g = sp.Matrix([
        [-(1 - 2 * M / r), 0],
        [0, 1 / (1 - 2 * M / r)],
    ])
    return F, g


# --------------------------------------------------------------------- #
# ComponentTensor base class                                            #
# --------------------------------------------------------------------- #


class TestComponentTensorBase:
    def test_construction_from_matrix(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.Matrix([[1, 0], [0, 1]]), signature=(0, 2))
        assert T.signature == (0, 2)
        assert T.rank == 2
        assert T.shape == (2, 2)

    def test_construction_from_nested_list(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        T = ComponentTensor(F, [[5]], signature=(0, 2))
        assert T[0, 0] == 5

    def test_construction_from_array(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        arr = sp.MutableDenseNDimArray([[1, 0], [0, 2]])
        T = ComponentTensor(F, arr, signature=(0, 2))
        assert T.shape == (2, 2)

    def test_shape_mismatch_rejected(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        with pytest.raises(ValueError, match="shape mismatch"):
            ComponentTensor(F, sp.Matrix([[1, 0, 0]]), signature=(0, 2))

    def test_invalid_signature(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        with pytest.raises(ValueError, match="signature"):
            ComponentTensor(F, sp.Matrix([[1]]), signature=(0,))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="signature"):
            ComponentTensor(F, sp.Matrix([[1]]), signature=(-1, 1))

    def test_non_frame_rejected(self) -> None:
        with pytest.raises(TypeError, match="Frame"):
            ComponentTensor("not a frame", [], signature=(0, 0))  # type: ignore[arg-type]

    def test_indexing_returns_components(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.Matrix([[t, 0], [0, r]]), signature=(0, 2))
        assert T[0, 0] == t
        assert T[1, 1] == r
        assert T[0, 1] == 0

    def test_indexing_wrong_arity(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        T = ComponentTensor(F, [[1]], signature=(0, 2))
        with pytest.raises(IndexError, match="expected 2 indices"):
            T[0]

    def test_indexing_out_of_range(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.eye(2), signature=(0, 2))
        with pytest.raises(IndexError, match="out of range"):
            T[2, 0]

    def test_all_indices_iteration(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.eye(2), signature=(0, 2))
        idxs = list(T.all_indices())
        assert idxs == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_higher_rank(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        # 2x2x2 tensor for (1, 2)
        comp = sp.MutableDenseNDimArray.zeros(2, 2, 2)
        comp[1, 0, 1] = sp.Symbol("X")
        T = ComponentTensor(F, comp, signature=(1, 2))
        assert T.shape == (2, 2, 2)
        assert T.rank == 3
        assert T[1, 0, 1] == sp.Symbol("X")
        assert T[0, 0, 0] == 0

    def test_repr(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t], name="line")
        T = ComponentTensor(F, [[1]], signature=(0, 2))
        rep = repr(T)
        assert "ComponentTensor" in rep
        assert "line" in rep
        assert "(0, 2)" in rep


# --------------------------------------------------------------------- #
# ComponentMetric                                                       #
# --------------------------------------------------------------------- #


class TestComponentMetric:
    def test_construction_polar(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        assert m.signature == (0, 2)
        r = sp.symbols("r", positive=True)
        assert m[1, 1] == r**2
        assert m[0, 1] == 0

    def test_non_symmetric_rejected(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        with pytest.raises(ValueError, match="not symmetric"):
            ComponentMetric(F, sp.Matrix([[1, 1], [0, 1]]))

    def test_matrix_round_trip(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        assert m.matrix() == g

    def test_det_polar(self, polar) -> None:
        F, g = polar
        r = sp.symbols("r", positive=True)
        m = ComponentMetric(F, g)
        assert sp.simplify(m.det() - r**2) == 0

    def test_det_schwarzschild_2d(self, schwarzschild_2d) -> None:
        F, g = schwarzschild_2d
        m = ComponentMetric(F, g)
        # det = -(1-2M/r) * 1/(1-2M/r) = -1
        assert sp.simplify(m.det() + 1) == 0

    def test_inverse_polar(self, polar) -> None:
        F, g = polar
        r = sp.symbols("r", positive=True)
        m = ComponentMetric(F, g)
        m_inv = m.inverse()
        assert isinstance(m_inv, ComponentMetricInverse)
        assert m_inv.signature == (2, 0)
        # g^{rr} = 1, g^{θθ} = 1/r²
        assert sp.simplify(m_inv[0, 0] - 1) == 0
        assert sp.simplify(m_inv[1, 1] - 1 / r**2) == 0
        assert sp.simplify(m_inv[0, 1]) == 0

    def test_inverse_schwarzschild_2d(self, schwarzschild_2d) -> None:
        F, g = schwarzschild_2d
        M, r = sp.symbols("M r", positive=True)
        m = ComponentMetric(F, g)
        m_inv = m.inverse()
        # g^{tt} = -1/(1-2M/r), g^{rr} = (1-2M/r)
        assert sp.simplify(m_inv[0, 0] + 1 / (1 - 2 * M / r)) == 0
        assert sp.simplify(m_inv[1, 1] - (1 - 2 * M / r)) == 0

    def test_inverse_round_trip(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        # (g^-1)^-1 == g
        m_recovered = m.inverse().inverse()
        assert m_recovered.equals(m)

    def test_inverse_satisfies_delta(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        m_inv = m.inverse()
        # g^{ac} g_{cb} = δ^a_b
        n = F.dim
        for a in range(n):
            for b in range(n):
                s = sum(m_inv[a, c] * m[c, b] for c in range(n))
                expected = 1 if a == b else 0
                assert sp.simplify(s - expected) == 0

    def test_abstract_frame_inverse_returns_opaque(self) -> None:
        """AbstractFrame inverse returns a ComponentMetricInverse with
        opaque InverseMetricEntryExpr atoms, the symbolic g^{ab}.
        """
        from jacopy.frame_calc.symbolic_atoms import InverseMetricEntryExpr

        F = AbstractFrame(dim=2)
        m = ComponentMetric(F, sp.eye(2))
        m_inv = m.inverse()
        assert isinstance(m_inv, ComponentMetricInverse)
        # Each entry is an opaque atom keyed on the metric id
        assert isinstance(m_inv[0, 0], InverseMetricEntryExpr)
        assert isinstance(m_inv[0, 1], InverseMetricEntryExpr)
        # Two distinct metrics get distinct atoms (keyed on id)
        m2 = ComponentMetric(F, sp.eye(2))
        assert m_inv[0, 0] != m2.inverse()[0, 0]

    def test_abstract_frame_det_raises(self) -> None:
        """det on AbstractFrame remains deferred, would need its own
        opaque-atom design (no polymorphic determinant on jacopy Expr)."""
        F = AbstractFrame(dim=2)
        m = ComponentMetric(F, sp.eye(2))
        with pytest.raises(NotImplementedError, match="Stage D"):
            m.det()


# --------------------------------------------------------------------- #
# ComponentMetricInverse                                                #
# --------------------------------------------------------------------- #


class TestComponentMetricInverse:
    def test_signature_is_2_0(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        m_inv = m.inverse()
        assert m_inv.signature == (2, 0)

    def test_matrix_round_trip(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        m_inv = m.inverse()
        # Build a fresh ComponentMetricInverse from same data, verify match
        m_inv2 = ComponentMetricInverse(F, m_inv.matrix())
        assert m_inv.equals(m_inv2)

    def test_inverse_returns_metric(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        m_inv = m.inverse()
        m_back = m_inv.inverse()
        assert isinstance(m_back, ComponentMetric)
        assert m_back.equals(m)


# --------------------------------------------------------------------- #
# ComponentConnection                                                   #
# --------------------------------------------------------------------- #


class TestComponentConnection:
    def test_construction_from_array(self, polar) -> None:
        F, _ = polar
        n = F.dim
        comp = sp.MutableDenseNDimArray.zeros(n, n, n)
        # Polar Christoffel: Γ^r_{θθ} = -r,  Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
        r = sp.symbols("r", positive=True)
        comp[0, 1, 1] = -r
        comp[1, 0, 1] = 1 / r
        comp[1, 1, 0] = 1 / r
        C = ComponentConnection(F, comp)
        assert C.signature == (1, 2)
        assert C.rank == 3
        assert C.shape == (2, 2, 2)
        assert C[0, 1, 1] == -r
        assert C.upper(0, 1, 1) == -r

    def test_zero_default(self, polar) -> None:
        F, _ = polar
        n = F.dim
        comp = sp.MutableDenseNDimArray.zeros(n, n, n)
        C = ComponentConnection(F, comp)
        assert C.is_zero()

    def test_nonzero_components_listing(self, polar) -> None:
        F, _ = polar
        n = F.dim
        comp = sp.MutableDenseNDimArray.zeros(n, n, n)
        r = sp.symbols("r", positive=True)
        comp[0, 1, 1] = -r
        comp[1, 0, 1] = 1 / r
        comp[1, 1, 0] = 1 / r
        C = ComponentConnection(F, comp)
        nz = C.nonzero_components()
        assert nz == {(0, 1, 1): -r, (1, 0, 1): 1 / r, (1, 1, 0): 1 / r}


# --------------------------------------------------------------------- #
# is_zero / nonzero / simplify integration                              #
# --------------------------------------------------------------------- #


class TestZeroDetection:
    def test_literal_zero_matrix_detected(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.zeros(2, 2), signature=(0, 2))
        assert T.is_zero()
        assert T.nonzero_components() == {}

    def test_trig_pythagorean_simplifies_to_zero(self) -> None:
        """`sin²θ + cos²θ - 1` should be recognised as zero via trigsimp."""
        t, theta = sp.symbols("t theta")
        F = CoordinateFrame([t, theta])
        # entry that simplifies to 0 only via trig identity
        e = sp.sin(theta) ** 2 + sp.cos(theta) ** 2 - 1
        T = ComponentTensor(F, sp.Matrix([[e, 0], [0, 0]]), signature=(0, 2))
        assert T.is_zero()

    def test_genuinely_nonzero_detected(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.Matrix([[t, 0], [0, 0]]), signature=(0, 2))
        assert not T.is_zero()
        assert T.nonzero_components() == {(0, 0): t}

    def test_simplify_returns_typed_subclass(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        m2 = m.simplify()
        assert isinstance(m2, ComponentMetric)
        assert m2.equals(m)


# --------------------------------------------------------------------- #
# Contraction                                                           #
# --------------------------------------------------------------------- #


class TestContraction:
    """Stage-C addition: ComponentTensor.contract(upper, lower)."""

    def test_trace_of_2d_identity_returns_dim(self) -> None:
        """δ^a_a = dim, trace of identity (1,1) tensor."""
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        delta = sp.MutableDenseNDimArray.zeros(2, 2)
        for i in range(2):
            delta[i, i] = 1
        T = ComponentTensor(F, delta, signature=(1, 1))
        trace = T.contract(upper=0, lower=1)
        assert trace == 2

    def test_trace_of_4d_identity(self) -> None:
        coords = list(sp.symbols("t x y z"))
        F = CoordinateFrame(coords)
        delta = sp.MutableDenseNDimArray.zeros(4, 4)
        for i in range(4):
            delta[i, i] = 1
        T = ComponentTensor(F, delta, signature=(1, 1))
        assert T.contract(upper=0, lower=1) == 4

    def test_trace_of_diagonal(self) -> None:
        """Trace = sum of diagonal entries."""
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        comp = sp.MutableDenseNDimArray.zeros(2, 2)
        comp[0, 0] = sp.Symbol("alpha")
        comp[1, 1] = sp.Symbol("beta")
        T = ComponentTensor(F, comp, signature=(1, 1))
        result = T.contract(upper=0, lower=1)
        assert sp.simplify(result - (sp.Symbol("alpha") + sp.Symbol("beta"))) == 0

    def test_signature_drops_correctly(self) -> None:
        """A (1, 3) tensor contracted gives (0, 2)."""
        t, r, theta = sp.symbols("t r theta")
        F = CoordinateFrame([t, r, theta])
        arr = sp.MutableDenseNDimArray.zeros(3, 3, 3, 3)
        T = ComponentTensor(F, arr, signature=(1, 3))
        result = T.contract(upper=0, lower=2)
        assert isinstance(result, ComponentTensor)
        assert result.signature == (0, 2)
        assert result.shape == (3, 3)

    def test_ricci_pattern_contraction(self) -> None:
        """Hand-built `R^c_{acb}` style contraction of a (1, 3) tensor.

        Construct R such that R[i, 0, i, 0] = i+1 for i = 0, 1.
        Then `R^c_{a c b}` with a = b = 0 contracts to:
            R[0, 0, 0, 0] + R[1, 0, 1, 0] = 1 + 2 = 3.
        Other free entries remain zero.
        """
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        arr = sp.MutableDenseNDimArray.zeros(2, 2, 2, 2)
        arr[0, 0, 0, 0] = 1
        arr[1, 0, 1, 0] = 2
        R = ComponentTensor(F, arr, signature=(1, 3))
        Ric = R.contract(upper=0, lower=2)
        assert Ric.signature == (0, 2)
        assert Ric[0, 0] == 3
        assert Ric[0, 1] == 0
        assert Ric[1, 0] == 0
        assert Ric[1, 1] == 0

    def test_remaining_indices_preserve_order(self) -> None:
        """For a (1, 3) tensor with index order [u, l1, l2, l3], contracting
        u with l2 should leave [l1, l3] in that order."""
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        arr = sp.MutableDenseNDimArray.zeros(2, 2, 2, 2)
        # Set R[0, 1, 0, 2] = sym for testing
        arr[0, 1, 0, 0] = sp.Symbol("X")
        arr[1, 1, 1, 0] = sp.Symbol("Y")
        T = ComponentTensor(F, arr, signature=(1, 3))
        result = T.contract(upper=0, lower=2)
        # Result[1, 0] = sum_k T[k, 1, k, 0] = X + Y
        expected = sp.Symbol("X") + sp.Symbol("Y")
        assert sp.simplify(result[1, 0] - expected) == 0

    def test_contract_runs_simplify(self) -> None:
        """Trace of a (1,1) with sin² + cos² = 1 entries simplifies."""
        theta = sp.Symbol("theta")
        F = CoordinateFrame([theta])
        comp = sp.MutableDenseNDimArray([[sp.sin(theta) ** 2 + sp.cos(theta) ** 2]])
        T = ComponentTensor(F, comp, signature=(1, 1))
        # Single (1, 1) entry; contract gives the scalar 1
        assert T.contract(upper=0, lower=1) == 1

    def test_invalid_upper_position(self) -> None:
        """Contracting upper=0 on a pure-lower tensor errors."""
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        T = ComponentTensor(F, [[1]], signature=(0, 2))
        with pytest.raises(IndexError, match="upper position 0 out of range"):
            T.contract(upper=0, lower=1)

    def test_invalid_lower_position(self) -> None:
        """Contracting lower=0 on a (1, 1) tensor errors (0 is upper, not lower)."""
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.eye(2), signature=(1, 1))
        with pytest.raises(IndexError, match="lower position 0 out of range"):
            T.contract(upper=0, lower=0)

    def test_invalid_lower_position_too_high(self) -> None:
        t, r = sp.symbols("t r")
        F = CoordinateFrame([t, r])
        T = ComponentTensor(F, sp.eye(2), signature=(1, 1))
        with pytest.raises(IndexError, match="out of range"):
            T.contract(upper=0, lower=5)


# --------------------------------------------------------------------- #
# Equality                                                              #
# --------------------------------------------------------------------- #


class TestEquality:
    def test_same_data_equal(self, polar) -> None:
        F, g = polar
        m1 = ComponentMetric(F, g)
        m2 = ComponentMetric(F, g)
        assert m1.equals(m2)

    def test_different_frames_unequal(self) -> None:
        t, r = sp.symbols("t r")
        F1 = CoordinateFrame([t, r])
        F2 = CoordinateFrame([t, r], name="other")
        m1 = ComponentMetric(F1, sp.eye(2))
        m2 = ComponentMetric(F2, sp.eye(2))
        assert not m1.equals(m2)

    def test_different_data_unequal(self, polar) -> None:
        F, g = polar
        m1 = ComponentMetric(F, g)
        # tweak [1,1] entry
        m2_data = g.copy()
        m2_data[1, 1] = sp.symbols("r", positive=True) ** 3
        m2 = ComponentMetric(F, m2_data)
        assert not m1.equals(m2)

    def test_unrelated_object_unequal(self, polar) -> None:
        F, g = polar
        m = ComponentMetric(F, g)
        assert not m.equals("not a tensor")
