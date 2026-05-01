"""Tests for the Cartan-structure equation wrapper, Faz 17.F.2."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.cartan_forms import (
    ConnectionForm,
    CurvatureForm,
    TorsionForm,
)
from jacopy.calculus.connection import connection
from jacopy.calculus.exterior_d import d
from jacopy.calculus.local_frame import FrameCovector, local_frame
from jacopy.core.expr import Sum, Symbol, Zero
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.library.cartan_structure import (
    CartanStructureProblem,
    CartanStructureProofResult,
)
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# construction / validation                                              #
# --------------------------------------------------------------------- #


def test_construct_records_connection_and_frame():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    assert P.connection is nabla
    assert P.frame is F
    assert isinstance(P.engine, ExpansionEngine)
    assert isinstance(P.registry, PropertyRegistry)


def test_construct_rejects_non_connection():
    F = local_frame("F", dim=3)
    with pytest.raises(TypeError):
        CartanStructureProblem("nabla", F)  # type: ignore[arg-type]


def test_construct_rejects_non_frame():
    nabla = connection("∇")
    with pytest.raises(TypeError):
        CartanStructureProblem(nabla, "F")  # type: ignore[arg-type]


def test_default_name_includes_connection_and_frame():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    assert "∇" in P.name and "F" in P.name


def test_custom_name_overrides_default():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F, name="custom")
    assert P.name == "custom"


def test_repr_contains_connection_and_frame():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    r = repr(P)
    assert "∇" in r and "F" in r


# --------------------------------------------------------------------- #
# registry: degree-1 declarations for FrameCovector + ConnectionForm     #
# --------------------------------------------------------------------- #


def test_registry_declares_framecovector_degree_one():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    e_a = F.coframe("a")
    from jacopy.algebra.derivation import degree_of

    assert degree_of(e_a, P.registry) == Degree.const(1)


def test_registry_declares_connectionform_degree_one():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    omega_ab = ConnectionForm(nabla, F, "a", "b")
    from jacopy.algebra.derivation import degree_of

    assert degree_of(omega_ab, P.registry) == Degree.const(1)


# --------------------------------------------------------------------- #
# engine bundle scoping                                                  #
# --------------------------------------------------------------------- #


def test_engine_does_not_unfold_other_connection_torsion_form():
    """Engine bundle is scoped to (∇, F); a second ∇' shape stays opaque."""
    nabla1 = connection("∇1")
    nabla2 = connection("∇2")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla1, F)
    other_T = MultiEval(
        TorsionForm(nabla2, F, "a"),
        Symbol("U"),
        Symbol("V"),
        alternating=True,
    )
    expanded, _ = P.engine.expand(other_T, max_steps=64)
    # Engine doesn't open the other connection's T^a head.
    assert isinstance(expanded, MultiEval)
    head = expanded.head
    assert isinstance(head, TorsionForm)
    assert head.connection is nabla2


# --------------------------------------------------------------------- #
# builders                                                               #
# --------------------------------------------------------------------- #


def test_torsion_form_builder_returns_torsion_form():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    t = P.torsion_form("a")
    assert isinstance(t, TorsionForm)
    assert t.connection is nabla
    assert t.frame is F


def test_connection_form_builder_returns_connection_form():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    w = P.connection_form("a", "b")
    assert isinstance(w, ConnectionForm)


def test_coframe_builder_returns_frame_covector():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    e_a = P.coframe("a")
    assert isinstance(e_a, FrameCovector)
    assert e_a.frame is F


def test_first_cartan_lhs_is_alternating_arity_two_multi_eval():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    lhs = P.first_cartan_lhs(U, V, "a")
    assert isinstance(lhs, MultiEval)
    assert lhs.arity == 2
    assert lhs.alternating
    assert isinstance(lhs.head, TorsionForm)


def test_first_cartan_rhs_is_sum_of_d_term_and_wedge_term():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    rhs = P.first_cartan_rhs(U, V, "a")
    assert isinstance(rhs, Sum)
    # First child: (de^a)(U,V), MultiEval over Act(d, e^a).
    d_term, wedge_term = rhs.children
    assert isinstance(d_term, MultiEval)
    assert isinstance(d_term.head, Act)
    assert d_term.head.op is d
    assert isinstance(d_term.head.arg, FrameCovector)
    # Second child: (Σ_b ω^a_b ∧ e^b)(U,V).
    assert isinstance(wedge_term, MultiEval)
    is_node = wedge_term.head
    assert isinstance(is_node, IndexedSum)
    assert isinstance(is_node.body, Wedge)


def test_first_cartan_rhs_uses_alpha_fresh_bound_b():
    """Each call mints a fresh bound dummy ``b̂``; ``upper_a`` stays free."""
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    rhs = P.first_cartan_rhs(U, V, "a")
    is_node = rhs.children[1].head
    assert is_node.dummy.is_bound
    assert is_node.dummy.name == "b"


# --------------------------------------------------------------------- #
# Cartan I closure                                                       #
# --------------------------------------------------------------------- #


def test_prove_first_cartan_closes_with_derivation_args():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_first_cartan(U, V, "a")
    assert isinstance(result, CartanStructureProofResult)
    assert result.ok
    assert result.final == Zero


def test_prove_first_cartan_records_lhs_and_rhs():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_first_cartan(U, V, "a")
    assert result.lhs_initial != result.rhs_initial
    # Engine produces a non-empty step transcript before reducing.
    assert len(result.steps) > 0


def test_prove_first_cartan_closes_for_alternative_index_name():
    """The proof works for any free upper index; ``a`` is just a default."""
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_first_cartan(U, V, "k")
    assert result.ok


def test_prove_first_cartan_closes_for_alternative_frame_dim():
    """Frame dim is symbolic / informational, proof closes regardless."""
    nabla = connection("∇")
    F = local_frame("F", dim=5)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_first_cartan(U, V, "a")
    assert result.ok


# --------------------------------------------------------------------- #
# proof-result shape                                                     #
# --------------------------------------------------------------------- #


def test_cartan_structure_proof_result_is_frozen():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_first_cartan(U, V, "a")
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]


# --------------------------------------------------------------------- #
# scoping: two connections / frames don't cross-fire                     #
# --------------------------------------------------------------------- #


def test_two_problems_have_independent_registries():
    nabla = connection("∇")
    F1 = local_frame("F1", dim=3)
    F2 = local_frame("F2", dim=3)
    P1 = CartanStructureProblem(nabla, F1)
    P2 = CartanStructureProblem(nabla, F2)
    assert P1.registry is not P2.registry


# --------------------------------------------------------------------- #
# Cartan II builders                                                     #
# --------------------------------------------------------------------- #


def test_curvature_form_builder_returns_curvature_form():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    R = P.curvature_form("a", "b")
    assert isinstance(R, CurvatureForm)
    assert R.connection is nabla
    assert R.frame is F


def test_second_cartan_lhs_is_alternating_arity_two_multi_eval():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    lhs = P.second_cartan_lhs(U, V, "a", "b")
    assert isinstance(lhs, MultiEval)
    assert lhs.arity == 2
    assert lhs.alternating
    assert isinstance(lhs.head, CurvatureForm)


def test_second_cartan_rhs_is_sum_of_d_term_and_wedge_term():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    rhs = P.second_cartan_rhs(U, V, "a", "b")
    assert isinstance(rhs, Sum)
    d_term, wedge_term = rhs.children
    # First child: (dω^a_b)(U, V).
    assert isinstance(d_term, MultiEval)
    assert isinstance(d_term.head, Act)
    assert d_term.head.op is d
    assert isinstance(d_term.head.arg, ConnectionForm)
    # Second child: (Σ_c ω^a_c ∧ ω^c_b)(U, V).
    assert isinstance(wedge_term, MultiEval)
    is_node = wedge_term.head
    assert isinstance(is_node, IndexedSum)
    assert isinstance(is_node.body, Wedge)


def test_second_cartan_rhs_uses_alpha_fresh_bound_c():
    """Each call mints a fresh bound dummy ``ĉ``; ``a``/``b`` stay free."""
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    rhs = P.second_cartan_rhs(U, V, "a", "b")
    is_node = rhs.children[1].head
    assert is_node.dummy.is_bound
    assert is_node.dummy.name == "c"


# --------------------------------------------------------------------- #
# Cartan II closure                                                      #
# --------------------------------------------------------------------- #


def test_prove_second_cartan_closes_with_derivation_args():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_second_cartan(U, V, "a", "b")
    assert isinstance(result, CartanStructureProofResult)
    assert result.ok
    assert result.final == Zero


def test_prove_second_cartan_records_lhs_and_rhs():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_second_cartan(U, V, "a", "b")
    assert result.lhs_initial != result.rhs_initial
    assert len(result.steps) > 0


def test_prove_second_cartan_closes_for_alternative_indices():
    """The proof works for any free upper / lower index pair."""
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_second_cartan(U, V, "k", "m")
    assert result.ok


def test_prove_second_cartan_closes_for_alternative_frame_dim():
    """Frame dim is symbolic / informational, proof closes regardless."""
    nabla = connection("∇")
    F = local_frame("F", dim=5)
    P = CartanStructureProblem(nabla, F)
    U, V = Derivation("U", 0), Derivation("V", 0)
    result = P.prove_second_cartan(U, V, "a", "b")
    assert result.ok


# --------------------------------------------------------------------- #
# Q9 Stage 9.E, Koszul-connection mode                                  #
# --------------------------------------------------------------------- #


class TestKoszulConnectionCartanStructure:
    """Cartan I/II closes for an affine connection equipped with a
    Koszul bracket and an anchor-pulled function action, the Q9 setting
    of a Poisson-induced ``∇̃`` on T*M.

    :class:`CartanStructureProblem` swaps in
    :class:`KoszulExteriorDIntrinsicDefinition` whenever the connection
    carries a bracket; that swap is what lines up the LHS's
    ``ρ(α)(…)`` shapes (sourced from the Y-Leibniz rule, which routes
    its function action through ``connection.function_action``) against
    the RHS's intrinsic d̃ expansion, and the LHS's ``[α, β]_K`` against
    the matching RHS bracket. With no other structural changes both
    Cartan equations close on the same engine machinery used for the
    standard smooth-manifold case.
    """

    def _setup(self):
        from jacopy.calculus.anchor import Anchor
        from jacopy.calculus.connection import koszul_connection
        from jacopy.brackets.koszul import KoszulBracket

        anchor = Anchor(name="ρ")
        bracket = KoszulBracket(anchor)
        nabla = koszul_connection("∇̃", anchor=anchor, bracket=bracket)
        F = local_frame("F̃", dim=3)
        return CartanStructureProblem(nabla, F)

    def test_first_cartan_closes(self):
        P = self._setup()
        U, V = Derivation("U", 0), Derivation("V", 0)
        result = P.prove_first_cartan(U, V, "a")
        assert result.ok
        assert result.final == Zero

    def test_second_cartan_closes(self):
        P = self._setup()
        U, V = Derivation("U", 0), Derivation("V", 0)
        result = P.prove_second_cartan(U, V, "a", "b")
        assert result.ok
        assert result.final == Zero

    def test_koszul_engine_picks_anchored_intrinsic_d_rule(self):
        from jacopy.calculus.intrinsic_axioms import (
            ExteriorDIntrinsicDefinition,
            KoszulExteriorDIntrinsicDefinition,
        )

        P = self._setup()
        rules = list(P.engine.definitions)
        assert any(
            isinstance(r, KoszulExteriorDIntrinsicDefinition) for r in rules
        )
        assert not any(
            isinstance(r, ExteriorDIntrinsicDefinition) for r in rules
        )

    def test_default_engine_picks_standard_intrinsic_d_rule(self):
        from jacopy.calculus.intrinsic_axioms import (
            ExteriorDIntrinsicDefinition,
            KoszulExteriorDIntrinsicDefinition,
        )

        nabla = connection("∇")
        F = local_frame("F", dim=3)
        P = CartanStructureProblem(nabla, F)
        rules = list(P.engine.definitions)
        assert any(
            isinstance(r, ExteriorDIntrinsicDefinition) for r in rules
        )
        assert not any(
            isinstance(r, KoszulExteriorDIntrinsicDefinition) for r in rules
        )
