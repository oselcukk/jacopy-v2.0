"""Tests for jacopy.core.properties."""

import pytest

from jacopy.core.properties import (
    AntiCommuting,
    Antisymmetric,
    Graded,
    GradedAntisymmetric,
    GradedCommutative,
    NonCommuting,
    ProofRef,
    Property,
    Provenance,
    Scalar,
    Symmetric,
)
from jacopy.core.symbolic_degree import Degree


class TestProvenance:
    def test_enum_members(self):
        assert Provenance.AXIOM.value == "axiom"
        assert Provenance.DERIVED.value == "derived"


class TestProofRef:
    def test_immutable_and_hashable(self):
        p = ProofRef("leibniz", ("derivation:d",))
        assert p.rule == "leibniz"
        assert p.sources == ("derivation:d",)
        # frozen dataclass -> hashable
        hash(p)
        with pytest.raises(Exception):
            p.rule = "other"  # type: ignore[misc]

    def test_default_sources_empty(self):
        p = ProofRef("axiom-install")
        assert p.sources == ()

    def test_equality(self):
        assert ProofRef("r", ("a",)) == ProofRef("r", ("a",))
        assert ProofRef("r", ()) != ProofRef("r", ("a",))


class TestProperty:
    def test_axiom_default(self):
        s = Scalar()
        assert s.provenance is Provenance.AXIOM
        assert s.proof is None
        assert s.is_axiom
        assert not s.is_derived

    def test_derived_requires_proof(self):
        with pytest.raises(ValueError):
            Scalar(provenance=Provenance.DERIVED)

    def test_axiom_cannot_carry_proof(self):
        with pytest.raises(ValueError):
            Scalar(proof=ProofRef("foo"))

    def test_derived_with_proof_ok(self):
        s = Scalar(provenance=Provenance.DERIVED, proof=ProofRef("r"))
        assert s.is_derived
        assert s.proof.rule == "r"

    def test_frozen(self):
        s = Scalar()
        with pytest.raises(Exception):
            s.provenance = Provenance.DERIVED  # type: ignore[misc]


class TestGraded:
    def test_carries_int_degree(self):
        g = Graded(degree=2)
        # Int is coerced to Degree but equality to int still works.
        assert g.degree == 2
        assert isinstance(g.degree, Degree)

    def test_default_degree_zero(self):
        assert Graded().degree == 0

    def test_accepts_symbolic_degree(self):
        p = Degree.var("p")
        g = Graded(degree=p)
        assert g.degree == p
        assert g.degree.as_int() is None

    def test_equality_includes_degree(self):
        assert Graded(degree=1) == Graded(degree=1)
        assert Graded(degree=1) != Graded(degree=2)

    def test_int_and_degree_const_equal(self):
        """Graded(degree=2) and Graded(degree=Degree.const(2)) match."""
        assert Graded(degree=2) == Graded(degree=Degree.const(2))

    def test_equality_separates_provenance(self):
        axiom = Graded(degree=1)
        derived = Graded(
            degree=1,
            provenance=Provenance.DERIVED,
            proof=ProofRef("r"),
        )
        assert axiom != derived

    def test_hashable(self):
        s = {Graded(degree=0), Graded(degree=1), Graded(degree=0)}
        assert len(s) == 2

    def test_rejects_invalid_degree_type(self):
        with pytest.raises(TypeError):
            Graded(degree="two")  # type: ignore[arg-type]


class TestSymmetryProperties:
    def test_three_distinct_types(self):
        assert Symmetric() != Antisymmetric()
        assert Antisymmetric() != GradedAntisymmetric()
        assert Symmetric() != GradedAntisymmetric()

    def test_same_type_equal(self):
        assert Symmetric() == Symmetric()
        assert Antisymmetric() == Antisymmetric()
        assert GradedAntisymmetric() == GradedAntisymmetric()


class TestPropertyHierarchy:
    def test_concrete_subclasses_are_property(self):
        for cls in [
            Scalar,
            Graded,
            Symmetric,
            Antisymmetric,
            GradedAntisymmetric,
            NonCommuting,
            AntiCommuting,
            GradedCommutative,
        ]:
            assert issubclass(cls, Property)


class TestCommutativityMarkers:
    def test_three_distinct_types(self):
        assert NonCommuting() != AntiCommuting()
        assert AntiCommuting() != GradedCommutative()
        assert NonCommuting() != GradedCommutative()

    def test_same_type_equal(self):
        assert NonCommuting() == NonCommuting()
        assert AntiCommuting() == AntiCommuting()
        assert GradedCommutative() == GradedCommutative()

    def test_carry_provenance(self):
        p = GradedCommutative(
            provenance=Provenance.DERIVED, proof=ProofRef("r")
        )
        assert p.is_derived
        assert p.proof.rule == "r"

    def test_axiom_cannot_carry_proof(self):
        with pytest.raises(ValueError):
            NonCommuting(proof=ProofRef("r"))

    def test_hashable(self):
        s = {
            NonCommuting(),
            AntiCommuting(),
            GradedCommutative(),
            NonCommuting(),
        }
        assert len(s) == 3
