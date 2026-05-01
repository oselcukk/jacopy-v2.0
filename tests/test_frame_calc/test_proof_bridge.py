"""Tests for `jacopy.frame_calc.proof_bridge` (Faz 18 Stage G)."""

import pytest
import sympy as sp

from jacopy.core.expr import Expr
from jacopy.frame_calc import (
    CoordinateFrame,
    ComponentMetric,
    SymPyAtom,
    curvature,
    levi_civita,
    ricci,
    steps_to_proof_chain,
)
from jacopy.frame_calc.library import minkowski, schwarzschild
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


@pytest.fixture
def polar_lc():
    r, theta = sp.symbols("r theta", positive=True)
    F = CoordinateFrame([r, theta])
    g = ComponentMetric(F, sp.Matrix([[1, 0], [0, r**2]]))
    return levi_civita(g)


# --------------------------------------------------------------------- #
# SymPyAtom basic behaviour                                             #
# --------------------------------------------------------------------- #


class TestSymPyAtom:
    def test_construction_from_sympy_expr(self) -> None:
        r = sp.Symbol("r")
        atom = SymPyAtom(r**2 + 1)
        assert isinstance(atom, Expr)
        assert atom.sympy == r**2 + 1

    def test_construction_from_int(self) -> None:
        atom = SymPyAtom(5)
        # Should sympify to a SymPy Integer
        assert atom.sympy == 5
        assert isinstance(atom.sympy, sp.Integer)

    def test_construction_from_zero(self) -> None:
        atom = SymPyAtom(sp.S.Zero)
        assert atom.sympy == 0

    def test_repr_uses_sympy_str(self) -> None:
        r = sp.Symbol("r")
        atom = SymPyAtom(r**2)
        assert "r**2" in str(atom) or "r^{2}" in str(atom)

    def test_equality_by_sympy_form(self) -> None:
        r = sp.Symbol("r")
        a = SymPyAtom(r**2)
        b = SymPyAtom(r**2)
        c = SymPyAtom(r**3)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c

    def test_is_atom(self) -> None:
        atom = SymPyAtom(sp.Symbol("x"))
        assert atom.is_atom


# --------------------------------------------------------------------- #
# steps_to_proof_chain                                                  #
# --------------------------------------------------------------------- #


class TestStepsToProofChain:
    def test_polar_lc_derivation_chain_works(self, polar_lc) -> None:
        chain = polar_lc.derivation_chain(0, 1, 1)
        assert isinstance(chain, ProofChain)
        # Should have 5 steps for coordinate frame:
        # Koszul, frame-deriv, γ (skipped narration), contract, simplify
        assert len(chain.steps) == 5

    def test_all_steps_tagged_computation(self, polar_lc) -> None:
        chain = polar_lc.derivation_chain(0, 1, 1)
        for step in chain.steps:
            assert step.provenance_tag == "computation"

    def test_first_step_carries_head_label(self, polar_lc) -> None:
        chain = polar_lc.derivation_chain(0, 1, 1)
        first = chain.steps[0]
        assert "via Koszul formula" in first.justification

    def test_final_after_matches_christoffel_value(self, polar_lc) -> None:
        chain = polar_lc.derivation_chain(0, 1, 1)
        final_after = chain.steps[-1].after
        # Should be SymPyAtom wrapping -r
        assert isinstance(final_after, SymPyAtom)
        r = sp.Symbol("r", positive=True)
        assert sp.simplify(final_after.sympy - (-r)) == 0

    def test_optimized_mode_raises(self) -> None:
        F, g = schwarzschild()
        LC = levi_civita(g, optimized=True)
        with pytest.raises(RuntimeError, match="optimized"):
            LC.derivation_chain(0, 0, 1)

    def test_curvature_chain(self, polar_lc) -> None:
        R = curvature(polar_lc)
        chain = R.derivation_chain(0, 0, 1, 0)  # canonical b<c entry
        assert isinstance(chain, ProofChain)
        assert all(s.provenance_tag == "computation" for s in chain.steps)

    def test_ricci_chain(self) -> None:
        F, g = minkowski()
        Ric = ricci(levi_civita(g))
        chain = Ric.derivation_chain(0, 0)
        assert isinstance(chain, ProofChain)
        assert all(s.provenance_tag == "computation" for s in chain.steps)

    def test_empty_steps_yields_empty_chain(self) -> None:
        chain = steps_to_proof_chain([])
        assert isinstance(chain, ProofChain)
        assert len(chain.steps) == 0

    def test_head_label_prepended(self) -> None:
        from jacopy.frame_calc.levi_civita import KoszulStep
        steps = [
            KoszulStep(rule="Test", description="something", expression=sp.S.One),
        ]
        chain = steps_to_proof_chain(steps, head_label="HeadLabel")
        assert "HeadLabel" in chain.steps[0].justification


# --------------------------------------------------------------------- #
# LaTeX rendering, ProofChain → LaTeX with SymPyAtom                   #
# --------------------------------------------------------------------- #


class TestLatexRendering:
    def test_chain_to_latex_runs(self, polar_lc) -> None:
        from jacopy.display import chain_to_latex

        chain = polar_lc.derivation_chain(0, 1, 1)
        text = chain_to_latex(chain)
        assert isinstance(text, str)
        assert "gather*" in text
        assert "computation" in text  # provenance tag rendered

    def test_sympy_atom_renders_via_sympy_latex(self) -> None:
        from jacopy.display import to_latex

        r = sp.Symbol("r")
        atom = SymPyAtom(r**2 + 1)
        latex = to_latex(atom)
        assert "r" in latex
        # SymPy's latex output includes ^{2} for r**2
        assert "^{2}" in latex or "r^2" in latex

    def test_sympy_atom_special_forms(self) -> None:
        """Verify that SymPyAtom renders SymPy's clean LaTeX forms
        for fractions, Greek letters, etc."""
        from jacopy.display import to_latex

        M, r = sp.symbols("M r", positive=True)
        # Fraction: M/(r²(r-2M))
        atom = SymPyAtom(M / (r**2 * (r - 2*M)))
        latex = to_latex(atom)
        # SymPy renders this as a \frac
        assert "\\frac" in latex

    def test_chain_renders_christoffel_value(self, polar_lc) -> None:
        from jacopy.display import chain_to_latex

        chain = polar_lc.derivation_chain(0, 1, 1)
        text = chain_to_latex(chain)
        # The final value -r should appear
        assert "- r" in text or "-r" in text


# --------------------------------------------------------------------- #
# ProofStep "computation" tag is now valid                              #
# --------------------------------------------------------------------- #


class TestComputationProvenanceTag:
    """Stage G adds 'computation' as a valid tag to ProofStep."""

    def test_tag_in_valid_set(self) -> None:
        assert "computation" in ProofStep._VALID_TAGS

    def test_construct_step_with_tag(self) -> None:
        from jacopy.core.expr import Integer

        step = ProofStep(
            before=Integer(0),
            after=Integer(1),
            rule="test",
            provenance_tag="computation",
        )
        assert step.provenance_tag == "computation"

    def test_existing_tags_still_valid(self) -> None:
        assert None in ProofStep._VALID_TAGS
        assert "axiom" in ProofStep._VALID_TAGS
        assert "theorem" in ProofStep._VALID_TAGS

    def test_invalid_tag_still_rejected(self) -> None:
        from jacopy.core.expr import Integer

        with pytest.raises(ValueError, match="provenance_tag"):
            ProofStep(
                before=Integer(0),
                after=Integer(0),
                rule="test",
                provenance_tag="bogus",
            )
