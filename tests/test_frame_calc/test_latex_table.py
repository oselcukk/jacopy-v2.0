"""Tests for `jacopy.frame_calc.latex_table.to_latex_table` (Faz 19 Chunk C.5)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    levi_civita,
    ricci,
    to_latex_table,
)
from jacopy.frame_calc.library import minkowski, schwarzschild


class TestLatexTableBasics:
    def test_align_block_wrapper(self) -> None:
        F, g = minkowski()
        out = to_latex_table(g)
        assert r"\begin{align*}" in out
        assert r"\end{align*}" in out

    def test_full_document(self) -> None:
        F, g = minkowski()
        out = to_latex_table(g, full_document=True)
        assert r"\documentclass{article}" in out
        assert r"\begin{document}" in out
        assert r"\end{document}" in out

    def test_title_in_full_document(self) -> None:
        F, g = minkowski()
        out = to_latex_table(g, full_document=True, title="My Metric")
        assert r"\section*{My Metric}" in out

    def test_zero_tensor_message(self) -> None:
        """All-zero tensor produces a clear placeholder."""
        F, g = schwarzschild()
        Ric = ricci(levi_civita(g))  # vacuum → zero
        out = to_latex_table(Ric)
        assert "all components zero" in out


class TestLatexTableSchwarzschild:
    def test_christoffel_listing(self) -> None:
        """Schwarzschild Christoffel rendering, verify a known entry."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        out = to_latex_table(LC)
        # Γ^t_{tr}, Γ^r_{tt} are canonical Schwarzschild entries.
        assert r"\Gamma^{t}{}_{tr}" in out
        assert r"\Gamma^{r}{}_{tt}" in out

    def test_symmetry_pruning(self) -> None:
        """Symmetric pair (b, c) in Christoffel: Γ^a_{bc} listed once."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        out = to_latex_table(LC)
        # Γ^t_{tr} is canonical (t ≤ r); Γ^t_{rt} should not appear.
        assert r"\Gamma^{t}{}_{tr}" in out
        assert r"\Gamma^{t}{}_{rt}" not in out

    def test_metric_diagonal_only(self) -> None:
        """Schwarzschild metric is diagonal: only 4 entries."""
        F, g = schwarzschild()
        out = to_latex_table(g)
        # Count `&=` (one per row)
        assert out.count("&=") == 4

    def test_custom_symbol(self) -> None:
        F, g = schwarzschild()
        Ric = ricci(levi_civita(g))
        # Use a custom symbol for an example
        out = to_latex_table(Ric, symbol="\\mathcal{R}")
        # Vacuum so empty, but the symbol won't appear; just verify
        # the call accepts the arg without error.
        assert isinstance(out, str)


class TestLatexTableGreekIndices:
    def test_theta_phi_become_latex(self) -> None:
        """Frame index names like `theta`, `phi` get backslashed."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        out = to_latex_table(LC)
        # Γ^θ_{rθ} in our index naming
        assert r"\theta" in out
        assert r"\phi" in out
