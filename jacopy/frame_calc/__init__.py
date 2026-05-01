"""
jacopy.frame_calc, frame-component differential geometry computations.

A separate submodule from the rest of jacopy: given a frame ``F`` and
a metric ``g`` (and optionally a connection ``∇``), compute Christoffel
symbols, torsion, curvature, Ricci tensor, Ricci scalar, and the
Einstein tensor, with step-by-step derivation transcripts that bridge
to :class:`~jacopy.proof.chain.ProofChain` for paper-ready LaTeX
output.

The module supports three frame types, :class:`CoordinateFrame`,
:class:`AbstractFrame`, :class:`Tetrad`, through a single
:class:`Frame` protocol; higher-level computations
(``levi_civita``, ``curvature``, ``ricci``, ``einstein_tensor``) are
frame-agnostic.

This module **requires SymPy**. Install via::

    pip install "jacopy[components]"

See :doc:`/tutorials/25_frame_calc` for an end-to-end walkthrough.
"""

from __future__ import annotations

try:  # noqa: SIM105, explicit ImportError message is the goal
    import sympy as _sp  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy. Install via "
        "`pip install \"jacopy[components]\"` to enable component-level "
        "differential geometry calculations."
    ) from exc

from jacopy.frame_calc.component_tensor import (
    ComponentConnection,
    ComponentMetric,
    ComponentMetricInverse,
    ComponentTensor,
)
from jacopy.frame_calc.curvature import (
    CurvatureStep,
    CurvatureTensor,
    curvature,
)
from jacopy.frame_calc.einstein import (
    EinsteinTensor,
    einstein_from_ricci,
    einstein_tensor,
)
from jacopy.frame_calc.analyze import analyze_metric
from jacopy.frame_calc.coord_transform import transform_metric
from jacopy.frame_calc.custom_connections import (
    connection_with_torsion,
    projective_connection,
    weyl_connection,
)
from jacopy.frame_calc.latex_table import to_latex_table
from jacopy.frame_calc.invariants import (
    cotton,
    kretschmann,
    ricci_squared,
)
from jacopy.frame_calc.frame import (
    AbstractFrame,
    CoordinateFrame,
    Frame,
    Tetrad,
)
from jacopy.frame_calc.levi_civita import (
    KoszulStep,
    LeviCivitaConnection,
    levi_civita,
)
from jacopy.frame_calc.ricci import (
    RicciStep,
    RicciTensor,
    ricci,
    ricci_from_curvature,
    ricci_scalar,
    ricci_scalar_from_ricci,
)
from jacopy.frame_calc.proof_bridge import steps_to_proof_chain
from jacopy.frame_calc.symbolic_atoms import (
    FrameDerivativeExpr,
    GammaExpr,
    SymPyAtom,
)
from jacopy.frame_calc.torsion import TorsionTensor, torsion

__all__ = [
    # Frames
    "Frame",
    "CoordinateFrame",
    "AbstractFrame",
    "Tetrad",
    # Symbolic atoms (abstract-mode opaques + ProofChain bridge)
    "FrameDerivativeExpr",
    "GammaExpr",
    "SymPyAtom",
    "steps_to_proof_chain",
    # Component tensors
    "ComponentTensor",
    "ComponentMetric",
    "ComponentMetricInverse",
    "ComponentConnection",
    # Levi-Civita
    "levi_civita",
    "LeviCivitaConnection",
    "KoszulStep",
    # Torsion
    "torsion",
    "TorsionTensor",
    # Curvature
    "curvature",
    "CurvatureTensor",
    "CurvatureStep",
    # Ricci
    "ricci",
    "ricci_from_curvature",
    "ricci_scalar",
    "ricci_scalar_from_ricci",
    "RicciTensor",
    "RicciStep",
    # Einstein
    "einstein_tensor",
    "einstein_from_ricci",
    "EinsteinTensor",
    # Curvature invariants
    "kretschmann",
    "ricci_squared",
    "cotton",
    # One-shot helper
    "analyze_metric",
    # LaTeX table render
    "to_latex_table",
    # Custom-connection helpers
    "connection_with_torsion",
    "weyl_connection",
    "projective_connection",
    # Coordinate transformation
    "transform_metric",
]
