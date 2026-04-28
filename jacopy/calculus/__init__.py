"""Calculus layer: Cartan operators on the exterior algebra."""

from jacopy.calculus.anchor import (
    Anchor,
    AnchoredVectorField,
    bracket_compatibility_obstruction,
)
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.exterior_d import (
    ExteriorDerivative,
    apply_d_squared_zero,
    d,
)
from jacopy.calculus.interior import (
    InteriorProduct,
    apply_iota_axioms,
    apply_iota_squared_zero,
    interior,
)
from jacopy.calculus.lie_derivative import (
    DEFINITIONS,
    LieDerivative,
    cartan_expansion,
    cartan_obstruction,
    lie_derivative,
)
from jacopy.calculus.cartan import CartanCalculus, RELATIONS
from jacopy.calculus.cartan_remainder import CartanRemainder, K
from jacopy.calculus.cartan_remainder_axioms import (
    CartanRemainderDefinition,
    TildeCartanRemainderDefinition,
)
from jacopy.calculus.connection import (
    AffineConnection,
    ConnectionEvalExpr,
    ConnectionXLinearityDefinition,
    ConnectionXScalarPullDefinition,
    ConnectionYAdditivityDefinition,
    ConnectionYLeibnizDefinition,
    connection,
    koszul_connection,
)
from jacopy.calculus.local_frame import (
    FrameCovector,
    FrameIndex,
    FramePairingDualityDefinition,
    FrameVectorField,
    KroneckerDelta,
    LocalFrame,
    local_frame,
)
from jacopy.calculus.metric import (
    MetricEvalExpr,
    MetricEvalLinearityDefinition,
    MetricEvalScalarPullDefinition,
    MetricEvalSymmetryDefinition,
    MetricTensor,
    metric,
)
from jacopy.calculus.non_metricity import (
    NonMetricityCompatibilityDefinition,
    NonMetricityEvalExpr,
    NonMetricityVLinearityDefinition,
    NonMetricityVScalarPullDefinition,
    NonMetricityXYSymmetryDefinition,
)
from jacopy.calculus.cartan_forms import (
    ConnectionForm,
    ConnectionFormDefinition,
    CurvatureForm,
    CurvatureFormDefinition,
    NonMetricityForm,
    NonMetricityFormDefinition,
    TorsionForm,
    TorsionFormDefinition,
)
from jacopy.calculus.indexed_sum_axioms import (
    ConnectionEvalIndexedSumPushInDefinition,
    IndexedSumKroneckerContractDefinition,
    IndexedSumNegPullDefinition,
    IndexedSumPairingPushInLeftDefinition,
    IndexedSumPairingPushInRightDefinition,
    IndexedSumScalarPullDefinition,
    IndexedSumSumDistributeDefinition,
    MultiEvalIndexedSumPushInDefinition,
)
from jacopy.calculus.frame_decomposition import (
    ConnectionEvalYFrameDecompositionDefinition,
    ConnectionFormDecompositionDefinition,
    FrameDecompositionDefinition,
)
from jacopy.calculus.torsion_curvature import (
    Curvature,
    CurvatureCovariantDerivative,
    CurvatureCovariantDerivativeDefinition,
    CurvatureDefinitionDefinition,
    CurvatureXLinearityDefinition,
    CurvatureXScalarPullDefinition,
    CurvatureXYAntiSymmetryDefinition,
    CurvatureYLinearityDefinition,
    CurvatureYScalarPullDefinition,
    Torsion,
    TorsionAntiSymmetryDefinition,
    TorsionCovariantDerivative,
    TorsionCovariantDerivativeDefinition,
    TorsionDefinitionDefinition,
    TorsionXLinearityDefinition,
    TorsionXScalarPullDefinition,
    TorsionYLinearityDefinition,
    TorsionYScalarPullDefinition,
)
from jacopy.calculus.derivator import derivator
from jacopy.calculus.hamiltonian_vf import (
    HamiltonianDefiningRelationDefinition,
    HamiltonianVectorField,
    HamiltonianVfDerivedDefinition,
    equivalence_condition,
    hamiltonian_vf,
    register_hamiltonian_defining_relation,
)
from jacopy.calculus.musical import (
    ArgNegLinearityDefinition,
    Flat,
    IotaFlatDefinition,
    MusicalCompatibility,
    MusicalCompatibilityBilinearDefinition,
    MusicalCompatibilityDefinition,
    Sharp,
    flat,
    sharp,
)
from jacopy.calculus.operator_equation import OperatorEquation
from jacopy.calculus.pairing import Pairing, pairing
from jacopy.calculus.pairing_axioms import (
    MultiEvalOneFormPairingBridgeDefinition,
    PairingLieLeibnizDefinition,
    PairingLinearityDefinition,
)
from jacopy.calculus.wedge_axioms import (
    WedgeMultiEvalAlternatingDefinition,
)
from jacopy.calculus.sharp_axioms import (
    SharpLinearityDefinition,
    SharpOnExactDefinition,
)
from jacopy.calculus.vf_axioms import (
    LieVfJacobiDefinition,
    OpCommutatorVfDefinition,
)
from jacopy.calculus.sn_axiom import SnBivectorFormulaDefinition
from jacopy.calculus.poisson_axioms import (
    HamiltonianCyclicSnFormulaDefinition,
    PoissonAsHamiltonianDefinition,
)
from jacopy.calculus.linearity_axioms import (
    ExteriorDerivativeLinearityDefinition,
    LieDerivativeArgLinearityDefinition,
)
from jacopy.calculus.multi_eval_axioms import (
    MultiEvalAlternatingNormalDefinition,
    MultiEvalArgLinearityDefinition,
    MultiEvalHeadLinearityDefinition,
    MultiEvalRepeatArgZeroDefinition,
)
from jacopy.calculus.intrinsic_axioms import (
    ExteriorDIntrinsicDefinition,
    InteriorProductIntrinsicDefinition,
    LieDerivativeIntrinsicDefinition,
)
from jacopy.calculus.intrinsic_engine import (
    IntrinsicFormulaMatch,
    IntrinsicFormulaRecognizer,
    intrinsic_engine,
    intrinsic_engine_with_closure,
    prove_intrinsic_equivalence,
)
from jacopy.calculus.closure_axioms import (
    IotaActAsScalarDefinition,
    LieBracketVfAntiSymmetryDefinition,
    LieBracketVfJacobiDefinition,
    VfActCommutatorDefinition,
)
from jacopy.calculus.closed_axioms import ClosedFormDefinition
from jacopy.calculus.nondegenerate_axioms import (
    NonDegenerateInteriorEqualityDefinition,
)
from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition
from jacopy.calculus.lie_rescaling_axioms import LieRescalingDefinition
from jacopy.calculus.pairing_linearity_axioms import (
    PairingScalarPullDefinition,
)
from jacopy.calculus.multi_eval_scalar_axioms import (
    MultiEvalScalarPullDefinition,
)
from jacopy.calculus.tilde import (
    K_tilde,
    TildeCartanRemainder,
    TildeDIntrinsicDefinition,
    TildeDOfFunctionDefinition,
    TildeDSquaredPoissonDefinition,
    TildeExteriorDerivative,
    TildeExteriorDLichnerowiczDefinition,
    TildeInteriorProduct,
    TildeIntrinsicFormulaMatch,
    TildeIntrinsicFormulaRecognizer,
    TildeIotaActAsScalarDefinition,
    TildeIotaIntrinsicDefinition,
    TildeIotaOnZeroVectorDefinition,
    TildeIotaSquaredZeroDefinition,
    TildeIotaSwapDefinition,
    TildeLieDerivative,
    TildeLieIntrinsicDefinition,
    TildeLieMagicDefinition,
    TildeLieOnZeroVectorDefinition,
    prove_tilde_cartan_relation,
    tilde_d,
    tilde_interior,
    tilde_intrinsic_engine,
    tilde_lie,
)

__all__ = [
    # exterior derivative
    "ExteriorDerivative",
    "d",
    "apply_d_squared_zero",
    # interior product
    "InteriorProduct",
    "interior",
    "apply_iota_axioms",
    "apply_iota_squared_zero",
    # Lie derivative
    "LieDerivative",
    "lie_derivative",
    "cartan_expansion",
    "cartan_obstruction",
    "DEFINITIONS",
    # anchor
    "Anchor",
    "AnchoredVectorField",
    "bracket_compatibility_obstruction",
    # exterior algebra skeleton
    "ExteriorAlgebra",
    # operator equation
    "OperatorEquation",
    # Cartan calculus framework
    "CartanCalculus",
    "RELATIONS",
    # Invariant formula for d on 1-forms
    "invariant_d_one_form",
    "InvariantDOneFormDefinition",
    "INVARIANT_D_CLASSIFICATIONS",
    # Hamiltonian vector field
    "HamiltonianVectorField",
    "HamiltonianVfDerivedDefinition",
    "HamiltonianDefiningRelationDefinition",
    "hamiltonian_vf",
    "equivalence_condition",
    "register_hamiltonian_defining_relation",
    # Pairing
    "Pairing",
    "pairing",
    # Musical isomorphisms
    "Flat",
    "flat",
    "Sharp",
    "sharp",
    "MusicalCompatibility",
    "MusicalCompatibilityDefinition",
    "MusicalCompatibilityBilinearDefinition",
    "IotaFlatDefinition",
    "ArgNegLinearityDefinition",
    # Sharp axioms (Faz 13.A)
    "SharpLinearityDefinition",
    "SharpOnExactDefinition",
    # Pairing axioms (Faz 13.B)
    "PairingLinearityDefinition",
    "PairingLieLeibnizDefinition",
    # Wedge alternating expansion + MultiEval→Pairing bridge (Faz 17.F.1.5/6)
    "WedgeMultiEvalAlternatingDefinition",
    "MultiEvalOneFormPairingBridgeDefinition",
    # Vector-field axioms (Faz 13.C)
    "OpCommutatorVfDefinition",
    "LieVfJacobiDefinition",
    # SN bivector formula (Faz 13.D)
    "SnBivectorFormulaDefinition",
    # Function-level Poisson axioms (Faz 13.E)
    "PoissonAsHamiltonianDefinition",
    "HamiltonianCyclicSnFormulaDefinition",
    # Operator R-linearity (supplementary expansion rules)
    "LieDerivativeArgLinearityDefinition",
    "ExteriorDerivativeLinearityDefinition",
    # Multilinear evaluation engine rules (Faz 12.A.0)
    "MultiEvalArgLinearityDefinition",
    "MultiEvalHeadLinearityDefinition",
    "MultiEvalRepeatArgZeroDefinition",
    "MultiEvalAlternatingNormalDefinition",
    # Intrinsic Cartan operator definitions (Faz 12.A.1+)
    "InteriorProductIntrinsicDefinition",
    "LieDerivativeIntrinsicDefinition",
    "ExteriorDIntrinsicDefinition",
    # Intrinsic-formula proof surface (Faz 12.A.5)
    "intrinsic_engine",
    "IntrinsicFormulaRecognizer",
    "IntrinsicFormulaMatch",
    "prove_intrinsic_equivalence",
    # Closure axioms + complete engine (Faz 12.A.6)
    "intrinsic_engine_with_closure",
    "VfActCommutatorDefinition",
    "LieBracketVfAntiSymmetryDefinition",
    "LieBracketVfJacobiDefinition",
    "IotaActAsScalarDefinition",
    # Closed-form axiom (Faz 12.B #7)
    "ClosedFormDefinition",
    # Non-degeneracy injectivity axiom (Faz 12.B #9)
    "NonDegenerateInteriorEqualityDefinition",
    # Registry-driven antisymmetry (Faz 12.B #11)
    "RegistryAntiSymCanonicalDefinition",
    # Lie-derivative rescaling (Faz 12.B #10)
    "LieRescalingDefinition",
    # Pairing C∞-linearity (Faz 12.B #12)
    "PairingScalarPullDefinition",
    # MultiEval C∞-linearity (Faz 12.B #6)
    "MultiEvalScalarPullDefinition",
    # Tilde calculus operators (Faz 14.A)
    "TildeInteriorProduct",
    "TildeExteriorDerivative",
    "TildeLieDerivative",
    "tilde_interior",
    "tilde_d",
    "tilde_lie",
    # Tilde calculus defining axioms (Faz 14.B)
    "TildeIotaSwapDefinition",
    "TildeExteriorDLichnerowiczDefinition",
    "TildeLieMagicDefinition",
    # Tilde calculus auxiliary axioms (Faz 14.D)
    "TildeIotaOnZeroVectorDefinition",
    "TildeIotaSquaredZeroDefinition",
    "TildeLieOnZeroVectorDefinition",
    "TildeDOfFunctionDefinition",
    "TildeDSquaredPoissonDefinition",
    "TildeIotaActAsScalarDefinition",
    # Tilde calculus intrinsic axioms (Faz 14.E)
    "TildeIotaIntrinsicDefinition",
    "TildeLieIntrinsicDefinition",
    "TildeDIntrinsicDefinition",
    "tilde_intrinsic_engine",
    "TildeIntrinsicFormulaRecognizer",
    "TildeIntrinsicFormulaMatch",
    "prove_tilde_cartan_relation",
    # Cartan remainder operators (Faz 15.A)
    "CartanRemainder",
    "K",
    "TildeCartanRemainder",
    "K_tilde",
    # Cartan-remainder defining axioms (Faz 15.B)
    "CartanRemainderDefinition",
    "TildeCartanRemainderDefinition",
    # Derivator helper (Faz 15.A)
    "derivator",
    # Affine connection (Faz 16.A; algebroid extension Q9 / Math 595)
    "AffineConnection",
    "ConnectionEvalExpr",
    "connection",
    "koszul_connection",
    "ConnectionXLinearityDefinition",
    "ConnectionXScalarPullDefinition",
    "ConnectionYAdditivityDefinition",
    "ConnectionYLeibnizDefinition",
    # Torsion & curvature (Faz 16.B)
    "Torsion",
    "Curvature",
    "TorsionDefinitionDefinition",
    "CurvatureDefinitionDefinition",
    # T / R C∞-bilinearity + antisymmetry (Faz 17.D)
    "TorsionXLinearityDefinition",
    "TorsionYLinearityDefinition",
    "TorsionXScalarPullDefinition",
    "TorsionYScalarPullDefinition",
    "TorsionAntiSymmetryDefinition",
    "CurvatureXLinearityDefinition",
    "CurvatureYLinearityDefinition",
    "CurvatureXScalarPullDefinition",
    "CurvatureYScalarPullDefinition",
    "CurvatureXYAntiSymmetryDefinition",
    # ∇-on-tensor Leibniz (Faz 16.C)
    "TorsionCovariantDerivative",
    "CurvatureCovariantDerivative",
    "TorsionCovariantDerivativeDefinition",
    "CurvatureCovariantDerivativeDefinition",
    # Local frame + duality (Faz 17.A)
    "FrameIndex",
    "KroneckerDelta",
    "LocalFrame",
    "local_frame",
    "FrameVectorField",
    "FrameCovector",
    "FramePairingDualityDefinition",
    # Metric + non-metricity Q (Faz 17.B)
    "MetricTensor",
    "metric",
    "MetricEvalExpr",
    "MetricEvalSymmetryDefinition",
    "MetricEvalLinearityDefinition",
    "MetricEvalScalarPullDefinition",
    "NonMetricityEvalExpr",
    "NonMetricityVLinearityDefinition",
    "NonMetricityVScalarPullDefinition",
    "NonMetricityXYSymmetryDefinition",
    "NonMetricityCompatibilityDefinition",
    # Cartan-form Expr nodes (Faz 17.C)
    "ConnectionForm",
    "ConnectionFormDefinition",
    "NonMetricityForm",
    "NonMetricityFormDefinition",
    "TorsionForm",
    "TorsionFormDefinition",
    "CurvatureForm",
    "CurvatureFormDefinition",
    # IndexedSum engine axioms (Faz 17.E.3-E.6)
    "IndexedSumSumDistributeDefinition",
    "IndexedSumNegPullDefinition",
    "IndexedSumScalarPullDefinition",
    "IndexedSumPairingPushInLeftDefinition",
    "IndexedSumPairingPushInRightDefinition",
    "IndexedSumKroneckerContractDefinition",
    # Connection-eval push-in over IndexedSum (Faz 17.F.1)
    "ConnectionEvalIndexedSumPushInDefinition",
    # MultiEval push-in over IndexedSum (Faz 17.F.2)
    "MultiEvalIndexedSumPushInDefinition",
    # Frame decomposition axioms (Faz 17.E.7 + 17.F.2)
    "FrameDecompositionDefinition",
    "ConnectionFormDecompositionDefinition",
    "ConnectionEvalYFrameDecompositionDefinition",
]


# Invariant-d re-exports are resolved lazily to break a circular import:
# ``invariant_d`` subclasses ``proof.expansion.Definition``, and
# ``proof.expansion`` itself depends on ``calculus.exterior_d``. Eagerly
# pulling ``invariant_d`` here would force ``proof.expansion`` to finish
# before it has had a chance to define ``Definition``. PEP 562's module
# ``__getattr__`` defers the pull to first access, by which time the
# import graph has settled.
_LAZY_INVARIANT_D_NAMES = frozenset(
    {
        "INVARIANT_D_CLASSIFICATIONS",
        "InvariantDOneFormDefinition",
        "invariant_d_one_form",
    }
)


def __getattr__(name: str):
    if name in _LAZY_INVARIANT_D_NAMES:
        from jacopy.calculus import invariant_d as _mod

        value = getattr(_mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
