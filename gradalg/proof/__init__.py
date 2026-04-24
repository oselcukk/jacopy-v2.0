"""Proof layer: step-by-step proof chains with pluggable strategies."""

from gradalg.proof.chain import ProofChain
from gradalg.proof.diagnostics import (
    DiagnosticHint,
    DiagnosticReport,
    DiagnosticRule,
    diagnose,
    register_rule,
)
# Register built-in rules by importing for side effects.
from gradalg.proof import diagnostic_rules  # noqa: F401
from gradalg.proof.expansion import (
    MODES,
    ActOverSumOpDefinition,
    Definition,
    DSquaredZeroDefinition,
    ExpansionEngine,
    IotaOnExactOneFormDefinition,
    IotaOnZeroFormDefinition,
    IotaSquaredZeroDefinition,
    LieDerivativeCartanDefinition,
    default_engine,
)
from gradalg.proof.recognizers import (
    AntisymmetryMatch,
    AntisymmetryRecognizer,
    CommutatorMatch,
    CommutatorRecognizer,
    LeibnizMatch,
    LeibnizRecognizer,
)
from gradalg.proof.step import ProofStep
from gradalg.proof.strategies import (
    AgreementOnGenerators,
    DerivedBracketStrategy,
    ExpandAndSimplify,
    ProofFailure,
    Strategy,
    UnrollToFoundations,
)
from gradalg.proof.verifier import (
    prove_equivalence,
    prove_jacobi,
    prove_operator_equation,
    show_equal,
    unroll_property,
)

__all__ = [
    # core data types
    "ProofStep",
    "ProofChain",
    # diagnostics
    "DiagnosticHint",
    "DiagnosticReport",
    "DiagnosticRule",
    "diagnose",
    "register_rule",
    # expansion
    "Definition",
    "ExpansionEngine",
    "MODES",
    "default_engine",
    "LieDerivativeCartanDefinition",
    "ActOverSumOpDefinition",
    "DSquaredZeroDefinition",
    "IotaSquaredZeroDefinition",
    "IotaOnZeroFormDefinition",
    "IotaOnExactOneFormDefinition",
    # strategies
    "Strategy",
    "ExpandAndSimplify",
    "AgreementOnGenerators",
    "UnrollToFoundations",
    "DerivedBracketStrategy",
    "ProofFailure",
    # recognizers
    "CommutatorRecognizer",
    "CommutatorMatch",
    "LeibnizRecognizer",
    "LeibnizMatch",
    "AntisymmetryRecognizer",
    "AntisymmetryMatch",
    # public API
    "show_equal",
    "prove_equivalence",
    "prove_jacobi",
    "prove_operator_equation",
    "unroll_property",
]
