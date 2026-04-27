"""Proof layer: step-by-step proof chains with pluggable strategies."""

from jacopy.proof.chain import ProofChain
from jacopy.proof.diagnostics import (
    DiagnosticHint,
    DiagnosticReport,
    DiagnosticRule,
    diagnose,
    register_rule,
)
# Register built-in rules by importing for side effects.
from jacopy.proof import diagnostic_rules  # noqa: F401
from jacopy.proof.expansion import (
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
from jacopy.proof.recognizers import (
    AntisymmetryMatch,
    AntisymmetryRecognizer,
    CommutatorMatch,
    CommutatorRecognizer,
    LeibnizMatch,
    LeibnizRecognizer,
)
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import (
    AgreementOnGenerators,
    DerivedBracketStrategy,
    ExpandAndSimplify,
    ProofFailure,
    Strategy,
    UnrollToFoundations,
)
from jacopy.proof.verifier import (
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
