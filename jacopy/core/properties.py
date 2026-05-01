"""
Property types with provenance.

A *property* is a piece of mathematical metadata attached to an
expression: "this symbol is a scalar", "this operator is graded
antisymmetric", "this bracket satisfies Jacobi". Properties are
immutable frozen dataclasses so they can be cached and hashed.

Every property carries a :class:`Provenance` tag:

* :attr:`Provenance.AXIOM`, declared by the user or library, taken as
  given for the purposes of proofs.
* :attr:`Provenance.DERIVED`, obtained from other properties via an
  algorithm. A :class:`ProofRef` records which rule produced it, so
  the unroll mode can re-derive it from axioms.

This distinction is what lets the same result be shown either as a
one-liner (``d² = 0`` as a declared axiom) or as an unrolled proof
(``d² = 0`` derived from ``[d,d] = 0`` which follows from graded
Jacobi on the Lie bracket).

Concrete property subclasses here are deliberately generic: anything
that's used across many layers (scalar, grading, symmetry). More
specialised properties, vector fields, differential forms, specific
brackets, will live alongside the modules that introduce them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union

from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree


# --------------------------------------------------------------------- #
# Provenance                                                            #
# --------------------------------------------------------------------- #


class Provenance(Enum):
    """Why does this property hold?"""

    AXIOM = "axiom"
    DERIVED = "derived"


@dataclass(frozen=True)
class ProofRef:
    """Lightweight pointer to a derivation.

    At this stage a proof reference only records the rule name and a
    tuple of *source specifiers*, each a string naming a property-type
    or axiom that was consumed. Later phases will link to a full
    :class:`ProofTree`; this keeps the core layer independent of the
    proof layer.
    """

    rule: str
    sources: Tuple[str, ...] = ()


# --------------------------------------------------------------------- #
# Base class                                                            #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Property:
    """Abstract base for every property.

    Subclasses should themselves be ``@dataclass(frozen=True)``.
    Concrete payload fields (e.g. a degree) are declared positionally;
    ``provenance`` and ``proof`` are kept keyword-only so subclasses
    can add their own positional fields without field-order conflicts.
    """

    provenance: Provenance = field(default=Provenance.AXIOM, kw_only=True)
    proof: Optional[ProofRef] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.provenance is Provenance.DERIVED and self.proof is None:
            raise ValueError(
                "Derived properties must carry a ProofRef"
            )
        if self.provenance is Provenance.AXIOM and self.proof is not None:
            raise ValueError(
                "Axiom properties cannot carry a ProofRef"
            )

    @property
    def is_axiom(self) -> bool:
        return self.provenance is Provenance.AXIOM

    @property
    def is_derived(self) -> bool:
        return self.provenance is Provenance.DERIVED


# --------------------------------------------------------------------- #
# Generic property types                                                #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scalar(Property):
    """Expression is a scalar, it commutes with everything.

    Scalars are the neutral case in the Koszul sign rule: swapping a
    scalar past anything never produces a sign.
    """


@dataclass(frozen=True)
class Graded(Property):
    """Expression has a degree in the ambient grading.

    The degree is a :class:`Degree`, concrete (``Degree.const(2)``) or
    symbolic (``Degree.var("|α|")``). Plain ``int`` is accepted and
    coerced. The degree is later consumed by the Koszul sign
    machinery: swapping two graded objects of degrees ``|a|``, ``|b|``
    produces ``(-1)^{|a||b|}``.
    """

    degree: Degree = field(default_factory=lambda: Degree.const(0))

    def __post_init__(self) -> None:
        # Parent's provenance / proof consistency check.
        super().__post_init__()
        # Coerce int → Degree without breaking frozen=True.
        if not isinstance(self.degree, Degree):
            object.__setattr__(self, "degree", as_degree(self.degree))


@dataclass(frozen=True)
class Symmetric(Property):
    """Binary operator satisfies ``B(a,b) = B(b,a)``."""


@dataclass(frozen=True)
class Antisymmetric(Property):
    """Binary operator satisfies ``B(a,b) = -B(b,a)``."""


@dataclass(frozen=True)
class GradedAntisymmetric(Property):
    """Binary operator with Koszul-signed antisymmetry.

    Satisfies ``B(a,b) = -(-1)^{|a||b|} B(b,a)``. This is the
    appropriate weakening for graded Lie brackets (Schouten-Nijenhuis,
    Koszul, etc.) where straight antisymmetry would be wrong.
    """


@dataclass(frozen=True)
class Closed(Property):
    """Form ``ω`` satisfies ``dω = 0``, declarative closedness.

    Attached to a form via :meth:`PropertyRegistry.declare`. When the
    :class:`~jacopy.calculus.closed_axioms.ClosedFormDefinition` rule
    is loaded into an engine, ``Act(d, ω)`` collapses to ``0`` for any
    ``ω`` carrying this property, no inline ``DOmegaClosed``
    :class:`~jacopy.proof.expansion.Definition` reproduction needed in
    notebook code (the 2a/2b pattern from the Faz 12 gap log).

    The property is type-agnostic about *why* ``ω`` is closed (an
    explicit symplectic 2-form, an exact form, a cohomology class
    representative, …), it just states the fact.
    """


@dataclass(frozen=True)
class NonDegenerate(Property):
    """Form ``ω`` is non-degenerate, ``ι_(·)ω: VF → 1-form`` is injective.

    Declarative analogue of :class:`Closed`: the wrapper or caller
    asserts the structural fact, and the
    :class:`~jacopy.calculus.nondegenerate_axioms.NonDegenerateInteriorEqualityDefinition`
    engine rule consumes it as a term-rewriting primitive, the
    obstruction ``ι_Y ω − ι_Z ω`` collapses to ``Y − Z`` whenever ``ω``
    carries this property.

    Mathematically equivalent to: the bundle map ``X ↦ ι_X ω`` is an
    isomorphism ``TM → T*M``. For symplectic 2-forms this is part of
    the definition; for almost-symplectic, volume, and metric forms it
    is the standalone property the reasoning machinery cites.
    """


@dataclass(frozen=True)
class Poisson(Property):
    """Bivector ``π`` satisfies ``[π, π]_SN = 0``, Poisson condition.

    Declarative analogue of :class:`Closed` / :class:`NonDegenerate`:
    the wrapper or caller asserts the Schouten-Nijenhuis self-bracket
    vanishes, and downstream engine rules (e.g. the tilde-d squared
    axiom in :mod:`jacopy.calculus.tilde.aux_axioms`) consume this
    as a primitive, ``d̃² V`` collapses to ``0`` whenever ``π``
    carries this property.

    A bivector that is both ``Antisymmetric`` and ``Poisson`` is the
    structural fingerprint of a Poisson manifold: skew + integrability.
    """


# --------------------------------------------------------------------- #
# Commutativity markers                                                 #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class NonCommuting(Property):
    """Expression has no a-priori commutativity law.

    The default for generic expressions, :class:`Product` is already
    non-commutative at the core level; this property is the explicit,
    registered counterpart for algorithms that want to assert the
    absence of commutativity rather than infer it from silence.
    """


@dataclass(frozen=True)
class AntiCommuting(Property):
    """Expression anti-commutes past other ``AntiCommuting`` factors.

    Satisfies ``a * b = - b * a`` at the element level. This is the
    ungraded version, :class:`GradedCommutative` is the right choice
    when signs depend on degrees.
    """


@dataclass(frozen=True)
class GradedCommutative(Property):
    """Expression obeys the Koszul sign rule: ``a*b = (-1)^{|a||b|} b*a``.

    Orthogonal to :class:`Graded`, which supplies the degree itself.
    An element that is both ``Graded(degree=d)`` and ``GradedCommutative``
    is the typical case for symbols living in a graded-commutative
    algebra (differential forms, polyvector fields, and so on).
    """
