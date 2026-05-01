"""Engine-driven proofs of the 6 tilde Cartan relations on a generic VF.

Mirrors the standard-side Faz 12.A.6 closure (which proves the textbook
Cartan relations on a 1-/2-form via :func:`prove_intrinsic_equivalence`)
on the Koszul side: each relation is stated as an operator equality
``LHS == RHS`` on a generic multivector ``V``, evaluated against a tuple
of generic 1-forms ``(η_1, …, η_q)`` and closed via
:func:`~jacopy.calculus.tilde.intrinsic_engine.prove_tilde_cartan_relation`.

The textbook poses the question for a *vector field* (1-VF). Where the
engine closes on a higher-degree ``V`` without extra cost, the test
evaluates on the degree the relation naturally fits, e.g. ``V`` a
2-vector for ι̃ ι̃ + ι̃ ι̃ = 0, since both sides drop two degrees and the
shortest non-trivial evaluation needs the result to still have at least
one slot.

Relations:

1. ``ι̃_ω ∘ ι̃_η + ι̃_η ∘ ι̃_ω = 0``         (anti-commute, no scope)
3. ``L̃_ω = d̃ ∘ ι̃_ω + ι̃_ω ∘ d̃``           (Cartan magic, defining)
5. ``[L̃_α, ι̃_β] = ι̃_{[α,β]_K}``           (commutator with ι̃)
6. ``[L̃_α, d̃] = 0``                        (Lie commutes with d̃)
4. ``[L̃_α, L̃_β] = L̃_{[α,β]_K}``           (Lie commutator)
2. ``d̃² = 0``                              (Poisson, needs ``[π,π]_SN = 0``)

Relations 2 and 4 are the hardest, they require Jacobi-style identities
of the Koszul bracket / Poisson bivector to close. Tests for them are
included so that any future closure axiom landing on the engine
immediately turns those into green lights without code changes.
"""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.calculus.tilde import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.expr import Integer, Neg, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.koszul_problem import KoszulProblem
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ProofFailure


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _build_problem(form_names=("ω", "η"), V_degree=1):
    r"""Return ``(prob, registry, π, V, *forms)``.

    The generic multivector ``V`` is declared :class:`Graded` with the
    requested SN degree; the form symbols are declared 1-forms. The
    problem registers all tilde aux + defining axioms; ``assume_poisson``
    is *not* called automatically, tests that need it invoke explicitly.
    """
    reg = PropertyRegistry()
    pi = Symbol("π")
    forms = tuple(Symbol(n) for n in form_names)
    for f in forms:
        reg.declare(f, Graded(degree=1))
    V = Symbol("V")
    reg.declare(V, Graded(degree=V_degree))
    prob = KoszulProblem(
        pi,
        forms,
        registry=reg,
        multivectors=((V, V_degree),),
    )
    return (prob, reg, pi, V) + forms


# --------------------------------------------------------------------- #
# Relation 1, ι̃_ω ι̃_η + ι̃_η ι̃_ω = 0                                  #
# --------------------------------------------------------------------- #


class TestTildeIotaAntiCommute:
    r"""``ι̃_ω ∘ ι̃_η + ι̃_η ∘ ι̃_ω = 0`` on a generic multivector.

    Each ``ι̃`` strips one form-slot. Evaluating on a 1-form ``ξ`` after
    a 2-vector ``V`` reduces to::

        V(ω, η, ξ) + V(η, ω, ξ) = 0

    which collapses through MultiEval alternating-canonicalisation.
    """

    def test_on_two_vector(self):
        prob, _, _, V, omega, eta = _build_problem(V_degree=2)
        xi = Symbol("ξ")
        prob.registry.declare(xi, Graded(degree=1))

        # ι̃_ω(ι̃_η V) + ι̃_η(ι̃_ω V)
        lhs = Sum.make(
            Act(TildeInteriorProduct(omega), Act(TildeInteriorProduct(eta), V)),
            Act(TildeInteriorProduct(eta), Act(TildeInteriorProduct(omega), V)),
        )
        chain = prob.prove_tilde_cartan(lhs, Integer(0), etas=(xi,))
        assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# Relation 3, L̃_ω = d̃ ι̃_ω + ι̃_ω d̃    (Cartan magic, defining)         #
# --------------------------------------------------------------------- #


class TestTildeCartanMagic:
    r"""``L̃_ω V = d̃(ι̃_ω V) + ι̃_ω(d̃ V)`` on a generic 1-vector.

    The defining identity for ``L̃`` on the tilde side. On a 1-vector
    ``V`` evaluated against a single 1-form ``η``, both sides collapse
    through the Faz 14.E intrinsic rules + the Aux-6 bridge that turns
    the bare ``ι̃_ω V`` produced by the d̃-arity-1 branch into a
    ``MultiEval(V, ω)`` scalar.
    """

    def test_on_one_vector(self):
        prob, _, _, V, omega, eta = _build_problem(V_degree=1)

        # L̃_ω V == d̃(ι̃_ω V) + ι̃_ω(d̃ V)
        lhs = Act(TildeLieDerivative(omega, prob.pi), V)
        rhs = Sum.make(
            Act(
                TildeExteriorDerivative(prob.pi),
                Act(TildeInteriorProduct(omega), V),
            ),
            Act(
                TildeInteriorProduct(omega),
                Act(TildeExteriorDerivative(prob.pi), V),
            ),
        )
        chain = prob.prove_tilde_cartan(lhs, rhs, etas=(eta,))
        assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# Relation 4, [L̃_α, L̃_β] = L̃_{[α,β]_K}                                  #
# --------------------------------------------------------------------- #


class TestTildeLieLieCommutator:
    r"""``L̃_α(L̃_β V) − L̃_β(L̃_α V) = L̃_{[α,β]_K} V`` on a 1-vector.

    Both sides preserve the SN degree of ``V``: with ``V`` deg-1, both
    sides are 1-vectors, and a single 1-form ``η`` evaluation yields
    a function. Closure runs through the Faz 14.G closure pipeline,
    after the slot-Lie commutator, anchor Lie homomorphism, pairing
    Leibniz, etc. lower the residue, the
    :class:`WrappedPairingAnchorAntisymmetryDefinition` cancels its 2
    pairs and :class:`TildeSnJacobiResidueDefinition` zeroes the
    remaining 5-term Schouten-Nijenhuis Jacobi obstruction (Poisson).
    Requires :meth:`assume_poisson` since both rules are Poisson-gated.
    """

    def test_on_one_vector(self):
        prob, _, _, V, alpha, beta = _build_problem(
            form_names=("α", "β"), V_degree=1
        )
        prob.assume_poisson()
        eta = Symbol("η")
        prob.registry.declare(eta, Graded(degree=1))

        lhs = Sum.make(
            Act(
                TildeLieDerivative(alpha, prob.pi),
                Act(TildeLieDerivative(beta, prob.pi), V),
            ),
            Neg(
                Act(
                    TildeLieDerivative(beta, prob.pi),
                    Act(TildeLieDerivative(alpha, prob.pi), V),
                )
            ),
        )
        bracket_form = prob.bracket(alpha, beta)
        rhs = Act(TildeLieDerivative(bracket_form, prob.pi), V)
        chain = prob.prove_tilde_cartan(lhs, rhs, etas=(eta,))
        assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# Relation 2, d̃² V = 0   (π Poisson)                                    #
# --------------------------------------------------------------------- #


class TestTildeDSquared:
    r"""``d̃(d̃ V) = 0`` on a 1-vector when ``π`` is Poisson.

    Closes via :class:`TildeDSquaredPoissonDefinition` (Aux-5), the
    closure axiom that consumes the
    :class:`~jacopy.core.properties.Poisson` flag set by
    :meth:`~jacopy.library.koszul_problem.KoszulProblem.assume_poisson`.
    Without ``assume_poisson()`` the rule is a strict no-op and the
    proof fails, exactly the behaviour that lets a script *check*
    whether a candidate ``π`` is Poisson.
    """

    def test_on_one_vector_with_poisson(self):
        prob, _, _, V, eta, xi = _build_problem(
            form_names=("η", "ξ"), V_degree=1
        )
        prob.assume_poisson()

        # d̃² V evaluated on (η, ξ): d̃V is a 2-vector, d̃²V a 3-vector,
        # so two slots leave a 1-vector residue. We need at least the
        # full arity to fully collapse, but the Aux-5 axiom matches
        # at the operator level (``Act(d̃, Act(d̃, V))``) before any
        # MultiEval expansion, so any non-empty etas tuple works.
        lhs = Act(
            TildeExteriorDerivative(prob.pi),
            Act(TildeExteriorDerivative(prob.pi), V),
        )
        chain = prob.prove_tilde_cartan(lhs, Integer(0), etas=(eta, xi))
        assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# Relation 6, [L̃_α, d̃] = 0      (Lie commutes with d̃)                  #
# --------------------------------------------------------------------- #


class TestTildeLieCommutesWithD:
    r"""``[L̃_α, d̃]V = 0``, Lie commutes with d̃ on a generic 1-vector.

    Closure runs through the Faz 14.G closure pipeline (same as rel-4):
    after the slot-Lie commutator, anchor Lie homomorphism, pairing
    Leibniz, etc. lower the residue, the
    :class:`WrappedPairingAnchorAntisymmetryDefinition` cancels its 2
    pairs and :class:`TildeSnJacobiResidueDefinition` zeroes the
    remaining 5-term Schouten-Nijenhuis Jacobi obstruction
    ``[π,π]_SN(α,η,ξ) = 0`` under Poisson. Requires
    :meth:`assume_poisson` since both rules are Poisson-gated.
    """

    def test_on_one_vector(self):
        prob, _, _, V, alpha, eta = _build_problem(
            form_names=("α", "η"), V_degree=1
        )
        prob.assume_poisson()
        xi = Symbol("ξ")
        prob.registry.declare(xi, Graded(degree=1))

        lhs = Sum.make(
            Act(
                TildeLieDerivative(alpha, prob.pi),
                Act(TildeExteriorDerivative(prob.pi), V),
            ),
            Neg(
                Act(
                    TildeExteriorDerivative(prob.pi),
                    Act(TildeLieDerivative(alpha, prob.pi), V),
                )
            ),
        )
        chain = prob.prove_tilde_cartan(lhs, Integer(0), etas=(eta, xi))
        assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# Relation 5, [L̃_α, ι̃_β] = ι̃_{[α,β]_K}                                  #
# --------------------------------------------------------------------- #


class TestTildeLieIotaCommutator:
    r"""``L̃_α(ι̃_β V) − ι̃_β(L̃_α V) = ι̃_{[α,β]_K} V`` on a 1-vector.

    Both sides are degree-0 (a 1-vector ι̃-contracted is a function).
    The Koszul bracket ``[α, β]_K`` is unfolded to its Lichnerowicz
    Cartan form by the bracket-expansion rule registered by
    :class:`KoszulProblem`; the engine then closes through
    L̃/ι̃ intrinsic + Sharp + multi-eval canonicalisation.
    """

    def test_on_two_vector(self):
        prob, _, _, V, alpha, beta = _build_problem(
            form_names=("α", "β"), V_degree=2
        )
        # V is a 2-vector: ι̃_β V is a 1-vector, ι̃_β V evaluated on
        # η gives a function. So a single eval slot fully collapses
        # both sides through the engine.
        eta = Symbol("η")
        prob.registry.declare(eta, Graded(degree=1))

        # LHS: L̃_α(ι̃_β V) − ι̃_β(L̃_α V)
        lhs = Sum.make(
            Act(
                TildeLieDerivative(alpha, prob.pi),
                Act(TildeInteriorProduct(beta), V),
            ),
            Neg(
                Act(
                    TildeInteriorProduct(beta),
                    Act(TildeLieDerivative(alpha, prob.pi), V),
                )
            ),
        )
        # RHS: ι̃_{[α,β]_K} V, the bracket-expansion rule unfolds
        # ``[α,β]_K`` into the Cartan form L_{ρα}β − L_{ρβ}α − d⟨ρα,β⟩
        # and ι̃-linearity then splits the contraction term-by-term.
        bracket_form = prob.bracket(alpha, beta)
        rhs = Act(TildeInteriorProduct(bracket_form), V)
        chain = prob.prove_tilde_cartan(lhs, rhs, etas=(eta,))
        assert isinstance(chain, ProofChain)
