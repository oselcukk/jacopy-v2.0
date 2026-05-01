"""
Schouten-Nijenhuis-with-function shortcut, Faz 15.C.

When the §3.1.5 derivator pre-pass expands ``d̃(f)`` (Lichnerowicz of a
0-form) it leaves behind ``[π, f]_SN``, the Schouten-Nijenhuis bracket
of the Poisson bivector with a function. This is the same vector field
as ``π^♯(df) = X_f``, but the engine's existing rules don't bridge the
two shapes: ``SharpOnExactDefinition`` consolidates ``π^♯(df) → X_f``,
and ``TildeDOfFunctionDefinition`` short-circuits ``d̃(f) → −π^♯(df)``,
but neither knows what to do with a bare ``BracketApply([·,·]_SN, π, f)``.

This module ships :class:`SnBracketOfFunctionDefinition`, the missing
bridge. It rewrites

    [π, f]_SN  →  −π^♯(df)

for any 0-form ``f``, scoped to a specific ``π``. Compose with the
existing Sharp pipeline and ``BracketApply([·,·]_SN, π, f)`` collapses
to ``−X_f``, which the form-side derivator residues can then cancel
against the ``X_f`` shape produced by the SharpOnExact path.

The sign is fixed by the codebase's tilde convention:
``d̃(f) := [π, f]_SN`` (Lichnerowicz definition) and
``d̃(f) → −π^♯(df)`` (the closed-form aux rule). For consistency this
module's rewrite must agree, so the output carries the leading
``Neg``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.brackets.base import BracketApply
from jacopy.brackets.schouten import SchoutenBracket, sn as default_sn
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.lie_derivative import (
    LieDerivative,
    lie_derivative as default_lie_derivative,
)
from jacopy.calculus.musical import Sharp
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.tilde.operators import TildeInteriorProduct
from jacopy.core.multi_eval import MultiEval
from jacopy.calculus.sharp_axioms import _is_degree_zero
from jacopy.core.expr import Expr, Neg, Sum, Symbol
from jacopy.core.properties import Graded, Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


class SnBracketOfFunctionDefinition(Definition):
    r"""``[π, f]_SN → −π^♯(df)`` for 0-form ``f``.

    Parameters
    ----------
    sharp
        The :class:`~jacopy.calculus.musical.Sharp` atom whose bivector
        ``π`` indexes which SN bracket the rule recognises. Two
        :class:`~jacopy.brackets.schouten.SchoutenBracket` instances
        share the same name, this rule discriminates on the operands
        rather than the bracket head, so it fires on any
        ``BracketApply(_, π, f)`` whose first slot equals ``sharp.bivector``.
    d
        Optional :class:`ExteriorDerivative` override; defaults to the
        module-level singleton.
    registry
        Optional :class:`PropertyRegistry` used to resolve the degree
        of the second slot (the function ``f``). Without a registry the
        rule declines to fire on any operand whose degree is not
        recognisable as zero.
    sn_bracket
        Optional :class:`SchoutenBracket` instance whose head must
        match ``BracketApply.bracket``. Defaults to the module
        singleton.
    """

    def __init__(
        self,
        sharp: Sharp,
        *,
        d: Optional[ExteriorDerivative] = None,
        registry: Optional[PropertyRegistry] = None,
        sn_bracket: Optional[SchoutenBracket] = None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError(
                "SnBracketOfFunctionDefinition requires a Sharp atom"
            )
        if sn_bracket is not None and not isinstance(sn_bracket, SchoutenBracket):
            raise TypeError(
                "SnBracketOfFunctionDefinition sn_bracket must be a SchoutenBracket"
            )
        self._sharp = sharp
        self._pi = sharp.bivector
        self._sn = sn_bracket if sn_bracket is not None else default_sn
        self._d = default_d if d is None else d
        self._registry = registry
        self.name = f"[π, f]_SN = −π♯(df) [{self._pi._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BracketApply):
            return False
        if expr.bracket != self._sn:
            return False
        if expr.a != self._pi:
            return False
        return _is_degree_zero(expr.b, self._registry)

    def rewrite(self, expr: Expr) -> Expr:
        f = expr.b
        return Neg(Act(self._sharp, Act(self._d, f)))


class HamiltonianPairingAntisymmetryDefinition(Definition):
    r"""``⟨X_f, μ⟩ → −π^♯(μ)(f)``, bivector antisymmetry on exact pairing.

    Mathematically: ``⟨π^♯(df), μ⟩ = π(df, μ) = −π(μ, df) = −⟨π^♯(μ), df⟩
    = −π^♯(μ)(f)``. The rule names this collapse so post-Sharp residues
    of the form ``Pairing(X_f, μ)`` reduce to a vector-field-applied-to-
    function shape, which the closure pipeline (VfActCommutator +
    LieBracketVfAntiSymmetry) can then unify with sibling residues
    inside derivator identities.

    Fires in either argument order: ``Pairing(X_f, μ)`` and
    ``Pairing(μ, X_f)``. The :class:`Pairing` constructor stores its
    arguments structurally without enforcing a 1-form-vs-vf convention,
    so both orientations turn up in practice, the engine lifts the
    antisymmetry regardless.

    Scoped to a specific :class:`~jacopy.calculus.musical.Sharp` (and
    therefore a specific bivector ``π``) so two coexisting Sharps in a
    proof don't cross-fire. The ``HamiltonianVectorField``'s recorded
    ``bivector`` must match.
    """

    def __init__(
        self,
        sharp: Sharp,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError(
                "HamiltonianPairingAntisymmetryDefinition requires a Sharp atom"
            )
        self._sharp = sharp
        self._pi = sharp.bivector
        self._registry = registry
        self.name = (
            f"⟨X_f, μ⟩ = −π♯(μ)(f) [{self._pi._repr_inner()}]"
        )

    def _is_one_form(self, expr: Expr) -> bool:
        try:
            return degree_of(expr, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def _classify(self, expr: Pairing):
        a, b = expr.alpha, expr.X
        if (
            isinstance(a, HamiltonianVectorField)
            and a.bivector == self._pi
            and self._is_one_form(b)
        ):
            return a, b
        if (
            isinstance(b, HamiltonianVectorField)
            and b.bivector == self._pi
            and self._is_one_form(a)
        ):
            return b, a
        return None

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        return self._classify(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        ham, mu = self._classify(expr)  # type: ignore[misc]
        f = ham.function
        return Neg(Act(Act(self._sharp, mu), f))


class MultiEvalOnHamiltonianDefinition(Definition):
    r"""``α(X_f) → −π^♯(α)(f)``, bivector antisymmetry on
    1-form-evaluated-on-Hamiltonian-vf.

    The same algebraic identity as
    :class:`HamiltonianPairingAntisymmetryDefinition` but in the
    :class:`MultiEval` shape that the engine produces from
    arity-1 ``MultiEval(α, X_f)`` evaluations of a 1-form on a single
    vector field. ``MultiEval`` and ``Pairing`` are structurally
    distinct in the codebase even though they encode the same scalar
    ``ι_{X_f} α``; both shapes turn up in §3.1.5 derivator residues
    and need a matching antisymmetry rule.

    Fires on arity-1 vector-slot ``MultiEval(α, X_f)`` where ``α`` is
    a 1-form and ``X_f`` is a Hamiltonian vector field whose bivector
    matches this rule's ``π``.
    """

    def __init__(
        self,
        sharp: Sharp,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError(
                "MultiEvalOnHamiltonianDefinition requires a Sharp atom"
            )
        self._sharp = sharp
        self._pi = sharp.bivector
        self._registry = registry
        self.name = (
            f"α(X_f) = −π♯(α)(f) [{self._pi._repr_inner()}]"
        )

    def _is_one_form(self, expr: Expr) -> bool:
        try:
            return degree_of(expr, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if expr.slot_kind != "vector":
            return False
        if expr.arity != 1:
            return False
        if not self._is_one_form(expr.head):
            return False
        (arg,) = expr.args
        if not isinstance(arg, HamiltonianVectorField):
            return False
        return arg.bivector == self._pi

    def rewrite(self, expr: Expr) -> Expr:
        alpha = expr.head
        (ham,) = expr.args
        f = ham.function  # type: ignore[union-attr]
        return Neg(Act(Act(self._sharp, alpha), f))


# --------------------------------------------------------------------- #
# Helpers, vf detection for Pairing-Leibniz                             #
# --------------------------------------------------------------------- #


_PAIRING_LEIBNIZ_RESERVED_OPS: Optional[tuple] = None


def _pairing_leibniz_reserved_ops() -> tuple:
    """Lazy lookup of operator atoms that are NOT plain vector fields.

    The Pairing-Leibniz rule must not fire when its outer ``Act`` head
    is one of the structural Cartan operators (``L_X``, ``ι_X``, ``d``)
    or their tilde counterparts, nor on bare ``Sharp``/``Flat``: those
    have specific operator semantics on forms / multivectors and are
    governed by their own axioms (e.g.
    :class:`~jacopy.calculus.pairing_axioms.PairingLieLeibnizDefinition`
    handles ``L_X⟨α, β⟩``).
    """
    global _PAIRING_LEIBNIZ_RESERVED_OPS
    if _PAIRING_LEIBNIZ_RESERVED_OPS is None:
        from jacopy.calculus.tilde.operators import (
            TildeExteriorDerivative,
            TildeInteriorProduct,
            TildeLieDerivative,
        )
        from jacopy.calculus.cartan_remainder import CartanRemainder
        from jacopy.calculus.tilde.cartan_remainder import TildeCartanRemainder
        from jacopy.calculus.musical import Flat

        _PAIRING_LEIBNIZ_RESERVED_OPS = (
            LieDerivative,
            InteriorProduct,
            ExteriorDerivative,
            TildeLieDerivative,
            TildeInteriorProduct,
            TildeExteriorDerivative,
            CartanRemainder,
            TildeCartanRemainder,
            Sharp,
            Flat,
        )
    return _PAIRING_LEIBNIZ_RESERVED_OPS


def _is_vf_for_pairing_leibniz(D: Expr) -> bool:
    """Whether ``D`` may be treated as a vector field acting on a scalar.

    Permissive: accepts bare :class:`Symbol` (the user's convention for
    declaring vfs by name + ``Graded(degree=1)``),
    :class:`Derivation` atoms, :class:`HamiltonianVectorField`,
    :class:`LieBracketVF`, and ``Act(Sharp(π), α)`` shapes (the anchor
    image, which is a vf even though structurally an ``Act``).

    Rejects the structural Cartan operators
    (:class:`LieDerivative`, :class:`InteriorProduct`,
    :class:`ExteriorDerivative` and their tilde counterparts,
    :class:`CartanRemainder`/:class:`TildeCartanRemainder`) and bare
    :class:`Sharp`/:class:`Flat` heads, those have their own operator
    semantics handled by other rules.
    """
    if isinstance(D, _pairing_leibniz_reserved_ops()):
        return False
    if isinstance(D, Symbol):
        return True
    if isinstance(D, (Derivation, LieBracketVF, HamiltonianVectorField)):
        return True
    if isinstance(D, Act) and isinstance(D.op, Sharp):
        return True
    return False


# --------------------------------------------------------------------- #
# VfActOnPairingLeibniz, D⟨a, b⟩ → ⟨L_D a, b⟩ + ⟨a, L_D b⟩              #
# --------------------------------------------------------------------- #


class VfActOnPairingLeibnizDefinition(Definition):
    r"""``Act(D, Pairing(a, b)) → Pairing(L_D a, b) + Pairing(a, L_D b)``.

    The Lie-Leibniz on a scalar pairing for a vector-field-like operator
    ``D``. Distinct from
    :class:`~jacopy.calculus.pairing_axioms.PairingLieLeibnizDefinition`
    which fires on ``Act(LieDerivative(X), Pairing)``, this rule fires
    on a *bare* vf-like atom (Symbol, Derivation, ``Act(Sharp, …)``,
    ``LieBracketVF``, ``HamiltonianVectorField``) acting as a degree-0
    derivation on the scalar produced by the pairing.

    Why both rules: §3.1.5 derivator residues produce the bare-vf shape
    ``Y(W(⟨π^♯ω, η⟩))`` where ``W`` is a user-declared vector-field
    Symbol, never wrapped in ``LieDerivative``. The existing
    pairing-Leibniz can't reach that shape; this rule bridges by
    expanding ``W(⟨…⟩)`` into the standard two-term Leibniz form. After
    the expansion, downstream intrinsic rules collapse
    ``Act(LieDerivative(W), vf)`` to ``LieBracketVF(W, vf)`` and
    ``Act(LieDerivative(W), 1-form)`` to its Cartan-magic expansion.
    """

    def __init__(self, *, lie_derivative=None) -> None:
        self._lie = (
            default_lie_derivative if lie_derivative is None else lie_derivative
        )
        self.name = "D⟨a, b⟩ → ⟨L_D a, b⟩ + ⟨a, L_D b⟩"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        if not isinstance(expr.arg, Pairing):
            return False
        return _is_vf_for_pairing_leibniz(expr.op)

    def rewrite(self, expr: Expr) -> Expr:
        D = expr.op
        p = expr.arg
        L_D = self._lie(D)
        return Sum(
            Pairing(Act(L_D, p.alpha), p.X),
            Pairing(p.alpha, Act(L_D, p.X)),
        )


# --------------------------------------------------------------------- #
# PoissonCommutatorOnInterior, ι_[π,W]_SN(ω) under Poisson              #
# --------------------------------------------------------------------- #


class PoissonCommutatorOnInteriorDefinition(Definition):
    r"""``ι_[π,W]_SN(ω) → π^♯(L_W ω) + [π^♯(ω), W]_VF`` when ``π`` is Poisson.

    The Lichnerowicz commutator identity:

    .. math::

        [L_W, \pi^\sharp](\omega) \;=\; -[\pi, W]_{\mathrm{SN}}^\sharp(\omega)

    rearranged so the SN-bracket-of-bivector-with-vf, contracted on a
    1-form, opens into a flat sum the §3.1.5 closure pipeline can
    cancel.

    The shape ``ι_{[π,W]_SN}(ω)`` arises in §3.1.5 form-side identity
    (3) after the surgical pre-pass canonicalizes
    ``[π,W]_SN(ω, ·)`` to its interior-product form. Treated here as
    ``[π,W]_SN^♯(ω)``: ``[π,W]_SN`` is a bivector (SN-degree
    ``2 + 1 - 1 = 2``), so ``ι_X`` of it on a 1-form ω returns a vector
    field, exactly ``[π,W]_SN^♯(ω)`` under the standard
    bivector-Sharp identification.

    Scoped to a specific :class:`Sharp` (and therefore a specific
    ``π``). Gated on ``Poisson(π)`` in the registry; declines to fire
    otherwise (the identity requires the Jacobi condition
    ``[π, π]_SN = 0`` to deduce the Lichnerowicz commutator).
    """

    def __init__(
        self,
        sharp: Sharp,
        *,
        registry: PropertyRegistry,
        lie_derivative=None,
        sn_bracket: Optional[SchoutenBracket] = None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError(
                "PoissonCommutatorOnInteriorDefinition requires a Sharp atom"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "PoissonCommutatorOnInteriorDefinition requires a PropertyRegistry"
            )
        self._sharp = sharp
        self._pi = sharp.bivector
        self._registry = registry
        self._lie = (
            default_lie_derivative if lie_derivative is None else lie_derivative
        )
        self._sn = sn_bracket if sn_bracket is not None else default_sn
        self.name = (
            f"ι_[π,W]_SN(ω) → π♯(L_W ω) + [π♯(ω), W]_VF [{self._pi._repr_inner()}]"
        )

    def _classify(self, iota_op: InteriorProduct) -> Optional[Expr]:
        """Return ``W`` if ``iota_op.vector_field == [π, W]_SN``, else ``None``."""
        vf_slot = iota_op.vector_field
        if not isinstance(vf_slot, BracketApply):
            return None
        if vf_slot.bracket != self._sn:
            return None
        if vf_slot.a != self._pi:
            return None
        return vf_slot.b

    def matches(self, expr: Expr) -> bool:
        if not self._registry.has(self._pi, Poisson):
            return False
        if not isinstance(expr, Act):
            return False
        iota = expr.op
        if not isinstance(iota, InteriorProduct):
            return False
        return self._classify(iota) is not None

    def rewrite(self, expr: Expr) -> Expr:
        iota = expr.op
        omega = expr.arg
        W = self._classify(iota)  # type: ignore[arg-type]
        assert W is not None, "matches() guarantees [π, W]_SN inside ι"
        L_W = self._lie(W)
        return Sum(
            Act(self._sharp, Act(L_W, omega)),
            LieBracketVF(Act(self._sharp, omega), W),
        )


# --------------------------------------------------------------------- #
# PairingToMultiEval bridge, ⟨α, X⟩ → α(X) for arity-1 vector slot      #
# --------------------------------------------------------------------- #


class PairingToMultiEvalBridgeDefinition(Definition):
    r"""``Pairing(α, X) → MultiEval(α, X, slot_kind="vector")`` and mirror.

    The codebase keeps :class:`Pairing` and arity-1 vector-slot
    :class:`MultiEval` as structurally distinct atoms even though they
    encode the same scalar ``⟨α, X⟩``. §3.1.5 derivator residues mix
    both shapes, one term arrives as ``Pairing(α, X)`` (from
    :class:`PairingLieLeibnizDefinition` /
    :class:`VfActOnPairingLeibnizDefinition` outputs) while a sibling
    arrives as ``MultiEval(α, X)`` (from intrinsic-rule expansion of
    Cartan operators). Without this bridge sibling cancellations don't
    line up.

    The rule canonicalizes towards :class:`MultiEval` (the engine's
    preferred scalar-pairing shape, Cartan intrinsic axioms produce
    MultiEval, and downstream antisymmetry rules
    (:class:`HamiltonianPairingAntisymmetryDefinition`,
    :class:`MultiEvalOnHamiltonianDefinition`) operate on that shape).

    Fires when the second slot is "vector-field-like"
    (per :func:`_is_vf_for_pairing_leibniz`) and the first is
    1-form-like (registry-declared ``Graded(degree=1)`` and not vf-like)
   , or vice versa, since :class:`Pairing` stores its slots without
    enforcing a 1-form-vs-vf order.
    """

    def __init__(self, *, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry
        self.name = "⟨α, X⟩ → α(X)"

    def _is_one_form(self, expr: Expr) -> bool:
        if self._registry is None:
            return False
        try:
            return degree_of(expr, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def _classify(self, p: Pairing) -> Optional[tuple]:
        """Return ``(alpha_1form, X_vf)`` if the pair fits the bridge, else None."""
        a, b = p.alpha, p.X
        if _is_vf_for_pairing_leibniz(a) and self._is_one_form(b):
            return (b, a)
        if _is_vf_for_pairing_leibniz(b) and self._is_one_form(a):
            return (a, b)
        # Allow Act(LieDerivative, 1-form) or Act(InteriorProduct, …) shapes
        # whose result-degree resolves to 1 via degree_of.
        if self._is_one_form(a) and _is_vf_for_pairing_leibniz(b):
            return (a, b)
        if self._is_one_form(b) and _is_vf_for_pairing_leibniz(a):
            return (b, a)
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Pairing)
            and self._classify(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        result = self._classify(expr)
        assert result is not None, "matches() guarantees a classification"
        alpha, X = result
        return MultiEval(alpha, X, alternating=True, slot_kind="vector")


# --------------------------------------------------------------------- #
# BivectorAntisymmetry on MultiEval, α(π^♯β) ↔ −β(π^♯α)                 #
# --------------------------------------------------------------------- #


class MultiEvalBivectorAntisymmetryDefinition(Definition):
    r"""``MultiEval(α, π^♯(β)) → −MultiEval(β, π^♯(α))`` for 1-forms ``α``, ``β``.

    The bivector antisymmetry ``π(α, β) = −π(β, α)`` lifted to the
    1-form-evaluated-on-Sharp-image shape. Same algebraic content as
    :class:`HamiltonianPairingAntisymmetryDefinition` /
    :class:`MultiEvalOnHamiltonianDefinition`, but for the
    *non-Hamiltonian* Sharp shape: the right slot is ``Act(Sharp, β)``
    where ``β`` is an arbitrary 1-form (not necessarily the differential
    of a function).

    Direction: rewrites only when the right slot's 1-form is
    structurally "smaller" by a deterministic ordering, to avoid an
    infinite swap ``α(π^♯β) ↔ −β(π^♯α) ↔ α(π^♯β)``. The ordering uses
    Python's tuple ``id``-stable comparison via ``repr``, sufficient
    for canonicalization since both shapes are reached but only one
    fires per pair.

    Scoped to a specific :class:`Sharp`. Skips when the right-slot
    1-form equals the head 1-form (the rule ``α(π^♯α) = π(α, α) = 0``
    is handled separately by π-antisymmetry on the bivector slot).
    """

    def __init__(
        self,
        sharp: Sharp,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError(
                "MultiEvalBivectorAntisymmetryDefinition requires a Sharp atom"
            )
        self._sharp = sharp
        self._pi = sharp.bivector
        self._registry = registry
        self.name = (
            f"α(π♯(β)) → −β(π♯(α)) [{self._pi._repr_inner()}]"
        )

    def _is_one_form(self, expr: Expr) -> bool:
        if self._registry is None:
            return False
        try:
            return degree_of(expr, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def _classify(self, expr: MultiEval) -> Optional[tuple]:
        if expr.slot_kind != "vector" or expr.arity != 1:
            return None
        alpha = expr.head
        if not self._is_one_form(alpha):
            return None
        (X,) = expr.args
        if not isinstance(X, Act):
            return None
        if X.op != self._sharp:
            return None
        beta = X.arg
        if not self._is_one_form(beta):
            return None
        if alpha == beta:
            return None
        # Direction: fire when ``repr(alpha) > repr(beta)`` so the
        # canonical form has the structurally smaller 1-form as head.
        if repr(alpha) <= repr(beta):
            return None
        return (alpha, beta)

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        return self._classify(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        result = self._classify(expr)  # type: ignore[arg-type]
        assert result is not None
        alpha, beta = result
        return Neg(
            MultiEval(
                beta,
                Act(self._sharp, alpha),
                alternating=True,
                slot_kind="vector",
            )
        )


# --------------------------------------------------------------------- #
# Permissive VF-commutator pair finder + LieBracketVF antisymmetry     #
# --------------------------------------------------------------------- #


def _strip_neg_pair(expr: Expr) -> Tuple[bool, Expr]:
    """Return ``(has_neg, inner)``, peeling a single :class:`Neg`."""
    if isinstance(expr, Neg):
        return True, expr.arg
    return False, expr


def _match_act_act_permissive(
    expr: Expr,
) -> Optional[Tuple[Expr, Expr, Expr]]:
    """Match ``Act(X, Act(Y, f))`` for permissive vf X, Y; return ``(X, Y, f)``.

    Permissive vf: see :func:`_is_vf_for_pairing_leibniz`, accepts
    Symbol, Derivation, LieBracketVF, HamiltonianVectorField,
    ``Act(Sharp, …)``. Rejects pairs where ``X == Y`` (commutator
    vanishes trivially).
    """
    if not isinstance(expr, Act):
        return None
    outer = expr.op
    inner_act = expr.arg
    if not isinstance(inner_act, Act):
        return None
    inner_op = inner_act.op
    if not (
        _is_vf_for_pairing_leibniz(outer)
        and _is_vf_for_pairing_leibniz(inner_op)
    ):
        return None
    if outer == inner_op:
        return None
    return outer, inner_op, inner_act.arg


class PermissiveVfActCommutatorDefinition(Definition):
    r"""``Act(X, Act(Y, f)) − Act(Y, Act(X, f)) → Act([X, Y]_VF, f)``.

    Same Sum-level pair-finder shape as
    :class:`~jacopy.calculus.closure_axioms.VfActCommutatorDefinition`,
    but with the permissive vf check
    :func:`_is_vf_for_pairing_leibniz` (accepts bare :class:`Symbol`,
    :class:`Derivation`, :class:`LieBracketVF`,
    :class:`HamiltonianVectorField`, ``Act(Sharp, …)``). The original
    closure rule restricts to plain ``Derivation`` atoms via
    ``_is_plain_vf``, which rejects :class:`Symbol`, but §3.1.5
    derivator probes declare vfs as Symbols carrying
    ``Graded(degree=1)``, so the original rule cannot reach them.

    This rule is intended for the Faz 15.C ``derivator_form_engine``
    bundle and should not be added to the generic intrinsic engine,
    it would compete with :class:`VfActCommutatorDefinition` on
    Derivation residues that the original was designed to handle.
    """

    name = "X(Y(f)) − Y(X(f)) = [X,Y]_VF(f)  [permissive]"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Sum) and self._find_pair(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_pair(expr)
        assert match is not None, "matches() guarantees a pair"
        i, j, X, Y, f = match
        bracket = LieBracketVF(X, Y)
        new_term = Act(bracket, f)
        kept = [c for k, c in enumerate(expr.children) if k != i and k != j]
        return Sum.make(new_term, *kept)

    def _find_pair(
        self, sum_expr: Sum
    ) -> Optional[Tuple[int, int, Expr, Expr, Expr]]:
        children = sum_expr.children
        for i, j in combinations(range(len(children)), 2):
            for a, b in ((i, j), (j, i)):
                neg_a, inner_a = _strip_neg_pair(children[a])
                neg_b, inner_b = _strip_neg_pair(children[b])
                if neg_a or not neg_b:
                    continue
                pos = _match_act_act_permissive(inner_a)
                neg = _match_act_act_permissive(inner_b)
                if pos is None or neg is None:
                    continue
                X1, Y1, f1 = pos
                Y2, X2, f2 = neg
                if X1 == X2 and Y1 == Y2 and f1 == f2:
                    return a, b, X1, Y1, f1
        return None


class LieBracketVfSumLinearityDefinition(Definition):
    r"""``LieBracketVF(Sum(a, b, …), Y) → Σ LieBracketVF(a_i, Y)`` (and mirror).

    Distributes a ``Sum`` operand through either slot of the
    Lie-bracket-of-vfs atom. Required by the §3.1.5 (1') closure path:
    after the L̃-Lichnerowicz emits a three-term Sum into a SN-bracket
    slot and the SN bilinearity peels that into per-summand brackets,
    a parallel residue path produces ``LBVF(Sum(π^♯(−L_U η), X_{η(U)}), V)``
    via SharpLinearity inside an LBVF.X slot. Without explicit Sum
    linearity the LBVF stays opaque and the eventual cancellation /
    Jacobi sweep cannot find its targets.
    """

    name = "[Sum(a,b,…), Y]_VF → Σ_i [a_i, Y]_VF  (and mirror)"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        return isinstance(expr.X, Sum) or isinstance(expr.Y, Sum)

    def rewrite(self, expr: Expr) -> Expr:
        X, Y = expr.X, expr.Y
        if isinstance(X, Sum):
            return Sum.make(*(LieBracketVF(c, Y) for c in X.children))
        return Sum.make(*(LieBracketVF(X, c) for c in Y.children))


class LieBracketVfNegLinearityDefinition(Definition):
    r"""``LieBracketVF(−X, Y) → −LieBracketVF(X, Y)`` (and mirror on right slot).

    Lie-bracket bilinearity over the additive sign: pushes a Neg
    wrapper out of either bracket slot. Necessary for the §3.1.5
    derivator residue: when
    :class:`LieBracketVfAntisymmetryDefinition` rewrites a nested
    bracket like ``[π^♯μ, Y]_VF`` to ``Neg([Y, π^♯μ]_VF)`` *inside*
    another bracket's argument, the result ``[Neg([Y,π^♯μ]), U]_VF``
    is no longer a recognized "nested bracket" by
    :func:`_peel_lie_bracket_jacobi`, the Jacobi rule needs the Neg
    pushed out to ``Neg([[Y,π^♯μ], U])``.
    """

    name = "[(−X), Y]_VF → −[X, Y]_VF"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        return isinstance(expr.X, Neg) or isinstance(expr.Y, Neg)

    def rewrite(self, expr: Expr) -> Expr:
        X, Y = expr.X, expr.Y
        sign_neg = False
        if isinstance(X, Neg):
            X = X.arg
            sign_neg = not sign_neg
        if isinstance(Y, Neg):
            Y = Y.arg
            sign_neg = not sign_neg
        new = LieBracketVF(X, Y)
        return Neg(new) if sign_neg else new


class LieBracketVfAntisymmetryDefinition(Definition):
    r"""``LieBracketVF(Y, X) → −LieBracketVF(X, Y)`` for ``repr(Y) > repr(X)``.

    Canonicalizes the arg order of vector-field Lie brackets via
    deterministic ``repr`` ordering. The direction guard prevents an
    infinite swap loop: only one of the two orientations satisfies
    ``repr(Y) > repr(X)`` for distinct X, Y.

    Why this matters: §3.1.5 derivator residues mix two paths that
    end up at the same vector-field commutator with opposite arg
    orders, one path emits ``[U, π^♯η]_VF`` (from the permissive
    pair finder), the other ``[π^♯η, U]_VF`` (from a Lichnerowicz
    expansion). Without this rule the two terms remain
    structurally distinct and ``collect_terms`` cannot fold them.
    With it, both canonicalize to a single orientation, and the
    sign flip lets them cancel.
    """

    name = "[Y, X]_VF → −[X, Y]_VF  [repr-canonical]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        if expr.X == expr.Y:
            return False
        return repr(expr.X) > repr(expr.Y)

    def rewrite(self, expr: Expr) -> Expr:
        return Neg(LieBracketVF(expr.Y, expr.X))


# --------------------------------------------------------------------- #
# Faz 15.C, multivector-side dual closures.                             #
# --------------------------------------------------------------------- #


class SnBracketOfOneVectorsToLieBracketVfDefinition(Definition):
    r"""``[U, V]_SN → [U, V]_VF`` when both operands are 1-vectors.

    The Schouten-Nijenhuis bracket of two 1-vectors equals their
    vector-field Lie bracket. The engine keeps these as structurally
    distinct atoms (``BracketApply(sn, U, V)`` vs :class:`LieBracketVF`)
    so the coercion has to be made explicit before downstream rules
    that expect :class:`LieBracketVF` (the form-side intrinsic
    ``L_X`` and ``d`` rules emit it; the iota-act-as-scalar bridges
    recognise it) can fire.

    Gated on a registry lookup: both arguments must carry
    :class:`Graded` ``degree=1``. Higher-degree multivectors keep their
    SN bracket, only the 1-vector case is the Lie bracket.
    """

    def __init__(
        self,
        *,
        registry: PropertyRegistry,
        sn_bracket: Optional[SchoutenBracket] = None,
    ) -> None:
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "SnBracketOfOneVectorsToLieBracketVfDefinition "
                "requires a PropertyRegistry"
            )
        if sn_bracket is not None and not isinstance(sn_bracket, SchoutenBracket):
            raise TypeError(
                "SnBracketOfOneVectorsToLieBracketVfDefinition "
                "sn_bracket must be a SchoutenBracket"
            )
        self._registry = registry
        self._sn = sn_bracket if sn_bracket is not None else default_sn
        self.name = "[U, V]_SN → [U, V]_VF  [both 1-vectors]"

    def _is_one_vector(self, expr: Expr) -> bool:
        # Structural 1-vector recognizers, opaque atoms whose
        # construction guarantees SN-degree 1, regardless of registry
        # state. Mirrors
        # :class:`MultiEvalCovectorPairingFlipDefinition._is_one_vector_like`,
        # extended with ``Act(TildeExteriorDerivative, _)`` (the
        # tilde-d of a scalar is a Hamiltonian vf, hence a 1-vector).
        if isinstance(expr, LieBracketVF):
            return True
        if isinstance(expr, HamiltonianVectorField):
            return True
        if isinstance(expr, Act) and isinstance(expr.op, Sharp):
            return True
        if isinstance(expr, Act):
            from jacopy.calculus.tilde.operators import TildeExteriorDerivative
            if isinstance(expr.op, TildeExteriorDerivative):
                return True
        graded = self._registry.get(expr, Graded)
        if graded is None:
            return False
        return graded.degree == Degree.const(1)

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BracketApply):
            return False
        if expr.bracket != self._sn:
            return False
        return self._is_one_vector(expr.a) and self._is_one_vector(expr.b)

    def rewrite(self, expr: Expr) -> Expr:
        return LieBracketVF(expr.a, expr.b)


class TildeIotaOnLieBracketVfActAsScalarDefinition(Definition):
    r"""Bridge ``Act(D, Act(ι̃_α, [U, V]_VF)) → Act(D, MultiEval([U,V]_VF, α, "covector"))``.

    Mirror of
    :class:`~jacopy.calculus.tilde.aux_axioms.TildeIotaActAsScalarDefinition`
    for :class:`~jacopy.algebra.lie_bracket_vf.LieBracketVF` operands.
    LieBracketVF instances are 1-vectors by construction, but the
    registry has no Graded entry on the synthetic atom, the original
    rule's ``Graded(1)`` guard skips them. This sibling fires
    unconditionally when the inner contraction is over a LieBracketVF,
    matching the same bridge shape and emitting the same
    arity-1 covector :class:`MultiEval`.

    Why a sibling instead of relaxing the original: keeping the
    ``Graded(1)`` guard there blocks misfires on higher-degree
    multivectors; opening it up to all 1-vector-shaped Exprs would
    require enumerating (LieBracketVF, ``Act(Sharp, _)``,
    ``BracketApply(sn, _, _)`` for two 1-vectors, …). One sibling per
    new shape keeps each rule's match condition explicit.
    """

    name = "Act(D, ι̃_α([U,V]_VF)) → Act(D, MultiEval([U,V]_VF, α, covector))"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, TildeInteriorProduct)):
            return False
        return isinstance(inner.arg, LieBracketVF)

    def rewrite(self, expr: Expr) -> Expr:
        outer_op = expr.op
        inner_act = expr.arg
        iota = inner_act.op
        V = inner_act.arg
        omega = iota.form
        return Act(
            outer_op,
            MultiEval(V, omega, alternating=True, slot_kind="covector"),
        )


class VfActOnExteriorDOfScalarDefinition(Definition):
    r"""``Act(D, Act(d, f)) → Act(D, f)`` for vf-like ``D`` and 0-form ``f``.

    Direct evaluation of ``D(df)`` as the directional derivative
    ``D(f)``, in this engine's semantics, ``Act(D, X)`` for vf ``D``
    only makes sense when ``X`` is a 0-form scalar; the contraction
    ``ι_D(df) = D(f)`` is the only meaningful reading. Mirrors the
    arity-1 branch of
    :class:`~jacopy.calculus.intrinsic_axioms.ExteriorDIntrinsicDefinition`
    (which fires inside :class:`MultiEval` wraps) for the bare-Act
    shape that emerges from §3.1.5 multivector-side dual residues
    after the d̃ intrinsic emits a directional-derivative factor on a
    ``d``-of-scalar.

    The 0-form guard on ``f`` is essential: for a higher-degree ``f``
    (e.g. a 1-form ω, where ``d(ω)`` is a 2-form), the rewrite
    ``D(d(ω)) → D(ω)`` would jump categories and produce a malformed
    expression. The ``_is_vf_for_pairing_leibniz`` guard on ``D``
    accepts the same vf-like atoms as
    :class:`PermissiveIotaActAsScalarDefinition` (Symbol, Derivation,
    LieBracketVF, HamiltonianVF, ``Act(Sharp, _)``).
    """

    def __init__(self, *, registry: PropertyRegistry) -> None:
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "VfActOnExteriorDOfScalarDefinition requires a PropertyRegistry"
            )
        self._registry = registry
        self.name = "Act(D, df) → Act(D, f)  [D vf-like, f 0-form]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        if not _is_vf_for_pairing_leibniz(expr.op):
            return False
        inner = expr.arg
        return (
            isinstance(inner, Act)
            and isinstance(inner.op, ExteriorDerivative)
        )

    def rewrite(self, expr: Expr) -> Expr:
        outer_op = expr.op
        f = expr.arg.arg
        return Act(outer_op, f)


class VfActOnLieOfScalarDefinition(Definition):
    r"""``Act(D, Act(L_X, f)) → Act(D, Act(X, f))`` for vf-like ``D``.

    Sibling of :class:`VfActOnExteriorDOfScalarDefinition`: the
    structural type-safety argument that an outer ``Act(D, _)`` with a
    vf-like ``D`` forces its inner argument to be a 0-form scalar. On a
    0-form, ``L_X`` collapses to the directional derivative ``X(f)``,
    so the rewrite is just lifting that identity through the outer
    ``D`` wrap.

    Why this is needed: the §3.1.5 multivector-side dual residue (2')
    produces the bare shape ``Act(V, Act(L_π♯(ξ), Act(ι_U, η)))``
    where ``ι_U η`` is structurally a 0-form. The form-side
    :class:`~jacopy.calculus.intrinsic_axioms.LieDerivativeIntrinsicDefinition`
    only fires inside :class:`MultiEval` wraps, so without this rule
    the bare ``L_X`` term sits unreduced. After this rule fires the
    resulting ``Act(D, Act(X, ι_U η))`` is then cracked open by
    :class:`PermissiveIotaActAsScalarDefinition`, matching the canonical
    ``Act(D, Act(X, MultiEval(η, U, "vector")))`` shape that the rest
    of the residue uses.
    """

    name = "Act(D, L_X(f)) → Act(D, X(f))  [D vf-like, f 0-form]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        if not _is_vf_for_pairing_leibniz(expr.op):
            return False
        inner = expr.arg
        return isinstance(inner, Act) and isinstance(inner.op, LieDerivative)

    def rewrite(self, expr: Expr) -> Expr:
        D = expr.op
        lie = expr.arg.op  # LieDerivative
        f = expr.arg.arg
        X = lie.vector_field
        return Act(D, Act(X, f))


class PermissiveIotaActAsScalarDefinition(Definition):
    r"""``Act(D, Act(ι_X, ω)) → Act(D, MultiEval(ω, X, "vector"))``.

    Mirror of
    :class:`~jacopy.calculus.closure_axioms.IotaActAsScalarDefinition`
    that accepts bare :class:`Symbol` vector fields (the original rule
    requires :class:`Derivation` typing on ``D`` via ``_is_plain_vf``,
    and rejects user-declared ``Symbol`` vfs).

    The §3.1.5 multivector-side derivator residues end up with
    ``Act(Sharp(π)(ξ), Act(U, Act(ι_V, η)))`` shapes where the inner
    operator ``U`` is a registry-declared :class:`Symbol` (the
    convention W/U/V stay as Symbol, converting them to
    :class:`Derivation` triggers misclassification by
    ``degree_of`` and breaks
    :class:`~jacopy.calculus.sn_function_axiom.SnBracketOfFunctionDefinition`).
    Without this rule the inner ``Act(ι_V, η)`` cannot be folded into
    the canonical ``MultiEval`` shape, and the residue stalls between
    ``U(η(V))`` (which one path produces) and ``U(ι_V(η))`` (the
    other).

    Matching mirrors the original rule but uses
    :func:`_is_vf_for_pairing_leibniz` (accepts :class:`Symbol`,
    :class:`Derivation`, :class:`LieBracketVF`,
    :class:`HamiltonianVectorField`, ``Act(Sharp, _)``) on ``D``.
    """

    name = "bare ι_X(ω) inside Act(D, _) → Act(D, MultiEval(ω, X))  [permissive]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        if not _is_vf_for_pairing_leibniz(expr.op):
            return False
        inner = expr.arg
        return (
            isinstance(inner, Act)
            and isinstance(inner.op, InteriorProduct)
        )

    def rewrite(self, expr: Expr) -> Expr:
        outer_op = expr.op
        inner_act = expr.arg
        iota = inner_act.op
        omega = inner_act.arg
        return Act(
            outer_op,
            MultiEval(
                omega,
                iota.vector_field,
                alternating=True,
                slot_kind="vector",
            ),
        )


class MultiEvalCovectorPairingFlipDefinition(Definition):
    r"""``MultiEval(X, ω, "covector", arity 1) → MultiEval(ω, X, "vector")``.

    The arity-1 scalar pairing of a 1-vector ``X`` against a 1-form
    ``ω`` is symmetric in shape: ``X(ω) = ω(X)``. The covector-slot
    evaluation appears after
    :class:`~jacopy.calculus.tilde.aux_axioms.TildeIotaActAsScalarDefinition`
    bridges ``ι̃_ω(V) → V(ω)``; the form-side intrinsic rules
    (:class:`~jacopy.calculus.intrinsic_axioms.LieDerivativeIntrinsicDefinition`,
    :class:`~jacopy.calculus.intrinsic_axioms.ExteriorDIntrinsicDefinition`,
    arity-1 ``d``) only fire on vector-slot evaluations, so the flip is
    needed before they can crack open ``ω`` when it is structurally a
    1-form (``Act(L_X, β)``, ``Act(d, f)``, …).

    Restrictions:

    * arity exactly 1 (single slot, the symmetry is only valid for
      pairings, not for genuine multivector evaluations on multiple
      forms);
    * ``slot_kind="covector"`` (the only side where a flip changes the
      semantics);
    * head ``X`` is 1-vector-like, :class:`Symbol` registered
      :class:`Graded` ``degree=1``, :class:`LieBracketVF`,
      ``Act(Sharp, _)`` (anchor image), :class:`HamiltonianVectorField`;
    * arg ``ω`` resolves under
      :func:`~jacopy.algebra.derivation.degree_of` to ``Degree.const(1)``
      (handles ``Act(d, scalar)``, ``Act(L_X, β)``, registered 1-form
      Symbols, etc.).
    """

    def __init__(self, *, registry: PropertyRegistry) -> None:
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "MultiEvalCovectorPairingFlipDefinition requires a PropertyRegistry"
            )
        self._registry = registry
        self.name = "X(ω) → ω(X)  [arity-1 covector pairing flip]"

    def _is_one_vector_like(self, expr: Expr) -> bool:
        if isinstance(expr, LieBracketVF):
            return True
        if isinstance(expr, HamiltonianVectorField):
            return True
        if isinstance(expr, Act) and isinstance(expr.op, Sharp):
            return True
        graded = self._registry.get(expr, Graded)
        if graded is None:
            return False
        return graded.degree == Degree.const(1)

    def _is_one_form_like(self, expr: Expr) -> bool:
        try:
            return degree_of(expr, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if expr.slot_kind != "covector":
            return False
        if len(expr.args) != 1:
            return False
        if not expr.alternating:
            return False
        if not self._is_one_vector_like(expr.head):
            return False
        return self._is_one_form_like(expr.args[0])

    def rewrite(self, expr: Expr) -> Expr:
        head = expr.head  # 1-vector
        omega = expr.args[0]  # 1-form
        return MultiEval(
            omega, head, alternating=True, slot_kind="vector"
        )


class SnBracketNegLinearityDefinition(Definition):
    r"""``[Neg(a), b]_SN / [a, Neg(b)]_SN → Neg([a, b]_SN)``.

    SN-side mirror of :class:`LieBracketVfNegLinearityDefinition`:
    pulls a ``Neg`` wrapper out of either operand so the SN-bracket
    atom ``BracketApply(sn, ·, ·)`` can keep being recognised by the
    1-vector predicate of
    :class:`SnBracketOfOneVectorsToLieBracketVfDefinition`. Without
    this rule the §3.1.5 (2') residue ends up with
    ``BracketApply(sn, Neg(X_f), V)`` (after the
    ``Act(d̃_π, f) → −X_f`` bridge) and stalls, neither the
    SN→LBVF rule (which gates on structural 1-vector shapes, not
    Neg-wrapped ones) nor the antisymmetry pair-finder can crack it.
    """

    def __init__(
        self,
        *,
        sn_bracket: Optional[SchoutenBracket] = None,
    ) -> None:
        if sn_bracket is not None and not isinstance(sn_bracket, SchoutenBracket):
            raise TypeError(
                "SnBracketNegLinearityDefinition sn_bracket must be a SchoutenBracket"
            )
        self._sn = sn_bracket if sn_bracket is not None else default_sn
        self.name = "[(−a), b]_SN → −[a, b]_SN  /  [a, (−b)]_SN → −[a, b]_SN"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BracketApply):
            return False
        if expr.bracket != self._sn:
            return False
        return isinstance(expr.a, Neg) or isinstance(expr.b, Neg)

    def rewrite(self, expr: Expr) -> Expr:
        a, b = expr.a, expr.b
        sign_neg = False
        if isinstance(a, Neg):
            a = a.arg
            sign_neg = not sign_neg
        if isinstance(b, Neg):
            b = b.arg
            sign_neg = not sign_neg
        new = BracketApply(self._sn, a, b)
        return Neg(new) if sign_neg else new


class TildeDOnScalarToHamiltonianVfDefinition(Definition):
    r"""``Act(d̃_π, f) → Neg(HamiltonianVectorField(f, bivector=π))``.

    Bridges the structural shape ``Act(TildeExteriorDerivative(π), f)``
    (which is ``π♯(d f)`` semantically) to the canonical Hamiltonian
    vector field, picking up a sign for the geometer's convention
    ``X_f = −π♯(d f)`` (sign ``'-'``, the
    :class:`HamiltonianVectorField` default). Without this bridge the
    §3.1.5 (2') residue ends with parallel terms ``ξ([V, X_f]_VF)`` and
    ``ξ([V, d̃_π(f)]_VF)``, both naming the same Lie bracket but in
    incompatible spellings, leaving the engine unable to spot the
    cancellation.

    Scoped to a single Poisson bivector ``π``: the rewrite identifies
    ``X_f`` only when the ``d̃_π`` operator's bivector matches
    ``self._pi``. Untwisted by construction (sign fixed to ``'-'``).
    """

    def __init__(self, pi: Expr) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeDOnScalarToHamiltonianVfDefinition pi must be an Expr"
            )
        self._pi = pi
        self.name = "Act(d̃_π, f) → −X_f  [geometer sign]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        from jacopy.calculus.tilde.operators import TildeExteriorDerivative
        if not isinstance(expr.op, TildeExteriorDerivative):
            return False
        return expr.op.bivector == self._pi

    def rewrite(self, expr: Expr) -> Expr:
        f = expr.arg
        return Neg(HamiltonianVectorField(f, bivector=self._pi))


class HamiltonianVfInnerIotaToMultiEvalDefinition(Definition):
    r"""``X_{Act(ι_X, ω)} → X_{MultiEval(ω, X, "vector")}``.

    Canonicalises the function carried by a
    :class:`~jacopy.calculus.hamiltonian_vf.HamiltonianVectorField` to
    the same scalar shape that the rest of the engine emits. The
    iota-as-scalar bridges (:class:`IotaActAsScalarDefinition` /
    :class:`PermissiveIotaActAsScalarDefinition`) only fire when
    ``Act(ι_X, ω)`` sits inside an outer ``Act(D, _)``; the bare
    ``Act(ι_X, ω)`` carried as the function of an HVF is invisible to
    them. Without this normalisation the (2') residue ends up with
    ``X_{Act(ι_U, η)}`` from one side and ``X_{MultiEval(η, U)}`` from
    the other, algebraically equal but structurally distinct.

    Preserves the bivector / symplectic_form / sign attributes of the
    incoming HVF so the rewrite is sign-stable.
    """

    name = "X_{ι_X(ω)} → X_{ω(X)} [HVF function canonical form]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, HamiltonianVectorField):
            return False
        f = expr.function
        if not (isinstance(f, Act) and isinstance(f.op, InteriorProduct)):
            return False
        return True

    def rewrite(self, expr: Expr) -> Expr:
        f = expr.function
        iota = f.op
        omega = f.arg
        new_f = MultiEval(
            omega, iota.vector_field, alternating=True, slot_kind="vector"
        )
        return HamiltonianVectorField(
            new_f,
            bivector=expr.bivector,
            symplectic_form=expr.symplectic_form,
            sign=expr.sign,
        )


class MultiEvalCovectorDArityOneDefinition(Definition):
    r"""``MultiEval(D, Act(d, f), arity=1, "covector") → Act(D, f)``.

    Covector-slot dual of the arity-1 branch of
    :class:`~jacopy.calculus.intrinsic_axioms.ExteriorDIntrinsicDefinition`
   , the form-side rule only fires on vector-slot evaluations, but the
    §3.1.5 multivector-side residues produce
    ``MultiEval(V, Act(d, scalar), slot_kind="covector")`` shapes after
    L̃ / d̃ intrinsic emission. Routes the same identity ``(d f)(D) =
    D(f)`` through the covector slot.

    Why this is not subsumed by
    :class:`MultiEvalCovectorPairingFlipDefinition`: the flip rule's
    1-form guard uses :func:`degree_of`, which interprets
    ``Act(vf, scalar)`` as a wedge product (degrees add) rather than a
    derivation evaluation. Concretely, ``π♯(ξ)(η(U))`` reads as a
    1-vector under ``degree_of`` (vf:1 + scalar:0 = 1), so
    ``Act(d, π♯(ξ)(η(U)))`` reads as a 2-form, blocking the flip. The
    structural constraint ``head is 1-vector-like`` + ``arg is
    Act(d, _)`` + ``slot_kind="covector"`` + ``arity=1`` forces the
    well-typed reading: the arg must be a 1-form, hence the inner ``f``
    must be a 0-form, hence ``Act(D, f)`` is the directional derivative
    on a scalar, well-defined regardless of how :func:`degree_of`
    classifies the syntactic shape.
    """

    def __init__(self, *, registry: PropertyRegistry) -> None:
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "MultiEvalCovectorDArityOneDefinition requires a PropertyRegistry"
            )
        self._registry = registry
        self.name = "(d f)(D) → D(f)  [arity-1 covector d on scalar]"

    def _is_one_vector_like(self, expr: Expr) -> bool:
        if isinstance(expr, LieBracketVF):
            return True
        if isinstance(expr, HamiltonianVectorField):
            return True
        if isinstance(expr, Act) and isinstance(expr.op, Sharp):
            return True
        graded = self._registry.get(expr, Graded)
        if graded is None:
            return False
        return graded.degree == Degree.const(1)

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if expr.slot_kind != "covector":
            return False
        if len(expr.args) != 1:
            return False
        if not expr.alternating:
            return False
        if not self._is_one_vector_like(expr.head):
            return False
        arg = expr.args[0]
        return isinstance(arg, Act) and isinstance(arg.op, ExteriorDerivative)

    def rewrite(self, expr: Expr) -> Expr:
        D = expr.head
        f = expr.args[0].arg  # inner of Act(d, f)
        return Act(D, f)


class TildeLieOnOneVectorLichnerowiczDefinition(Definition):
    r"""``Act(L̃_α, U) → −X_{α(U)} + π^♯(L_U α) − [U, π^♯ α]_VF`` for 1-form α, 1-vector U.

    Closed-form Lichnerowicz expansion of the tilde-Lie derivative on a
    1-vector. Derived by combining the tilde Cartan magic
    ``L̃_α V = d̃(ι̃_α V) + ι̃_α(d̃ V)`` with the bookkeeping
    identities ``d̃(f) = π^♯(df) = X_f`` (on 0-forms) and
    ``ι̃_η(d̃ U) = π^♯(L_U α) − [U, π^♯ α]_VF`` (Sharp identity for the
    bracket-of-vf case).

    On §3.1.5 identity (1') the residue carries an undecomposed
    ``Act(L̃_η, U)`` inside ``BracketApply([·,·]_SN, ·, V)``. The
    intrinsic L̃ rule fires only on a MultiEval-wrapped head; the
    generic tilde Cartan magic over-decomposes (its ``ι̃_η(d̃ U)``
    branch routes through opaque ``[π, U]_SN`` shapes that the engine
    can't crack on a 1-vf operand without further machinery). This
    rule short-circuits both paths with the analytic closed form.

    Scoped to a Poisson bivector ``π``: matches only when the outer
    head is a :class:`TildeLieDerivative` whose
    :attr:`~jacopy.calculus.tilde.operators.TildeLieDerivative.bivector`
    equals ``π``. Gated on registry-Graded(1) on both the operator's
    form ``α`` and the operand ``U``; structural 1-vector recognisers
    (Sharp-image, HVF, LieBracketVF) also qualify ``U``.
    """

    def __init__(
        self,
        sharp: Sharp,
        *,
        registry: PropertyRegistry,
        lie_derivative=None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError(
                "TildeLieOnOneVectorLichnerowiczDefinition requires a Sharp atom"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "TildeLieOnOneVectorLichnerowiczDefinition requires a PropertyRegistry"
            )
        self._sharp = sharp
        self._pi = sharp.bivector
        self._registry = registry
        self._lie = (
            default_lie_derivative if lie_derivative is None else lie_derivative
        )
        self.name = (
            f"L̃_α(U) = −X_{{α(U)}} + π♯(L_U α) − [U, π♯ α]_VF "
            f"[{self._pi._repr_inner()}]"
        )

    def _is_one_form(self, expr: Expr) -> bool:
        try:
            return degree_of(expr, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def _is_one_vector(self, expr: Expr) -> bool:
        if isinstance(expr, (LieBracketVF, HamiltonianVectorField)):
            return True
        if isinstance(expr, Act) and isinstance(expr.op, Sharp):
            return True
        try:
            return degree_of(expr, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def matches(self, expr: Expr) -> bool:
        from jacopy.calculus.tilde.operators import TildeLieDerivative
        if not isinstance(expr, Act):
            return False
        if not isinstance(expr.op, TildeLieDerivative):
            return False
        if expr.op.bivector != self._pi:
            return False
        if not self._is_one_form(expr.op.form):
            return False
        return self._is_one_vector(expr.arg)

    def rewrite(self, expr: Expr) -> Expr:
        alpha = expr.op.form
        U = expr.arg
        # α(U) in canonical scalar shape (matches the form emitted by
        # HamiltonianVfInnerIotaToMultiEvalDefinition).
        alpha_U = MultiEval(
            alpha, U, alternating=True, slot_kind="vector"
        )
        L_U = self._lie(U)
        return Sum(
            Neg(HamiltonianVectorField(alpha_U, bivector=self._pi)),
            Act(self._sharp, Act(L_U, alpha)),
            Neg(LieBracketVF(U, Act(self._sharp, alpha))),
        )


class SnBracketSumLinearityDefinition(Definition):
    r"""``[Sum(a, b, …), Z]_SN → [a, Z]_SN + [b, Z]_SN + …`` (and right-slot mirror).

    Distributes a ``Sum`` operand through either slot of the
    Schouten-Nijenhuis bracket. The :class:`SchoutenBracket` ``expand``
    method handles wedge products and the four base cases but leaves a
    ``Sum`` operand opaque, the §3.1.5 (1') closure path emits a
    three-term Sum into one slot (from the tilde-Lie Lichnerowicz
    expansion on a 1-vector), so the bracket needs an explicit
    distributor before the SN→LBVF coercion can fire on each piece.
    """

    def __init__(
        self,
        *,
        sn_bracket: Optional[SchoutenBracket] = None,
    ) -> None:
        if sn_bracket is not None and not isinstance(sn_bracket, SchoutenBracket):
            raise TypeError(
                "SnBracketSumLinearityDefinition sn_bracket must be a SchoutenBracket"
            )
        self._sn = sn_bracket if sn_bracket is not None else default_sn
        self.name = "[Sum(a,b,…), Z]_SN → Σ_i [a_i, Z]_SN  (and mirror)"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BracketApply):
            return False
        if expr.bracket != self._sn:
            return False
        return isinstance(expr.a, Sum) or isinstance(expr.b, Sum)

    def rewrite(self, expr: Expr) -> Expr:
        a, b = expr.a, expr.b
        if isinstance(a, Sum):
            return Sum.make(
                *(BracketApply(self._sn, c, b) for c in a.children)
            )
        return Sum.make(
            *(BracketApply(self._sn, a, c) for c in b.children)
        )
