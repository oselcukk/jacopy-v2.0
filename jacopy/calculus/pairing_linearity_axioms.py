r"""
Pairing C∞-linearity axiom, Faz 12.B #12.

Engine rewrite that pulls a scalar factor out of either slot of a
:class:`~jacopy.calculus.pairing.Pairing`:

.. math::

    \langle \alpha, f\,X \rangle \;\longrightarrow\; f \cdot \langle \alpha, X \rangle,
    \qquad
    \langle f\,\alpha, X \rangle \;\longrightarrow\; f \cdot \langle \alpha, X \rangle.

The pairing is C∞-bilinear (a 1-form is a C∞-linear functional on
vector fields; both sides of the pairing are C∞(M)-modules), so the
rule fires whenever either slot is a :class:`~jacopy.core.expr.Product`
of two or more factors. By convention the **last** factor in the slot
product is the underlying covector / vector and the leading factors
are scalars, matching the same split used by
:class:`~jacopy.calculus.lie_rescaling_axioms.LieRescalingDefinition`.

Both slots are unconditional: callers building tutorials over a
classical or algebroid manifold can rely on the rule firing without a
registry. Adding registry-conditioned variants would only be needed
for pairings on non-modular structures, which the current core does
not model.
"""

from __future__ import annotations

from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Expr, Product
from jacopy.proof.expansion import Definition


class PairingScalarPullDefinition(Definition):
    r"""``⟨α, f·X⟩ → f·⟨α, X⟩`` and ``⟨f·α, X⟩ → f·⟨α, X⟩``.

    Fires on a :class:`Pairing` node whenever either child is a
    :class:`Product` with two or more factors. The leading factors of
    that child are folded into the scalar coefficient ``f`` and the
    last factor is taken as the underlying covector / vector. When
    *both* slots are products, the rule pulls from the first slot in a
    given pass; the bottom-up engine walk fires the rule a second time
    on the resulting pairing to catch the other slot.
    """

    name = "Pairing C∞-linearity: ⟨α, f·X⟩ = f·⟨α, X⟩"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        return self._slot_to_pull(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        slot = self._slot_to_pull(expr)
        # _slot_to_pull is consulted from matches(), this branch is
        # unreachable if the engine respects the matches contract, but
        # keep the helper honest for direct callers.
        assert slot is not None
        if slot == "alpha":
            prod: Product = expr.alpha  # type: ignore[assignment]
            *scalars, head = prod.children
            f = Product.make(*scalars)
            return Product(f, Pairing(head, expr.X))
        prod: Product = expr.X  # type: ignore[assignment]
        *scalars, head = prod.children
        f = Product.make(*scalars)
        return Product(f, Pairing(expr.alpha, head))

    @staticmethod
    def _slot_to_pull(expr: Pairing) -> str | None:
        if isinstance(expr.alpha, Product) and len(expr.alpha.children) >= 2:
            return "alpha"
        if isinstance(expr.X, Product) and len(expr.X.children) >= 2:
            return "X"
        return None
