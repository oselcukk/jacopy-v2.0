"""
Courant bracket on sections of ``TM ⊕ T*M``.

The Courant bracket is the skew-symmetric cousin of the Dorfman bracket.
On section pairs ``(X, α), (Y, β)`` it is

    [(X, α), (Y, β)]_C = ( [X, Y],
                           L_X β − L_Y α − ½ d(ι_X β − ι_Y α) ).

Compared to :class:`~jacopy.brackets.dorfman.DorfmanBracket`:

* **Courant is graded antisymmetric**; Dorfman is not.
* **Dorfman satisfies Leibniz**; Courant does not.
* The two agree modulo an exact correction term: their difference is
  ``½ d(ι_X β + ι_Y α)``. This module doesn't encode the bridge
  explicitly (deferred to Faz 9 library); users who want to compare
  them assemble the correction by hand.

Courant's graded Jacobi is *conditional*. In the plain case it holds
exactly, the "Courant algebroid Jacobi" is the identity on
``(TM ⊕ T*M, [·,·]_C)``. In the H-twisted variant (pass
``background_H=H`` to the constructor) the form part gains a
contraction correction ``ι_Y ι_X H``, and Jacobi now holds iff ``H``
is *closed*: ``dH = 0``. The helper :meth:`jacobi_condition` returns
the corresponding :class:`VanishingCondition`.

Implementation mirrors :class:`DorfmanBracket`: operands are
:class:`~jacopy.brackets.dorfman.SectionPair` instances, the vector
half is routed through a caller-supplied vector bracket (default Lie),
and the form half is assembled from the standard Cartan operators
(``d``, ``L``, ``ι``). The ``background_H`` kwarg is optional and
defaults to ``None``, when absent, the bracket is the plain Courant
bracket with unconditional (exact) Jacobi behaviour in the sense of a
Courant algebroid.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.dorfman import SectionPair
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior as default_interior
from jacopy.calculus.lie_derivative import lie_derivative as default_lie_derivative
from jacopy.core.expr import Expr, Neg, Product, Rational, Sum
from jacopy.core.registry import PropertyRegistry


LieDerivativeFactory = Callable[[Expr], Derivation]
InteriorFactory = Callable[[Expr], Derivation]


class CourantBracket(GradedBracket):
    """Courant bracket on :class:`SectionPair` operands.

    Parameters
    ----------
    name
        Display name; defaults to ``"[·,·]_C"``.
    vector_bracket
        Bracket on the vector-field halves. Defaults to
        :class:`LieBracket`.
    d, lie_derivative, interior
        Cartan operators. Default to the smooth-manifold singletons.
    background_H
        Optional closed 3-form ``H``. When supplied, the bracket is
        the *H-twisted* Courant bracket: the form component gains
        ``ι_Y ι_X H`` and graded Jacobi holds iff ``dH = 0`` (exposed
        by :meth:`jacobi_condition`). ``None`` (default) gives the
        plain Courant bracket.

    Notes
    -----
    * Degree 0; graded-antisymmetric; Leibniz fails (the Courant
      side of the Dorfman/Courant dichotomy). Jacobi is reported as
      ``None``, "conditional", because the H-twisted case needs
      ``dH = 0`` and even the untwisted case is more cleanly
      expressed as "Courant algebroid Jacobi holds" at the proof
      layer rather than as an unconditional flag.
    """

    def __init__(
        self,
        name: str = "[·,·]_C",
        *,
        vector_bracket: Optional[GradedBracket] = None,
        d: Optional[Derivation] = None,
        lie_derivative: Optional[LieDerivativeFactory] = None,
        interior: Optional[InteriorFactory] = None,
        background_H: Optional[Expr] = None,
    ) -> None:
        if background_H is not None and not isinstance(background_H, Expr):
            raise TypeError(
                f"CourantBracket background_H must be an Expr, "
                f"got {type(background_H).__name__}"
            )
        # Courant is the skew twin of Dorfman, antisymmetric holds,
        # Leibniz fails. Jacobi is tracked as conditional (None) so the
        # proof layer is the one that discharges it, exactly against
        # dH = 0 in the H-twisted case.
        super().__init__(
            name,
            degree=0,
            is_graded_antisymmetric=True,
            satisfies_leibniz=False,
            satisfies_graded_jacobi=None,
        )
        self._vector_bracket = (
            vector_bracket if vector_bracket is not None else LieBracket()
        )
        self._d = d if d is not None else default_d
        self._lie_derivative = (
            lie_derivative if lie_derivative is not None else default_lie_derivative
        )
        self._interior = interior if interior is not None else default_interior
        self._background_H = background_H

    @property
    def vector_bracket(self) -> GradedBracket:
        return self._vector_bracket

    @property
    def background_H(self) -> Optional[Expr]:
        return self._background_H

    @property
    def is_twisted(self) -> bool:
        return self._background_H is not None

    # ---- expansion -------------------------------------------------- #

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``[(X, α), (Y, β)]_C``, classical (and H-twisted) Courant.

        Form part assembled as ``L_X β − L_Y α − ½ d(ι_X β − ι_Y α)``,
        plus ``ι_Y ι_X H`` when ``background_H`` is set. Raises
        :class:`TypeError` if either operand is not a
        :class:`SectionPair`.
        """
        if not isinstance(a, SectionPair):
            raise TypeError(
                f"CourantBracket left operand must be a SectionPair, "
                f"got {type(a).__name__}"
            )
        if not isinstance(b, SectionPair):
            raise TypeError(
                f"CourantBracket right operand must be a SectionPair, "
                f"got {type(b).__name__}"
            )
        X, alpha = a.vector, a.form
        Y, beta = b.vector, b.form

        vector_part = self._vector_bracket.expand(X, Y, registry)

        L_X = self._lie_derivative(X)
        L_Y = self._lie_derivative(Y)
        iota_X = self._interior(X)
        iota_Y = self._interior(Y)
        half = Rational(1, 2)

        # ½ d(ι_X β − ι_Y α). Building the inner Sum then wrapping Neg
        # and the ½ factor keeps the shape visible to the proof layer.
        iota_diff = Sum(Act(iota_X, beta), Neg(Act(iota_Y, alpha)))
        half_d_correction = Product(half, Act(self._d, iota_diff))

        form_terms = [
            Act(L_X, beta),
            Neg(Act(L_Y, alpha)),
            Neg(half_d_correction),
        ]

        if self._background_H is not None:
            # H-twist: ``+ ι_Y ι_X H``. No sign on this term, the
            # classical convention is that the twist is additive.
            form_terms.append(Act(iota_Y, Act(iota_X, self._background_H)))

        form_part = Sum(*form_terms)
        return SectionPair(vector_part, form_part)

    # ---- Jacobi condition ------------------------------------------ #

    def jacobi_condition(
        self,
        registry: Optional[PropertyRegistry] = None,
    ) -> VanishingCondition:
        """Return the :class:`VanishingCondition` for Courant Jacobi.

        * **Untwisted** (``background_H is None``): the obstruction is
          :class:`Zero`-valued, Courant algebroid Jacobi holds
          unconditionally, and the condition is reported as vacuous.
        * **H-twisted**: the obstruction is ``dH``. Classical result:
          the H-twisted Courant bracket satisfies graded Jacobi iff
          the twist 3-form is closed.
        """
        if self._background_H is None:
            from jacopy.core.expr import Zero
            return VanishingCondition(
                obstruction=Zero,
                name="Courant Jacobi (untwisted, vacuous)",
            )
        obstruction = Act(self._d, self._background_H)
        return VanishingCondition(
            obstruction=obstruction,
            name=(
                f"Courant Jacobi condition "
                f"(H-twisted by {self._background_H._repr_inner()})"
            ),
        )

    # ---- identity -------------------------------------------------- #

    def _identity_key(self) -> Any:
        return super()._identity_key() + (
            self._vector_bracket,
            self._d,
            self._lie_derivative,
            self._interior,
            self._background_H,
        )
