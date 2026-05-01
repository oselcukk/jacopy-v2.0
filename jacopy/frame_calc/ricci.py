r"""
Ricci tensor + Ricci scalar, Stage F.

The Ricci tensor is the contraction of the Riemann curvature on
the upper index against the second lower index:

.. math::

    \operatorname{Ric}(\nabla)_{ab} \;=\; R(\nabla)^c{}_{acb}.

The Ricci scalar is the further metric trace:

.. math::

    \mathcal{R}(\nabla, g) \;=\; \operatorname{Ric}(\nabla)_{ab}\,g^{ab}.

Both quantities follow directly from
:func:`~jacopy.frame_calc.curvature.curvature`. This module exposes
them as high-level entry points (:func:`ricci`,
:func:`ricci_scalar`) plus from-existing-tensor helpers for callers
who already have the curvature tensor and want to avoid recomputing
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc

from jacopy.frame_calc.component_tensor import (
    ComponentConnection,
    ComponentMetric,
    ComponentTensor,
)
from jacopy.frame_calc.curvature import CurvatureTensor, curvature


# --------------------------------------------------------------------- #
# Step records                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class RicciStep:
    """One step in a Ricci-tensor entry derivation."""

    rule: str
    description: str
    expression: Any = None


# --------------------------------------------------------------------- #
# RicciTensor                                                           #
# --------------------------------------------------------------------- #


class RicciTensor(ComponentTensor):
    r"""``(0, 2)`` symmetric tensor ``Ric_{ab}`` from a contraction
    of the Riemann curvature.
    """

    __slots__ = ("_derivations", "_optimized")

    def __init__(
        self,
        frame: Any,
        components: Any,
        derivations: Dict[Tuple[int, int], List[RicciStep]] | None = None,
        *,
        optimized: bool = False,
    ) -> None:
        super().__init__(frame, components, signature=(0, 2))
        self._derivations = (
            dict(derivations) if derivations is not None else {}
        )
        self._optimized = bool(optimized)

    @property
    def optimized(self) -> bool:
        """``True`` if this Ricci tensor was built via the fast path."""
        return self._optimized

    def derivation_steps(
        self, a: int, b: int
    ) -> Tuple[RicciStep, ...]:
        """Recorded :class:`RicciStep`s for ``Ric_{ab}``.

        Raises :class:`RuntimeError` in optimized mode (no traces
        recorded).
        """
        if self._optimized:
            raise RuntimeError(
                "derivation_steps unavailable: this Ricci tensor was "
                "built with optimized=True."
            )
        for label, value in (("a", a), ("b", b)):
            if not isinstance(value, int):
                raise TypeError(
                    f"derivation_steps: index {label} must be int"
                )
            if not 0 <= value < self._frame.dim:
                raise IndexError(
                    f"derivation_steps: index {label} = {value} "
                    f"out of range for dim={self._frame.dim}"
                )
        canonical = (min(a, b), max(a, b))
        steps = self._derivations.get(canonical)
        if steps is None:
            raise KeyError(
                f"No derivation recorded for Ric_{{{a}{b}}}"
            )
        return tuple(steps)

    def derivation_chain(
        self, a: int, b: int
    ) -> "ProofChain":  # noqa: F821
        r"""Return a :class:`~jacopy.proof.chain.ProofChain` for ``Ric_{ab}``.

        Raises :class:`RuntimeError` in optimized mode.
        """
        from jacopy.frame_calc.proof_bridge import steps_to_proof_chain

        steps = self.derivation_steps(a, b)
        names = self._frame.index_names()
        head = f"Ric_{{{names[a]}{names[b]}}} (Ricci contraction)"
        return steps_to_proof_chain(steps, head_label=head)

    def format_derivation(
        self, a: int, b: int, *, indent: str = "  "
    ) -> str:
        if self._optimized:
            raise RuntimeError(
                "format_derivation unavailable in optimized mode."
            )
        names = self._frame.index_names()
        title = f"Ric_{{{names[a]}{names[b]}}}  (Ricci tensor)"
        lines = [title, "─" * len(title)]
        for i, step in enumerate(
            self.derivation_steps(a, b), start=1
        ):
            lines.append(f"{indent}[{i}] {step.rule}")
            if step.description:
                lines.append(f"{indent}    {step.description}")
            if step.expression is not None:
                lines.append(f"{indent}    = {step.expression}")
        return "\n".join(lines)

    def matrix(self) -> sp.Matrix:
        """Return the Ricci tensor as a SymPy ``Matrix``."""
        return sp.Matrix(self._components.tolist())

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "RicciTensor":
        out = object.__new__(RicciTensor)
        ComponentTensor.__init__(
            out, self._frame, new_arr, signature=(0, 2)
        )
        out._derivations = dict(self._derivations)
        out._optimized = self._optimized
        return out


# --------------------------------------------------------------------- #
# Ricci factories                                                       #
# --------------------------------------------------------------------- #


def ricci(
    connection: ComponentConnection, *, optimized: bool = False
) -> RicciTensor:
    r"""Compute the Ricci tensor from a connection.

    Internally calls :func:`~jacopy.frame_calc.curvature.curvature`
    and contracts via :meth:`ComponentTensor.contract` on the
    canonical Ricci index pattern ``R^c_{acb}``.

    Sign convention
    ---------------
    Following the operator definition
    ``R(U, V) W := ∇_U∇_V W − ∇_V∇_U W − ∇_{[U,V]} W`` and the
    contraction ``Ric_{ab} := R^c_{acb}``, the Ricci tensor on a
    constant-curvature space carries the **opposite** sign of the
    common Wald / Carroll physics convention. Concretely, on a
    round 2-sphere of radius ``R₀``::

        Ric = −(1/R₀²) g
        ricci_scalar = −2/R₀²

    Schwarzschild's vacuum identity ``G_{ab} = 0`` is unaffected,
    both ``Ric`` and ``½ R g`` flip sign together in the definition
    of the Einstein tensor.
    """
    if not isinstance(connection, ComponentConnection):
        raise TypeError(
            "ricci expects a ComponentConnection, got "
            f"{type(connection).__name__}"
        )
    R = curvature(connection, optimized=optimized)
    return ricci_from_curvature(R, optimized=optimized)


def ricci_from_curvature(
    R: CurvatureTensor, *, optimized: bool = False
) -> RicciTensor:
    r"""Build the Ricci tensor by contracting an existing curvature.

    Avoids recomputing the curvature when the caller already has it.
    Contracts ``R^c_{acb}``, i.e. position 0 (upper) with position 2
    (the second lower index). The ``optimized`` flag is propagated
    forward, if the input curvature was built optimized, set
    ``optimized=True`` here too to skip the per-entry derivation
    trace.
    """
    if not isinstance(R, CurvatureTensor):
        raise TypeError(
            "ricci_from_curvature expects a CurvatureTensor, got "
            f"{type(R).__name__}"
        )

    contracted = R.contract(upper=0, lower=2)
    frame = R.frame
    n = frame.dim
    derivations: Dict[Tuple[int, int], List[RicciStep]] = {}

    if not optimized:
        names = frame.index_names()
        for a in range(n):
            for b in range(a, n):
                steps: List[RicciStep] = [
                    RicciStep(
                        rule="Ricci contraction",
                        description=(
                            f"Ric_{{{names[a]}{names[b]}}} = "
                            f"R^c_{{{names[a]} c {names[b]}}}"
                        ),
                    ),
                    RicciStep(
                        rule="Sum over c",
                        description=(
                            f"Σ_{{c=0..{n-1}}} R^c_{{{names[a]} c {names[b]}}}"
                        ),
                        expression=contracted[a, b],
                    ),
                ]
                derivations[(a, b)] = steps

    return RicciTensor(
        frame, contracted.components, derivations, optimized=optimized
    )


# --------------------------------------------------------------------- #
# Ricci scalar                                                          #
# --------------------------------------------------------------------- #


def ricci_scalar(
    connection: ComponentConnection,
    g: ComponentMetric,
    *,
    optimized: bool = False,
) -> Any:
    r"""Compute the Ricci scalar from a connection and metric.

    Convenience shortcut that runs the full pipeline:

        ``connection → curvature → Ricci → contract with g^{ab}``.
    """
    if not isinstance(connection, ComponentConnection):
        raise TypeError(
            "ricci_scalar expects a ComponentConnection, got "
            f"{type(connection).__name__}"
        )
    if not isinstance(g, ComponentMetric):
        raise TypeError(
            "ricci_scalar expects a ComponentMetric, got "
            f"{type(g).__name__}"
        )
    Ric = ricci(connection, optimized=optimized)
    return ricci_scalar_from_ricci(Ric, g, optimized=optimized)


def ricci_scalar_from_ricci(
    Ric: RicciTensor,
    g: ComponentMetric,
    *,
    optimized: bool = False,
) -> Any:
    r"""``R = Ric_{ab} g^{ab}`` from existing Ricci and metric.

    Both tensors must live on the same frame.
    """
    if not isinstance(Ric, RicciTensor):
        raise TypeError(
            "ricci_scalar_from_ricci expects a RicciTensor, got "
            f"{type(Ric).__name__}"
        )
    if not isinstance(g, ComponentMetric):
        raise TypeError(
            "ricci_scalar_from_ricci expects a ComponentMetric, got "
            f"{type(g).__name__}"
        )
    if Ric.frame != g.frame:
        raise ValueError(
            "ricci_scalar_from_ricci: Ric and g must share a frame"
        )

    g_inv = g.inverse()
    n = Ric.frame.dim
    s: Any = sp.S.Zero
    for a in range(n):
        for b in range(n):
            s += g_inv[a, b] * Ric[a, b]
    if not optimized:
        try:
            s = sp.simplify(s)
        except (TypeError, AttributeError):
            pass
    return s
