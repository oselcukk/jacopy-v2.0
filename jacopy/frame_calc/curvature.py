r"""
Curvature tensor from a connection — Stage E.

The Riemann curvature tensor of a connection ``∇`` has frame
components

.. math::

    R(\nabla)^a{}_{bcd}
    = e_b(\Gamma^a{}_{cd})
    - e_c(\Gamma^a{}_{bd})
    + \Gamma^e{}_{cd}\,\Gamma^a{}_{be}
    - \Gamma^e{}_{bd}\,\Gamma^a{}_{ce}
    - \gamma^e{}_{bc}\,\Gamma^a{}_{ed}.

The five terms split as: two **derivative** terms, two **product**
terms (Christoffel-on-Christoffel contractions), and one **structure
constant** term (zero for coordinate frames).

The result is antisymmetric in the lower index pair ``(b, c)`` —
``R^a_{bcd} = -R^a_{cbd}``.
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
    ComponentTensor,
)


# --------------------------------------------------------------------- #
# Derivation step record (mirrors KoszulStep from levi_civita.py)        #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CurvatureStep:
    """One step in the per-``(a, b, c, d)`` curvature derivation.

    Same role as :class:`~jacopy.frame_calc.levi_civita.KoszulStep`
    on the Christoffel side: human-readable narration of what the
    curvature formula computed for a single component, ready to be
    lifted into a :class:`~jacopy.proof.chain.ProofChain` by the
    Stage G bridge.
    """

    rule: str
    description: str
    expression: Any = None


# --------------------------------------------------------------------- #
# CurvatureTensor                                                        #
# --------------------------------------------------------------------- #


class CurvatureTensor(ComponentTensor):
    r"""``(1, 3)`` tensor ``R^a_{bcd}`` — Riemann curvature.

    Antisymmetric in ``(b, c)``: ``R^a_{bcd} = -R^a_{cbd}``. Stage E
    exploits this for a ~2× speedup; Stage F's Ricci contraction
    consumes the resulting tensor through
    :meth:`ComponentTensor.contract`.

    Per-entry derivation traces accessible via
    :meth:`derivation_steps`.
    """

    __slots__ = ("_derivations",)

    def __init__(
        self,
        frame: Any,
        components: Any,
        derivations: Dict[
            Tuple[int, int, int, int], List[CurvatureStep]
        ],
    ) -> None:
        super().__init__(frame, components, signature=(1, 3))
        self._derivations = dict(derivations)

    def derivation_steps(
        self, a: int, b: int, c: int, d: int
    ) -> Tuple[CurvatureStep, ...]:
        """Recorded :class:`CurvatureStep`s for ``R^a_{bcd}``.

        Antisymmetry in ``(b, c)`` is honoured: only the canonical
        ordering ``b ≤ c`` is recorded internally; ``b > c`` lookups
        flip the sign in the rendering layer.
        """
        for label, value in (("a", a), ("b", b), ("c", c), ("d", d)):
            if not isinstance(value, int):
                raise TypeError(
                    f"derivation_steps: index {label} must be int"
                )
            if not 0 <= value < self._frame.dim:
                raise IndexError(
                    f"derivation_steps: index {label} = {value} out "
                    f"of range for dim={self._frame.dim}"
                )
        canonical = (a, min(b, c), max(b, c), d)
        steps = self._derivations.get(canonical)
        if steps is None:
            raise KeyError(
                f"No derivation recorded for R^{a}_{{{b}{c}{d}}}"
            )
        return tuple(steps)

    def format_derivation(
        self, a: int, b: int, c: int, d: int, *, indent: str = "  "
    ) -> str:
        """Plain-text format of the derivation for ``R^a_{bcd}``."""
        names = self._frame.index_names()
        title = (
            f"R^{names[a]}_{{{names[b]}{names[c]}{names[d]}}}"
            f"  (Riemann curvature)"
        )
        lines = [title, "─" * len(title)]
        for i, step in enumerate(
            self.derivation_steps(a, b, c, d), start=1
        ):
            lines.append(f"{indent}[{i}] {step.rule}")
            if step.description:
                lines.append(f"{indent}    {step.description}")
            if step.expression is not None:
                lines.append(f"{indent}    = {step.expression}")
        return "\n".join(lines)

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "CurvatureTensor":
        out = object.__new__(CurvatureTensor)
        ComponentTensor.__init__(
            out, self._frame, new_arr, signature=(1, 3)
        )
        out._derivations = dict(self._derivations)
        return out


# --------------------------------------------------------------------- #
# Factory                                                                #
# --------------------------------------------------------------------- #


def curvature(connection: ComponentConnection) -> CurvatureTensor:
    r"""Compute the Riemann curvature of a connection.

    For each ``(a, b, c, d)``::

        R^a_{bcd} = e_b(Γ^a_{cd}) - e_c(Γ^a_{bd})
                    + Γ^e_{cd} Γ^a_{be} - Γ^e_{bd} Γ^a_{ce}
                    - γ^e_{bc} Γ^a_{ed}

    Antisymmetry in ``(b, c)`` is exploited — only ``b < c`` is
    computed (``b == c`` is identically zero); ``b > c`` filled via
    sign flip.

    Parameters
    ----------
    connection
        Any :class:`ComponentConnection` whose components are SymPy
        expressions. The frame's :meth:`derivative` and :meth:`gamma`
        methods are called on those components.

    Returns
    -------
    CurvatureTensor
        Of signature ``(1, 3)``, shape ``(dim,)*4``, with per-entry
        derivation traces.

    Examples
    --------
    Schwarzschild → vacuum curvature non-zero, but Ricci collapses
    (Stage F)::

        R = curvature(levi_civita(g_schwarzschild))
        assert not R.is_zero()        # not flat
    """
    if not isinstance(connection, ComponentConnection):
        raise TypeError(
            "curvature expects a ComponentConnection, got "
            f"{type(connection).__name__}"
        )

    frame = connection.frame
    n = frame.dim
    components = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    derivations: Dict[
        Tuple[int, int, int, int], List[CurvatureStep]
    ] = {}

    for a in range(n):
        for d in range(n):
            for b in range(n):
                for c in range(b + 1, n):  # antisym: b < c only
                    value, steps = _curvature_at(
                        connection, frame, a, b, c, d
                    )
                    components[a, b, c, d] = value
                    components[a, c, b, d] = -value if value != 0 else 0
                    derivations[(a, b, c, d)] = steps
                # b == c gives zero (antisymmetry)
                derivations.setdefault((a, b, b, d), [])

    return CurvatureTensor(frame, components, derivations)


# --------------------------------------------------------------------- #
# Per-entry computation                                                  #
# --------------------------------------------------------------------- #


def _curvature_at(
    connection: ComponentConnection,
    frame: Any,
    a: int,
    b: int,
    c: int,
    d: int,
) -> Tuple[Any, List[CurvatureStep]]:
    """Compute one Riemann entry plus its derivation trace."""
    n = frame.dim
    names = frame.index_names()
    a_n, b_n, c_n, d_n = (names[i] for i in (a, b, c, d))
    steps: List[CurvatureStep] = []

    steps.append(
        CurvatureStep(
            rule="Riemann curvature formula",
            description=(
                f"R^{a_n}_{{{b_n}{c_n}{d_n}}} = "
                f"e_{b_n}(Γ^{a_n}_{{{c_n}{d_n}}}) "
                f"- e_{c_n}(Γ^{a_n}_{{{b_n}{d_n}}}) "
                f"+ Γ^e_{{{c_n}{d_n}}} Γ^{a_n}_{{{b_n}e}} "
                f"- Γ^e_{{{b_n}{d_n}}} Γ^{a_n}_{{{c_n}e}} "
                f"- γ^e_{{{b_n}{c_n}}} Γ^{a_n}_{{e{d_n}}}"
            ),
        )
    )

    # 1-2: derivative terms
    deriv_b = frame.derivative(connection[a, c, d], b)
    deriv_c = frame.derivative(connection[a, b, d], c)
    deriv_diff = deriv_b - deriv_c
    steps.append(
        CurvatureStep(
            rule="Derivative terms",
            description=(
                f"e_{b_n}(Γ^{a_n}_{{{c_n}{d_n}}}) "
                f"- e_{c_n}(Γ^{a_n}_{{{b_n}{d_n}}})"
            ),
            expression=deriv_diff,
        )
    )

    # 3-4: Christoffel-on-Christoffel product terms
    prod_terms = sp.S.Zero
    for e in range(n):
        prod_terms += connection[e, c, d] * connection[a, b, e]
        prod_terms -= connection[e, b, d] * connection[a, c, e]
    steps.append(
        CurvatureStep(
            rule="Product terms",
            description=(
                f"Σ_e (Γ^e_{{{c_n}{d_n}}} Γ^{a_n}_{{{b_n}e}} "
                f"- Γ^e_{{{b_n}{d_n}}} Γ^{a_n}_{{{c_n}e}})"
            ),
            expression=prod_terms,
        )
    )

    # 5: γ-correction term
    gamma_term = sp.S.Zero
    gamma_nonzero = False
    for e in range(n):
        g_struct = frame.gamma(e, b, c)
        if g_struct != 0:
            gamma_nonzero = True
        gamma_term -= g_struct * connection[a, e, d]

    if gamma_nonzero:
        steps.append(
            CurvatureStep(
                rule="γ-correction term",
                description=(
                    f"-Σ_e γ^e_{{{b_n}{c_n}}} Γ^{a_n}_{{e{d_n}}}"
                ),
                expression=gamma_term,
            )
        )
    else:
        steps.append(
            CurvatureStep(
                rule="γ-correction (coordinate frame)",
                description=(
                    "γ^a_{bc} ≡ 0 for a holonomic frame; skipped."
                ),
            )
        )

    raw_value = deriv_diff + prod_terms + gamma_term
    try:
        value = sp.simplify(raw_value)
    except (TypeError, AttributeError):
        value = raw_value

    steps.append(
        CurvatureStep(
            rule="Combine + simplify",
            description=(
                "Sum all five terms and apply sympy.simplify"
            ),
            expression=value,
        )
    )

    return value, steps
