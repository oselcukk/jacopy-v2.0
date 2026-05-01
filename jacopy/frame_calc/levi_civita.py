r"""
Levi-Civita connection from a metric, Stage D.

The :func:`levi_civita` factory takes a :class:`ComponentMetric` and
returns a :class:`LeviCivitaConnection` whose components are the
Christoffel symbols ``Γ^e_{ab}``, computed via the **Koszul formula**:

.. math::

    \Gamma^e{}_{ab}
    \;=\;
    \tfrac{1}{2}\,g^{ec}\!\left(
       e_a(g_{bc}) + e_b(g_{ac}) - e_c(g_{ab})
       - \gamma^d{}_{bc}\,g_{da}
       - \gamma^d{}_{ac}\,g_{db}
       + \gamma^d{}_{ab}\,g_{dc}
    \right).

For a coordinate frame the structure constants ``γ^d_{bc}`` vanish and
the formula reduces to the textbook three-term form. The
implementation handles the general case unconditionally, the γ terms
just evaluate to zero when the frame is holonomic.

Each entry's derivation is recorded as a list of
:class:`KoszulStep` records; users access the trace via
:meth:`LeviCivitaConnection.derivation_steps` (and, in Stage G, the
:class:`~jacopy.proof.chain.ProofChain`-rendered
:meth:`derivation_chain`).

Concrete frames only at this stage. :class:`AbstractFrame` raises
:class:`NotImplementedError` until the symbolic ``g^{ab}`` opaque-atom
path is filled in (Stage D follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ComponentMetricInverse,
)
from jacopy.frame_calc.frame import (
    AbstractFrame,
    CoordinateFrame,
    Frame,
    Tetrad,
)


# --------------------------------------------------------------------- #
# Per-entry derivation step records                                     #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class KoszulStep:
    """One step in the per-``(e, a, b)`` Koszul-formula derivation.

    Stage D stores derivation traces as a list of these records, one
    list per Christoffel entry. Stage G will provide a bridge that
    lifts the list to a :class:`~jacopy.proof.chain.ProofChain` for
    rendering via the existing
    :func:`~jacopy.display.chain_to_latex_document` machinery.

    Attributes
    ----------
    rule
        Short label naming the step (``"Koszul formula"``,
        ``"Frame derivatives"``, ``"Multiply by ½ g^{ec}"``,
        ``"Simplify"``, …).
    description
        Human-readable expansion of the rule, parameterised by the
        ``(e, a, b)`` triple of the entry under derivation.
    expression
        The intermediate expression at this step. ``None`` for early
        narration steps that don't carry a numeric form yet.
    """

    rule: str
    description: str
    expression: Any = None


# --------------------------------------------------------------------- #
# LeviCivitaConnection                                                  #
# --------------------------------------------------------------------- #


class LeviCivitaConnection(ComponentConnection):
    r"""``Γ(^g∇)^e_{ab}`` computed via the Koszul formula.

    Carries the same component data as a :class:`ComponentConnection`
    plus a per-entry derivation trace accessible via
    :meth:`derivation_steps`. Equality / hashing inherit from the
    base class, two Levi-Civita connections compare equal iff their
    components match (the trace metadata is presentation-only).

    The :attr:`optimized` attribute records whether this connection
    was built via the fast path (no per-entry simplify, no derivation
    traces). In optimized mode :meth:`derivation_steps` raises
    :class:`RuntimeError`.
    """

    __slots__ = ("_derivations", "_optimized")

    def __init__(
        self,
        frame: Frame,
        christoffel: Any,
        derivations: Dict[Tuple[int, int, int], List[KoszulStep]],
        *,
        optimized: bool = False,
    ) -> None:
        super().__init__(frame, christoffel)
        self._derivations = dict(derivations)
        self._optimized = bool(optimized)

    @property
    def optimized(self) -> bool:
        """``True`` if this connection was built via the fast path."""
        return self._optimized

    def derivation_steps(
        self, e: int, a: int, b: int
    ) -> Tuple[KoszulStep, ...]:
        """Return the recorded :class:`KoszulStep`s for ``Γ^e_{ab}``.

        For symmetric pairs ``(e, a, b)`` and ``(e, b, a)`` the same
        underlying derivation is returned, the connection is
        torsion-free, so only the canonical ordering ``a ≤ b`` is
        recorded internally.

        Raises
        ------
        RuntimeError
            If this connection was built with ``optimized=True``;
            derivation traces are not recorded in the fast path.
        """
        if self._optimized:
            raise RuntimeError(
                "derivation_steps unavailable: this connection was "
                "built with optimized=True (fast path). Rebuild via "
                "levi_civita(g) without the optimized flag to record "
                "per-entry KoszulStep traces."
            )
        for label, value in (("e", e), ("a", a), ("b", b)):
            if not isinstance(value, int):
                raise TypeError(
                    f"derivation_steps: index {label} must be int"
                )
            if not 0 <= value < self._frame.dim:
                raise IndexError(
                    f"derivation_steps: index {label} = {value} out of "
                    f"range for dim={self._frame.dim}"
                )
        # Honour symmetry: stored under (e, min(a,b), max(a,b)).
        canonical = (e, min(a, b), max(a, b))
        steps = self._derivations.get(canonical)
        if steps is None:
            raise KeyError(
                f"No derivation recorded for Γ^{e}_{{{a}{b}}}"
            )
        return tuple(steps)

    def derivation_chain(
        self, e: int, a: int, b: int
    ) -> "ProofChain":  # noqa: F821, string form to avoid import cycle
        r"""Return a :class:`~jacopy.proof.chain.ProofChain` for ``Γ^e_{ab}``.

        The chain wraps the recorded :class:`KoszulStep`s into
        :class:`~jacopy.proof.step.ProofStep`s tagged
        ``provenance_tag="computation"``, ready for paper-grade
        LaTeX output via
        :func:`~jacopy.display.chain_to_latex.chain_to_latex_document`.

        Raises :class:`RuntimeError` in optimized mode.
        """
        from jacopy.frame_calc.proof_bridge import steps_to_proof_chain

        steps = self.derivation_steps(e, a, b)  # raises if optimized
        names = self._frame.index_names()
        head = (
            f"Γ^{names[e]}_{{{names[a]}{names[b]}}} via Koszul formula"
        )
        return steps_to_proof_chain(steps, head_label=head)

    def format_derivation(
        self, e: int, a: int, b: int, *, indent: str = "  "
    ) -> str:
        """Format the per-entry derivation as plain text.

        Useful for terminal / Jupyter prints. Stage G will produce
        a publication-grade LaTeX render through ProofChain.

        Raises :class:`RuntimeError` in optimized mode (no traces
        recorded).
        """
        if self._optimized:
            raise RuntimeError(
                "format_derivation unavailable in optimized mode."
            )
        names = self._frame.index_names()
        title = (
            f"Γ^{names[e]}_{{{names[a]}{names[b]}}}  "
            f"(via Koszul formula)"
        )
        lines = [title, "─" * len(title)]
        for i, step in enumerate(self.derivation_steps(e, a, b), start=1):
            lines.append(f"{indent}[{i}] {step.rule}")
            if step.description:
                lines.append(f"{indent}    {step.description}")
            if step.expression is not None:
                lines.append(f"{indent}    = {step.expression}")
        return "\n".join(lines)

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "LeviCivitaConnection":
        # Preserve derivation table on simplify; Stage F may want to
        # overwrite this with re-derived chains in foundational mode.
        out = object.__new__(LeviCivitaConnection)
        ComponentConnection.__init__(out, self._frame, new_arr)
        out._derivations = dict(self._derivations)
        out._optimized = self._optimized
        return out


# --------------------------------------------------------------------- #
# Factory + Koszul formula                                              #
# --------------------------------------------------------------------- #


def levi_civita(
    g: ComponentMetric, *, optimized: bool = False
) -> LeviCivitaConnection:
    r"""Compute the Levi-Civita connection from a metric ``g``.

    Returns a :class:`LeviCivitaConnection` whose ``[e, a, b]`` entry
    is ``Γ^e_{ab}`` from the Koszul formula. Symmetry of the
    connection (torsion-free: ``Γ^e_{ab} = Γ^e_{ba}``) is exploited,
    only the canonical ordering ``a ≤ b`` is computed; the mirror
    is filled in directly.

    Parameters
    ----------
    g
        A symmetric :class:`ComponentMetric`.
    optimized
        When ``False`` (default), the full pipeline runs:
        every Christoffel entry is :func:`sympy.simplify`-d to a
        clean form, and per-entry :class:`KoszulStep` derivation
        traces are recorded for paper-grade transcript output.

        When ``True``, the **fast path** runs: no per-entry
        ``simplify``, no derivation traces. Use for production
        computations where the bottleneck is mid-formula
        simplification on complex metrics (e.g. Kerr-class).
        :meth:`LeviCivitaConnection.derivation_steps` will raise
        :class:`RuntimeError` for entries computed in optimized
        mode. Components remain mathematically correct, they are
        just stored in raw, unsimplified form. Apply
        :func:`sympy.simplify` (or
        :meth:`ComponentTensor.simplify`) at user-access time when a
        clean form is needed.

    Returns
    -------
    LeviCivitaConnection
        Christoffel-symbol components. With ``optimized=False`` the
        connection also carries per-entry derivation traces.

    Examples
    --------
    Default (full transparency, slower for complex metrics)::

        LC = levi_civita(g)
        LC.derivation_steps(0, 1, 1)   # full Koszul-formula trace

    Optimized (fast path, no traces)::

        LC_fast = levi_civita(g, optimized=True)
        LC_fast[0, 1, 1]                # raw expression
        sp.simplify(LC_fast[0, 1, 1])   # clean form on demand
    """
    if not isinstance(g, ComponentMetric):
        raise TypeError(
            "levi_civita expects a ComponentMetric, got "
            f"{type(g).__name__}"
        )
    frame = g.frame
    if isinstance(frame, AbstractFrame):
        raise NotImplementedError(
            "levi_civita on AbstractFrame requires polymorphic "
            "arithmetic that mixes jacopy Expr atoms and SymPy "
            "expressions in a single formula. The opaque "
            "InverseMetricEntryExpr atom for g^{ab} landed in this "
            "stage, but the full pipeline (jacopy Sum / division by "
            "Rational, mixed-type contraction) is non-trivial, "
            "deferred to a dedicated abstract-mode pass."
        )
    if not isinstance(frame, (CoordinateFrame, Tetrad)):
        raise TypeError(
            f"levi_civita: unsupported frame type {type(frame).__name__}"
        )

    n = frame.dim
    g_inv = g.inverse()

    christoffel = sp.MutableDenseNDimArray.zeros(n, n, n)
    derivations: Dict[Tuple[int, int, int], List[KoszulStep]] = {}

    for e in range(n):
        for a in range(n):
            for b in range(a, n):  # symmetry: only a ≤ b
                if optimized:
                    value = _koszul_at_optimized(
                        g, g_inv, frame, e, a, b
                    )
                else:
                    value, steps = _koszul_at(
                        g, g_inv, frame, e, a, b
                    )
                    derivations[(e, a, b)] = steps
                christoffel[e, a, b] = value
                if a != b:
                    christoffel[e, b, a] = value

    return LeviCivitaConnection(
        frame, christoffel, derivations, optimized=optimized
    )


# --------------------------------------------------------------------- #
# Per-entry computation                                                  #
# --------------------------------------------------------------------- #


def _koszul_at(
    g: ComponentMetric,
    g_inv: ComponentMetricInverse,
    frame: Frame,
    e: int,
    a: int,
    b: int,
) -> Tuple[Any, List[KoszulStep]]:
    """Compute one Christoffel entry plus its derivation trace.

    Returns ``(value, steps)`` where ``value`` is the simplified
    SymPy expression and ``steps`` is a list of
    :class:`KoszulStep` records narrating the computation.
    """
    n = frame.dim
    names = frame.index_names()
    e_name = names[e]
    a_name = names[a]
    b_name = names[b]

    steps: List[KoszulStep] = []

    # 1. Open the Koszul formula
    steps.append(
        KoszulStep(
            rule="Koszul formula",
            description=(
                f"2 g(∇_{{e_{a_name}}} e_{b_name}, e_{e_name}) = "
                f"e_{a_name}(g_{{{b_name}c}}) "
                f"+ e_{b_name}(g_{{{a_name}c}}) "
                f"- e_c(g_{{{a_name}{b_name}}}) "
                f"- γ^d_{{{b_name}c}} g_{{d{a_name}}} "
                f"- γ^d_{{{a_name}c}} g_{{d{b_name}}} "
                f"+ γ^d_{{{a_name}{b_name}}} g_{{dc}}"
            ),
        )
    )

    # 2. Frame-derivative terms (one per c)
    deriv_terms_per_c: List[Any] = []
    for c in range(n):
        d_a = frame.derivative(g[b, c], a)  # e_a(g_{bc})
        d_b = frame.derivative(g[a, c], b)  # e_b(g_{ac})
        d_c = frame.derivative(g[a, b], c)  # e_c(g_{ab})
        deriv_terms_per_c.append(d_a + d_b - d_c)

    steps.append(
        KoszulStep(
            rule="Frame-derivative terms",
            description=(
                f"For each c: e_{a_name}(g_{{{b_name}c}}) "
                f"+ e_{b_name}(g_{{{a_name}c}}) "
                f"- e_c(g_{{{a_name}{b_name}}})"
            ),
            expression=tuple(deriv_terms_per_c),
        )
    )

    # 3. γ correction terms (zero for coordinate frames; included
    # unconditionally so Tetrad / non-holonomic frames work later).
    gamma_nonzero = False
    gamma_terms_per_c: List[Any] = []
    for c in range(n):
        gt = sp.S.Zero
        for d in range(n):
            gt -= frame.gamma(d, b, c) * g[d, a]
            gt -= frame.gamma(d, a, c) * g[d, b]
            gt += frame.gamma(d, a, b) * g[d, c]
        gamma_terms_per_c.append(gt)
        if gt != 0:
            gamma_nonzero = True

    if gamma_nonzero:
        steps.append(
            KoszulStep(
                rule="γ correction terms",
                description=(
                    "For each c: -γ^d_{bc} g_{da} - γ^d_{ac} g_{db} "
                    "+ γ^d_{ab} g_{dc} (sum over d)"
                ),
                expression=tuple(gamma_terms_per_c),
            )
        )
    else:
        # For coordinate frames, narrate that γ ≡ 0 so the user knows
        # why these terms dropped.
        steps.append(
            KoszulStep(
                rule="γ correction (coordinate frame)",
                description="γ^a_{bc} ≡ 0 for a holonomic frame; skipped.",
            )
        )

    # 4. Multiply by ½ g^{ec} and contract over c
    contraction = sp.S.Zero
    for c in range(n):
        inner = deriv_terms_per_c[c] + gamma_terms_per_c[c]
        contraction += g_inv[e, c] * inner
    raw_value = contraction / 2
    steps.append(
        KoszulStep(
            rule="Multiply by ½ g^{ec} and contract over c",
            description=(
                f"Γ^{e_name}_{{{a_name}{b_name}}} = ½ Σ_c g^{{{e_name}c}}"
                f" · (frame-deriv + γ-terms)"
            ),
            expression=raw_value,
        )
    )

    # 5. Simplify
    try:
        value = sp.simplify(raw_value)
    except (TypeError, AttributeError):
        value = raw_value
    steps.append(
        KoszulStep(
            rule="Simplify",
            description="sympy.simplify",
            expression=value,
        )
    )

    return value, steps


def _koszul_at_optimized(
    g: ComponentMetric,
    g_inv: ComponentMetricInverse,
    frame: Frame,
    e: int,
    a: int,
    b: int,
) -> Any:
    """Fast-path Koszul: no per-step trace, no per-entry simplify.

    Returns the **raw** Koszul-formula result for ``Γ^e_{ab}``. The
    expression is mathematically correct but not simplified, call
    :func:`sympy.simplify` (or :meth:`ComponentTensor.simplify`) at
    the user-access layer when a clean form is desired.

    Used by :func:`levi_civita` when ``optimized=True``.
    """
    n = frame.dim
    contraction = sp.S.Zero
    for c in range(n):
        # Three frame-derivative terms, summed:
        d_a = frame.derivative(g[b, c], a)
        d_b = frame.derivative(g[a, c], b)
        d_c = frame.derivative(g[a, b], c)
        deriv = d_a + d_b - d_c

        # γ-correction (zero for coordinate frames, but we evaluate
        # to keep the path uniform across frame types).
        gamma_term = sp.S.Zero
        for d in range(n):
            gamma_term -= frame.gamma(d, b, c) * g[d, a]
            gamma_term -= frame.gamma(d, a, c) * g[d, b]
            gamma_term += frame.gamma(d, a, b) * g[d, c]

        contraction += g_inv[e, c] * (deriv + gamma_term)

    return contraction / 2
