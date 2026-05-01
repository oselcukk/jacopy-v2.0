r"""
Coordinate transformation helper, Faz 19 Chunk E.

Pull a metric back from one coordinate chart to another via the
Jacobian of a user-supplied diffeomorphism:

.. math::

    g'_{ab}(x') = \frac{\partial x^c}{\partial x'^a}
                  \frac{\partial x^d}{\partial x'^b}\,
                  g_{cd}\!\bigl(x(x')\bigr).

The helper is **fully general**, no built-in registry of "named"
coordinate systems. The caller supplies a dict mapping each old
coordinate symbol to a SymPy expression in the new coordinates, and
the helper handles the Jacobian + substitution.

Example: Cartesian to spherical on flat 3D
-------------------------------------------

.. code-block:: python

    import sympy as sp
    from jacopy.frame_calc import CoordinateFrame, ComponentMetric
    from jacopy.frame_calc.coord_transform import transform_metric

    x, y, z = sp.symbols("x y z", real=True)
    F_cart = CoordinateFrame([x, y, z])
    g_cart = ComponentMetric(F_cart, sp.eye(3))

    r, theta, phi = sp.symbols("r theta phi", positive=True)
    F_sph, g_sph = transform_metric(
        g_cart, [r, theta, phi],
        {x: r * sp.sin(theta) * sp.cos(phi),
         y: r * sp.sin(theta) * sp.sin(phi),
         z: r * sp.cos(theta)},
    )
    # g_sph.matrix() == diag(1, r**2, r**2 * sin(theta)**2)
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def transform_metric(
    g: ComponentMetric,
    new_coords: Sequence[sp.Symbol],
    transform_map: Mapping[sp.Symbol, sp.Expr],
    *,
    name: Optional[str] = None,
    simplify: bool = True,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Pull a metric back to a new coordinate chart via the Jacobian.

    Parameters
    ----------
    g
        Source :class:`ComponentMetric` on a :class:`CoordinateFrame`.
        Tetrad / AbstractFrame metrics are rejected — coordinate
        transformations only make sense on a coordinate frame.
    new_coords
        Sequence of SymPy ``Symbol`` s for the new coordinates. Length
        must match the source frame's dimension.
    transform_map
        Mapping ``{old_sym: expression in new_coords}``. You only
        need entries for coordinates that **change**; any old
        coordinate that also appears in ``new_coords`` (as the same
        ``Symbol`` object) is treated as identity automatically.
        Every value must be a SymPy expression that depends only on
        the new coordinates.
    name
        Optional name for the constructed :class:`CoordinateFrame`.
    simplify
        Apply :func:`sympy.simplify` to each transformed entry.
        Default ``True``. Pass ``False`` for Kerr-class metrics where
        mid-formula simplify dominates runtime.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        A new ``(F', g')`` pair on the new coordinates. ``g'`` is the
        pull-back of ``g`` along the diffeomorphism encoded by
        ``transform_map``.

    Raises
    ------
    TypeError
        If ``g.frame`` is not a :class:`CoordinateFrame`, or
        ``new_coords`` contains non-Symbol entries.
    ValueError
        If ``len(new_coords) != g.frame.dim``, ``transform_map`` is
        missing an entry for some old coordinate, or one of its
        values references an old coordinate (transform must be closed
        in the new coordinates).

    Notes
    -----
    The helper assumes the user-supplied transformation is a smooth
    bijection (a diffeomorphism). It does **not** check invertibility
    or non-degeneracy: if the dict encodes a singular or many-to-one
    map, the resulting metric components may be undefined or
    degenerate. The error will surface downstream (e.g. when
    :meth:`ComponentMetric.inverse` fails) rather than here.

    The pull-back formula

    .. math::

        g'_{ab}(x') = J^c{}_a J^d{}_b \, g_{cd}(x(x')),
        \qquad J^c{}_a := \frac{\partial x^c}{\partial x'^a}

    is computed entry-by-entry. Curvature data (Christoffel, Ricci,
    Einstein) of the new metric is recovered by re-running the
    pipeline on ``g'`` — no separate ``transform_connection`` is
    needed because the Levi-Civita connection of the pulled-back
    metric is the pull-back of the original connection.
    """
    if not isinstance(g, ComponentMetric):
        raise TypeError("transform_metric: g must be a ComponentMetric")
    if not isinstance(g.frame, CoordinateFrame):
        raise TypeError(
            "transform_metric: source frame must be a CoordinateFrame; "
            f"got {type(g.frame).__name__}. Coordinate transformations "
            "do not apply to Tetrad / AbstractFrame inputs."
        )

    n = g.frame.dim
    new_coords_t = tuple(new_coords)
    if len(new_coords_t) != n:
        raise ValueError(
            "transform_metric: new_coords length must match the source "
            f"dimension ({n}); got {len(new_coords_t)}"
        )
    for c in new_coords_t:
        if not isinstance(c, sp.Symbol):
            raise TypeError(
                "transform_metric: new_coords entries must be SymPy "
                f"Symbols; got {type(c).__name__}"
            )

    old_coords = g.frame.coords
    old_set = set(old_coords)
    new_set = set(new_coords_t)

    # Build the effective transform: user-provided entries take precedence;
    # any old coord that also appears in new_coords is treated as identity.
    effective_map: dict = {}
    for old_sym in old_coords:
        if old_sym in transform_map:
            effective_map[old_sym] = sp.sympify(transform_map[old_sym])
        elif old_sym in new_set:
            # Identity: this coord is unchanged
            effective_map[old_sym] = old_sym
        else:
            raise ValueError(
                f"transform_metric: old coordinate {old_sym} is missing "
                "from transform_map and does not appear in new_coords; "
                "either provide a transform expression or include the "
                "symbol in new_coords for an identity mapping."
            )

    # Reject keys in transform_map that aren't source coords.
    extras = set(transform_map.keys()) - old_set
    if extras:
        raise ValueError(
            f"transform_metric: transform_map has keys "
            f"{sorted(str(s) for s in extras)} that are not source "
            "coordinates"
        )

    # Validate values reference only new coords (no leaks of unmapped old).
    unmapped_old = old_set - new_set
    for old_sym, expr in effective_map.items():
        leaks = expr.free_symbols & unmapped_old - {old_sym}
        if leaks:
            raise ValueError(
                f"transform_metric: transform_map[{old_sym}] = {expr} "
                f"references old coordinates {sorted(str(s) for s in leaks)}; "
                "the new expression must be closed in the new coordinates."
            )

    # Build Jacobian: J[c, a] = ∂(old[c]) / ∂(new[a]).
    J = sp.Matrix(
        n, n,
        lambda c, a: sp.diff(effective_map[old_coords[c]], new_coords_t[a]),
    )

    # Construct new frame.
    F_new = (
        CoordinateFrame(list(new_coords_t), name=name)
        if name is not None
        else CoordinateFrame(list(new_coords_t))
    )

    # Pull back: g'_{ab} = sum_{c, d} J[c, a] * J[d, b] * g[c, d],
    # then substitute old → new so the result lives on new_coords.
    new_matrix_rows = []
    sub_dict = dict(effective_map)
    for a in range(n):
        row = []
        for b in range(n):
            s = sp.S.Zero
            for c in range(n):
                for d in range(n):
                    s += J[c, a] * J[d, b] * g[c, d]
            s = s.subs(sub_dict)
            if simplify:
                try:
                    s = sp.simplify(s)
                except (TypeError, AttributeError):
                    pass
            row.append(s)
        new_matrix_rows.append(row)

    new_g = ComponentMetric(F_new, sp.Matrix(new_matrix_rows))
    return F_new, new_g
