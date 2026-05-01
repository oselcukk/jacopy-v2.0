r"""
Minkowski metric on flat spacetime.

The 4D Minkowski metric in Cartesian coordinates ``(t, x, y, z)``:

.. math::

    ds^2 = -dt^2 + dx^2 + dy^2 + dz^2.

Christoffel symbols, Ricci, Einstein tensor, all identically zero.
"""

from __future__ import annotations

from typing import Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def minkowski(
    *, signature: str = "-+++"
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""4D Minkowski metric in Cartesian coordinates ``(t, x, y, z)``.

    Parameters
    ----------
    signature
        Either ``"-+++"`` (default; timelike-negative, spacelike-positive)
        or ``"+---"`` (timelike-positive, spacelike-negative). Both
        conventions are used in the literature; the
        Einstein tensor / Ricci-flat verifications give the same
        result either way.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        The frame uses standard symbol names ``t, x, y, z``; both
        frame and metric live on the same coordinate symbols.
    """
    if signature not in ("-+++", "+---"):
        raise ValueError(
            f"minkowski signature must be '-+++' or '+---', "
            f"got {signature!r}"
        )
    t, x, y, z = sp.symbols("t x y z", real=True)
    F = CoordinateFrame([t, x, y, z], name=f"minkowski({signature})")
    if signature == "-+++":
        g = ComponentMetric(F, sp.diag(-1, 1, 1, 1))
    else:
        g = ComponentMetric(F, sp.diag(1, -1, -1, -1))
    return F, g
