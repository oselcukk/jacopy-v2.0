r"""
Torsion tensor from a connection, Stage E.

The torsion of a connection ``∇`` on a frame ``F`` has frame
components

.. math::

    T(\nabla)^a{}_{bc}
    = \Gamma^a{}_{bc} - \Gamma^a{}_{cb} - \gamma^a{}_{bc}.

For the Levi-Civita connection (and any torsion-free connection in a
holonomic frame) every entry vanishes, :class:`TorsionTensor` is
mainly useful for testing and for connections built by hand.
"""

from __future__ import annotations

from typing import Any

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


class TorsionTensor(ComponentTensor):
    r"""``(1, 2)`` tensor ``T^a_{bc}``, torsion of a connection.

    Same shape as :class:`ComponentConnection` (one upper, two lower),
    but antisymmetric in the lower pair: ``T^a_{bc} = -T^a_{cb}``.
    """

    __slots__ = ()

    def __init__(self, frame: Any, components: Any) -> None:
        super().__init__(frame, components, signature=(1, 2))

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "TorsionTensor":
        out = object.__new__(TorsionTensor)
        ComponentTensor.__init__(
            out, self._frame, new_arr, signature=(1, 2)
        )
        return out


def torsion(connection: ComponentConnection) -> TorsionTensor:
    r"""Compute the torsion of a connection from its Christoffel symbols.

    For each ``(a, b, c)``::

        T^a_{bc} = Γ^a_{bc} - Γ^a_{cb} - γ^a_{bc}

    where ``γ^a_{bc}`` are the frame's Lie bracket structure constants
    (zero for coordinate frames; non-trivial for tetrads /
    non-holonomic frames).

    Parameters
    ----------
    connection
        A :class:`ComponentConnection` (or subclass such as
        :class:`~jacopy.frame_calc.levi_civita.LeviCivitaConnection`).

    Returns
    -------
    TorsionTensor
        Antisymmetric in its lower pair.

    Examples
    --------
    The Levi-Civita connection is torsion-free::

        T = torsion(levi_civita(g))
        assert T.is_zero()
    """
    if not isinstance(connection, ComponentConnection):
        raise TypeError(
            "torsion expects a ComponentConnection, got "
            f"{type(connection).__name__}"
        )

    frame = connection.frame
    n = frame.dim
    components = sp.MutableDenseNDimArray.zeros(n, n, n)

    for a in range(n):
        for b in range(n):
            for c in range(n):
                gamma_struct = frame.gamma(a, b, c)
                value = (
                    connection[a, b, c]
                    - connection[a, c, b]
                    - gamma_struct
                )
                try:
                    value = sp.simplify(value)
                except (TypeError, AttributeError):
                    pass
                components[a, b, c] = value

    return TorsionTensor(frame, components)
