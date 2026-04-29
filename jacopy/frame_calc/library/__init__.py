r"""
Concrete metric fixtures — Stage H.

Ready-made ``(frame, metric)`` factories for the metrics most papers
calibrate against. Each returns ``tuple[CoordinateFrame, ComponentMetric]``
that the user can drop into the Faz 18 pipeline:

.. code-block:: python

    from jacopy.frame_calc import levi_civita, einstein_tensor
    from jacopy.frame_calc.library import schwarzschild

    F, g = schwarzschild()
    LC = levi_civita(g)
    G = einstein_tensor(LC, g)
    assert G.is_vacuum()    # Schwarzschild is a vacuum solution

The factories accept optional ``Symbol`` / ``Function`` overrides so
the metrics can be composed with user-supplied parameters or
specialised to specific limits.
"""

from jacopy.frame_calc.library.frw import frw
from jacopy.frame_calc.library.kerr import kerr
from jacopy.frame_calc.library.minkowski import minkowski
from jacopy.frame_calc.library.schwarzschild import schwarzschild

__all__ = ["minkowski", "schwarzschild", "frw", "kerr"]
