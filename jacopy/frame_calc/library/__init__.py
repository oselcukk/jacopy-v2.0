r"""
Concrete metric fixtures, Stage H + Faz 19 expansion.

Ready-made ``(frame, metric)`` factories for standard metrics. Each
returns ``tuple[CoordinateFrame, ComponentMetric]`` ready for the
Faz 18 pipeline:

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

**Available fixtures:**

| Fixture                                  | Vacuum?  | Symmetry              |
|------------------------------------------|----------|-----------------------|
| :func:`minkowski`                        | yes      | maximal (Poincaré)    |
| :func:`schwarzschild`                    | yes      | static, spherical     |
| :func:`reissner_nordstrom`               | no (EM)  | static, spherical     |
| :func:`kerr`                             | yes      | stationary, axial     |
| :func:`frw`                              | no       | homogeneous, isotropic|
| :func:`de_sitter` / :func:`anti_de_sitter` | Λ-vac  | maximally symmetric   |
| :func:`vaidya`                           | no (null radiation) | static, spherical |
| :func:`bianchi_I` / :func:`bianchi_V` / :func:`bianchi_IX` | depends | homogeneous anisotropic |
| :func:`godel`                            | no (dust+Λ) | rotating, CTCs     |
"""

from jacopy.frame_calc.library.anti_de_sitter import anti_de_sitter
from jacopy.frame_calc.library.bianchi import bianchi_I, bianchi_IX, bianchi_V
from jacopy.frame_calc.library.de_sitter import de_sitter
from jacopy.frame_calc.library.frw import frw
from jacopy.frame_calc.library.godel import godel
from jacopy.frame_calc.library.kerr import kerr
from jacopy.frame_calc.library.minkowski import minkowski
from jacopy.frame_calc.library.reissner_nordstrom import reissner_nordstrom
from jacopy.frame_calc.library.schwarzschild import schwarzschild
from jacopy.frame_calc.library.vaidya import vaidya

__all__ = [
    "minkowski",
    "schwarzschild",
    "reissner_nordstrom",
    "kerr",
    "frw",
    "de_sitter",
    "anti_de_sitter",
    "vaidya",
    "bianchi_I",
    "bianchi_V",
    "bianchi_IX",
    "godel",
]
