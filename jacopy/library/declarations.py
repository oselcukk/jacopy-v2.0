"""Role-driven declaration helpers.

Every tutorial opens with the same boilerplate: create a few
:class:`~jacopy.core.expr.Symbol`s and declare each one's grading on a
:class:`~jacopy.core.registry.PropertyRegistry`. That is two lines of
ceremony hiding a one-line intent, "these are three vector fields" or
"this is a Poisson bivector". The helpers below bundle the Symbol
construction with the :class:`~jacopy.core.properties.Graded`
declaration into a single call that reads like a mathematical
declaration.

All helpers take an explicit ``registry=`` keyword. There is no
module-level default; a tutorial or user script must bring its own
:class:`PropertyRegistry`. That keeps property state local and avoids
the cross-notebook leakage that a hidden singleton would invite.
"""

from __future__ import annotations

from typing import Tuple

from jacopy.core.expr import Symbol
from jacopy.core.properties import Antisymmetric, Graded
from jacopy.core.registry import PropertyRegistry


def _parse_names(names: str) -> Tuple[str, ...]:
    if not isinstance(names, str):
        raise TypeError("names must be a whitespace-separated string")
    parts = tuple(names.split())
    if not parts:
        raise ValueError("at least one name required")
    return parts


def _declare_each(
    names: Tuple[str, ...],
    degree: int,
    registry: PropertyRegistry,
) -> Tuple[Symbol, ...]:
    syms = tuple(Symbol(n) for n in names)
    for s in syms:
        registry.declare(s, Graded(degree=degree))
    return syms


def Functions(
    names: str,
    *,
    degree: int = 0,
    registry: PropertyRegistry,
) -> Tuple[Symbol, ...]:
    """Declare one or more function symbols at a chosen grading.

    ``Functions("f g h", registry=reg)`` returns three :class:`Symbol`
    instances, each already carrying :class:`Graded(degree=0)` on
    ``reg``. Even for a single name, the return is a tuple, unpack
    with ``(f,) = Functions("f", registry=reg)``.

    The default ``degree=0`` matches the classical form-grading.
    Pass ``degree=-1`` in Schouten–Nijenhuis contexts (Poisson
    geometry, Cartan calculus over multivectors) where a 0-form has
    SN-degree ``-1``; the
    :class:`~jacopy.brackets.schouten.ScoutenNijenhuis` expansion
    relies on that grading to recognise a function.
    """
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    parts = _parse_names(names)
    return _declare_each(parts, degree, registry)


def VectorFields(
    names: str, *, registry: PropertyRegistry
) -> Tuple[Symbol, ...]:
    """Declare one or more vector-field symbols.

    Vector fields carry :class:`Graded(degree=0)` in the classical
    grading convention used by :mod:`jacopy.brackets.lie`, this is
    what the Koszul sign machinery needs to expand a Jacobi
    obstruction. The registry footprint is identical to
    :func:`Functions`; the two helpers differ only in the name a
    tutorial reader sees at the call site.
    """
    parts = _parse_names(names)
    return _declare_each(parts, 0, registry)


def Forms(
    names: str,
    *,
    degree: int,
    registry: PropertyRegistry,
) -> Tuple[Symbol, ...]:
    """Declare one or more differential-form symbols of a given degree.

    ``Forms("α β", degree=1, registry=reg)`` produces two symbols with
    :class:`Graded(degree=1)`. Any non-negative integer degree is
    accepted.
    """
    if not isinstance(degree, int):
        raise TypeError("degree must be an int")
    parts = _parse_names(names)
    return _declare_each(parts, degree, registry)


def Bivector(
    name: str, *, registry: PropertyRegistry
) -> Symbol:
    """Declare a single bivector symbol.

    Matches the Schouten–Nijenhuis grading used throughout
    :mod:`jacopy.library.poisson`: a 2-vector carries
    :class:`Graded(degree=1)` (SN-degree ``k - 1`` for a k-vector). A
    2-vector is anti-symmetric in its covector arguments,
    ``π(α, β) = -π(β, α)``, so the helper additionally declares
    :class:`Antisymmetric` on the symbol. The declaration is consumed
    by :class:`~jacopy.calculus.antisym_axioms.RegistryAntiSymCanonicalDefinition`
    when the bivector appears as the head of a 2-arg
    :class:`~jacopy.core.multi_eval.MultiEval`, no per-callsite
    ``alternating=True`` flag needed.

    Returns the symbol itself, *not* a 1-tuple, because Bivector is
    singular by construction.
    """
    pieces = _parse_names(name)
    if len(pieces) != 1:
        raise ValueError("Bivector takes exactly one name")
    sym = Symbol(pieces[0])
    registry.declare(sym, Graded(degree=1))
    registry.declare(sym, Antisymmetric())
    return sym


__all__ = ["Functions", "VectorFields", "Forms", "Bivector"]
