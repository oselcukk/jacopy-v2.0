"""
Single proof step.

A :class:`ProofStep` records one rewrite: ``before`` is the expression
before the rule fired, ``after`` is the expression after, and ``rule``
/ ``justification`` name what was applied and why. Steps nest, a
strategy that decomposes its work into sub-proofs records them under
:attr:`children`, which lets the transcript be rendered at any
verbosity level by choosing how deep to unfold each step.

The class is deliberately thin: no checks beyond type validation on
``before`` / ``after``, no automatic semantic interpretation of
``rule``. The proof-validity invariant, that ``after`` actually
follows from ``before`` under ``rule``, is the responsibility of
whatever code constructed the step. This keeps the data type a pure
record that tracers, strategies, and display modules can all share
without coupling.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.core.expr import Expr


class ProofStep:
    """One rewrite: ``before → after`` under ``rule``."""

    __slots__ = (
        "_before",
        "_after",
        "_rule",
        "_justification",
        "_children",
        "_provenance_tag",
    )

    #: Recognised provenance tags. ``None`` means the step carries no
    #: provenance claim (e.g. a structural rewrite like ``product-rule``).
    #:
    #: * ``"axiom"``, primitive equation accepted as input.
    #: * ``"theorem"``, derived from other primitives; foundational
    #:   mode can recover the sub-proof.
    #: * ``"computation"``, a frame-component / numerical-symbolic
    #:   step (typically Stage G of :mod:`jacopy.frame_calc`); not a
    #:   citation but a calculation. Carries the same paper-grade
    #:   render path as axiom / theorem.
    _VALID_TAGS = (None, "axiom", "theorem", "computation")

    def __init__(
        self,
        before: Expr,
        after: Expr,
        rule: str,
        justification: str = "",
        children: Optional[List["ProofStep"]] = None,
        *,
        provenance_tag: Optional[str] = None,
    ) -> None:
        if not isinstance(before, Expr):
            raise TypeError("ProofStep.before must be an Expr")
        if not isinstance(after, Expr):
            raise TypeError("ProofStep.after must be an Expr")
        if not isinstance(rule, str):
            raise TypeError("ProofStep.rule must be a str")
        if not isinstance(justification, str):
            raise TypeError("ProofStep.justification must be a str")
        if provenance_tag not in self._VALID_TAGS:
            raise ValueError(
                f"provenance_tag must be one of {self._VALID_TAGS}, "
                f"got {provenance_tag!r}"
            )
        self._before = before
        self._after = after
        self._rule = rule
        self._justification = justification
        self._provenance_tag = provenance_tag
        self._children: List[ProofStep] = []
        if children:
            for ch in children:
                self.add_child(ch)

    @property
    def before(self) -> Expr:
        return self._before

    @property
    def after(self) -> Expr:
        return self._after

    @property
    def rule(self) -> str:
        return self._rule

    @property
    def justification(self) -> str:
        return self._justification

    @property
    def children(self) -> Tuple["ProofStep", ...]:
        return tuple(self._children)

    @property
    def provenance_tag(self) -> Optional[str]:
        """Classification of this step: ``"axiom"``, ``"theorem"``, or ``None``.

        Set by the expansion engine when a :class:`Definition` fires, the
        tag records whether the rewrite was taken axiomatically or whether
        it's derivable, which :class:`UnrollToFoundations` reads to decide
        whether to attach a sub-proof.
        """
        return self._provenance_tag

    def add_child(self, step: "ProofStep") -> None:
        """Nest ``step`` under this one as a sub-proof."""
        if not isinstance(step, ProofStep):
            raise TypeError("ProofStep.add_child expects a ProofStep")
        self._children.append(step)

    def format(self, indent: int = 0, max_depth: int = 64) -> str:
        """Render this step as text; recurse into children up to ``max_depth``."""
        pad = "  " * indent
        tag = f" ({self._provenance_tag})" if self._provenance_tag else ""
        head = (
            f"{pad}[{self._rule}]{tag} "
            f"{self._before._repr_inner()} → {self._after._repr_inner()}"
        )
        if self._justification:
            head = f"{head} , {self._justification}"
        lines = [head]
        if max_depth > 0 and self._children:
            for ch in self._children:
                lines.append(ch.format(indent + 1, max_depth - 1))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ProofStep(rule={self._rule!r}, "
            f"{self._before!r} → {self._after!r})"
        )
