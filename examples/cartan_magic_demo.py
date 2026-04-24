"""End-to-end proof demo: Cartan's magic formula + d² = 0.

Two canonical Cartan-calculus identities, each proved via
:func:`gradalg.proof.show_equal` and rendered through the display
layer (terminal tree + ASCII + LaTeX).

Demo 1 — *Cartan's magic formula*, efficient mode::

    L_X(ω) == (d ∘ ι_X + ι_X ∘ d)(ω)

Demo 2 — *d² = 0*, foundational mode::

    d(d(f)) == 0

In foundational mode the ``d² = 0`` rule is classified as a theorem
and its :class:`ProofStep` carries a sub-proof citing the generator
axiom ``d(df) = 0``. The tree renderer nests the sub-proof under the
parent step, which is the whole point of the mode switch.

Run::

    python3 examples/cartan_magic_demo.py

The terminal output is coloured if ``rich`` is installed; otherwise
the ASCII fallback is used (text is identical in structure).
"""

from __future__ import annotations

from gradalg.algebra.derivation import Act, Derivation, compose
from gradalg.calculus.exterior_d import d
from gradalg.calculus.interior import interior
from gradalg.calculus.lie_derivative import lie_derivative
from gradalg.core.expr import Sum, Symbol
from gradalg.display import (
    HAS_RICH,
    chain_to_ascii,
    chain_to_latex,
    render_chain,
)
from gradalg.proof import show_equal
from gradalg.proof.expansion import default_engine


def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}\n")


def demo_cartan_magic() -> None:
    """``L_X(ω) == (d∘ι_X + ι_X∘d)(ω)`` — one-liner via the default engine."""
    _section("Demo 1 — Cartan's magic formula (efficient mode)")

    # A degree-0 vector field and a differential form on the same manifold.
    # The Lie derivative is built in the Cartan definition so the expansion
    # engine's `LieDerivativeCartanDefinition` will fire on it.
    X = Derivation("X", degree=0)
    omega = Symbol("ω")
    L_X = lie_derivative(X, definition="cartan")
    iota_X = interior(X)

    lhs = Act(L_X, omega)
    rhs = Sum(
        Act(compose(d, iota_X), omega),
        Act(compose(iota_X, d), omega),
    )

    print(f"  lhs = {lhs}")
    print(f"  rhs = {rhs}\n")

    chain = show_equal(lhs, rhs)

    print("--- terminal tree " + ("(rich)" if HAS_RICH else "(ASCII fallback)"))
    print(render_chain(chain))

    print("\n--- ASCII transcript")
    print(chain_to_ascii(chain))

    print("\n--- LaTeX (align* body)")
    print(chain_to_latex(chain))


def demo_d_squared_zero() -> None:
    """``d(d(f)) == 0`` under foundational mode — sub-proof gets attached."""
    _section("Demo 2 — d² = 0 (foundational mode, theorem classification)")

    f = Symbol("f")
    lhs = Act(d, Act(d, f))

    # Foundational mode + theorem classification together: the engine tags
    # the step "theorem" and attaches the generator-axiom sub-proof as
    # children. The terminal tree shows the nested structure directly.
    engine = default_engine(mode="foundational", d_squared_mode="theorem")

    from gradalg.core.expr import Integer

    chain = show_equal(lhs, Integer(0), engine=engine)

    print(f"  lhs = {lhs}")
    print(f"  rhs = 0\n")

    print("--- terminal tree " + ("(rich)" if HAS_RICH else "(ASCII fallback)"))
    print(render_chain(chain))

    print("\n--- ASCII transcript (nested sub-proof visible)")
    print(chain_to_ascii(chain))

    # Surface the theorem step's children explicitly so the
    # axiom-bottoming-out structure is visible without needing the
    # renderer to parse.
    theorem_steps = [s for s in chain if s.provenance_tag == "theorem"]
    if theorem_steps:
        print("\n--- inspecting the theorem step")
        t = theorem_steps[0]
        print(f"  rule         = {t.rule}")
        print(f"  provenance   = {t.provenance_tag}")
        print(f"  #children    = {len(t.children)}")
        for i, child in enumerate(t.children):
            print(f"  child[{i}]     = [{child.provenance_tag}] {child.rule}")


def main() -> None:
    demo_cartan_magic()
    demo_d_squared_zero()
    print()


if __name__ == "__main__":
    main()
