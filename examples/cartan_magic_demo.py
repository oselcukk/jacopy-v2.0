"""End-to-end proof demo: Cartan's magic formula + d² = 0.

Three canonical Cartan-calculus closures, each rendered through the
display layer (terminal tree + ASCII + LaTeX). The three demos form a
progression from trivial element-level rewriting to a genuine
operator-level proof on an exterior algebra.

Demo 1, *Cartan's magic formula, element-level*, efficient mode::

    L_X(ω) == (d ∘ ι_X + ι_X ∘ d)(ω)

Closed by :func:`jacopy.proof.show_equal`: definition-level unfold of
``L_X`` (classified as an axiom by default), graded-Leibniz product
rule, canonicalize. Three steps. Fast, but circular: the rule that
fires *is* the formula being proved.

Demo 2, *d² = 0*, foundational mode::

    d(d(f)) == 0

Default engine with ``d_squared_mode="theorem"`` classifies ``d² = 0``
as a theorem; foundational mode attaches its generator-axiom sub-proof
as a child step. Shows the axiom/theorem distinction in the tree.

Demo 3, *Cartan's magic formula, operator-level*::

    [d, ι_X] = L_X    (as an equation of operators on Ω*(M))

Closed by :class:`CartanCalculus.verify`, which routes through
:class:`AgreementOnGenerators`: both sides must have the same degree
and agree on each generator of the supplied exterior algebra. The
proof is fundamentally reshaped, degree check + per-generator
sub-proofs, instead of element-level simplify. Run twice, once in
efficient mode and once in foundational mode, to expose the
:class:`UnrollToFoundations` wrapper.

Run::

    python3 examples/cartan_magic_demo.py

The terminal output is coloured if ``rich`` is installed; otherwise
the ASCII fallback is used (text is identical in structure).
"""

from __future__ import annotations

from jacopy.algebra.derivation import Act, Derivation, compose
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Sum, Symbol
from jacopy.display import (
    HAS_RICH,
    chain_to_ascii,
    chain_to_latex,
    render_chain,
)
from jacopy.proof import show_equal
from jacopy.proof.expansion import default_engine


def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}\n")


def demo_cartan_magic() -> None:
    """``L_X(ω) == (d∘ι_X + ι_X∘d)(ω)``, one-liner via the default engine."""
    _section("Demo 1, Cartan's magic formula (efficient mode)")

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
    """``d(d(f)) == 0`` under foundational mode, sub-proof gets attached."""
    _section("Demo 2, d² = 0 (foundational mode, theorem classification)")

    f = Symbol("f")
    lhs = Act(d, Act(d, f))

    # Foundational mode + theorem classification together: the engine tags
    # the step "theorem" and attaches the generator-axiom sub-proof as
    # children. The terminal tree shows the nested structure directly.
    engine = default_engine(mode="foundational", d_squared_mode="theorem")

    from jacopy.core.expr import Integer

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


def demo_cartan_magic_operator_level() -> None:
    """``[d, ι_X] = L_X`` as an operator equation, AgreementOnGenerators.

    This is the *non-trivial* closure of Cartan's magic formula: the
    identity is stated between operators (not between their evaluations
    on a chosen form), and the proof discharges it on every generator
    of a finite exterior algebra ``Ω*(M)``. Two modes of the same
    verification are shown, so the effect of the foundational unroll
    is visible.
    """
    _section("Demo 3, Cartan's magic formula, operator-level")

    # Build a concrete Cartan-calculus bundle and a minimal
    # exterior algebra with one 0-form generator f (and its
    # differential df, which ExteriorAlgebra derives automatically).
    from jacopy.brackets.lie import LieBracket
    from jacopy.calculus.cartan import CartanCalculus
    from jacopy.calculus.exterior_algebra import ExteriorAlgebra
    from jacopy.core.properties import Graded
    from jacopy.core.registry import PropertyRegistry

    reg = PropertyRegistry()
    f = Symbol("f")
    reg.declare(f, Graded(degree=0))
    algebra = ExteriorAlgebra((f,))
    print(f"  algebra generators = {algebra.generators}\n")

    X = Derivation("X", degree=0)
    calc = CartanCalculus(
        d=d,
        lie_derivative=lie_derivative,
        interior=interior,
        vector_bracket=LieBracket(),
    )

    # -- efficient mode -------------------------------------------- #
    print("--- efficient mode")
    chain_eff = calc.verify(
        "cartan_magic",
        algebra=algebra,
        X=X,
        registry=reg,
        mode="efficient",
    )
    print(render_chain(chain_eff))

    # -- foundational mode ----------------------------------------- #
    # Wire d² = 0 as a theorem so the foundational unroll has
    # something to expand under the per-generator sub-proof, on
    # generator ``df`` the proof hits ι_X(d(df)) which fires d² = 0.
    foundational_engine = default_engine(
        registry=reg, mode="foundational", d_squared_mode="theorem"
    )
    print("\n--- foundational mode (d² = 0 classified as theorem)")
    chain_fnd = calc.verify(
        "cartan_magic",
        algebra=algebra,
        X=X,
        registry=reg,
        mode="foundational",
        engine=foundational_engine,
    )
    print(render_chain(chain_fnd))

    # Per-generator sub-proof inspection, makes the shape explicit
    # even when the tree renderer wraps lines.
    print("\n--- shape of the efficient-mode proof")
    root = chain_eff.steps[0]
    print(f"  root rule       = {root.rule}")
    print(f"  root children   = {len(root.children)} (one per generator)")
    for i, gen_step in enumerate(root.children):
        print(
            f"    child[{i}] rule = {gen_step.rule}  "
            f"({len(gen_step.children)} sub-step(s))"
        )

    print(
        "\n  Note: efficient and foundational trees are identical here\n"
        "  because the (A + B) - (A + B) cancellation in ExpandAndSimplify\n"
        "  reaches 0 without any theorem-classified rule firing, d² = 0\n"
        "  is never invoked, so there's nothing for UnrollToFoundations\n"
        "  to unroll. The mode only shows a visible difference when a\n"
        "  theorem step is actually reached (cf. Demo 2)."
    )


def main() -> None:
    demo_cartan_magic()
    demo_d_squared_zero()
    demo_cartan_magic_operator_level()
    print()


if __name__ == "__main__":
    main()
