"""Build the companion Jupyter notebooks for the tutorial markdowns.

Each tutorial pairs a ``NN_topic.md`` with an executable ``NN_topic.ipynb``.
Rather than maintain two copies by hand, the notebook sources live here
as Python dicts, one ``(markdown cells, code cells)`` tuple per
tutorial, and :func:`build_all` serialises them to ``.ipynb`` via
:mod:`nbformat`.

Run::

    python3 docs/tutorials/_build_notebooks.py

to regenerate every notebook. The test suite
(``tests/test_docs/test_notebooks.py``) executes whatever ``.ipynb``
files live alongside the markdowns, so running this script is a
prerequisite any time a tutorial's code path changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

THIS_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------- #
# Tutorial sources                                                       #
# --------------------------------------------------------------------- #


# Every notebook opens with a bootstrap cell that ensures ``jacopy``
# is importable even when the notebook is opened directly in the IDE
# (not via pytest, which has its own PYTHONPATH fixture). We try a
# plain import first and only touch ``sys.path`` on failure, so a
# user who ``pip install -e .``'d the repo sees no side-effects.
_BOOTSTRAP = (
    "code",
    "# Ensure jacopy is importable when this notebook is opened\n"
    "# directly (not via pytest). Walks up from the notebook's\n"
    "# directory to the repo root and prepends it to sys.path if\n"
    "# jacopy isn't already installed into this kernel.\n"
    "try:\n"
    "    import jacopy  # noqa: F401\n"
    "except ModuleNotFoundError:\n"
    "    import sys\n"
    "    from pathlib import Path\n"
    "    here = Path.cwd().resolve()\n"
    "    for candidate in (here, *here.parents):\n"
    "        if (candidate / \"jacopy\" / \"__init__.py\").is_file():\n"
    "            sys.path.insert(0, str(candidate))\n"
    "            break\n"
    "    import jacopy  # noqa: F401",
)


TUTORIAL_01: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 01, First steps\n\n"
        "Companion notebook to [01_first_steps.md](01_first_steps.md). "
        "Building a symbolic expression in `jacopy`, declaring properties, "
        "and simplifying, the basics.",
    ),
    (
        "markdown",
        "## Symbols and basic construction\n\n"
        "`Symbol(name)` builds an atomic expression; `Integer(n)` is a "
        "literal constant; `+`, `*`, `-` produce `Sum`, `Product`, `Neg` "
        "under the hood.",
    ),
    (
        "code",
        "from jacopy.core.expr import Symbol, Integer, Sum, Product, Neg\n\n"
        "x, y, z = Symbol(\"x\"), Symbol(\"y\"), Symbol(\"z\")\n\n"
        "expr = x + y - z\n"
        "assert expr == Sum(x, y, Neg(z))\n"
        "print(expr)",
    ),
    (
        "code",
        "mult = 2 * x\n"
        "assert mult == Product(Integer(2), x)\n"
        "print(mult)",
    ),
    (
        "markdown",
        "## Declaring properties (`PropertyRegistry`)\n\n"
        "Properties, degree, scalarity, graded antisymmetry, …, are "
        "declared **externally** on symbols. That's what lets the same "
        "symbol be reused in different contexts (different degrees, "
        "different algebras).",
    ),
    (
        "code",
        "from jacopy.core.properties import Graded, Scalar\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "reg.declare(x, Scalar())\n"
        "reg.declare(y, Scalar())\n"
        "reg.declare(z, Graded(degree=1))  # z behaves like a 1-form",
    ),
    (
        "markdown",
        "### Role-driven shortcuts\n\n"
        "For common patterns, functions, vector fields, forms, bivectors "
        ", `jacopy.library.declarations` exposes `Functions`, "
        "`VectorFields`, `Forms`, `Bivector` helpers. Each collapses "
        "`Symbol(...)` plus the matching `reg.declare(...)` into a single "
        "call.",
    ),
    (
        "code",
        "from jacopy import Functions, VectorFields, Forms, Bivector\n\n"
        "reg2 = PropertyRegistry()\n"
        "f, g = Functions(\"f g\", registry=reg2)\n"
        "X, Y = VectorFields(\"X Y\", registry=reg2)\n"
        "alpha, beta = Forms(\"α β\", degree=1, registry=reg2)\n"
        "pi = Bivector(\"π\", registry=reg2)\n"
        "print(f, g, X, Y, alpha, beta, pi)",
    ),
    (
        "markdown",
        "## `simplify`, canonical form\n\n"
        "The `simplify(expr, registry)` pipeline: flatten → canonicalize → "
        "distribute → flatten → sort_product → collect_terms. With a "
        "registry the registered properties (commutativity, degree) are "
        "consulted.",
    ),
    (
        "code",
        "from jacopy.algorithms.simplify import simplify\n\n"
        "assert simplify(x + x - x) == x\n"
        "assert simplify(Product(Integer(2), x, Integer(3)), reg) \\\n"
        "    == Product(Integer(6), x)\n"
        "# Two scalars sort alphabetically:\n"
        "assert simplify(Product(y, x), reg) == Product(x, y)\n"
        "print('simplify checks passed')",
    ),
    (
        "markdown",
        "## Display\n\n"
        "The `display` layer gives `to_ascii`, `to_latex`, and "
        "(if `rich` is installed) a coloured terminal tree.",
    ),
    (
        "code",
        "from jacopy.display import to_ascii, to_latex\n\n"
        "e = x + y - z\n"
        "print('ascii:', to_ascii(e))\n"
        "print('latex:', to_latex(e))",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "Layering a bracket on top → Jacobi proof: "
        "[02_jacobi_identity.md](02_jacobi_identity.md).",
    ),
]


TUTORIAL_02: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 02, The Jacobi identity\n\n"
        "Companion notebook to [02_jacobi_identity.md](02_jacobi_identity.md). "
        "Closing the Jacobi identity for a Lie bracket in a single "
        "`prove_jacobi` call.",
    ),
    (
        "markdown",
        "## The Lie bracket and three vector fields\n\n"
        "`jacopy.brackets.lie.lie` is the module-level singleton for the "
        "standard manifold Lie bracket. The three vector fields are "
        "declared with the `VectorFields` helper, each gets "
        "`Graded(degree=0)` attached, which is what the Jacobi expansion "
        "consults for sign rules.",
    ),
    (
        "code",
        "from jacopy import VectorFields\n"
        "from jacopy.brackets.lie import lie\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "X, Y, Z = VectorFields(\"X Y Z\", registry=reg)",
    ),
    (
        "markdown",
        "## `prove_jacobi`, does the obstruction vanish?\n\n"
        "The graded Jacobi identity:\n"
        "\n"
        "$$[X,[Y,Z]] + (-1)^{|X||Y|+|X||Z|}[Y,[Z,X]] + (-1)^{|Y||Z|+|X||Z|}[Z,[X,Y]] = 0$$\n"
        "\n"
        "On the Lie bracket every degree is zero, so every sign is `+1`. "
        "`prove_jacobi(lie, X, Y, Z, registry=reg)` returns a `ProofChain` "
        "that simplifies the obstruction down to zero.",
    ),
    (
        "code",
        "from jacopy.proof import prove_jacobi\n\n"
        "chain = prove_jacobi(lie, X, Y, Z, registry=reg)\n"
        "print('chain length:', len(chain))\n"
        "print('final:', chain.steps[-1].after)",
    ),
    (
        "markdown",
        "## Step-by-step rendering\n\n"
        "The `display` layer renders `ProofChain`s as ASCII or LaTeX. "
        "Inside Jupyter, `display_chain` produces a LaTeX `align*` block "
        "that MathJax renders inline.",
    ),
    (
        "code",
        "from jacopy.display import chain_to_ascii\n\n"
        "print(chain_to_ascii(chain))",
    ),
    (
        "code",
        "# Jupyter auto-renders the LaTeX (align* blocks):\n"
        "from jacopy.display import display_chain\n"
        "display_chain(chain)",
    ),
    (
        "markdown",
        "## Why does this work?\n\n"
        "The Lie bracket's class definition sets "
        "`is_graded_antisymmetric=True` and "
        "`satisfies_graded_jacobi=True`. `prove_jacobi` first builds the "
        "obstruction, then simplifies it through the expansion engine's "
        "Jacobi recogniser, collapsing the residue to zero. For brackets "
        "with conditional Jacobi (such as `CourantBracket`) the same "
        "function does **not** drive the obstruction to zero; instead the "
        "returned chain records the condition (e.g. `dH = 0`) explicitly.",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "Onwards to Poisson geometry: "
        "[03_poisson_geometry.md](03_poisson_geometry.md).",
    ),
]


TUTORIAL_03: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 03, Poisson geometry\n\n"
        "Companion notebook to [03_poisson_geometry.md](03_poisson_geometry.md). "
        "Three equivalent views of the Poisson bracket on a symplectic "
        "manifold (derived, Hamiltonian, Koszul) plus the Jacobi proof "
        "reduced to the single condition `[π, π]_SN = 0`.",
    ),
    (
        "markdown",
        "## Symplectic manifold, `(ω, π, ♭, ♯)` bundle\n\n"
        "`SymplecticManifold(ω, bivector=π)` keeps the form, the inverse "
        "bivector, the musical maps, and the `MusicalCompatibility` "
        "axiom together in one object. On the registry `ω` is a 2-form "
        "and `π` is a 2-vector with SN-degree 1, the `Bivector` helper "
        "handles that automatically.",
    ),
    (
        "code",
        "from jacopy import Bivector, Forms, Functions\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.library.symplectic import SymplecticManifold\n\n"
        "reg = PropertyRegistry()\n"
        "(omega,) = Forms(\"ω\", degree=2, registry=reg)\n"
        "pi = Bivector(\"π\", registry=reg)\n\n"
        "M = SymplecticManifold(omega, bivector=pi, name=\"(M, ω, π)\")\n"
        "print(M)\n"
        "print('flat:', M.flat)\n"
        "print('sharp:', M.sharp)\n"
        "print('compat:', M.compatibility.name)",
    ),
    (
        "markdown",
        "## `PoissonBracket`, three equivalent views\n\n"
        "Functions live at SN-shifted degree `−1`; the `Functions` "
        "helper takes the `degree=-1` kwarg for this context.",
    ),
    (
        "code",
        "from jacopy.library.poisson import PoissonBracket\n\n"
        "f, g, h = Functions(\"f g h\", degree=-1, registry=reg)\n"
        "poisson = PoissonBracket.from_bivector(pi)\n"
        "print('bracket:', poisson)",
    ),
    (
        "markdown",
        "### View 1, the derived bracket\n\n"
        "`{f, g}_π = [[f, π]_SN, g]_SN`.",
    ),
    (
        "code",
        "poisson.expand(f, g, reg)",
    ),
    (
        "markdown",
        "### View 2, the Hamiltonian vector field\n\n"
        "`{f, g}_π = X_f(g)`. On a symplectic manifold this is "
        "equivalent to `ι_{X_f} ω + df = 0`; "
        "`prove_hamiltonian_equivalence` closes that in five steps using "
        "the musical compatibility.",
    ),
    (
        "code",
        "from jacopy.display import chain_to_ascii\n\n"
        "print('X_f =', poisson.hamiltonian_vf(f))\n"
        "print('X_f(g) =', poisson.via_hamiltonian(f, g))\n"
        "\n"
        "chain = M.prove_hamiltonian_equivalence(f, registry=reg)\n"
        "print('chain length:', len(chain))\n"
        "print(chain_to_ascii(chain))",
    ),
    (
        "markdown",
        "### View 3, the Koszul three-term formula\n\n"
        "On 1-forms `{α, β}_π = L_{π♯(α)} β − L_{π♯(β)} α − "
        "d⟨π♯(α), β⟩`. The classical Koszul bracket and the derived "
        "bracket are *structurally equal* on this operand type, "
        "`prove_koszul_equivalence` records that in a single reflexive "
        "step.",
    ),
    (
        "code",
        "alpha, beta = Forms(\"α β\", degree=1, registry=reg)\n"
        "print('koszul expand:', poisson.koszul_expand(alpha, beta, reg))\n\n"
        "chain_k = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)\n"
        "print('koszul chain length:', len(chain_k))\n"
        "print('rule:', chain_k.steps[0].rule)",
    ),
    (
        "markdown",
        "## `[π, π]_SN = 0`, the single condition\n\n"
        "The Derived Bracket Theorem reduces Jacobi for `{·, ·}_π` to "
        "one condition: `[π, π]_SN = 0`. The three-input reduction "
        "chain reaches the obstruction in a single `DerivedBracketTheorem` "
        "step; for atomic `π` the obstruction stays opaque, the Jacobi "
        "identity closes once the Poisson hypothesis is supplied.",
    ),
    (
        "code",
        "print('obstruction:', poisson.jacobi_obstruction(reg))\n"
        "print('condition:', poisson.jacobi_condition(reg))\n\n"
        "chain_j = poisson.prove_jacobi_reduction(f, g, h, registry=reg)\n"
        "print('chain length:', len(chain_j))\n"
        "print('rule:', chain_j.steps[0].rule)\n"
        "print('reduces to:', chain_j.steps[0].after)",
    ),
    (
        "markdown",
        "## Theorem Book, the seeded theorem\n\n"
        "The library carries this reduction as a ready `Theorem` entry "
        "under `poisson_jacobi`, downstream code wires the result "
        "through a single citation.",
    ),
    (
        "code",
        "from jacopy.library import theorem_book\n\n"
        "thm = theorem_book.get(\"poisson_jacobi\")\n"
        "print('statement:', thm.statement)\n"
        "print('from_axioms:', thm.from_axioms)",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "The Lie algebroid framework applies the same derivation "
        "strategy to a bracket living on a vector bundle, "
        "[04_lie_algebroid.md](04_lie_algebroid.md).",
    ),
]


TUTORIAL_04: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 04, Lie algebroid\n\n"
        "Companion notebook to [04_lie_algebroid.md](04_lie_algebroid.md). "
        "The `(E, [·,·]_E, ρ)` triple as a single object, anchor "
        "compatibility as a separate axiom, and the algebroid Cartan "
        "bundle.",
    ),
    (
        "markdown",
        "## The triple `(E, [·,·]_E, ρ)`\n\n"
        "`LieAlgebroid` keeps the bundle name, the section bracket, the "
        "anchor, and the target `TM` bracket together in one object.",
    ),
    (
        "code",
        "from jacopy import VectorFields\n"
        "from jacopy.brackets.lie import LieBracket\n"
        "from jacopy.calculus.anchor import Anchor\n"
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.library.lie_algebroid import LieAlgebroid\n\n"
        "reg = PropertyRegistry()\n"
        "E = Symbol(\"E\")\n"
        "bracket_E = LieBracket(name=\"[·,·]_E\")\n"
        "rho = Anchor(name=\"ρ\")\n\n"
        "A = LieAlgebroid(E, bracket=bracket_E, anchor=rho, name=\"E-algebroid\")\n"
        "print(A)",
    ),
    (
        "markdown",
        "## Anchor compatibility, a separate axiom\n\n"
        "`ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}` is part of the Lie algebroid "
        "**definition**; the bracket's own axioms do not entail it. "
        "Three presentations: an obstruction (Expr), a condition "
        "(VanishingCondition), and a single-step `axiom`-tagged "
        "`ProofChain`.",
    ),
    (
        "code",
        "X, Y = VectorFields(\"X Y\", registry=reg)\n\n"
        "print('obstruction:')\n"
        "print(' ', A.anchor_compatibility_obstruction(X, Y, reg))\n"
        "print()\n"
        "print('condition:', A.anchor_compatibility_condition(X, Y, reg))\n"
        "print()\n"
        "chain = A.prove_anchor_compatibility(X, Y, registry=reg)\n"
        "print('chain rule:', chain.steps[0].rule)\n"
        "print('provenance:', chain.steps[0].provenance_tag)\n"
        "print('justification:', chain.steps[0].justification)",
    ),
    (
        "markdown",
        "## Algebroid Cartan bundle\n\n"
        "Same `CartanCalculus` API but with `E`-tagged operators: "
        "`d_E`, `L_{E,X}`, `ι_{E,X}`. The operator-level `relation()` "
        "call returns an `OperatorEquation`. Note: algebroid Cartan "
        "magic does not auto-close under `verify` (the engine's "
        "Cartan-definition rewrite is bound to TM); this is expected "
        "and recorded. Live proofs of the five Cartan relations on TM "
        "live in [05_cartan_calculus.md](05_cartan_calculus.md).",
    ),
    (
        "code",
        "cart = A.cartan\n"
        "print('d:', A.d)\n"
        "print('L_E,X:', cart.lie_derivative(X))\n"
        "print('ι_E,X:', cart.interior(X))\n\n"
        "eq = cart.relation(\"cartan_magic\", X=X)\n"
        "print('magic:', eq.lhs, '=', eq.rhs)",
    ),
    (
        "markdown",
        "## Seeded theorem\n\n"
        "The compatibility axiom is registered in `theorem_book`, "
        "downstream theorems (algebroid Cartan, Courant–Dorfman bridge) "
        "cite it in a single step.",
    ),
    (
        "code",
        "from jacopy.library import theorem_book\n\n"
        "thm = theorem_book.get(\"lie_algebroid_anchor_compat\")\n"
        "print('statement:', thm.statement)\n"
        "print('from_axioms:', thm.from_axioms)",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "Live proofs of the five Cartan relations on TM, in two modes: "
        "[05_cartan_calculus.md](05_cartan_calculus.md).",
    ),
]


TUTORIAL_05: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 05, Cartan calculus\n\n"
        "Companion notebook to [05_cartan_calculus.md](05_cartan_calculus.md). "
        "The five Cartan relations as `OperatorEquation`s, the "
        "axiom/theorem split for `d² = 0`, live verification of the "
        "magic formula in both modes, and the invariant-d helper.",
    ),
    (
        "markdown",
        "## The bundle\n\n"
        "`CartanCalculus(d, L, ι, [·,·])`, four ingredients in one "
        "object.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Derivation\n"
        "from jacopy.brackets.lie import LieBracket\n"
        "from jacopy.calculus.cartan import CartanCalculus, RELATIONS\n"
        "from jacopy.calculus.exterior_algebra import ExteriorAlgebra\n"
        "from jacopy.calculus.exterior_d import d\n"
        "from jacopy.calculus.interior import interior\n"
        "from jacopy.calculus.lie_derivative import lie_derivative\n"
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "cart = CartanCalculus(\n"
        "    d=d, lie_derivative=lie_derivative,\n"
        "    interior=interior, vector_bracket=LieBracket(),\n"
        ")\n"
        "print('RELATIONS:', RELATIONS)",
    ),
    (
        "markdown",
        "## Five relations, five `OperatorEquation`s",
    ),
    (
        "code",
        "reg = PropertyRegistry()\n"
        "f = Symbol(\"f\")\n"
        "reg.declare(f, Graded(degree=0))\n"
        "algebra = ExteriorAlgebra((f,))\n"
        "X = Derivation(\"X\", degree=0)\n"
        "Y = Derivation(\"Y\", degree=0)\n\n"
        "for name, kw in [\n"
        "    (\"d_squared_zero\", {}),\n"
        "    (\"cartan_magic\", {\"X\": X}),\n"
        "    (\"d_lie\", {\"X\": X}),\n"
        "    (\"lie_lie\", {\"X\": X, \"Y\": Y}),\n"
        "    (\"lie_iota\", {\"X\": X, \"Y\": Y}),\n"
        "]:\n"
        "    eq = cart.relation(name, algebra=algebra, **kw)\n"
        "    print(f\"{name:16s}: {eq.lhs} = {eq.rhs}\")",
    ),
    (
        "markdown",
        "## `d² = 0`, axiom mode vs theorem mode\n\n"
        "The plain helper `apply_d_squared_zero` always rewrites to "
        "`0`. The default engine with "
        "`d_squared_mode=\"theorem\"` and foundational mode records "
        "`d(d(x)) → 0` as a `ProofStep`.",
    ),
    (
        "code",
        "from jacopy.calculus.exterior_d import apply_d_squared_zero\n"
        "from jacopy.proof.expansion import default_engine\n\n"
        "x = Symbol(\"x\")\n"
        "reg.declare(x, Graded(degree=0))\n"
        "print('axiom rewrite:', apply_d_squared_zero(d(d(x))))\n\n"
        "engine = default_engine(\n"
        "    registry=reg, mode=\"foundational\", d_squared_mode=\"theorem\"\n"
        ")\n"
        "expanded, steps = engine.expand(d(d(x)))\n"
        "print('theorem-mode expanded:', expanded)\n"
        "print('theorem-mode step rule:', steps[0].rule)",
    ),
    (
        "markdown",
        "## All five relations, live proof via `verify`\n\n"
        "`cartan_magic` closes in a single step in both modes on "
        "`ExteriorAlgebra((f,))`; the other four close at the "
        "generator level via `AgreementOnGenerators` + "
        "`ExpandAndSimplify`.",
    ),
    (
        "code",
        "chain = cart.verify(\"cartan_magic\", algebra=algebra, X=X, registry=reg)\n"
        "print('efficient len:', len(chain), 'rule:', chain.steps[0].rule)\n\n"
        "chain_f = cart.verify(\n"
        "    \"cartan_magic\", algebra=algebra, X=X, registry=reg,\n"
        "    mode=\"foundational\",\n"
        ")\n"
        "print('foundational len:', len(chain_f), 'rule:', chain_f.steps[0].rule)\n\n"
        "results = cart.verify_all(algebra=algebra, X=X, Y=Y, registry=reg)\n"
        "for name, c in results.items():\n"
        "    print(f'{name:16s}: len={len(c)}')",
    ),
    (
        "markdown",
        "## `invariant_d`, theorem derived from magic + lie_iota\n\n"
        "`dω(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y])`, the Koszul-Cartan "
        "invariant formula for 1-forms. `InvariantDOneFormDefinition`'s "
        "default classification is `\"theorem\"` (in contrast to "
        "`d²=0`'s `\"axiom\"`), the formula falls out naturally from "
        "magic + lie_iota.",
    ),
    (
        "code",
        "from jacopy.calculus.invariant_d import invariant_d_one_form\n"
        "from jacopy.brackets.lie import lie\n\n"
        "omega = Symbol(\"ω\")\n"
        "reg.declare(omega, Graded(degree=1))\n"
        "print(invariant_d_one_form(omega, X, Y, bracket=lie))",
    ),
    (
        "markdown",
        "## Twisted Cartan bundle, `d_H = d + H∧`\n\n"
        "For a closed 3-form `H`, the H-twisted exterior derivative "
        "`d_H` carries the Cartan calculus through the same five "
        "relations. The `TwistedCartanBundle(H)` wrapper builds `d_H` "
        "as a fresh `ExteriorDerivative` and rewires the Lie-derivative "
        "factory so `d_H` slots into the bundle. Constructing the "
        "bundle is making the assumption `dH = 0`.",
    ),
    (
        "code",
        "from jacopy.library import TwistedCartanBundle\n\n"
        "H = Symbol(\"H\")\n"
        "reg.declare(H, Graded(degree=3))\n"
        "bundle = TwistedCartanBundle(H)\n"
        "print('bundle.d:', bundle.d)\n"
        "print('bundle.cartan.d:', bundle.cartan.d)\n\n"
        "algebra_H = ExteriorAlgebra((f,), d=bundle.d)\n"
        "results = bundle.cartan.verify_all(\n"
        "    algebra=algebra_H, X=X, Y=Y, registry=reg\n"
        ")\n"
        "for name, c in results.items():\n"
        "    print(f'{name:16s}: len={len(c)}')",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "Building your own bracket + the Jacobi test, "
        "[06_custom_bracket.md](06_custom_bracket.md) (Stage C).",
    ),
]


TUTORIAL_06: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 06, Custom bracket\n\n"
        "Companion notebook to [06_custom_bracket.md](06_custom_bracket.md). "
        "Define your own rule with `CustomBracket`, declare an axiom "
        "profile via flags, and test it through `prove_jacobi`'s "
        "generic dispatch path.",
    ),
    (
        "markdown",
        "## Minimum profile, the commutator rule\n\n"
        "`CustomBracket(name, expand_fn, *, degree=..., "
        "is_graded_antisymmetric=..., satisfies_leibniz=..., "
        "satisfies_graded_jacobi=...)`. The `expand_fn` signature is "
        "`(a, b, registry) → Expr`.",
    ),
    (
        "code",
        "from jacopy.brackets.custom import CustomBracket\n"
        "from jacopy.core.expr import Neg, Product, Sum, Symbol\n\n"
        "def commutator(a, b, registry):\n"
        "    return Sum(Product(a, b), Neg(Product(b, a)))\n\n"
        "B = CustomBracket(\"[·,·]\", commutator)\n"
        "print('name:', B.name, 'degree:', B.degree)\n"
        "print('antisym:', B.is_graded_antisymmetric)\n"
        "print('expand X,Y:', B(Symbol('X'), Symbol('Y')).expand())",
    ),
    (
        "markdown",
        "## Axiom flags\n\n"
        "Override the defaults with flags when your bracket has a "
        "different axiom profile. `satisfies_graded_jacobi=None` is "
        "**conditional** Jacobi (as on a derived bracket).",
    ),
    (
        "code",
        "B_asym = CustomBracket(\n"
        "    \"asym\",\n"
        "    lambda a, b, reg: Product(a, b),\n"
        "    is_graded_antisymmetric=False,\n"
        "    satisfies_leibniz=False,\n"
        "    satisfies_graded_jacobi=False,\n"
        ")\n"
        "print('antisym:', B_asym.is_graded_antisymmetric,\n"
        "      'leibniz:', B_asym.satisfies_leibniz,\n"
        "      'jacobi:', B_asym.satisfies_graded_jacobi)",
    ),
    (
        "markdown",
        "## `prove_jacobi`, generic dispatch\n\n"
        "For the commutator rule the chain is bracket-expand → "
        "simplify → 0. A wrong / asymmetric rule leaves a residual "
        "and raises `ProofFailure`.",
    ),
    (
        "code",
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.proof.verifier import prove_jacobi\n"
        "from jacopy.proof.strategies import ProofFailure\n\n"
        "reg = PropertyRegistry()\n"
        "for s in (Symbol('X'), Symbol('Y'), Symbol('Z')):\n"
        "    reg.declare(s, Graded(degree=0))\n\n"
        "chain = prove_jacobi(B, Symbol('X'), Symbol('Y'), Symbol('Z'), registry=reg)\n"
        "print('commutator chain len:', len(chain))\n"
        "for st in chain.steps:\n"
        "    print(' ', st.rule)\n"
        "print('final:', chain.steps[-1].after)\n\n"
        "try:\n"
        "    prove_jacobi(B_asym, Symbol('X'), Symbol('Y'), Symbol('Z'), registry=reg)\n"
        "except ProofFailure as exc:\n"
        "    print('\\nasym rule fails (as expected):')\n"
        "    print(' ', str(exc)[:110])",
    ),
    (
        "markdown",
        "## Axiom-obstruction helpers\n\n"
        "Inherited from `GradedBracket`: each helper returns the "
        "explicit Expr asserted by the axiom. Useful for probing a "
        "rule without going into a full proof.",
    ),
    (
        "code",
        "a, b, c = Symbol('a'), Symbol('b'), Symbol('c')\n"
        "for s in (a, b, c):\n"
        "    reg.declare(s, Graded(degree=0))\n\n"
        "print('antisym obs:', B.graded_antisymmetry_obstruction(a, b, reg))\n"
        "print('jacobi obs :', B.graded_jacobi_obstruction(a, b, c, reg))\n"
        "print('leibniz obs:', B.leibniz_obstruction(a, b, c, reg))",
    ),
    (
        "markdown",
        "## Equality, callable identity\n\n"
        "Two `CustomBracket`s compare equal only if they share the "
        "same `expand_fn` callable.",
    ),
    (
        "code",
        "rule_a = lambda a, b, reg: Sum(Product(a, b), Neg(Product(b, a)))\n"
        "rule_b = lambda a, b, reg: Sum(Product(a, b), Product(b, a))\n"
        "print('same rule:', CustomBracket('B', rule_a) == CustomBracket('B', rule_a))\n"
        "print('diff rule:', CustomBracket('B', rule_a) == CustomBracket('B', rule_b))",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "Generator-based automatic bracket construction, "
        "[07_derived_bracket.md](07_derived_bracket.md).",
    ),
]


TUTORIAL_07: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 07, Derived bracket\n\n"
        "Companion notebook to [07_derived_bracket.md](07_derived_bracket.md). "
        "The construction `{a, b}_Q := [[a, Q]_base, b]_base`, Jacobi "
        "reduced to a single equation (`[Q, Q]_base = 0`), Koszul "
        "equivalence, and the Poisson / H-twisted Courant corners.",
    ),
    (
        "markdown",
        "## The construction\n\n"
        "`DerivedBracket(base, Q, degree_Q=...)`, pick a degree-1 `Q` "
        "on a Lie base. `|{·,·}_Q| = |Q| − 2 = −1`. Leibniz is "
        "universal; antisymmetry / Jacobi are conditional (flag=None).",
    ),
    (
        "code",
        "from jacopy.brackets.derived import DerivedBracket\n"
        "from jacopy.brackets.lie import LieBracket\n"
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "Q = Symbol('Q')\n"
        "reg.declare(Q, Graded(degree=1))\n"
        "lie = LieBracket()\n\n"
        "d = DerivedBracket(lie, Q, degree_Q=1)\n"
        "print('name   :', d.name)\n"
        "print('degree :', d.degree)\n"
        "print('antisym:', d.is_graded_antisymmetric,\n"
        "      'leibniz:', d.satisfies_leibniz,\n"
        "      'jacobi :', d.satisfies_graded_jacobi)",
    ),
    (
        "markdown",
        "## Two-faced expansion\n\n"
        "`expand`, both inner and outer base layers resolved; "
        "`expand_definition`, both layers kept as inert "
        "`BracketApply` nodes.",
    ),
    (
        "code",
        "a, b = Symbol('a'), Symbol('b')\n"
        "for s in (a, b):\n"
        "    reg.declare(s, Graded(degree=0))\n\n"
        "print('expand           :', d.expand(a, b, reg))\n"
        "print('expand_definition:', d.expand_definition(a, b, reg))",
    ),
    (
        "markdown",
        "## Jacobi obstruction, three faces\n\n"
        "A single condition: `[Q, Q]_base = 0`. On a Lie base it's "
        "trivial (`Q*Q − Q*Q`).",
    ),
    (
        "code",
        "print('expanded :', d.jacobi_obstruction(reg))\n"
        "print('raw      :', d.jacobi_obstruction_raw())\n"
        "cond = d.jacobi_condition(reg)\n"
        "print('condition:', cond.name)\n"
        "print('holds?   :', cond.holds(reg))",
    ),
    (
        "markdown",
        "## `prove_jacobi`, `DerivedBracketStrategy`\n\n"
        "Auto-dispatched on bracket type. Three steps: "
        "`DerivedBracketTheorem` → `base-bracket-expand` → `simplify`.",
    ),
    (
        "code",
        "from jacopy.proof.verifier import prove_jacobi\n\n"
        "# a, b already declared Graded(0) above, add c.\n"
        "c = Symbol('c')\n"
        "reg.declare(c, Graded(degree=0))\n\n"
        "chain = prove_jacobi(d, a, b, c, registry=reg)\n"
        "print('chain len:', len(chain))\n"
        "for st in chain.steps:\n"
        "    print(' ', st.rule)\n"
        "print('final:', chain.steps[-1].after)",
    ),
    (
        "markdown",
        "## `acting_on`, Koszul equivalence\n\n"
        "SN base + π generator + anchor ρ: `expand` auto-emits the "
        "Koszul three-term form. Structurally equal to "
        "`KoszulBracket(ρ).expand`.",
    ),
    (
        "code",
        "from jacopy.brackets.schouten import sn\n"
        "from jacopy.brackets.koszul import KoszulBracket\n"
        "from jacopy.calculus.anchor import Anchor\n\n"
        "reg2 = PropertyRegistry()\n"
        "pi = Symbol('π')\n"
        "reg2.declare(pi, Graded(degree=1))\n"
        "alpha, beta = Symbol('α'), Symbol('β')\n"
        "for s in (alpha, beta):\n"
        "    reg2.declare(s, Graded(degree=1))\n\n"
        "rho = Anchor('ρ')\n"
        "koszul_derived = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho)\n"
        "koszul_classical = KoszulBracket(rho)\n\n"
        "lhs = koszul_derived.expand(alpha, beta)\n"
        "rhs = koszul_classical.expand(alpha, beta)\n"
        "print('derived :', lhs)\n"
        "print('classic :', rhs)\n"
        "print('equal?  :', lhs == rhs)",
    ),
    (
        "markdown",
        "## Poisson-as-derived, the library wrapper\n\n"
        "Mathematically `DerivedBracket(sn, π, degree_Q=1)` is the "
        "Poisson bracket. The generic `prove_jacobi` reduces the "
        "obstruction to `[·,·]_SN(π, π)` and surfaces it as an "
        "honest `ProofFailure`, the Poisson hypothesis "
        "`[π, π]_SN = 0` must be carried as an explicit assumption. "
        "`PoissonBracket.prove_jacobi_reduction` closes it in one "
        "step via the seeded theorem citation.",
    ),
    (
        "code",
        "from jacopy.library import theorem_book\n"
        "from jacopy.library.declarations import Bivector, Functions\n"
        "from jacopy.library.poisson import PoissonBracket\n\n"
        "reg3 = PropertyRegistry()\n"
        "pi3 = Bivector('π', registry=reg3)\n"
        "f, g, h = Functions('f g h', degree=-1, registry=reg3)\n\n"
        "poisson = PoissonBracket.from_bivector(pi3)\n"
        "chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg3)\n"
        "print('reduction chain len:', len(chain))\n"
        "print('  rule  :', chain.steps[0].rule)\n"
        "print('  after :', chain.steps[0].after)\n\n"
        "thm = theorem_book.get('poisson_jacobi')\n"
        "print('theorem from_axioms:', thm.from_axioms)",
    ),
    (
        "markdown",
        "## H-twist, the Courant conditional Jacobi\n\n"
        "`CourantBracket(background_H=H)`: Jacobi ⟺ `dH = 0`. The "
        "untwisted default (`H=None`) is vacuous.",
    ),
    (
        "code",
        "from jacopy.brackets.courant import CourantBracket\n\n"
        "reg4 = PropertyRegistry()\n"
        "H = Symbol('H')\n"
        "reg4.declare(H, Graded(degree=3))\n\n"
        "print('untwisted:', CourantBracket().jacobi_condition(reg4).name)\n\n"
        "C = CourantBracket(background_H=H)\n"
        "print('twisted  :', C.is_twisted)\n"
        "cond = C.jacobi_condition(reg4)\n"
        "print('  name       :', cond.name)\n"
        "print('  obstruction:', cond.obstruction)",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "Stage D: the unified picture + foundations, "
        "[08_unified_picture.md](08_unified_picture.md).",
    ),
]


TUTORIAL_08: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 08, The unified picture\n\n"
        "Companion notebook to [08_unified_picture.md](08_unified_picture.md). "
        "A single hypothesis `[π, π]_SN = 0`, same obstruction at both "
        "the function and form levels. Theorem Book citation chain. "
        "A parallel: `dH = 0`.",
    ),
    (
        "markdown",
        "## One hypothesis, two faces\n\n"
        "`jacobi_condition` and `koszul_jacobi_condition` point to the "
        "same `Expr`, only the display name differs.",
    ),
    (
        "code",
        "from jacopy.library.declarations import Bivector, Forms, Functions\n"
        "from jacopy.library.poisson import PoissonBracket\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "pi = Bivector('π', registry=reg)\n"
        "poisson = PoissonBracket.from_bivector(pi)\n\n"
        "c1 = poisson.jacobi_condition(reg)\n"
        "c2 = poisson.koszul_jacobi_condition(reg)\n"
        "print('func obs:', c1.obstruction)\n"
        "print('form obs:', c2.obstruction)\n"
        "print('same expr?:', c1.obstruction == c2.obstruction)",
    ),
    (
        "markdown",
        "## Function vs form, meeting at the same obstruction",
    ),
    (
        "code",
        "f, g, h = Functions('f g h', degree=-1, registry=reg)\n"
        "alpha, beta, gamma = Forms('α β γ', degree=1, registry=reg)\n\n"
        "func_chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)\n"
        "form_chain = poisson.prove_koszul_jacobi_reduction(\n"
        "    alpha, beta, gamma, registry=reg\n"
        ")\n\n"
        "print('func rule:', func_chain.steps[0].rule)\n"
        "print('form rule:', form_chain.steps[0].rule)\n"
        "print('func after:', func_chain.steps[0].after)\n"
        "print('form after:', form_chain.steps[0].after)\n"
        "print('same after?:', func_chain.steps[0].after == form_chain.steps[0].after)",
    ),
    (
        "markdown",
        "## The classical–derived bridge, the `reflexive` step\n\n"
        "The package's expand rules bring both sides to the same Expr "
        "tree; Koszul equivalence closes structurally.",
    ),
    (
        "code",
        "chain_eq = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)\n"
        "print('len:', len(chain_eq), 'rule:', chain_eq.steps[0].rule)",
    ),
    (
        "markdown",
        "## Seeded theorems, the citation chain",
    ),
    (
        "code",
        "from jacopy.library import theorem_book\n\n"
        "for name in (\n"
        "    'poisson_jacobi',\n"
        "    'poisson_koszul_equivalence',\n"
        "    'poisson_koszul_jacobi',\n"
        "):\n"
        "    thm = theorem_book.get(name)\n"
        "    print(name)\n"
        "    for ax in thm.from_axioms:\n"
        "        print('   -', ax)",
    ),
    (
        "markdown",
        "## `dH = 0`, the same pattern on the Courant side",
    ),
    (
        "code",
        "from jacopy.brackets.courant import CourantBracket\n"
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.properties import Graded\n\n"
        "reg_h = PropertyRegistry()\n"
        "H = Symbol('H')\n"
        "reg_h.declare(H, Graded(degree=3))\n\n"
        "C = CourantBracket(background_H=H)\n"
        "cond = C.jacobi_condition(reg_h)\n"
        "print('name       :', cond.name)\n"
        "print('obstruction:', cond.obstruction)\n\n"
        "twist = theorem_book.get('courant_jacobi_twist')\n"
        "print('twist from_axioms:', twist.from_axioms)",
    ),
    (
        "markdown",
        "## Next step\n\n"
        "Where does the axiom *itself* come from? "
        "[09_foundations.md](09_foundations.md).",
    ),
]


TUTORIAL_09: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 09, Foundations\n\n"
        "Companion notebook to [09_foundations.md](09_foundations.md). "
        "Axiom vs theorem classification, efficient vs foundational "
        "mode, and `d² = 0` derived from the generator-level axiom.",
    ),
    (
        "markdown",
        "## Default engine, every rule is an axiom",
    ),
    (
        "code",
        "from jacopy.proof.expansion import default_engine\n\n"
        "eng = default_engine()\n"
        "for d in eng.definitions:\n"
        "    label = 'theorem' if d.is_theorem else 'axiom'\n"
        "    print(f'{label:<8} | {d.name}')",
    ),
    (
        "markdown",
        "## `d_squared_mode=\"theorem\"`, reclassify",
    ),
    (
        "code",
        "eng_th = default_engine(d_squared_mode='theorem')\n"
        "for d in eng_th.definitions:\n"
        "    if d.name == 'd² = 0':\n"
        "        print('is_theorem    :', d.is_theorem)\n"
        "        print('has builder?  :', d.theorem_proof_builder() is not None)",
    ),
    (
        "markdown",
        "## Efficient vs foundational, same step, different sub-proof\n\n"
        "Efficient mode carries no children; foundational mode attaches "
        "a generator-level citation when a theorem-class rule fires.",
    ),
    (
        "code",
        "from jacopy.calculus.invariant_d import default_d\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.core.expr import Symbol, Integer\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.proof.verifier import prove_equivalence\n\n"
        "reg = PropertyRegistry()\n"
        "omega = Symbol('ω')\n"
        "reg.declare(omega, Graded(degree=2))\n"
        "expr = default_d(default_d(omega))\n\n"
        "eng_eff = default_engine(registry=reg, mode='efficient',\n"
        "                         d_squared_mode='theorem')\n"
        "eng_fnd = default_engine(registry=reg, mode='foundational',\n"
        "                         d_squared_mode='theorem')\n\n"
        "eff = prove_equivalence(expr, Integer(0), registry=reg, engine=eng_eff)\n"
        "fnd = prove_equivalence(expr, Integer(0), registry=reg, engine=eng_fnd)\n\n"
        "print('efficient steps:')\n"
        "for s in eff.steps:\n"
        "    print(f'  {s.rule:<15} children={len(s.children)}')\n"
        "print('foundational steps:')\n"
        "for s in fnd.steps:\n"
        "    print(f'  {s.rule:<15} children={len(s.children)}')\n"
        "    for c in s.children:\n"
        "        print(f'     ↳ {c.rule}')",
    ),
    (
        "markdown",
        "## What does the sub-proof say?\n\n"
        "The foundational sub-proof's only input is `d(df) = 0`, "
        "the generic axiom the package treats as primitive. `d² = 0` "
        "extends from it at every form degree.",
    ),
    (
        "code",
        "child = fnd.steps[0].children[0]\n"
        "print('child rule         :', child.rule)\n"
        "print('child justification:\\n  ', child.justification)",
    ),
    (
        "markdown",
        "## Custom Definition, axiom class\n\n"
        "`ExpansionEngine([YourDef()])` lets you assemble your own "
        "axiom set and run the engine on it. Below: a single rule "
        "that drives a `c_zero` symbol to zero.",
    ),
    (
        "code",
        "from jacopy.proof.expansion import Definition, ExpansionEngine\n"
        "from jacopy.core.expr import Sum\n\n"
        "class ZeroConstAxiom(Definition):\n"
        "    name = 'c_zero := 0 (axiom)'\n\n"
        "    def matches(self, expr):\n"
        "        return isinstance(expr, Symbol) and expr.name == 'c_zero'\n\n"
        "    def rewrite(self, expr):\n"
        "        return Integer(0)\n\n"
        "custom_engine = ExpansionEngine([ZeroConstAxiom()])\n"
        "c, x = Symbol('c_zero'), Symbol('x')\n"
        "expanded, steps = custom_engine.expand(Sum(c, x))\n"
        "print('expanded:', expanded)\n"
        "print('rule   :', steps[0].rule)\n"
        "print('prov   :', steps[0].provenance_tag)",
    ),
    (
        "markdown",
        "## Custom Definition, theorem class\n\n"
        "Override `theorem_proof_builder` and the same rule becomes a "
        "theorem; foundational mode attaches the sub-proof.",
    ),
    (
        "code",
        "from jacopy.proof.chain import ProofChain\n"
        "from jacopy.proof.step import ProofStep\n\n"
        "class ZeroConstTheorem(Definition):\n"
        "    name = 'c_zero := 0 (theorem)'\n\n"
        "    def matches(self, expr):\n"
        "        return isinstance(expr, Symbol) and expr.name == 'c_zero'\n\n"
        "    def rewrite(self, expr):\n"
        "        return Integer(0)\n\n"
        "    def theorem_proof_builder(self):\n"
        "        def build(matched):\n"
        "            return ProofChain(steps=[\n"
        "                ProofStep(\n"
        "                    rule='c_zero = c_zero − c_zero (axiom)',\n"
        "                    before=matched, after=Integer(0),\n"
        "                    justification='self-annihilation axiom',\n"
        "                    provenance_tag='axiom',\n"
        "                ),\n"
        "            ])\n"
        "        return build\n\n"
        "eff = ExpansionEngine([ZeroConstTheorem()], mode='efficient')\n"
        "fnd = ExpansionEngine([ZeroConstTheorem()], mode='foundational')\n\n"
        "c = Symbol('c_zero')\n"
        "_, eff_steps = eff.expand(c)\n"
        "_, fnd_steps = fnd.expand(c)\n\n"
        "print('efficient children :', len(eff_steps[0].children))\n"
        "print('foundational child :', len(fnd_steps[0].children))\n"
        "print('  ↳ sub-rule       :', fnd_steps[0].children[0].rule)",
    ),
    (
        "markdown",
        "## Theorem Book structure\n\n"
        "The `Theorem` dataclass carries five fields: `name`, "
        "`statement`, `from_axioms`, `proof`, `notes`. The singleton "
        "`theorem_book` registry holds seeded theorems; downstream "
        "code pulls `proof` and embeds the chain inside a larger "
        "proof.",
    ),
    (
        "code",
        "from jacopy.library.theorem_book import Theorem\n"
        "from jacopy.library import theorem_book\n"
        "import dataclasses\n\n"
        "print('fields:', [f.name for f in dataclasses.fields(Theorem)])\n"
        "print('registry size:', len(theorem_book))\n"
        "print()\n"
        "for name in theorem_book.names():\n"
        "    t = theorem_book.get(name)\n"
        "    print(f'{name:32s} proof_len={len(t.proof)}')",
    ),
    (
        "markdown",
        "## Three provenance layers\n\n"
        "Property (symbol), Definition (expansion), Theorem, all "
        "three carry the `axiom`/`theorem` distinction. "
        "`Theorem.from_axioms` lands a single citation inside a "
        "paper-style proof.",
    ),
    (
        "code",
        "for name in ('poisson_jacobi', 'courant_jacobi_twist'):\n"
        "    thm = theorem_book.get(name)\n"
        "    print(name, '->', thm.from_axioms)",
    ),
    (
        "markdown",
        "## End of the tutorial series\n\n"
        "Nine chapters complete. The package is now a tool ready to "
        "be extended with your own brackets, your own theorems, and "
        "your own axiom sets.",
    ),
]


TUTORIAL_10: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 10, Diagnostics & proof debugging\n\n"
        "Companion notebook to [10_diagnostics.md](10_diagnostics.md). "
        "What to do when `prove_equivalence` raises `ProofFailure`, "
        "read the residual, inspect the diagnostic report, extend the "
        "rule catalogue.",
    ),
    (
        "markdown",
        "## A failing proof\n\n"
        "Build an `ExpansionEngine` deliberately missing the d² = 0 "
        "rule. Forming `d(d(ω)) == 0` should close trivially in the "
        "default engine but stalls here, so we get a real failure to "
        "look at.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Act\n"
        "from jacopy.calculus.exterior_d import d\n"
        "from jacopy.core.expr import Symbol, Integer\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.proof import prove_equivalence, ProofFailure\n"
        "from jacopy.proof.expansion import (\n"
        "    ExpansionEngine,\n"
        "    LieDerivativeCartanDefinition,\n"
        "    ActOverSumOpDefinition,\n"
        "    IotaSquaredZeroDefinition,\n"
        "    IotaOnZeroFormDefinition,\n"
        "    IotaOnExactOneFormDefinition,\n"
        ")\n\n"
        "reg = PropertyRegistry()\n"
        "omega = Symbol(\"ω\"); reg.declare(omega, Graded(degree=2))\n\n"
        "engine_no_d2 = ExpansionEngine([\n"
        "    LieDerivativeCartanDefinition(),\n"
        "    ActOverSumOpDefinition(),\n"
        "    IotaSquaredZeroDefinition(),\n"
        "    IotaOnZeroFormDefinition(),\n"
        "    IotaOnExactOneFormDefinition(d=d),\n"
        "])\n\n"
        "try:\n"
        "    prove_equivalence(\n"
        "        Act(d, Act(d, omega)),\n"
        "        Integer(0),\n"
        "        registry=reg,\n"
        "        engine=engine_no_d2,\n"
        "    )\n"
        "except ProofFailure as exc:\n"
        "    print(exc)",
    ),
    (
        "markdown",
        "The exception message is two-tier, a one-line summary "
        "(`ExpandAndSimplify left residual ...`) followed by the "
        "structured `DiagnosticReport` block. The summary tells you "
        "*that* the proof stalled; the report tells you *why* the "
        "engine couldn't reduce the residual further.",
    ),
    (
        "markdown",
        "## Reading the report\n\n"
        "`ProofFailure.report` is a `DiagnosticReport` (or `None`). It "
        "carries `report.residual` (the surviving `Expr`) and a list of "
        "`DiagnosticHint`s, each with `category`, `message`, "
        "`location`, and `suggestion` fields.",
    ),
    (
        "code",
        "try:\n"
        "    prove_equivalence(\n"
        "        Act(d, Act(d, omega)), Integer(0),\n"
        "        registry=reg, engine=engine_no_d2,\n"
        "    )\n"
        "except ProofFailure as exc:\n"
        "    report = exc.report\n"
        "    print(f\"residual : {report.residual}\")\n"
        "    print(f\"hints    : {len(report)}\")\n"
        "    for hint in report:\n"
        "        print(f\"  [{hint.category}] {hint.message}\")\n"
        "        if hint.suggestion is not None:\n"
        "            print(f\"    fix: {hint.suggestion}\")",
    ),
    (
        "markdown",
        "## Calling `diagnose()` directly\n\n"
        "You don't need a `ProofFailure` to use the diagnostic layer. "
        "`diagnose(expr, registry=...)` runs the full rule catalogue "
        "against any expression, useful for inspecting intermediate "
        "terms.",
    ),
    (
        "code",
        "from jacopy.core.expr import Product\n"
        "from jacopy.proof import diagnose\n\n"
        "mystery = Symbol(\"M\")  # never registered\n"
        "report = diagnose(Product(mystery, omega), registry=reg)\n"
        "print(report.format())",
    ),
    (
        "markdown",
        "## Built-in rule catalogue\n\n"
        "Rules in `jacopy.proof.diagnostic_rules` register themselves "
        "on import. Each fires on a specific *stalled shape*:\n\n"
        "| Category | Trigger |\n|---|---|\n"
        "| `stalled-d-squared` | `Act(d, Act(d, x))` survived |\n"
        "| `stalled-iota-squared` | `Act(ι_X, Act(ι_X, x))` survived |\n"
        "| `stalled-act-over-zero` | `Act(op, 0)` reached the residual |\n"
        "| `stalled-act-over-neg-op` | `Act(Neg(op), x)` failed to peel |\n"
        "| `unreduced-iota-on-df` | `ι_V(d(f))` with V a sum/product of derivations |\n"
        "| `unclassified-factor` | a `Product` factor has no grading evidence |\n"
        "| `symbol-vector-field` | a bare `Symbol` plays a vector-field role |\n\n"
        "Filter to a single category with `report.by_category(...)`.",
    ),
    (
        "code",
        "# Trigger several categories side by side, just by passing\n"
        "# different residual shapes through diagnose().\n"
        "from jacopy.core.expr import Sum, Neg\n\n"
        "shapes = [\n"
        "    (\"d² stall\",     Act(d, Act(d, omega))),\n"
        "    (\"Act on 0\",     Act(d, Integer(0))),\n"
        "    (\"unclassified\", Product(Symbol(\"M\"), omega)),\n"
        "]\n"
        "for label, expr in shapes:\n"
        "    rep = diagnose(expr, registry=reg)\n"
        "    cats = sorted({h.category for h in rep})\n"
        "    print(f\"{label:14s} → categories: {cats}\")",
    ),
    (
        "markdown",
        "## Adding your own rule\n\n"
        "A rule is `(expr, registry, engine) → Iterable[DiagnosticHint]`. "
        "Register it with `@register_rule`, it joins the catalogue "
        "immediately, no engine wiring needed because diagnostics are "
        "read-only on the residual tree.",
    ),
    (
        "code",
        "from jacopy.proof import DiagnosticHint, register_rule\n\n"
        "@register_rule\n"
        "def warn_on_unregistered_omega(expr, registry, engine):\n"
        "    \"\"\"Toy rule: flag bare 'ω' that lost its Graded declaration.\"\"\"\n"
        "    if registry is None:\n"
        "        return\n"
        "    stack = [expr]\n"
        "    while stack:\n"
        "        cur = stack.pop()\n"
        "        if isinstance(cur, Symbol) and cur.name == \"ω\":\n"
        "            if not registry.has(cur, Graded):\n"
        "                yield DiagnosticHint(\n"
        "                    category=\"omega-unregistered\",\n"
        "                    message=\"ω appears without a registered grading\",\n"
        "                    location=cur,\n"
        "                    suggestion=\"reg.declare(ω, Graded(degree=...))\",\n"
        "                )\n"
        "        stack.extend(getattr(cur, \"children\", ()))\n\n"
        "# Exercise the new rule on an empty registry\n"
        "empty = PropertyRegistry()\n"
        "report = diagnose(omega, registry=empty)\n"
        "for h in report:\n"
        "    if h.category == \"omega-unregistered\":\n"
        "        print(h)",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `prove_equivalence` raises `ProofFailure`; `exc.report` "
        "is a `DiagnosticReport`.\n"
        "* The report bundles `residual` + a list of `DiagnosticHint`s "
        "with `category` / `message` / `location` / `suggestion`.\n"
        "* `diagnose(expr, registry=...)` runs the same pipeline "
        "directly on any `Expr`.\n"
        "* Built-in catalogue covers d² / ι² stalls, `Act` linearity "
        "gaps, unreduced iota-on-df, and unclassified factors.\n"
        "* Extend the catalogue with `@register_rule`, pure tree-walk "
        "function, no engine plumbing.",
    ),
]


TUTORIAL_11: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 11, Publication-ready output\n\n"
        "Companion notebook to "
        "[11_publication_output.md](11_publication_output.md). "
        "`chain_to_latex` / `chain_to_tikz` and their `_document` "
        "wrappers turn a `ProofChain` into LaTeX you can paste into a "
        "paper or compile into a standalone PDF.",
    ),
    (
        "markdown",
        "## A worked chain\n\n"
        "Tiny but real proof: `d(d(ω)) == 0`. Two steps, enough to "
        "exercise the renderers without flooding the output.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Act\n"
        "from jacopy.calculus.exterior_d import d\n"
        "from jacopy.core.expr import Symbol, Integer\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.proof import prove_equivalence\n\n"
        "reg = PropertyRegistry()\n"
        "omega = Symbol(\"ω\"); reg.declare(omega, Graded(degree=2))\n\n"
        "chain = prove_equivalence(\n"
        "    Act(d, Act(d, omega)),\n"
        "    Integer(0),\n"
        "    registry=reg,\n"
        ")\n"
        "print(f\"steps: {len(chain)}\")\n"
        "for s in chain.steps:\n"
        "    print(f\"  [{s.rule}] {s.before} → {s.after}\")",
    ),
    (
        "markdown",
        "## Inline LaTeX\n\n"
        "`chain_to_latex(chain)` returns a `gather*` block, one math "
        "row per step, wrapped in `\\allowdisplaybreaks\\scriptsize` "
        "so long chains page-break properly. Paste straight into any "
        "`.tex` source that loads `amsmath`.",
    ),
    (
        "code",
        "from jacopy.display import chain_to_latex\n\n"
        "body = chain_to_latex(chain)\n"
        "print(body)",
    ),
    (
        "markdown",
        "Each row uses `\\to` (not `=`), explicit rewriting direction. "
        "The `\\quad \\text{[rule] (provenance)}` annotation reads as the "
        "step's justification.",
    ),
    (
        "markdown",
        "## Standalone document\n\n"
        "`chain_to_latex_document` adds a full `\\documentclass{article}` "
        "preamble plus optional `\\title`/`\\author`/`\\maketitle`. Write "
        "to disk and `pdflatex out.tex` compiles directly.",
    ),
    (
        "code",
        "from jacopy.display import chain_to_latex_document\n\n"
        "doc = chain_to_latex_document(\n"
        "    chain,\n"
        "    title=\"d² = 0\",\n"
        "    author=\"jacopy\",\n"
        ")\n"
        "print(doc[:400])\n"
        "print(\"...\")\n"
        "print(doc[-80:])",
    ),
    (
        "markdown",
        "`preamble_extras=\"...\"` splices project-specific macros "
        "(e.g. `\\input{macros.tex}`) between the default preamble and "
        "`\\begin{document}`.",
    ),
    (
        "markdown",
        "## TikZ diagram\n\n"
        "`chain_to_tikz` renders the same chain as a vertical "
        "`tikzpicture`. Each `before`/`after` becomes a boxed node, "
        "`n + 1` nodes for `n` steps, with arrows labelled by rule "
        "name and provenance tag.",
    ),
    (
        "code",
        "from jacopy.display import chain_to_tikz\n\n"
        "diagram = chain_to_tikz(chain)\n"
        "print(diagram)",
    ),
    (
        "markdown",
        "## Standalone TikZ document\n\n"
        "`chain_to_tikz_document` wraps the diagram in a full document "
        "with the `tikz` + `positioning` preamble.",
    ),
    (
        "code",
        "from jacopy.display import chain_to_tikz_document\n\n"
        "doc = chain_to_tikz_document(\n"
        "    chain,\n"
        "    title=\"d² = 0 (diagram)\",\n"
        "    node_distance=\"1.4cm\",\n"
        ")\n"
        "print(doc[:400])",
    ),
    (
        "markdown",
        "`node_distance` tunes vertical spacing, bump it up when "
        "expression labels crowd each other.",
    ),
    (
        "markdown",
        "## Round-trip to PDF (optional)\n\n"
        "Both `_document` outputs compile with any LaTeX engine. The "
        "cell below is illustrative, skip executing it unless "
        "`pdflatex` is on your PATH.",
    ),
    (
        "code",
        "import shutil\n\n"
        "if shutil.which(\"pdflatex\") is None:\n"
        "    print(\"pdflatex not found on PATH, skipping the round-trip demo.\")\n"
        "else:\n"
        "    import subprocess, tempfile\n"
        "    from pathlib import Path\n"
        "    with tempfile.TemporaryDirectory() as tmp:\n"
        "        src = Path(tmp) / \"proof.tex\"\n"
        "        src.write_text(doc, encoding=\"utf-8\")\n"
        "        subprocess.run([\n"
        "            \"pdflatex\", \"-interaction=nonstopmode\", str(src),\n"
        "        ], cwd=tmp, check=True, capture_output=True)\n"
        "        print(sorted(p.name for p in Path(tmp).iterdir()))",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* Four helpers, two inline (`chain_to_latex`, "
        "`chain_to_tikz`), two standalone (`chain_to_latex_document`, "
        "`chain_to_tikz_document`).\n"
        "* Inline: paste into existing `.tex`. Standalone: write + "
        "`pdflatex`.\n"
        "* `title` / `author` / `preamble_extras` are kwargs on the "
        "`_document` variants, empty strings produce a body-only "
        "document.\n"
        "* Nested sub-proofs flatten in both renderers; compose by hand "
        "if you want a tree.",
    ),
]


TUTORIAL_14: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 14, Solving textbook problems with Problem wrappers\n\n"
        "Companion notebook to "
        "[14_problem_wrappers.md](14_problem_wrappers.md). The wrapper "
        "layer (`SymplecticProblem`, `KoszulProblem`, `BianchiProblem`, "
        "...) bundles `(structure, designated operands, registry, "
        "engine)` so you don't re-wire the same axioms for every "
        "related question.",
    ),
    (
        "markdown",
        "## Setting up a `SymplecticProblem`\n\n"
        "Three inputs: the symplectic form `ω`, the functions you want "
        "as Hamiltonians, and a registry. Everything else is auto-wired "
        ", `Closed(ω)` and `NonDegenerate(ω)` are declared on the "
        "registry, `X_f`/`X_g` are built with their defining relation "
        "`ι_{X_f} ω = ±df` registered on the engine.",
    ),
    (
        "code",
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.properties import Graded, Scalar\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.library.symplectic_problem import SymplecticProblem\n\n"
        "reg = PropertyRegistry()\n"
        "omega = Symbol(\"ω\"); reg.declare(omega, Graded(degree=2))\n"
        "f = Symbol(\"f\"); reg.declare(f, Scalar())\n"
        "g = Symbol(\"g\"); reg.declare(g, Scalar())\n\n"
        "prob = SymplecticProblem(omega, [f, g], registry=reg)\n"
        "print(prob)",
    ),
    (
        "markdown",
        "## Question 2a, Hamiltonian invariance\n\n"
        "The canonical first question on a symplectic form: "
        "`L_{X_f} ω = 0`. Cartan magic + closed form + d² = 0 close "
        "this in eight named steps.",
    ),
    (
        "code",
        "chain = prob.prove_hamiltonian_invariance(f)\n"
        "print(f\"closure: {chain.initial} → {chain.final}\")\n"
        "print(f\"steps  : {len(chain)}\")\n"
        "for i, step in enumerate(chain.steps):\n"
        "    print(f\"  [{i:02d}] {step.rule:35s} {step.before} → {step.after}\")",
    ),
    (
        "markdown",
        "Every step has a named rule, `L_X := d∘ι_X + ι_X∘d`, "
        "`Closed: d(ω) = 0`, `d² = 0`, ..., so the chain reads end-to-"
        "end as the textbook computation.",
    ),
    (
        "markdown",
        "## Question 2c, Hamiltonian equality\n\n"
        "Two helpers close the family: `prove_vector_field_equality` "
        "(reduces `Y = Z` via non-degeneracy) and "
        "`prove_hamiltonian_equality` (closes `ι_Y ω = ±dh`). The "
        "simplest demonstration is the reflexive case, each Hamiltonian "
        "equals itself.",
    ),
    (
        "code",
        "Xf = prob.hamiltonian(f)\n"
        "chain = prob.prove_vector_field_equality(Xf, Xf)\n"
        "print(f\"steps: {len(chain)}, final: {chain.final}\")",
    ),
    (
        "markdown",
        "Three steps: the obstruction `ι_{X_f} ω − ι_{X_f} ω` peels "
        "through the `NonDegenerate` rule to `X_f − X_f`, which "
        "`simplify` collapses to `0`. The non-trivial use is when `Y` "
        "is a Lie bracket of two Hamiltonians and you want it "
        "recognised as a third Hamiltonian, same call, different "
        "operands.",
    ),
    (
        "markdown",
        "## Setting up a `KoszulProblem`\n\n"
        "Form-side counterpart: a Poisson bivector `π` with a list of "
        "forms. The wrapper auto-declares `Antisymmetric(π)` and "
        "exposes the Koszul bracket plus the tilde calculus rules "
        "(L̃ / ι̃ / d̃) on its engine.",
    ),
    (
        "code",
        "from jacopy.library.koszul_problem import KoszulProblem\n\n"
        "reg2 = PropertyRegistry()\n"
        "pi    = Symbol(\"π\"); reg2.declare(pi,    Graded(degree=2))\n"
        "alpha = Symbol(\"α\"); reg2.declare(alpha, Graded(degree=1))\n"
        "beta  = Symbol(\"β\"); reg2.declare(beta,  Graded(degree=1))\n\n"
        "kprob = KoszulProblem(pi, [alpha, beta], registry=reg2)\n"
        "print(kprob)\n"
        "print(f\"bracket: {kprob.koszul_bracket}\")\n"
        "print(f\"sharp  : {kprob.sharp}\")",
    ),
    (
        "markdown",
        "The bracket-expansion rule is exposed directly, useful when "
        "you want to reduce a `[α,β]_K` expression by hand without "
        "running a full proof closure.",
    ),
    (
        "code",
        "from jacopy.brackets.base import BracketApply\n\n"
        "raw = BracketApply(kprob.koszul_bracket, alpha, beta)\n"
        "print(\"input :\", raw)\n"
        "print(\"output:\", kprob.bracket_expansion_rule.rewrite(raw))",
    ),
    (
        "markdown",
        "The rewrite recovers the classical Koszul formula\n"
        "`L_{π^♯α} β − L_{π^♯β} α − d⟨π^♯α, β⟩` term by term.",
    ),
    (
        "markdown",
        "## When to step outside the wrapper\n\n"
        "The wrapper is a convenience, not a wall.\n\n"
        "1. **`prob.engine` is the pre-wired engine**, feed it to "
        "`prove_equivalence(..., engine=prob.engine)` to drive any "
        "equality under the wrapper's axiom set.\n"
        "2. **`prob.hamiltonian(f)` returns the registered "
        "Hamiltonian**, build expressions with it directly.\n"
        "3. **The wrapper never overrides existing declarations**, "
        "pre-declaring is the way to opt out of a default convention.",
    ),
    (
        "code",
        "# Demo: drive an arbitrary equality through prob.engine.\n"
        "from jacopy.algebra.derivation import Act\n"
        "from jacopy.calculus.exterior_d import d\n"
        "from jacopy.core.expr import Integer\n"
        "from jacopy.proof import prove_equivalence\n\n"
        "# d(d(f)) = 0, the engine knows d² = 0 because the wrapper\n"
        "# inherited it from default_engine.\n"
        "chain = prove_equivalence(\n"
        "    Act(d, Act(d, f)), Integer(0),\n"
        "    registry=prob._registry, engine=prob._engine,\n"
        ")\n"
        "print(f\"closure: {chain.initial} → {chain.final} ({len(chain)} steps)\")",
    ),
    (
        "markdown",
        "## Anatomy of the wrapper family\n\n"
        "| Wrapper | Bundles |\n"
        "|---|---|\n"
        "| `SymplecticProblem` | `(M, ω, π?, {f_i}, registry, engine)` |\n"
        "| `KoszulProblem` | `(π, ρ = π^♯, K, {α_i})` + tilde calculus |\n"
        "| `BianchiProblem` | `(connection, registry)`, T̃-Bianchi I/II |\n"
        "| `CartanFormPropertyProblem` | `(connection, frame)`, §3.1.6 |\n"
        "| `CartanStructureProblem` | `(connection, frame)`, Cartan I/II |\n"
        "| `KoszulConnectionProblem` | facade over the three above |\n\n"
        "Each `__init__` validates inputs + declares structural axioms; "
        "properties expose the underlying objects; `prove_*` methods "
        "drive closures via a pre-built engine.",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* Problem wrappers bundle `(structure, designated operands, "
        "registry, engine)` so you don't re-wire axioms every "
        "question.\n"
        "* `SymplecticProblem` covers form-side problems, Hamiltonian "
        "invariance, vector-field equality, Hamiltonian equality.\n"
        "* `KoszulProblem` covers Poisson form-bracket problems, "
        "`[α, β]_K` expansion, tilde calculus, derivator engines.\n"
        "* The pre-built engine (`prob.engine`) is exposed for "
        "closures outside the named helpers.\n"
        "* Wrappers never override existing registry declarations, "
        "pre-declaring is the override mechanism.",
    ),
]


TUTORIAL_12: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 12, The Schouten–Nijenhuis bracket\n\n"
        "Companion notebook to "
        "[12_schouten_nijenhuis.md](12_schouten_nijenhuis.md). "
        "`[·,·]_SN` extends the Lie bracket to multivector fields with "
        "the shifted grading `|X| = k − 1`, and supplies the universal "
        "Poisson obstruction `[π, π]_SN`.",
    ),
    (
        "markdown",
        "## Shifted grading\n\n"
        "Functions are SN-degree `−1`, vector fields are SN-degree `0`, "
        "bivectors are SN-degree `1`. Use `Graded(degree=...)` to "
        "declare these, **not `Scalar()`**, which the SN engine reads "
        "as `degree=0` (i.e. a 1-vector).",
    ),
    (
        "code",
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "X = Symbol(\"X\"); reg.declare(X, Graded(degree=0))\n"
        "Y = Symbol(\"Y\"); reg.declare(Y, Graded(degree=0))\n"
        "f = Symbol(\"f\"); reg.declare(f, Graded(degree=-1))\n"
        "g = Symbol(\"g\"); reg.declare(g, Graded(degree=-1))",
    ),
    (
        "markdown",
        "## Four base cases\n\n"
        "`sn.expand(a, b, registry)` returns the closed form when both "
        "operands are atomic and SN-degrees are concrete integers in "
        "{−1, 0}. These are the characterising rules:\n\n"
        "* `[X, Y]_SN = X*Y − Y*X`, the Lie bracket on 1-vectors.\n"
        "* `[f, g]_SN = 0`, `C^∞` is commutative.\n"
        "* `[f, X]_SN = −X(f)`, graded antisymmetric of the next.\n"
        "* `[X, f]_SN = X(f)`, vectors act as derivations on functions.",
    ),
    (
        "code",
        "from jacopy.brackets.schouten import sn\n\n"
        "print(\"[X, Y]_SN =\", sn.expand(X, Y, reg))\n"
        "print(\"[f, g]_SN =\", sn.expand(f, g, reg))\n"
        "print(\"[f, X]_SN =\", sn.expand(f, X, reg))\n"
        "print(\"[X, f]_SN =\", sn.expand(X, f, reg))",
    ),
    (
        "markdown",
        "## Wedge Leibniz\n\n"
        "On `Product(X, Y)` (= `X ∧ Y`) the bracket descends via\n\n"
        "```\n"
        "[X ∧ Y, Z]_SN = X ∧ [Y, Z]_SN + (−1)^{|Y||Z|} [X, Z]_SN ∧ Y\n"
        "```\n\n"
        "so a 2-vector against a function reads as `X ∧ Y(f) + X(f) ∧ Y` "
        "(both signs `+1` because `|Y|·|f| = 0·(−1) = 0` is even).",
    ),
    (
        "code",
        "from jacopy.core.expr import Product\n\n"
        "bivec = Product(X, Y)              # X ∧ Y\n"
        "print(\"[X ∧ Y, f]_SN =\", sn.expand(bivec, f, reg))",
    ),
    (
        "markdown",
        "## Atomic higher-order multivectors stay opaque\n\n"
        "A bare `Symbol` declared `Graded(degree=1)` plays the role of "
        "an atomic bivector, there's no wedge to peel, so SN can't "
        "descend. Rather than raise, `expand` returns the inert "
        "`BracketApply` node. That handle is exactly what makes "
        "`[π, π]_SN = 0` a usable hypothesis.",
    ),
    (
        "code",
        "pi = Symbol(\"π\"); reg.declare(pi, Graded(degree=1))\n"
        "obstr = sn.self_bracket(pi, reg)\n"
        "print(\"[π, π]_SN =\", obstr)\n"
        "print(\"type      :\", type(obstr).__name__)",
    ),
    (
        "markdown",
        "## Bridge to `PoissonBracket`\n\n"
        "`PoissonBracket(π)` exposes the same obstruction via "
        "`jacobi_obstruction()` and the textbook statement via "
        "`jacobi_condition()`. The killer move is "
        "`prove_jacobi_reduction(f, g, h)`, the cyclic Jacobi sum "
        "collapses to `[·,·]_SN(π, π)` in **one** proof step (the "
        "Derived Bracket Theorem).",
    ),
    (
        "code",
        "from jacopy.library.poisson import PoissonBracket\n\n"
        "h = Symbol(\"h\"); reg.declare(h, Graded(degree=-1))\n\n"
        "P = PoissonBracket(pi)\n"
        "print(\"obstruction:\", P.jacobi_obstruction())\n"
        "print(\"condition  :\", P.jacobi_condition())\n\n"
        "chain = P.prove_jacobi_reduction(f, g, h, registry=reg)\n"
        "print(f\"initial: {chain.initial}\")\n"
        "print(f\"final  : {chain.final}\")\n"
        "print(f\"steps  : {len(chain)}  rule: {chain.steps[0].rule}\")",
    ),
    (
        "markdown",
        "## When SN stays inert\n\n"
        "Three situations:\n\n"
        "1. **Atomic higher-order multivector**, useful opacity, the "
        "handle drives the proof.\n"
        "2. **Symbolic SN-degree**, if any operand has a non-integer "
        "`Graded` degree, the wedge-Leibniz parity can't be decided.\n"
        "3. **Forms**, `sn.expand(α, π)` for a form `α` is **not "
        "defined**. SN is the multivector-only bracket; the form-level "
        "operation is the Koszul bracket via "
        "`DerivedBracket(sn, π, acting_on=Sharp(π))` (tutorial 7).",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `sn = SchoutenBracket()`, graded Lie bracket of degree 0 in "
        "the shifted grading `|X| = k − 1`.\n"
        "* Four base cases close on 1-vector / function pairs; wedge "
        "Leibniz climbs into `Product`s.\n"
        "* Atomic higher multivectors return an opaque `BracketApply`, "
        "`sn.self_bracket(π)` is the universal Poisson obstruction.\n"
        "* `PoissonBracket.prove_jacobi_reduction` collapses the "
        "cyclic Jacobi sum to that obstruction in one step.",
    ),
]


TUTORIAL_13: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 13, Closure properties & axiom flags\n\n"
        "Companion notebook to "
        "[13_closure_axioms.md](13_closure_axioms.md). The three "
        "closure flags (`Closed`, `Antisymmetric`, `NonDegenerate`) "
        "are declarative facts on the registry that paired engine "
        "rules cash in as rewrite primitives, no per-form "
        "`Definition` subclass needed.",
    ),
    (
        "markdown",
        "## `Closed`, `dω = 0` on demand\n\n"
        "Declare the property; layer `ClosedFormDefinition` onto the "
        "default engine; `Act(d, ω)` rewrites to `0` for every "
        "registered form.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Act\n"
        "from jacopy.calculus.closed_axioms import ClosedFormDefinition\n"
        "from jacopy.calculus.exterior_d import d\n"
        "from jacopy.core.expr import Symbol, Integer\n"
        "from jacopy.core.properties import Graded, Closed\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.proof import prove_equivalence\n"
        "from jacopy.proof.expansion import ExpansionEngine, default_engine\n\n"
        "reg = PropertyRegistry()\n"
        "omega = Symbol(\"ω\")\n"
        "reg.declare(omega, Graded(degree=2))\n"
        "reg.declare(omega, Closed())\n\n"
        "base = default_engine(registry=reg)\n"
        "engine = ExpansionEngine(\n"
        "    list(base.definitions) + [ClosedFormDefinition(registry=reg)]\n"
        ")\n\n"
        "chain = prove_equivalence(Act(d, omega), Integer(0),\n"
        "                          registry=reg, engine=engine)\n"
        "print(f\"d(ω) = 0 in {len(chain)} steps: {chain.steps[0].rule}\")",
    ),
    (
        "markdown",
        "The single named step *is* the proof artefact: the chain "
        "transcript reads as \"because ω is closed\". With "
        "`registry=None` the rule is dormant, a safety hatch, not a "
        "default.",
    ),
    (
        "markdown",
        "## `Antisymmetric`, bivectors with a sign rule\n\n"
        "`Antisymmetric()` flags a binary head whose `MultiEval(head, "
        "α, β)` swap-pair canonicalises to `-head(β, α)`. Typical use: "
        "a Schouten–Nijenhuis bivector `π` whose pairing should sort "
        "args by `repr` order with a sign.",
    ),
    (
        "code",
        "from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition\n"
        "from jacopy.core.multi_eval import MultiEval\n"
        "from jacopy.core.properties import Antisymmetric\n\n"
        "reg2 = PropertyRegistry()\n"
        "pi    = Symbol(\"π\"); reg2.declare(pi, Antisymmetric())\n"
        "alpha = Symbol(\"α\"); reg2.declare(alpha, Graded(degree=1))\n"
        "beta  = Symbol(\"β\"); reg2.declare(beta,  Graded(degree=1))\n\n"
        "engine2 = ExpansionEngine([\n"
        "    RegistryAntiSymCanonicalDefinition(registry=reg2)\n"
        "])\n\n"
        "raw = MultiEval(pi, beta, alpha)  # out of canonical order\n"
        "expanded, steps = engine2.expand(raw)\n"
        "print(f\"input    : {raw}\")\n"
        "print(f\"expanded : {expanded}\")\n"
        "print(f\"rule     : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "Cancellation pattern: `π(α, β) + π(β, α)` collapses to zero "
        "when this rule meets the simplify pipeline.",
    ),
    (
        "code",
        "from jacopy.algorithms.simplify import simplify\n\n"
        "eq = MultiEval(pi, alpha, beta) + MultiEval(pi, beta, alpha)\n"
        "expanded, _ = engine2.expand(eq)\n"
        "print(f\"π(α,β) + π(β,α) → {simplify(expanded, reg2)}\")",
    ),
    (
        "markdown",
        "## `NonDegenerate`, peeling `ι_(·) ω` off both sides\n\n"
        "`NonDegenerate()` encodes injectivity of the bundle map "
        "`X ↦ ι_X ω`. The paired rule fires on a two-term `Sum` "
        "whose children are interior products of the *same* form "
        "against vector fields with opposite signs, exactly the "
        "obstruction shape a vector-field equality produces.",
    ),
    (
        "code",
        "from jacopy.calculus.interior import interior\n"
        "from jacopy.calculus.nondegenerate_axioms import (\n"
        "    NonDegenerateInteriorEqualityDefinition,\n"
        ")\n"
        "from jacopy.core.properties import NonDegenerate\n\n"
        "reg3 = PropertyRegistry()\n"
        "omega = Symbol(\"ω\")\n"
        "reg3.declare(omega, Graded(degree=2))\n"
        "reg3.declare(omega, NonDegenerate())\n\n"
        "Y = Symbol(\"Y\"); reg3.declare(Y, Graded(degree=1))\n"
        "Z = Symbol(\"Z\"); reg3.declare(Z, Graded(degree=1))\n\n"
        "engine3 = ExpansionEngine([\n"
        "    NonDegenerateInteriorEqualityDefinition(registry=reg3)\n"
        "])\n\n"
        "obstruction = Act(interior(Y), omega) - Act(interior(Z), omega)\n"
        "expanded, steps = engine3.expand(obstruction)\n"
        "print(f\"input    : {obstruction}\")\n"
        "print(f\"expanded : {expanded}\")\n"
        "print(f\"rule     : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## Why declarative beats inline\n\n"
        "Each closure rule is registry-aware: at construction it "
        "stores a `PropertyRegistry`, at `matches` time it queries "
        "that registry for the relevant flag. Practical consequences:\n\n"
        "1. **One rule, every form.** A single `ClosedFormDefinition` "
        "handles every form declared `Closed()`, no per-form "
        "`Definition` subclass.\n"
        "2. **Pre-declaring opts out.** Problem wrappers (tutorial 14) "
        "see existing flags and don't re-declare.\n"
        "3. **`registry=None` is a no-op.** Engines without a "
        "registry keep these rules dormant.",
    ),
    (
        "markdown",
        "## When the wrapper does it for you\n\n"
        "| Wrapper | Auto-declares | Wires |\n"
        "|---|---|---|\n"
        "| `SymplecticProblem` | `Closed(ω)`, `NonDegenerate(ω)` | both rules on `prob.engine` |\n"
        "| `KoszulProblem` | `Antisymmetric(π)` | antisym rule + tilde calculus |\n\n"
        "Tutorial 14 walks those wrappers. The point of *this* "
        "tutorial is the layer underneath: when the wrapper isn't a "
        "fit, declare the flag and layer the rule yourself.",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* Three closure properties, `Closed`, `Antisymmetric`, "
        "`NonDegenerate`, each paired with a single registry-aware "
        "engine rule.\n"
        "* Rules constructed with `registry=` keyword; `None` is a "
        "no-op default.\n"
        "* Pre-declaring on the registry is the opt-out mechanism "
        "for the problem-wrapper layer.\n"
        "* The same flag-plus-rule recipe scales to `Poisson`, the "
        "Lie-bracket antisymmetry / Jacobi axioms, and any future "
        "structural fact.",
    ),
]


TUTORIAL_15: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 15, The intrinsic engine\n\n"
        "Companion notebook to "
        "[15_intrinsic_engine.md](15_intrinsic_engine.md). "
        "`intrinsic_engine()` bundles the textbook expansions of "
        "`ι_X`, `L_X`, `d` on a `MultiEval` plus the multi-eval "
        "book-keeping helpers; `prove_intrinsic_equivalence` drives "
        "them to fix-point.",
    ),
    (
        "markdown",
        "## The base bundle\n\n"
        "Seven rules: three intrinsic operator expansions plus four "
        "`MultiEval` helpers. Match order matters, operator-specific "
        "rules fire before head-linearity scans the wrapper.",
    ),
    (
        "code",
        "from jacopy.calculus.intrinsic_engine import intrinsic_engine\n\n"
        "eng = intrinsic_engine()\n"
        "print(f\"rules : {len(eng.definitions)}\")\n"
        "for r in eng.definitions:\n"
        "    print(f\"  - {r.name}\")",
    ),
    (
        "markdown",
        "## `prove_intrinsic_equivalence` on `ι² = 0`\n\n"
        "Vector fields are constructed via `Derivation(name, 0)` "
        "(degree 0), the intrinsic rules look for that shape rather "
        "than a generic `Symbol` with a `Graded` declaration.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Act, Derivation\n"
        "from jacopy.calculus.interior import interior\n"
        "from jacopy.calculus.intrinsic_engine import prove_intrinsic_equivalence\n"
        "from jacopy.core.expr import Symbol, Integer\n"
        "from jacopy.core.multi_eval import multi_eval\n\n"
        "omega = Symbol(\"ω\")\n"
        "X, Y = Derivation(\"X\", 0), Derivation(\"Y\", 0)\n\n"
        "lhs = multi_eval(Act(interior(X), Act(interior(X), omega)), Y)\n"
        "chain = prove_intrinsic_equivalence(lhs, Integer(0))\n"
        "print(f\"ι_X ι_X ω = 0 closes in {len(chain)} steps\")\n"
        "for s in chain.steps:\n"
        "    print(f\"  - {s.rule}\")",
    ),
    (
        "markdown",
        "## Cartan magic on a 2-form\n\n"
        "The flagship 12.A test: `(ι_X d + d ι_X) ω = L_X ω` on a "
        "2-form, evaluated on `(Y, Z)`. The base bundle alone closes "
        "this in twelve named steps.",
    ),
    (
        "code",
        "from jacopy.calculus.exterior_d import d as default_d\n"
        "from jacopy.calculus.lie_derivative import lie_derivative\n"
        "from jacopy.core.expr import Sum\n\n"
        "X, Y, Z = (Derivation(s, 0) for s in (\"X\", \"Y\", \"Z\"))\n\n"
        "lhs = Sum(\n"
        "    multi_eval(Act(interior(X), Act(default_d, omega)), Y, Z),\n"
        "    multi_eval(Act(default_d, Act(interior(X), omega)), Y, Z),\n"
        ")\n"
        "rhs = multi_eval(Act(lie_derivative(X), omega), Y, Z)\n\n"
        "chain = prove_intrinsic_equivalence(lhs, rhs)\n"
        "print(f\"Cartan magic closes in {len(chain)} steps\")",
    ),
    (
        "markdown",
        "## Closure-complete bundle\n\n"
        "`intrinsic_engine_with_closure()` adds four 12.A.6 rules "
        "that fold post-expansion residues, VF-commutator, bracket "
        "antisymmetry / Jacobi, the iota-as-scalar 1-form bridge. "
        "Together they close `[L_X, ι_Y] ω = ι_{[X,Y]_VF} ω` and "
        "`d² = 0` on 1- / 2-forms.",
    ),
    (
        "code",
        "from jacopy.algebra.lie_bracket_vf import lie_bracket_vf\n"
        "from jacopy.calculus.intrinsic_engine import intrinsic_engine_with_closure\n"
        "from jacopy.core.expr import Neg\n\n"
        "XY = lie_bracket_vf(X, Y)\n\n"
        "lhs = Sum(\n"
        "    multi_eval(Act(lie_derivative(X), Act(interior(Y), omega)), Z),\n"
        "    Neg(multi_eval(Act(interior(Y), Act(lie_derivative(X), omega)), Z)),\n"
        ")\n"
        "rhs = multi_eval(Act(interior(XY), omega), Z)\n\n"
        "chain = prove_intrinsic_equivalence(\n"
        "    lhs, rhs, engine=intrinsic_engine_with_closure(),\n"
        ")\n"
        "print(f\"[L_X, ι_Y] ω closes in {len(chain)} steps\")",
    ),
    (
        "markdown",
        "## `IntrinsicFormulaRecognizer`, shape inspection\n\n"
        "Pure-shape inspection of `MultiEval(Act(op, ω), Y_1, …)`: "
        "given an expression, returns the operator label, the form, "
        "the vector field (if any), and the eval slots, without "
        "running any rewriting. Use it when you want to *dispatch* "
        "on shape without committing to a closure.",
    ),
    (
        "code",
        "from jacopy.calculus.intrinsic_engine import IntrinsicFormulaRecognizer\n\n"
        "expr = multi_eval(Act(lie_derivative(X), omega), Y, Z)\n"
        "match = IntrinsicFormulaRecognizer().recognize(expr)\n"
        "print(f\"operator     : {match.operator}\")\n"
        "print(f\"vector_field : {match.vector_field}\")\n"
        "print(f\"omega        : {match.omega}\")\n"
        "print(f\"args         : {match.args}\")",
    ),
    (
        "markdown",
        "## When the intrinsic engine is the wrong choice\n\n"
        "**Use it for:** equalities involving `MultiEval(Act(op, ω), "
        "Y_1, …)` shapes, when you want the textbook intrinsic-"
        "formula transcript.\n\n"
        "**Don't use it for:** generic operator-equation work (raw "
        "`L²`, `d²` without multi-eval), that belongs to "
        "`prove_equivalence` with `default_engine`. Problem-specific "
        "axioms belong in the closure-axiom layer (tutorial 13) or a "
        "problem wrapper (tutorial 14).\n\n"
        "**Known failure mode:** `d²` and `[L_X, L_Y]` on a 3-form or "
        "higher don't close, the 12.A.6 closure axioms are "
        "calibrated for 1- / 2-forms. Tutorial 10 walks the "
        "diagnostic surface for those residues.",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `intrinsic_engine()`, 7 base rules: 3 intrinsic operator "
        "expansions + 4 multi-eval helpers.\n"
        "* `intrinsic_engine_with_closure()`, adds 4 closure rules "
        "to fold post-expansion residues; closes `[L_X, ι_Y] ω`, "
        "`d² = 0`, `[L_X, L_Y] ω` on 1- / 2-forms.\n"
        "* `prove_intrinsic_equivalence`, runs the engine to "
        "fix-point, returns a `ProofChain`.\n"
        "* `IntrinsicFormulaRecognizer`, pure-shape inspector for "
        "intrinsic-operator multi-evals, no rewriting.",
    ),
]


TUTORIAL_16: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 16, Phase 13 deep dive: the `[π, π]_SN` obstruction\n\n"
        "Companion notebook to "
        "[16_phase_13_deep_dive.md](16_phase_13_deep_dive.md). Phase "
        "13 closes the cyclic Poisson Jacobi sum to `[·,·]_SN(π, π)` "
        "*without* citing the seeded `poisson_jacobi` theorem, using "
        "only engine-level rewrite axioms. This walks the machinery: "
        "`LieBracketVF` atom + the four rewrite rules.",
    ),
    (
        "markdown",
        "## `LieBracketVF`, Lie bracket of vector fields as an atom\n\n"
        "`[X, Y]_VF` is an opaque `Derivation` subclass with structural "
        "identity over `(X, Y)`. Why opaque: after the operator-"
        "commutator fold, downstream Cartan rules need a single "
        "derivation, not a `X*Y − Y*X` expansion. Literal expansion "
        "is still available through `LieBracket.expand` for callers "
        "who need it.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Derivation\n"
        "from jacopy.algebra.lie_bracket_vf import lie_bracket_vf\n\n"
        "X = Derivation(\"X\", 0)\n"
        "Y = Derivation(\"Y\", 0)\n\n"
        "bracket = lie_bracket_vf(X, Y)\n"
        "print(f\"name      : {bracket}\")\n"
        "print(f\"X         : {bracket.X}\")\n"
        "print(f\"Y         : {bracket.Y}\")\n"
        "print(f\"degree    : {bracket._degree}\")\n"
        "print(f\"same atom : {bracket == lie_bracket_vf(X, Y)}\")",
    ),
    (
        "markdown",
        "## The vector-field axioms (Faz 13.C)\n\n"
        "`OpCommutatorVfDefinition` folds the operator commutator: "
        "`L_X(L_Y(ω)) − L_Y(L_X(ω)) → L_{[X,Y]_VF}(ω)`. Order-"
        "permissive on the Sum's children, the upstream pipeline "
        "doesn't have to canonicalise first.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Act\n"
        "from jacopy.calculus.lie_derivative import lie_derivative\n"
        "from jacopy.calculus.vf_axioms import OpCommutatorVfDefinition\n"
        "from jacopy.core.expr import Symbol, Sum, Neg\n"
        "from jacopy.proof.expansion import ExpansionEngine\n\n"
        "omega = Symbol(\"ω\")\n"
        "LX, LY = lie_derivative(X), lie_derivative(Y)\n\n"
        "expr = Sum(\n"
        "    Act(LX, Act(LY, omega)),\n"
        "    Neg(Act(LY, Act(LX, omega))),\n"
        ")\n\n"
        "engine = ExpansionEngine([OpCommutatorVfDefinition()])\n"
        "out, steps = engine.expand(expr)\n"
        "print(f\"input  : {expr}\")\n"
        "print(f\"output : {out}\")\n"
        "print(f\"rule   : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "The match is structural, a positive `Act(L_X, Act(L_Y, ω))` "
        "paired with its sign-flipped twin `Neg(Act(L_Y, Act(L_X, "
        "ω)))`. `LieVfJacobiDefinition` is the same kind of matcher "
        "one level up: three cyclically-permuted `Act(L_{[X,[Y,Z]_VF]"
        "_VF}, ω)` terms collapse to `Integer(0)`.",
    ),
    (
        "markdown",
        "## Function-side closure (2g-deep, end-to-end)\n\n"
        "Two axioms in `jacopy.calculus.poisson_axioms`, "
        "`PoissonAsHamiltonianDefinition` (rewrites `{f, g}_π → "
        "X_f(g)` for a *pinned* `DerivedBracket`) and "
        "`HamiltonianCyclicSnFormulaDefinition` (collapses cyclic "
        "`Act(X_a, Act(X_b, c))` to `[·,·]_SN(π, π)`), close the "
        "cyclic Poisson Jacobi sum end-to-end.",
    ),
    (
        "code",
        "from jacopy.brackets.derived import DerivedBracket\n"
        "from jacopy.brackets.schouten import sn\n"
        "from jacopy.calculus.poisson_axioms import (\n"
        "    PoissonAsHamiltonianDefinition,\n"
        "    HamiltonianCyclicSnFormulaDefinition,\n"
        ")\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "pi = Symbol(\"π\"); reg.declare(pi, Graded(degree=1))\n"
        "f = Symbol(\"f\"); reg.declare(f, Graded(degree=-1))\n"
        "g = Symbol(\"g\"); reg.declare(g, Graded(degree=-1))\n"
        "h = Symbol(\"h\"); reg.declare(h, Graded(degree=-1))\n\n"
        "P = DerivedBracket(sn, pi, name=\"[·,·]_π\")\n"
        "obs = P.graded_jacobi_obstruction(f, g, h, registry=reg)\n"
        "print(f\"input : {obs}\")",
    ),
    (
        "code",
        "engine = ExpansionEngine([\n"
        "    PoissonAsHamiltonianDefinition(bracket=P, bivector=pi),\n"
        "    HamiltonianCyclicSnFormulaDefinition(bivector=pi),\n"
        "])\n\n"
        "result, steps = engine.expand(obs)\n"
        "print(f\"steps : {len(steps)}\")\n"
        "for i, s in enumerate(steps):\n"
        "    print(f\"  [{i:02d}] {s.rule}\")\n"
        "print(f\"final : {result}\")",
    ),
    (
        "markdown",
        "Seven steps: six `PoissonAsHamiltonian` rewrites (each cyclic "
        "term `{f, {g, h}}` peels twice, outer then inner, into "
        "`X_f(X_g(h))`) plus one `HamiltonianCyclicSn` collapse to "
        "`−[·,·]_SN(π, π)`. The leading `Neg` is the sign carried in "
        "from `graded_jacobi_obstruction`'s shape, not a sign error.",
    ),
    (
        "markdown",
        "## Form-side asymmetry (2f-deep)\n\n"
        "The form-side chain, cyclic Koszul Jacobi on three 1-forms "
        ", closes through `SnBivectorFormulaDefinition` (Faz 13.D), "
        "but doesn't reach a clean `[·,·]_SN(π, π)` residue without "
        "extra bookkeeping. After Cartan-layer expansion of "
        "`{α, β}_K = L_{π^♯α}β − L_{π^♯β}α − d⟨π^♯α, β⟩` and operator-"
        "commutator folding, three pieces remain:\n\n"
        "1. **Named-bracket cyclic** `Σ_cyc L_{[π^♯·, π^♯·]_VF}(·)`, "
        "what `SnBivectorFormulaDefinition` rewrites.\n"
        "2. **Iterated Lie tails** `L_{π^♯·}(L_{π^♯·}(·))`, separate "
        "cancellation pass.\n"
        "3. **`d⟨·, ·⟩` residues** from the Koszul third term, "
        "don't enter the SN handle directly.\n\n"
        "(2) and (3) are the bookkeeping burden. They cancel "
        "algebraically but need user-driven simplification or extra "
        "rules. The function-side chain doesn't have this burden, "
        "the `Act(X_f, X_g(h))` shape absorbs everything into a "
        "single iterated derivation. The asymmetry is structural: "
        "1-forms carry more Cartan-layer machinery than functions.",
    ),
    (
        "markdown",
        "## When to use the deep machinery\n\n"
        "| Goal | Use |\n"
        "|---|---|\n"
        "| One-step Jacobi reduction citing a seeded theorem | `PoissonBracket.prove_jacobi_reduction` (tutorial 12) |\n"
        "| Step-by-step `LieBracketVF` fold transcript | This tutorial's two-axiom engine, function-side |\n"
        "| Custom derived bracket without a seeded theorem | Pin a `DerivedBracket`, layer the same axioms |\n"
        "| Form-side cancellation with the 3-form pairing | Faz 13.D `SnBivectorFormulaDefinition` + manual residue work |\n\n"
        "Default workflow stays at the seeded-theorem level, "
        "`prove_jacobi_reduction` is shorter and reads as \"by the "
        "Derived Bracket Theorem\". The deeper machinery sits "
        "*underneath* that one-line citation, ready when the seed "
        "doesn't apply.",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `LieBracketVF(X, Y)`, opaque `Derivation` atom, kept "
        "unexpanded for downstream Cartan rule uniformity.\n"
        "* `OpCommutatorVfDefinition` folds operator commutators into "
        "`L_{[X,Y]_VF}`; `LieVfJacobiDefinition` zeros the cyclic "
        "Lie-Jacobi triple.\n"
        "* `PoissonAsHamiltonianDefinition` + "
        "`HamiltonianCyclicSnFormulaDefinition` close the cyclic "
        "Poisson Jacobi sum to `[·,·]_SN(π, π)` in 7 engine steps.\n"
        "* Form-side (2f-deep) needs additional bookkeeping for "
        "iterated Lie tails and `d⟨·, ·⟩` residues, a structural "
        "consequence of 1-forms' deeper Cartan layer.",
    ),
]


TUTORIAL_17: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 17, The tilde calculus\n\n"
        "Companion notebook to "
        "[17_tilde_calculus.md](17_tilde_calculus.md). The tilde "
        "calculus dualises `ι_X`, `L_X`, `d` on a Poisson manifold "
        "`(M, π)` to operators `ι̃_ω`, `L̃_ω`, `d̃` acting on "
        "*multivector* fields, indexed by *forms* (and `π`). Six "
        "Cartan-style identities mirror the form-side ones; this "
        "notebook walks the operator atoms, the three defining "
        "rewrites, and `prove_tilde_cartan_relation`.",
    ),
    (
        "markdown",
        "## The three operator atoms\n\n"
        "`tilde_interior(ω)` lowers multivector degree by 1, "
        "`tilde_d(π)` raises by 1, `tilde_lie(ω, π)` preserves it.",
    ),
    (
        "code",
        "from jacopy.calculus.tilde import tilde_interior, tilde_d, tilde_lie\n"
        "from jacopy.core.expr import Symbol\n\n"
        "omega = Symbol(\"ω\")\n"
        "pi    = Symbol(\"π\")\n\n"
        "i_til = tilde_interior(omega)\n"
        "d_til = tilde_d(pi)\n"
        "L_til = tilde_lie(omega, pi)\n\n"
        "print(f\"{i_til}: degree {i_til._degree}\")\n"
        "print(f\"{d_til}: degree {d_til._degree}\")\n"
        "print(f\"{L_til}: degree {L_til._degree}\")",
    ),
    (
        "markdown",
        "## Defining-identity rewrites\n\n"
        "Three rules in `jacopy.calculus.tilde.axioms` realise the "
        "defining identities:\n\n"
        "* `TildeIotaSwapDefinition`, `ι̃_ω V → ι_V ω` (notation swap)\n"
        "* `TildeExteriorDLichnerowiczDefinition(π)`, `d̃ V → [π, V]_SN`\n"
        "* `TildeLieMagicDefinition(π)`, `L̃_ω V → d̃ ι̃_ω V + ι̃_ω d̃ V`",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Act\n"
        "from jacopy.calculus.tilde import (\n"
        "    TildeIotaSwapDefinition,\n"
        "    TildeExteriorDLichnerowiczDefinition,\n"
        "    TildeLieMagicDefinition,\n"
        ")\n"
        "from jacopy.proof.expansion import ExpansionEngine\n\n"
        "V = Symbol(\"V\")\n\n"
        "engine = ExpansionEngine([\n"
        "    TildeIotaSwapDefinition(),\n"
        "    TildeExteriorDLichnerowiczDefinition(pi),\n"
        "    TildeLieMagicDefinition(pi),\n"
        "])\n"
        "for inp in [Act(i_til, V), Act(d_til, V), Act(L_til, V)]:\n"
        "    out, steps = engine.expand(inp)\n"
        "    print(f\"{inp} → {out}  via {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## Auxiliary axioms, `Poisson` flag unlocks `d̃² = 0`\n\n"
        "Five auxiliary rules in `tilde.aux_axioms` cover the special "
        "cases the three defining rules don't reach:\n\n"
        "| Rule | Folds |\n"
        "|---|---|\n"
        "| `TildeIotaOnZeroVectorDefinition` | `ι̃_ω f → 0` (f deg 0) |\n"
        "| `TildeIotaSquaredZeroDefinition` | `ι̃_ω(ι̃_ω V) → 0` |\n"
        "| `TildeLieOnZeroVectorDefinition` | `L̃_ω f → π^♯(ω)(f)` |\n"
        "| `TildeDOfFunctionDefinition` | `d̃ f → −π^♯(df)` |\n"
        "| `TildeDSquaredPoissonDefinition` | `d̃² V → 0` (π Poisson) |",
    ),
    (
        "code",
        "from jacopy.calculus.tilde import TildeDSquaredPoissonDefinition\n"
        "from jacopy.core.properties import Poisson\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "reg.declare(pi, Poisson())\n\n"
        "engine = ExpansionEngine([\n"
        "    TildeDSquaredPoissonDefinition(pi, registry=reg),\n"
        "])\n"
        "expr = Act(d_til, Act(d_til, V))\n"
        "out, steps = engine.expand(expr)\n"
        "print(f\"{expr} → {out}\")\n"
        "print(f\"rule  : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## `prove_tilde_cartan_relation`, Cartan magic on a 1-vector\n\n"
        "`tilde_intrinsic_engine(pi, koszul, …)` bundles every rule "
        "above plus standard MultiEval / Sharp / Pairing helpers. "
        "Pair it with `prove_tilde_cartan_relation` and the magic "
        "formula closes mechanically. The `slot_kind=\"covector\"` "
        "discipline keeps the engine routed to *tilde* rules.",
    ),
    (
        "code",
        "from jacopy.brackets.koszul import KoszulBracket\n"
        "from jacopy.calculus.musical import Sharp\n"
        "from jacopy.calculus.tilde import (\n"
        "    tilde_intrinsic_engine, prove_tilde_cartan_relation,\n"
        ")\n"
        "from jacopy.core.expr import Sum\n"
        "from jacopy.core.properties import Graded\n\n"
        "reg = PropertyRegistry()\n"
        "reg.declare(pi,    Graded(degree=1)); reg.declare(pi, Poisson())\n"
        "reg.declare(omega, Graded(degree=1))\n"
        "reg.declare(V,     Graded(degree=1))\n"
        "eta = Symbol(\"η\"); reg.declare(eta, Graded(degree=1))\n\n"
        "sharp  = Sharp(pi)\n"
        "koszul = KoszulBracket(sharp)\n"
        "eng    = tilde_intrinsic_engine(pi, koszul, sharp=sharp, registry=reg)\n\n"
        "lhs = Act(L_til, V)\n"
        "rhs = Sum(Act(d_til, Act(i_til, V)), Act(i_til, Act(d_til, V)))\n\n"
        "chain = prove_tilde_cartan_relation(\n"
        "    lhs, rhs, etas=(eta,), engine=eng, registry=reg,\n"
        ")\n"
        "print(f\"L̃_ω = d̃ ι̃_ω + ι̃_ω d̃ closes in {len(chain)} steps\")",
    ),
    (
        "markdown",
        "The anti-commute relation `ι̃_ω(ι̃_μ V) + ι̃_μ(ι̃_ω V) = 0` "
        "(relation 1 in §3.1.3) closes through the same machinery.",
    ),
    (
        "code",
        "from jacopy.core.expr import Integer\n\n"
        "mu = Symbol(\"μ\");  reg.declare(mu, Graded(degree=1))\n"
        "W  = Symbol(\"W\");  reg.declare(W,  Graded(degree=2))\n"
        "i_mu = tilde_interior(mu)\n\n"
        "lhs = Sum(Act(i_til, Act(i_mu, W)), Act(i_mu, Act(i_til, W)))\n"
        "chain = prove_tilde_cartan_relation(\n"
        "    lhs, Integer(0), etas=(eta,), engine=eng, registry=reg,\n"
        ")\n"
        "print(f\"ι̃ anti-commute closes in {len(chain)} steps\")",
    ),
    (
        "markdown",
        "## `K̃_η`, the tilde Cartan remainder\n\n"
        "`K̃_η := −L̃_η + d̃ ∘ ι̃_η` is the polarity-flipped magic "
        "formula. The atom is inert; "
        "`TildeCartanRemainderDefinition` realises the defining "
        "expansion. Two `K̃` operators on different bivectors stay "
        "distinct, useful for deformation arguments.",
    ),
    (
        "code",
        "from jacopy.calculus.cartan_remainder_axioms import (\n"
        "    TildeCartanRemainderDefinition,\n"
        ")\n"
        "from jacopy.calculus.tilde import K_tilde\n\n"
        "K = K_tilde(eta, pi)\n"
        "print(f\"{K}: degree {K._degree}, form {K.form}, bivector {K.bivector}\")\n\n"
        "engine = ExpansionEngine([TildeCartanRemainderDefinition()])\n"
        "expr = Act(K, V)\n"
        "out, steps = engine.expand(expr)\n"
        "print(f\"{expr} → {out}\")\n"
        "print(f\"rule  : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `tilde_interior(ω)`, `tilde_d(π)`, `tilde_lie(ω, π)`, "
        "opaque `Derivation` atoms acting on multivectors.\n"
        "* Three defining rewrites (`TildeIotaSwap`, "
        "`TildeExteriorDLichnerowicz`, `TildeLieMagic`) plus five "
        "auxiliary rules close the engine machinery.\n"
        "* `tilde_intrinsic_engine` + `prove_tilde_cartan_relation` "
        "close magic in 7 steps and ι̃ anti-commute in 12. "
        "`slot_kind=\"covector\"` keeps the tilde and form-side "
        "engines from aliasing.\n"
        "* The `Poisson` flag unlocks `d̃² V → 0` in 4 engine steps "
        ", same flag drives the SN-bracket Jacobi chain (tutorial 12).\n"
        "* `K̃_η` and `TildeCartanRemainderDefinition` are the "
        "polarity-flipped shortcut for §3.1.4 derived identities; "
        "the atom keys on `(form, bivector)` so multiple `K̃` "
        "operators coexist without aliasing.",
    ),
]


TUTORIAL_18: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 18, Derivator identities (§3.1.5)\n\n"
        "Companion notebook to "
        "[18_derivator_identities.md](18_derivator_identities.md). The "
        "six §3.1.5 identities equate Cartan / SN derivators to sums "
        "of Cartan-remainder corrections (`K_V` / `K̃_η`). All six "
        "close mechanically through `KoszulProblem.prove_derivator`.",
    ),
    (
        "markdown",
        "## Setup, `KoszulProblem` as the entry point\n\n"
        "`KoszulProblem` carries the bivector, form / multivector "
        "inventories, and exposes pre-bundled `derivator_form_engine` "
        "/ `derivator_multivector_engine` accessors. "
        "`assume_poisson()` flags `π` as `Poisson`, that's what "
        "unlocks `d̃² V → 0` and the SN-bracket Jacobi inside the "
        "multivector engine.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Act, Derivation\n"
        "from jacopy.brackets.base import BracketApply\n"
        "from jacopy.brackets.schouten import sn as default_sn\n"
        "from jacopy.calculus.cartan_remainder import K\n"
        "from jacopy.calculus.derivator import derivator\n"
        "from jacopy.calculus.exterior_d import d as default_d\n"
        "from jacopy.calculus.interior import interior\n"
        "from jacopy.calculus.lie_derivative import lie_derivative\n"
        "from jacopy.calculus.tilde import (\n"
        "    K_tilde, tilde_d, tilde_interior, tilde_lie,\n"
        ")\n"
        "from jacopy.core.expr import Neg, Sum, Symbol\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.library.koszul_problem import KoszulProblem\n\n"
        "reg = PropertyRegistry()\n"
        "pi = Symbol(\"π\")\n"
        "omega, eta, mu = (Symbol(s) for s in (\"ω\", \"η\", \"μ\"))\n"
        "U, V, W = (Symbol(s) for s in (\"U\", \"V\", \"W\"))\n"
        "for f in (omega, eta, mu, U, V, W):\n"
        "    reg.declare(f, Graded(degree=1))\n\n"
        "Y = Derivation(\"Y\", 0)\n"
        "xi = Symbol(\"ξ\"); reg.declare(xi, Graded(degree=1))\n\n"
        "prob = KoszulProblem(\n"
        "    pi, (omega, eta, mu),\n"
        "    registry=reg,\n"
        "    multivectors=((U, 1), (V, 1), (W, 1)),\n"
        ")\n"
        "prob.assume_poisson()\n"
        "K_b = prob.koszul_bracket\n\n"
        "print(f\"form engine        : {len(prob.derivator_form_engine().definitions)} rules\")\n"
        "print(f\"multivector engine : {len(prob.derivator_multivector_engine().definitions)} rules\")",
    ),
    (
        "markdown",
        "## Form-side identity (1)\n\n"
        "$$D^{T^*M}_{L_U}(\\eta, \\mu) = L_{\\tilde K_\\eta U} \\mu "
        "+ K_{\\tilde K_\\mu U} \\eta$$\n\n"
        "The cyclic identity, three nested Koszul-bracket expansions "
        "feed into the obstruction. The 109 steps cover Cartan-magic "
        "expansions, operator-commutator folds, and `K̃_η U → "
        "−L̃_η U + d̃ ι̃_η U` polarity flips.",
    ),
    (
        "code",
        "lhs = derivator(lie_derivative(U), K_b, eta, mu)\n\n"
        "K_tilde_eta_U = Act(K_tilde(eta, pi), U)\n"
        "K_tilde_mu_U  = Act(K_tilde(mu, pi),  U)\n"
        "rhs = Sum(\n"
        "    Act(lie_derivative(K_tilde_eta_U), mu),\n"
        "    Act(K(K_tilde_mu_U),               eta),\n"
        ")\n\n"
        "chain = prob.prove_derivator(lhs, rhs, eval_args=(Y,), side=\"form\")\n"
        "print(f\"(1) form-side closes in {len(chain)} steps\")",
    ),
    (
        "markdown",
        "## Dual multivector-side identity (1')\n\n"
        "$$\\tilde D^{SN}_{\\tilde L_\\eta}(U, V) = "
        "\\tilde L_{K_U \\eta} V + \\tilde K_{K_V \\eta} U$$\n\n"
        "`side=\"multivector\"` routes through "
        "`derivator_multivector_engine` and uses "
        "`slot_kind=\"covector\"` for the `MultiEval` wrap, same "
        "discipline as `prove_tilde_cartan_relation` (tutorial 17).",
    ),
    (
        "code",
        "lhs = derivator(tilde_lie(eta, pi), default_sn, U, V)\n\n"
        "K_U_eta = Act(K(U), eta)\n"
        "K_V_eta = Act(K(V), eta)\n"
        "rhs = Sum(\n"
        "    Act(tilde_lie(K_U_eta, pi), V),\n"
        "    Act(K_tilde(K_V_eta, pi),   U),\n"
        ")\n\n"
        "chain = prob.prove_derivator(\n"
        "    lhs, rhs, eval_args=(xi,), side=\"multivector\",\n"
        ")\n"
        "print(f\"(1') multivector-side closes in {len(chain)} steps\")",
    ),
    (
        "markdown",
        "## The full table\n\n"
        "All six identities close in one `prob.prove_derivator(...)` "
        "call. Step counts:\n\n"
        "| # | side | steps |\n"
        "|---|---|---|\n"
        "| (1) `D^{T*M}_{L_U}(η, μ) = L_{K̃_η U} μ + K_{K̃_μ U} η` | form | 109 |\n"
        "| (2) `L_{d̃ ι̃_η U} μ = −[d ι_U η, μ]_K` | form | 21 |\n"
        "| (3) `0 = d ι_{L̃_ω W} η − d ι_{d̃ ι̃_η W} ω + d ι_W [ω, η]_K` | form | 30 |\n"
        "| (1') `D̃^{SN}_{L̃_η}(U, V) = L̃_{K_U η} V + K̃_{K_V η} U` | multivec | 117 |\n"
        "| (2') `L̃_{d ι_U η} V = −[d̃ ι̃_η U, V]_SN` | multivec | 25 |\n"
        "| (3') `0 = d̃ ι̃_{L_U η} V − d̃ ι̃_{d ι_V η} U + d̃ ι̃_η [U, V]_SN` | multivec | 23 |\n\n"
        "The cyclic (1)/(1') are the heaviest because three nested "
        "bracket expansions feed the obstruction. (2)/(2') and "
        "(3)/(3') skip the cycle and run lean.",
    ),
    (
        "markdown",
        "## When you'd reach past the wrapper\n\n"
        "Use `prove_derivator_identity` from "
        "`jacopy.calculus.derivator` directly when:\n\n"
        "* you need a custom rule layered on "
        "(`derivator_form_engine()` returns a fresh engine to extend);\n"
        "* the bracket isn't Koszul or SN (write your own engine "
        "factory mirroring the `derivator_form_engine` pattern);\n"
        "* you're proving an identity outside §3.1.5's derivator "
        "shape (Cartan magic, `d² = 0`, those belong to "
        "tutorial 15's `intrinsic_engine`).",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* Derivator `D^E_φ(u, v) := φ[u,v]_E − [φu, v]_E − [u, φv]_E` "
        "measures `φ`'s failure to be a graded derivation of "
        "`[·, ·]_E`.\n"
        "* Six §3.1.5 identities pin Cartan / SN derivators to sums "
        "of `K_V` / `K̃_η` corrections, three form-side, three dual.\n"
        "* `KoszulProblem.prove_derivator(lhs, rhs, *, eval_args, "
        "side)` closes all six in one call: `side=\"form\"` for "
        "(1)/(2)/(3), `side=\"multivector\"` for (1')/(2')/(3').\n"
        "* Step counts 109/21/30 (form) and 117/25/23 (multivec), "
        "(1)/(1') are heaviest because three nested bracket "
        "expansions feed the obstruction.\n"
        "* `prove_derivator_identity` is the engine-level entry "
        "point when you've assembled your own engine.",
    ),
]


TUTORIAL_19: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 19, The Courant family: Dorfman, Courant, Dirac\n\n"
        "Companion notebook to "
        "[19_courant_family.md](19_courant_family.md). "
        "`CourantAlgebroid` carries both Courant and Dorfman brackets "
        "on shared Cartan operators; `DiracStructure` pins a "
        "maximally-isotropic involutive subbundle. The bridge "
        "identity, H-twisted Jacobi, and Dirac axioms all close as "
        "single citation-tagged proof steps.",
    ),
    (
        "markdown",
        "## Section pairs\n\n"
        "`SectionPair(vector, form)` wraps `(X, α) ∈ Γ(TM ⊕ T*M)`. "
        "Both brackets consume `SectionPair`s and produce one.",
    ),
    (
        "code",
        "from jacopy.brackets.dorfman import SectionPair\n"
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "X = Symbol(\"X\");  reg.declare(X, Graded(degree=0))\n"
        "Y = Symbol(\"Y\");  reg.declare(Y, Graded(degree=0))\n"
        "alpha = Symbol(\"α\"); reg.declare(alpha, Graded(degree=1))\n"
        "beta  = Symbol(\"β\"); reg.declare(beta,  Graded(degree=1))\n\n"
        "a = SectionPair(X, alpha)\n"
        "b = SectionPair(Y, beta)\n"
        "print(f\"a = ({a.vector}, {a.form})\")\n"
        "print(f\"b = ({b.vector}, {b.form})\")",
    ),
    (
        "markdown",
        "## CourantAlgebroid, both brackets, shared operators\n\n"
        "Construct with no arguments for untwisted; pass "
        "`background_H=H` to twist by a closed 3-form. Both brackets "
        "use the **same** Cartan operators, that sharing is what "
        "makes the bridge identity exact.",
    ),
    (
        "code",
        "from jacopy.library.courant_algebroid import CourantAlgebroid\n\n"
        "C = CourantAlgebroid()\n"
        "print(f\"name           : {C.name}\")\n"
        "print(f\"twisted        : {C.is_twisted}\")\n\n"
        "dorf = C.expand_dorfman(a, b, registry=reg)\n"
        "cour = C.expand(a, b,         registry=reg)\n"
        "print(f\"Dorfman form half : {dorf.form}\")\n"
        "print(f\"Courant form half : {cour.form}\")",
    ),
    (
        "markdown",
        "## The bridge identity\n\n"
        "$$[a, b]_D - [a, b]_C = (0, \\tfrac12\\, d(\\iota_X \\beta "
        "+ \\iota_Y \\alpha))$$\n\n"
        "A single `theorem`-tagged step. Cartan's magic formula "
        "`L_Y α = d(ι_Y α) + ι_Y(dα)` makes the cancellation work.",
    ),
    (
        "code",
        "chain = C.prove_courant_dorfman_bridge(a, b, registry=reg)\n"
        "step = chain.steps[0]\n"
        "print(f\"steps  : {len(chain)}\")\n"
        "print(f\"rule   : {step.rule}\")\n"
        "print(f\"tag    : {step.provenance_tag}\")\n\n"
        "correction = C.bridge_correction(a, b)\n"
        "print(f\"correction.vector : {correction.vector}\")\n"
        "print(f\"correction.form   : {correction.form}\")",
    ),
    (
        "markdown",
        "## H-twisted Jacobi\n\n"
        "Untwisted: Jacobi holds exactly (obstruction is `0`). "
        "H-twisted: Jacobi closes iff `dH = 0`, the obstruction "
        "lands on `dH`. Both produce single `axiom`-tagged steps.",
    ),
    (
        "code",
        "H = Symbol(\"H\"); reg.declare(H, Graded(degree=3))\n"
        "C_H = CourantAlgebroid(background_H=H)\n"
        "print(f\"name           : {C_H.name}\")\n"
        "print(f\"twisted        : {C_H.is_twisted}\")\n\n"
        "chain = C_H.prove_jacobi_reduction(registry=reg)\n"
        "step = chain.steps[0]\n"
        "print(f\"rule           : {step.rule}\")\n"
        "print(f\"justification  : {step.justification}\")",
    ),
    (
        "markdown",
        "## DiracStructure, isotropy + involutivity\n\n"
        "Pairing `⟨a, b⟩ = ½(ι_X β + ι_Y α)`; diagonal "
        "`⟨a, a⟩ = ι_X α` is the isotropy obstruction. Involutivity "
        "is `[a, b]_C ∈ Γ(L)`, surfaced as a placeholder symbol "
        "since subbundle membership isn't an Expr-level predicate.",
    ),
    (
        "code",
        "from jacopy.library.dirac import DiracStructure\n\n"
        "L_sym = Symbol(\"L\")\n"
        "D     = DiracStructure(C, L_sym)\n"
        "print(f\"Dirac          : {D.name}\")\n"
        "print(f\"⟨a, b⟩         : {D.pairing(a, b)}\")\n"
        "print(f\"⟨a, a⟩         : {D.isotropy_obstruction(a)}\")\n\n"
        "chain_iso = D.prove_isotropy(a)\n"
        "chain_inv = D.prove_involutivity(a, b)\n"
        "print(f\"isotropy   : {len(chain_iso)} step, rule={chain_iso.steps[0].rule}\")\n"
        "print(f\"involutivity: {len(chain_inv)} step, rule={chain_inv.steps[0].rule}\")",
    ),
    (
        "markdown",
        "## The two canonical Dirac structures\n\n"
        "| Factory | Subbundle | Source |\n"
        "|---|---|---|\n"
        "| `poisson_dirac(π)` | `L_π = {(π^♯ α, α)}` | Poisson bivector |\n"
        "| `presymplectic_dirac(ω)` | `L_ω = {(X, ω^♭ X)}` | closed 2-form |\n\n"
        "Each factory records the subbundle name; isotropy / "
        "involutivity remain axioms (the conditional theorems "
        "`dω = 0 ⇒ L_ω` Dirac, `[π,π] = 0 ⇒ L_π` Dirac are out of "
        "scope here).",
    ),
    (
        "code",
        "from jacopy.library.dirac import poisson_dirac, presymplectic_dirac\n\n"
        "pi = Symbol(\"π\"); reg.declare(pi, Graded(degree=1))\n"
        "omega = Symbol(\"ω\"); reg.declare(omega, Graded(degree=2))\n\n"
        "L_pi    = poisson_dirac(pi,        courant=C)\n"
        "L_omega = presymplectic_dirac(omega, courant=C)\n\n"
        "print(f\"poisson_dirac        : {L_pi.name}\")\n"
        "print(f\"presymplectic_dirac  : {L_omega.name}\")",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `SectionPair(X, α)` operands for both brackets; "
        "`.vector` / `.form` accessors.\n"
        "* `CourantAlgebroid` exposes `expand` (Courant) and "
        "`expand_dorfman` on shared Cartan operators.\n"
        "* `prove_courant_dorfman_bridge`, single theorem-step "
        "asserting the exact correction `(0, ½ d(ι_X β + ι_Y α))`.\n"
        "* `prove_jacobi_reduction`, vacuous when untwisted, lands "
        "obstruction on `dH` when H-twisted.\n"
        "* `DiracStructure` carries pairing + isotropy + "
        "involutivity, all as axiom-tagged proof steps.\n"
        "* `poisson_dirac` / `presymplectic_dirac` factories for the "
        "two source geometries.",
    ),
]


TUTORIAL_20: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 20, Connection, curvature, and Bianchi identities\n\n"
        "Companion notebook to "
        "[20_connection_curvature.md](20_connection_curvature.md). "
        "`AffineConnection` carries `∇_X Y`; `Torsion` / `Curvature` "
        "are the structural obstructions; `BianchiProblem` bundles "
        "every rule needed to close the two Bianchi identities "
        "mechanically.",
    ),
    (
        "markdown",
        "## `AffineConnection` and `∇_X Y`\n\n"
        "An atom + an `eval(X, Y)` builder. Four engine rules realise "
        "X-linearity, X-scalar-pull, Y-additivity, and Y-Leibniz.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Derivation\n"
        "from jacopy.calculus.connection import AffineConnection\n\n"
        "nabla = AffineConnection(\"∇\")\n"
        "X, Y, W, Z = (Derivation(s, 0) for s in (\"X\", \"Y\", \"W\", \"Z\"))\n\n"
        "print(f\"connection : {nabla}\")\n"
        "print(f\"∇_X Y      : {nabla.eval(X, Y)}\")",
    ),
    (
        "markdown",
        "## Torsion and curvature\n\n"
        "$$T(X, Y) := \\nabla_X Y - \\nabla_Y X - [X, Y]$$\n"
        "$$R(X, Y) Z := \\nabla_X \\nabla_Y Z - \\nabla_Y \\nabla_X Z "
        "- \\nabla_{[X,Y]} Z$$\n\n"
        "Both inert until `TorsionDefinitionDefinition` / "
        "`CurvatureDefinitionDefinition` fire.",
    ),
    (
        "code",
        "from jacopy.calculus.torsion_curvature import Torsion, Curvature\n\n"
        "T = Torsion(nabla, X, Y)\n"
        "R = Curvature(nabla, X, Y, W)\n"
        "print(f\"T(X, Y)   : {T}\")\n"
        "print(f\"R(X, Y) W : {R}\")",
    ),
    (
        "markdown",
        "## `BianchiProblem`, the wrapper\n\n"
        "Bundles every rule needed: four connection axioms, "
        "torsion / curvature definitions, covariant-derivative "
        "definitions, and the LBVF (or `BracketApply`) closure "
        "family. The `registry` powers the Y-Leibniz `Graded(degree=0)` "
        "test on function factors.",
    ),
    (
        "code",
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.library.bianchi_problem import BianchiProblem\n\n"
        "reg = PropertyRegistry()\n"
        "prob = BianchiProblem(nabla, registry=reg)\n"
        "print(f\"engine rules : {len(prob.engine.definitions)}\")\n"
        "print(f\"connection   : {prob.connection}\")",
    ),
    (
        "markdown",
        "## First Bianchi identity\n\n"
        "$$\\mathrm{cycl}_{X,Y,Z}\\, R(X, Y) Z = "
        "\\mathrm{cycl}_{X,Y,Z}\\, "
        "[(\\nabla_X T)(Y, Z) + T(T(X, Y), Z)]$$\n\n"
        "`prove_first_bianchi` builds the difference of both sides, "
        "expands through the engine, and reports `ok=True` iff the "
        "residue collapses to `0`.",
    ),
    (
        "code",
        "res1 = prob.prove_first_bianchi(X, Y, W)\n"
        "print(f\"Bianchi I : ok={res1.ok}, steps={len(res1.lhs_steps)}\")",
    ),
    (
        "markdown",
        "## Second Bianchi identity\n\n"
        "$$\\mathrm{cycl}_{X,Y,Z}\\, (\\nabla_X R)(Y, Z) W = "
        "\\mathrm{cycl}_{X,Y,Z}\\, R(X, T(Y, Z)) W$$\n\n"
        "Same protocol, one extra fixed argument.",
    ),
    (
        "code",
        "res2 = prob.prove_second_bianchi(X, Y, W, Z)\n"
        "print(f\"Bianchi II : ok={res2.ok}, steps={len(res2.lhs_steps)}\")",
    ),
    (
        "markdown",
        "## Koszul connection, same identities on `T*M`\n\n"
        "`koszul_connection(name, *, anchor)` produces an algebroid "
        "connection on `T*M` for Poisson problems. Same "
        "`BianchiProblem` wrapper works, engine swaps in "
        "`BracketApply` closure rules and routes function-action "
        "through the anchor.",
    ),
    (
        "code",
        "from jacopy.calculus.connection import koszul_connection\n"
        "from jacopy.calculus.anchor import Anchor\n\n"
        "pi_sharp = Anchor(\"π^♯\")\n"
        "nabla_tilde = koszul_connection(\"∇̃\", anchor=pi_sharp)\n"
        "print(f\"connection : {nabla_tilde}\")\n"
        "print(f\"anchor     : {nabla_tilde.anchor}\")\n"
        "print(f\"bracket    : {nabla_tilde.bracket}\")",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `AffineConnection(name)` atom + `eval(X, Y)` builder + "
        "four defining-axiom engine rules.\n"
        "* `Torsion(∇, X, Y)` / `Curvature(∇, X, Y, Z)` inert until "
        "their definitions fire.\n"
        "* `BianchiProblem` bundles every rule; "
        "`prove_first_bianchi` / `prove_second_bianchi` close in "
        "~60-63 steps.\n"
        "* `koszul_connection` for the cotangent variant, same "
        "wrapper, `BracketApply` closure family swapped in.",
    ),
]


TUTORIAL_21: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 21, IndexedSum, Wedge, MultiEval\n\n"
        "Companion notebook to "
        "[21_indexed_sum_wedge_multi_eval.md](21_indexed_sum_wedge_multi_eval.md). "
        "Three structural Expr nodes underpin everything from Faz 17 "
        "onwards. They carry no algebraic content, antisymmetry, "
        "distribution, Kronecker contraction, etc. live as engine "
        "Definitions in the companion `*_axioms` modules.",
    ),
    (
        "markdown",
        "## `MultiEval`, multilinear evaluation\n\n"
        "`multi_eval(head, *args, alternating=True, slot_kind=\"vector\")`. "
        "Slot kind is declarative, `\"vector\"` (form-on-vectors) or "
        "`\"covector\"` (multivector-on-forms). Arity NOT validated at "
        "construction (head's degree may need a registry lookup).",
    ),
    (
        "code",
        "from jacopy.core.expr import Symbol\n"
        "from jacopy.core.multi_eval import multi_eval, has_repeated_arg\n\n"
        "omega = Symbol(\"ω\")\n"
        "X, Y  = Symbol(\"X\"), Symbol(\"Y\")\n\n"
        "me = multi_eval(omega, X, Y)\n"
        "print(f\"ω(X, Y)        : {me}\")\n"
        "print(f\"arity          : {me.arity}\")\n"
        "print(f\"alternating    : {me.alternating}\")\n"
        "print(f\"slot_kind      : {me.slot_kind}\")\n\n"
        "swapped, sign = me.swapped(0, 1)\n"
        "print(f\"swap → {swapped}, sign={sign}\")\n\n"
        "me_repeat = multi_eval(omega, X, X)\n"
        "print(f\"has_repeated_arg : {has_repeated_arg(me_repeat)}\")",
    ),
    (
        "markdown",
        "## `Wedge`, graded-antisymmetric product\n\n"
        "Distinct from `Product` (which is non-commutative scalar / "
        "operator product). `Wedge.make` flattens, absorbs `0`, drops "
        "`1`. Degree-aware identities (`α ∧ α = 0`) live in "
        "algorithms layer.",
    ),
    (
        "code",
        "from jacopy.core.expr import Integer\n"
        "from jacopy.core.wedge import Wedge\n\n"
        "alpha, beta, gamma = Symbol(\"α\"), Symbol(\"β\"), Symbol(\"γ\")\n\n"
        "w = Wedge.make(alpha, beta, gamma)\n"
        "print(f\"α ∧ β ∧ γ : {w}\")\n"
        "print(f\"absorb 0  : {Wedge.make(alpha, Integer(0), beta)}\")\n"
        "print(f\"drop 1    : {Wedge.make(alpha, Integer(1), beta)}\")",
    ),
    (
        "markdown",
        "## `WedgeMultiEvalAlternatingDefinition`\n\n"
        "$$(\\alpha_1 \\wedge \\dots \\wedge \\alpha_p)(X_1, \\dots, X_p) "
        "= \\sum_\\sigma \\mathrm{sign}(\\sigma) \\prod_i \\alpha_i(X_{\\sigma(i)})$$",
    ),
    (
        "code",
        "from jacopy.calculus.wedge_axioms import WedgeMultiEvalAlternatingDefinition\n"
        "from jacopy.core.properties import Graded\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.proof.expansion import ExpansionEngine\n\n"
        "reg = PropertyRegistry()\n"
        "for f in (alpha, beta):\n"
        "    reg.declare(f, Graded(degree=1))\n"
        "for f in (X, Y):\n"
        "    reg.declare(f, Graded(degree=0))\n\n"
        "w_eval = multi_eval(Wedge.make(alpha, beta), X, Y)\n"
        "engine = ExpansionEngine([WedgeMultiEvalAlternatingDefinition(registry=reg)])\n"
        "out, steps = engine.expand(w_eval)\n"
        "print(f\"(α ∧ β)(X, Y) → {out}\")\n"
        "print(f\"rule          : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## `IndexedSum`, α-equivalence as `==`\n\n"
        "`indexed_sum(dummy, range_, body)`. Equality compares "
        "α-renamed forms via a depth-aware sentinel; nested binders "
        "with overlapping names still distinguish correctly.",
    ),
    (
        "code",
        "from jacopy.calculus.local_frame import FrameIndex, LocalFrame\n"
        "from jacopy.core.indexed_sum import indexed_sum\n\n"
        "i = FrameIndex(\"i\")\n"
        "j = FrameIndex(\"j\")\n"
        "frame = LocalFrame(\"e\", dim=3)\n"
        "T_i = Symbol(\"T_i\")\n\n"
        "S   = indexed_sum(i, frame, T_i)\n"
        "S_j = S.with_dummy(j)\n"
        "print(f\"Σ_i T_i        : {S}\")\n"
        "print(f\"Σ_j (renamed)  : {S_j}\")\n"
        "print(f\"S == S_j       : {S == S_j}\")",
    ),
    (
        "markdown",
        "## `IndexedSum` axioms\n\n"
        "| Rule | Folds |\n"
        "|---|---|\n"
        "| `IndexedSumSumDistributeDefinition` | `Σ_i (A + B) → Σ_i A + Σ_i B` |\n"
        "| `IndexedSumNegPullDefinition` | `Σ_i Neg(X) → Neg(Σ_i X)` |\n"
        "| `IndexedSumScalarPullDefinition` | `Σ_i (c · X) → c · Σ_i X` (c dummy-free) |\n"
        "| `IndexedSumKroneckerContractDefinition` | `Σ_i δ_i^j A_i → A_j` |\n"
        "| `MultiEval/Pairing/ConnectionEvalIndexedSumPushIn` | pull contraction past Σ |",
    ),
    (
        "code",
        "from jacopy.calculus.indexed_sum_axioms import (\n"
        "    IndexedSumSumDistributeDefinition,\n"
        ")\n"
        "from jacopy.core.expr import Sum\n\n"
        "A, B = Symbol(\"A_i\"), Symbol(\"B_i\")\n"
        "S = indexed_sum(i, frame, Sum(A, B))\n"
        "engine = ExpansionEngine([IndexedSumSumDistributeDefinition()])\n"
        "out, steps = engine.expand(S)\n"
        "print(f\"Σ_i (A + B) → {out}\")\n"
        "print(f\"rule        : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `MultiEval(head, *args)`, multilinear evaluation. Slot "
        "kind declarative; arity not validated. `swapped(i, j)` "
        "carries the alternating sign.\n"
        "* `Wedge.make(α, β, …)`, graded-antisymmetric product. "
        "Smart constructor handles 0/1/associativity; degree-aware "
        "`α ∧ α = 0` lives in algorithms layer.\n"
        "* `IndexedSum(dummy, range_, body)`, bound-index sum with "
        "α-equivalence as `==`. `with_dummy(new)` for explicit rename.\n"
        "* Engine rules in `wedge_axioms` / `indexed_sum_axioms` "
        "realise alternating expansion, sum-distribute, scalar-pull, "
        "Kronecker contract, push-in past contraction nodes.\n"
        "* Direct use rare, debugging Cartan / frame proofs is the "
        "usual entry point.",
    ),
]


TUTORIAL_22: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 22, Local frames and frame decomposition\n\n"
        "Companion notebook to "
        "[22_frame_decomposition.md](22_frame_decomposition.md). "
        "`LocalFrame` is a library wrapper (not an Expr) bundling a "
        "frame's identity and display symbols. It produces frame VFs "
        "`X_a`, dual coframes `e^a`, and the frame-scoped duality "
        "rule. Frame-decomposition rules are opt-in to avoid loops.",
    ),
    (
        "markdown",
        "## `LocalFrame`, the wrapper\n\n"
        "Frames with the same `(name, dim, vf_symbol, coframe_symbol)` "
        "compare equal. `dim=None` is the symbolic-dimension mode "
        "Faz 17 proofs use.",
    ),
    (
        "code",
        "from jacopy.calculus.local_frame import LocalFrame, FrameIndex\n\n"
        "F = LocalFrame(\"F\", dim=3)\n"
        "print(f\"frame    : {F}\")\n"
        "print(f\"name     : {F.name}\")\n"
        "print(f\"dim      : {F.dim}\")\n"
        "print(f\"vf sym   : {F.vf_symbol}\")\n"
        "print(f\"coframe  : {F.coframe_symbol}\")",
    ),
    (
        "markdown",
        "## Indices and basis elements\n\n"
        "`FrameVectorField` subclasses `Derivation` (degree 0), "
        "every existing pass picks frame VFs up automatically. "
        "`FrameCovector` is an `Atom` mediated through `Pairing`. "
        "Equality includes the frame name to keep coexisting frames "
        "distinct.",
    ),
    (
        "code",
        "a, b = F.index(\"a\"), F.index(\"b\")\n"
        "X_a, X_b = F.X(a),       F.X(b)\n"
        "e_a, e_b = F.coframe(a), F.coframe(b)\n\n"
        "print(f\"X_a : {X_a}    class={X_a.__class__.__name__}\")\n"
        "print(f\"e_a : {e_a}    class={e_a.__class__.__name__}\")",
    ),
    (
        "markdown",
        "## `KroneckerDelta`, the contraction unit\n\n"
        "`KroneckerDelta(i, j)` collapses to `One` when the indices "
        "are structurally equal; stays opaque otherwise.",
    ),
    (
        "code",
        "from jacopy.calculus.local_frame import KroneckerDelta\n\n"
        "print(f\"δ^a_a = {KroneckerDelta(a, a)}\")\n"
        "print(f\"δ^a_b = {KroneckerDelta(a, b)}\")",
    ),
    (
        "markdown",
        "## `FramePairingDualityDefinition`, `⟨e^a, X_b⟩ → δ^a_b`\n\n"
        "Frame-scoped: fires only when both halves belong to the "
        "same `LocalFrame`. Build via `F.duality_definition()` to "
        "wire the scoping correctly.",
    ),
    (
        "code",
        "from jacopy.calculus.pairing import Pairing\n"
        "from jacopy.proof.expansion import ExpansionEngine\n\n"
        "p = Pairing(e_b, X_a)\n"
        "print(f\"raw pairing : {p}\")\n\n"
        "engine = ExpansionEngine([F.duality_definition()])\n"
        "out, steps = engine.expand(p)\n"
        "print(f\"after rule  : {out}\")\n"
        "print(f\"rule        : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## Frame decomposition, opt-in rules\n\n"
        "`FrameDecompositionDefinition(F)` rewrites `W → Σ_a e^a(W) · X_a`. "
        "**Opt-in**: pairing it with duality creates a loop. "
        "`CartanStructureProblem` (tutorial 23) turns it on for a "
        "specific sub-pass.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Derivation\n"
        "from jacopy.calculus.frame_decomposition import FrameDecompositionDefinition\n\n"
        "W = Derivation(\"W\", 0)\n"
        "rule = FrameDecompositionDefinition(F)\n\n"
        "# Apply once directly (running the engine would loop).\n"
        "print(f\"W → {rule.rewrite(W)}\")",
    ),
    (
        "markdown",
        "## The three frame-decomposition rules\n\n"
        "| Rule | Folds |\n"
        "|---|---|\n"
        "| `FrameDecompositionDefinition(F)` | `W → Σ_a e^a(W) · X_a` for any non-frame VF |\n"
        "| `ConnectionEvalYFrameDecompositionDefinition(F, ∇)` | `∇_X Y → ∇_X (Σ_a e^a(Y) · X_a)` |\n"
        "| `ConnectionFormDecompositionDefinition(F, ∇, ω)` | `∇_V X_b → Σ_c ω^c_b(V) · X_c` |\n\n"
        "The third introduces the **connection form** `ω^c_b(∇)`, "
        "the keystone of Cartan structure equation proofs. Every "
        "Christoffel-symbol calculation funnels through it.",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `LocalFrame` library wrapper; `dim=None` for symbolic "
        "dimension. Equality on `(name, dim, vf_symbol, coframe_symbol)` "
        "tuples.\n"
        "* Four Expr shapes: `FrameIndex`, `FrameVectorField` "
        "(Derivation subclass), `FrameCovector` (Atom), "
        "`KroneckerDelta` (collapses on matching indices).\n"
        "* `FramePairingDualityDefinition`, frame-scoped duality "
        "rule. Build via `F.duality_definition()`.\n"
        "* Three opt-in frame-decomposition rules; "
        "`ConnectionFormDecompositionDefinition` introduces "
        "`ω^c_b`. `CartanStructureProblem` (tutorial 23) handles "
        "the loop-avoidance bookkeeping.",
    ),
]


TUTORIAL_23: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 23, Cartan structure equations\n\n"
        "Companion notebook to "
        "[23_cartan_structure_equations.md](23_cartan_structure_equations.md). "
        "`CartanStructureProblem(∇, F)` proves both Cartan I and II "
        "mechanically:\n\n"
        "$$T^a = de^a + \\sum_b \\omega^a{}_b \\wedge e^b "
        "\\quad\\text{(I)}$$\n"
        "$$R^a{}_b = d\\omega^a{}_b + \\sum_c \\omega^a{}_c \\wedge "
        "\\omega^c{}_b \\quad\\text{(II)}$$",
    ),
    (
        "markdown",
        "## The three form atoms\n\n"
        "`ConnectionForm` (`ω^a_b`), `TorsionForm` (`T^a`), "
        "`CurvatureForm` (`R^a_b`), all inert until their "
        "definitions fire. All carry degree 1.",
    ),
    (
        "code",
        "from jacopy.calculus.connection import AffineConnection\n"
        "from jacopy.calculus.local_frame import LocalFrame\n"
        "from jacopy.calculus.cartan_forms import (\n"
        "    ConnectionForm, TorsionForm, CurvatureForm,\n"
        ")\n\n"
        "nabla = AffineConnection(\"∇\")\n"
        "F     = LocalFrame(\"F\", dim=3)\n\n"
        "omega_ab = ConnectionForm(nabla, F, \"a\", \"b\")\n"
        "T_a      = TorsionForm   (nabla, F, \"a\")\n"
        "R_ab     = CurvatureForm (nabla, F, \"a\", \"b\")\n\n"
        "print(f\"connection form : {omega_ab}\")\n"
        "print(f\"torsion form    : {T_a}\")\n"
        "print(f\"curvature form  : {R_ab}\")",
    ),
    (
        "markdown",
        "## `CartanStructureProblem`\n\n"
        "24-rule engine bundling torsion / curvature unfolding, "
        "connection axioms, frame decomposition, indexed-sum "
        "machinery, wedge expansion, intrinsic `d`, frame duality. "
        "Per-problem registry declares `FrameCovector` and "
        "`ConnectionForm` as degree 1.",
    ),
    (
        "code",
        "from jacopy.library.cartan_structure import CartanStructureProblem\n\n"
        "prob = CartanStructureProblem(nabla, F)\n"
        "print(f\"name           : {prob.name}\")\n"
        "print(f\"engine rules   : {len(prob.engine.definitions)}\")",
    ),
    (
        "markdown",
        "## LHS / RHS builders\n\n"
        "`first_cartan_lhs` / `_rhs` (and their second-equation "
        "counterparts) build the side terms as `MultiEval`s on a "
        "`(U, V)` pair. The bound dummies (`b` in I, `c` in II) "
        "are minted fresh per call.",
    ),
    (
        "code",
        "from jacopy.algebra.derivation import Derivation\n\n"
        "U, V = Derivation(\"U\", 0), Derivation(\"V\", 0)\n\n"
        "lhs1 = prob.first_cartan_lhs(U, V, \"a\")\n"
        "rhs1 = prob.first_cartan_rhs(U, V, \"a\")\n"
        "print(f\"Cartan I LHS : {lhs1}\")\n"
        "print(f\"Cartan I RHS : {rhs1}\")",
    ),
    (
        "markdown",
        "## `prove_first_cartan`, Cartan I\n\n"
        "~49 steps cover torsion-form opening, frame decomposition "
        "of `U`/`V`, Y-Leibniz on each `∇_U V`, connection-form "
        "decomposition of `∇_V X_b`, intrinsic `d` on `e^a`, wedge "
        "alternating expansion, and Kronecker contractions + "
        "pairing duality collapsing `⟨e^a, X_b⟩` matches.",
    ),
    (
        "code",
        "res1 = prob.prove_first_cartan(U, V, \"a\")\n"
        "print(f\"Cartan I  : ok={res1.ok}, steps={len(res1.steps)}\")",
    ),
    (
        "markdown",
        "## `prove_second_cartan`, Cartan II\n\n"
        "~54 steps. Structurally similar to Cartan I, both are "
        "shadows of the same abstract identity.",
    ),
    (
        "code",
        "res2 = prob.prove_second_cartan(U, V, \"a\", \"b\")\n"
        "print(f\"Cartan II : ok={res2.ok}, steps={len(res2.steps)}\")",
    ),
    (
        "markdown",
        "## When to use what\n\n"
        "| If you want… | Reach for… |\n"
        "|---|---|\n"
        "| Index-laden Cartan structure equation | `CartanStructureProblem` |\n"
        "| Coordinate-free Bianchi (no frame) | `BianchiProblem` (tutorial 20) |\n"
        "| Generic operator equality on forms | `prove_intrinsic_equivalence` (tutorial 15) |\n"
        "| Q9 Koszul mode (custom bracket on `∇`) | Same wrapper, auto-detects via `_intrinsic_d_rule` |",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `ConnectionForm`, `TorsionForm`, `CurvatureForm`, three "
        "inert degree-1 form atoms; definitions in `cartan_forms` "
        "engine rules.\n"
        "* `CartanStructureProblem(∇, F)`, 24-rule engine spanning "
        "seven phases (torsion/curvature, connection axioms, frame "
        "decomp, indexed-sum, wedge, intrinsic d, duality).\n"
        "* `prove_first_cartan` ~49 steps; `prove_second_cartan` "
        "~54 steps. Both run engine + simplify (with `sort_product`) "
        "to a fix-point.\n"
        "* Auto-detects custom-bracket connections (Q9) and swaps "
        "in `KoszulExteriorDIntrinsicDefinition`, no caller action "
        "needed.\n"
        "* Skip when the problem is coordinate-free (no frame, no "
        "index).",
    ),
]


TUTORIAL_24: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 24, Writing your own Problem wrapper\n\n"
        "Companion notebook to "
        "[24_custom_problem_wrapper.md](24_custom_problem_wrapper.md). "
        "Five-step recipe walked end-to-end via a worked "
        "`AlmostSymplecticProblem` (non-degenerate but **not** "
        "closed `ω`), illustrates every moving part without "
        "overlapping any existing wrapper.",
    ),
    (
        "markdown",
        "## The five steps\n\n"
        "1. Pick the geometric data the wrapper carries.\n"
        "2. Auto-declare structural axioms on the registry "
        "(check `has(...)` first, pre-declared flags take "
        "precedence).\n"
        "3. Assemble the engine, layer your rules onto "
        "`default_engine(registry=…)`.\n"
        "4. Write builder + prover methods.\n"
        "5. (optional) Register seeded theorems for one-step "
        "citations.",
    ),
    (
        "markdown",
        "## Worked example, `AlmostSymplecticProblem(ω)`\n\n"
        "An almost-symplectic form is non-degenerate but not "
        "necessarily closed. We get the vector-field-equality "
        "rule (`ι_X ω = ι_Y ω ⇒ X = Y`) but *not* the "
        "Hamiltonian / Poisson structure that closure unlocks.",
    ),
    (
        "code",
        "from typing import Optional\n"
        "from jacopy.algebra.derivation import Act\n"
        "from jacopy.calculus.interior import interior\n"
        "from jacopy.calculus.nondegenerate_axioms import (\n"
        "    NonDegenerateInteriorEqualityDefinition,\n"
        ")\n"
        "from jacopy.core.expr import Expr\n"
        "from jacopy.core.properties import Graded, NonDegenerate\n"
        "from jacopy.core.registry import PropertyRegistry\n"
        "from jacopy.proof.expansion import ExpansionEngine, default_engine\n\n"
        "class AlmostSymplecticProblem:\n"
        "    \"\"\"`(ω, registry)`, non-degenerate (not necessarily closed) 2-form.\"\"\"\n\n"
        "    __slots__ = (\"_omega\", \"_registry\", \"_engine\", \"_name\")\n\n"
        "    def __init__(self, omega, *, registry=None, name=None):\n"
        "        self._omega    = omega\n"
        "        self._registry = registry or PropertyRegistry()\n"
        "        # Step 2, auto-declare; pre-declared flags take precedence.\n"
        "        if not self._registry.has(omega, Graded):\n"
        "            self._registry.declare(omega, Graded(degree=2))\n"
        "        if not self._registry.has(omega, NonDegenerate):\n"
        "            self._registry.declare(omega, NonDegenerate())\n"
        "        # Step 3, engine assembly.\n"
        "        base = default_engine(registry=self._registry)\n"
        "        self._engine = ExpansionEngine(\n"
        "            list(base.definitions) + [\n"
        "                NonDegenerateInteriorEqualityDefinition(\n"
        "                    registry=self._registry,\n"
        "                ),\n"
        "            ]\n"
        "        )\n"
        "        self._name = name or f\"AlmostSymplecticProblem({omega._repr_inner()})\"\n\n"
        "    @property\n"
        "    def omega(self): return self._omega\n"
        "    @property\n"
        "    def registry(self): return self._registry\n"
        "    @property\n"
        "    def engine(self): return self._engine\n"
        "    @property\n"
        "    def name(self): return self._name\n\n"
        "    # Step 4, builders + provers.\n"
        "    def musical_flat(self, X):\n"
        "        \"\"\"`ω^♭(X) = ι_X ω`.\"\"\"\n"
        "        return Act(interior(X), self._omega)\n",
    ),
    (
        "markdown",
        "## Try it, the rule fires on the difference\n\n"
        "On the obstruction `ι_X ω − ι_Y ω`, the engine reduces "
        "to `X − Y` in one step. That **is** the proof "
        "transcript of `ι_X ω = ι_Y ω ⇒ X = Y`, the residue "
        "`X − Y` is the equality the caller discharges.",
    ),
    (
        "code",
        "from jacopy.core.expr import Symbol, Sum, Neg\n\n"
        "omega = Symbol(\"ω\")\n"
        "X = Symbol(\"X\"); Y = Symbol(\"Y\")\n"
        "reg = PropertyRegistry()\n"
        "reg.declare(X, Graded(degree=1))\n"
        "reg.declare(Y, Graded(degree=1))\n\n"
        "prob = AlmostSymplecticProblem(omega, registry=reg)\n"
        "print(f\"name         : {prob.name}\")\n"
        "print(f\"engine rules : {len(prob.engine.definitions)}\")\n\n"
        "obstruction = Sum(prob.musical_flat(X), Neg(prob.musical_flat(Y)))\n"
        "out, steps = prob.engine.expand(obstruction)\n"
        "print(f\"input  : {obstruction}\")\n"
        "print(f\"output : {out}\")\n"
        "print(f\"rule   : {steps[0].rule}\")",
    ),
    (
        "markdown",
        "## Picking your axioms, flags vs Definitions\n\n"
        "| Mechanism | Use when | Example |\n"
        "|---|---|---|\n"
        "| Registry flag | Property is a one-bit fact about a single object | `Closed`, `NonDegenerate`, `Poisson` |\n"
        "| Custom `Definition` subclass | Property is a non-trivial rewrite shape | tilde closure axioms |\n"
        "| Seeded `Theorem` | Identity should appear as a single citation step | `poisson_jacobi`, `courant_dorfman_bridge` |\n\n"
        "Prefer flags when you can, declarative, opt-out via "
        "pre-declaration, single rule fires per object.",
    ),
    (
        "markdown",
        "## Where to look in the codebase\n\n"
        "| Source | Read for |\n"
        "|---|---|\n"
        "| `library/symplectic.py` | Smallest non-trivial wrapper |\n"
        "| `library/courant_algebroid.py` | Wrapper with seeded theorems + bridge identity |\n"
        "| `library/bianchi_problem.py` | Custom proof loop (`_expand_to_canonical`) |\n"
        "| `library/koszul_problem.py` | Largest wrapper, multi-engine + canonicalize_indices pre-pass |\n"
        "| `library/cartan_structure.py` | Index-laden wrapper with per-problem registry |\n\n"
        "Read the wrapper closest to your shape; the pattern repeats.",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* Five-step recipe: data → axioms → engine → methods → "
        "(seeded theorems).\n"
        "* `registry.has(...)` check is the **opt-out mechanism**: "
        "pre-declared flags take precedence.\n"
        "* `default_engine(registry=…)` is the right base for most "
        "form-side problems; layer your rules onto its "
        "`definitions` list.\n"
        "* Engine order: definitions before linearity; frame-scoped "
        "rules before generic; opt-in rules for loop-prone pairs "
        "(decomp + duality).\n"
        "* Read `library/symplectic.py` for the smallest template, "
        "`library/koszul_problem.py` for the largest.",
    ),
]


TUTORIAL_25: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 25, Frame-component differential geometry (`jacopy.frame_calc`)\n\n"
        "Companion notebook to "
        "[25_frame_calc.md](25_frame_calc.md). "
        "`jacopy.frame_calc` is jacopy's **component-level submodule** "
        "for concrete metric calculations: given a metric `g` on a "
        "frame, compute Christoffel symbols, Riemann curvature, "
        "Ricci tensor, scalar curvature, Einstein tensor, with "
        "step-by-step derivation transcripts that bridge to "
        "`ProofChain` for paper-grade LaTeX output.\n\n"
        "Requires SymPy: `pip install \"jacopy[components]\"`.",
    ),
    (
        "markdown",
        "## Quick taste, Schwarzschild vacuum in five lines",
    ),
    (
        "code",
        "from jacopy.frame_calc import einstein_tensor, levi_civita\n"
        "from jacopy.frame_calc.library import schwarzschild\n\n"
        "F, g = schwarzschild()\n"
        "G = einstein_tensor(levi_civita(g), g)\n"
        "print(f'G.is_vacuum() = {G.is_vacuum()}')",
    ),
    (
        "markdown",
        "## Frame setup, `CoordinateFrame`\n\n"
        "Most physics literature uses coordinate frames "
        "(`e_a = ∂/∂x^a`). The frame's `derivative(f, a)` is "
        "`∂f/∂x^a`; `gamma(a, b, c) = 0` (coordinate frames are "
        "holonomic).",
    ),
    (
        "code",
        "from jacopy.frame_calc import CoordinateFrame\n"
        "import sympy as sp\n\n"
        "t, r, theta, phi = sp.symbols('t r theta phi')\n"
        "F = CoordinateFrame([t, r, theta, phi])\n"
        "print(F)\n"
        "print('dim:', F.dim)\n"
        "print('e_r(r²) =', F.derivative(r**2, 1))\n"
        "print('γ^a_bc =', F.gamma(0, 1, 0))",
    ),
    (
        "markdown",
        "## `ComponentMetric` and `inverse()`\n\n"
        "Symmetry checked at construction. `inverse()` returns "
        "`g^{ab}` as a `(2, 0)` tensor.",
    ),
    (
        "code",
        "from jacopy.frame_calc import ComponentMetric\n\n"
        "M = sp.Symbol('M', positive=True)\n"
        "g = ComponentMetric(F, sp.Matrix([\n"
        "    [-(1 - 2*M/r),   0,                0,    0],\n"
        "    [0,              1/(1 - 2*M/r),    0,    0],\n"
        "    [0,              0,                r**2, 0],\n"
        "    [0,              0,                0,    r**2 * sp.sin(theta)**2],\n"
        "]))\n"
        "print('g[0,0] =', g[0, 0])\n"
        "print('g_inv[0,0] =', g.inverse()[0, 0])\n"
        "# Verify g^{ac} g_{cb} = δ^a_b for one entry\n"
        "g_inv = g.inverse()\n"
        "delta_00 = sp.simplify(sum(g_inv[0, c] * g[c, 0] for c in range(4)))\n"
        "print('δ^0_0 =', delta_00)",
    ),
    (
        "markdown",
        "## Levi-Civita Christoffel symbols via Koszul formula\n\n"
        "The unique torsion-free metric-compatible connection.",
    ),
    (
        "code",
        "from jacopy.frame_calc import levi_civita\n\n"
        "LC = levi_civita(g)\n"
        "print(f'# non-zero Christoffel: {len(LC.nonzero_components())}')\n"
        "print(f'Γ^t_tr = {LC[0, 0, 1]}')\n"
        "print(f'Γ^θ_rθ = {LC[2, 1, 2]}')\n"
        "print(f'Γ^r_θθ = {LC[1, 2, 2]}')",
    ),
    (
        "markdown",
        "## Step-by-step derivation transcript\n\n"
        "Each Christoffel computation records `KoszulStep`s. Use "
        "`format_derivation` for plain text or `derivation_chain` "
        "for `ProofChain` → LaTeX.",
    ),
    (
        "code",
        "print(LC.format_derivation(0, 0, 1))",
    ),
    (
        "markdown",
        "## Curvature, Ricci, Einstein\n\n"
        "Schwarzschild is Ricci-flat, `Ric = 0`, `R = 0`, `G = 0`. "
        "This is the vacuum field equation result.",
    ),
    (
        "code",
        "from jacopy.frame_calc import (\n"
        "    curvature, ricci, ricci_scalar, einstein_tensor,\n"
        ")\n\n"
        "R = curvature(LC)\n"
        "Ric = ricci(LC)\n"
        "R_scalar = ricci_scalar(LC, g)\n"
        "G = einstein_tensor(LC, g)\n\n"
        "print(f'curvature.is_zero():  {R.is_zero()}  (NOT flat)')\n"
        "print(f'Ric.is_zero():        {Ric.is_zero()}')\n"
        "print(f'R_scalar:             {R_scalar}')\n"
        "print(f'G.is_vacuum():        {G.is_vacuum()}')",
    ),
    (
        "markdown",
        "## Optimised mode for Kerr-class metrics\n\n"
        "Default mode runs `sympy.simplify` on every Christoffel / "
        "Ricci / curvature entry. For Kerr (off-diagonal + complex "
        "denominators), this blows up. **Optimised mode** skips "
        "per-entry simplify; expressions stay raw but mathematically "
        "correct. Trade-off: no derivation traces in optimised mode.",
    ),
    (
        "code",
        "from jacopy.frame_calc.library import kerr\n"
        "import time\n\n"
        "F_kerr, g_kerr = kerr()\n"
        "t0 = time.perf_counter()\n"
        "G_kerr = einstein_tensor(\n"
        "    levi_civita(g_kerr, optimized=True), g_kerr, optimized=True\n"
        ")\n"
        "elapsed = time.perf_counter() - t0\n"
        "print(f'Kerr full pipeline: {elapsed:.1f} s')\n"
        "print(f'G.is_vacuum(): {G_kerr.is_vacuum()}')",
    ),
    (
        "markdown",
        "## Library fixtures\n\n"
        "Ready-made factories: `minkowski`, `schwarzschild`, `frw`, "
        "`kerr`. Each accepts `Symbol` / `Function` overrides.",
    ),
    (
        "code",
        "from jacopy.frame_calc.library import minkowski, frw\n\n"
        "# Minkowski 4D, flat\n"
        "F_m, g_m = minkowski()\n"
        "G_m = einstein_tensor(levi_civita(g_m), g_m)\n"
        "print(f'Minkowski G.is_vacuum(): {G_m.is_vacuum()}')\n\n"
        "# FRW (k=0, a(t) symbolic), non-vacuum cosmology\n"
        "F_frw, g_frw = frw()\n"
        "G_frw = einstein_tensor(levi_civita(g_frw), g_frw)\n"
        "print(f'FRW G.is_zero(): {G_frw.is_zero()} (Friedmann eq form)')\n"
        "print(f'FRW G[0,0] = {sp.simplify(G_frw[0, 0])}')",
    ),
    (
        "markdown",
        "## ProofChain bridge, paper-grade LaTeX\n\n"
        "Each tracked tensor's `derivation_chain(...)` returns a "
        "`ProofChain` compatible with `chain_to_latex_document`. "
        "The `SymPyAtom` wrapper bridges SymPy expressions into "
        "jacopy's `Expr` for ProofStep storage.",
    ),
    (
        "code",
        "from jacopy.display import chain_to_latex\n\n"
        "chain = LC.derivation_chain(0, 0, 1)   # Γ^t_tr\n"
        "print(f'chain length: {len(chain.steps)}')\n"
        "print(f'first step rule: {chain.steps[0].rule}')\n"
        "print(f'first step tag: {chain.steps[0].provenance_tag}')",
    ),
    (
        "markdown",
        "## Drop-in template, paste your metric, get everything\n\n"
        "Below is the **paper-workflow template**: change *only* the "
        "metric-matrix block; the rest of the pipeline runs as-is on "
        "whatever metric you provided. The default example uses "
        "Reissner-Nordström (charged Schwarzschild), not in the "
        "library, just to show that any metric matrix works.\n\n"
        "**To compute on a different metric**, edit the matrix in "
        "block 2; everything else is unchanged.",
    ),
    (
        "code",
        "import sympy as sp\n"
        "from jacopy.frame_calc import (\n"
        "    CoordinateFrame, ComponentMetric,\n"
        "    levi_civita, ricci, ricci_scalar, einstein_tensor,\n"
        ")\n\n"
        "# ─────────────────────────────────────────────────────────\n"
        "# 1. Coordinates, adjust to your metric's chart\n"
        "# ─────────────────────────────────────────────────────────\n"
        "t, r, theta, phi = sp.symbols('t r theta phi')\n"
        "coords = [t, r, theta, phi]\n\n"
        "# Extra parameters (mass, charge, cosmological constant, …):\n"
        "M = sp.Symbol('M', positive=True)\n"
        "Q = sp.Symbol('Q', positive=True)\n\n"
        "# ─────────────────────────────────────────────────────────\n"
        "# 2. METRIC MATRIX, REPLACE THIS BLOCK WITH YOUR OWN\n"
        "# ─────────────────────────────────────────────────────────\n"
        "# Reissner-Nordström: charged static spherical black hole\n"
        "factor = 1 - 2*M/r + Q**2 / r**2\n"
        "metric_matrix = sp.Matrix([\n"
        "    [-factor,         0,        0,                          0],\n"
        "    [0,        1/factor,        0,                          0],\n"
        "    [0,               0,     r**2,                          0],\n"
        "    [0,               0,        0,    r**2 * sp.sin(theta)**2],\n"
        "])\n\n"
        "# ─────────────────────────────────────────────────────────\n"
        "# 3. Pipeline, runs as-is on whatever metric is above\n"
        "# ─────────────────────────────────────────────────────────\n"
        "F = CoordinateFrame(coords)\n"
        "g = ComponentMetric(F, metric_matrix)\n"
        "LC = levi_civita(g)\n"
        "Ric = ricci(LC)\n"
        "R = ricci_scalar(LC, g)\n"
        "G = einstein_tensor(LC, g)\n\n"
        "# ─────────────────────────────────────────────────────────\n"
        "# 4. Output, summary + all non-zero entries\n"
        "# ─────────────────────────────────────────────────────────\n"
        "names = F.index_names()\n"
        "print(f'# non-zero Christoffel: {len(LC.nonzero_components())}')\n"
        "print(f'Ricci scalar R   = {sp.simplify(R)}')\n"
        "print(f'Ric.is_zero()    = {Ric.is_zero()}')\n"
        "print(f'G.is_vacuum()    = {G.is_vacuum()}')\n\n"
        "print('\\nChristoffel symbols (non-zero):')\n"
        "for (e, a, b), val in LC.nonzero_components().items():\n"
        "    print(f'  Γ^{names[e]}_{{{names[a]}{names[b]}}} = {val}')\n\n"
        "print('\\nEinstein tensor (non-zero entries):')\n"
        "for a in range(F.dim):\n"
        "    for b in range(a, F.dim):\n"
        "        val = sp.simplify(sp.trigsimp(G[a, b]))\n"
        "        if val != 0:\n"
        "            print(f'  G_{{{names[a]}{names[b]}}} = {val}')",
    ),
    (
        "markdown",
        "### Other metrics you can drop in\n\n"
        "Replace block 2 with any of these (or whatever your paper "
        "uses):\n\n"
        "**Schwarzschild-de Sitter (cosmological constant):**\n"
        "```python\n"
        "Lambda = sp.Symbol('Lambda')\n"
        "factor = 1 - 2*M/r - Lambda*r**2/3\n"
        "metric_matrix = sp.Matrix([\n"
        "    [-factor, 0, 0, 0],\n"
        "    [0, 1/factor, 0, 0],\n"
        "    [0, 0, r**2, 0],\n"
        "    [0, 0, 0, r**2 * sp.sin(theta)**2],\n"
        "])\n"
        "```\n\n"
        "**Vaidya (radiating mass, advanced time):**\n"
        "```python\n"
        "u = sp.Symbol('u'); coords = [u, r, theta, phi]\n"
        "m = sp.Function('m')(u)\n"
        "metric_matrix = sp.Matrix([\n"
        "    [-(1 - 2*m/r), 1, 0, 0],\n"
        "    [1, 0, 0, 0],\n"
        "    [0, 0, r**2, 0],\n"
        "    [0, 0, 0, r**2 * sp.sin(theta)**2],\n"
        "])\n"
        "```\n\n"
        "**Kerr-class metrics** (off-diagonal, complex denominators), "
        "add `optimized=True` to every pipeline call:\n"
        "```python\n"
        "LC = levi_civita(g, optimized=True)\n"
        "Ric = ricci(LC, optimized=True)\n"
        "R = ricci_scalar(LC, g, optimized=True)\n"
        "G = einstein_tensor(LC, g, optimized=True)\n"
        "```\n\n"
        "The output is raw (un-simplified) but mathematically "
        "correct; `G.is_vacuum()` etc. still work via SymPy's basic "
        "arithmetic. Apply `sp.simplify(LC[a,b,c])` on the specific "
        "entries you want to clean up.",
    ),
    (
        "markdown",
        "## Custom connection, independent of the metric\n\n"
        "**Connection and metric are independent geometric objects.** "
        "The Levi-Civita connection is the *unique* connection that's "
        "both torsion-free and metric-compatible for a given metric "
        ", but it's just one of many possible connections. In "
        "Einstein-Cartan theory, teleparallel gravity, Palatini "
        "formulations, and other modified gravity frameworks, the "
        "connection is **not** Levi-Civita.\n\n"
        "`einstein_tensor(connection, g)` accepts **any** "
        "`ComponentConnection`, not just `LeviCivitaConnection`. "
        "Build a custom connection with `ComponentConnection(F, "
        "christoffel_table)` and the rest of the pipeline runs "
        "as-is.",
    ),
    (
        "markdown",
        "### Symbol-domain matching (important pitfall!)\n\n"
        "When you supply Christoffel symbols by hand, **use the "
        "symbols the frame already carries**, not freshly-created "
        "ones. Library factories like `schwarzschild()` create "
        "symbols with specific assumptions (`r > 0`, `M > 0`); your "
        "hand-written `sp.symbols('r')` is a *different* symbol "
        "object even though the name matches.\n\n"
        "Pattern:\n"
        "```python\n"
        "F, g = schwarzschild()\n"
        "t, r, theta, phi = F.coords            # ← use these\n"
        "M = sp.Symbol('M', positive=True)      # ← match factory's assumption\n"
        "```",
    ),
    (
        "markdown",
        "### Sanity check, manual Schwarzschild matches Levi-Civita\n\n"
        "Build the textbook Schwarzschild Christoffels by hand, wrap "
        "them in `ComponentConnection`, and verify the result matches "
        "`levi_civita(g)` exactly.",
    ),
    (
        "code",
        "from jacopy.frame_calc import (\n"
        "    ComponentConnection, einstein_tensor, levi_civita,\n"
        ")\n"
        "from jacopy.frame_calc.library import schwarzschild\n"
        "import sympy as sp\n\n"
        "F_sw, g_sw = schwarzschild()\n"
        "t_sw, r_sw, theta_sw, phi_sw = F_sw.coords\n"
        "M_sw = sp.Symbol('M', positive=True)\n\n"
        "# 13 non-zero textbook Schwarzschild Christoffels\n"
        "manual = sp.MutableDenseNDimArray.zeros(F_sw.dim, F_sw.dim, F_sw.dim)\n"
        "factor = 1 - 2*M_sw/r_sw\n"
        "val_t = M_sw / (r_sw**2 * factor)\n"
        "manual[0, 0, 1] = val_t                                          # Γ^t_tr\n"
        "manual[0, 1, 0] = val_t                                          # Γ^t_rt\n"
        "manual[1, 0, 0] = M_sw * factor / r_sw**2                        # Γ^r_tt\n"
        "manual[1, 1, 1] = -M_sw / (r_sw**2 * factor)                     # Γ^r_rr\n"
        "manual[1, 2, 2] = -(r_sw - 2*M_sw)                               # Γ^r_θθ\n"
        "manual[1, 3, 3] = -(r_sw - 2*M_sw) * sp.sin(theta_sw)**2         # Γ^r_φφ\n"
        "manual[2, 1, 2] = manual[2, 2, 1] = 1/r_sw                       # Γ^θ_rθ\n"
        "manual[2, 3, 3] = -sp.sin(theta_sw) * sp.cos(theta_sw)           # Γ^θ_φφ\n"
        "manual[3, 1, 3] = manual[3, 3, 1] = 1/r_sw                       # Γ^φ_rφ\n"
        "manual[3, 2, 3] = manual[3, 3, 2] = sp.cos(theta_sw)/sp.sin(theta_sw)\n\n"
        "manual_conn = ComponentConnection(F_sw, manual)\n"
        "LC_sw = levi_civita(g_sw)\n\n"
        "# Entry-by-entry consistency\n"
        "all_match = all(\n"
        "    sp.simplify(sp.trigsimp(LC_sw[a, b, c] - manual_conn[a, b, c])) == 0\n"
        "    for a in range(F_sw.dim) for b in range(F_sw.dim) for c in range(F_sw.dim)\n"
        ")\n"
        "print(f'manual vs Levi-Civita match? {all_match}')\n"
        "print(f'einstein_tensor(manual_conn, g).is_vacuum() = {einstein_tensor(manual_conn, g_sw).is_vacuum()}')\n"
        "print(f'einstein_tensor(LC, g).is_vacuum()         = {einstein_tensor(LC_sw, g_sw).is_vacuum()}')",
    ),
    (
        "markdown",
        "### Non-trivial use: same metric, different connection\n\n"
        "For modified-gravity work, you'd add a torsion correction "
        "or use a fully independent connection. Here's the same "
        "Schwarzschild metric with a torsion-perturbed connection, "
        "the Einstein tensor is no longer vacuum because the "
        "connection is no longer torsion-free.",
    ),
    (
        "code",
        "from jacopy.frame_calc import torsion\n\n"
        "# Levi-Civita baseline\n"
        "LC = levi_civita(g_sw)\n\n"
        "# Add antisymmetric torsion: T^t_{rθ} = sin θ\n"
        "new_christoffel = sp.MutableDenseNDimArray.zeros(F_sw.dim, F_sw.dim, F_sw.dim)\n"
        "for a in range(F_sw.dim):\n"
        "    for b in range(F_sw.dim):\n"
        "        for c in range(F_sw.dim):\n"
        "            new_christoffel[a, b, c] = LC[a, b, c]\n\n"
        "new_christoffel[0, 1, 2] += sp.Rational(1, 2) * sp.sin(theta_sw)\n"
        "new_christoffel[0, 2, 1] -= sp.Rational(1, 2) * sp.sin(theta_sw)\n\n"
        "torsion_conn = ComponentConnection(F_sw, new_christoffel)\n"
        "T = torsion(torsion_conn)\n"
        "print(f'Torsion is zero?              {T.is_zero()}')\n"
        "print(f'T^t_{{rθ}} = {T[0, 1, 2]}')\n\n"
        "G_torsion = einstein_tensor(torsion_conn, g_sw)\n"
        "print(f'Same metric, custom connection: G.is_vacuum() = {G_torsion.is_vacuum()}')\n"
        "print(f'Levi-Civita on same metric:     G.is_vacuum() = {einstein_tensor(LC, g_sw).is_vacuum()}')",
    ),
    (
        "markdown",
        "### When you'd actually use this\n\n"
        "| Scenario | Custom connection? |\n"
        "|---|---|\n"
        "| Standard GR (vacuum, Einstein-Maxwell, Schwarzschild family) | No, `levi_civita(g)` |\n"
        "| Einstein-Cartan theory (torsion present) | Yes |\n"
        "| Teleparallel gravity (`R = 0`, torsion only) | Yes |\n"
        "| Palatini formulation (`g`, `Γ` varied independently) | Yes |\n"
        "| Affine theory (no metric) | Yes |\n\n"
        "For the standard-GR cases the metric → Levi-Civita → "
        "tensors chain is all you need. The custom-connection path "
        "opens up when the physics requires it.",
    ),
    (
        "markdown",
        "### API stress test, arbitrary symbols\n\n"
        "Before showing physically-motivated patterns, let's check "
        "the API accepts completely arbitrary symbol parameters.\n\n"
        "**Schwarzschild with made-up A, B:** the API runs, but the "
        "result is non-vacuum because A, B don't match Levi-Civita "
        "and the angular-block Christoffels are missing.",
    ),
    (
        "code",
        "F_sw, g_sw = schwarzschild()\n"
        "t_sw, r_sw, theta_sw, phi_sw = F_sw.coords\n"
        "A, B = sp.symbols('A B')\n\n"
        "manual = sp.MutableDenseNDimArray.zeros(F_sw.dim, F_sw.dim, F_sw.dim)\n"
        "manual[0, 0, 1] = manual[0, 1, 0] = A\n"
        "manual[1, 0, 0] = B\n"
        "manual[1, 1, 1] = -B\n"
        "manual[2, 1, 2] = manual[2, 2, 1] = 1 / r_sw\n"
        "manual[3, 1, 3] = manual[3, 3, 1] = 1 / r_sw\n\n"
        "manual_conn = ComponentConnection(F_sw, manual)\n"
        "G_manual = einstein_tensor(manual_conn, g_sw)\n"
        "print(f'is_vacuum?  {G_manual.is_vacuum()}')\n"
        "print('non-zero G entries:')\n"
        "for a in range(F_sw.dim):\n"
        "    for b in range(a, F_sw.dim):\n"
        "        val = sp.simplify(sp.trigsimp(G_manual[a, b]))\n"
        "        if val != 0:\n"
        "            print(f'  G[{a},{b}] = {val}')",
    ),
    (
        "markdown",
        "**2D polar with made-up A, B:** the result will reveal that "
        "B doesn't appear in G, Lovelock 2D theorem in action!",
    ),
    (
        "code",
        "x, y = sp.symbols('x y')\n"
        "A, B = sp.symbols('A B')\n"
        "F2 = CoordinateFrame([x, y])\n"
        "g2 = ComponentMetric(F2, sp.Matrix([[1, 0], [0, x**2]]))\n\n"
        "manual = sp.MutableDenseNDimArray.zeros(F2.dim, F2.dim, F2.dim)\n"
        "manual[0, 0, 0] = A          # Γ^x_{xx}\n"
        "manual[0, 1, 1] = B*x        # Γ^x_{yy}\n"
        "manual[1, 0, 1] = 1/x + A    # Γ^y_{xy}\n"
        "manual[1, 1, 0] = 1/x - A    # Γ^y_{yx}\n\n"
        "manual_conn = ComponentConnection(F2, manual)\n"
        "G = einstein_tensor(manual_conn, g2)\n"
        "for a in range(F2.dim):\n"
        "    for b in range(F2.dim):\n"
        "        val = sp.simplify(sp.trigsimp(G[a, b]))\n"
        "        if val != 0:\n"
        "            print(f'G[{a},{b}] = {val}')\n"
        "print(f'\\nA=0 → vacuum?  {all(sp.simplify(G[a,b].subs(A, 0)) == 0 for a in range(2) for b in range(2))}')",
    ),
    (
        "markdown",
        "**Lovelock 2D detected**: B doesn't appear in `G`. The "
        "torsion `T^y_{xy} = 2A` is the only thing breaking the "
        "2D Lovelock theorem (`G ≡ 0` for any torsion-free "
        "connection in 2D). At `A = 0`, the connection becomes "
        "torsion-free regardless of B, and `G` collapses to zero.",
    ),
    (
        "markdown",
        "### Physically-motivated deformation patterns\n\n"
        "Real paper work uses one of these patterns. Each is a "
        "Levi-Civita connection plus a specific deformation tensor "
        "parameterised by a small number of physical quantities.",
    ),
    (
        "markdown",
        "#### Pattern 1: Levi-Civita + antisymmetric torsion (2D polar)\n\n"
        "Single scalar `α` controls a torsion-violating perturbation. "
        "At `α = 0` recovers Levi-Civita (vacuum, by Lovelock 2D).",
    ),
    (
        "code",
        "x, y = sp.symbols('x y')\n"
        "alpha = sp.Symbol('alpha', real=True)\n"
        "F = CoordinateFrame([x, y])\n"
        "g = ComponentMetric(F, sp.Matrix([[1, 0], [0, x**2]]))\n"
        "LC = levi_civita(g)\n\n"
        "Gamma = sp.MutableDenseNDimArray(LC.components)\n"
        "Gamma[0, 0, 1] += alpha\n"
        "Gamma[0, 1, 0] -= alpha\n"
        "torsion_conn = ComponentConnection(F, Gamma)\n"
        "G_torsion = einstein_tensor(torsion_conn, g)\n\n"
        "print('Pattern 1: 2D polar + α torsion')\n"
        "print(f'  LC vacuum?       {einstein_tensor(LC, g).is_vacuum()}')\n"
        "print(f'  deformed vacuum? {G_torsion.is_vacuum()}')\n"
        "for a in range(F.dim):\n"
        "    for b in range(F.dim):\n"
        "        val = sp.simplify(sp.trigsimp(G_torsion[a, b]))\n"
        "        if val != 0:\n"
        "            print(f'  G[{a},{b}] = {val}')\n"
        "print(f'  α=0 recovers LC? {all(sp.simplify(G_torsion[a,b].subs(alpha, 0)) == 0 for a in range(2) for b in range(2))}')",
    ),
    (
        "markdown",
        "#### Pattern 2: Levi-Civita + Weyl non-metricity (2D polar)\n\n"
        "Single scalar `W_x` controls a Weyl-type deformation. "
        "**Surprise**: `G ≡ 0` even with `W_x ≠ 0`, because Weyl "
        "preserves torsion-freeness, Lovelock 2D still applies. "
        "Pedagogical lesson: torsion breaks Lovelock; non-metricity "
        "doesn't.",
    ),
    (
        "code",
        "W_x = sp.Symbol('W_x', real=True)\n"
        "LC = levi_civita(g)\n"
        "g_mat = g.matrix()\n"
        "g_inv = g_mat.inv()\n"
        "W = [W_x, 0]\n"
        "W_up = [sum(g_inv[a, b] * W[b] for b in range(F.dim)) for a in range(F.dim)]\n\n"
        "Gamma = sp.MutableDenseNDimArray(LC.components)\n"
        "for a in range(F.dim):\n"
        "    for b in range(F.dim):\n"
        "        for c in range(F.dim):\n"
        "            d_ab = 1 if a == b else 0\n"
        "            d_ac = 1 if a == c else 0\n"
        "            Gamma[a, b, c] += sp.Rational(1, 2) * (\n"
        "                d_ab * W[c] + d_ac * W[b] - g_mat[b, c] * W_up[a]\n"
        "            )\n\n"
        "weyl_conn = ComponentConnection(F, Gamma)\n"
        "G_weyl = einstein_tensor(weyl_conn, g)\n"
        "print('Pattern 2: 2D polar + Weyl non-metricity')\n"
        "print(f'  deformed vacuum? {G_weyl.is_vacuum()}  ← Lovelock 2D, even with W_x ≠ 0')",
    ),
    (
        "markdown",
        "#### Pattern 3: Schwarzschild + antisymmetric torsion (4D)\n\n"
        "Single scalar `ε` adds a `t-φ` cross-term torsion. The "
        "result is compact: only `G_{tφ}` and `G_{φt}` non-zero.",
    ),
    (
        "code",
        "F_sw, g_sw = schwarzschild()\n"
        "t, r, theta, phi = F_sw.coords\n"
        "epsilon = sp.Symbol('epsilon', real=True)\n"
        "LC = levi_civita(g_sw)\n\n"
        "Gamma = sp.MutableDenseNDimArray(LC.components)\n"
        "Gamma[3, 1, 0] += epsilon       # Γ^φ_{rt}\n"
        "Gamma[3, 0, 1] -= epsilon       # Γ^φ_{tr}\n"
        "torsion_conn = ComponentConnection(F_sw, Gamma)\n\n"
        "G_torsion = einstein_tensor(torsion_conn, g_sw)\n"
        "print('Pattern 3: Schwarzschild + ε torsion')\n"
        "print(f'  deformed vacuum? {G_torsion.is_vacuum()}')\n"
        "for a in range(F_sw.dim):\n"
        "    for b in range(F_sw.dim):\n"
        "        val = sp.simplify(sp.trigsimp(G_torsion[a, b]))\n"
        "        if val != 0:\n"
        "            print(f'  G[{a},{b}] = {val}')\n"
        "print(f'  ε=0 recovers LC? {all(sp.simplify(G_torsion[a,b].subs(epsilon, 0) - einstein_tensor(LC, g_sw)[a,b]) == 0 for a in range(4) for b in range(4))}')",
    ),
    (
        "markdown",
        "#### Pattern 4: Schwarzschild + Weyl non-metricity (4D)\n\n"
        "Same Weyl construction, on Schwarzschild. **Use `optimized=True`** "
        "because the 4D pipeline is much slower without it.",
    ),
    (
        "code",
        "W_r = sp.Symbol('W_r', real=True)\n"
        "LC = levi_civita(g_sw, optimized=True)   # optimized!\n"
        "g_mat = g_sw.matrix()\n"
        "g_inv = g_mat.inv()\n"
        "W = [0, W_r, 0, 0]\n"
        "W_up = [sum(g_inv[mu, nu] * W[nu] for nu in range(F_sw.dim)) for mu in range(F_sw.dim)]\n\n"
        "Gamma = sp.MutableDenseNDimArray(LC.components)\n"
        "for mu in range(F_sw.dim):\n"
        "    for nu in range(F_sw.dim):\n"
        "        for rho in range(F_sw.dim):\n"
        "            d_munu = 1 if mu == nu else 0\n"
        "            d_murho = 1 if mu == rho else 0\n"
        "            Gamma[mu, nu, rho] += sp.Rational(1, 2) * (\n"
        "                d_munu * W[rho] + d_murho * W[nu] - g_mat[nu, rho] * W_up[mu]\n"
        "            )\n\n"
        "weyl_conn = ComponentConnection(F_sw, Gamma)\n"
        "G_weyl = einstein_tensor(weyl_conn, g_sw, optimized=True)\n"
        "print('Pattern 4: Schwarzschild + W_r Weyl')\n"
        "print(f'  deformed vacuum? {G_weyl.is_vacuum()}')\n"
        "for a in range(F_sw.dim):\n"
        "    for b in range(F_sw.dim):\n"
        "        val = sp.simplify(sp.trigsimp(G_weyl[a, b]))\n"
        "        if val != 0:\n"
        "            print(f'  G[{a},{b}] = {val}')",
    ),
    (
        "markdown",
        "#### Pattern 5: FLRW + scalar-gradient projective deformation\n\n"
        "Connection deformed by a scalar field's gradient, typical "
        "scalar-tensor gravity setup. Coupling form `Γ + δ A_a + δ A_a` "
        "where `A_μ = ∂_μ φ`.",
    ),
    (
        "code",
        "tt, rr, thh, ph = sp.symbols('t r theta phi')\n"
        "a_func = sp.Function('a')(tt)\n"
        "varphi = sp.Function('varphi')(tt)\n"
        "F_flrw = CoordinateFrame([tt, rr, thh, ph])\n\n"
        "g_flrw = ComponentMetric(F_flrw, sp.Matrix([\n"
        "    [-1, 0, 0, 0],\n"
        "    [0, a_func**2, 0, 0],\n"
        "    [0, 0, a_func**2 * rr**2, 0],\n"
        "    [0, 0, 0, a_func**2 * rr**2 * sp.sin(thh)**2],\n"
        "]))\n"
        "LC = levi_civita(g_flrw)\n\n"
        "A = [sp.diff(varphi, c) for c in F_flrw.coords]\n"
        "Gamma = sp.MutableDenseNDimArray(LC.components)\n"
        "for mu in range(F_flrw.dim):\n"
        "    for nu in range(F_flrw.dim):\n"
        "        for rho in range(F_flrw.dim):\n"
        "            d_munu = 1 if mu == nu else 0\n"
        "            d_murho = 1 if mu == rho else 0\n"
        "            Gamma[mu, nu, rho] += d_munu * A[rho] + d_murho * A[nu]\n\n"
        "scalar_conn = ComponentConnection(F_flrw, Gamma)\n"
        "G_def = einstein_tensor(scalar_conn, g_flrw)\n"
        "print('Pattern 5: FLRW + scalar-gradient')\n"
        "print(f'  LC G_tt = {sp.simplify(einstein_tensor(LC, g_flrw)[0, 0])}')\n"
        "print(f'  G_def[0,0] = {sp.simplify(G_def[0, 0])}')",
    ),
    (
        "markdown",
        "### Insight summary\n\n"
        "Five patterns, three structural lessons:\n\n"
        "| Pattern | Recovers LC at | Lovelock 2D? |\n"
        "|---|---|---|\n"
        "| 2D + α torsion | `α = 0` | No (torsion breaks it) |\n"
        "| 2D + W_x Weyl | `W_x = 0` | **Yes** (`G ≡ 0` always) |\n"
        "| Schwarzschild + ε torsion | `ε = 0` | n/a (4D) |\n"
        "| Schwarzschild + W_r Weyl | `W_r = 0` | n/a (4D) |\n"
        "| FLRW + scalar-grad | `varphi'(t) = 0` | n/a (4D, non-vacuum) |\n\n"
        "Key takeaways:\n\n"
        "- **API stress-tests with arbitrary symbols** are valid "
        "and accidentally surface deep theorems (Lovelock 2D from "
        "the A, B examples).\n"
        "- **Physically-meaningful examples** parameterise the "
        "deformation by a small number of fields/constants, with the "
        "`parameter → 0` limit recovering Levi-Civita.",
    ),
    (
        "markdown",
        "## Summary\n\n"
        "* `jacopy.frame_calc` is the component-level submodule for "
        "concrete metric calculations.\n"
        "* Three frame types (CoordinateFrame, Tetrad, AbstractFrame) "
        "share a common `Frame` protocol; higher-level operations are "
        "frame-agnostic.\n"
        "* Pipeline: `g → LC → R → Ric → R_scalar → G`. Default mode "
        "records full derivation traces; optimised mode skips them "
        "for Kerr-class performance.\n"
        "* Library fixtures cover standard metrics; users build "
        "custom metrics on any frame.\n"
        "* `derivation_chain(...)` lifts any per-entry trace to a "
        "`ProofChain` for paper-grade LaTeX rendering.\n"
        "* SymPy is an opt-in dependency under `[components]`.",
    ),
]


# --------------------------------------------------------------------- #
# Builder                                                                #
# --------------------------------------------------------------------- #


def _build(cells: Iterable[tuple[str, str]]) -> "nbformat.NotebookNode":
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    built = []
    for kind, source in cells:
        if kind == "markdown":
            built.append(new_markdown_cell(source))
        elif kind == "code":
            built.append(new_code_cell(source))
        else:
            raise ValueError(f"unknown cell kind: {kind!r}")
    nb.cells = built
    return nb


def build_all() -> None:
    sources = {
        "01_first_steps.ipynb": TUTORIAL_01,
        "02_jacobi_identity.ipynb": TUTORIAL_02,
        "03_poisson_geometry.ipynb": TUTORIAL_03,
        "04_lie_algebroid.ipynb": TUTORIAL_04,
        "05_cartan_calculus.ipynb": TUTORIAL_05,
        "06_custom_bracket.ipynb": TUTORIAL_06,
        "07_derived_bracket.ipynb": TUTORIAL_07,
        "08_unified_picture.ipynb": TUTORIAL_08,
        "09_foundations.ipynb": TUTORIAL_09,
        "10_diagnostics.ipynb": TUTORIAL_10,
        "11_publication_output.ipynb": TUTORIAL_11,
        "12_schouten_nijenhuis.ipynb": TUTORIAL_12,
        "13_closure_axioms.ipynb": TUTORIAL_13,
        "14_problem_wrappers.ipynb": TUTORIAL_14,
        "15_intrinsic_engine.ipynb": TUTORIAL_15,
        "16_phase_13_deep_dive.ipynb": TUTORIAL_16,
        "17_tilde_calculus.ipynb": TUTORIAL_17,
        "18_derivator_identities.ipynb": TUTORIAL_18,
        "19_courant_family.ipynb": TUTORIAL_19,
        "20_connection_curvature.ipynb": TUTORIAL_20,
        "21_indexed_sum_wedge_multi_eval.ipynb": TUTORIAL_21,
        "22_frame_decomposition.ipynb": TUTORIAL_22,
        "23_cartan_structure_equations.ipynb": TUTORIAL_23,
        "24_custom_problem_wrapper.ipynb": TUTORIAL_24,
        "25_frame_calc.ipynb": TUTORIAL_25,
    }
    for fname, cells in sources.items():
        nb = _build(cells)
        out = THIS_DIR / fname
        with out.open("w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        print(f"wrote {out}")


if __name__ == "__main__":
    build_all()
