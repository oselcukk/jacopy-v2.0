"""Build the companion Jupyter notebooks for the tutorial markdowns.

Each tutorial pairs a ``NN_topic.md`` with an executable ``NN_topic.ipynb``.
Rather than maintain two copies by hand, the notebook sources live here
as Python dicts — one ``(markdown cells, code cells)`` tuple per
tutorial — and :func:`build_all` serialises them to ``.ipynb`` via
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


# Every notebook opens with a bootstrap cell that ensures ``gradalg``
# is importable even when the notebook is opened directly in the IDE
# (not via pytest, which has its own PYTHONPATH fixture). We try a
# plain import first and only touch ``sys.path`` on failure, so a
# user who ``pip install -e .``'d the repo sees no side-effects.
_BOOTSTRAP = (
    "code",
    "# Ensure gradalg is importable when this notebook is opened\n"
    "# directly (not via pytest). Walks up from the notebook's\n"
    "# directory to the repo root and prepends it to sys.path if\n"
    "# gradalg isn't already installed into this kernel.\n"
    "try:\n"
    "    import gradalg  # noqa: F401\n"
    "except ModuleNotFoundError:\n"
    "    import sys\n"
    "    from pathlib import Path\n"
    "    here = Path.cwd().resolve()\n"
    "    for candidate in (here, *here.parents):\n"
    "        if (candidate / \"gradalg\" / \"__init__.py\").is_file():\n"
    "            sys.path.insert(0, str(candidate))\n"
    "            break\n"
    "    import gradalg  # noqa: F401",
)


TUTORIAL_01: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 01 — İlk Adımlar\n\n"
        "Bu notebook [01_first_steps.md](01_first_steps.md) markdown'ının "
        "çalıştırılabilir sürümüdür. `gradalg`'de sembolik ifade kurma, "
        "özellik atama ve sadeleştirmenin temelleri.",
    ),
    (
        "markdown",
        "## Semboller ve temel inşa\n\n"
        "`Symbol(name)` atomik bir ifade üretir; `Integer(n)` literal sabit; "
        "`+`, `*`, `-` arka planda `Sum`, `Product`, `Neg` inşa eder.",
    ),
    (
        "code",
        "from gradalg.core.expr import Symbol, Integer, Sum, Product, Neg\n\n"
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
        "## Özellik atama (PropertyRegistry)\n\n"
        "Özellikler — derece, skalerlik, graded antisymmetry, … — semboller "
        "üzerine *dışsal* olarak ilan edilir. Bu, aynı sembolün farklı "
        "bağlamlarda (farklı derecelerde, farklı cebirlerde) yeniden "
        "kullanılmasına izin verir.",
    ),
    (
        "code",
        "from gradalg.core.properties import Graded, Scalar\n"
        "from gradalg.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "reg.declare(x, Scalar())\n"
        "reg.declare(y, Scalar())\n"
        "reg.declare(z, Graded(degree=1))  # z bir 1-form gibi davranır",
    ),
    (
        "markdown",
        "### Role-driven kısayollar\n\n"
        "Sık tekrarlanan desenler — fonksiyon, vektör alanı, form, "
        "bivector — için `gradalg.library.declarations` altında "
        "`Functions`, `VectorFields`, `Forms`, `Bivector` yardımcıları "
        "var. Her biri `Symbol(...)` + uygun `reg.declare(...)` "
        "çağrısını tek satıra indirir.",
    ),
    (
        "code",
        "from gradalg import Functions, VectorFields, Forms, Bivector\n\n"
        "reg2 = PropertyRegistry()\n"
        "f, g = Functions(\"f g\", registry=reg2)\n"
        "X, Y = VectorFields(\"X Y\", registry=reg2)\n"
        "alpha, beta = Forms(\"α β\", degree=1, registry=reg2)\n"
        "pi = Bivector(\"π\", registry=reg2)\n"
        "print(f, g, X, Y, alpha, beta, pi)",
    ),
    (
        "markdown",
        "## `simplify`: canonical forma indirme\n\n"
        "`simplify(expr, registry)` pipeline'ı: flatten → canonicalize → "
        "distribute → flatten → sort_product → collect_terms. "
        "Registry verilirse kayıtlı özellikler (commutativity, derece) "
        "işleme alınır.",
    ),
    (
        "code",
        "from gradalg.algorithms.simplify import simplify\n\n"
        "assert simplify(x + x - x) == x\n"
        "assert simplify(Product(Integer(2), x, Integer(3)), reg) \\\n"
        "    == Product(Integer(6), x)\n"
        "# Scalar iki sembol alfabetik sıraya alınır:\n"
        "assert simplify(Product(y, x), reg) == Product(x, y)\n"
        "print('simplify checks passed')",
    ),
    (
        "markdown",
        "## Görselleştirme\n\n"
        "`display` katmanı `to_ascii`, `to_latex` ve (rich yüklüyse) renkli "
        "terminal ağacı verir.",
    ),
    (
        "code",
        "from gradalg.display import to_ascii, to_latex\n\n"
        "e = x + y - z\n"
        "print('ascii:', to_ascii(e))\n"
        "print('latex:', to_latex(e))",
    ),
    (
        "markdown",
        "## Sonraki adım\n\n"
        "Bir bracket üstüne Jacobi ispatı: "
        "[02_jacobi_identity.md](02_jacobi_identity.md).",
    ),
]


TUTORIAL_02: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 02 — Jacobi Özdeşliği\n\n"
        "Bu notebook [02_jacobi_identity.md](02_jacobi_identity.md) "
        "markdown'ının çalıştırılabilir sürümüdür. Lie bracket üzerinde "
        "Jacobi özdeşliğinin `prove_jacobi` ile tek satırda nasıl "
        "kapatıldığını gösterir.",
    ),
    (
        "markdown",
        "## Lie bracket ve üç vector field\n\n"
        "`gradalg.brackets.lie.lie` standart manifold Lie bracket'inin "
        "modül-seviyesi singleton'ıdır. Üç vector field'ı `VectorFields` "
        "yardımcısıyla tek satırda deklare ediyoruz — her birine "
        "`Graded(degree=0)` iliştirilir, Jacobi expansion'ın işaret "
        "kuralları için gereken budur.",
    ),
    (
        "code",
        "from gradalg import VectorFields\n"
        "from gradalg.brackets.lie import lie\n"
        "from gradalg.core.registry import PropertyRegistry\n\n"
        "reg = PropertyRegistry()\n"
        "X, Y, Z = VectorFields(\"X Y Z\", registry=reg)",
    ),
    (
        "markdown",
        "## `prove_jacobi`: Jacobi obstruction sıfır mı?\n\n"
        "Graded Jacobi özdeşliği\n"
        "\n"
        "$$[X,[Y,Z]] + (-1)^{|X||Y|+|X||Z|}[Y,[Z,X]] + (-1)^{|Y||Z|+|X||Z|}[Z,[X,Y]] = 0$$\n"
        "\n"
        "Lie bracket'te dereceler sıfır olduğu için tüm işaretler `+1`'dir. "
        "`prove_jacobi(lie, X, Y, Z, registry=reg)` obstruction'ı sadeleştirip "
        "sıfıra indirgeyen bir `ProofChain` döndürür.",
    ),
    (
        "code",
        "from gradalg.proof import prove_jacobi\n\n"
        "chain = prove_jacobi(lie, X, Y, Z, registry=reg)\n"
        "print('chain length:', len(chain))\n"
        "print('final:', chain.steps[-1].after)",
    ),
    (
        "markdown",
        "## Zincirin adım adım görselleştirilmesi\n\n"
        "`display` katmanı `ProofChain`'leri ASCII veya LaTeX olarak "
        "render eder. Jupyter içinde `display_chain` MathJax tarafından "
        "render edilen bir LaTeX `align*` bloğu verir.",
    ),
    (
        "code",
        "from gradalg.display import chain_to_ascii\n\n"
        "print(chain_to_ascii(chain))",
    ),
    (
        "code",
        "# Jupyter'da otomatik LaTeX render (align* blokları):\n"
        "from gradalg.display import display_chain\n"
        "display_chain(chain)",
    ),
    (
        "markdown",
        "## Neden bu çalışıyor?\n\n"
        "Lie bracket'in sınıf tanımında `is_graded_antisymmetric=True` ve "
        "`satisfies_graded_jacobi=True` bayrakları işaretlidir. "
        "`prove_jacobi` ilk önce obstruction'ı kuruyor, sonra expansion "
        "engine'in Jacobi tanıyıcısı ile sadeleştirip kalıntıyı sıfıra "
        "indiriyor. Koşullu Jacobi taşıyan bracket'lerde "
        "(`CourantBracket` gibi) aynı fonksiyon obstruction'ı sıfıra "
        "indirmez; bu durumda dönen chain koşulun (ör. `dH = 0`) ne "
        "olduğunu açıkça kayda alır.",
    ),
    (
        "markdown",
        "## Sonraki adım\n\n"
        "Poisson geometriye geçiş: [03_poisson_geometry.md](03_poisson_geometry.md) "
        "bir sonraki stage'de.",
    ),
]


TUTORIAL_03: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 03 — Poisson Geometri\n\n"
        "Bu notebook [03_poisson_geometry.md](03_poisson_geometry.md) "
        "markdown'ının çalıştırılabilir sürümüdür. Symplectic manifold "
        "üstünde Poisson bracket'inin üç eşdeğer görüşü (derived, "
        "Hamiltonian, Koszul) ve `[π, π]_SN = 0` tek koşuluna indirgenmiş "
        "Jacobi ispatı.",
    ),
    (
        "markdown",
        "## Symplectic manifold — (ω, π, ♭, ♯) demet\n\n"
        "`SymplecticManifold(ω, bivector=π)` formu, ters-bivector'ı, "
        "musical map'leri ve `MusicalCompatibility` aksiyomunu tek "
        "objede tutar. Registry'de `ω` 2-form, `π` SN-derecesi 1'lik "
        "2-vektör olarak deklare edilir — `Bivector` yardımcısı bunu "
        "otomatik yapar.",
    ),
    (
        "code",
        "from gradalg import Bivector, Forms, Functions\n"
        "from gradalg.core.registry import PropertyRegistry\n"
        "from gradalg.library.symplectic import SymplecticManifold\n\n"
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
        "## `PoissonBracket` — üç eşdeğer görüş\n\n"
        "Fonksiyonlar SN-shifted grading'de degree `−1` taşır; "
        "`Functions` yardımcısına `degree=-1` kwarg'ı verilir.",
    ),
    (
        "code",
        "from gradalg.library.poisson import PoissonBracket\n\n"
        "f, g, h = Functions(\"f g h\", degree=-1, registry=reg)\n"
        "poisson = PoissonBracket.from_bivector(pi)\n"
        "print('bracket:', poisson)",
    ),
    (
        "markdown",
        "### Görüş 1 — derived bracket\n\n"
        "`{f, g}_π = [[f, π]_SN, g]_SN`.",
    ),
    (
        "code",
        "poisson.expand(f, g, reg)",
    ),
    (
        "markdown",
        "### Görüş 2 — Hamiltonian vector field\n\n"
        "`{f, g}_π = X_f(g)`. Symplectic manifold üstünde bu eşitlik "
        "`ι_{X_f} ω + df = 0`'ya denktir; `prove_hamiltonian_equivalence` "
        "musical kompatibiliteyi kullanarak beş adımda kapatır.",
    ),
    (
        "code",
        "from gradalg.display import chain_to_ascii\n\n"
        "print('X_f =', poisson.hamiltonian_vf(f))\n"
        "print('X_f(g) =', poisson.via_hamiltonian(f, g))\n"
        "\n"
        "chain = M.prove_hamiltonian_equivalence(f, registry=reg)\n"
        "print('chain length:', len(chain))\n"
        "print(chain_to_ascii(chain))",
    ),
    (
        "markdown",
        "### Görüş 3 — Koszul üç-terim formülü\n\n"
        "1-formlar üstünde `{α, β}_π = L_{π♯(α)} β − L_{π♯(β)} α − "
        "d⟨π♯(α), β⟩`. Klasik Koszul bracket ile derived bracket'in "
        "bu operand tipinde *yapısal eşitliği* `prove_koszul_equivalence` "
        "ile tek reflexive adımda kayda geçer.",
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
        "## `[π, π]_SN = 0` — tek koşul\n\n"
        "Derived Bracket Teoremi, `{·, ·}_π` üstündeki Jacobi'yi tek "
        "koşula indirger: `[π, π]_SN = 0`. Üç-giriş reduction zinciri "
        "`DerivedBracketTheorem` rule'u ile tek adımda obstruction'a "
        "varır; atomik `π` için obstruction opak kalır — Poisson "
        "hipotezi devreye girince Jacobi kapanır.",
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
        "## Theorem Book — seeded teorem\n\n"
        "Kütüphane bu indirgemeyi `poisson_jacobi` altında hazır bir "
        "`Theorem` kaydı olarak tutar — downstream kod tek citation ile "
        "sonuca bağlanır.",
    ),
    (
        "code",
        "from gradalg.library import theorem_book\n\n"
        "thm = theorem_book.get(\"poisson_jacobi\")\n"
        "print('statement:', thm.statement)\n"
        "print('from_axioms:', thm.from_axioms)",
    ),
    (
        "markdown",
        "## Sonraki adım\n\n"
        "Lie algebroid çerçevesi aynı derivation stratejisini bir vector "
        "bundle'ın üstünde yaşayan bir bracket'e uygular — "
        "[04_lie_algebroid.md](04_lie_algebroid.md).",
    ),
]


TUTORIAL_04: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 04 — Lie Algebroid\n\n"
        "Bu notebook [04_lie_algebroid.md](04_lie_algebroid.md) "
        "markdown'ının çalıştırılabilir sürümüdür. `(E, [·,·]_E, ρ)` "
        "üçlüsünün `gradalg` içindeki nesneleşmesi, anchor "
        "compatibility'nin ayrı aksiyom olarak ele alınışı, ve "
        "algebroid Cartan bundle'ına giriş.",
    ),
    (
        "markdown",
        "## Üçlü: (E, [·,·]_E, ρ)\n\n"
        "`LieAlgebroid` bundle adı, section bracket'i, anchor'u ve "
        "uyum hedefi olan TM bracket'ini tek objede tutar.",
    ),
    (
        "code",
        "from gradalg import VectorFields\n"
        "from gradalg.brackets.lie import LieBracket\n"
        "from gradalg.calculus.anchor import Anchor\n"
        "from gradalg.core.expr import Symbol\n"
        "from gradalg.core.registry import PropertyRegistry\n"
        "from gradalg.library.lie_algebroid import LieAlgebroid\n\n"
        "reg = PropertyRegistry()\n"
        "E = Symbol(\"E\")\n"
        "bracket_E = LieBracket(name=\"[·,·]_E\")\n"
        "rho = Anchor(name=\"ρ\")\n\n"
        "A = LieAlgebroid(E, bracket=bracket_E, anchor=rho, name=\"E-algebroid\")\n"
        "print(A)",
    ),
    (
        "markdown",
        "## Anchor compatibility — ayrı aksiyom\n\n"
        "`ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}` Lie algebroid *tanımı*nın "
        "bir parçasıdır; bracket'in kendi aksiyomları içermez. Üç "
        "sunumu vardır: obstruction (Expr), koşul "
        "(VanishingCondition), ve aksiyom etiketli tek adımlık "
        "ProofChain.",
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
        "Aynı `CartanCalculus` API'si ama `E`-etiketli operatörler ile: "
        "`d_E`, `L_{E,X}`, `ι_{E,X}`. Operator-seviyesi `relation()` "
        "çağrısı `OperatorEquation` döner. Not: algebroid Cartan "
        "magic'i otomatik `verify` ile kapanmıyor (engine cartan-def "
        "rewrite'ı TM'ye bağlı); bu beklenen ve kayıtlı. Beş Cartan "
        "bağıntısının canlı ispatı için [05_cartan_calculus.md](05_cartan_calculus.md) "
        "TM üstünde çalışır.",
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
        "## Seeded teorem\n\n"
        "Compatibility aksiyomu `theorem_book`'ta kayıtlı — downstream "
        "teoremler (algebroid Cartan, Courant–Dorfman köprüsü) tek "
        "citation ile bu aksiyomu kullanır.",
    ),
    (
        "code",
        "from gradalg.library import theorem_book\n\n"
        "thm = theorem_book.get(\"lie_algebroid_anchor_compat\")\n"
        "print('statement:', thm.statement)\n"
        "print('from_axioms:', thm.from_axioms)",
    ),
    (
        "markdown",
        "## Sonraki adım\n\n"
        "Beş Cartan bağıntısının TM üstünde iki modda canlı ispatı: "
        "[05_cartan_calculus.md](05_cartan_calculus.md).",
    ),
]


TUTORIAL_05: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 05 — Cartan Calculus\n\n"
        "Bu notebook [05_cartan_calculus.md](05_cartan_calculus.md) "
        "markdown'ının çalıştırılabilir sürümüdür. Beş Cartan "
        "bağıntısının `OperatorEquation` olarak inşası, `d² = 0` için "
        "axiom/theorem mod farkı, magic formülünün iki modda canlı "
        "ispatı, ve invariant-d helper'ı.",
    ),
    (
        "markdown",
        "## Bundle\n\n"
        "`CartanCalculus(d, L, ι, [·,·])` — dört ingredient tek objede.",
    ),
    (
        "code",
        "from gradalg.algebra.derivation import Derivation\n"
        "from gradalg.brackets.lie import LieBracket\n"
        "from gradalg.calculus.cartan import CartanCalculus, RELATIONS\n"
        "from gradalg.calculus.exterior_algebra import ExteriorAlgebra\n"
        "from gradalg.calculus.exterior_d import d\n"
        "from gradalg.calculus.interior import interior\n"
        "from gradalg.calculus.lie_derivative import lie_derivative\n"
        "from gradalg.core.expr import Symbol\n"
        "from gradalg.core.properties import Graded\n"
        "from gradalg.core.registry import PropertyRegistry\n\n"
        "cart = CartanCalculus(\n"
        "    d=d, lie_derivative=lie_derivative,\n"
        "    interior=interior, vector_bracket=LieBracket(),\n"
        ")\n"
        "print('RELATIONS:', RELATIONS)",
    ),
    (
        "markdown",
        "## Beş bağıntı, beş `OperatorEquation`",
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
        "## `d² = 0` — axiom mode vs theorem mode\n\n"
        "Sade helper `apply_d_squared_zero` her zaman 0'a çevirir. "
        "Default engine `d_squared_mode=\"theorem\"` + foundational "
        "modda d(d(x)) → 0 rewrite'ını ProofStep olarak kaydeder.",
    ),
    (
        "code",
        "from gradalg.calculus.exterior_d import apply_d_squared_zero\n"
        "from gradalg.proof.expansion import default_engine\n\n"
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
        "## Cartan magic — `verify` üzerinden canlı ispat\n\n"
        "`cartan_magic` iki modda da `ExteriorAlgebra((f,))` üstünde "
        "tek adımda kapanıyor.",
    ),
    (
        "code",
        "chain = cart.verify(\"cartan_magic\", algebra=algebra, X=X, registry=reg)\n"
        "print('efficient len:', len(chain), 'rule:', chain.steps[0].rule)\n\n"
        "chain_f = cart.verify(\n"
        "    \"cartan_magic\", algebra=algebra, X=X, registry=reg,\n"
        "    mode=\"foundational\",\n"
        ")\n"
        "print('foundational len:', len(chain_f), 'rule:', chain_f.steps[0].rule)",
    ),
    (
        "markdown",
        "## `d_lie`, `lie_lie`, `lie_iota` — henüz verify kapsamı dışında\n\n"
        "Bu üçünün `relation()`'ı `OperatorEquation` üretiyor; `verify()` "
        "mevcut baseline'da kapanmıyor (derece/grading sebepleri). "
        "Üzerinde deney yapacak kullanıcı bağıntıyı elle parçalar. "
        "Kapanış sonraki pass'te.",
    ),
    (
        "markdown",
        "## `invariant_d` — magic + lie_iota türevi teorem\n\n"
        "`dω(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y])` — 1-formlar için "
        "Koszul-Cartan invariant formülü. `InvariantDOneFormDefinition`'ın "
        "default classification'ı `\"theorem\"` (d²=0'ın tersine) — "
        "çünkü formül doğal olarak magic + lie_iota'dan türüyor.",
    ),
    (
        "code",
        "from gradalg.calculus.invariant_d import invariant_d_one_form\n"
        "from gradalg.brackets.lie import lie\n\n"
        "omega = Symbol(\"ω\")\n"
        "reg.declare(omega, Graded(degree=1))\n"
        "print(invariant_d_one_form(omega, X, Y, bracket=lie))",
    ),
    (
        "markdown",
        "## Sonraki adım\n\n"
        "Kendi bracket'iniz + Jacobi testi — "
        "[06_custom_bracket.md](06_custom_bracket.md) (Stage C).",
    ),
]


TUTORIAL_06: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 06 — Custom Bracket\n\n"
        "Bu notebook [06_custom_bracket.md](06_custom_bracket.md) "
        "markdown'ının çalıştırılabilir sürümüdür. `CustomBracket` "
        "ile kendi rule'unu tanımla, flag'lerle aksiyom profilini "
        "beyan et, `prove_jacobi` ile generic yolda test et.",
    ),
    (
        "markdown",
        "## Minimum profil — commutator rule\n\n"
        "`CustomBracket(name, expand_fn, *, degree=..., "
        "is_graded_antisymmetric=..., satisfies_leibniz=..., "
        "satisfies_graded_jacobi=...)`. expand_fn imzası "
        "`(a, b, registry) → Expr`.",
    ),
    (
        "code",
        "from gradalg.brackets.custom import CustomBracket\n"
        "from gradalg.core.expr import Neg, Product, Sum, Symbol\n\n"
        "def commutator(a, b, registry):\n"
        "    return Sum(Product(a, b), Neg(Product(b, a)))\n\n"
        "B = CustomBracket(\"[·,·]\", commutator)\n"
        "print('name:', B.name, 'degree:', B.degree)\n"
        "print('antisym:', B.is_graded_antisymmetric)\n"
        "print('expand X,Y:', B(Symbol('X'), Symbol('Y')).expand())",
    ),
    (
        "markdown",
        "## Aksiyom flag'leri\n\n"
        "Default profili komşu bracket'lerden farklı bir şey kurmak "
        "istersen flag'lerle beyan et. `satisfies_graded_jacobi=None` "
        "— koşullu Jacobi (derived bracket'teki gibi).",
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
        "## `prove_jacobi` — generic dispatch\n\n"
        "Commutator rule için zincir: bracket-expand → simplify → 0. "
        "Asimetrik kötü rule ise residual bırakır ve `ProofFailure` "
        "fırlatır.",
    ),
    (
        "code",
        "from gradalg.core.properties import Graded\n"
        "from gradalg.core.registry import PropertyRegistry\n"
        "from gradalg.proof.verifier import prove_jacobi\n"
        "from gradalg.proof.strategies import ProofFailure\n\n"
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
        "## Axiom obstruction helper'ları\n\n"
        "`GradedBracket`'ten miras: her aksiyomun iddia ettiği ifadeyi "
        "açıkça döner. İspata girmeden rule'u probe etmek için.",
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
        "## Eşitlik — callable kimliği\n\n"
        "İki `CustomBracket` ancak aynı `expand_fn` callable'ını "
        "paylaşırsa eşit.",
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
        "## Sonraki adım\n\n"
        "Generator tabanlı otomatik bracket inşası — "
        "[07_derived_bracket.md](07_derived_bracket.md).",
    ),
]


TUTORIAL_07: list[tuple[str, str]] = [
    _BOOTSTRAP,
    (
        "markdown",
        "# 07 — Derived Bracket\n\n"
        "Bu notebook [07_derived_bracket.md](07_derived_bracket.md) "
        "markdown'ının çalıştırılabilir sürümüdür. `{a, b}_Q := "
        "[[a, Q]_base, b]_base` inşası, Jacobi'nin tek bir "
        "denkleme (`[Q, Q]_base = 0`) indirilmesi, Koszul "
        "eşdeğerliği, Poisson ve H-twisted Courant köşeleri.",
    ),
    (
        "markdown",
        "## İnşa\n\n"
        "`DerivedBracket(base, Q, degree_Q=...)` — Lie base üstünde "
        "degree-1 `Q` seçelim. `|{·,·}_Q| = |Q| − 2 = −1`. Leibniz "
        "evrensel, antisymmetry/Jacobi koşullu (flag=None).",
    ),
    (
        "code",
        "from gradalg.brackets.derived import DerivedBracket\n"
        "from gradalg.brackets.lie import LieBracket\n"
        "from gradalg.core.expr import Symbol\n"
        "from gradalg.core.properties import Graded\n"
        "from gradalg.core.registry import PropertyRegistry\n\n"
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
        "## İki yüzlü expansion\n\n"
        "`expand` — iç/dış iki katman base açılmış; "
        "`expand_definition` — iki katman base `BracketApply` inert.",
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
        "## Jacobi obstruction — üç yüz\n\n"
        "Tek bir koşul: `[Q, Q]_base = 0`. Lie base için trivial "
        "(`Q*Q − Q*Q`).",
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
        "## `prove_jacobi` — DerivedBracketStrategy\n\n"
        "Bracket tipinden otomatik dispatch. Üç adım: "
        "DerivedBracketTheorem → base-bracket-expand → simplify.",
    ),
    (
        "code",
        "from gradalg.proof.verifier import prove_jacobi\n\n"
        "# a, b zaten yukarıda Graded(0) olarak kayıtlı — sadece c'yi ekle.\n"
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
        "## `acting_on` — Koszul eşdeğerliği\n\n"
        "SN base + π generator + anchor ρ: `expand` otomatik olarak "
        "Koszul 3-terim formunu emit eder. `KoszulBracket(ρ)` ile "
        "structurally eşit.",
    ),
    (
        "code",
        "from gradalg.brackets.schouten import sn\n"
        "from gradalg.brackets.koszul import KoszulBracket\n"
        "from gradalg.calculus.anchor import Anchor\n\n"
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
        "## Poisson-as-derived — library wrapper\n\n"
        "Matematiksel olarak `DerivedBracket(sn, π, degree_Q=1)` "
        "Poisson bracket. `[π, π]_SN`'nin SN-atomik olması sebebiyle "
        "`prove_jacobi` generic simplify yolu kapanmaz; "
        "`PoissonBracket.prove_jacobi_reduction` seeded teorem "
        "citation'ı ile bir adımda biter.",
    ),
    (
        "code",
        "from gradalg.library import theorem_book\n"
        "from gradalg.library.declarations import Bivector, Functions\n"
        "from gradalg.library.poisson import PoissonBracket\n\n"
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
        "## H-twist — Courant koşullu Jacobi\n\n"
        "`CourantBracket(background_H=H)`: Jacobi ⟺ dH = 0. "
        "Default (H=None) vacuous.",
    ),
    (
        "code",
        "from gradalg.brackets.courant import CourantBracket\n\n"
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
        "## Sonraki adım\n\n"
        "Stage D: birleşik tur + foundations — "
        "[08_unified_picture.md](08_unified_picture.md).",
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
    }
    for fname, cells in sources.items():
        nb = _build(cells)
        out = THIS_DIR / fname
        with out.open("w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        print(f"wrote {out}")


if __name__ == "__main__":
    build_all()
