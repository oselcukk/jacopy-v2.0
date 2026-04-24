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


TUTORIAL_01: list[tuple[str, str]] = [
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
        "modül-seviyesi singleton'ıdır. Üç derece-0 sembolü `X, Y, Z` "
        "declare ediyoruz — vector field olarak davranacaklar.",
    ),
    (
        "code",
        "from gradalg.brackets.lie import lie\n"
        "from gradalg.core.expr import Symbol\n"
        "from gradalg.core.properties import Graded\n"
        "from gradalg.core.registry import PropertyRegistry\n\n"
        "X, Y, Z = Symbol(\"X\"), Symbol(\"Y\"), Symbol(\"Z\")\n"
        "reg = PropertyRegistry()\n"
        "for s in (X, Y, Z):\n"
        "    reg.declare(s, Graded(degree=0))",
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
    }
    for fname, cells in sources.items():
        nb = _build(cells)
        out = THIS_DIR / fname
        with out.open("w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        print(f"wrote {out}")


if __name__ == "__main__":
    build_all()
