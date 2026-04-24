# 02 — Jacobi Özdeşliği

Bu tutorial, Lie bracket üzerinde Jacobi özdeşliğinin `gradalg` içinde
nasıl bir `ProofChain` olarak kapatıldığını gösteriyor. [İlk adımlar
tutorial'ına](01_first_steps.md) aşinalık önerilir.

## Lie bracket

Modül `gradalg.brackets.lie` iki şey sağlar:

- `LieBracket` — isim alabilen `GradedBracket` alt sınıfı.
- `lie` — standart ("TM") Lie bracket'in process-wide singleton'ı.

Tutorial'larda hep bu singleton'ı kullanıyoruz. Özel bir manifold veya
cebroid üstündeyken kendi `LieBracket(name="[·,·]_E")` instance'ınızı
yaratırsınız.

```python
from gradalg.brackets.lie import lie
```

## Üç vector field

Jacobi için üç sembol gerekli. `VectorFields` yardımcısı her sembole
`Graded(degree=0)` declare eder — `PropertyRegistry` Jacobi expansion
sırasında işaret kurallarını bu dereceden türetiyor.

```python
from gradalg import VectorFields
from gradalg.core.registry import PropertyRegistry

reg = PropertyRegistry()
X, Y, Z = VectorFields("X Y Z", registry=reg)
```

Alternatif (primitive) yol için [01_first_steps.md](01_first_steps.md)
altındaki "Özellik atama" bölümüne bakın — yardımcı yalnızca
`Symbol(...) + reg.declare(sym, Graded(degree=0))` deseninin bir
sargısı.

## `prove_jacobi`

Graded Jacobi özdeşliği

$$[X,[Y,Z]] + (-1)^{|X||Y|+|X||Z|}\,[Y,[Z,X]] + (-1)^{|Y||Z|+|X||Z|}\,[Z,[X,Y]] = 0$$

Derece-0 durumunda tüm işaretler `+1`; sonuç standart Jacobi:

$$[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0.$$

`prove_jacobi`, bracket + üç operand + registry alır ve bir `ProofChain`
döndürür:

```python
from gradalg.proof import prove_jacobi

chain = prove_jacobi(lie, X, Y, Z, registry=reg)
assert chain.steps[-1].after  # final state; 0 olmalı
```

Zincirin uzunluğu bracket'in iç rewrite kuralı sayısına bağlı. `lie`
üzerinde iki adımlık bir zincir üretiliyor; ilk adım Jacobi obstruction'ı
bracket'i açıp canonical forma alır, ikincisi kalıntıyı sıfıra indirir.

## Görselleştirme

ASCII:

```python
from gradalg.display import chain_to_ascii
print(chain_to_ascii(chain))
```

Jupyter içinde LaTeX (`align*` blokları olarak):

```python
from gradalg.display import display_chain
display_chain(chain)
```

## Koşullu Jacobi

Tüm bracket'ler Jacobi'yi unconditional sağlamaz. Örneğin
`CourantBracket`: `satisfies_graded_jacobi=None`; H-twist var ise
obstruction `dH`'ye eşit olur, yoksa `0`. `prove_jacobi` aynı sinyali
taşıyan bir `ProofChain` döndürmez — bunun yerine `bracket.jacobi_condition()`
üzerinden bir `VanishingCondition` alır, `ProofChain` ise kütüphane
seviyesindeki helper'lar (örn. `CourantAlgebroid.prove_jacobi_reduction`)
aracılığıyla kurulur.

Sonraki tutorial ([03_poisson_geometry.md](03_poisson_geometry.md))
Poisson bracket'in derived-bracket teoreminden Jacobi'yi nasıl
türettiğini gösteriyor.
