# 06 — Custom Bracket

Daha önceki tutorial'lar paketin içinde hazır gelen bracket'lerle
çalıştı: `LieBracket`, `sn`, `KoszulBracket`, `CourantBracket`…
Kullanıcı kendi tanım-kuralını (expansion function) bir bracket
olarak sokmak istediğinde — sınıf yazıp `GradedBracket`'ten türemeden —
`gradalg.brackets.custom.CustomBracket` devreye girer. Bu tutorial
(a) `CustomBracket`'in minimum veri profilini, (b) flag'ler aracılığıyla
aksiyom profilini nasıl deklare ettiğini, (c) `prove_jacobi`'nin bu
bracket üstünde hangi yolu takip ettiğini, ve (d) ne zaman
`CustomBracket`'ten çıkıp `GradedBracket` alt-sınıfına geçilmesi
gerektiğini gösteriyor.

[05 — Cartan calculus](05_cartan_calculus.md) hazır bracket'lerin
operatör-seviyesi bağıntılarını inceledi; burada bir adım geri
çıkıp kendi bracket'imizi sembol-seviyesinde tanımlıyoruz.

## Minimum profil — iki argüman

`CustomBracket(name, expand_fn, *, degree, is_graded_antisymmetric,
satisfies_leibniz, satisfies_graded_jacobi)`. Zorunlu alanlar sadece
isim ve expand çağrısı — geri kalanı makul default'larla geliyor
(derece 0, antisymmetric, Leibniz var, Jacobi var).

`expand_fn` imzası sabit: `(a, b, registry) → Expr`. Registry rule'un
işine yaramasa bile her çağrıda geçilir — diğer bracket'lerle
çağrı-uyumluluğu için.

```python
from gradalg.brackets.custom import CustomBracket
from gradalg.core.expr import Neg, Product, Sum, Symbol


def commutator(a, b, registry):
    return Sum(Product(a, b), Neg(Product(b, a)))


B = CustomBracket("[·,·]", commutator)
B(Symbol("X"), Symbol("Y")).expand()
# ((X * Y) + (-(Y * X)))
```

`B(X, Y)` bir `BracketApply` düğümü verir; üstünde `.expand()` ya da
paket seviyesinde `expand_bracket(...)` çağrılınca `commutator` rule'u
tetiklenir.

## Aksiyom profili flag'leri

Dört flag bracket'in aksiyomatik iddialarını sembolik olarak tutar.
Engine'in hangi kısayolları seçebileceğini ve `prove_jacobi`
dispatch'inin beklenti seviyesini bunlar belirler:

| flag | anlam | default |
|------|-------|---------|
| `degree` | `[·,·]`'nin derece kayması (`|[a,b]| = |a|+|b|+degree`) | 0 |
| `is_graded_antisymmetric` | `[a,b] = −(−1)^{|a||b|}[b,a]` | `True` |
| `satisfies_leibniz` | Leibniz kuralı 2. slot'ta | `True` |
| `satisfies_graded_jacobi` | Graded Jacobi | `True` / `False` / `None` |

`None` bir üçüncü seçenek — *koşullu Jacobi*. `DerivedBracket` buna
kanonik örnek; kendi CustomBracket'inin Jacobi'si ayrı bir koşula
bağlıysa (örneğin `[Q,Q]_base = 0`), flag'ı `None` ver, ispat katmanı
uygun stratejiyi seçsin.

```python
B_asym = CustomBracket(
    "asym",
    lambda a, b, reg: Product(a, b),
    is_graded_antisymmetric=False,
    satisfies_leibniz=False,
    satisfies_graded_jacobi=False,
)
B_asym.is_graded_antisymmetric, B_asym.satisfies_graded_jacobi
# (False, False)
```

## `prove_jacobi` — generic dispatch yolu

`CustomBracket` bir `DerivedBracket` değil; `prove_jacobi` dispatch
tablosunda `GradedBracket` generic yoluna düşer. Bu yol:

1. `graded_jacobi_obstruction(a, b, c, registry)` — triple cyclic sum
   `(−1)^{|a||c|}[a,[b,c]] + …`.
2. Bracket düğümlerini `expand_fn` ile tamamen açar (`bracket-expand`
   adımı).
3. `ExpandAndSimplify` stratejisini `Integer(0)`'a karşı koşar.

Commutator rule'u için tüm bu zincir sıfıra kapanır:

```python
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.proof.verifier import prove_jacobi

reg = PropertyRegistry()
for s in (Symbol("X"), Symbol("Y"), Symbol("Z")):
    reg.declare(s, Graded(degree=0))

chain = prove_jacobi(B, Symbol("X"), Symbol("Y"), Symbol("Z"), registry=reg)
len(chain)                        # 2
chain.steps[0].rule               # 'bracket-expand'
chain.steps[1].rule               # 'simplify'
chain.steps[1].after              # 0
```

Yanlış bir kural seçerseniz aynı pipeline residual bırakır ve
`ProofFailure` fırlatır:

```python
from gradalg.proof.strategies import ProofFailure

try:
    prove_jacobi(B_asym, Symbol("X"), Symbol("Y"), Symbol("Z"), registry=reg)
except ProofFailure as exc:
    str(exc)
# "ExpandAndSimplify left residual (3 * X * Y * Z) when proving ... == 0"
```

Mesajdaki residual — `3 * X * Y * Z` — `asym` kuralının Jacobi
özdeşliğini çözmediğinin doğrudan sembolik kanıtı.

## Axiom obstruction helper'ları

`GradedBracket`'ten miras gelen üç yardımcı, aksiyomun iddiası olan
ifadeyi açıkça verir. İspata gitmeden önce rule'u bir sembolik
üçlü/ikili üstünde probe etmek için pratik:

```python
a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
for s in (a, b, c):
    reg.declare(s, Graded(degree=0))

B.graded_antisymmetry_obstruction(a, b, reg)
# ([·,·](a, b) + [·,·](b, a))

B.graded_jacobi_obstruction(a, b, c, reg)
# ([·,·](a, [·,·](b, c)) + [·,·](b, [·,·](c, a)) + [·,·](c, [·,·](a, b)))

B.leibniz_obstruction(a, b, c, reg)
# ([·,·](a, (b * c)) + (-([·,·](a, b) * c)) + (-(b * [·,·](a, c))))
```

Her biri bir `Expr` — `simplify(..., reg)` ile sıfıra indirmek rule'un
o aksiyomu sağladığı iddiasının testidir. Parite karar-dışıysa
(`None`), `ValueError` fırlatır ve operand derecelerini daraltmanız
gerekir. Bu, hatanın erken sembolik düzeyde yakalanması için
tasarlandı — ispat katmanı yerine burada yakalamak daha ucuz.

## `_identity_key` ve eşitlik

İki `CustomBracket` ancak *aynı* expand callable'ını paylaşırsa eşit
sayılır. Python fonksiyon kimliği kullanılır — isim ve derece aynı
olsa bile iki farklı `lambda` farklı bracket'tir:

```python
rule_a = lambda a, b, reg: Sum(Product(a, b), Neg(Product(b, a)))
rule_b = lambda a, b, reg: Sum(Product(a, b), Product(b, a))
CustomBracket("B", rule_a) == CustomBracket("B", rule_b)   # False
```

Bu tasarım kasıtlı: aksi halde `DerivedBracket(base=B_a, ...)`'nın
hash table anahtarı yanlış eş-çakılırdı.

## Ne zaman CustomBracket'ten çıkılmalı

`CustomBracket` iki durumda doğru araç:

- Tutorial / keşif: hızlıca bir rule yazıp üstünde Jacobi / Leibniz
  testleri koşmak.
- Aksiyomları henüz netleşmemiş bir bracket'in çalışma tezgâhı.

Ama `expand_definition`, obstruction hook'ları, anchor lift'i,
özel kimlik alanı (`_identity_key` extension'ı), veya theorem_book
tarafından başvurulacak bir `Theorem` kaydı istiyorsan `GradedBracket`'e
alt-sınıf yazmak gerekir. Paketin kendi bracket'leri
(`KoszulBracket`, `CourantBracket`, `DerivedBracket`, `SchoutenBracket`)
bu ikincisine örnek.

## Sonraki adım

`CustomBracket` bir rule gösterir ama bir *yapı* çıkarmaz. Derived
bracket ise tek bir "generator" `Q` seçerek üstteki bracket
aksiyomlarını otomatik inşa eder — Poisson, Koszul, Courant
bracket'lerinin hepsi bu inşanın ayrı örnekleri.
[07 — Derived bracket](07_derived_bracket.md).
