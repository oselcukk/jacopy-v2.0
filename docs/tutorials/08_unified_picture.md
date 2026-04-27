# 08 — Birleşik Tablo

Bu paketin pedagojik iddiası: Poisson geometri + Lie algebroid +
Cartan calculus + Courant geometri tek bir matematiksel mekanizmanın
yüzleri. Merkezde *Derived Bracket Teoremi* duruyor — üstüne oturan
bracket'in bracket-aksiyomları (antisymmetry + graded Jacobi)
`[Q, Q]_base = 0` tek denklemiyle kontrol ediliyor. Bu tutorial aynı
hipotezin iki farklı yüzde nasıl hem fonksiyon-Jacobi hem
form-Jacobi'yi kapattığını, theorem_book'un bu hiyerarşiyi nasıl
tek citation zinciriyle servis ettiğini, ve "tek varsayım, çok
sonuç" pedagojisinin neden bu mimarinin doğal bir yan ürünü olduğunu
gösteriyor.

[07 — Derived bracket](07_derived_bracket.md) mekanizmayı tanıttı;
[05 — Cartan calculus](05_cartan_calculus.md) operatör-seviyesi
ispatları açtı. Burada bu parçaları birbirine bağlıyoruz.

## Tek hipotez: `[π, π]_SN = 0`

Bir `π` bivector'ünün "Poisson bivector" olması, Schouten-Nijenhuis
bracket'i altında kendisiyle anti-commutation yapması demektir:
`[π, π]_SN = 0`. `jacopy` bu denklemi evrensel bir
`VanishingCondition` olarak üretir:

```python
from jacopy.library.declarations import Bivector, Forms, Functions
from jacopy.library.poisson import PoissonBracket
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
pi = Bivector("π", registry=reg)
poisson = PoissonBracket.from_bivector(pi)

poisson.jacobi_condition(reg).obstruction     # [·,·]_SN(π, π)
poisson.koszul_jacobi_condition(reg).obstruction  # aynısı
```

İki koşul aynı `Expr`'e işaret eder — sadece display ismi farklı.
Bu bir tesadüf değil: derived bracket'in obstruction'ı yalnız
`(base, Q)` ikilisine bağlı; `acting_on=π^♯` yeniden yazımını
eklemek Jacobi obstruction'ını değiştirmez.

## Aynı koşul, iki ispat yüzü

**Fonksiyon seviyesi.** Klasik Poisson Jacobi özdeşliği üç fonksiyon
üstünde cyclic sum'ı sıfır kılar. `prove_jacobi_reduction` bunu tek
adımlık *DerivedBracketTheorem* step'iyle obstruction'a indirger:

```python
f, g, h = Functions("f g h", degree=-1, registry=reg)
func_chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
func_chain.steps[0].rule    # 'DerivedBracketTheorem'
func_chain.steps[0].after   # [·,·]_SN(π, π)
```

**Form seviyesi.** Aynı mekanizma 1-formlar üstünde Koszul
Jacobi'sini verir. `prove_koszul_jacobi_reduction` aynı teorem
citation'ını üretir ve *aynı* `[π, π]_SN` obstruction'ına varır —
`π^♯` anchor'ı sadece operand-lift için devrede:

```python
alpha, beta, gamma = Forms("α β γ", degree=1, registry=reg)
form_chain = poisson.prove_koszul_jacobi_reduction(
    alpha, beta, gamma, registry=reg
)
form_chain.steps[0].rule    # 'DerivedBracketTheorem'
form_chain.steps[0].after   # [·,·]_SN(π, π)

# Aynı obstruction'da kesişiyorlar:
func_chain.steps[0].after == form_chain.steps[0].after   # True
```

Matematiksel içerik: "klasik Poisson Jacobi" ve "klasik Koszul
Jacobi" iki farklı teorem *değil* — her ikisi de Derived Bracket
Teoremi'nin aynı universal obstruction'a indirdiği iki görüntü.

## Klasik–derived köprü

Aynı hipotez, ek olarak "klasik Koszul bracket = SN-derived bracket
(π^♯ ile)" teoremini de verir. `PoissonBracket` onu tek satırda
doğrular:

```python
chain = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)
len(chain)                # 1
chain.steps[0].rule       # 'reflexive' — iki tarafın canonical form'ı eşit
```

`reflexive` step'inin anlamı: paketin expand kuralları her iki
tarafı da aynı `Expr` ağacına getirdi; eşitlik "yapısal" olarak
sağlandı. İşte tam bu yapısal kimlik, Koszul Jacobi'nin fonksiyon
Jacobi'siyle aynı obstruction'a düşmesinin sebebi.

## Seeded teoremlerle citation zinciri

Üç seeded teorem `theorem_book` içinde bekliyor. Her biri `from_axioms`
üstünden hangi aksiyomlara dayandığını beyan ediyor — bu paketin
*property provenance* felsefesinin teorem-seviyesi karşılığı:

```python
from jacopy.library import theorem_book

theorem_book.get("poisson_jacobi").from_axioms
# ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')

theorem_book.get("poisson_koszul_equivalence").from_axioms
# ('derived bracket definition', 'classical Koszul bracket definition',
#  'π^♯ = Sharp(π) as common anchor')

theorem_book.get("poisson_koszul_jacobi").from_axioms
# ('Derived Bracket Theorem', 'π^♯ = Sharp(π) as form-lift anchor',
#  '[π, π]_SN = 0 (Poisson hypothesis)')
```

Her citation'ın arkasında bir `ProofChain` duruyor — teorem'in
kanonik ispatı. Downstream kod `theorem_book.get(...)` ile bu
chain'i olduğu gibi alıp daha büyük bir ispata gömebilir:

```python
thm = theorem_book.get("poisson_jacobi")
thm.proof                 # ProofChain(1 steps)
thm.proof.steps[0].rule   # 'DerivedBracketTheorem'
```

Aynı pattern Lie algebroid (`lie_algebroid_anchor_compat`) ve Courant
geometri (`courant_jacobi_twist`, `courant_dorfman_bridge`,
`dirac_isotropy`, `dirac_involutivity`) için de var — paket şu an 8
seeded teorem taşıyor.

## Paralel bir örnek: `dH = 0`

Aynı "tek denklem, çok sonuç" deseni Courant tarafında farklı bir
hipotezle tekrar ediyor. H-twisted Courant bracket'in graded
Jacobi'si ancak ve ancak twist 3-formu kapalıysa (`dH = 0`) tutar:

```python
from jacopy.brackets.courant import CourantBracket
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded

reg_h = PropertyRegistry()
H = Symbol("H")
reg_h.declare(H, Graded(degree=3))

C = CourantBracket(background_H=H)
cond = C.jacobi_condition(reg_h)
cond.obstruction          # d(H)

theorem_book.get("courant_jacobi_twist").from_axioms
# ('Courant algebroid Jacobi axiom', 'dH = 0 (closed-3-form hypothesis)')
```

Yapısal analoji: Poisson tarafında `[π, π]_SN = 0` tek denklemi,
Courant tarafında `dH = 0` tek denklemi — her iki durumda da
bracket'in iki aksiyomu (antisymmetry + Jacobi) tek bir koşula
indirgeniyor ve teorem kitabı bu indirgemeyi tek citation olarak
dışarı veriyor.

## Pedagojik özet

Paket tasarımının gözlemlenebilir sonuçları:

1. **Tek teorem, çok sonuç.** Derived Bracket Teoremi bir kez
   ispatlanır; Poisson function Jacobi, Poisson form Jacobi,
   Koszul Jacobi, Courant Jacobi — hepsi onun instantiate edilmiş
   halleri.
2. **Obstruction paylaşımı.** İki farklı ispat yolu (fonksiyon vs
   form) aynı `Expr`'e inanıyorsa, arkalarında aynı teorem
   yatıyordur. Paket bunu mekanik olarak yakalar — display isimleri
   farklı olsa bile.
3. **Citation zinciri izlenebilir.** `theorem_book.get(name).from_axioms`
   teorem'in hangi atomik varsayımlara dayandığını gösterir; bir
   makalenin ispat akışını `from_axioms` listeleriyle birlikte
   çizmek mümkün.
4. **Yeni bracket, eski teorem.** Yeni bir derived bracket tanımlayan
   kullanıcı Jacobi ispatını sıfırdan yapmaz — `prove_jacobi`
   otomatik olarak `DerivedBracketStrategy`'ye gider ve aynı teorem
   citation'ı'nı yeniden kullanır.

## Sonraki adım

Bu tutorial "hangi teoremi nereye bağlarım?" sorusunu yanıtladı.
Son tutorial — [09 — Temeller](09_foundations.md) — "bir aksiyom
nereden geliyor?" sorusuna iniyor. `d² = 0` neden bir aksiyom;
foundational mode'da bunu generator seviyesinden üreten sub-proof
nedir; özel bir aksiyom seti ile çalışırken ispat katmanı kendini
nasıl yeniden kurar.
