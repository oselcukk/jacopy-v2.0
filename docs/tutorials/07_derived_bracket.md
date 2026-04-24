# 07 — Derived Bracket

Paketin matematiksel kalbi: tek bir bracket üstüne seçilmiş bir
*generator* ile bütün bir bracket ailesi doğar. Bu inşa — *derived
bracket* — `{a, b}_Q := [[a, Q]_base, b]_base`. Leibniz aksiyomu
`Q`'nun ne olduğundan bağımsız sağlanır; antisymmetry ve Jacobi ise
tek bir denkleme, `[Q, Q]_base = 0`'a bağlıdır. Poisson, Koszul,
Courant bracket'leri bu inşanın ayrı örnekleridir — Derived Bracket
Teoremi'ni bir kez ispatlayıp her birine tek citation'la uygularız.

[06 — Custom bracket](06_custom_bracket.md) genel `GradedBracket`
iskeletinin içine kendi rule'unu koymayı gösterdi. Burada ise rule
otomatik inşa ediliyor — siz yalnızca `(base, Q)`'yu seçiyorsunuz.

## İnşa — `{a, b}_Q := [[a, Q]_base, b]_base`

```python
from gradalg.brackets.derived import DerivedBracket, derived_bracket
from gradalg.brackets.lie import LieBracket
from gradalg.core.expr import Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry

reg = PropertyRegistry()
Q = Symbol("Q")
reg.declare(Q, Graded(degree=1))
lie = LieBracket()

d = DerivedBracket(lie, Q, degree_Q=1)
d.name           # '{·,·}_Q'
d.degree         # Degree.const(-1) — |Q| − 2 formülü
d.satisfies_leibniz          # True  (evrensel)
d.is_graded_antisymmetric    # True  (teoremin sonucu olarak beyan)
d.satisfies_graded_jacobi    # None  (koşullu — [Q,Q]_base = 0'a bağlı)
```

`d.satisfies_graded_jacobi` yukarıda `None` raporlanır: koşulsuz bir
`True`/`False` değil, *koşullu*. İspat katmanı bu koşulu
`[Q, Q]_base`'in sıfıra indirgenmesiyle ele alır.

Derece formülü: `|{·,·}_Q| = |Q| − 2`. `degree_Q` verilmediğinde
varsayılan 0'dır; o zaman derived bracket −2 kayar, ki bu genel
durumda pek anlamlı değildir. Canlı örnekler `degree_Q=1` (Poisson,
Koszul, Courant) etrafındadır.

## İki yüzlü expansion

İki kademeli açılımı *görmek* için iki yüz:

- `expand(a, b, registry)` — tam açılım; iç ve dış `[·, ·]_base`
  düğümleri çözülmüş halde döner.
- `expand_definition(a, b, registry)` — tanımın *yüzeyi*: dış
  `BracketApply` ve iç `BracketApply` iki katman inert bırakılmış.

```python
a, b = Symbol("a"), Symbol("b")
for s in (a, b):
    reg.declare(s, Graded(degree=0))

d.expand(a, b, reg)
# ((((a * Q) + (-(Q * a))) * b) + (-(b * ((a * Q) + (-(Q * a))))))

d.expand_definition(a, b, reg)
# [·,·]([·,·](a, Q), b)
```

`expand_definition`, ispat sunumunda "işte derived bracket'in kanonik
formu" olarak göstermek için; `expand` ise simplify pipeline'ına
beslenir.

## Jacobi obstruction — evrensel indirgeme

Derived Bracket Teoremi: graded Jacobi özdeşliği `{·,·}_Q` üstünde
**her** üçlü için geçerlidir, ancak ve ancak *tek bir* ifade sıfırsa:
`[Q, Q]_base`. Bu, paketteki her derived bracket için aynı matematiksel
argüman — teorem bir kez ispatlanır, her instantiation'a citation
olarak akar.

API üç yüz verir:

```python
d.jacobi_obstruction(reg)
# ((Q * Q) + (-(Q * Q)))  — base bracket'in π üstünde tam açılımı

d.jacobi_obstruction_raw()
# [·,·](Q, Q)              — base bracket inert, display için

d.jacobi_condition(reg)
# VanishingCondition(obstruction=..., name='Jacobi condition on {·,·}_Q')
```

`VanishingCondition.holds(registry)` obstruction'ı `simplify`'dan
geçirir ve `Integer(0)`'a kapanıp kapanmadığına bakar. Lie base için
`[Q, Q] = Q*Q − Q*Q → 0` anında düşer:

```python
d.jacobi_condition(reg).holds(reg)   # True
```

## `prove_jacobi` — DerivedBracketStrategy dispatch'i

`gradalg.proof.verifier.prove_jacobi` bracket'in tipine göre yol
seçer. Bir `DerivedBracket` için otomatik olarak
`DerivedBracketStrategy`'ye gider — üç adımlık zincir:

1. `DerivedBracketTheorem` (theorem step): triple cyclic Jacobi
   toplamı evrensel obstruction `[Q, Q]_base`'e indirgenir.
2. `base-bracket-expand`: base bracket'in kendi rule'u `[Q, Q]`
   düğümünü açar.
3. `simplify`: canonical formda sıfıra iner.

```python
from gradalg.proof.verifier import prove_jacobi

a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
for s in (a, b, c):
    reg.declare(s, Graded(degree=0))

chain = prove_jacobi(d, a, b, c, registry=reg)
len(chain)                    # 3
[s.rule for s in chain.steps]
# ['DerivedBracketTheorem', 'base-bracket-expand', 'simplify']
chain.steps[-1].after         # 0
```

Obstruction sıfıra indirgenemezse (örneğin base bracket'in `[Q, Q]`'yi
çözecek yeniden-yazımı yoksa), `DerivedBracketStrategy` residual'ı
`ProofFailure` içinde döndürür — teoremin koşulu sağlanmıyor demektir.

## `acting_on` — Koszul eşdeğerliği

Poisson manifold'unda klasik Koszul bracket 1-form'lar üstünde yaşar
ve üç-terim formülü verir. Aynı yapı, Schouten-Nijenhuis base üstüne
oturan bir derived bracket'in — ancak `acting_on=π^♯` (veya genel
bir anchor `ρ`) ile ele alınmasıyla — bire bir çıkar.

`acting_on` kwarg'ı verildiğinde `expand` otomatik olarak Koszul
formunu emit eder:

```python
from gradalg.brackets.derived import DerivedBracket
from gradalg.brackets.koszul import KoszulBracket
from gradalg.brackets.schouten import sn
from gradalg.calculus.anchor import Anchor

reg = PropertyRegistry()
pi = Symbol("π")
reg.declare(pi, Graded(degree=1))
alpha, beta = Symbol("α"), Symbol("β")
for s in (alpha, beta):
    reg.declare(s, Graded(degree=1))

rho = Anchor("ρ")
koszul_derived = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho)

koszul_derived.expand(alpha, beta)
# (L_ρ(α)(β) + (-L_ρ(β)(α)) + (-d(⟨ρ(α), β⟩)))

KoszulBracket(rho).expand(alpha, beta)
# (L_ρ(α)(β) + (-L_ρ(β)(α)) + (-d(⟨ρ(α), β⟩)))
```

Sonuçlar *structurally* eşit. Bu, "klasik Koszul bracket, Sharp (π^♯)
anchor'ı üstünden SN base ile `(π, π)`-derived bracket'tir" teoreminin
sembolik seviyede doğrulanmasıdır.

`acting_on=None` (default) verilirse kanonik `{a,b}_Q = [[a,Q],b]`
yolu korunur — hiçbir sessiz yeniden-yazım olmaz. Anchor aynı zamanda
kimlik anahtarına katılır: `DerivedBracket(sn, π, degree_Q=1,
acting_on=Anchor("ρ1"))` ile `acting_on=Anchor("ρ2")` iki farklı
bracket'tir.

## Poisson-as-derived

`DerivedBracket(sn, π, degree_Q=1)` matematiksel olarak Poisson
bracket'in derived inşasıdır. Generic dispatcher üstünden
`prove_jacobi` çağrılınca obstruction `[·,·]_SN(π, π)` şekline iner ve
`ProofFailure` olarak yüzeyleşir — bu dürüst matematiksel teşhistir:
Poisson hipotezi `[π, π]_SN = 0` generic bir simplify kuralı değildir,
explicit varsayım olarak taşınmalıdır. Üretim kullanımı için
`gradalg.library.poisson.PoissonBracket` wrapper'ı tercih edilir —
seeded theorem `poisson_jacobi`'yi tek adımlık citation olarak
verir:

```python
from gradalg.library import theorem_book
from gradalg.library.declarations import Bivector, Functions
from gradalg.library.poisson import PoissonBracket

reg = PropertyRegistry()
pi = Bivector("π", registry=reg)
f, g, h = Functions("f g h", degree=-1, registry=reg)

poisson = PoissonBracket.from_bivector(pi)
chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
len(chain)                    # 1
chain.steps[0].rule           # 'DerivedBracketTheorem'
chain.steps[0].after          # [·,·]_SN(π, π)

theorem_book.get("poisson_jacobi").from_axioms
# ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')
```

[03 — Poisson geometri](03_poisson_geometry.md) aynı yolun üç-görüş
sunumunu koyar; burada sadece derived yapının Poisson'un arkasında
yattığını işaretliyoruz.

## H-twist — Courant bracket'in koşullu Jacobi'si

Aynı indirgeme, bir başka bracket ailesi için farklı bir denkleme
iner. H-twisted Courant bracket `[(X,α),(Y,β)]_C = ([X,Y], L_X β −
L_Y α − ½ d(ι_X β − ι_Y α) + ι_Y ι_X H)` — graded Jacobi ancak ve
ancak `dH = 0`'sa tutar. `CourantBracket(background_H=H)`
`jacobi_condition` üstünden bu koşulu doğrudan verir:

```python
from gradalg.brackets.courant import CourantBracket

reg = PropertyRegistry()
H = Symbol("H")
reg.declare(H, Graded(degree=3))

C = CourantBracket(background_H=H)
C.is_twisted                  # True
cond = C.jacobi_condition(reg)
cond.name                     # 'Courant Jacobi condition (H-twisted by H)'
cond.obstruction              # d(H)

CourantBracket().jacobi_condition(reg).name
# 'Courant Jacobi (untwisted, vacuous)'
```

`CourantBracket`'in kendisi bir `DerivedBracket` alt-sınıfı değildir —
section-pair (TM ⊕ T*M) operandları ve yarı-yarıya expansion Dorfman
köprüsü gibi gömme detayları ayrı tutulmuş. Ama koşullu Jacobi'nin
sunumu derived bracket mekaniğinin aynısı: tek bir denklem
(`dH = 0`), tek bir `VanishingCondition`.

## Özet

Derived bracket'in tek ifadelik koşul indirmesi, paketi tek bir
teorem etrafında örer:

| bracket | base | Q | koşul |
|---------|------|---|-------|
| Poisson (`{f,g}_π`) | `sn` | `π` | `[π, π]_SN = 0` |
| Koszul classical | `sn` + `acting_on=π^♯` | `π` | (eş) |
| Courant (H-twisted) | — | — | `dH = 0` |
| genel | herhangi | seçilen | `[Q, Q]_base = 0` |

## Sonraki adım

Yukarıdaki tabloyu gerçekten tek bir matematiksel manzara haline
getiren kapanış: `DerivedBracket` + `CartanCalculus` + `TheoremBook`
+ `ProofChain`'in birlikte nasıl kullanıldığını gösteren birleşik
tur — [08 — Unified picture](08_unified_picture.md) (Stage D).
