# 09 — Temeller

Bu son tutorial bir soruya iniyor: "`d² = 0` *neden* bir aksiyom?" Yanıt
paketin pedagojik omurgasını açıklar — property provenance, efficient vs
foundational mode, axiom vs theorem sınıflandırması, özel aksiyom setleri
ile çalışma. Daha önceki tutorial'lar bracket ve teoremleri verdi; burada
o teoremlerin altında hangi zeminin olduğunu ve paketin bu zemini nasıl
açıkça tuttuğunu görüyoruz.

[08 — Birleşik tablo](08_unified_picture.md) teoremleri birleştirdi;
burada teoremlerin *ardındaki* aksiyom katmanına iniyoruz.

## İki hipotez seviyesi

Paket iki katman iddia taşır:

- **Axiom.** Primitive kabul edilen eşitlik. Örnek: `d(df) = 0` 0-form'lar
  üstünde jenerik aksiyom — Ω*(M) generator'larına empoze edilmiş.
- **Theorem.** Aksiyomlardan çıkan sonuç. Örnek: `d² = 0` operatör
  özdeşliği — genel derece formlar için `d(df) = 0` + Leibniz ile
  agreement-on-generators argümanıyla türer.

`ExpansionEngine` bu ayrımı `Definition.is_theorem` üstünden taşır. Her
kural ya aksiyom (`is_theorem=False`) ya da theorem (`is_theorem=True`);
theorem olanlar `theorem_proof_builder()`'ından bir sub-proof çıkarabilir.

```python
from gradalg.proof.expansion import default_engine

eng = default_engine()
for d in eng.definitions:
    label = "theorem" if d.is_theorem else "axiom"
    print(f"{label:<8} | {d.name}")
# axiom    | L_X := d∘ι_X + ι_X∘d (Cartan definition)
# axiom    | Act linearity: (A + B)(x) = A(x) + B(x)
# axiom    | d² = 0
# axiom    | ι_X ∘ ι_X = 0
# axiom    | ι_X(f) = 0 on 0-forms
# axiom    | ι_X(df) = X(f)
```

Default engine konservatiftir: her şeyi axiom sayar. Hiçbir kural daha
derinden türetilmez. Bu "efficient" mode'un davranışı — kısa, hızlı,
ispat-altyapısı yok.

## `d_squared_mode="theorem"` — d² = 0'ı türet

`d² = 0`'ı *theorem* olarak işaretlemek için engine'i özel konfigüre
edin:

```python
eng_th = default_engine(d_squared_mode="theorem")
[d for d in eng_th.definitions if d.name == "d² = 0"][0].is_theorem
# True
```

Bu bayrak tek başına ispat katmanını değiştirmez; sadece "bu kural
aksiyom değil, daha derin bir türevdir" der. Türemeyi *görmek* için
foundational mode gerekli.

## Efficient vs foundational mode

`mode="efficient"` (default): hızlı tag-only ispatlar. Her kural tek
adımda atar, arkasında sub-proof yok.

`mode="foundational"`: theorem-sınıfı kurallar tetiklendiğinde
`ProofStep.children` alanına sub-proof iliştirir — kuralın hangi daha
primitive aksiyom(lar)a dayandığını gösterir.

```python
from gradalg.calculus.invariant_d import default_d
from gradalg.core.registry import PropertyRegistry
from gradalg.core.expr import Symbol, Integer
from gradalg.core.properties import Graded
from gradalg.proof.verifier import prove_equivalence

reg = PropertyRegistry()
omega = Symbol("ω")
reg.declare(omega, Graded(degree=2))
expr = default_d(default_d(omega))     # d(d(ω))

eng_eff = default_engine(registry=reg, mode="efficient",
                         d_squared_mode="theorem")
eng_fnd = default_engine(registry=reg, mode="foundational",
                         d_squared_mode="theorem")

eff = prove_equivalence(expr, Integer(0), registry=reg, engine=eng_eff)
fnd = prove_equivalence(expr, Integer(0), registry=reg, engine=eng_fnd)

[(s.rule, len(s.children)) for s in eff.steps]
# [('d² = 0', 0), ('simplify', 0)]

[(s.rule, len(s.children)) for s in fnd.steps]
# [('d² = 0', 1), ('simplify', 0)]

fnd.steps[0].children[0].rule
# 'd(df) = 0 on 0-forms (generator axiom)'
fnd.steps[0].children[0].justification
# 'operator identity d ∘ d = 0 extends from the generator-level axiom
#  d(df) = 0 by agreement on the generators of Ω*(M) ...'
```

Aynı `d² = 0` adımı iki mode'da aynı `after` değerine (`0`) iner — ama
foundational mode "peki nereden?" sorusuna cevap taşıyor. Generator-level
axiom (`d(df) = 0`) mevcut argümanın *tek* primitive girdisidir; geri
kalan her şey "generator üstünde eşitse Ω*(M)'nin tamamında eşittir"
prensibiyle extend edilmiş.

## Custom aksiyom setleri

`default_engine` bir kolaylık — asıl veri `ExpansionEngine.definitions`
listesi. Bu listeyi kendiniz inşa ederek paketin aksiyomatik tabanını
değiştirebilirsiniz:

- `d² = 0` yerine sadece generator-level `d(df) = 0`'ı tutup
  `DSquaredZeroDefinition`'ı tamamen dışarı bırakmak.
- `LieDerivativeCartanDefinition`'ı *aksiyom* yerine *tanım* olarak
  kullanıp Cartan magic formula üstünden türetilmiş varsaymak.
- Kendi cebirsel teorinizin aksiyomlarını `Definition` alt-sınıfı
  yazarak enjekte etmek.

`Definition` API minimal — `matches(expr)`, `rewrite(expr)`, ve opsiyonel
`theorem_proof_builder()` (foundational mode'da sub-proof veren):

```python
from gradalg.proof.expansion import Definition, DSquaredZeroDefinition

class MyAxiom(Definition):
    name = "my rule"

    def matches(self, expr): ...
    def rewrite(self, expr): ...
    # theorem_proof_builder() bırakıyor → axiom olarak kalır
```

`theorem_proof_builder` `None` döndürürse kural aksiyom — foundational
mode'da bile çocuğu yok. `ProofChain` döndürürse kural theorem — mode
foundational olduğunda bu chain step'in altına iliştirilir.

## Property provenance tekrar

[02 — Property provenance](02_property_provenance.md) property'lerin
`axiom` ve `theorem` olarak etiketlenmesini göstermişti. Aynı ayrım
burada `Definition.classification` üstünden karşımıza çıkıyor — aslında
paketin tek omurgası: "bir iddia primitive mi, yoksa başka
primitive'lerden mi türüyor?" sorusu hem sembol seviyesinde
(property'ler) hem operatör seviyesinde (expansion kuralları)
takibe alınır.

Birleştirme:

| katman | taşıyıcı | primitive | türeyen |
|--------|----------|-----------|---------|
| sembol | `Property.provenance` | `"axiom"` | `"theorem"` |
| expansion | `Definition.is_theorem` | `False` | `True` |
| teorem | `Theorem.from_axioms` | atomic string | — |

Üç katmanda da aynı felsefe: **her iddianın kaynağı takip edilir**.
Kullanıcı "bu sonuç hangi aksiyoma dayanıyor?" diye sorduğunda paket
mekanik yolla cevap verebilir.

## Pedagojik kapanış

Bütün paket tek bir mimari kararın etrafında örülü: *provenance'ı
asla kaybetme*. Bu karar:

1. **Property-level.** Her `Property` (graded-antisymmetry, Leibniz,
   Jacobi, …) hangi aksiyomdan türediğini taşır.
2. **Expansion-level.** Her `Definition` axiom mu theorem mi deklare
   eder; foundational mode kişisel ispat altında ne var sorusuna
   mekanik cevap verir.
3. **Theorem-level.** Her `Theorem.from_axioms` hangi atomic
   varsayımlara dayandığını beyan eder; her `theorem_book.get(name)`
   kullanımı citation zinciri üretir.
4. **Bracket-level.** Derived Bracket Teoremi gibi yapı teoremleri
   "tek hipotez, çok sonuç" çerçevesinde aynı obstruction'ı paylaşır —
   paket bunu otomatik tespit eder.

Paket *nereden biliyorsun?* sorusunu sembol'den teorem'e kadar her
seviyede cevaplanabilir tutmak için inşa edildi. Her `ProofChain` bir
argüman ağacı — kökleri axiom, yaprakları `Integer(0)`. Pedagojik
değer buradan geliyor: kullanıcı bir teoremi kara-kutu olarak değil,
altındaki yapı ile birlikte kavrıyor.

## Tutorial serisinin sonu

Dokuz bölüm:

1. [01 — Expression agacı](01_expressions.md)
2. [02 — Property provenance](02_property_provenance.md)
3. [03 — Poisson geometri](03_poisson_geometry.md)
4. [04 — Lie algebroid](04_lie_algebroid.md)
5. [05 — Cartan calculus](05_cartan_calculus.md)
6. [06 — Custom bracket](06_custom_bracket.md)
7. [07 — Derived bracket](07_derived_bracket.md)
8. [08 — Birleşik tablo](08_unified_picture.md)
9. **09 — Temeller** ← buradasınız

Sembolden teoreme, aksiyomdan birleşik tabloya — her adımda paketin
gerçekten sunabildiğini canlı API örnekleriyle gördük. Buradan sonra
paket bir araç: kendi bracket'inizi, kendi teoreminizi, kendi aksiyom
setinizi ekleyip üstüne çalıştırabilirsiniz.
