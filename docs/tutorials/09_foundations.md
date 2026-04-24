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
`theorem_proof_builder()` (foundational mode'da sub-proof veren).

### Axiom sınıfı — en kısa yol

Aşağıda `c_zero` sembolünü sıfıra indiren bir kural. Kendi engine'ini
`ExpansionEngine([...])` ile kuruyorsun; default kuralları devre dışı
bırakıp sadece bu kuralı çalıştırıyorsun:

```python
from gradalg.proof.expansion import Definition, ExpansionEngine
from gradalg.core.expr import Symbol, Integer, Sum

class ZeroConstAxiom(Definition):
    name = "c_zero := 0 (axiom)"

    def matches(self, expr):
        return isinstance(expr, Symbol) and expr.name == "c_zero"

    def rewrite(self, expr):
        return Integer(0)

engine = ExpansionEngine([ZeroConstAxiom()])
expanded, steps = engine.expand(Sum(Symbol("c_zero"), Symbol("x")))
# expanded:  (0 + x)
# steps:     [ProofStep(rule='c_zero := 0 (axiom)', provenance_tag='axiom')]
```

`theorem_proof_builder` override etmediği için `is_theorem=False` —
foundational mode'da bile çocuk adım yok.

### Theorem sınıfı — sub-proof iliştir

Aynı kuralı theorem olarak sunmak için `theorem_proof_builder` bir
`ProofChain` builder'ı döndürür:

```python
from gradalg.proof.chain import ProofChain
from gradalg.proof.step import ProofStep

class ZeroConstTheorem(Definition):
    name = "c_zero := 0 (theorem)"

    def matches(self, expr):
        return isinstance(expr, Symbol) and expr.name == "c_zero"

    def rewrite(self, expr):
        return Integer(0)

    def theorem_proof_builder(self):
        def build(matched):
            step = ProofStep(
                rule="c_zero = c_zero − c_zero (axiom)",
                before=matched,
                after=Integer(0),
                justification="self-annihilation axiom on c_zero",
                provenance_tag="axiom",
            )
            return ProofChain(steps=[step])
        return build

eff = ExpansionEngine([ZeroConstTheorem()], mode="efficient")
fnd = ExpansionEngine([ZeroConstTheorem()], mode="foundational")

c = Symbol("c_zero")
eff_exp, eff_steps = eff.expand(c)
fnd_exp, fnd_steps = fnd.expand(c)

len(eff_steps[0].children)   # 0
len(fnd_steps[0].children)   # 1
fnd_steps[0].children[0].rule
# 'c_zero = c_zero − c_zero (axiom)'
```

Aynı `Definition`; efficient mode'da atomik olarak atıyor, foundational
mode'da altına tek adımlık sub-proof iliştiriyor. Paket'in kendi
`DSquaredZeroDefinition`'ı da aynen böyle çalışıyor — sadece sub-proof
builder'ı `d(df) = 0` generator axiom'una atıf yapıyor.

## Theorem Book yapısı

Expansion kuralları operatör-seviyesi provenance taşır; teorem-seviyesi
provenance ise [`gradalg.library.theorem_book`](../../gradalg/library/theorem_book.py)
altında. Veri yapısı:

```python
from gradalg.library.theorem_book import Theorem
import dataclasses

[f.name for f in dataclasses.fields(Theorem)]
# ['name', 'statement', 'from_axioms', 'proof', 'notes']
```

Beş alan:

- `name` — registry key (ör. `"poisson_jacobi"`).
- `statement` — insan-okunur teorem iddiası.
- `from_axioms` — atomic dayandığı aksiyomların `Tuple[str, ...]`'u.
- `proof` — teoremin kanonik `ProofChain`'i.
- `notes` — ek bağlam (opsiyonel).

Singleton registry `theorem_book`'a sorgu atmak:

```python
from gradalg.library import theorem_book

theorem_book.names()
# ('poisson_jacobi', 'poisson_koszul_equivalence',
#  'poisson_koszul_jacobi', 'lie_algebroid_anchor_compat',
#  'courant_jacobi_twist', 'courant_dorfman_bridge',
#  'dirac_isotropy', 'dirac_involutivity')

thm = theorem_book.get("poisson_jacobi")
thm.statement
# '{f, g, h}_π cyclic sum = 0 when [π, π]_SN = 0'
thm.from_axioms
# ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')
thm.proof.steps[0].rule
# 'DerivedBracketTheorem'
```

Seeded teoremler paket initialization sırasında register edilir
(`gradalg/library/__init__.py` submodule yüklemeleri). Downstream kod
bir teoremi *yeniden ispatlamaz*; `theorem_book.get(name).proof`
chain'ini alıp daha büyük bir `ProofChain`'in içine gömer. Bu paketin
"tek citation, çok kullanım" stratejisinin omurgası — her yeni library
modülü kendi teoremlerini register eder, Theorem Book büyür.

## Property provenance tekrar

[02 — Property provenance](02_property_provenance.md) property'lerin
`axiom` ve `theorem` olarak etiketlenmesini göstermişti. Aynı ayrım
burada `Definition.is_theorem` üstünden karşımıza çıkıyor — aslında
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
