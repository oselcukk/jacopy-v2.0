# gradalg — Graded Algebra ve Bracket Hesabı için Sembolik Paket

## Proje Özeti

`gradalg`, graded cebir, bracket yapıları ve Cartan calculus alanında
sembolik hesap ve **adım adım ispat** üretmek için tasarlanmış bir Python
paketidir. Hedef kitlesi: diferansiyel geometri, Poisson geometrisi,
Lie/Courant algebroidleri ve ilgili konularda çalışan matematikçiler ve
matematiksel fizikçilerdir.

Paket soyut (koordinatsız) seviyede çalışır. Bir sembolik ifadeyi tanım ve
aksiyomlara göre açar, tanıdık cebirsel yapıları (commutator, cyclic sum,
Leibniz) tanır, sadeleştirir ve her adımı okunabilir biçimde (terminal,
LaTeX, Jupyter) sunar. Her matematiksel "gerçek"in aksiyom mı yoksa daha
temel aksiyomlardan türetilmiş teorem mi olduğunu takip eder; kullanıcının
talebine göre bir property'yi kısa kullanır veya en temel aksiyomlara
kadar açar.

## Tasarım Felsefesi

Paketin üzerine kurulu olduğu temel ilkeler:

- **Koordinatsız / soyut**: Manifold koordinatlarına inmeyiz. Hesaplar
  bracket tanımları, derivasyon aksiyomları ve defining relation'lar
  üzerinden yürür.

- **Cadabra-tarzı mimari**: Expression tree + global property registry +
  modüler algoritma sınıfları. Kod Cadabra'nın değil, tasarımı bizim.

- **Adım adım ispat**: Sonuç değil süreç önemli. Her rewrite adımı
  kaydedilir ve istenen formatta görüntülenir.

- **Derived bracket birleştirici mimari**: Poisson, Koszul, Courant ve pek
  çok klasik bracket aslında aynı derived bracket inşasının farklı
  elemanlara uygulanmış halidir. Paket bu birleşik bakışı birinci sınıf
  vatandaş olarak tutar: teorem bir kez kanıtlanır, her yerde kullanılır.

- **Property provenance**: Her property'nin statüsü açık tutulur —
  *axiom* (ilkel, tanımın parçası) ya da *theorem* (daha temel
  aksiyomlardan türetilmiş). Theorem olanların ispatı sistemde saklanır.

- **Unroll modu**: Kullanıcı herhangi bir theorem'i ispatı görülsün diye
  en temel aksiyomlara kadar açtırabilir. "Hızlı mod" property'leri kısa
  yoldan kullanır; "foundational mod" her şeyi dipten türetir.

- **Operatör-seviye ve eleman-seviye ispat**: Cartan relations gibi
  operatör özdeşlikleri ya operatörler arası denklem olarak
  (`AgreementOnGenerators` meta-teoremi ile) ya da belirli p-formlar
  üzerinde (brute force açılım) ispatlanabilir. Kullanıcı seçer.

- **Genişletilebilir**: Kullanıcı kendi bracket'ını, kendi derivasyonunu,
  kendi aksiyomlarını ve kendi calculus'ünü tanımlayabilmeli.

- **Saf Python**: Bağımlılığımız minimum. Opsiyonel zengin çıktı için
  `rich`, Jupyter için yerleşik destek.

## Aksiyom Hiyerarşisi ve Unroll Modu

Paketin bilgi sistemi iki katmanlıdır: **aksiyomlar** (türetilemeyen
ilkeller) ve **teoremler** (daha aşağıdan türetilmiş, ispatı sistemde
kayıtlı).

### Hiyerarşi (en dipten başlayarak)

**Seviye 0 — Cebirsel temel**
- Vektör uzayı (toplama, skaler çarpım, 0 ve 1)
- Associative cebir (wedge ürünü birleşmeli)
- Wedge'in graded commutative olması: `α ∧ β = (-1)^{|α||β|} β ∧ α`
- Graded structure: her elemanın bir derecesi var

**Seviye 1 — Derivasyon aksiyomları**
- Derivasyonun tanımı: R-lineer + graded Leibniz rule
- Graded commutator tanımı: `[A,B] = AB − (−1)^{|A||B|} BA`
- İki graded derivation'ın graded commutator'u yine graded derivation
  (dereceleri toplanır)

**Seviye 2 — Temel operatörlerin tanımları**
- `d` (exterior derivative): deg +1 graded anti-derivation, fonksiyonlarda
  `d(f) = df` ile tanımlı, invariant formülle 1-formlara uzatılır
- `ι_X` (interior product): deg -1 graded anti-derivation, `ι_X(df) = X(f)`
- `L_X` (Lie derivative): iki seçenek — (a) flow ile tanımlı, Cartan'ın
  büyülü formülü teorem olarak çıkar, (b) `L_X := d ∘ ι_X + ι_X ∘ d` ile
  tanımlı, Cartan formülü tautoloji olur

**Seviye 3 — Bracket aksiyomları**
- Lie bracket tanımı (vektör alanları üzerinde): `[X,Y](f) = X(Y(f)) − Y(X(f))`
- Jacobi identity for Lie bracket: türetilebilir veya axiom olarak kabul
  edilir (konvansiyon seçimi)
- Schouten-Nijenhuis bracket: Lie bracket + wedge Leibniz ile genişletme

**Türetilebilir theorem'ler (sistemde hazır ispatlarıyla saklanır):**
- `d² = 0` (Jacobi identity for Lie bracket'tan türer — bkz. Tutorial 9)
- `ι_X ∘ ι_X = 0` (form antisymmetry'den)
- `[ι_X, ι_Y] = 0` (anti-commuting anti-derivations)
- Cartan's magic formula `[d, ι_X] = L_X` (tanım seçimine göre)
- `[L_X, L_Y] = L_{[X,Y]}` (Lie bracket Jacobi + Cartan'dan)
- `[L_X, ι_Y] = ι_{[X,Y]}`
- Derived Bracket Theorem: `{a,b}_Q = [[a,Q],b]` Jacobi ⟺ `[Q,Q] = 0`

### Provenance sistemi

Her property bir `status` alanı taşır:

```python
class PropertySpec:
    name: str
    statement: Expr              # denklem veya koşul
    status: Literal["axiom", "theorem", "conjecture"]
    dependencies: list["PropertySpec"]   # theorem ise bağlı olduğu aksiyomlar
    proof: Optional[ProofChain]          # theorem ise ispat zinciri
```

### İspat modları

Kullanıcı ispat istediğinde paket üç moddan birini seçer:

1. **`mode="efficient"`** (default)
   - Property'ler axiom gibi kullanılır, kısa ispat
   - Hızlı, günlük araştırma için

2. **`mode="foundational"`**
   - Her theorem kendi ispatına açılır, o da bağımlılıklarına açılır
   - En dipte sadece Seviye 0–2 aksiyomları kalır
   - Uzun ama tam şeffaf

3. **`mode="custom"` ile `axioms={...}`**
   - Kullanıcı hangi property'leri axiom kabul edeceğini söyler
   - Diğerleri türetilir
   - Standart dışı calculus'ler için ideal (örn. twisted, Lie algebroid)

### `library/theorem_book.py`

Türetilmiş teoremler burada merkezi olarak saklanır. Her teorem:
- İfadesi
- Hangi aksiyomlara dayandığı
- İspatının `ProofChain` formatı
- Unroll sırasında yerine yerleştirilir

Örnek kayıtlar:
- `THEOREM_d_squared_zero`
- `THEOREM_cartan_magic_formula`
- `THEOREM_LX_LY_commutator`
- `THEOREM_derived_bracket_jacobi`
- `THEOREM_poisson_bracket_equals_Xf_g`

## Birleştirici Mimari: Derived Bracket Teoremi

Paketin matematiksel omurgasında **Kosmann-Schwarzbach'ın derived bracket
teoremi** var. Graded Lie cebri `(g, [·,·])` ve `Q ∈ g` verildiğinde:

```
{a, b}_Q  :=  [[a, Q], b]
```

**Teorem:** `{·,·}_Q` Leibniz aksiyomunu sağlar, ve
**Jacobi identity'yi sağlar ⟺ [Q, Q] = 0**.

Bu tek teorem sayesinde aşağıdaki klasik bracket'ların hepsi aynı
çerçevede ele alınır:

| Bracket | Base cebir `g` | Generator `Q` | Etki ettiği | Jacobi koşulu |
|---|---|---|---|---|
| Poisson `{·,·}_π` | Polyvector + SN | `π` (bivektör, deg 2) | Fonksiyonlar | `[π,π]_SN = 0` |
| Koszul `[·,·]_K` | Polyvector + SN | `π` (bivektör, deg 2) | 1-formlar | `[π,π]_SN = 0` |
| Extended Koszul | Polyvector + SN | `π` (bivektör, deg 2) | k-formlar | `[π,π]_SN = 0` |
| Courant `[·,·]_C` | Graded Leibniz | `Θ` (Courant tensor) | `TM ⊕ T*M` | `[Θ,Θ] = 0` |
| Twisted Courant | Graded Leibniz | `Θ + H` (3-form ile) | `TM ⊕ T*M` | `dH = 0` |
| Lie (Kirillov) | Uygun graded | Anchor-compatibl Q | Cebir | Anchor + Jacobi |

### Mimari sonuç

- `DerivedBracket` sınıfı `brackets/derived.py`'da bir kez yazılır
- `DerivedBracketTheorem` verifier `proof/verifier.py`'da bir kez kanıtlanır
- Poisson, Koszul, Courant bracket'ları bu sınıftan türetilir
- Jacobi ispatı istendiğinde paket şunu üretir:

  > "Bu bir derived bracket'tir (generator: Q). Derived Bracket Theorem
  > uyarınca Jacobi ⟺ `[Q, Q] = 0`. Dolayısıyla Jacobi ⟺ (somut koşul)."

- Aynı generator `Q`'nun farklı derecelerdeki elemanlara etkisi, aynı
  `[Q,Q] = 0` koşuluyla **bütün seviyelerde aynı anda** Jacobi verir
- Kullanıcı "brute force" mod da isteyebilir: `ExpansionEngine` ifadeyi
  klasik tanımdan açar, uzun ama öğretici çıktı verir

## Yetkinlikler (Nihai Hedef)

Bitirdiğimizde paket şunları yapabilmeli:

- Soyut graded cebir üzerinde sembolik hesap
- Herhangi bir bracket için Jacobi, antisymmetry, Leibniz kontrolü — adım
  adım ispatla
- **Derived bracket teoremi** üzerinden Poisson, Koszul, Courant gibi
  bracket'ların Jacobi'sinin tek bir koşula (`[Q,Q] = 0`) indirgenmesi ve
  bunun ispat olarak sunulması
- Aynı generator'ın (örn. bir Poisson bivektör π) farklı derecelerdeki
  elemanlara (fonksiyonlar, 1-formlar, k-formlar) etki edişinin
  gösterilmesi — `{f,g}_π`, `[α,β]_K`, extended Koszul hepsinin aynı
  `[π,π]_SN = 0` koşulundan çıktığını ispatlama
- Cartan calculus ilişkilerinin (`[L_X, L_Y] = L_{[X,Y]}`,
  `[d, ι_X] = L_X`, vb.) farklı calculus'lerde doğrulanması —
  **hem operatör-seviye** (`AgreementOnGenerators` ile p-forms için
  otomatik) **hem eleman-seviye** (sembolik `Form(degree=p)` üzerinde
  açılım)
- `d² = 0` gibi "bedava bilinen" property'lerin aslında hangi aksiyomlardan
  türediğinin ispatla gösterilmesi (Lie bracket Jacobi → `d² = 0`)
- Hamiltonian vektör alanları, Poisson bracket, symplectic form hesabı
  (örn. `{f,g} = ω(X_f, X_g) = X_f(g)` zincirinin ispatı)
- `{f,g} = X_f(g)` identity'sinin derived bracket tanımından türetilmesi:
  `X_f = -[f, π]_SN` üzerinden
- Lie algebroid, Courant algebroid gibi yapıların aksiyomlarının test
  edilmesi
- Schouten-Nijenhuis, Koszul, Courant, Dorfman bracket'ları ile çalışma
- Bir bracket için iki alternatif tanımın eşdeğerliğinin adım adım ispatı
  (örn. Koszul bracket'ın klasik formülü
  `[α,β]_K = L_{ρα}β - L_{ρβ}α - d⟨ρα,β⟩` ile derived bracket
  tanımı `[[α,π]_SN, β]_SN` arasındaki eşitliğin gösterilmesi)
- Kullanıcının "bu aksiyomları kabul et, gerisi türesin" şeklinde özel
  aksiyom seti tanımlayıp tüm Cartan relations'ı o set üzerinden
  ispatlatabilmesi
- "Brute force" modu: kullanıcı isterse paket tanımları açarak uzun adım
  adım ispat üretir (öğretici); isterse derived bracket teoremiyle kısa,
  yapısal ispat verir
- LaTeX çıktı (makale kalitesinde), terminal güzel çıktı, Jupyter yerleşik
  entegrasyonu
- Kullanıcının özel yapı tanımlama akışı: yeni bracket, yeni derivasyon,
  yeni property, yeni aksiyom seti

## Mimari Katmanlar

```
┌─────────────────────────────────────────────────────┐
│  8. Display (LaTeX / Terminal / Jupyter / ASCII)    │
├─────────────────────────────────────────────────────┤
│  7. Proof System                                    │
│     ProofChain, Tracer, Expansion, Recognizers,     │
│     Strategies (Unroll, AgreementOnGenerators, ...)│
├─────────────────────────────────────────────────────┤
│  6. Theorem Book (library/theorem_book.py)          │
│     Önceden ispatlanmış teoremlerin merkezi kaydı   │
├─────────────────────────────────────────────────────┤
│  5. Calculus (d, ι, L, Anchor, CartanCalculus)      │
├─────────────────────────────────────────────────────┤
│  4. Brackets (Lie, SN, Koszul, Courant, Derived)    │
├─────────────────────────────────────────────────────┤
│  3. Algebra (Derivations, Commutators, Tensor)      │
├─────────────────────────────────────────────────────┤
│  2. Algorithms                                      │
│     Distribute, SortProduct (graded),               │
│     Substitute, ProductRule, CollectTerms, ...      │
├─────────────────────────────────────────────────────┤
│  1. Core (Expression Tree + Property Registry       │
│          with Provenance + Symbolic Degrees)        │
└─────────────────────────────────────────────────────┘
```

Alt katmanlar yukarısını bilmez. Proof system tüm alt katmanları sarar
(`Tracer` her algoritma çağrısını kaydeder). Theorem Book teoremleri
saklar ve Unroll stratejisi onları kullanır. Display bağımsızdır.

## Neden Kendimiz Yazıyoruz

SymPy, SageMath ve Cadabra2 incelendi. Kararımız:

- **SymPy**: Expression tree ve LaTeX için güçlü, ama graded cebir
  altyapısı yok; bunları sıfırdan eklemek gerekiyor.
- **SageMath**: `CombinatorialFreeModule` ve category sistemi çok güçlü,
  ama ağır bağımlılık ve soyut-bracket ispat mantığı için hâlâ iş var.
- **Cadabra2**: Tam olarak bu problem sınıfı için tasarlanmış, ama kendi
  preprocessor syntax'ı, sınırlı programmatik kontrol ve ispat gösterimi
  için özel hook eksikliği var.

Bu nedenle **Cadabra'nın mimari desenini** (Expr tree + PropertyRegistry +
Algorithm sınıfları) taklit eden, saf Python ile yazılmış, graded cebire
odaklı, ispat sistemi birinci sınıf vatandaş olan, property provenance'ı
ve unroll modunu desteklen kendi paketimizi yazıyoruz.

## Paket Dizin Yapısı

```
gradalg/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
│   ├── index.md
│   ├── tutorials/
│   │   ├── 01_first_steps.md
│   │   ├── 02_jacobi_identity.md
│   │   ├── 03_poisson_geometry.md
│   │   ├── 04_lie_algebroid.md
│   │   ├── 05_cartan_calculus.md
│   │   ├── 06_custom_bracket.md
│   │   ├── 07_derived_bracket.md
│   │   ├── 08_unified_picture.md
│   │   └── 09_foundations.md
│   ├── examples/                 (Jupyter notebooks)
│   └── api/                      (otomatik API reference)
│
├── gradalg/
│   ├── __init__.py
│   │
│   ├── core/                     [Faz 1]
│   │   ├── __init__.py
│   │   ├── expr.py               (Expression tree, Sum, Product, ...)
│   │   ├── symbolic_degree.py    (Sembolik tamsayı derece parametreleri)
│   │   ├── properties.py         (Property sınıfları, provenance alanı)
│   │   ├── registry.py           (PropertyRegistry)
│   │   ├── wildcards.py          (Pattern matching primitives)
│   │   └── equality.py           (Structural equality + canonical hash)
│   │
│   ├── algorithms/               [Faz 2-3]
│   │   ├── __init__.py
│   │   ├── base.py               (Algorithm ABC, StepResult)
│   │   ├── distribute.py         (A(B+C) → AB+AC)
│   │   ├── flatten.py            (Associativity)
│   │   ├── sort_product.py       (Graded sign + Koszul convention)
│   │   ├── collect_terms.py      (Like terms)
│   │   ├── substitute.py         (Pattern-based rewrite engine)
│   │   ├── product_rule.py       (Graded Leibniz)
│   │   ├── unwrap.py             (Türevleri parantez dışına çıkar)
│   │   ├── expand_bracket.py     (Bracket'ı tanıma göre aç)
│   │   └── simplify.py           (Top-level pipeline)
│   │
│   ├── algebra/                  [Faz 4]
│   │   ├── __init__.py
│   │   ├── graded_element.py     (Degree-taşıyan elemanlar)
│   │   ├── derivation.py         (Graded derivasyon)
│   │   ├── commutator.py         (Graded commutator [A,B])
│   │   └── tensor.py             (Tensor product)
│   │
│   ├── brackets/                 [Faz 5]
│   │   ├── __init__.py
│   │   ├── base.py               (GradedBracket ABC)
│   │   ├── lie.py                (Lie bracket)
│   │   ├── schouten.py           (Schouten-Nijenhuis, fonksiyonlar dahil)
│   │   ├── koszul.py             (Klasik + derived tanım, eşdeğerlik)
│   │   ├── courant.py            (Courant, twisted dahil)
│   │   ├── dorfman.py            (Dorfman bracket)
│   │   ├── derived.py            (DerivedBracket — paketin kalbi)
│   │   └── custom.py             (Kullanıcı bracket helper'ı)
│   │
│   ├── calculus/                 [Faz 6]
│   │   ├── __init__.py
│   │   ├── exterior_d.py         (d operatörü — iki mod: d²=0 axiom/theorem)
│   │   ├── interior.py           (ι_X — interior product)
│   │   ├── lie_derivative.py     (L_X — iki tanım seçeneği)
│   │   ├── anchor.py             (Anchor ρ: E → TM)
│   │   ├── hamiltonian_vf.py     (X_f via ι_Xω = -df veya [f,π]_SN)
│   │   ├── exterior_algebra.py   (Ω*(M), generator yapısı)
│   │   ├── operator_equation.py  (Operatör-seviye denklem sınıfı)
│   │   └── cartan.py             (CartanCalculus framework)
│   │
│   ├── proof/                    [Faz 7]
│   │   ├── __init__.py
│   │   ├── step.py               (ProofStep, with provenance)
│   │   ├── chain.py              (ProofChain, nested)
│   │   ├── tracer.py             (TracingAlgorithm wrapper)
│   │   ├── expansion.py          (Tanım-tabanlı açılım)
│   │   ├── recognizers.py        (Commutator, cyclic sum, Leibniz,
│   │   │                          Schouten, derived bracket patterns)
│   │   ├── strategies.py         (ExpandAndSimplify,
│   │   │                          AgreementOnGenerators,
│   │   │                          UnrollToFoundations,
│   │   │                          PatternGuided, ...)
│   │   └── verifier.py           (prove_jacobi, check_cartan, ...)
│   │
│   ├── display/                  [Faz 8]
│   │   ├── __init__.py
│   │   ├── latex.py              (LaTeX printer)
│   │   ├── terminal.py           (rich-tabanlı terminal çıktı)
│   │   ├── jupyter.py            (_repr_latex_, _repr_html_)
│   │   └── ascii.py              (Plain text fallback)
│   │
│   └── library/                  [Faz 9]
│       ├── __init__.py
│       ├── theorem_book.py       (Türetilmiş teoremlerin kaydı)
│       ├── symplectic.py         (Symplectic manifold)
│       ├── poisson.py            (Poisson geometri, üç tanım)
│       ├── lie_algebroid.py      (Lie algebroid)
│       ├── courant_algebroid.py  (Courant algebroid)
│       └── dirac.py              (Dirac structure)
│
└── tests/
    ├── test_core/
    ├── test_algorithms/
    ├── test_algebra/
    ├── test_brackets/
    ├── test_calculus/
    ├── test_proof/
    ├── test_library/
    ├── test_display/
    └── integration/              (uçtan uca ispat testleri)
```

## Geliştirme Fazları

### Faz 0 — Altyapı

**Amaç:** Paketin geliştirme ortamını kurmak.

**İşler:**
- `pyproject.toml` (build system, metadata, deps)
- Dizin yapısı + boş `__init__.py`'ler
- `pytest` konfigürasyonu
- Opsiyonel: `pre-commit` (black, ruff, mypy)
- Opsiyonel: `mkdocs` veya `sphinx` docs skeleton
- `README.md` taslağı
- `.gitignore`

**Çıktı:** `pip install -e .` ile kurulabilen, `pytest` çalıştırılabilen
boş paket.

---

### Faz 1 — Core (Expression Tree + Properties + Provenance)

**Amaç:** Paketin kalbini atmak. Her şey bunun üzerine oturuyor.

#### `core/expr.py` — Expression Tree

- `Expr` base class: `head`, `children`, `indices`, `metadata`
- Operatör overloading: `+`, `-`, `*`, `**`, `-` (unary)
- Concrete node tipleri:
  - `Symbol(name)`
  - `Sum(*args)`
  - `Product(*args)` (non-commutative)
  - `Power(base, exponent)`
  - `Rational(p, q)`, `Integer(n)`
  - `Zero`, `One` (singleton)
  - `Neg(expr)` — rahat işaret yönetimi için
- Traversal: `walk()`, `find(predicate)`, `replace_at(path, new)`
- `clone()` — deep copy
- Temel `__repr__`

#### `core/symbolic_degree.py` — Sembolik Dereceler

Cartan relations'ı generic p-formlar üzerinde ispatlayabilmek için
dereceler **sembolik tamsayı** olabilmeli:

- `SymbolicDegree(name)` — sembolik tamsayı parametre (`p`, `q`, vb.)
- `DegreeExpr` — dereceler arası aritmetik (`p + 1`, `p + q`)
- Cebirsel ilişkiler: `|dω| = |ω| + 1`, `|α∧β| = |α| + |β|`
- Paritelerle çalışma: `(-1)^p`, `(-1)^{p+q}`
- Sembolik derecelerle çalışırken (-1)^... ifadelerinin doğru takibi

#### `core/properties.py` — Property sınıfları + Provenance

Her property'nin statüsü vardır:

```python
class Property:
    name: str
    statement: Expr                        # ne söylüyor
    status: Literal["axiom", "theorem"]
    dependencies: list["Property"]         # theorem ise bağlı olduğu aksiyomlar
    proof: Optional[ProofChain]            # theorem ise ispat
    provenance_note: str                   # "bu bir ilkel" veya "Tutorial 9'dan türer"
```

Property sınıfları:
- `Graded(degree)` — degree sembolik veya concrete
- `NonCommuting`, `AntiCommuting`, `GradedCommutative`
- `Symmetric`, `Antisymmetric`
- `Derivation(degree=0)`
- `Closed` (d²=0) — **hem axiom hem theorem olabilen özel durum**
- `NonDegenerate`
- `Vector`, `Form(degree)`, `Function`, `MultiVector(degree)`
- Her property kendi statüsünü kontrol edebilir: `is_axiom()`, `is_theorem()`

#### `core/registry.py` — PropertyRegistry

- Pattern → property listesi map'i
- `attach(pattern, prop)` — tek obje veya liste
- `get(node, prop_type)` — sorgu
- Pattern tipleri: exact name, type, parent-class
- Scope yönetimi: `with registry.scope():`
- **Axiom seti yönetimi**: `with registry.axioms({...}):` — geçici olarak
  bazı property'leri axiom muamelesi yaptır
- Global default registry + özel registry oluşturma

#### `core/wildcards.py` — Pattern primitives

- `Wildcard(name, constraint=None)`
- Tipler: generic, typed (Vector/Form), indexed
- `match(pattern, tree)` → `bindings | None`

#### `core/equality.py`

- Structural equality
- Up-to-canonical equality (sort + sign sonrası)
- `canonical_hash(expr)`
- `structurally_equal`, `semantically_equal`

**Testler:**
- Operatör overloading doğru ağaç üretimi
- `clone()` ve equality tutarlılığı
- `walk()` tüm düğümleri ziyaret
- Property atama + provenance round-trip
- Axiom ↔ theorem status geçişi
- Wildcard eşleşme örnekleri
- Sembolik derece aritmetiği

---

### Faz 2 — Basic Algorithms

**Amaç:** Cadabra'nın temel algoritmalarının Python karşılıkları.

#### `algorithms/base.py`
- `Algorithm` ABC: `can_apply`, `apply`, `run`
- `StepResult(before, after, changed)`
- Traversal helper'ları

#### `algorithms/distribute.py`
- `A * (B + C) → A*B + A*C`
- Non-commutative korumalı

#### `algorithms/flatten.py`
- Associativity

#### `algorithms/sort_product.py` — Kritik dosya
- Property'lere göre davran: NonCommuting swap'lemez, AntiCommuting sign
  üretir, GradedCommutative + Graded(d) Koszul sign
- Stabil sorting algoritması
- Sembolik derecelerle doğru `(-1)^{|a||b|}` üretimi

#### `algorithms/collect_terms.py`
- Aynı terimleri topla (structural equality)

#### `algorithms/simplify.py`
- Pipeline: flatten → distribute → sort_product → collect_terms
- Fix-point algoritması

**Testler:** her algoritma için birim testler + pipeline testleri.

---

### Faz 3 — Pattern Matching ve Substitute

**Amaç:** Rewrite kuralları uygulayabilen motor.

#### `core/wildcards.py` — Genişletme
- Head/subtree/index wildcards
- Property-constrained wildcards
- Variadic (çoklu) wildcard'lar

#### `algorithms/substitute.py`
- Structural pattern matching with backtracking
- Binding consistency
- LHS → RHS rewrite
- Alt-ağaçlara uygulama
- `converge` modu (fix-point)
- Koşullu rule'lar

**Testler:** basit rewrite, wildcard rewrite, consistency, converge.

---

### Faz 4 — Derivations ve Leibniz

**Amaç:** Graded derivasyonlar ve Leibniz kuralı.

#### `algebra/derivation.py`
- `Derivation(degree)` sınıfı
- Kompozisyon, commutator
- Derivasyon uygulaması node'u

#### `algorithms/product_rule.py`
- Graded Leibniz: `D(a*b) = D(a)*b + (-1)^{|D||a|} a*D(b)`
- Çok faktörlü çarpımlar
- Derivation-of-derivation

#### `algebra/commutator.py`
- Graded commutator: `[A,B] = AB - (-1)^{|A||B|} BA`
- Derivasyonların graded commutator'u
- `d² = 0` özel durumu: `[d,d] = 2d²`

**Testler:** Leibniz doğruluğu, Koszul sign tutarlılığı, commutator
antisymmetry, compose → derivation (derece toplamı).

---

### Faz 5 — Brackets

**Amaç:** Soyut bracket yapılarının çerçevesi ve derived bracket.

#### `brackets/base.py`
- `GradedBracket` ABC
- `degree`, `is_graded_antisymmetric`
- `__call__(a, b)`
- Abstract aksiyom testleri

#### `brackets/lie.py`
- Standart Lie bracket (degree 0)

#### `brackets/schouten.py` — Özel önem

Schouten-Nijenhuis bracket paketin en önemli base bracket'ıdır çünkü
Poisson ve Koszul'ün derived bracket olarak kurulduğu cebir budur.

- Multivector fields üzerinde: `⊕_k Γ(Λ^k TM)`
- **Fonksiyonlar (0-vektörler) dahil** — Poisson bracket türetebilmek için
- Degree konvansiyonu: `|X| = k - 1` for `X ∈ Λ^k TM`
- Graded antisymmetric, graded Jacobi
- Özel durumlar:
  - `[X, Y]_SN = [X, Y]_Lie` (iki 1-vektör)
  - `[f, X]_SN = -X(f)` (fonksiyon + vektör)
  - `[X, f]_SN = X(f)`
  - `[f, g]_SN = 0` (iki fonksiyon)
- `[π, π]_SN` bivektörün kendisiyle — Poisson koşulunun merkezi 3-vektör

#### `brackets/koszul.py`

**İki tanımı var ve paket ikisinin aynı olduğunu ispatlayabilmeli:**

- **Klasik:** `[α,β]_K = L_{ρα}β - L_{ρβ}α - d⟨ρα,β⟩`
- **Derived:** `KoszulBracket(π) = DerivedBracket(SN, π, acting_on=Forms)`
- `prove_equivalence(classical, derived)` ile eşdeğerlik adım adım
- Jacobi: derived bracket teoreminden → `[π,π]_SN = 0`

#### `brackets/courant.py`

- Klasik: `[(X,α), (Y,β)]_C = ([X,Y], L_X β - L_Y α + ½d(ι_X β - ι_Y α))`
- **Derived bracket perspektifi**: generator `Θ`, koşul `[Θ,Θ] = 0`
- H-twisted: `[Θ + H, Θ + H] = 0 ⟺ dH = 0`
- Jacobiator'u `[Θ,Θ]` cinsinden

#### `brackets/dorfman.py`
- `[(X,α), (Y,β)]_D = ([X,Y], L_X β - ι_Y dα)`
- Leibniz sağlar, antisymmetric değil

#### `brackets/derived.py` — Paketin matematiksel kalbi

**`DerivedBracket` sınıfı:**

Parametreler:
- `base: GradedBracket` (genelde Schouten-Nijenhuis)
- `Q` — generator
- `degree_Q` — generator'ın derecesi
- `acting_on` — hangi derecedeki elemanlar

Tanım: `{a, b}_Q := [[a, Q]_base, b]_base`

Otomatik özellikler:
- `degree = degree_Q - 2`
- Graded Leibniz — her zaman (koşulsuz)
- Graded antisymmetry (konvansiyon gereği)
- Jacobi: koşullu — `[Q, Q]_base = 0` ⟺ Jacobi

API:
- `__call__(a, b)`
- `jacobi_obstruction()` — `[Q, Q]_base` ifadesi
- `jacobi_condition()` — condition object
- `expand_definition(a, b)` — proof chain üretir

**`DerivedBracketTheorem` verifier** (`proof/verifier.py`'dan kullanılır):

Teorem paket içinde bir kez, soyut olarak kanıtlanır ve sonra her derived
bracket instance'ı için otomatik çağrılır. Her sorgulandığında:

> "Bu bir derived bracket'tir. Derived Bracket Theorem (bkz. Tutorial 7,
> THEOREM_derived_bracket_jacobi) uyarınca Jacobi ⟺ `[Q, Q]_base = 0`."

**İki tanımın eşdeğerlik ispatı:**

`prove_equivalence(classical_def, derived_def)` ile klasik formül ile
derived bracket tanımının eşit olduğunu gösterir.

**`derived_bracket(bracket, Q, **opts) → GradedBracket`** helper.

#### `brackets/custom.py`
- Kullanıcı bracket tanımı helper'ı

**Testler:**
- Lie bracket Jacobi
- SN bracket 1-vektörlerde Lie'ye indirgeniyor
- Koszul iki tanımın eşdeğerliği
- Courant Jacobiator
- Derived: `[Q,Q]=0` ⟺ Jacobi

---

### Faz 6 — Calculus (Cartan Operators)

**Amaç:** Soyut Cartan calculus framework'ü, operatör-seviye ispat.

#### `calculus/exterior_d.py` — `d`

Bu dosya property provenance'ın ilk gerçek kullanım alanı.

- Degree +1, graded anti-derivation
- Fonksiyonlarda `d(f) = df` (axiom)
- Invariant formülle genişletme (axiom veya theorem — seçim)
- **`d² = 0` iki modda**:
  - `d²=0` axiom olarak (`Closed` property, hızlı rewrite)
  - `d²=0` theorem olarak (Lie bracket Jacobi'sinden türetilmiş ispat)
  - Registry'de her iki versiyon da saklı; kullanıcı mod seçer

#### `calculus/interior.py` — `ι_X`
- Degree -1, graded anti-derivation
- `ι_X(f) = 0` fonksiyonlarda
- `ι_X(df) = X(f)` 1-formlarda
- `ι_X ∘ ι_X = 0` (theorem: form antisymmetry'den)

#### `calculus/lie_derivative.py` — `L_X`

İki tanım seçeneği:

**A) Flow tanımı (axiom)**
- `L_X(ω) = d/dt|_0 φ_t^* ω` (axiom)
- Cartan's magic formula theorem olarak çıkar

**B) Cartan tanımı (axiom)**
- `L_X := d ∘ ι_X + ι_X ∘ d` (axiom)
- Cartan's magic formula tautoloji

Paket her iki tanımı destekler; kullanıcı hangisini kullandığını seçer.

#### `calculus/anchor.py` — Anchor
- `ρ: E → TM`, lineer
- Bracket uyumu: `ρ([X,Y]_E) = [ρX, ρY]`

#### `calculus/hamiltonian_vf.py`
- `X_f` iki tanım:
  - Symplectic: `ι_{X_f}ω = -df`
  - Derived: `X_f = -[f, π]_SN`
- Eşdeğerlik ispatı

#### `calculus/exterior_algebra.py` — Dış cebir (YENİ)

`AgreementOnGenerators` stratejisi için cebirin generator yapısını
açıkça bilmek gerek:

```python
class ExteriorAlgebra:
    """Ω*(M) — dış cebir"""

    @property
    def generators(self):
        return [
            Function,                # C^∞(M) (deg 0)
            OneForm("df"),            # exact 1-formlar (deg 1 generator'ları)
        ]

    def is_generated_by(self, elements):
        """Verilen elemanlar cebiri Leibniz altında üretir mi?"""
```

#### `calculus/operator_equation.py` (YENİ)

```python
class OperatorEquation:
    """A = B operatör denklemi (form'a uygulandığında)"""

    lhs: Operator
    rhs: Operator
    algebra: Algebra   # hangi cebir üzerinde

    def prove(self, strategy=None):
        """Default: AgreementOnGenerators"""
```

#### `calculus/cartan.py` — CartanCalculus

- `CartanCalculus(d, L, ι, bracket)`
- Aksiyomlar olarak Cartan relations:
  - `d² = 0`
  - `[L_X, L_Y] = L_{[X,Y]}`
  - `[L_X, ι_Y] = ι_{[X,Y]}`
  - `[d, ι_X] = L_X`
  - `[d, L_X] = 0`
- `verify_all(mode="efficient"|"foundational")`
- `verify(relation, mode=...)` — tek relation
- Farklı calculus varyantları:
  - Standart manifold
  - Lie algebroid (d_E, L_E, ι_E)
  - Twisted (d_H = d + H∧ where H kapalı 3-form)

**Testler:**
- `ω(X_f, X_g) = X_f(g)` zinciri — iki strateji
- Cartan relations standart manifold (her iki modda)
- Cartan relations Lie algebroid calculus'ünde
- Custom calculus tanımı
- Operator-level vs element-level mod karşılaştırması

---

### Faz 7 — Proof System

**Amaç:** Paketin kullanıcıya görünen yüzü.

#### `proof/step.py` — ProofStep

- `before`, `after` (Expr)
- `rule_applied`
- `justification`
- `parent`, `children` (nested)
- `provenance_tag`: hangi statüde kullanıldı (axiom/theorem)

#### `proof/chain.py` — ProofChain

- Sıralı adım listesi
- Nesting
- `append`, `extend`, `merge`, `fold`
- Verbosity: full, summary, compact, latex, terminal

#### `proof/tracer.py`

- `TracingAlgorithm(inner, chain)`
- Her çağrıyı kaydet
- Otomatik justification üretimi

#### `proof/expansion.py` — ExpansionEngine

- Tanım kullanarak iç içe açma
- Her açılım bir ProofStep

#### `proof/recognizers.py`

- `CommutatorRecognizer`
- `CyclicSumRecognizer`
- `LeibnizRecognizer`
- `AntisymmetryRecognizer`
- `SchoutenBracketRecognizer` — SN bracket pattern'leri
- `DerivedBracketRecognizer` — bir bracket'ın derived form olduğunu tanır
- `InvariantDerivativeFormulaRecognizer` — `dα(X,Y) = X(α(Y)) - Y(α(X)) - α([X,Y])`

#### `proof/strategies.py`

Artık genişletilmiş bir strateji katalogu:

- `ExpandAndSimplify`: LHS aç, sadeleştir, 0'a düşmeli
- `ExpandBothSidesAndCompare`: iki tarafı aç
- `InductionOnDegree`: derece üzerinde
- `PatternGuided`: kullanıcı rehberli
- **`AgreementOnGenerators`** (YENİ, operatör denklemleri için)
  - İki derivasyonun aynı derecede olduğunu kontrol et
  - Cebirin generator'larını al
  - Her generator'da eşitliği doğrula
  - Derivation extension ile sonlandır
- **`UnrollToFoundations`** (YENİ)
  - Kullanılan theorem'leri recursive olarak aç
  - Durma koşulu: verilen axiom set'i
  - Cache: aynı theorem iki kez açılmıyor
- **`DerivedBracketStrategy`** (YENİ)
  - Bracket'ın derived form olduğunu tanı
  - Derived bracket teoremini uygula
  - `[Q,Q]=0` koşulunu çıkar
- `OperatorLevelProof` vs `ElementLevelProof` — mode toggle

#### `proof/verifier.py` — Yüksek seviye API

- `prove_jacobi(bracket, a, b, c, mode="efficient") → ProofChain`
- `prove_antisymmetry(bracket, a, b)`
- `prove_leibniz(derivation, bracket, a, b)`
- `prove_cartan_relations(calculus, mode=...)`
- `show_equal(lhs, rhs, strategy=None)`
- `prove_equivalence(statement1, statement2)`
- `prove_operator_equation(op1, op2, algebra)` (YENİ)
- `unroll_property(prop) → ProofChain` (YENİ, bir property'nin dayandığı
  aksiyomları gösterir)

**Testler:**
- Her strateji için referans ispatlar
- Mode karşılaştırmaları
- Nested proof doğruluğu
- Başarısız ispatlarda hata mesajı

---

### Faz 8 — Display  *(KAPALI — Stage A + B + C + verbosity + collapsible)*

**Amaç:** LaTeX, terminal, Jupyter çıktısı.

#### Mimari sapma: dispatch vs. `_latex_()` metodu

Plan başlangıçta "`_latex_()` metodu her Expr'da" öngörüyordu; gerçek
uygulamada **MRO tabanlı dispatch fonksiyonları** tercih edildi
(`to_ascii(expr)`, `to_latex(expr)`). Nedeni:

- Core `Expr` hiyerarşisi render şekline bağımsız kalır — `display/`
  paketi olmadan da derlenir/test edilir.
- Yeni render hedefi (HTML collapsible, rich tree) eklerken Expr
  sınıflarına tekrar metod eklemek gerekmez.
- Subclass'lar (örn. `ExteriorDerivative : Derivation`) genel
  `Derivation` handler'ına MRO ile düşer; her subclass'ın kendi
  metodunu kaydetmesine gerek yok.
- Jupyter'ın `_repr_latex_` / `_repr_html_` / `_repr_mimebundle_`
  sözleşmesi `Expr` üzerinde değil, açık opt-in wrapper'larda
  (`LatexDisplay`, `HtmlProofDisplay`) bulunur — test ederken bir
  notebook boot etmek gerekmez.

#### `display/ascii.py`
- MRO dispatch renderer: `to_ascii`, `step_to_ascii`, `chain_to_ascii`.
- Precedence rung'ları + sign normalisation (`Sum(a, Neg(b))` → `a - b`).
- `VERBOSITY_MODES = ("full", "summary", "compact")` — tüm renderer
  katmanlarının paylaştığı sabit.

#### `display/latex.py`
- `to_latex(expr)` dispatch fonksiyonu — Expr'a metod eklenmez.
- `latex_name(...)`: Greek / musical / algebraic glyph translation +
  multi-char subscript bracing (`X_ab` → `X_{ab}`).
- `_escape_text(...)`: rule/justification metinlerinde hem ASCII
  özel karakterleri (`\_#%&$`) hem de Unicode glyph'leri
  `\ensuremath{...}` ile sarar — pdfLaTeX Unicode hatası engellenir.
- `chain_to_latex` → `\begin{align*} … \end{align*}` bloğu
  (paper-ready flat form).

#### `display/terminal.py`
- `rich` opsiyonel; yüklü değilse sessizce `chain_to_ascii`'ye fallback.
- `HAS_RICH` flag kullanıcıya görünür.
- Renkli `Tree` hiyerarşisi; recording Console `file=io.StringIO()`
  ile kurulur → stdout duplikasyonu yok.
- Verbosity: `full` / `summary` / `compact` (compact hem justification
  hem child'ları suppress eder → flat table-of-contents).

Örnek terminal çıktısı (Cartan magic formula için):

```
═══ Cartan's Magic Formula: [d, ι_X] = L_X on Ω*(M) ═══
 
Mode: efficient  (d²=0 used as property)

Step 1: Both sides are graded derivations
  |[d, ι_X]| = |d| + |ι_X| = +1 + (-1) = 0
  |L_X| = 0
  ✓ Same type

Step 2: Agreement on generators (AgreementOnGenerators strategy)
  Generators of Ω*(M): f ∈ C^∞, df

Step 3: Check on f:
  [d, ι_X](f) = d(ι_X f) + ι_X(df) = 0 + X(f) = X(f) = L_X(f)  ✓

Step 4: Check on df:
  [d, ι_X](df) = d(X(f)) + ι_X(d²f) = d(X(f)) + 0 = d(X(f))
  L_X(df) = d(L_X f) = d(X(f))  [using d²=0 as property]
  ✓

Step 5: Both derivations agree on generators → equal on Ω*(M)

═══ Switch to foundational mode for d²=0 derivation ═══
(kullanıcı istediğinde unroll edilir)
```

#### `display/jupyter.py`
- İki tamamlayıcı wrapper; Expr hiyerarşisi monkey-patch *edilmez*:
  - `LatexDisplay` — inline `$…$` veya `\begin{align*}…\end{align*}`
    payload; `_repr_latex_` / `_repr_html_` / `_repr_mimebundle_`
    sözleşmesini yerine getirir.
  - `HtmlProofDisplay` — collapsible proof tree (yalnız `text/html`).
- Helpers:
  - `display_expr`, `display_step`, `display_chain` (=
    `display_proof`) — flat `align*` çıktı, paper-ready.
  - `display_step_collapsible`, `display_chain_collapsible` —
    `<details open>` HTML ağaç; her adım `[rule] (tag) \(before \to
    after\) — just` biçiminde MathJax'e bırakılmış matematik içerir.
    `max_depth`, `title`, `verbosity` opsiyonları terminal renderer'ı
    ile aynı semantikte.

**Testler:** 173 display testi (`tests/test_display/`), golden-value
assert'ler — tüm renderer'ların verbosity / tipe hata / Unicode
sanitisation / stdout leak regresyon testleri dâhil.

---

### Faz 9 — Library (Hazır Yapılar ve Teorem Kütüphanesi)

**Amaç:** Kullanıcının hemen kullanabileceği yapılar + merkezi teorem
deposu.

#### `library/theorem_book.py` (YENİ, kritik)

Türetilmiş tüm önemli teoremlerin ispatlarıyla birlikte saklandığı
merkezi dosya. `UnrollToFoundations` stratejisi buradaki ispatları
kullanır.

```python
THEOREM_d_squared_zero = Theorem(
    name="d² = 0",
    statement=...,
    from_axioms=[
        "d is graded derivation of degree +1",
        "invariant formula on 1-forms",
        "Lie bracket definition",
        "[X,Y] commutator formula"
    ],
    proof=<ProofChain showing d(df) via invariant formula → 0>
)

THEOREM_cartan_magic = Theorem(
    name="[d, ι_X] = L_X",
    statement=...,
    from_axioms=[
        "d axioms",
        "ι_X axioms",
        "L_X on functions = X(f)",
        "d² = 0"
    ],
    proof=<ProofChain via AgreementOnGenerators>
)

THEOREM_LX_LY_commutator = Theorem(
    name="[L_X, L_Y] = L_{[X,Y]}",
    from_axioms=[...],
    proof=...
)

THEOREM_derived_bracket_jacobi = Theorem(
    name="Derived Bracket Theorem",
    statement="{a,b}_Q Jacobi ⟺ [Q,Q] = 0",
    from_axioms=["base bracket Jacobi", "base bracket Leibniz"],
    proof=<soyut ispat, değişkenler seviyesinde>
)

THEOREM_poisson_equals_Xf_g = Theorem(
    name="{f,g}_π = X_f(g)",
    from_axioms=["derived bracket definition", "SN bracket on functions"],
    proof=<bir-iki adımlık türetim>
)

THEOREM_koszul_classical_equals_derived = Theorem(
    name="[α,β]_K (classical) = [[α,π]_SN, β]_SN",
    from_axioms=[...],
    proof=<uzun ispat, Cartan relations ve anchor uyumu kullanır>
)

THEOREM_iota_squared_zero = Theorem(
    name="ι_X ∘ ι_X = 0",
    from_axioms=["form antisymmetry"],
    proof=...
)
```

Theorem Book, paketin "matematiksel hafızası"dır. Her teorem:
- Adı
- Statement
- Bağlı olduğu aksiyomlar
- İspat zinciri
- Kullanım örneği

Paket "bu property theorem'dir, ispatı bkz. THEOREM_X" diyerek kullanır.

#### `library/symplectic.py`
- `SymplecticManifold`
- Non-degenerate + closed 2-form
- `hamiltonian_vf(f) → X_f`

#### `library/poisson.py`

Paketin "birleşik bakış" felsefesinin gösterim alanı.

Üç eşdeğer tanım:

1. **Symplectic:** `{f, g} = ω(X_f, X_g)`  where  `ι_{X_f} ω = -df`
2. **Bivector:** `{f, g} = π(df, dg)`
3. **Derived bracket:** `{f, g}_π = [[f, π]_SN, g]_SN`
   - `X_f = -[f, π]_SN` türetilir (aksiyom değil)

```python
pi = PoissonBivector("π")
poisson = PoissonBracket.from_bivector(pi)

prove_jacobi(poisson)
# → "Derived bracket (Q=π). Theorem: Jacobi ⟺ [π,π]_SN = 0."

show_equal(poisson(f,g), X_f(g), strategy="derived")  # kısa
show_equal(poisson(f,g), X_f(g), strategy="expand")   # uzun, öğretici
```

Tek varsayım `[π,π]_SN = 0` üzerinden tüm hiyerarşi:
- `{·,·}_π` Jacobi
- `[·,·]_K` Jacobi
- `{f,g} = X_f(g)`
- `X_{{f,g}} = [X_f, X_g]`

#### `library/lie_algebroid.py`
- `LieAlgebroid(E, bracket, anchor)`
- Aksiyomlar: anchor uyumu, Leibniz, Jacobi
- Cartan calculus otomatik: `d_E`, `L_E`, `ι_E`

#### `library/courant_algebroid.py`
- Full Courant aksiyomları
- Pairing, anchor, bracket uyumu
- Dirac structure

#### `library/dirac.py`
- Dirac structure (izotropik + involutive)
- Poisson, presymplectic özel halleri

**Testler:** her yapı için smoke testi (kur → aksiyom kontrol).

---

### Faz 10 — Dokümantasyon ve Tutorials

1. **`01_first_steps.md`** — İlk adımlar
   - Expr, Symbol, toplama/çarpma
   - Property atama
   - Basit sadeleştirme

2. **`02_jacobi_identity.md`** — Jacobi gösterme
   - Lie bracket tanımı
   - Graded cebirde Jacobi
   - `prove_jacobi` kullanımı

3. **`03_poisson_geometry.md`** — Poisson geometri
   - Symplectic form kurma
   - Hamiltonian VF
   - `{f,g} = ω(X_f,X_g) = X_f(g)` zinciri (expand + derived modlar)
   - Poisson Jacobi (derived bracket teoremi)
   - `[π,π]_SN = 0` merkezi rolü

4. **`04_lie_algebroid.md`** — Lie algebroid
   - Anchor map
   - Bracket aksiyomları
   - Leibniz kontrolü
   - Algebroid Cartan calculus

5. **`05_cartan_calculus.md`** — Cartan relations
   - Standart manifold (iki modda)
   - Lie algebroid
   - Twisted
   - Cartan's magic formula
   - Operator-level ispat (AgreementOnGenerators)
   - Element-level ispat (p-form açılımı)

6. **`06_custom_bracket.md`** — Kendi bracket'ı
   - Tanım formundan bracket
   - Aksiyom atama
   - Jacobi test

7. **`07_derived_bracket.md`** — Derived bracket birleştirici
   - `{a,b}_Q = [[a,Q],b]` inşası
   - Kosmann-Schwarzbach teoremi
   - Poisson derived, Koszul derived
   - Klasik vs derived eşdeğerlik
   - Courant derived ve H-twist

8. **`08_unified_picture.md`** — Tek teorem, çok sonuç
   - `[π,π]_SN = 0` tek koşulundan hiyerarşi
   - Tek varsayımla çok ispat
   - Pedagojik özet

9. **`09_foundations.md`** (YENİ) — Aksiyomdan teoreme
   - `d² = 0` nereden gelir? (Lie bracket Jacobi'sinden)
   - Foundational mod kullanımı
   - Özel aksiyom seti ile çalışma
   - Theorem Book yapısı

Her tutorial için çalıştırılabilir Jupyter notebook.

---

### Faz 11 — İleri Özellikler

- Performans: hashing + memoization, pattern indexing
- Konfigürasyon: sign convention seçimi, L_X tanımı seçimi
- Diagnostic: doğrulama başarısızsa hangi aksiyom eksik
- Export: ProofChain → .tex, TikZ diagram
- CLI: `gradalg verify calculus.yaml`

## Geliştirme Sırası (Bağımlılık Grafiği)

```
Faz 0 (altyapı)
  │
  ▼
Faz 1 (core: Expr + Properties + Provenance + Symbolic degrees)
  │
  ▼
Faz 2 (basic algorithms)
  │
  ▼
Faz 3 (pattern matching + substitute)
  │
  ▼
Faz 4 (derivations + Leibniz)
  │
  ▼
Faz 5 (brackets + derived bracket)    ║    Faz 8 (display) — paralel
  │
  ▼
Faz 6 (calculus + exterior algebra + operator equations)
  │
  ▼
Faz 7 (proof system — tam)
  │
  ▼
Faz 9 (library + Theorem Book)
  │
  ▼
Faz 10 (docs + 9 tutorial)
  │
  ▼
Faz 11 (iyileştirmeler)
```

Proof system'in minimal versiyonu Faz 3'ten itibaren devrede olmalı
(`TracingAlgorithm`), tam entegrasyon Faz 7'de. Theorem Book Faz 9'da
doldurulur ama her faz tamamlandığında ilgili teoremler eklenir.

## Hedef Kullanıcı API'si (Tam Önizleme)

Paket bittiğinde kullanıcı kodu:

```python
from gradalg import *

# ---------- Temel ----------
f, g, h = Functions("f g h")
X, Y, Z = VectorFields("X Y Z")
alpha, beta = Forms("α β", degree=1)
p = SymbolicDegree("p")
omega_p = Form("ω", degree=p)   # sembolik p-form

# ---------- Symplectic & Poisson ----------
omega = SymplecticForm("ω")
pi = PoissonBivector("π")

# İki tanım eşdeğer
poisson_symp = PoissonBracket.from_symplectic(omega)
poisson_pi   = PoissonBracket.from_bivector(pi)
prove_equivalence(poisson_symp, poisson_pi).display()

# ---------- İspat zincirleri ----------
# {f,g} = ω(X_f, X_g) = X_f(g) — iki mod
show_equal(poisson_symp(f,g), Action(X_f, g),
           strategy="expand").display()      # uzun, öğretici
show_equal(poisson_pi(f,g), Action(X_f, g),
           strategy="derived").display()     # kısa, yapısal

# ---------- Jacobi: tek varsayım, çok sonuç ----------
with assume(sn_bracket(pi, pi) == 0):
    prove_jacobi(poisson_pi).display()       # derived bracket theorem
    prove_jacobi(KoszulBracket(pi)).display()  # aynı teorem, aynı koşul

# ---------- Cartan relations: iki mod ----------
d = ExteriorDerivative()
iota_X = InteriorProduct(X)
L_X = LieDerivative(X)
calc = CartanCalculus(d, L_X, iota_X, LieBracket())

# Hızlı mod
calc.verify_all(mode="efficient").display()

# Foundational mod — d² = 0 bile türetilsin
calc.verify_all(mode="foundational").display()

# Custom aksiyom seti
calc.verify_all(
    mode="custom",
    axioms={"d acts as derivation", "[X,Y] = XY - YX on functions"}
).display()
# → "d² = 0 türetilecek, Lie Jacobi türetilecek, Cartan magic türetilecek"

# ---------- Operatör denklemi: p-form üzerinde ----------
cartan_magic = OperatorEquation(
    lhs=commutator(d, iota_X),
    rhs=L_X,
    algebra=ExteriorAlgebra()
)
cartan_magic.prove(strategy=AgreementOnGenerators()).display()

# ---------- p-form üzerinde explicit ----------
show_equal(
    commutator(d, iota_X)(omega_p),
    L_X(omega_p),
    strategy="expand"    # p sembolik, açılım hesabı
).display()

# ---------- Provenance sorgulama ----------
unroll_property(THEOREM_d_squared_zero).display()
# → "d² = 0 is derived from: d axioms + invariant formula + Lie Jacobi"

# ---------- Lie algebroid ----------
anchor = Anchor("ρ")
E_bracket = CustomBracket("[·,·]_E", antisymmetric=True)
E = LieAlgebroid(bundle="E", bracket=E_bracket, anchor=anchor)
E.verify_axioms().display()
E.cartan_calculus().verify_all().display()

# ---------- Courant algebroid ----------
Theta = CourantTensor("Θ")
C = CourantAlgebroid.from_generator(Theta)
C.verify_axioms(mode="derived").display()
# → "Assumes [Θ,Θ] = 0"

# H-twisted
H = ClosedThreeForm("H")
C_H = CourantAlgebroid.from_generator(Theta + H)
prove_equivalence(
    C_H.jacobi_condition(),
    (d(H) == 0)
).display()

# ---------- Derived bracket hiyerarşisi ----------
# Tek bir π üzerinden tüm bracket'lar
bracket_zoo = [
    derived_bracket(SN, pi, acting_on=Functions),   # Poisson
    derived_bracket(SN, pi, acting_on=OneForms),    # Koszul
    derived_bracket(SN, pi, acting_on=TwoForms),    # Extended Koszul
]

# Hepsi tek varsayımla Jacobi sağlar
with assume(sn_bracket(pi, pi) == 0):
    for br in bracket_zoo:
        prove_jacobi(br).summary()
# Çıktı: her biri için aynı justifikasyon (derived bracket theorem)
```

## Başarı Kriterleri

Proje "bitti" dediğimizde şunların hepsi çalışıyor olmalı:

1. `{f,g} = ω(X_f, X_g) = X_f(g)` zinciri adım adım ispatlanır, hem
   klasik açılımla hem derived bracket yoluyla güzel çıktı verir
2. Lie bracket için Jacobi identity tam ispatı üretilir
3. **Derived Bracket Theorem** soyut olarak bir kez kanıtlanır ve her
   derived bracket instance'ı için otomatik uygulanır
4. Schouten-Nijenhuis bracket fonksiyonlar dahil tüm multivector'larda
   doğru çalışır; `[π,π]_SN` kurulabilir ve sıfırlık koşulu varsayım
   olarak kullanılabilir
5. Poisson bracket için Jacobi, `[π,π]_SN = 0` koşulundan derived bracket
   teoremi ile türetilir — kısa ve yapısal ispat
6. Koszul bracket için Jacobi aynı `[π,π]_SN = 0` koşulundan, aynı
   teoremle gösterilir
7. Koszul bracket'ın klasik tanımı ile derived tanımının eşdeğerliği
   adım adım ispatlanır
8. Koszul bracket'ın graded antisymmetric olduğu gösterilir
9. Courant bracket için `[Θ,Θ] = 0` koşulu ve H-twisted versiyonu
   `dH = 0` ispatlanır
10. Cartan relations hem operatör-seviye (`AgreementOnGenerators` ile) hem
    eleman-seviye (p-form açılımı ile) ispatlanır
11. `d² = 0` hem axiom olarak kullanılabilir hem theorem olarak Lie
    bracket Jacobi'sinden türetilebilir
12. **Foundational mod**: herhangi bir theorem istenirse en temel
    aksiyomlara kadar unroll edilebilir
13. **Custom axiom mod**: kullanıcı hangi property'leri axiom kabul
    edeceğini söyleyip gerisini türettirebilir
14. Bir Lie algebroid tanımlanıp tüm aksiyomları test edilir
15. Kullanıcı kendi bracket'ını tanımlayıp aksiyom testlerini çalıştırabilir
16. "Brute force" modu: kullanıcı klasik tanım üzerinden uzun ispat alabilir
17. Theorem Book'ta en az 10 önemli teorem ispatlarıyla kayıtlı
18. LaTeX çıktıları makaleye direkt yapıştırılabilir kalitede
19. Jupyter notebook'ta yerleşik render çalışır (provenance tree collapse/expand)
20. Dokümantasyon ve 9 tutorial tamamlanmış (foundations tutorial dahil)

## Sınırlamalar (Bilinçli Tercihler)

Bu paketin yapmayacakları:

- Koordinat bazlı somut hesap (SageManifolds kullanın)
- Tensor index canonicalisation (Butler-Portugal gerektirir, kapsam dışı)
- Sayısal hesap
- PDE çözümü
- Gamma matrix algebra, spinor manipulation (Cadabra2 kullanın)

## Bağımlılıklar

**Zorunlu:**
- Python ≥ 3.10
- Standart kütüphane (dataclasses, abc, typing, functools)

**Opsiyonel:**
- `rich` — zengin terminal çıktısı
- `pytest` — test
- `mkdocs` veya `sphinx` — docs
- `hypothesis` — property-based testler

**Kesinlikle yok:**
- SymPy, SageMath, Cadabra — bağımsız paket hedefi

## Test Stratejisi

- **Birim testler**: Her algoritma, property, Expr tipi için
- **Entegrasyon testleri**: Uçtan uca ispat akışları
- **Golden file testleri**: ProofChain çıktılarının byte-exact sabitliği
- **Provenance testleri**: theorem'lerin ispatları aksiyomlara indirgeniyor mu
- **Mod karşılaştırma testleri**: `efficient` ve `foundational` mod aynı
  sonuca varıyor mu (farklı yollarla)
- **Property-based testler** (hypothesis, opsiyonel):
  - Rastgele graded ifadeler için Koszul sign tutarlılığı
  - Rastgele derivasyonlar için Leibniz
  - Rastgele bracket'lar için antisymmetry ↔ graded antisymmetry
- **Regression testler**: bulunan bug'lar için kalıcı test

## İlk Oturumda Yapılacak İşler

Pratik başlangıç:

1. **Faz 0 tamamla** — `pyproject.toml`, dizin, pytest
2. **Faz 1'in ilk yarısı** — `core/expr.py`:
   - `Expr` base
   - `Symbol`, `Sum`, `Product`, `Zero`, `One`
   - Operatör overloading
   - Basit `__repr__`
3. **İlk testler** — structural equality, property atama skeleton
4. **Provenance iskeleti** — `Property` sınıfının `status`, `dependencies`,
   `proof` alanlarının taslağı (doldurulacak)

Bu dört adım bittiğinde paket yaşamaya başlar ve üzerine her şey
yapılandırılır.

---

Plan canlı bir dokümandır. Geliştirme sırasında kararlar değişirse
buraya yansıtılır.
