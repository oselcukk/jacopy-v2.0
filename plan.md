# jacopy, Graded Algebra ve Bracket Hesabı için Sembolik Paket

## Proje Özeti

`jacopy`, graded cebir, bracket yapıları ve Cartan calculus alanında
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

- **Property provenance**: Her property'nin statüsü açık tutulur,
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

**Seviye 0, Cebirsel temel**
- Vektör uzayı (toplama, skaler çarpım, 0 ve 1)
- Associative cebir (wedge ürünü birleşmeli)
- Wedge'in graded commutative olması: `α ∧ β = (-1)^{|α||β|} β ∧ α`
- Graded structure: her elemanın bir derecesi var

**Seviye 1, Derivasyon aksiyomları**
- Derivasyonun tanımı: R-lineer + graded Leibniz rule
- Graded commutator tanımı: `[A,B] = AB − (−1)^{|A||B|} BA`
- İki graded derivation'ın graded commutator'u yine graded derivation
  (dereceleri toplanır)

**Seviye 2, Temel operatörlerin tanımları**
- `d` (exterior derivative): deg +1 graded anti-derivation, fonksiyonlarda
  `d(f) = df` ile tanımlı, invariant formülle 1-formlara uzatılır
- `ι_X` (interior product): deg -1 graded anti-derivation, `ι_X(df) = X(f)`
- `L_X` (Lie derivative): iki seçenek, (a) flow ile tanımlı, Cartan'ın
  büyülü formülü teorem olarak çıkar, (b) `L_X := d ∘ ι_X + ι_X ∘ d` ile
  tanımlı, Cartan formülü tautoloji olur

**Seviye 3, Bracket aksiyomları**
- Lie bracket tanımı (vektör alanları üzerinde): `[X,Y](f) = X(Y(f)) − Y(X(f))`
- Jacobi identity for Lie bracket: türetilebilir veya axiom olarak kabul
  edilir (konvansiyon seçimi)
- Schouten-Nijenhuis bracket: Lie bracket + wedge Leibniz ile genişletme

**Türetilebilir theorem'ler (sistemde hazır ispatlarıyla saklanır):**
- `d² = 0` (Jacobi identity for Lie bracket'tan türer, bkz. Tutorial 9)
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
- Herhangi bir bracket için Jacobi, antisymmetry, Leibniz kontrolü, adım
  adım ispatla
- **Derived bracket teoremi** üzerinden Poisson, Koszul, Courant gibi
  bracket'ların Jacobi'sinin tek bir koşula (`[Q,Q] = 0`) indirgenmesi ve
  bunun ispat olarak sunulması
- Aynı generator'ın (örn. bir Poisson bivektör π) farklı derecelerdeki
  elemanlara (fonksiyonlar, 1-formlar, k-formlar) etki edişinin
  gösterilmesi, `{f,g}_π`, `[α,β]_K`, extended Koszul hepsinin aynı
  `[π,π]_SN = 0` koşulundan çıktığını ispatlama
- Cartan calculus ilişkilerinin (`[L_X, L_Y] = L_{[X,Y]}`,
  `[d, ι_X] = L_X`, vb.) farklı calculus'lerde doğrulanması,
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
jacopy/
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
├── jacopy/
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
│   │   ├── derived.py            (DerivedBracket, paketin kalbi)
│   │   └── custom.py             (Kullanıcı bracket helper'ı)
│   │
│   ├── calculus/                 [Faz 6]
│   │   ├── __init__.py
│   │   ├── exterior_d.py         (d operatörü, iki mod: d²=0 axiom/theorem)
│   │   ├── interior.py           (ι_X, interior product)
│   │   ├── lie_derivative.py     (L_X, iki tanım seçeneği)
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

### Faz 0, Altyapı

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

### Faz 1, Core (Expression Tree + Properties + Provenance)

**Amaç:** Paketin kalbini atmak. Her şey bunun üzerine oturuyor.

#### `core/expr.py`, Expression Tree

- `Expr` base class: `head`, `children`, `indices`, `metadata`
- Operatör overloading: `+`, `-`, `*`, `**`, `-` (unary)
- Concrete node tipleri:
  - `Symbol(name)`
  - `Sum(*args)`
  - `Product(*args)` (non-commutative)
  - `Power(base, exponent)`
  - `Rational(p, q)`, `Integer(n)`
  - `Zero`, `One` (singleton)
  - `Neg(expr)`, rahat işaret yönetimi için
- Traversal: `walk()`, `find(predicate)`, `replace_at(path, new)`
- `clone()`, deep copy
- Temel `__repr__`

#### `core/symbolic_degree.py`, Sembolik Dereceler

Cartan relations'ı generic p-formlar üzerinde ispatlayabilmek için
dereceler **sembolik tamsayı** olabilmeli:

- `SymbolicDegree(name)`, sembolik tamsayı parametre (`p`, `q`, vb.)
- `DegreeExpr`, dereceler arası aritmetik (`p + 1`, `p + q`)
- Cebirsel ilişkiler: `|dω| = |ω| + 1`, `|α∧β| = |α| + |β|`
- Paritelerle çalışma: `(-1)^p`, `(-1)^{p+q}`
- Sembolik derecelerle çalışırken (-1)^... ifadelerinin doğru takibi

#### `core/properties.py`, Property sınıfları + Provenance

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
- `Graded(degree)`, degree sembolik veya concrete
- `NonCommuting`, `AntiCommuting`, `GradedCommutative`
- `Symmetric`, `Antisymmetric`
- `Derivation(degree=0)`
- `Closed` (d²=0), **hem axiom hem theorem olabilen özel durum**
- `NonDegenerate`
- `Vector`, `Form(degree)`, `Function`, `MultiVector(degree)`
- Her property kendi statüsünü kontrol edebilir: `is_axiom()`, `is_theorem()`

#### `core/registry.py`, PropertyRegistry

- Pattern → property listesi map'i
- `attach(pattern, prop)`, tek obje veya liste
- `get(node, prop_type)`, sorgu
- Pattern tipleri: exact name, type, parent-class
- Scope yönetimi: `with registry.scope():`
- **Axiom seti yönetimi**: `with registry.axioms({...}):`, geçici olarak
  bazı property'leri axiom muamelesi yaptır
- Global default registry + özel registry oluşturma

#### `core/wildcards.py`, Pattern primitives

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

### Faz 2, Basic Algorithms

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

#### `algorithms/sort_product.py`, Kritik dosya
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

### Faz 3, Pattern Matching ve Substitute

**Amaç:** Rewrite kuralları uygulayabilen motor.

#### `core/wildcards.py`, Genişletme
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

### Faz 4, Derivations ve Leibniz

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

### Faz 5, Brackets

**Amaç:** Soyut bracket yapılarının çerçevesi ve derived bracket.

#### `brackets/base.py`
- `GradedBracket` ABC
- `degree`, `is_graded_antisymmetric`
- `__call__(a, b)`
- Abstract aksiyom testleri

#### `brackets/lie.py`
- Standart Lie bracket (degree 0)

#### `brackets/schouten.py`, Özel önem

Schouten-Nijenhuis bracket paketin en önemli base bracket'ıdır çünkü
Poisson ve Koszul'ün derived bracket olarak kurulduğu cebir budur.

- Multivector fields üzerinde: `⊕_k Γ(Λ^k TM)`
- **Fonksiyonlar (0-vektörler) dahil**, Poisson bracket türetebilmek için
- Degree konvansiyonu: `|X| = k - 1` for `X ∈ Λ^k TM`
- Graded antisymmetric, graded Jacobi
- Özel durumlar:
  - `[X, Y]_SN = [X, Y]_Lie` (iki 1-vektör)
  - `[f, X]_SN = -X(f)` (fonksiyon + vektör)
  - `[X, f]_SN = X(f)`
  - `[f, g]_SN = 0` (iki fonksiyon)
- `[π, π]_SN` bivektörün kendisiyle, Poisson koşulunun merkezi 3-vektör

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

#### `brackets/derived.py`, Paketin matematiksel kalbi

**`DerivedBracket` sınıfı:**

Parametreler:
- `base: GradedBracket` (genelde Schouten-Nijenhuis)
- `Q`, generator
- `degree_Q`, generator'ın derecesi
- `acting_on`, hangi derecedeki elemanlar

Tanım: `{a, b}_Q := [[a, Q]_base, b]_base`

Otomatik özellikler:
- `degree = degree_Q - 2`
- Graded Leibniz, her zaman (koşulsuz)
- Graded antisymmetry (konvansiyon gereği)
- Jacobi: koşullu, `[Q, Q]_base = 0` ⟺ Jacobi

API:
- `__call__(a, b)`
- `jacobi_obstruction()`, `[Q, Q]_base` ifadesi
- `jacobi_condition()`, condition object
- `expand_definition(a, b)`, proof chain üretir

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

### Faz 6, Calculus (Cartan Operators)

**Amaç:** Soyut Cartan calculus framework'ü, operatör-seviye ispat.

#### `calculus/exterior_d.py`, `d`

Bu dosya property provenance'ın ilk gerçek kullanım alanı.

- Degree +1, graded anti-derivation
- Fonksiyonlarda `d(f) = df` (axiom)
- Invariant formülle genişletme (axiom veya theorem, seçim)
- **`d² = 0` iki modda**:
  - `d²=0` axiom olarak (`Closed` property, hızlı rewrite)
  - `d²=0` theorem olarak (Lie bracket Jacobi'sinden türetilmiş ispat)
  - Registry'de her iki versiyon da saklı; kullanıcı mod seçer

#### `calculus/interior.py`, `ι_X`
- Degree -1, graded anti-derivation
- `ι_X(f) = 0` fonksiyonlarda
- `ι_X(df) = X(f)` 1-formlarda
- `ι_X ∘ ι_X = 0` (theorem: form antisymmetry'den)

#### `calculus/lie_derivative.py`, `L_X`

İki tanım seçeneği:

**A) Flow tanımı (axiom)**
- `L_X(ω) = d/dt|_0 φ_t^* ω` (axiom)
- Cartan's magic formula theorem olarak çıkar

**B) Cartan tanımı (axiom)**
- `L_X := d ∘ ι_X + ι_X ∘ d` (axiom)
- Cartan's magic formula tautoloji

Paket her iki tanımı destekler; kullanıcı hangisini kullandığını seçer.

#### `calculus/anchor.py`, Anchor
- `ρ: E → TM`, lineer
- Bracket uyumu: `ρ([X,Y]_E) = [ρX, ρY]`

#### `calculus/hamiltonian_vf.py`
- `X_f` iki tanım:
  - Symplectic: `ι_{X_f}ω = -df`
  - Derived: `X_f = -[f, π]_SN`
- Eşdeğerlik ispatı

#### `calculus/exterior_algebra.py`, Dış cebir (YENİ)

`AgreementOnGenerators` stratejisi için cebirin generator yapısını
açıkça bilmek gerek:

```python
class ExteriorAlgebra:
    """Ω*(M), dış cebir"""

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

#### `calculus/cartan.py`, CartanCalculus

- `CartanCalculus(d, L, ι, bracket)`
- Aksiyomlar olarak Cartan relations:
  - `d² = 0`
  - `[L_X, L_Y] = L_{[X,Y]}`
  - `[L_X, ι_Y] = ι_{[X,Y]}`
  - `[d, ι_X] = L_X`
  - `[d, L_X] = 0`
- `verify_all(mode="efficient"|"foundational")`
- `verify(relation, mode=...)`, tek relation
- Farklı calculus varyantları:
  - Standart manifold
  - Lie algebroid (d_E, L_E, ι_E)
  - Twisted (d_H = d + H∧ where H kapalı 3-form)

**Testler:**
- `ω(X_f, X_g) = X_f(g)` zinciri, iki strateji
- Cartan relations standart manifold (her iki modda)
- Cartan relations Lie algebroid calculus'ünde
- Custom calculus tanımı
- Operator-level vs element-level mod karşılaştırması

---

### Faz 7, Proof System

**Amaç:** Paketin kullanıcıya görünen yüzü.

#### `proof/step.py`, ProofStep

- `before`, `after` (Expr)
- `rule_applied`
- `justification`
- `parent`, `children` (nested)
- `provenance_tag`: hangi statüde kullanıldı (axiom/theorem)

#### `proof/chain.py`, ProofChain

- Sıralı adım listesi
- Nesting
- `append`, `extend`, `merge`, `fold`
- Verbosity: full, summary, compact, latex, terminal

#### `proof/tracer.py`

- `TracingAlgorithm(inner, chain)`
- Her çağrıyı kaydet
- Otomatik justification üretimi

#### `proof/expansion.py`, ExpansionEngine

- Tanım kullanarak iç içe açma
- Her açılım bir ProofStep

#### `proof/recognizers.py`

- `CommutatorRecognizer`
- `CyclicSumRecognizer`
- `LeibnizRecognizer`
- `AntisymmetryRecognizer`
- `SchoutenBracketRecognizer`, SN bracket pattern'leri
- `DerivedBracketRecognizer`, bir bracket'ın derived form olduğunu tanır
- `InvariantDerivativeFormulaRecognizer`, `dα(X,Y) = X(α(Y)) - Y(α(X)) - α([X,Y])`

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
- `OperatorLevelProof` vs `ElementLevelProof`, mode toggle

#### `proof/verifier.py`, Yüksek seviye API

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

### Faz 8, Display  *(KAPALI, Stage A + B + C + verbosity + collapsible)*

**Amaç:** LaTeX, terminal, Jupyter çıktısı.

#### Mimari sapma: dispatch vs. `_latex_()` metodu

Plan başlangıçta "`_latex_()` metodu her Expr'da" öngörüyordu; gerçek
uygulamada **MRO tabanlı dispatch fonksiyonları** tercih edildi
(`to_ascii(expr)`, `to_latex(expr)`). Nedeni:

- Core `Expr` hiyerarşisi render şekline bağımsız kalır, `display/`
  paketi olmadan da derlenir/test edilir.
- Yeni render hedefi (HTML collapsible, rich tree) eklerken Expr
  sınıflarına tekrar metod eklemek gerekmez.
- Subclass'lar (örn. `ExteriorDerivative : Derivation`) genel
  `Derivation` handler'ına MRO ile düşer; her subclass'ın kendi
  metodunu kaydetmesine gerek yok.
- Jupyter'ın `_repr_latex_` / `_repr_html_` / `_repr_mimebundle_`
  sözleşmesi `Expr` üzerinde değil, açık opt-in wrapper'larda
  (`LatexDisplay`, `HtmlProofDisplay`) bulunur, test ederken bir
  notebook boot etmek gerekmez.

#### `display/ascii.py`
- MRO dispatch renderer: `to_ascii`, `step_to_ascii`, `chain_to_ascii`.
- Precedence rung'ları + sign normalisation (`Sum(a, Neg(b))` → `a - b`).
- `VERBOSITY_MODES = ("full", "summary", "compact")`, tüm renderer
  katmanlarının paylaştığı sabit.

#### `display/latex.py`
- `to_latex(expr)` dispatch fonksiyonu, Expr'a metod eklenmez.
- `latex_name(...)`: Greek / musical / algebraic glyph translation +
  multi-char subscript bracing (`X_ab` → `X_{ab}`).
- `_escape_text(...)`: rule/justification metinlerinde hem ASCII
  özel karakterleri (`\_#%&$`) hem de Unicode glyph'leri
  `\ensuremath{...}` ile sarar, pdfLaTeX Unicode hatası engellenir.
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
  - `LatexDisplay`, inline `$…$` veya `\begin{align*}…\end{align*}`
    payload; `_repr_latex_` / `_repr_html_` / `_repr_mimebundle_`
    sözleşmesini yerine getirir.
  - `HtmlProofDisplay`, collapsible proof tree (yalnız `text/html`).
- Helpers:
  - `display_expr`, `display_step`, `display_chain` (=
    `display_proof`), flat `align*` çıktı, paper-ready.
  - `display_step_collapsible`, `display_chain_collapsible`,
    `<details open>` HTML ağaç; her adım `[rule] (tag) \(before \to
    after\), just` biçiminde MathJax'e bırakılmış matematik içerir.
    `max_depth`, `title`, `verbosity` opsiyonları terminal renderer'ı
    ile aynı semantikte.

**Testler:** 173 display testi (`tests/test_display/`), golden-value
assert'ler, tüm renderer'ların verbosity / tipe hata / Unicode
sanitisation / stdout leak regresyon testleri dâhil.

---

### Faz 9, Library (Hazır Yapılar ve Teorem Kütüphanesi)

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

### Faz 10, Dokümantasyon ve Tutorials

1. **`01_first_steps.md`**, İlk adımlar
   - Expr, Symbol, toplama/çarpma
   - Property atama
   - Basit sadeleştirme

2. **`02_jacobi_identity.md`**, Jacobi gösterme
   - Lie bracket tanımı
   - Graded cebirde Jacobi
   - `prove_jacobi` kullanımı

3. **`03_poisson_geometry.md`**, Poisson geometri
   - Symplectic form kurma
   - Hamiltonian VF
   - `{f,g} = ω(X_f,X_g) = X_f(g)` zinciri (expand + derived modlar)
   - Poisson Jacobi (derived bracket teoremi)
   - `[π,π]_SN = 0` merkezi rolü

4. **`04_lie_algebroid.md`**, Lie algebroid
   - Anchor map
   - Bracket aksiyomları
   - Leibniz kontrolü
   - Algebroid Cartan calculus

5. **`05_cartan_calculus.md`**, Cartan relations
   - Standart manifold (iki modda)
   - Lie algebroid
   - Twisted
   - Cartan's magic formula
   - Operator-level ispat (AgreementOnGenerators)
   - Element-level ispat (p-form açılımı)

6. **`06_custom_bracket.md`**, Kendi bracket'ı
   - Tanım formundan bracket
   - Aksiyom atama
   - Jacobi test

7. **`07_derived_bracket.md`**, Derived bracket birleştirici
   - `{a,b}_Q = [[a,Q],b]` inşası
   - Kosmann-Schwarzbach teoremi
   - Poisson derived, Koszul derived
   - Klasik vs derived eşdeğerlik
   - Courant derived ve H-twist

8. **`08_unified_picture.md`**, Tek teorem, çok sonuç
   - `[π,π]_SN = 0` tek koşulundan hiyerarşi
   - Tek varsayımla çok ispat
   - Pedagojik özet

9. **`09_foundations.md`** (YENİ), Aksiyomdan teoreme
   - `d² = 0` nereden gelir? (Lie bracket Jacobi'sinden)
   - Foundational mod kullanımı
   - Özel aksiyom seti ile çalışma
   - Theorem Book yapısı

Her tutorial için çalıştırılabilir Jupyter notebook.

---

### Faz 11, İleri Özellikler

- Performans: hashing + memoization, pattern indexing
- Konfigürasyon: sign convention seçimi, L_X tanımı seçimi
- Diagnostic: doğrulama başarısızsa hangi aksiyom eksik
- Export: ProofChain → .tex, TikZ diagram
- CLI: `jacopy verify calculus.yaml`

#### Faz 11 ertelemeleri (küçük paketler, talep gelince açılır)

Stage A (Diagnostic) + Stage B (Export) + Stage C (L_X bundle slots)
kapandıktan sonra hâlâ duran üç ince iş, engine genişletmesi gerekmiyor,
kapsam dar. Kullanıcı talebi gelince sırayla alınır.

1. **Flow-mode `L_X` rewrite kuralları.** Şu an engine'de `L_X` sadece
   `definition="cartan"` instance'ı için `d∘ι_X + ι_X∘d` olarak açılıyor.
   `"flow"` mode'da `L_X` primitive gibi davranıyor, Cartan's magic
   formula "teorem olarak çıkar" vaadi engine tarafında doldurulmuş
   değil. İki definition eklenmesi yeter:
   - `LieDerivativeOnZeroFormDefinition`, `L_X(f) → X(f)` when `f` is
     a declared 0-form.
   - `LieDerivativeCommutesWithDDefinition`, `L_X(d ω) → d(L_X ω)`
     (flow axiom: Lie derivative commutes with `d`).
   Bu ikisiyle `AgreementOnGenerators` flow-mode magic formula'yı
   generator'lar üzerinde kapatır. +3-4 definition, +~10 test.

2. **Algebroid `d_E² = 0` / `ι_E² = 0` axiom wiring.** Stage C
   `cartan_magic`'i kapattı; kalan dört relation (`d_squared_zero`,
   `d_lie`, `lie_lie`, `lie_iota`) algebroid tarafında `d_E`/`ι_E`
   axiom'larının engine'e kaydedilmemesinden kapanmıyor.
   `DSquaredZeroDefinition(target=d_E)` ve
   `IotaSquaredZeroDefinition(target=iota_E_factory)` gibi algebroid
   aware kayıtları `LieAlgebroid` içinden default engine'e eklemek
   yeter. Bu da Courant-Dorfman bridge'in foundational unroll'u için
   ön şart.

---

### Faz 12, Intrinsik (koordinatsız) tanımları tanıma *(opsiyonel, kaçıcı)*

**Amaç:** Kullanıcının klasik diferansiyel geometri metinlerinde
gördüğü multilinear / rank-p formül tanımlarını framework düzeyinde
ifade edilebilir ve açılabilir kılmak.

**Motivasyon:** Şu an engine sadece **operatör-seviyesi** tanımları
tanıyor (`L_X := d∘ι_X + ι_X∘d`, `ι_X(df) = X(f)`, `[L_X, L_Y] =
L_{[X,Y]}`, vb.). Kullanıcı ders kitabı standardı olan intrinsik
formülleri, örneğin

$$
(L_X \omega)(Y_1,\dots,Y_p) = X(\omega(Y_1,\dots,Y_p))
  - \sum_{i} \omega(Y_1,\dots,[X,Y_i],\dots,Y_p)
$$

ya da Koszul formülü

$$
(d\omega)(X_0,\dots,X_p) = \sum_i (-1)^i X_i\bigl(\omega(\dots,\hat{X}_i,\dots)\bigr)
  + \sum_{i<j} (-1)^{i+j}\,\omega\bigl([X_i,X_j],\dots,\hat{X}_i,\dots,\hat{X}_j,\dots\bigr)
$$

ya da `(ι_X \omega)(X_1,\dots,X_{p-1}) = \omega(X, X_1,\dots,X_{p-1})`
yazdığında sistemin bunu bir **tanım** olarak alıp operatör-seviyesine
redüksiyonu (veya tersini) ispat zinciri olarak üretmesini istiyor.

Bugünkü çerçeve bunu **yapamıyor**, çünkü `ω(Y_1,…,Y_p)` biçimindeki
rank-p multilinear evaluation için Expr node tipi yok, ve bunu
eklemeden intrinsik formüller sentaks düzeyinde ifade edilemez.

#### Gereken altyapı parçaları

1. **Multilinear evaluation node**, `MultiEval(form, *vector_fields)`.
   Derecesi: `|form| − len(vector_fields)`; argüman sayısı uyuşmazsa
   hata. Cadabra benzeri head + variadic children. Leibniz'in genellemesi
   için graded-antisymmetry bayrağı (p-formlar için argümanlarda
   antisymmetrik).

2. **Skip-index ("hat") semantiği**, `MultiEval` üzerinde
   `omit(index)` operasyonu veya `HatMultiEval(form, args, omit=i)`
   sabit-*i* varyantı. Somut p için döngü açılımı + sembolik p için
   yerleşik "hat iterator", generator olarak tanımlı.

3. **Parametrik (indeksli) sembolik toplam**, `SymbolicSum(index,
   range, body)`. Şu an `Sum` variadic ama "∑ᵢ (-1)ⁱ … " tipinde
   parametrik ifadelere uygun değil. Somut p için `SymbolicSum` →
   `Sum` expansion'ı bir rewrite kuralı. Sembolik p için alternating
   sum cebri (Koszul sign compatibility) ayrı bir mini-calculus.

4. **Intrinsik definition sınıfları** (engine için):
   - `InteriorProductIntrinsicDefinition`,
     `MultiEval(ι_X ω, Y_1,…,Y_{p-1}) → MultiEval(ω, X, Y_1,…,Y_{p-1})`.
     En kolay; ι'nın rank-1 pairing sözleşmesinin rank-p genellemesi.
   - `LieDerivativeIntrinsicDefinition`, yukarıdaki Leibniz formülü.
     Rank-1 halini `lie_iota` relation'ının Leibniz'li versiyonu
     olarak türetmek mümkün; p sembolikse parametrik sum motoru
     gerekir.
   - `ExteriorDIntrinsicDefinition` (Koszul formülü), üç altyapı
     parçasının hepsini aynı anda kullanır: hat notation, parametrik
     sum, `(-1)^i` işaret cebri. En ağır parça.

5. **`IntrinsicFormulaRecognizer`** (`proof/recognizers.py`'ye),
   kullanıcının yazdığı multilinear ifade bir klasik intrinsik tanımın
   şablonunu tutuyorsa tanısın; `prove_equivalence` ile operatör-seviyesi
   tanıma bağlansın. Örneğin kullanıcı Koszul formülünü LHS olarak
   girerse sistem `d` operatör tanımıyla eşdeğerliğini
   `AgreementOnGenerators` + `MultiEval` açılımıyla kapatabilir.

6. **Bivector / multivector evaluation**, `MultiEval` form-on-vectors
   için; **covector-on-bivector** eşi için de aynı makine lazım
   (`π(α, β)`, `[·,·]_SN(α, β)` gibi bilinear contract'lar). Tek
   `MultiEval` node'u symmetry flag'i ile her iki yönü karşılayabilir
   (form = antisymmetric-on-vectors, bivector = antisymmetric-on-covectors).
   Bu, Poisson bracket'i `{f,g} := π(df, dg)` olarak **Expr seviyesinde**
   ifade edip intrinsik olarak açmanın yolu.

7. **`Closed` registry property** (`core/properties.py`'ye),
   `Graded(degree=p)` gibi declarative bir bayrak. Engine'de yeni bir
   `ClosedFormDefinition(registry)` kuralı: `Act(d, ω)` gördüğünde
   `ω`'nın registry'de `Closed` olup olmadığına bakar; öyleyse
   `Integer(0)`'a yazar. Böylece kullanıcı "`dω = 0`" için inline
   `Definition` yazmak zorunda kalmaz, `registry.declare(ω, Closed())`
   yeter. Symplectic form, volume form, H-twist 3-form'u gibi tipik
   sabit closed form'lar için natural API.

8. **Musical compatibility bilinear genişlemesi**, mevcut
   `MusicalCompatibility` axiom'u sadece 1-form seviyesinde
   (`ω^♭ ∘ π^♯ = id`) iş görüyor. Bilinear seviye (`MultiEval` üstünden):
   $\omega(\pi^\sharp\alpha, \pi^\sharp\beta) = \pi(\alpha, \beta)$.
   Bu identity'yi bir ``MusicalCompatibilityBilinearDefinition`` olarak
   eklemek, Poisson bracket'in symplectic formüllü hali
   (`{f,g} = ω(X_f, X_g)`) ile bivector tanımlı hali
   (`{f,g} = π(df, dg)`) arasındaki eşitliği aksiyomsuz kapatır.
   Altyapı olarak `MultiEval` + Sharp/Flat operatörleri birlikte
   olmalı; Musical layer'ın rank-p desteği yeni alt-pass.

9. **`NonDegenerate` property + injectivity dispatch** *(deferred,
   12.C(e) birikene kadar bekletilir)*, `core/properties.py` zaten
   `NonDegenerate`'i bir class adı olarak listeliyor (plan §460-467),
   ama Faz 12 altyapısında karşılığı yok. İki ayrı iş:
   - **Registry property**: `registry.declare(ω, NonDegenerate())` ile
     ω'nın non-degenerate olduğunu declarative biçimde söyleyebilmek.
     `Closed` property'sine (#7) paralel, bir bayrak, bir engine
     kuralı değil.
   - **Injectivity dispatch (yeni engine primitivi)**: "$\iota_Y \omega
     = \iota_{Y'} \omega$ zinciri kapandıysa $Y = Y'$" meta-kuralı.
     Bugünkü engine term-rewriting ($A \to B$) temelli; bu ise
     **equation-closure-under-injectivity**, farklı bir ispat
     primitivi. Blast radius büyük.
   - **Tetikleyici**: `examples/2c.ipynb` pass'i (2026-04-25) bunu
     açıkça ortaya çıkardı, $[X_f, X_g] = X_{\{f,g\}}$ operator-level
     sonucu engine'e indirilemedi, son adım markdown'a yazıldı. Tek
     başına bir notebook için 12.C(e) `prove_hamiltonian_equality`
     wrapper'ı yeterli (aşağıda); #9'un kendisi, non-degeneracy
     paterni başka use-case'lerde (volume form + divergence-free vektör
     alanı, Riemann metric + musical izomorfizma) birikirse açılır.
   - **Kapsam dışı (şimdilik)**: injectivity dispatch'in engine-level
     tasarımı, ayrı pass, kendi planı.

10. **$L_{fX}$ rescaling rule** *(yeni altyapı, `examples/2d.ipynb`
    pass'inden, 2026-04-25)*, engine-level rewrite:
    $\mathcal{L}_{fX}\omega \to f\,\mathcal{L}_X\omega + df\wedge
    \iota_X\omega$ (1-form için $df\wedge\iota_X\omega$, genel rank için
    $df\wedge\iota_X\omega$). Cartan magic'in skalar-ölçeklendirilmiş
    vektör alanı versiyonu; bugünkü `LieDerivative` slot mimarisinin
    (Faz 11 Stage C) doğal genişlemesi. Tetikleyici: 2d'nin A-K1
    aksiyomu **üç farklı geometrik olguyu** ($[α,β]_K$ tanımı + sharp
    $C^\infty$-linearity + $L_{fX}$ açılımı) tek "fold edilmiş" satıra
    sıkıştırıyor; bu kural landing ettiğinde A-K1 saf Koszul tanımına
    iner, sharp linearity #6'ya, $L_{fX}$ ise bu kurala düşer. Blast
    radius dar, yalnızca `LieDerivative` rewrite slot'u, registry
    yüzeyi yok. Sadece 2d için değil, **her skalar-ölçeklendirilmiş
    vektör alanı problemi için** lazım (Hamiltonian transport, Lie
    algebroid morphism testleri, vb.).

11. **`AntiSymmetric` registry property + bivector evaluation rewrite**
    *(yeni altyapı, `examples/2d.ipynb` pass'inden, 2026-04-25)*,
    `core/properties.py`'a yeni bayrak: `registry.declare(π,
    AntiSymmetric())`. Engine-level rewrite: anti-simetrik bir
    `MultiEval`/bivector evaluation gördüğünde
    $\pi(\alpha,\beta) \to -\pi(\beta,\alpha)$ kanonikleştirir (veya
    sıfıra düşürür slot'lar eşitse). #8 (musical compat bilinear) ile
    birleştiğinde 2d'nin **A-π-antisym** aksiyomu
    ($\iota_{X_\eta}(\omega) = -\langle X_\omega,\eta\rangle$) tamamen
    engine'e iner: $\iota_{\pi^\sharp\eta}\omega \xrightarrow{\#8}
    \pi(\eta,\omega) \xrightarrow{\#11} -\pi(\omega,\eta)
    \xrightarrow{\#8} -\langle\pi^\sharp\omega,\eta\rangle$. #8 tek
    başına yetmez, anti-symmetry ayrı bir geometrik olgudur. `Closed`
    property'sine paralel declarative bayrak.

12. **Pairing $C^\infty$-linearity built-in** *(yeni altyapı,
    `examples/2d.ipynb` pass'inden, 2026-04-25)*, `Pairing` node'unun
    intrinsik özelliği: covector slot $C^\infty(M)$-modül lineer.
    Engine'e otomatik kural: `Pairing(α, Product(f, X))` formundaki
    bir node $f \cdot \mathrm{Pairing}(\alpha, X)$ olarak rewrite
    edilir, $f$ 0-form olduğu sürece. 2d'deki **A-pairing** aksiyomu
    ($\langle X_\omega, f\eta\rangle = f\langle X_\omega,\eta\rangle$)
    bu kuralla aksiyomsuz kapanır. Şu anki `Pairing` Expr'ı sadece
    syntactic; bu özellik ona bilinearity semantiği ekler. Blast
    radius dar, sadece `Pairing` evaluation, `Act` veya `Product`'a
    dokunmaz.

#### Somut vs. sembolik p

- **Somut p (örn. p=2, p=3)**, yukarıdaki üç altyapıdan sadece
  `MultiEval` ve `ExteriorDIntrinsicDefinition` açılımı gerekir,
  sembolik toplam mekanizması olmadan döngü açılımıyla biter. Bu
  alt-kümenin tek başına faydası var: kullanıcı 2-formlar, 3-formlar
  üzerinde intrinsik formülleri test edebilir. Faz 12.A olarak
  ayrılabilir, ~orta iş.

- **Sembolik p**, parametrik sum + hat generator + alternating sign
  cebri birlikte. Faz 12.B. Ağır, symbolic_degree.py'nin paralel
  genişlemesi lazım (sembolik indeksli toplam'ın derece hesabı).

#### Ergonomi wrapper'ları, `examples/2a`, `examples/2b` pass'inden

2026-04-25'te symplectic manifold problem-kitabı şıklarını (`examples/2a.ipynb`
$\mathcal{L}_{X_f}\omega = 0$ ve `examples/2b.ipynb` $\{f,g\} = X_f(g)$)
notebook'a döken pass, intrinsik altyapı olmadan da çalışıyor ama
**her problemde aynı iskelet inline `Definition` olarak tekrar
yazılıyor**. Bu yüzden Faz 12 kapsamına aşağıdaki library-layer
ergonomi wrapper'ları da girer, core altyapı değişmeden çalışır,
ama intrinsik altyapı landing ettiğinde doğal olarak onunla
birleşirler:

a. **`SymplecticProblem(omega, functions=(f,g,...), registry=...)`**,
   `SymplecticManifold`'un problem-odaklı kardeşi. Inşa edildiğinde
   engine'e otomatik register eder:
   - `d ω = 0` (ω closed, yukarıdaki #7 property'si sayesinde),
   - her `f_i` için Hamiltonian defining relation (sign convention
     kwarg'ıyla),
   - `X_{f_i}` derivation'larını factory olarak.
   Notebook şablonu: `SymplecticProblem(ω, (f, g))` → `engine` →
   `prove(...)`. Aksiyom sayısı sıfıra iner, problem-özel olmayan
   her şey wrapper'dan gelir.

b. **Sign convention flag**, library'nin default'u
   `ι_{X_f}ω = -df`, ders kitaplarının bir kısmı `+df`. Hem
   `HamiltonianVectorField` hem yeni `SymplecticProblem`
   `sign="+"` / `sign="-"` kwarg'ı taşısın; aksiyom rewrite'ı
   flag'e göre şekil alsın. `examples/2a.ipynb`'de şu an inline
   aksiyom yazılıyor çünkü library'nin eksisi sabit.

c. **`register_hamiltonian_defining_relation(X, f, omega, engine)`**
   helper'ı, `SymplecticProblem` kullanmayan ama tek-atım
   problem çözen kullanıcılar için mini API. Şu an 2a/2b'de her
   fonksiyon için el-yazımı `Definition` sınıfı yazıyoruz; bu
   tek satıra iner.

d. **Poisson bracket Expr-level entegrasyonu**, `library/poisson.py`'deki
   `PoissonBracket` wrapper'ı yüksek seviyede kalıyor; Expr'ın kendi
   içinde `{f,g}` ifadesi yok (2b'de `Symbol("{f,g}")` aliasıyla
   çözdük). #6 (bivector evaluation) + musical genişlemesi landing
   ettiğinde `PoissonBracket.eval(f, g)` bir `MultiEval(π, df, dg)`
   döndürsün; `{f,g} = X_f(g)` zinciri aksiyomsuz kapansın.

e. **`SymplecticProblem.prove_hamiltonian_equality(Y, h)` helper'ı**,
   `examples/2c.ipynb` pass'inden (2026-04-25). Paternin özeti:
   "$Y$ vektör alanı ve $h$ fonksiyonu verildiğinde, $\iota_Y \omega
   = dh$ zincirini engine'de kapat, sonra non-degeneracy ile $Y =
   X_h$ sonucunu transcript'e **cited axiom step** olarak ekle."
   Bu wrapper bugünkü engine'e dokunmadan çalışır, son adımı
   markdown prose yerine kayıtlı bir ProofChain step'i yapar.
   `examples/2c.ipynb`'in §7 markdown bölümü (non-degeneracy +
   (b) cite) bu helper'a iner, callsite bir satıra düşer:
   `sp.prove_hamiltonian_equality(lie_bracket(X_f, X_g), poisson_fg)`.
   Altyapı #9 (gerçek engine-level injectivity dispatch) açılmadan
   da değerli; açılınca helper'ın iç implementation'ı aksiyom
   cite'ını engine-derived reduction'a refactor eder, callsite
   değişmez.

f. **`KoszulProblem(pi, forms, engine=...)` library wrapper'ı**,
   `examples/2d.ipynb` pass'inden (2026-04-25). `SymplecticProblem`'in
   Koszul-bracket kardeşi. İnşa edildiğinde otomatik kaydeder:
   - Koszul defining axiom'u her form çifti için: $[\alpha,\beta]_K =
     \mathcal{L}_{\pi^\sharp\alpha}\beta - \mathcal{L}_{\pi^\sharp
     \beta}\alpha - d\langle\pi^\sharp\alpha,\beta\rangle$,
   - $\pi^\sharp$ $C^\infty$-linearity (altyapı #6 landing ettiyse
     theorem'e iner; o zamana kadar Definition olarak),
   - $\pi$ anti-symmetry (altyapı #11 landing ettiyse declarative
     `AntiSymmetric` flag, yoksa Definition).
   2d'deki 4-aksiyom paterni (A-K1, A-K2, A-pairing, A-π-antisym)
   tek satıra iner: `kp = KoszulProblem(π, (ω, η, f, f*η),
   engine=engine)`. Altyapı #6/#10/#11/#12 hepsi landing ettiğinde
   `kp` yalnızca Koszul defining'i taşır, sharp linearity, $L_{fX}$,
   anti-sym, pairing linearity hepsi engine-derived olur. Callsite
   her iki dünyada aynı.

#### Kapsam dışı

- Koordinat bazlı açılım (local chart, Christoffel sembolleri, vb.)
 , paketin "koordinatsız" felsefesine aykırı.
- Butler-Portugal tarzı index canonicalisation, multilinear
  değerlendirme antisymmetry'si `MultiEval`'ın bayrağıyla sorunsuz
  halledilir, genel index cebrine gerek yok.

#### Neden opsiyonel / kaçıcı

- Mevcut hedef kitlenin (derived bracket / Cartan calculus / Poisson-
  Lie-Courant geometri üzerinde **operatör-seviyesi** ispat) ihtiyacı
  bu genişleme olmadan karşılanıyor. Faz 10'daki 9 tutorial ve Faz 9'un
  8 seeded theorem'i intrinsik formüllere bağlı değil.
- Altyapı genişlemesi core katmanına dokunuyor (yeni Expr node, yeni
  symbolic_degree modu), blast radius büyük, ciddi regression
  yüzeyi. Talep gelmeden başlatılmamalı.
- "Ders kitabı okuyucusuna intrinsik formülü doğrulama aracı" özel
  bir kullanıcı profili; standart araştırma akışında karşılığı zaten
  operatör-seviyesi Cartan relations üzerinden veriliyor.

#### Ne zaman açılır

Aşağıdakilerden biri olduğunda:
1. Kullanıcı intrinsik formül doğrulaması için somut bir pedagoji
   ihtiyacı dile getirdiğinde (ör. "Tutorial 10'da Koszul formülünün
   operatör tanımıyla eşdeğerliği gösterilsin").
2. Bir research workflow'u `MultiEval` olmadan ifade edilemeyecek bir
   yapı gerektirirse (ör. derecesi parametrik algebroid üzerinde
   Cartan formülünü çıkarma).
3. **Problem-kitabı akışı birikmeye başladığında**,
   `examples/2a.ipynb`, `examples/2b.ipynb` bir dizi halinde
   çoğalırsa (2c, 2d, …). Her yeni problem aynı boilerplate'i
   tekrar etmeye başladığında, yukarıdaki **ergonomi wrapper'ları**
   (`SymplecticProblem` + sign flag + auto-register helper) kritik
   eşiği geçer. Wrapper'lar intrinsik altyapı beklemeden **ayrı bir
   mini-pass** olarak da landing edebilir, Faz 12.C. Intrinsik
   altyapı sonra geldiğinde aynı wrapper'ların axiomlarını theorem'e
   indirmek **in-place** refactor olur, callsite'lar değişmez.

O zamana kadar deferral notu bu başlık altında kayıtlı, `memory/`
tarafında ayrı bir "faz12_intrinsic.md" açıldığında cross-ref buraya
atılır.

#### Faz 12 alt-pass'leri özet

- **Faz 12.A**, somut p intrinsik (MultiEval + p=2,3 için
  InteriorProduct / LieDerivative intrinsic definitions).
- **Faz 12.B**, sembolik p (parametrik sum, hat generator,
  ExteriorDIntrinsicDefinition).
- **Faz 12.C**, `examples/2*.ipynb` pass'inden gelen ergonomi
  wrapper'ları (SymplecticProblem + sign flag + Closed property +
  Hamiltonian auto-register + Poisson bracket Expr entegrasyonu).
  İntrinsik altyapı (12.A) beklemeden de **yararlı**, textbook
  şıkkı yazma iş yükünü doğrudan azaltır.

### Faz 13, Derived Bracket Theorem'in makina-seviyesi ispatı *(opsiyonel, ağır)*

**Tetikleyici.** `examples/2f-theo.ipynb` (2026-04-25) Koszul Jacobi
$\Leftrightarrow [π,π]_{SN}=0$ özdeşliğini paketin seeded
`poisson_koszul_jacobi` teoremini cite ederek tek-adım kapatıyor;
Poisson Jacobi $\Leftrightarrow [π,π]_{SN}=0$ için seeded
`poisson_jacobi` (Faz 9 Stage B.1) eşdeğer 1-adım yol sunar.
Pedagojik tamamlayıcı sual: aynı sonuç **paketin teoremi siteden
kullanmadan**, sıfırdan, 27-terim açılımıyla kapatılabilir mi? Bu pass
o sorunun cevabıdır, `2f-deep` (form-level) ve `2g-deep`
(function-level) notebook'ları + altında yatan engine genişlemesi.

**Amaç.** Derived Bracket Theorem'in iki özel hâli için
**engine-derived** ispat:
1. Koszul Jacobi (1-form'lar) ↔ SN öz-bracket, 27-ground-term
   cancellation zinciri, ~80-150 adım (2f-deep, 13.A-D).
2. Poisson Jacobi (fonksiyonlar) ↔ SN öz-bracket, function-level
   cyclic sum, ~30-50 adım (2g-deep, 13.A axiom 2 + 13.C + 13.E).

Sonuç iki notebook olarak iner; teoremleri siteden kullanmazlar.

**Niye Faz 12 değil.** Faz 12 *intrinsik tanımları* tanımak veya
*ergonomi wrapper'ları* için. Bu pass farklı: yeni geometrik
axiom'lar + yeni `Expr` node ($[X,Y]_{VF}$ vector field Lie bracket)
+ yeni engine rule grupları. Blast radius core katmanına dokunuyor
(yeni node, yeni rewrite kategorisi). Dolayısıyla ayrı faz.

#### 4 alt-faz iskeleti

**13.A, Sharp $\mathbb{R}$-linearity + Hamiltonian VF.**

- *Yeni axiom 1, Sharp on Sum:* $\pi^\sharp(A+B+C) = \pi^\sharp A +
  \pi^\sharp B + \pi^\sharp C$. Engine-level rewrite: `Sharp(π)` bir
  `Sum` üzerinde uygulandığında dağıtsın.
- *Yeni axiom 2, Sharp on $df$:* $\pi^\sharp(df)$ Hamiltonian VF
  $X_f$'e iner. Engine-level: `Sharp(π)` bir `Act(d, f)` üzerinde
  uygulandığında named `Derivation` $X_f$'e replace edilsin (factory
  paterni: ad çakışmasını önlemek için pairing-symmetric ad).
- *Test:* 2f-deep probe, 27-shape açılımı 9-shape'ten 18-terim
  ground'a açılır (sharp linearity ile). Birim test seti: ~10 vaka.

**13.B, Pairing $\mathbb{R}$-linearity + Pairing-Lie Leibniz.**

- *Yeni axiom 3, Pairing on Sum:* $\langle X, A+B+C\rangle =
  \langle X,A\rangle + \langle X,B\rangle + \langle X,C\rangle$
  (her iki slot için). 12 #12 (Pairing $C^\infty$-linearity) ile
  uyumlu, ondan farklı: bu $\mathbb{R}$-linearity, slot
  $C^\infty$-modül lineerliğin daha zayıf hâli. Aynı node üzerinde
  ortak rewrite.
- *Yeni axiom 4, Pairing-Lie Leibniz:* $L_X\langle Y,\beta\rangle =
  \langle L_X Y,\beta\rangle + \langle Y, L_X \beta\rangle$. Pairing
  bilinear olarak Lie türevi altında Leibniz uyar; engine bunu
  bilmiyor.
- *Test:* 2f-deep probe, $d\langle X_α, [β,γ]_K\rangle$ türü grup
  iç açılım yapsın, 27-ground şekli netleşsin. Birim test: ~12 vaka.

**13.C, Vector field Lie bracket node + commutator + Lie-Jacobi.**

- *Yeni Expr node:* `LieBracketVF(X, Y)`, vektör alanlarının Lie
  bracket'i, kendisi `Derivation` (graded degree 0, anti-symmetric).
  `jacopy/algebra/lie_bracket_vf.py` içinde tanımlanır.
- *Yeni axiom 5, Operator commutator → VF Lie:*
  $L_X \circ L_Y - L_Y \circ L_X = L_{[X,Y]_{VF}}$. Engine rewrite:
  Sum içinde `Act(L_X, Act(L_Y, ω)) - Act(L_Y, Act(L_X, ω))` paterni
  görüldüğünde `Act(L_{[X,Y]_VF}, ω)`'e indir.
- *Yeni axiom, Lie-Jacobi for VF:* $[X,[Y,Z]_{VF}]_{VF} +
  [Y,[Z,X]_{VF}]_{VF} + [Z,[X,Y]_{VF}]_{VF} = 0$. Cyclic operator
  pattern.
- *Test:* 2f-deep'in 27-ground açılımının iterated-Lie grubu
  cyclic toplamda sıfıra düşsün; residue $L_X \pi$ türü bivector
  türevlerine collapse etsin. Birim test: ~15 vaka.

**13.D, SN-bivector formula + 2f-deep notebook'u kapatma.**

- *Yeni axiom 6, SN expansion on bivector:* $[\pi,\pi]_{SN}$'ın
  evaluator formülü, pairing yoluyla operator residue ile eşleşmesi.
  Ya `SchoutenBracket._try_base_cases`'e bivector self-bracket
  durumu eklenir, ya da inline definition olarak. Tetikte kalan
  cyclic residue'nun bu formüle iner şekilde rewrite edilmesi.
- *2f-deep notebook'u:* 6 axiom yüklenmiş engine üzerinde,
  cyclic Koszul Jacobi sum LHS, $[\pi,\pi]_{SN}$ bivector formülü
  RHS, `ExpandAndSimplify().prove(...)` ile zincirin kapanması.
  Tahmini: 80-150 step.
- *Test:* 2f-deep notebook'u execute edilir; tam-suite (mevcut 1496
  test) yeşil kalır; yeni axiom'lar ~30 birim test.

**13.E, Function-level Poisson Jacobi (2g-deep), opsiyonel paralel pass.**

- *Tetik:* `examples/2g-theo.ipynb` (planlı) Poisson Jacobi
  $\Leftrightarrow [π,π]_{SN}=0$'ı seeded `poisson_jacobi` teoremini
  cite ederek 1-adımda kapatır. 2g-deep aynı sonucu **fonksiyon-level
  cyclic sum**'ı 27-terim üzerinden açarak ispatlar, 2f-deep'in
  function-ikizi.
- *Probe bulgusu (2026-04-25):* Mevcut
  `DerivedBracket(acting_on=Sharp(π))._koszul_expand` 0-form operandlar
  için `Sharp(π)(f)` üretiyor (anlamsız: sharp 1-form'a uygulanır).
  Function-level dispatch yapısal olarak farklı: anchor değil,
  **anchor ∘ d**. Yani $\{f, g\}_\pi := X_f(g)$ where
  $X_f := \pi^\sharp(df)$.
- *2f-deep ile axiom paylaşımı:*
  - **Reused:** 13.A axiom 2 (`Sharp(d(f)) → X_f`), 13.C tamamı
    (`LieBracketVF` Expr node + Op-commutator → VF-Lie + Lie-Jacobi).
  - **Skipped:** 13.B tamamı (Pairing R-lin + Pairing-Lie Leibniz).
    Function chain'de nested 1-form pairing yapısı yok; gereksiz.
  - **Replaced:** 13.D (form-level SN formülü) → function-level
    Hamiltonian morphism failure ile yer değiştirir.
- *Yeni axiom-2g-1, Function-level Poisson defining:*
  $\{f, g\}_\pi \to X_f(g)$. 2g-deep'in inline axiom'u olarak eklenir
  (core'a dokunmaz; 2f-deep zaten core değişikliği gerektirir,
  2g-deep'i saf inline axiom pass'i olarak korumak tercih edilir).
- *Yeni axiom-2g-2, Hamiltonian morphism failure (kalp):*
  $[X_f, X_g]_{VF} - X_{\{f,g\}} = \tfrac12\langle [\pi,\pi]_{SN},
  df \wedge dg\rangle^\sharp$. Bu axiom $\pi$'in Poisson olması
  $\Leftrightarrow$ Hamiltonian map $f \mapsto X_f$'in Lie morphism
  olması içeriğini taşır. 2g-deep zincirinin $[\pi,\pi]_{SN}$'i sokan
  tek noktası budur. Pratik formülasyonu: cyclic toplamda
  $X_a X_b(c) - X_b X_a(c) = X_{\{a,b\}}(c) + \text{SN-pair}(a,b,c)$.
- *2g-deep notebook'u:* 13.A axiom 2 + 13.C tamamı + 13.E iki yeni
  axiom yüklü engine üzerinde, function-level cyclic Jacobi sum LHS,
  $[\pi,\pi]_{SN}$-pairing residue RHS, `ExpandAndSimplify().prove(...)`
  ile zincirin kapanması. Tahmini: 30-50 step (2f-deep'in %30-40'ı).
- *Test:* 2g-deep notebook'u execute edilir; tam-suite yeşil; yeni
  axiom'lar ~15 birim test.
- *Bağımsızlık:* 13.E yalnızca 13.A axiom 2 + 13.C'ye bağlı.
  **2f-deep'i beklemeden 13.A+13.C+13.E olarak kısaltılmış pass
  açılabilir** (tek başına lighter çalışma). Faz 13 narrative'i iki
  notebook'un birlikte oluşu, ama sıralama pratik (hangi pass açılırsa).

#### Toplam

- **Yeni axiom sayısı:** 8 (13.A-D'den 6 + 13.E'den 2: function-level
  Poisson defining + Hamiltonian morphism failure).
- **Yeni Expr node:** 1 (`LieBracketVF`, 13.C; 13.E reuse).
- **Yeni engine rewrite kategorisi:** ~6 (Sharp distribution, Pairing
  distribution, Pairing-Lie Leibniz, Operator commutator collapse,
  SN bivector expansion, function-level Poisson dispatch).
- **Yeni notebook:** `examples/2f-deep.ipynb` + `examples/2g-deep.ipynb`.
- **Tahmini test eklemesi:** ~82 birim test (2f-deep ~67 + 2g-deep ~15).

#### Niye opsiyonel

- Pratik kullanıcı için 2f-theo + 2g-theo (1-adım theorem cite)
  yeterli, sonuç kanıtlanmış teorem, paket onu zaten taşıyor
  (Faz 9 Stage B.1 ve B.3 seeded).
- 2f-deep / 2g-deep'in değer önerisi pedagojik: "engine teoremi
  *yeniden* türetir, derived-bracket evrenselliğini terim cancellation
  seviyesinde gözlemler".
- Blast radius core katmanına dokunuyor (yeni Expr node `LieBracketVF`)
 , talep gelmeden başlatılmaz.

#### Ne zaman açılır

- Kullanıcı 2f-theo / 2g-theo cite-zincirinin yetersizliğini ifade
  ettiğinde, "engine teoremi cite etmesin, ispatlasın" istemi (bu
  pass'in başlangıç tetikleyicisi: 2026-04-25).
- Vinogradov / big-bracket / $T^*[1]M$ büyük cebir ispatlarına
  pedagojik giriş gerekirse, Faz 13 onun engine-katmanlı önçalışması
  olur.
- 2g-deep tek başına (13.A+13.C+13.E) açılabilir, 2f-deep'in 13.B
  yükünü beklemeden function-level pedagojik narrative.

### Faz 14, Tilde Calculus (Koszul-tarafı Cartan operatörleri) *(opsiyonel, multivektör katmanı)*

**Tetikleyici.** 3.1.2 (`examples/3.1.2.ipynb`, 2026-04-25) standart
Cartan'ın 7 ilişkisini 0-form ve 1-form üzerinde kapatıyor. Pedagojik
bir-sonraki adım §3.1.3: aynı kapanışı *Koszul tarafında*,
$\tilde{d}, \tilde{\iota}_\omega, \tilde{\mathcal{L}}_\omega$, yapmak.
Sistem şu anda KoszulBracket + KoszulProblem + SN bracket + Sharp
katmanlarını taşıyor; tilde *operatörleri* henüz Expr seviyesinde
yok. Bu pass o boşluğu kapatır.

**Amaç.** Üç tilde operatörünü Expr olarak tanımlamak, defining
axiom'larını engine kuralı olarak yazmak, mevcut KoszulProblem
wrapper'ını tilde-aware hâle getirmek; ardından §3.1.3'ün hedef
6 Cartan ilişkisinin **engine ile kapanmasını** sağlamak.

**Niye Faz 12/13 değil.** Faz 12 standart Cartan tarafında
*intrinsik* tanımları + ergonomi'yi tamamladı. Faz 13 derived bracket
Jacobi-zincirinin makina-seviyesi ispatı için core'a `LieBracketVF`
node'u soktu. Faz 14 farklı bir eksen: tilde tarafının operatör
katmanı, yeni *operatör tipleri* ve onları açan rewrite kuralları.
Standart `LieDerivative`/`InteriorProduct`/`ExteriorDerivative`
sınıflarına sığmıyor (parametre tipleri farklı: tilde ι forma indeksli,
tilde d bivektör-bağımlı), dolayısıyla ayrı bir submodül.

**Dayanak tanımlar (3.1.3 textbook formülasyonu).** Bir Poisson
bivektörü $\pi$ ve onun anchor'ı $\rho = \pi^\sharp$ verildiğinde:

* $\tilde{\iota}_\omega V := \iota_V \omega$, formla indekslenmiş,
  multivektör operandı; çıktı türü $\iota_V \omega$'nın türü
  (1-vektör + 1-form girdisinde sonuç bir 0-form / fonksiyon).
* $\tilde{d}V := [\pi, V]_{SN}$, Lichnerowicz türevi; multivektör
  $V$'nin SN-derecesini $+1$ artırır.
* $\tilde{\mathcal{L}}_\omega := \tilde{d}\circ\tilde{\iota}_\omega +
  \tilde{\iota}_\omega\circ\tilde{d}$, Cartan magic (Koszul
  tarafının tanımı).

**Hedef 6 Cartan ilişkisi (3.1.3, ispat fazında doğrulanacak).**

| # | İlişki | Tip yorumu |
|---|---|---|
| 1 | $\tilde{\iota}_\omega \tilde{\iota}_\eta + \tilde{\iota}_\eta \tilde{\iota}_\omega = 0$ | tilde-iota anti-commute |
| 2 | $\tilde{\mathcal{L}}_\omega = \tilde{d}\tilde{\iota}_\omega + \tilde{\iota}_\omega \tilde{d}$ | tilde Cartan magic (tanım) |
| 3 | $[\tilde{\mathcal{L}}_\omega, \tilde{\iota}_\eta] = \tilde{\iota}_{[\omega,\eta]_K}$ | tilde komütatör Koszul bracket'e iner |
| 4 | $[\tilde{d}, \tilde{\mathcal{L}}_\omega] = 0$ | tilde-d ve tilde-L commute |
| 5 | $\tilde{d}^2 = 0$ | $\Leftrightarrow [\pi,\pi]_{SN}=0$ Poisson koşulu |
| 6 | $[\tilde{\mathcal{L}}_\omega, \tilde{\mathcal{L}}_\eta] = \tilde{\mathcal{L}}_{[\omega,\eta]_K}$ | tilde-Lie bracket'i Koszul'a iner |

#### Yeniden kullanılan altyapı

| Bileşen | Konum | Rolü |
|---|---|---|
| `sn` SN bracketi | [jacopy/brackets/schouten.py](jacopy/brackets/schouten.py) | $\tilde{d}V = [\pi,V]_{SN}$ açılımı |
| `KoszulBracket` | [jacopy/brackets/koszul.py](jacopy/brackets/koszul.py) | Relasyon (3)/(6) sağ tarafları |
| `KoszulProblem` | [jacopy/library/koszul_problem.py](jacopy/library/koszul_problem.py) | π + sharp + engine bundle (genişletilecek) |
| `Sharp(π)` | [jacopy/calculus/musical.py](jacopy/calculus/musical.py) | $\rho = \pi^\sharp$ anchor |
| `InteriorProduct`, `multi_eval` | calculus/interior + core/multi_eval | Tilde-iota swap aksiyomunun hedefi |
| `Antisymmetric(π)`, registry | core/properties + core/registry | Π'nin antisimetrisi (KoszulProblem auto-declare) |
| `MultiEvalScalarPullDefinition`, `PairingScalarPullDefinition`, `LieRescalingDefinition` | Faz 12.B #6/#10/#12 | C∞-linearity altyapısı (tilde-side de gerekiyor) |

#### 4 alt-faz iskeleti (A → C zorunlu sıra; D ayrı pass)

**14.A, Tilde operatör Expr tipleri (skeleton).**

- *Yeni dosya:* `jacopy/calculus/tilde/operators.py`.
- *Üç `Derivation` subclass:*
  - `TildeInteriorProduct(omega)`, `omega` bir form Expr; `degree =
    -1` (multivektör-derecesi azaltıcı). Hash key'inde `omega` taşır.
    LaTeX: `\tilde{\iota}_{\omega}`.
  - `TildeExteriorDerivative(pi)`, `pi` bir bivektör Expr; `degree =
    +1`. Singleton-per-π via `pi` kimliği. LaTeX: `\tilde{d}` (alt-
    indis isteğe bağlı π adı).
  - `TildeLieDerivative(omega, pi)`, `degree = 0`; iki parametre
    (form + bivektör). LaTeX: `\tilde{\mathcal{L}}_{\omega}`.
- *Factory shortcut'lar:* `tilde_interior(omega)`, `tilde_d(pi)`,
  `tilde_lie(omega, pi)`.
- *Display dispatch:* `display/latex.py` + `display/ascii.py` üç tipi
  de tanır; mevcut iota/d/L dispatch'lerinin yanına eklenir.
- *Submodül ihracı:* `jacopy/calculus/__init__.py`'de tilde
  operatörleri ve factory'ler `__all__` listesinde görünür.
- *Test bütçesi:* ~12 birim test, construction, hash/eq tutarlılığı,
  derece, LaTeX/ASCII render, factory caching.

**14.B, Tilde defining-axiom'ları (engine kuralları).**

- *Yeni dosya:* `jacopy/calculus/tilde/axioms.py`.
- *Üç `Definition`:*

  1. `TildeIotaSwapDefinition`, köprü aksiyom.
     - matches: `Act(TildeInteriorProduct(ω), V)`.
     - rewrite: `Act(InteriorProduct(V), ω)` (V bir 1-vektör
       (`Derivation`) ise; daha yüksek multivektör için
       `multi_eval(ω, V_1, …, V_k)` formuna açılır).
     - Köşe durumu: V bir 0-vektör (skaler fonksiyon) ise sonuç
       `Zero`, tilde-iota'nın yutucu kuralı (registry'de
       `Graded(degree=0)` tanılı operandlar için).

  2. `TildeExteriorDLichnerowiczDefinition(pi)`, Lichnerowicz tanımı.
     - matches: `Act(TildeExteriorDerivative(π), V)`.
     - rewrite: `BracketApply(sn, π, V)`.
     - Engine fix-point'inde `sn.expand` zincirleri devralır
       (mevcut SN base-cases + wedge Leibniz).
     - Scoped to specific π (instance-bound; iki tilde-d kuralı
       birbirine sızmaz).

  3. `TildeLieMagicDefinition(pi)`, tilde Cartan magic.
     - matches: `Act(TildeLieDerivative(ω, π), V)`.
     - rewrite: `Sum(Act(d̃, Act(ι̃_ω, V)), Act(ι̃_ω, Act(d̃, V)))`
       (standart `LieDerivativeCartanDefinition`'ın dual karşılığı).

- *Test bütçesi:* ~18 birim test, her aksiyom için match/no-match
  eksen vakaları, rewrite hedef şekli, iki π-tipli tilde-d karışmaz,
  registry-aware iota swap'in 0-vektör'de Zero üretmesi.

**14.C, KoszulProblem entegrasyonu.**

- *Dosya değişikliği:* [jacopy/library/koszul_problem.py](jacopy/library/koszul_problem.py).
- *KoszulProblem `__init__` genişlemesi:*
  - `multivectors` (opsiyonel) parametresi: SN-graded multivektör
    operandlarının listesi (`(f, X, V, ...)`); auto-declare ederek
    her birine uygun derece koyar (skaler `Graded(degree=0)`,
    1-vektör `Graded(degree=0)` + `Derivation`, 2-vektör
    `Graded(degree=1)` SN-shifted, vb.).
  - Tilde aksiyomlarını engine'e register eder:
    `TildeIotaSwapDefinition`, `TildeExteriorDLichnerowiczDefinition(pi)`,
    `TildeLieMagicDefinition(pi)`.
  - 14.D'de eklenecek auxiliary 0-vektör/1-vektör closure aksiyomları
    da varsayılan olarak register edilir (D bağımsız landed olduğunda).
- *Yeni metodlar:*
  - `tilde_d() -> TildeExteriorDerivative`, π-bound singleton.
  - `tilde_interior(omega) -> TildeInteriorProduct`, caller-supplied
    1-form üzerinden factory.
  - `tilde_lie(omega) -> TildeLieDerivative`, caller-supplied 1-form
    + π üzerinden factory.
  - `assume_poisson() -> None`, `[π,π]_{SN} = 0` aksiyomunu
    declarative bayrak olarak işaretler (yeni `Poisson(π)` registry
    property; ilgili engine kuralı 14.D'de). Idempotent.
- *Test bütçesi:* ~15 birim test, auto-declare, factory'lerin
  doğru registry/π wire'ı, `assume_poisson` idempotency, iki ayrı
  KoszulProblem birbirinden bağımsız.

**14.D, İspat-için-gerekli auxiliary aksiyomlar (3.1.3 kapanışı).**

3.1.3'ün 6 ilişkisini bir 0-vektör $f$ ve bir 1-vektör $X$ üzerinde
test ederken, 14.A-C'nin üç-aksiyomu yetmez. Aşağıdaki 5 ek aksiyom
gereklidir; her biri standart Cartan tarafının doğal karşılığıdır.

  *Aux-1.* `TildeIotaOnZeroVectorDefinition`, $\tilde{\iota}_\omega f
  = 0$ (f registry'de `Graded(degree=0)`).
  - 14.B'deki `TildeIotaSwapDefinition`'ın "0-vektör girdisinde Zero"
    köşe durumunu *bağımsız* bir kural olarak ayrıştırır (engine
    log'unda görünür kalsın).

  *Aux-2.* `TildeIotaSquaredZeroDefinition`, $\tilde{\iota}_\omega^2
  = 0$ on multivectors (relasyon (1)'in $\eta = \omega$ özel hâli).
  - matches: `Act(ι̃_ω, Act(ι̃_ω, V))` → `Zero`.
  - Standart `IotaSquaredZeroDefinition` paterninin tilde-aynası.

  *Aux-3.* `TildeLieOnZeroVectorDefinition`, $\tilde{\mathcal{L}}_\omega
  f$ skalere iner.
  - rewrite: `Act(Sharp(π), ω)(f)` türünde → registry'deki
    Hamiltonian VF $X_\omega$'a eşleşirse o; aksi hâlde
    `Act(Sharp(π)·ω, df)` kanonik şeklinde tutar.
  - Bu, 14.B magic kuralının fix-point sonrasında `Sum(d̃(ι̃_ω(f)),
    ι̃_ω(d̃(f))) = Sum(0, ι̃_ω(-X_f)) = ω(X_f)` zincirini biriktirir;
    auxiliary kural yalnızca son skalere collapse adımını ekler.

  *Aux-4.* `TildeDOfFunctionDefinition`, $\tilde{d}f = -X_f$
  (Hamiltonian VF). Mevcut SN base-cases'in tilde-aware aynası;
  `TildeExteriorDLichnerowiczDefinition` zaten `[π, f]_{SN}` üretiyor,
  ama SN base case 2 ($[f,X]_{SN} = -X(f)$) doğrudan eşleşmiyor,
  `[π, f]_{SN}` SN-Leibniz ile $-X_f$'e iner; o zincirde Sharp +
  exterior-d kuralları lazım. Bu aksiyom kestirme ekler:
  doğrudan `Act(d̃, f) → Neg(Act(Sharp(π), Act(d, f)))`. Engine
  derinliğini ~5 adım keser.

  *Aux-5.* `TildeDSquaredPoissonDefinition`, $\tilde{d}^2 = 0$ aksiyomu
  (registry'de `Poisson(π)` tanılı).
  - matches: `Act(d̃, Act(d̃, V))` AND `registry.has(π, Poisson)`.
  - rewrite: `Zero`.
  - 14.C `assume_poisson()` ile bayrak yakılır; bu aksiyom o bayrağı
    tüketir. Relasyon (5)'in tek adımda kapanmasını sağlar.

- *Test bütçesi:* ~25 birim test, 5 auxiliary'nin her biri için
  match/no-match + rewrite hedefi + registry-aware fall-through;
  bayrak yokluğunda Aux-5 no-op.

#### Stage E, 6 Cartan ilişkisinin engine ile kapanması (sonraki pass)

14.A-D landed olduktan sonra, 3.1.3 ispatları **mevcut**
`prove_intrinsic_equivalence` API'siyle yazılır (notebook bu plan
kapsamı dışı; ayrı pedagojik pass). Her ilişkinin beklenen kapanış
maliyeti:

| # | Engine adımı tahmini (0-vektör + 1-vektör) | Yer-tutucu darboğaz |
|---|---|---|
| 1 | 1 + 3 | Aux-2 (anti-commute reduce) |
| 2 | 0 (tanım gereği) + 0 | tilde magic *tanım*, ispat değil, `cartan_obstruction`-tipi 0-residue testi yeterli |
| 3 | 5 + 8-12 | Cartan-magic + Koszul expansion + Aux-1/3 |
| 4 | 6 + 10-15 | Aux-3 + Aux-4 + Sharp linearity |
| 5 | 1 + 1 | Aux-5 (Poisson bayrağı tüketir) |
| 6 | 8-12 + 15-20 | İki tilde-magic + Koszul expansion + cyclic SN-Leibniz |

Stage E sırasında çıkacak muhtemel ek ihtiyaçlar (ön-keşif, plan
yapısı içinde gelişigüzel artırılmaması için listelenir):

- **Sharp(π)·ω = X_ω hizalaması**: relasyon (3)/(4)'te
  $\tilde{\iota}_{[\omega,\eta]_K}V$ açılımı `[ω,η]_K`'yı Cartan
  formülüne açar; engine'in fix-point'inde sonuç `Sharp(π)`'a
  uygulanmış sum'a iner. Mevcut `SharpLinearityDefinition` +
  `SharpOnExactDefinition` (Faz 13.A) yeter; ek aksiyom
  beklenmiyor, ama keşif sırasında darboğaz bulunursa
  `tilde/aux_axioms.py`'a Aux-6 olarak eklenir.
- **SN-Leibniz çift sarımı**: relasyon (6)'da
  `[π, [π, X]_{SN}]_{SN}` yapısı görünür; mevcut SN expand'in
  `Antisymmetric(π)` + Aux-5 (Poisson) etkileşimi gerek; engine
  sırasının doğru olması (önce SN açılımı, sonra
  registry-canonicalize) `KoszulProblem`'in mevcut
  registration sırasıyla uyumlu, ama Stage E'de doğrulanır.
- **Cyclic SN-Jacobi**: relasyon (6)'nın derinliğine bağlı olarak
  Faz 13.D'deki `SnBivectorFormulaDefinition`-tipi cyclic bir
  formül-aksiyomu gerekebilir; eğer çıkarsa Aux-7 olarak landed.

Stage E'nin her ilişkisi için ölçüm: `prove_intrinsic_equivalence`
adım sayısı + zincirin SN/Sharp/Koszul aksiyom dağılımı. Tablodaki
tahminler aşılırsa hangi auxiliary'in eksik olduğu tespit edilir;
14.D listesi büyütülebilir.

#### Toplam

- **Yeni Expr tipi:** 3 (`TildeInteriorProduct`,
  `TildeExteriorDerivative`, `TildeLieDerivative`).
- **Yeni axiom sayısı:** 8 (14.B'den 3 + 14.D'den 5).
- **Yeni registry property:** 1 (`Poisson(π)`; standart
  `Antisymmetric(π)` yanına bayrak).
- **Yeni library API:** `KoszulProblem.tilde_d / tilde_interior /
  tilde_lie / assume_poisson` + `multivectors` constructor parametresi.
- **Yeni submodül:** `jacopy/calculus/tilde/{operators.py, axioms.py,
  aux_axioms.py}`.
- **Tahmini test eklemesi:** ~70 birim test (14.A ~12 + 14.B ~18 +
  14.C ~15 + 14.D ~25).
- **Stage E (ayrı pass):** 6 ilişki ispatı, tahmini ~50-80 toplam
  engine adımı, ek test ~30-50.

#### Niye opsiyonel

- Standart Cartan tarafı 3.1.2 ile zaten kapalı; tilde tarafı
  pedagojik tamamlayıcı (Lie algebroid $T^*M$ örneğinin
  Cartan-katmanı). Pratik kullanıcı bu pass olmadan da Poisson +
  Koszul + SN'i tam kullanabilir (Faz 9 Stage B + Faz 12.C).
- Blast radius görece kontrollü: yeni submodül, yeni 3 Expr tipi,
  KoszulProblem'in genişlemesi; *core* katmanına dokunmaz.
- Yine de yeni Expr tipi olduğu için "talep gelmeden açılmaz"
  sınıfında, Faz 13 ile aynı disiplin.

#### Ne zaman açılır

- Kullanıcı 3.1.3 (Koszul tarafı Cartan ilişkileri) ispatlarını
  istediğinde.
- Lie algebroid intrinsik Koszul-engine'i (anchor-Koszul rule
  parametrizasyonu) talep edildiğinde, Faz 14 bu pass'in operatör
  katmanlı önçalışmasıdır; sonraki adım `intrinsic_axioms.py`'ı
  anchor + algebroid bracket parametresine açmaktır (ayrı bir
  Faz, talep gelirse).

### Faz 16, Bianchi kimlikleri (afin bağlantı ∇) *(opsiyonel, yeni operatör katmanı)*

**Tetikleyici.** §3.1.5 derivator identities (Faz 15.C) §3.1.6'a
geçişin önünü açtı: aynı pedagojik kalıbın bir sonraki örneği bir
afin bağlantı $\nabla$ üzerindeki iki Bianchi kimliği:

- **Bianchi I:**
  $\operatorname{cycl}_{U,V,W} R(\nabla)(U,V)W
   = \operatorname{cycl}_{U,V,W}\bigl[(\nabla_U T(\nabla))(V,W)
   + T(\nabla)(T(\nabla)(U,V), W)\bigr]$
- **Bianchi II:**
  $\operatorname{cycl}_{U,V,W} (\nabla_U R(\nabla))(V,W)W'
   = \operatorname{cycl}_{U,V,W} R(\nabla)(U, T(\nabla)(V,W))W'$

Burada $T(\nabla)(X,Y) = \nabla_X Y - \nabla_Y X - [X,Y]_{VF}$ ve
$R(\nabla)(X,Y)Z = \nabla_X\nabla_Y Z - \nabla_Y\nabla_X Z -
\nabla_{[X,Y]_{VF}} Z$. `LieBracketVF` Faz 13.C'den geliyor;
**`AffineConnection`, `Torsion`, `Curvature` katmanı yok**, bu pass
o boşluğu kapatır.

**Amaç.** İki Bianchi kimliğinin **engine ile kapanması**. Bianchi I
$\sim$ 80-150 adım (LBVF Jacobi cyclic kapatır), Bianchi II $\sim$
100-180 adım (∇-on-curvature Leibniz açılımı + cyclic). Sonuç bir
notebook (`examples/3.1.6.ipynb`), §3.1.5 stilinde
`display_chain` rendering'iyle PDF'e iner.

**Niye Faz 14 / 15 değil.** Faz 14 tilde-Cartan operatörleri
(Koszul tarafı), Faz 15.C derivator-on-Koszul-bracket §3.1.5
identities. Bianchi farklı bir eksen: **afin bağlantı katmanı**,
ne SN-derivator ne tilde-Cartan; tamamen yeni operatör tipleri ve
yeni axiom ailesi. Cartan / SN / Koszul / tilde altyapısının hiçbiri
∇'yı tanımıyor.

**Mimari karar (en temiz yol).** $T(\nabla)$ ve $R(\nabla)$
**MultiEval-tabanlı operatör atomları** olarak modellenir
(`Torsion(nabla)` 2-arity, `Curvature(nabla)` 3-arity). Bu sayede
Faz 15.C'nin `AtomSlotLift` pre-pass borcuna girilmez,
`operator_atom_index_opacity.md` deferred sorunu burada baştan
yoktur. Slotlar engine-walkable, antisimetri / Sum-linearity /
Neg-linearity ailesi bedavadan üzerine oturur.

#### Yeniden kullanılan altyapı

| Bileşen | Konum | Rolü |
|---|---|---|
| `LieBracketVF` Expr | [jacopy/algebra/lie_bracket_vf.py](jacopy/algebra/lie_bracket_vf.py) | $[X,Y]_{VF}$ T tanımında ve LBVF Jacobi'de |
| LBVF Jacobi axiom | [jacopy/calculus/sn_function_axiom.py](jacopy/calculus/sn_function_axiom.py) | Faz 13.C, Bianchi I cyclic kapanışı |
| `Sum`/`Neg`/`MultiEval` | core | T/R MultiEval şekli + cyclic 3-terim Sum |
| `collect_terms` | proof/simplify | Cyclic pair-cancel |
| `IntrinsicEngine` factory | [jacopy/calculus/intrinsic_engine.py](jacopy/calculus/intrinsic_engine.py) | Bianchi engine bundle (KoszulProblem analogu) |
| `display_chain` + LaTeX `gather*` | [jacopy/display/jupyter.py](jacopy/display/jupyter.py) | §3.1.5'te düzelmiş PDF rendering, birebir reuse |

#### 5 alt-faz iskeleti (16.A → 16.D zorunlu sıra; 16.E paralel pass)

**16.A, AffineConnection Expr + ∇ defining axiom'ları.**

- *Yeni dosya:* `jacopy/calculus/connection.py`.
- *Yeni Expr tipi:* `AffineConnectionAct(nabla, X, Y)`, `nabla`
  bağlantı sembolü (`Symbol`), `X` ikinci slot (Derivation), `Y`
  üçüncü slot (Derivation veya tensor argümanı). Kendisi
  `Derivation`-benzeri (degree 0); X-slot'unda C∞-lineer, Y-slot'unda
  Leibniz uyar. LaTeX: `\nabla_{X} Y`.
- *Factory:* `connection("∇")` → `AffineConnection` wrapper, içinde
  `act(X, Y)` çağrısı `AffineConnectionAct` üretir.
- *Engine kuralları (yeni axiom'lar):*
  1. **`ConnectionXLinearityDefinition`**, $\nabla_{fX+gY} Z = f
     \nabla_X Z + g \nabla_Y Z$ (X-slot C∞-lineer). Sum + scalar-
     pull paterni Faz 12.B #6 ile uyumlu, MultiEval-üstü reuse.
  2. **`ConnectionYLeibnizDefinition`**, $\nabla_X (fY) = X(f) Y +
     f \nabla_X Y$ (Y-slot Leibniz). `Act(X, f)` + `f \nabla_X Y`
     iki-terim Sum'a açılır.
  3. **`ConnectionYAdditivityDefinition`**, $\nabla_X (Y+Z) =
     \nabla_X Y + \nabla_X Z$ (Y-slot Sum-linearity).
- *Display dispatch:* `display/latex.py` + `display/ascii.py`
  `AffineConnectionAct`'ı tanır.
- *Test bütçesi:* ~14 birim test, construction, hash/eq, üç axiom
  fire kuralı, slot tipleri.

**16.B, Torsion + Curvature MultiEval atomları + tanım/antisimetri.**

- *Yeni dosya:* `jacopy/calculus/torsion_curvature.py`.
- *İki MultiEval-tabanlı operatör atomu:*
  - `TorsionOp(nabla)`, 2-arity. `MultiEval(TorsionOp(nabla),
    [X, Y])`. LaTeX: `T(\nabla)(X, Y)`.
  - `CurvatureOp(nabla)`, 3-arity. `MultiEval(CurvatureOp(nabla),
    [X, Y, Z])`. LaTeX: `R(\nabla)(X, Y)Z`.
- *Engine kuralları (yeni axiom'lar):*
  4. **`TorsionDefinitionDefinition`**, `MultiEval(TorsionOp(∇),
     [X,Y]) → ∇_X Y - ∇_Y X - [X,Y]_VF`. Üç-terim Sum (iki Neg).
  5. **`CurvatureDefinitionDefinition`**, `MultiEval(CurvatureOp(∇),
     [X,Y,Z]) → ∇_X(∇_Y Z) - ∇_Y(∇_X Z) - ∇_{[X,Y]_VF} Z`.
     Üç-terim Sum.
  6. **`TorsionAntisymmetryDefinition`**, `T(∇)(X,Y) → -T(∇)(Y,X)`.
     MultiEval-bilinear-arg flip. (Tanım aksiyomundan türetilebilir
     ama doğrudan vermek 16.D'de cyclic kapanışı kısaltır.)
  7. **`CurvatureFirstSlotAntisymmetryDefinition`**, `R(∇)(X,Y)Z →
     -R(∇)(Y,X)Z`. Aynı şekilde shortcut.
- *Test bütçesi:* ~16 birim test, iki tanım fire kuralı, iki
  antisimetri, MultiEval slot opacity yokluğunun verifikasyonu
  (engine T/R'nin iç slotlarına doğal yürüyebiliyor mu).

**16.C, ∇-on-tensor Leibniz axiom'ları.**

- *Aynı dosya:* `torsion_curvature.py` (devam).
- *Engine kuralları (yeni axiom'lar):*
  8. **`ConnectionOnTorsionLeibnizDefinition`**, $(\nabla_U T)(V,W)
     = \nabla_U(T(V,W)) - T(\nabla_U V, W) - T(V, \nabla_U W)$.
     `Act(∇_U, MultiEval(T,[V,W]))` paterni → 3-terim Sum'a açılır.
     Bu Bianchi I'in sağ tarafının ana açılım kuralı.
  9. **`ConnectionOnCurvatureLeibnizDefinition`**, $(\nabla_U R)
     (V,W)W' = \nabla_U(R(V,W)W') - R(\nabla_U V, W)W' -
     R(V, \nabla_U W)W' - R(V,W)(\nabla_U W')$. 4-terim Sum
     (üçüncü argüman dahil tensor-ekstensiyon). Bu Bianchi II'nin
     sağ tarafının ana açılım kuralı.
- *Test bütçesi:* ~10 birim test, iki Leibniz fire kuralı, slot
  tipleri (1-vektör + multi_eval arg), Sum açılımı doğru sayıda
  terim üretiyor mu.

**16.D, `cyclic_sum` helper + `BianchiProblem` + Bianchi I notebook.**

- *Yeni dosya:* `jacopy/library/bianchi_problem.py`.
- *Helper:* `cyclic_sum(expr, args=[U,V,W]) → Sum`.
  Argümanların 3 cyclic permütasyonunu (U→V→W→U) `expr`'e yerleştirip
  3-terim Sum üretir. Saf expression-builder, axiom değil.
- *Wrapper:* `BianchiProblem(nabla)`, `AffineConnection` taşır,
  engine bundle olarak 16.A-C'nin 9 axiom'unu + LBVF Jacobi'yi +
  collect_terms'i taşır. Metot:
  - `prove_first_bianchi(U, V, W) → ProofChain`. Lhs
    `cyclic_sum(R(∇)(U,V)W, [U,V,W])`; rhs
    `cyclic_sum(∇_U T(∇)(V,W) + T(∇)(T(∇)(U,V),W), [U,V,W])`.
    Strateji: lhs'i Curvature-tanımıyla aç → 9-terim
    Sum (3 cyclic × 3-terim); cyclic permütasyon altında
    $\nabla_X\nabla_Y Z$ çiftleri tek $\nabla_U(...)$ ve T
    terimlerine grupanır; LBVF Jacobi cyclic toplamda
    $-\nabla_{[U,V]_{VF}} W + \text{cycl}$ kalanını yutar.
- *Notebook:* `examples/3.1.6.ipynb`, §3.1.5 birebir kalıbı:
  setup hücresi, Bianchi I bölümü, `display_chain(chain)`
  rendering. PDF §3.1.5 düzeltmeleriyle (gather* + scriptsize +
  allowdisplaybreaks) sorunsuz çıkmalı.
- *Test bütçesi:* ~14 birim test, `cyclic_sum` correctness,
  `BianchiProblem.prove_first_bianchi` ProofChain hash kararlılığı,
  notebook execute.

**16.E, Bianchi II notebook *(paralel pass; 16.D'den bağımsız)*.**

- *Aynı `BianchiProblem` üzerinde:*
  `prove_second_bianchi(U, V, W, Wp) → ProofChain`. Lhs
  `cyclic_sum((∇_U R(∇))(V,W)Wp, [U,V,W])`; rhs
  `cyclic_sum(R(∇)(U, T(∇)(V,W))Wp, [U,V,W])`.
  Strateji: lhs'i 16.C axiom 9 (∇-on-Curvature Leibniz) ile aç →
  cyclic toplamda $\nabla_U \nabla_X (...)$ çiftleri yine LBVF Jacobi
  ile çöker; Torsion-tanımı (16.B axiom 4) sağ taraftaki
  $T(V,W)$'yi açar ve $\nabla_{V}\nabla_W - \nabla_W\nabla_V -
  \nabla_{[V,W]}$ ile R'ye yeniden grupanır.
- *Notebook:* `examples/3.1.6.ipynb`'a ikinci bölüm olarak eklenir
  (veya `3.1.6_b.ipynb`, §3.1.5'in iki notebook'lu örüntüsünden
  bağımsız tek-notebook tercihi).
- *Test bütçesi:* ~10 birim test, sadece `prove_second_bianchi`
  ProofChain kararlılığı + notebook execute.
- *Bağımsızlık:* 16.E sadece 16.A+B+C'ye bağlı, 16.D'siz
  çalıştırılabilir; ama Bianchi I'i kapatmadan açmak pedagojik
  olarak ters.

#### Toplam

- **Yeni axiom sayısı:** 9 (16.A: 3, 16.B: 4, 16.C: 2).
- **Yeni Expr tipi:** 1 (`AffineConnectionAct`); **2 yeni operatör
  atomu** (`TorsionOp`, `CurvatureOp`, MultiEval'a oturan).
- **Yeni engine bundle:** `BianchiProblem` (KoszulProblem analogu).
- **Yeni notebook:** `examples/3.1.6.ipynb`.
- **Tahmini test eklemesi:** ~64 birim test (16.A: 14, 16.B: 16,
  16.C: 10, 16.D: 14, 16.E: 10).
- **Tahmini ispat zincir uzunluğu:** Bianchi I 80-150 step,
  Bianchi II 100-180 step.

#### Niye opsiyonel

- Mevcut paket Cartan + SN + Koszul + tilde + derivator katmanlarını
  taşıyor; afin bağlantı bunlardan farklı bir geometrik objedir,
  Poisson/multivektör hattının doğal devamı değil.
- Pratik kullanıcı simplektik / Poisson / Lie algebroid hattında
  bu pass olmadan tam donanımlı.
- Blast radius: yeni Expr + 2 operatör atomu + 9 axiom + 1 wrapper
 , orta. Faz 13'e benzer büyüklük.

#### Ne zaman açılır

- §3.1.5 derivator pass'inden sonra §3.1.6 (Bianchi) talep
  edildiğinde, bu pass'in tetikleyicisi.
- Riemann / Cartan geometri tarafına pedagojik açılım istendiğinde.
- Bianchi I tek başına (16.A+B+C+D) Bianchi II'siz açılabilir;
  16.E'nin değeri tamamlayıcı.

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
Faz 5 (brackets + derived bracket)    ║    Faz 8 (display), paralel
  │
  ▼
Faz 6 (calculus + exterior algebra + operator equations)
  │
  ▼
Faz 7 (proof system, tam)
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
from jacopy import *

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
# {f,g} = ω(X_f, X_g) = X_f(g), iki mod
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

# Foundational mod, d² = 0 bile türetilsin
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
   teoremi ile türetilir, kısa ve yapısal ispat
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
- `rich`, zengin terminal çıktısı
- `pytest`, test
- `mkdocs` veya `sphinx`, docs
- `hypothesis`, property-based testler

**Kesinlikle yok:**
- SymPy, SageMath, Cadabra, bağımsız paket hedefi

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

1. **Faz 0 tamamla**, `pyproject.toml`, dizin, pytest
2. **Faz 1'in ilk yarısı**, `core/expr.py`:
   - `Expr` base
   - `Symbol`, `Sum`, `Product`, `Zero`, `One`
   - Operatör overloading
   - Basit `__repr__`
3. **İlk testler**, structural equality, property atama skeleton
4. **Provenance iskeleti**, `Property` sınıfının `status`, `dependencies`,
   `proof` alanlarının taslağı (doldurulacak)

Bu dört adım bittiğinde paket yaşamaya başlar ve üzerine her şey
yapılandırılır.

---

Plan canlı bir dokümandır. Geliştirme sırasında kararlar değişirse
buraya yansıtılır.
