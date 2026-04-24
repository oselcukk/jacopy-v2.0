# 05 — Cartan Calculus

Diferansiyel geometrinin aritmetik çekirdeği beş bağıntıya indirger:

1. `d² = 0` — exterior türevin karesi
2. `[d, ι_X] = L_X` — Cartan magic formülü
3. `[d, L_X] = 0`
4. `[L_X, L_Y] = L_{[X, Y]}`
5. `[L_X, ι_Y] = ι_{[X, Y]}`

`gradalg` bu beşliyi `CartanCalculus(d, lie_derivative, interior, vector_bracket)`
bundle'ı etrafında toplar ve her birini bir `OperatorEquation` olarak
verir. Bu tutorial (a) bağıntıları *operatör denklemi* olarak nasıl
okunacağını, (b) hangi bağıntıların hangi mod'da (efficient vs
foundational) canlı ispatla kapandığını, ve (c) türeten bir teorem
olarak `invariant-d` formülünü gösteriyor.

[04 — Lie algebroid](04_lie_algebroid.md) aynı API'yi bundle etiketli
operatörlerle çalıştırıyor; buradaki canlı ispatlar `TM` üstünde.

## Bundle

```python
from gradalg.algebra.derivation import Derivation
from gradalg.brackets.lie import LieBracket
from gradalg.calculus.cartan import CartanCalculus, RELATIONS
from gradalg.calculus.exterior_algebra import ExteriorAlgebra
from gradalg.calculus.exterior_d import d
from gradalg.calculus.interior import interior
from gradalg.calculus.lie_derivative import lie_derivative
from gradalg.core.expr import Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry

cart = CartanCalculus(
    d=d,
    lie_derivative=lie_derivative,
    interior=interior,
    vector_bracket=LieBracket(),
)
RELATIONS
# ('d_squared_zero', 'cartan_magic', 'd_lie', 'lie_lie', 'lie_iota')
```

## Beş bağıntı, beş `OperatorEquation`

`cart.relation(name, X=..., Y=..., algebra=...)` her bir bağıntının
operatör-seviyesi denklemini döner. Bağıntıya göre hangi parametrelerin
gerektiği farklı — `d²` hiçbir şey istemez, `magic` / `d_lie` tek `X`,
`lie_lie` / `lie_iota` ikisini de:

```python
reg = PropertyRegistry()
f = Symbol("f")
reg.declare(f, Graded(degree=0))
algebra = ExteriorAlgebra((f,))
X = Derivation("X", degree=0)
Y = Derivation("Y", degree=0)

cart.relation("d_squared_zero", algebra=algebra)
# (d * d) = 0

cart.relation("cartan_magic", X=X, algebra=algebra)
# ((d * ι_X) + (ι_X * d)) = L_X

cart.relation("lie_lie", X=X, Y=Y, algebra=algebra)
# ((L_X * L_Y) + (-(L_Y * L_X))) = L_([X, Y])
```

## `d² = 0` — axiom mode vs theorem mode

`d²` default engine rewrite'ı iki sınıflama taşır: `axiom` (varsayılan,
doğrudan aksiyom olarak yazar) ve `theorem` (generator-seviyesi kanıtı
foundational modda açar). `default_engine(..., d_squared_mode=...)`
ile seçilir:

```python
from gradalg.calculus.exterior_d import apply_d_squared_zero
from gradalg.proof.expansion import default_engine

x = Symbol("x")
reg.declare(x, Graded(degree=0))

# Sade axiom rewrite — calculation helper
apply_d_squared_zero(d(d(x)))       # 0

# Theorem-mode expansion (foundational)
engine = default_engine(registry=reg, mode="foundational", d_squared_mode="theorem")
expanded, steps = engine.expand(d(d(x)))
expanded                            # 0
steps                               # [ProofStep(rule='d² = 0', d(d(x)) → 0)]
```

`theorem` sınıflamasında sub-proof `d(df) = 0` aksiyomunu 0-formlar
üstünde generator-seviyesinden türetir; bu, operatör-seviyesi
`d ∘ d = 0`'ın generator'ler üstünde anlaşmayla genelleştiği
`AgreementOnGenerators` stratejisinin tam yoludur.

## Cartan magic — `verify` üzerinden canlı ispat

`cartan_magic` şu an `CartanCalculus.verify(...)` üstünden `ExteriorAlgebra`
bir fonksiyon generator'ı ile iki modda da net kapanıyor:

```python
chain = cart.verify("cartan_magic", algebra=algebra, X=X, registry=reg)
len(chain)                # 1
chain.steps[0].rule       # definition citation

chain_f = cart.verify(
    "cartan_magic",
    algebra=algebra,
    X=X,
    registry=reg,
    mode="foundational",
)
len(chain_f)              # 1 — UnrollToFoundations sarmalıyor
```

Magic formülü `LieDerivativeCartanDefinition`'ın default classification'ı
gereği tek adımlık teorem olarak tetikleniyor; foundational mod aynı
sub-proof'u citation halinde tutuyor. Algebroid variant için
bak: [04_lie_algebroid.md](04_lie_algebroid.md) — o tarafta
`verify`'ın bundle-etiketli operatörlerle tetiklenememesi bilinen bir
deferral.

## `d_lie`, `lie_lie`, `lie_iota` — inşa hazır, verify henüz kapsanmıyor

Üç parametrik bağıntının `OperatorEquation`'ı `relation()` ile sağlam
kuruluyor ama `verify()` bugünün baseline'ında iki nedenle kapanmıyor:

- `d_lie` için `lhs`/`rhs` derecelerinin otomatik anlaşılması gereken
  sembolik `Derivation` operatör vakaları;
- `lie_lie` / `lie_iota` için `L_Y`, `ι_Y` faktörlerinin kaydedilmiş
  bir grading beklemesi — default `AgreementOnGenerators`
  normalizasyonu şu an açık-ifadeli bu objeler için yapılmıyor.

Bu üçünün "sağlamca build" + "henüz verify değil" durumu bilerek
korunuyor: tutorial seviyesinde üzerlerinde gerçek operatör-hesabı
deneyen kullanıcılar bağıntıları elle parçalayıp sub-relation'a
düşürebilir. Sonra detaylı verify yolları eklenince bu paragraf
güncellenecek.

## `invariant_d` — magic + lie_iota → d formülü

Classical Koszul-Cartan "invariant d" formülü
`dω(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y])` — bu, magic ve lie_iota
bağıntılarından türetilen *teorem*. `gradalg` bunu tek helper ile
verir:

```python
from gradalg.calculus.invariant_d import invariant_d_one_form
from gradalg.brackets.lie import lie

omega = Symbol("ω")
reg.declare(omega, Graded(degree=1))
for s in (X, Y):
    pass  # already declared Graded(0) via fixtures above

invariant_d_one_form(omega, X, Y, bracket=lie)
# (X(ι_Y(ω)) + (-Y(ι_X(ω))) + (-ι_((X * Y) + (-(Y * X)))(ω)))
```

Formül `InvariantDOneFormDefinition` altında bir `Definition` olarak
da sunuluyor; default classification'ı `"theorem"` (sub-proof magic +
lie_iota citation'ı). İstisna: default classification `d²=0`
aksiyomunda `"axiom"` ama burada `"theorem"` — çünkü formül doğal
olarak bu iki bağıntıdan çıkıyor, aksiyomatik bir giriş noktası değil.

## Sonraki adım

Kendi bracket'inizi yazıp onun üzerinde Jacobi testini koşma —
CustomBracket, flag'ler ve `prove_jacobi` etkileşimi:
[06_custom_bracket.md](06_custom_bracket.md) (Stage C).
