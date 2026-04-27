# 04 — Lie Algebroid

Bir Lie algebroid, bir `M` manifoldu üstündeki vector bundle `E`
üzerinde yaşayan bir bracket ve anchor morfizmi `ρ: E → TM`'den ibaret
üç parçalı yapıdır: `(E, [·,·]_E, ρ)`. Bu tutorial, `jacopy`'in
bu üçlüyü tek bir `LieAlgebroid` objesi etrafında nasıl topladığını,
*anchor compatibility* aksiyomunun neden ayrıca ele alındığını, ve
algebroid üstünde yaşayan Cartan calculus'un (d_E, L_{E,X}, ι_{E,X})
neye benzediğini gösterir.

[03 — Poisson geometri](03_poisson_geometry.md) sonrası okunmak üzere
yazıldı; orada derived bracket'lerin ne işe yaradığını gördük.

## Üçlü: (E, [·,·]_E, ρ)

`LieAlgebroid(bundle, bracket=..., anchor=..., vector_bracket=...)` bu
dörtlüyü tek objede tutar:

- `bundle` — `E`'nin sembolik adı (display için).
- `bracket` — `E`'nin section'ları üstündeki `GradedBracket`;
  graded-antisymmetric, Jacobi ve Leibniz aksiyomları bu bracket'in
  *kendi* flag'lerinde taşınır.
- `anchor` — `Anchor(name="ρ")`: `E → TM` linear bir morphism
  (Derivation sınıfının 0-derecesi altında oturur; Leibniz'i trivial).
- `vector_bracket` — uyum hedefi olan `TM` bracket'i. Varsayılan
  `jacopy.brackets.lie.lie` singleton'ı.

```python
from jacopy import VectorFields
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.anchor import Anchor
from jacopy.core.expr import Symbol
from jacopy.core.registry import PropertyRegistry
from jacopy.library.lie_algebroid import LieAlgebroid

reg = PropertyRegistry()
E = Symbol("E")
bracket_E = LieBracket(name="[·,·]_E")
rho = Anchor(name="ρ")

A = LieAlgebroid(E, bracket=bracket_E, anchor=rho, name="E-algebroid")
```

## Anchor compatibility — ayrı aksiyom

Bracket'in kendi üç aksiyomu (antisymmetry, Jacobi, Leibniz) anchor
compatibility'yi *içermez*. Yani `ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}`
klasik Lie bracket aksiyomlarından türetilmez; Lie algebroid
*tanımının* bir parçasıdır. `jacopy` bunu üç farklı şekilde sunar:

1. **Ham obstruction (Expr):** farkı eşit olması gereken sıfır olarak
   verir — simplify ile indirgemek kullanıcının seçimi.
2. **VanishingCondition:** ham obstruction + isimlendirilmiş koşul.
3. **ProofChain:** tek adımlık `axiom` etiketli zincir — aksiyomu
   "var kabul edip" obstruction'ı sıfıra düşürür.

```python
X, Y = VectorFields("X Y", registry=reg)

A.anchor_compatibility_obstruction(X, Y, reg)
# (ρ(((X * Y) + (-(Y * X)))) + (-((ρ(X) * ρ(Y)) + (-(ρ(Y) * ρ(X))))))

A.anchor_compatibility_condition(X, Y, reg)
# VanishingCondition(..., name='anchor compatibility on E-algebroid')

chain = A.prove_anchor_compatibility(X, Y, registry=reg)
chain.steps[0].rule              # 'LieAlgebroidAnchorCompat'
chain.steps[0].provenance_tag    # 'axiom'
```

Obstruction'ı `simplify` ile sıfıra indirmek istesek başarısız olur —
`TM` bracket'i atomik olduğundan iki tarafı birleştiren yeniden-yazım
kuralı yoktur. Bu, tasarımın amacı: compatibility'nin bir *aksiyom
seçimi* olduğunu saklamak yerine açıkça işaretler.

## Algebroid Cartan bundle

`E` üstündeki exterior algebra `Λ*E*`'de yaşayan bir Cartan calculus'u
`A.cartan` verir:

- `d_E` — algebroid exterior türev (derece +1). `LieAlgebroid`
  inşasında `ExteriorDerivative(name=f"d_{E}")` olarak dökülür ve
  `A.d` erişilir.
- `L_{E,X}` — algebroid Lie türev factory'si: `cart.lie_derivative(X)`.
  İsmine bundle tag gömülür (`L_E,X`), böylece aynı ifade içinde
  manifold Lie türevlerinden ayrışabilir.
- `ι_{E,X}` — algebroid interior product factory'si: `cart.interior(X)`.
  Benzer şekilde `ι_E,X` olarak işaretlenir.

```python
cart = A.cartan
A.d                 # d_E
cart.lie_derivative(X)   # L_E,X
cart.interior(X)         # ι_E,X
```

Beş Cartan bağıntısı (`d_E² = 0`, magic, `[d_E, L]`, `[L, L]`,
`[L, ι]`) ayni API üstünden çağrılır — `cart.relation(name, X=..., Y=...)`
bir `OperatorEquation` döner:

```python
eq = cart.relation("cartan_magic", X=X)
# [d_E, ι_E,X] = L_E,X biçiminde OperatorEquation
```

**Not.** Algebroid Cartan üstünde `cart.verify(...)` mevcut expansion
engine'in "Cartan magic formülünün tanım rewrite'ı"nı default `d/ι_X`
TM operatörleriyle eşleştirmesi sebebiyle özel-adlı `d_E / ι_E,X`
ile otomatik kapanmıyor — bu biliniyor ve
`engine_cartan_definition_deferral` olarak kayıtlı. Beş bağıntının
canlı doğrulanmış örneklerini [05 — Cartan calculus](05_cartan_calculus.md)
TM üstünde yürütür; algebroid'de yapısal simetri aynıdır.

## Seeded teorem

Compatibility aksiyomu `theorem_book` içine kaydedildi:

```python
from jacopy.library import theorem_book

thm = theorem_book.get("lie_algebroid_anchor_compat")
thm.statement    # "ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}"
thm.from_axioms  # ('Lie algebroid anchor compatibility axiom',)
```

Kayıt, aşağı yöndeki teoremlerin (algebroid Cartan, Courant–Dorfman
köprüsü) tek citation ile bu aksiyomu kullanmasını sağlar.

## Sonraki adım

Cartan calculus'un beş bağıntısını — `d² = 0`, magic, `[d, L]`,
`[L, L]`, `[L, ι]` — `TM` üstünde iki ayrı modda (efficient vs
foundational) canlı ispatlayan tur: [05_cartan_calculus.md](05_cartan_calculus.md).
