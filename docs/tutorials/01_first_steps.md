# 01 — İlk Adımlar

Bu tutorial `jacopy`'de bir sembolik ifade kurmayı, özellik (property)
atamayı ve sadeleştirmeyi gösterir. Bitirdiğinizde `Expr`, `Symbol`,
`Sum`, `Product`, `Neg` ile küçük cebirsel ifadeler yazabilecek,
`PropertyRegistry` üzerinden semboller arasına derece/commutativity gibi
ilişkiler kurabilecek ve `simplify` ile bunları canonical hale
getirebileceksiniz.

## Semboller ve temel inşa

Bir sembol `Symbol(name)` ile; literal tamsayılar `Integer(n)` ile;
ikisini toplayıp çarpmak Python'un `+`, `*`, `-` operatörleriyle olur.
Arka planda `Sum`, `Product`, `Neg` inşa edilir — yani `x + y` ile
`Sum(x, y)` birebir aynı nesneyi üretir.

```python
from jacopy.core.expr import Symbol, Integer, Sum, Product, Neg

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

expr = x + y - z
assert expr == Sum(x, y, Neg(z))

mult = 2 * x
assert mult == Product(Integer(2), x)
```

`Expr` bir değer nesnesidir: hash'lenebilir, eşittirlenebilir,
değiştirilemez. İki aynı-şekilli ifade `==` ile eşit çıkar.

## Özellik atama (PropertyRegistry)

Semboller öntanımlı olarak hiçbir cebirsel özellik taşımaz. İlişkileri
`PropertyRegistry` üzerinden *dışsal* olarak ilan ediyoruz — bu, aynı
sembolün farklı bağlamlarda (örneğin farklı derecelerde) yeniden
kullanılmasına izin veriyor.

Derece için `Graded(degree=k)`, çarpımda yer-değiştirmeye izin veren
skalerler için `Scalar()` kullanılır:

```python
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry

reg = PropertyRegistry()
reg.declare(x, Scalar())
reg.declare(y, Scalar())
reg.declare(z, Graded(degree=1))  # z bir 1-form gibi davranır
```

Scalar olarak ilan edilmiş iki sembol çarpımda takas edilebilir;
graded semboller arasındaki değişim `(−1)^{|a||b|}` işaretini taşır.

### Role-driven kısayollar

Sık tekrarlanan desenler (fonksiyon, vektör alanı, form, bivector)
için `jacopy.library.declarations` altında role-driven yardımcılar
var. Her biri `Symbol(...)` + uygun `reg.declare(...)` çağrısını tek
satıra indirir:

```python
from jacopy import Functions, VectorFields, Forms, Bivector

reg2 = PropertyRegistry()
f, g = Functions("f g", registry=reg2)         # Graded(degree=0)
X, Y = VectorFields("X Y", registry=reg2)      # Graded(degree=0)
alpha, beta = Forms("α β", degree=1, registry=reg2)  # Graded(degree=1)
pi = Bivector("π", registry=reg2)              # Graded(degree=1), SN-derecesi
```

Tek isim verildiğinde bile dönen değer bir demet (`(f,) = Functions("f", ...)`);
`Bivector` tek istisna — doğrudan sembolü döner.

## simplify: canonical forma indirme

`simplify(expr, registry)` pipeline'ı:

1. `flatten` — iç içe `Sum`/`Product`'ları düzleştirir.
2. `canonicalize` — `Neg`'leri toplamlara dağıtır, işaretleri toplar.
3. `distribute` — `Product` içindeki `Sum`'ları açar (gerekliyse).
4. `sort_product` — kayıtlı özelliklere göre çarpanları sıralar.
5. `collect_terms` — aynı terimleri birleştirir (`x + x → 2x` vb.).

```python
from jacopy.algorithms.simplify import simplify

assert simplify(x + x - x) == x
assert simplify(Product(Integer(2), x, Integer(3)), reg) == Product(Integer(6), x)
assert simplify(Product(y, x), reg) == Product(x, y)  # alfabetik sıralama
```

Sadeleştirme belirli bir `registry` ile çağrıldığında ilan edilmiş
özellikleri dikkate alır; `registry=None` verildiğinde yalnızca
sembollerden bağımsız olan adımlar çalışır (flatten + canonicalize +
sabit aritmetiği).

## Görselleştirme

`display` katmanı üç ayrı render verir:

- `to_ascii(expr)` — düz metin, monospace terminaller için.
- `to_latex(expr)` — LaTeX string; Jupyter'da doğrudan `LatexDisplay`
  ile render edilir.
- `print_expr(expr)` — `rich` yüklüyse renkli ağaç, değilse ASCII
  fallback.

```python
from jacopy.display import to_ascii, to_latex

to_ascii(x + y - z)       # 'x + y - z'
to_latex(x + y - z)       # 'x + y - z'
```

## Sonraki adım

Aynı `PropertyRegistry` üzerine bir Lie bracket oturttuğumuzda
`simplify`'ın üzerine `prove_jacobi` gibi ispat yardımcıları yükseliyor.
[`02_jacobi_identity.md`](02_jacobi_identity.md) Jacobi özdeşliğinin
nasıl tek satırda kapatıldığını gösteriyor.
