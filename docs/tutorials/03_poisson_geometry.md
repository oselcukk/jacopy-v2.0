# 03 — Poisson Geometri

Bu tutorial bir symplectic manifold üstüne Poisson bracket'i yerleştirir
ve `{f, g}_π` bracket'inin üç eşdeğer görüşünü — derived, Hamiltonian,
Koszul — tek bir `PoissonBracket` objesi üstünden dolaşır. Amaç, hem
elemen-seviyesi hesapları hem de `[π, π]_SN = 0` universal koşuluna
indirgenmiş Jacobi ispatını `gradalg` içinde ayağa kaldırmak.

[İlk adımlar](01_first_steps.md) ve [Jacobi özdeşliği](02_jacobi_identity.md)
tutorial'larına aşinalık varsayılır.

## Symplectic manifold — (ω, π, ♭, ♯) demet

`SymplecticManifold(ω, bivector=π)` dört veriyi bir arada tutar:
form `ω`, ters-bivector `π`, musical map'ler `ω^♭` / `π^♯`, ve bu ikili
arasındaki `MusicalCompatibility` aksiyomu — `ω^♭ ∘ π^♯ = id`.
Sarıcı matematiksel açıdan yeni bir şey eklemez; değeri, tüm
`(ω, π, ♭, ♯, compat)` beşlisini inşa anında tutarlı tutmasıdır.

```python
from gradalg import Bivector, Forms, Functions
from gradalg.core.registry import PropertyRegistry
from gradalg.library.symplectic import SymplecticManifold

reg = PropertyRegistry()
(omega,) = Forms("ω", degree=2, registry=reg)
pi = Bivector("π", registry=reg)

M = SymplecticManifold(omega, bivector=pi, name="(M, ω, π)")
M.flat          # ω♭
M.sharp         # π♯
M.compatibility # MusicalCompatibility(ω, π, ...)
```

Dikkat: `Bivector` yardımcısı `π` için `Graded(degree=1)` declare eder —
bu bir 2-vektörün SN-shifted derecesi (`k − 1 = 1`). Aynı registry
üstünde `SymplecticManifold` derler.

## `PoissonBracket` — üç eşdeğer görüş

`PoissonBracket.from_bivector(π)` bracket'in üç sunumunu aynı objenin
metodlarına bağlar. Çalışma registry'sinde fonksiyonların SN-shifted
derecesi `−1` olmalı (bir 0-formu SN'de `−1` taşır) — `Functions`
yardımcısının `degree=-1` kwarg'ı tam bu bağlam için.

```python
from gradalg.library.poisson import PoissonBracket

f, g, h = Functions("f g h", degree=-1, registry=reg)
poisson = PoissonBracket.from_bivector(pi)
```

### Görüş 1 — derived bracket

`{f, g}_π = [[f, π]_SN, g]_SN` — Schouten-Nijenhuis kaldıracının üstüne
oturan derived bracket formu:

```python
poisson.expand(f, g, reg)   # [·,·]_SN(f, π)(g)
```

### Görüş 2 — Hamiltonian vector field

`{f, g}_π = X_f(g)`, yani `f`'nin Hamiltonian vektör alanının `g`
üzerine etkisi. `via_hamiltonian` sembolik form'u döner; aynı
bracket'in arkasındaki yapıyı gösterir.

```python
poisson.via_hamiltonian(f, g)  # X_f(g)
poisson.hamiltonian_vf(f)      # X_f
```

İki görüşün *aynı* bracket'i adlandırdığını kanıtlamak için symplectic
manifold'daki `ι_{X_f} ω + df = 0` eşitliği yeterli — `prove_hamiltonian_equivalence`
`MusicalCompatibility` aksiyomunu kullanarak bunu adım adım kapatır:

```python
from gradalg.display import chain_to_ascii

chain = M.prove_hamiltonian_equivalence(f, registry=reg)
print(chain_to_ascii(chain))
```

Zincirin 5 adımı şu yolu izliyor: `ι_X ω` musical eşitlik ile `ω^♭(X)`'e
dönüşür → `X_f = −π^♯(df)` yerleştirmesi → `−` işareti dışarı çekilir →
`ω^♭ ∘ π^♯ = id` kompatibilitesi `d f`'i serbest bırakır → simplify
kalan `−df + df`'i sıfıra indirir.

### Görüş 3 — Koszul üç-terim formülü

Form seviyesinde aynı bracket 1-form'lara uygulanır; 
`{α, β}_π = L_{π^♯(α)} β − L_{π^♯(β)} α − d⟨π^♯(α), β⟩`:

```python
alpha, beta = Forms("α β", degree=1, registry=reg)
poisson.koszul_expand(alpha, beta, reg)
# (L_π♯(α)(β) + (-L_π♯(β)(α)) + (-d(⟨π♯(α), β⟩)))
```

Klasik Koszul bracket (`KoszulBracket(Sharp(π))`) ile derived
bracket'in bu 1-form girişi üstünde *yapısal olarak* eşit olduğu
`prove_koszul_equivalence` ile tek reflexive adımda kapatılır:

```python
chain = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)
len(chain)                           # 1
chain.steps[0].rule                  # 'reflexive'
```

## `[π, π]_SN = 0` — tek koşul

Derived Bracket Teoremi, `{·, ·}_π` üstündeki Jacobi özdeşliğinin tek
koşula indirgendiğini söyler: `[π, π]_SN = 0`. `PoissonBracket` bu
obstruction'ı ham `Expr` olarak, `VanishingCondition` olarak ve
`(f, g, h)` üstünde bir reduction chain olarak verir.

```python
poisson.jacobi_obstruction(reg)   # [·,·]_SN(π, π)
poisson.jacobi_condition(reg)     # VanishingCondition(..., name='Poisson Jacobi condition on {·,·}_π')
```

Üç-giriş reduction zinciri tek adımlık:

```python
chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
len(chain)                        # 1
chain.steps[0].rule               # 'DerivedBracketTheorem'
chain.steps[0].after              # [·,·]_SN(π, π)
```

Zincir obstruction'ı *boşaltmaz* — atomik `π` için `[π, π]_SN`
opak kalır. Poisson hipotezi devreye girdiğinde (yani kullanıcı
`π`'nin bir Poisson bivector olduğunu ilan ettiğinde) aynı adım
Jacobi özdeşliğini kapatır.

## Theorem Book — seeded teorem

Kütüphane bu indirgemeyi `theorem_book.get("poisson_jacobi")` altında
hazır bir `Theorem` kaydı olarak barındırır — `prove_jacobi_reduction`
her çağrılırken zincir yeniden inşa edilmez; downstream kod tek bir
citation ile sonuca bağlanır.

```python
from gradalg.library import theorem_book

thm = theorem_book.get("poisson_jacobi")
thm.statement     # "{f, g, h}_π cyclic sum = 0 when [π, π]_SN = 0"
thm.from_axioms   # ('Derived Bracket Theorem', '[π, π]_SN = 0 (Poisson hypothesis)')
```

`thm.proof` teoremin kanonik `ProofChain`'ini (yani yukarıda
gördüğümüz bir adımlık reduction'ı) taşır.

## Sonraki adım

Lie algebroid çerçevesi aynı derivation stratejisini bir manifold'un
dışında, bir vector bundle'ın üstünde yaşayan bir bracket'e uygular.
Anchor uyumluluk aksiyomu, algebroid Cartan bundle, ve bu yapıyı
`theorem_book` ile nasıl kaydedildiğini
[04_lie_algebroid.md](04_lie_algebroid.md) gösteriyor.
