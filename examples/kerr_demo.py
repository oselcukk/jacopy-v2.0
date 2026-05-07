"""
Kerr-Newman-de Sitter (KNdS) — optimized pipeline.

Optimizations:
  1. Tensor symmetries:
       • Christoffel Γ^a_{bc} = Γ^a_{cb}    → only b ≤ c (40 unique vs 64)
       • Riemann R^a_{bcd} = -R^a_{cbd}     → only b < c, b≠c (96 vs 256)
       • Ricci/Einstein symmetric             → only a ≤ b (10 vs 16)
  2. CSE (Common Subexpression Elimination): cross-tensor shared
     subexpressions extracted, simplified once, substituted back.
  3. Smart simplify chain: together → cancel → trigsimp → factor
     (replaces brute-force sp.simplify).
  4. Multiprocess (joblib loky): paralel simplify.
  5. Auxiliary symbols: Σ, Δ_r, Δ_θ, Ξ as SymPy Functions, kept atomic
     during computation, expanded only at the end.
"""

import os
import sympy as sp
import time
from itertools import product
from joblib import Parallel, delayed

from jacopy.frame_calc import (
    CoordinateFrame, ComponentMetric, levi_civita,
)

n_jobs = os.cpu_count() or 4

# ============================================================== #
# 1) Coordinates + Parameters                                     #
# ============================================================== #

t, r, theta, varphi = sp.symbols("t r θ φ", real=True)
c       = sp.symbols("c", positive=True)
M       = sp.symbols("M", positive=True)
a_spin  = sp.symbols("a", real=True)
Q       = sp.symbols("Q", real=True)
Lam     = sp.symbols("Λ", real=True)

# ============================================================== #
# 2) Auxiliary symbols (atomic Functions)                          #
# ============================================================== #

Sigma       = sp.Function("Σ")(r, theta, a_spin)
Delta_r     = sp.Function("Δ_r")(r, M, a_spin, Q, Lam)
Delta_theta = sp.Function("Δ_θ")(theta, a_spin, Lam)
Xi          = sp.Function("Ξ")(a_spin, Lam)

Sigma_def       = r**2 + a_spin**2 * sp.cos(theta)**2
Delta_r_def     = (r**2 + a_spin**2)*(1 - (Lam/3)*r**2) - 2*M*r + Q**2
Delta_theta_def = 1 + (Lam/3)*a_spin**2 * sp.cos(theta)**2
Xi_def          = 1 + (Lam/3)*a_spin**2

expand_aux = {
    Sigma:       Sigma_def,
    Delta_r:     Delta_r_def,
    Delta_theta: Delta_theta_def,
    Xi:          Xi_def,
}

# ============================================================== #
# 3) KNdS metric                                                   #
# ============================================================== #

g_tt = -(Delta_r / (Sigma * Xi**2)) + (
    Delta_theta * a_spin**2 * sp.sin(theta)**2 / (Sigma * Xi**2)
)
g_tphi = (a_spin * sp.sin(theta)**2 / (Sigma * Xi**2)) * (
    Delta_r - (r**2 + a_spin**2) * Delta_theta
)
g_rr         = Sigma / Delta_r
g_thetatheta = Sigma / Delta_theta
g_phiphi     = (sp.sin(theta)**2 / (Sigma * Xi**2)) * (
    Delta_theta * (r**2 + a_spin**2)**2
    - Delta_r * a_spin**2 * sp.sin(theta)**2
)

g_mat = sp.Matrix([
    [g_tt,    0,    0,             g_tphi  ],
    [0,       g_rr, 0,             0       ],
    [0,       0,    g_thetatheta,  0       ],
    [g_tphi,  0,    0,             g_phiphi],
])

F = CoordinateFrame((t, r, theta, varphi))
g = ComponentMetric(F, sp.MutableDenseNDimArray(g_mat))

coord_names = ["t", "r", "θ", "φ"]
coord_syms  = (t, r, theta, varphi)
n = 4

# ============================================================== #
# 4) Smart simplify (Optimization 3)                              #
# ============================================================== #

def smart_simplify(expr):
    """together → cancel → trigsimp → factor.
    sp.simplify yerine targeted chain — 2-3x daha hızlı."""
    e = sp.together(expr)
    e = sp.cancel(e)
    e = sp.trigsimp(e)
    e = sp.factor(e)
    return e

def expand_and_simplify(expr, aux_dict=None):
    """Substitute auxiliaries + .doit() + smart_simplify."""
    if aux_dict is None:
        aux_dict = expand_aux
    expanded = expr.subs(aux_dict).doit()
    return smart_simplify(expanded)

# ============================================================== #
# 5) Levi-Civita                                                   #
# ============================================================== #

print(f"[{time.strftime('%H:%M:%S')}] Computing Levi-Civita (auxiliaries atomic)...", flush=True)
t0 = time.time()
LC = levi_civita(g, optimized=True)
print(f"[{time.strftime('%H:%M:%S')}] LC done ({time.time()-t0:.1f}s)", flush=True)

# ============================================================== #
# 6) Riemann (Optimization 1: only j < k, antisymmetry mirror)    #
# ============================================================== #

print(f"[{time.strftime('%H:%M:%S')}] Computing Riemann (96 unique entries instead of 256)...", flush=True)
t0 = time.time()
LC_comp = [[[LC[i, j, k] for k in range(n)] for j in range(n)] for i in range(n)]

R = sp.MutableDenseNDimArray.zeros(n, n, n, n)
# R^i_{jkl} antisymmetric in (j, k): R[i, j, k, l] = -R[i, k, j, l]
# Diagonal (j = k): R = 0 automatic
for i, j, k, l in product(range(n), repeat=4):
    if j == k:
        continue                    # 0 by antisymmetry
    if j > k:
        R[i, j, k, l] = -R[i, k, j, l]   # mirror from already-computed
        continue
    # j < k: compute
    term1 = sp.diff(LC_comp[i][k][l], coord_syms[j])
    term2 = sp.diff(LC_comp[i][j][l], coord_syms[k])
    prod1 = sum(LC_comp[i][j][e] * LC_comp[e][k][l] for e in range(4))
    prod2 = sum(LC_comp[i][k][e] * LC_comp[e][j][l] for e in range(4))
    R[i, j, k, l] = term1 - term2 + prod1 - prod2
print(f"[{time.strftime('%H:%M:%S')}] Riemann done ({time.time()-t0:.1f}s)", flush=True)

# ============================================================== #
# 7) Ricci, R-scalar, Einstein (Opt 1: only a ≤ b, mirror)        #
# ============================================================== #

print(f"[{time.strftime('%H:%M:%S')}] Computing Ric, R-scalar, G (10 unique each instead of 16)...", flush=True)
t0 = time.time()

Ric_unique = [(j, l) for j in range(n) for l in range(j, n)]
Ric = sp.MutableDenseNDimArray.zeros(n, n)
for j, l in Ric_unique:
    val = sum(R[i, j, i, l] for i in range(n))
    Ric[j, l] = val
    if j != l:
        Ric[l, j] = val   # mirror

g_inv_mat = sp.Matrix(g_mat).inv()
R_scalar  = sum(g_inv_mat[i, j] * Ric[i, j] for i in range(n) for j in range(n))

G = sp.MutableDenseNDimArray.zeros(n, n)
for i, j in Ric_unique:
    val = Ric[i, j] - sp.Rational(1, 2) * g_mat[i, j] * R_scalar
    G[i, j] = val
    if i != j:
        G[j, i] = val

print(f"[{time.strftime('%H:%M:%S')}] Assembled ({time.time()-t0:.1f}s)", flush=True)

# ============================================================== #
# 8) CSE + parallel simplify (Optimizations 2 + 3 + 4)            #
# ============================================================== #

# Index sets — sadece unique entry'ler
LC_unique = [(i, j, k) for i in range(n) for j in range(n) for k in range(j, n)]

print(f"[{time.strftime('%H:%M:%S')}] CSE: collecting unique expressions...", flush=True)
t0 = time.time()

# Tüm unique entry'leri tek liste yap — CSE tüm bunlara birden bakar
all_raw = (
    [LC[i, j, k] for i, j, k in LC_unique] +
    [Ric[i, j] for i, j in Ric_unique] +
    [G[i, j] for i, j in Ric_unique] +
    [R_scalar]
)
n_lc, n_ric = len(LC_unique), len(Ric_unique)
print(f"  Total unique entries: {len(all_raw)} "
      f"(LC={n_lc}, Ric={n_ric}, G={n_ric}, R=1)", flush=True)

substitutions, simplified_exprs = sp.cse(all_raw)
print(f"  CSE found {len(substitutions)} shared subexpressions.", flush=True)
print(f"  CSE took {time.time()-t0:.1f}s", flush=True)

# Shared subs — auxiliary expand + smart_simplify (paralel)
print(f"[{time.strftime('%H:%M:%S')}] Simplifying {len(substitutions)} shared subexpressions ({n_jobs} workers)...", flush=True)
t0 = time.time()

if substitutions:
    shared_simp_vals = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(expand_and_simplify)(val) for sym, val in substitutions
    )
    shared_simp = dict(zip([sym for sym, _ in substitutions], shared_simp_vals))
else:
    shared_simp = {}

print(f"  Shared simplify: {time.time()-t0:.1f}s", flush=True)

# Substitute shared back — STREAMING DISK-CACHED VERSION
#
# Memory-safety: instead of holding all 61 large expressions in RAM
# simultaneously (8 workers × ~big expr ≈ swap pressure → freeze),
# process ONE entry at a time, write result to disk, free memory,
# move on. Final read-back happens just before print.
#
# Also: `xreplace` instead of `subs` — atomic replace, faster + less RAM.

import pickle
import gc
from pathlib import Path

CACHE_DIR = Path("kerr_cache")
CACHE_DIR.mkdir(exist_ok=True)
print(f"[{time.strftime('%H:%M:%S')}] Cache dir: {CACHE_DIR.absolute()}", flush=True)

# CSE state — save immediately as crash-recovery checkpoint
sub_pairs = list(substitutions)
with open(CACHE_DIR / "cse_state.pkl", "wb") as f:
    pickle.dump({"sub_pairs": sub_pairs, "shared_simp": shared_simp}, f)
print(f"  CSE checkpoint saved ({len(sub_pairs)} sub_pairs, "
      f"{len(shared_simp)} shared values).", flush=True)

# Build a pure xreplace-friendly dict in REVERSE topo order
# Single dict for atomic replacement — values still chain, but xreplace
# in reverse order resolves cleanly.
reverse_chain = list(reversed(sub_pairs))   # xN, xN-1, ..., x0

def _stream_resolve_save(idx, expr, reverse_chain, shared_dict, aux, cache_dir):
    """Process ONE entry: resolve CSE chain (reverse order, xreplace),
    expand auxiliaries, cancel, write to disk, return only index."""
    out = expr
    # xreplace ile tek atomic replace, sequential reverse order
    for sym, _ in reverse_chain:
        if sym in shared_dict:
            out = out.xreplace({sym: shared_dict[sym]})
    out = out.subs(aux).doit()
    out = sp.cancel(out)
    # Hemen diske yaz, RAM'den çıkar
    with open(Path(cache_dir) / f"entry_{idx:03d}.pkl", "wb") as f:
        pickle.dump(out, f)
    del out
    gc.collect()
    return idx

# Streaming back-substitute — DÜŞÜK paralelizm (memory için)
streaming_jobs = max(2, n_jobs // 4)   # 8 yerine 2 — 4x az RAM
print(f"[{time.strftime('%H:%M:%S')}] Streaming back-substitute "
      f"({streaming_jobs} workers, disk cache, low-mem mode)...", flush=True)
t0 = time.time()
done_indices = Parallel(n_jobs=streaming_jobs, backend="loky", verbose=5)(
    delayed(_stream_resolve_save)(idx, e, reverse_chain, shared_simp,
                                  expand_aux, str(CACHE_DIR))
    for idx, e in enumerate(simplified_exprs)
)
print(f"  Streaming back-substitute: {time.time()-t0:.1f}s", flush=True)

# Sonuçları diskten oku — print öncesi
print(f"[{time.strftime('%H:%M:%S')}] Loading {len(done_indices)} results from cache...", flush=True)
t0 = time.time()
final_results = []
for idx in range(len(simplified_exprs)):
    with open(CACHE_DIR / f"entry_{idx:03d}.pkl", "rb") as f:
        final_results.append(pickle.load(f))
print(f"  Cache load: {time.time()-t0:.1f}s", flush=True)

# Sonuçları LC_simp, Ric_simp, G_simp, R_scalar_s'e dağıt + simetri yansıt
LC_simp = {}
for idx, (i, j, k) in enumerate(LC_unique):
    val = final_results[idx]
    LC_simp[(i, j, k)] = val
    if j != k:
        LC_simp[(i, k, j)] = val   # Christoffel symmetry mirror

Ric_simp = {}
for idx, (i, j) in enumerate(Ric_unique):
    val = final_results[n_lc + idx]
    Ric_simp[(i, j)] = val
    if i != j:
        Ric_simp[(j, i)] = val

G_simp = {}
for idx, (i, j) in enumerate(Ric_unique):
    val = final_results[n_lc + n_ric + idx]
    G_simp[(i, j)] = val
    if i != j:
        G_simp[(j, i)] = val

R_scalar_s = final_results[-1]

print(f"[{time.strftime('%H:%M:%S')}] All done.\n", flush=True)

# ============================================================== #
# 9) PRINT — full output (every entry, with index)                #
# ============================================================== #

print("=" * 70)
print("Levi-Civita coefficients — Γ^a_{bc}:\n")
idx = 0
for up in range(4):
    for lo1 in range(4):
        for lo2 in range(4):
            val = LC_simp[(up, lo1, lo2)]
            print(f"  {idx:>3}  Γ^{coord_names[up]}_{{{coord_names[lo1]}{coord_names[lo2]}}}  =  {val}")
            idx += 1

print("\n" + "=" * 70)
print("Ricci tensor — Ric_{ab}:\n")
idx = 0
for row in range(4):
    for col in range(4):
        val = Ric_simp[(row, col)]
        print(f"  {idx:>3}  Ric_{{{coord_names[row]}{coord_names[col]}}}  =  {val}")
        idx += 1

print("\n" + "=" * 70)
print("Ricci scalar:\n")
print(f"  R = {R_scalar_s}")

print("\n" + "=" * 70)
print("Einstein tensor — G_{ab}:\n")
idx = 0
for row in range(4):
    for col in range(4):
        val = G_simp[(row, col)]
        print(f"  {idx:>3}  G_{{{coord_names[row]}{coord_names[col]}}}  =  {val}")
        idx += 1
