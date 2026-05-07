"""
Kerr-Newman-de Sitter (KNdS) — full pipeline.

Christoffel (Γ) → Riemann → Ricci → R-scalar → Einstein, all entries
printed with index numbering.

Optimizations:
  1. Auxiliary symbols Σ, Δ_r, Δ_θ, Ξ as SymPy Function — kept atomic
     during computation, expanded only at the end.
  2. Tensor symmetries:
       • Christoffel Γ^a_{bc} = Γ^a_{cb}    → only b ≤ c (40 vs 64)
       • Riemann R^a_{bcd} = -R^a_{cbd}     → only j < k (96 vs 256)
       • Ricci/Einstein symmetric             → only a ≤ b (10 vs 16)
  3. CSE (Common Subexpression Elimination) — shared subexpressions
     across LC + Ric + G + R simplified once.
  4. Smart simplify chain: together → cancel → trigsimp → factor
     (replaces brute-force sp.simplify).
  5. Multiprocess (joblib loky) — parallel simplify.

Tested on Colab Pro (25GB RAM, 4-8 cores) — bottleneck is `sp.cancel`
on the heaviest Ric/G entries, but RAM stays under control.
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
# 4) Smart simplify                                                #
# ============================================================== #

def smart_simplify(expr):
    """together → cancel → trigsimp → factor."""
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
# 6) Riemann (j < k antisymmetry exploitation)                     #
# ============================================================== #

print(f"[{time.strftime('%H:%M:%S')}] Computing Riemann (96 unique entries)...", flush=True)
t0 = time.time()
LC_comp = [[[LC[i, j, k] for k in range(n)] for j in range(n)] for i in range(n)]

R = sp.MutableDenseNDimArray.zeros(n, n, n, n)
for i, j, k, l in product(range(n), repeat=4):
    if j == k:
        continue
    if j > k:
        R[i, j, k, l] = -R[i, k, j, l]
        continue
    term1 = sp.diff(LC_comp[i][k][l], coord_syms[j])
    term2 = sp.diff(LC_comp[i][j][l], coord_syms[k])
    prod1 = sum(LC_comp[i][j][e] * LC_comp[e][k][l] for e in range(4))
    prod2 = sum(LC_comp[i][k][e] * LC_comp[e][j][l] for e in range(4))
    R[i, j, k, l] = term1 - term2 + prod1 - prod2
print(f"[{time.strftime('%H:%M:%S')}] Riemann done ({time.time()-t0:.1f}s)", flush=True)

# ============================================================== #
# 7) Ricci, R-scalar, Einstein                                    #
# ============================================================== #

print(f"[{time.strftime('%H:%M:%S')}] Computing Ric, R-scalar, G...", flush=True)
t0 = time.time()

Ric_unique = [(j, l) for j in range(n) for l in range(j, n)]
Ric = sp.MutableDenseNDimArray.zeros(n, n)
for j, l in Ric_unique:
    val = sum(R[i, j, i, l] for i in range(n))
    Ric[j, l] = val
    if j != l:
        Ric[l, j] = val

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
# 8) CSE                                                           #
# ============================================================== #

LC_unique = [(i, j, k) for i in range(n) for j in range(n) for k in range(j, n)]

print(f"[{time.strftime('%H:%M:%S')}] CSE: collecting unique expressions...", flush=True)
t0 = time.time()

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
print(f"  CSE found {len(substitutions)} shared subexpressions ({time.time()-t0:.1f}s)", flush=True)

# ============================================================== #
# 9) Parallel simplify of shared subexpressions                    #
# ============================================================== #

print(f"[{time.strftime('%H:%M:%S')}] Simplifying {len(substitutions)} "
      f"shared subexpressions ({n_jobs} workers)...", flush=True)
t0 = time.time()

if substitutions:
    shared_simp_vals = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(expand_and_simplify)(val) for sym, val in substitutions
    )
    shared_simp = dict(zip([sym for sym, _ in substitutions], shared_simp_vals))
else:
    shared_simp = {}

print(f"  Shared simplify: {time.time()-t0:.1f}s", flush=True)

# ============================================================== #
# 10) Back-substitute (reverse CSE order) + cancel                 #
# ============================================================== #

sub_pairs = list(substitutions)
reverse_chain = list(reversed(sub_pairs))

def back_substitute_resolve(expr, reverse_chain, shared_dict, aux):
    """Apply CSE subs in REVERSE order (chained refs fully resolved) +
    expand auxiliaries + cancel."""
    out = expr
    for sym, _ in reverse_chain:
        if sym in shared_dict:
            out = out.xreplace({sym: shared_dict[sym]})
    out = out.subs(aux).doit()
    return sp.cancel(out)

print(f"[{time.strftime('%H:%M:%S')}] Back-substituting + cancel ({n_jobs} workers)...", flush=True)
t0 = time.time()
final_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
    delayed(back_substitute_resolve)(e, reverse_chain, shared_simp, expand_aux)
    for e in simplified_exprs
)
print(f"  Back-substitute: {time.time()-t0:.1f}s", flush=True)

# ============================================================== #
# 11) Distribute + symmetry mirror                                 #
# ============================================================== #

LC_simp = {}
for idx, (i, j, k) in enumerate(LC_unique):
    val = final_results[idx]
    LC_simp[(i, j, k)] = val
    if j != k:
        LC_simp[(i, k, j)] = val

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
# 12) PRINT — full output                                          #
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
