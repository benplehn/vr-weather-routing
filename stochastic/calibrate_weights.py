"""Calibration a posteriori du filtre bayésien des scénarios (game_wind.scenario_weights) sur la course réelle.

Pour chaque bulletin C (membres GEFS data/ens_C/gefs*3h.npz, GFS C) et le vent vrai du jeu (hindsight/data/gfs3h.npz) :
  d_j = écart RMS (vecteur, nds) membre j / GFS C sur les échéances 6..18 h  (ce que le filtre voit au moment du calcul)
  s_j = écart RMS membre j / vent vrai sur C+24..C+96 h, dans le couloir de la route suivie (ce qui compte pour décider)
Questions : (1) d prédit-il s (corrélation de rang) ? (2) quelle température t du poids w_j ∝ exp(-t (d_j² - min d²))
maximise le score logarithmique du vent vrai sous le mélange des membres ? (3) GFS C seul vs moyenne d'ensemble.
Usage : calibrate_weights.py [LEAD_A LEAD_B]  -> stochastic/calibration_weights.json
"""
import os, sys, json, glob, datetime as dt
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'hindsight'))
import bellman_fine as bf
from track import TRACK
T0 = dt.datetime(2026, 11, 1)
LA, LB = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (24, 96)
truth = bf.Field(R / 'hindsight/data/gfs3h.npz')
trk = np.array([[(dt.datetime.strptime(u, '%Y%m%d%H%M') - T0).total_seconds() / 3600, la, lo] for u, la, lo, *_ in TRACK])


def uv(ws, wd): return -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))


def corridor(t0, t1, r=4.):
    """Points (1°) à moins de r degrés de la trace suivie entre t0 et t1 (heures depuis T0)."""
    m = (trk[:, 0] >= t0) & (trk[:, 0] <= t1)
    if m.sum() < 2: m = np.argsort(np.abs(trk[:, 0] - (t0 + t1) / 2))[:3]
    la, lo = np.meshgrid(np.arange(8, 56, 1.), np.arange(-74, 0, 1.), indexing='ij')
    keep = np.zeros(la.shape, bool)
    for _, a, b in trk[m]: keep |= (np.abs(la - a) <= r) & (np.abs(lo - b) <= r * 1.3)
    return la[keep], lo[keep]


rows = []
for d in sorted(glob.glob(str(R / 'data/ens_20261*'))):
    cyc = d[-10:]
    if not cyc.isdigit(): continue
    ch = (dt.datetime.strptime(cyc, '%Y%m%d%H') - T0).total_seconds() / 3600
    if ch + LB > truth.h[-1] - 1 or ch + LB > trk[-1, 0] + 6: continue
    mem = sorted(glob.glob(d + '/gefs*3h.npz'))
    if len(mem) < 10 or not os.path.exists(d + '/gfs3h.npz'): continue
    la, lo = corridor(ch + LA, ch + LB)
    G = bf.Field(d + '/gfs3h.npz'); Ms = [bf.Field(m) for m in mem]
    early = list(range(6, 19, 3)); late = list(range(LA, LB + 1, 6))
    ref_e = [uv(*G(L, la, lo)) for L in early]; ref_l = [uv(*truth(ch + L, la, lo)) for L in late]
    dd, ss, U = [], [], []
    for F in Ms:
        e = np.mean([np.mean((a - c) ** 2 + (b - e2) ** 2) for (a, b), (c, e2) in zip([uv(*F(L, la, lo)) for L in early], ref_e)])
        fl = [uv(*F(L, la, lo)) for L in late]; U.append(fl)
        s = np.mean([np.mean((a - c) ** 2 + (b - e2) ** 2) for (a, b), (c, e2) in zip(fl, ref_l)])
        dd.append(np.sqrt(e)); ss.append(np.sqrt(s))
    sg = np.sqrt(np.mean([np.mean((a - c) ** 2 + (b - e2) ** 2) for (a, b), (c, e2) in zip([uv(*G(L, la, lo)) for L in late], ref_l)]))
    mu = [(np.mean([U[j][k][0] for j in range(len(U))], 0), np.mean([U[j][k][1] for j in range(len(U))], 0)) for k in range(len(late))]
    sm = np.sqrt(np.mean([np.mean((a - c) ** 2 + (b - e2) ** 2) for (a, b), (c, e2) in zip(mu, ref_l)]))
    rows.append(dict(cycle=cyc, d=dd, s=ss, s_gfs=float(sg), s_mean=float(sm), n_points=int(la.size)))
    print(cyc, f'membres {len(mem)} pts {la.size} | GFS {sg:.2f} moy.ens {sm:.2f} meilleur membre {min(ss):.2f} | rang(d)~rang(s) '
          f'{np.corrcoef(np.argsort(np.argsort(dd)), np.argsort(np.argsort(ss)))[0, 1]:+.2f}', flush=True)

# Score : log-vraisemblance du vent vrai sous le mélange pondéré, noyau gaussien de largeur tau (= médiane des s, par bulletin)
T = np.r_[0, np.logspace(-4, 1.5, 34)]


def score(t, r):
    d, s = np.array(r['d']), np.array(r['s']); w = np.exp(-t * (d ** 2 - d.min() ** 2)); w /= w.sum()
    tau = np.median(s); return float(np.log(np.sum(w * np.exp(-(s ** 2 - s.min() ** 2) / (2 * tau ** 2)))))


def ess_t(r, frac=.5):
    d = np.array(r['d']); x = d ** 2 - d.min() ** 2; n = len(d); W = lambda t: np.exp(-x * t); E = lambda w: w.sum() ** 2 / (w ** 2).sum()
    lo, hi = 0., 1.
    while E(W(hi)) > frac * n and hi < 1e6: hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if E(W(mid)) > frac * n: lo = mid
        else: hi = mid
    return lo


S = np.array([[score(t, r) for t in T] for r in rows]); tot = S.mean(0)
# validation croisée (un bulletin laissé de côté) pour la température choisie
cv = []
for i in range(len(rows)):
    m = np.ones(len(rows), bool); m[i] = False; k = int(np.argmax(S[m].mean(0))); cv.append(S[i, k] - S[i, 0])
cur = np.mean([score(ess_t(r), r) for r in rows]) - tot[0]
rho = [float(np.corrcoef(np.argsort(np.argsort(r['d'])), np.argsort(np.argsort(r['s'])))[0, 1]) for r in rows]
k = int(np.argmax(tot))
res = dict(leads=[LA, LB], n_cycles=len(rows), rank_corr_mean=float(np.mean(rho)), rank_corr=rho,
           t_grid=T.tolist(), score_vs_uniform=(tot - tot[0]).tolist(), t_best=float(T[k]), gain_best=float(tot[k] - tot[0]),
           gain_best_crossval=float(np.mean(cv)), gain_current_ess05=float(cur),
           gfs_vs_ensmean=[(r['cycle'], round(r['s_gfs'], 2), round(r['s_mean'], 2)) for r in rows])
(R / f"stochastic/calibration_weights_{LA}_{LB}.json").write_text(json.dumps(res, indent=1))
print(f"\n{len(rows)} bulletins, échéances {LA}-{LB} h. Corrélation de rang moyenne d->s : {np.mean(rho):+.2f}")
print(f"Température optimale t={T[k]:.4g} : gain de log-score vs poids égaux {tot[k] - tot[0]:+.3f} (validation croisée {np.mean(cv):+.3f})")
print(f"Filtre actuel (ESS 50 %) : {cur:+.3f} vs poids égaux")
print('GFS seul vs moyenne d\'ensemble (RMS nds) : GFS meilleur dans', sum(r['s_gfs'] < r['s_mean'] for r in rows), '/', len(rows))
