"""Bellman grossier puis fin (point 2 du plan Vendée Globe).

1. Table grossière (COARSE_STEP, 1° par défaut) sur toute la boîte : ~16x moins de nœuds qu'à 0,25°.
2. Couloir : routes gloutonnes sur la table grossière depuis les positions de départ (pour chaque voile/amure de
   départ), élargies de BUFFER_NM milles, plus la zone d'arrivée.
3. Table fine (0,25°) calculée SEULEMENT sur les nœuds du couloir ; hors couloir = mur (2000), comme hors boîte.
Même mise à jour semi-lagrangienne que bellman_fine.build (mêmes angles, pénalités, terre, ligne d'arrivée).
Usage : bellman_c2f.py ens_dir lead0 lat lon [lat lon ...]  -> ens_dir/bellman_fine.npz (+ corridor dans le fichier)
"""
import os, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
R = Path(__file__).resolve().parent; sys.path.insert(0, str(R))
import bellman_fine as bf
p = bf.p
COARSE = float(os.environ.get('COARSE_STEP', '1.0')); BUFFER_NM = float(os.environ.get('BUFFER_NM', '300'))


def coarse_routes(G, lead0, starts, hours):
    """Routes gloutonnes (pas 1 h, angles bf.ANGLES, Jib/Spi) sur la table G depuis chaque départ, pour les 4 couches."""
    pts = []
    W = G['_W']
    for la0, lo0 in starts:
        for s0 in (0, 1):
            for t0 in (-1, 1):
                la, lo, s, tk = np.array([la0]), np.array([lo0]), s0, t0
                for k in range(int(hours)):
                    L = lead0 + k; ws, wd = W(L + .5, la, lo); best = None
                    for a in bf.ANGLES:
                        for ns in (0, 1):
                            nt = 1 if a > 0 else -1
                            _, pen = p.maneuver(ws, bf.TYPICAL_ENERGY, ns != s, nt != tk)
                            ln, on = p.move(la, lo, (wd - a) % 360, p.polar(a, ws, ns) * (1 - .5 * np.minimum(1, pen)))
                            if not p.leg_allowed(la, lo, ln, on)[0]: continue
                            v = 1 + bf.value(G, L + 1, ln, on, np.array([ns]), np.array([nt]))[0]
                            if best is None or v < best[0]: best = (v, ln, on, ns, nt)
                    if best is None or best[0] >= 1000: break
                    _, la, lo, s, tk = best; pts.append((float(la[0]), float(lo[0])))
                    if bf.value(G, L + 1, la, lo, np.array([s]), np.array([tk]))[0] < .5: break
    return np.array(pts)


def build(ens_dir, lead0, starts, horizon=None, dt=1.):
    tic = time.time(); ens_dir = Path(ens_dir)
    horizon = horizon or int(os.environ.get('FINE_HORIZON', '180'))
    # 1. grossier (bellman_fine.build au pas COARSE -> bellman_fine_{COARSE}_1.npz)
    cf = bf.build(str(ens_dir), COARSE, 1., lead0=lead0, horizon=horizon)
    G = dict(np.load(cf)); G['_W'] = bf.Field(ens_dir / 'gfs3h.npz'); t1 = time.time() - tic
    # 2. couloir
    pts = coarse_routes(G, lead0, starts, horizon)
    las = np.arange(p.LAT_LO, p.LAT_HI + 1e-9, bf.STEP); los = np.arange(p.LON_MIN, p.LON_HI + 1e-9, bf.STEP)
    lo2, la2 = np.meshgrid(los, las)
    mask = np.zeros(la2.shape, bool)
    if len(pts):
        i = np.clip(np.round((pts[:, 0] - las[0]) / bf.STEP).astype(int), 0, len(las) - 1)
        j = np.clip(np.round((pts[:, 1] - los[0]) / bf.STEP).astype(int), 0, len(los) - 1); mask[i, j] = True
    r = int(np.ceil(BUFFER_NM / 60 / bf.STEP))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]; disk = yy ** 2 + xx ** 2 <= r * r
    mask = binary_dilation(mask, structure=disk)
    target, _ = p.goal_target(la2, lo2, bf.STEP)
    mask |= binary_dilation(target, structure=disk)  # arrivée toujours dans le couloir
    t2 = time.time() - tic - t1
    # 3. fin, sur le couloir seulement
    out = fine_masked(ens_dir, lead0, horizon, mask, dt)
    print(f'C2F grossier {t1:.0f}s couloir {t2:.0f}s ({mask.mean():.1%} des nœuds) fin {time.time() - tic - t1 - t2:.0f}s total {time.time() - tic:.0f}s', flush=True)
    return out


def fine_masked(ens_dir, lead0, horizon, mask, dt=1.):
    step = bf.STEP
    W = bf.Field(Path(ens_dir) / 'gfs3h.npz')
    las = np.arange(p.LAT_LO, p.LAT_HI + 1e-9, step); los = np.arange(p.LON_MIN, p.LON_HI + 1e-9, step)
    g = dict(lat0=las[0], lon0=los[0], step=step)
    lo2, la2 = np.meshgrid(los, las); ok = p.allowed(la2, lo2) & mask
    target, toff = p.goal_target(la2, lo2, step)
    P = ok | target; Pla, Plo = la2[P], lo2[P]; tP = target[P]; okP = ok[P]
    n = int(round(horizon / dt)); per = int(round(1 / dt))
    V = np.full((horizon + 1, 4, *la2.shape), 2000., np.float32)
    term = np.where(target, toff, 200 + p.distance(la2, lo2) / 20); term[~ok] = 2000
    cur = np.repeat(term[None], 4, 0).astype(np.float32); V[-1] = cur
    NW = int(os.environ.get('FINE_WORKERS', '1')); pool = None
    bf._P.update(la=Pla, lo=Plo, g=g, dt=dt)
    if NW > 1:
        import multiprocessing as mp; pool = mp.get_context('fork').Pool(NW)
    for k in range(n - 1, -1, -1):
        ws, wd = W(lead0 + (k + .5) * dt, Pla, Plo)
        if pool is not None:
            best = np.min(pool.map(bf._partial_ws, [(ch, cur, ws, wd) for ch in np.array_split(bf.ANGLES, NW)]), axis=0)
        else:
            bf._P.update(ws=ws, wd=wd); best = bf._partial((bf.ANGLES, cur))
        best[:, tP] = toff[P][tP]; best[:, ~okP & ~tP] = 2000
        cur = np.full_like(cur, 2000.); cur[:, P] = best
        if k % per == 0: V[k // per] = cur
    if pool is not None: pool.close(); pool.join()
    out = Path(ens_dir) / os.environ.get('C2F_OUT', 'bellman_fine.npz')
    np.savez(out, values=V, lead0=lead0, lat=las, lon=los, typical_energy=bf.TYPICAL_ENERGY, dt=dt, corridor=mask)
    return out


if __name__ == '__main__':
    a = sys.argv[1:]; xs = [float(x) for x in a[2:]]
    print('WROTE', build(a[0], float(a[1]), list(zip(xs[::2], xs[1::2]))))
