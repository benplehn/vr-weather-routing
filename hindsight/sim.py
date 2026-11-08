"""Simulation a posteriori avec physique complète (énergie, pénalités qui débordent sur les pas suivants, régulateur
d'allure = TWA constante), vent vrai du jeu, et recherche par faisceau guidée par la valeur Bellman.

Usage : sim.py TAG [MODE] [BEAM] [DT_MIN] [START_IDX]
  MODE = free  : ordres libres à chaque pas (borne « routeur parfait »)
         night : contrainte utilisateur — de 22:30 à 10:00 Paris (21:30–09:00 UTC) au plus 3 ordres de TWA, voile figée
  START_IDX : indice de TRACK comme point de départ (0 = départ)
Sortie : hindsight/out/route_TAG_MODE_START.json (trajectoire, ordres, ETA).
"""
import os, sys, json, time, datetime as dt
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
H = Path(__file__).resolve().parent; R = H.parent
sys.path.insert(0, str(H)); sys.path.insert(0, str(R))
TAG = sys.argv[1] if len(sys.argv) > 1 else 'w74'
OUTSFX = os.environ.get('OUTSFX', '')
os.environ.update(LON_MIN='-65' if TAG == 'w65' else '-74', BOX_LAT_LO='10', BOX_LAT_HI='52', BOX_LON_HI='-1')
os.environ['ROUTING_SCENARIO'] = str(H / 'scenario.json'); os.environ.setdefault('FULL_PACK', '1')
import optimal_route as p
from bellman_fine import Field
from track import TRACK

MODE = sys.argv[2] if len(sys.argv) > 2 else 'free'
BEAM = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
DT = (float(sys.argv[4]) if len(sys.argv) > 4 else 20.) / 60
I0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
T0 = dt.datetime(2026, 11, 1)
W = Field(H / 'data/gfs3h.npz')
G = dict(np.load(H / 'data' / TAG / 'bellman_fine.npz'))
TWA = np.r_[np.arange(40, 170, 5), 168.]
SAILS = [int(x) for x in os.environ.get('SIM_SAILS', '0,1').split(',')]; NORD = int(os.environ.get('SIM_ORDERS', '3'))
LAYER_SAIL = np.array([0, 1, 0, 0, 0, 1, 1])
ACT = [(s * a, sail) for a in TWA for s in (-1, 1) for sail in SAILS]
TAIL = 2.5


def V(t, lat, lon, layer):
    x = t - float(G['lead0']); k = int(np.clip(np.floor(x), 0, len(G['values']) - 2)); f = float(np.clip(x - k, 0, 1))
    st = float(G['lat'][1] - G['lat'][0]); c = [(lat - G['lat'][0]) / st, (lon - G['lon'][0]) / st]; out = np.empty(lat.shape)
    for j in range(4):
        m = layer == j
        if m.any():
            cc = [c[0][m], c[1][m]]
            out[m] = map_coordinates(G['values'][k, j], cc, order=1, cval=2000, prefilter=False) * (1 - f) + \
                     map_coordinates(G['values'][k + 1, j], cc, order=1, cval=2000, prefilter=False) * f
    return out


def night(t):  # t heures depuis T0 (UTC) ; nuit utilisateur 21:30–09:00 UTC
    h = t % 24; return h >= 21.5 or h < 9.


def night_id(t):  # identifiant de la nuit (jour du réveil)
    return int((t + 2.5) // 24)


def step(S, twa, sail, t):
    """Avance tous les états S d'un pas DT avec la commande (twa signé, voile). Retourne le nouvel état et validité."""
    lat, lon, e, s0, tk0, pen, twa0 = S['lat'], S['lon'], S['e'], S['sail'], S['tack'], S['pen'], S['twa']
    tk = np.full(lat.shape, 1 if twa > 0 else -1); cs = sail != s0; ct = tk != tk0
    ws, wd = W(t, lat, lon)
    used, newpen = p.maneuver(ws, e, cs, ct)
    e1 = np.maximum(0, e - used); pen1 = np.where(cs | ct, np.maximum(pen, newpen), pen)
    # deux demi-pas, cap recalculé (régulateur TWA)
    la, lo, pl = lat, lon, pen1
    for half in (0, 1):
        tt = t + (half + .5) * DT / 2
        w2, d2 = W(tt, la, lo); hd = (d2 - twa) % 360
        sp = p.polar(twa, w2, sail)
        inpen = np.minimum(DT / 2, pl); eff = DT / 2 - p.PEN_LOSS * inpen
        la2, lo2 = p.move(la, lo, hd, sp * eff)
        ok = p.leg_allowed(la, lo, la2, lo2) if half == 0 else ok & p.leg_allowed(la, lo, la2, lo2)
        rec = p.recover_seconds(w2); e1 = np.minimum(100, e1 + (DT / 2 - inpen) * 3600 / rec)
        la, lo, pl = la2, lo2, pl - inpen
    return dict(lat=la, lon=lo, e=e1, sail=np.broadcast_to(sail, la.shape).copy(), tack=tk, pen=pl,
                twa=np.broadcast_to(twa, la.shape).astype(float)), ok


def crossed(lat0, lon0, lat1, lon1):
    """Passage de la ligne 16.25N entre 62.8W et 61.88W (fin des tables) ; retourne (masque, fraction, longitude)."""
    m = (lat0 > 16.25) & (lat1 <= 16.25)
    f = np.where(m, (lat0 - 16.25) / np.maximum(lat0 - lat1, 1e-9), 0); lx = lon0 + f * (lon1 - lon0)
    m &= (lx <= -61.88) & (lx >= -62.8)
    d = p.distance(np.full(lx.shape, 16.25), lx, p.TAIL_WP); d0 = float(p.distance(np.array([16.25]), np.array([-61.88]), p.TAIL_WP)[0])
    return m, f, np.maximum(0, d - d0) / p.TAIL_KN


def run():
    u, la, lo, s, tk, _ = TRACK[I0]
    t = (dt.datetime.strptime(u, '%Y%m%d%H%M') - T0).total_seconds() / 3600
    S = dict(lat=np.array([la]), lon=np.array([lo]), e=np.array([100. if I0 == 0 else 85.]), sail=np.array([s]),
             tack=np.array([tk]), pen=np.zeros(1), twa=np.array([0.]))
    S['nord'] = np.zeros(1, int); S['nid'] = np.full(1, -1); S['hist'] = [[]]
    best = (np.inf, None); tic = time.time(); n = 0
    while True:
        cand = []
        nt = night(t) and MODE == 'night'
        for twa, sail in ACT:
            if nt:  # nuit : voile figée ; ordre = changement de TWA, compté
                keep = S['sail'] == sail
                if not keep.any(): continue
            else:
                keep = np.ones(len(S['lat']), bool)
            idx = np.nonzero(keep)[0]
            Sk = {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]) for k, v in S.items()}
            if nt:
                same = np.abs(Sk['twa'] - twa) < 1e-6; nid = night_id(t)
                cnt = np.where(Sk['nid'] == nid, Sk['nord'], 0) + (~same)
                ok_o = cnt <= NORD
                if not ok_o.any(): continue
            else:
                cnt = np.zeros(len(idx), int); ok_o = np.ones(len(idx), bool); nid = -1
            N, ok = step(Sk, float(twa), sail, t)
            ok &= ok_o
            N['nord'] = cnt; N['nid'] = np.full(len(idx), nid)
            N['par'] = idx; N['act'] = np.full(len(idx), twa); N['as'] = np.full(len(idx), sail)
            m, f, toff = crossed(Sk['lat'], Sk['lon'], N['lat'], N['lon'])
            arr = m & ok
            if arr.any():
                eta = t + f[arr] * DT + toff[arr] + TAIL; i = int(np.argmin(eta))
                if eta[i] < best[0]:
                    pi = idx[np.nonzero(arr)[0][i]]
                    best = (float(eta[i]), S['hist'][pi] + [(t, float(N['lat'][arr][i]), float(N['lon'][arr][i]), float(twa), int(sail), float(N['e'][arr][i]))])
            score = t + DT + V(t + DT, N['lat'], N['lon'], 2 * LAYER_SAIL[N['sail']] + (N['tack'] > 0)) + TAIL
            score = np.where(ok & ~m, score, np.inf)
            # petit terme énergie : une énergie basse allonge les pénalités futures
            score = score + .02 * (100 - N['e']) / 10
            N['score'] = score; cand.append(N)
        C = {k: np.concatenate([c[k] for c in cand]) for k in cand[0]}
        fin = np.isfinite(C['score'])
        if not fin.any() or C['score'][fin].min() > best[0]: break
        key = (np.floor(C['lat'] / .03).astype(np.int64) * 100000 + np.floor(C['lon'] / .03).astype(np.int64)) * 40 + \
              (2 * LAYER_SAIL[C['sail']] + (C['tack'] > 0)) * 10 + np.floor(C['e'] / 25).astype(int) + C['sail'] * 1000000000000 + (C['nord'] if MODE == 'night' else 0) * 4000000000
        order = np.lexsort((C['score'], key)); first = np.r_[True, key[order][1:] != key[order][:-1]]
        sel = order[first]; sel = sel[np.isfinite(C['score'][sel])]
        sel = sel[np.argsort(C['score'][sel])][:BEAM]
        hist = [S['hist'][C['par'][i]] + [(t, float(C['lat'][i]), float(C['lon'][i]), float(C['act'][i]), int(C['as'][i]), float(C['e'][i]))] for i in sel]
        S = {k: C[k][sel] for k in ('lat', 'lon', 'e', 'sail', 'tack', 'pen', 'twa', 'nord', 'nid')}; S['hist'] = hist
        t += DT; n += 1
        if n % 36 == 0:
            print(f't={t:6.1f} {(T0 + dt.timedelta(hours=t)):%d/%m %H:%M} états {len(sel)} meilleur score {C["score"][sel[0]]:.2f} '
                  f'pos {S["lat"][0]:.2f} {S["lon"][0]:.2f} best_eta {best[0]:.2f} ({time.time() - tic:.0f}s)', flush=True)
    eta, hist = best
    out = H / 'out'; out.mkdir(exist_ok=True)
    orders = []
    for i, r in enumerate(hist):
        if i == 0 or r[3] != hist[i - 1][3] or r[4] != hist[i - 1][4]: orders.append(r)
    res = dict(tag=TAG, mode=MODE, beam=BEAM, dt_min=DT * 60, start=TRACK[I0][0], eta_h=eta,
               eta_utc=f'{T0 + dt.timedelta(hours=eta):%Y-%m-%d %H:%M}Z', n_orders=len(orders),
               n_tacks=int(sum(1 for i in range(1, len(hist)) if np.sign(hist[i][3]) != np.sign(hist[i - 1][3]))),
               min_energy=min(r[5] for r in hist), track=hist, orders=orders)
    f = out / f'route_{TAG}_{MODE}_{I0}{OUTSFX}.json'; f.write_text(json.dumps(res))
    print('ETA', res['eta_utc'], 'ordres', len(orders), 'empannages/virements', res['n_tacks'], 'énergie mini', round(res['min_energy']), '->', f)


if __name__ == '__main__':
    run()
