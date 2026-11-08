"""Faisceau STOCHASTIQUE : un programme commun (non anticipatif) évalué dans tous les scénarios à la fois.

Chaque nœud du faisceau = une suite d'ordres commune ; son état = la position/énergie/pénalité du bateau dans CHAQUE
scénario (GFS + membres GEFS, vent du jeu game_wind : GFS connu jusqu'à C+12, puis le scénario). Coût d'un nœud =
moyenne pondérée sur les scénarios de (temps écoulé + valeur restante de CE scénario) ; les poids viennent du filtre
bayésien (game_wind.scenario_weights) avec la température calibrée a posteriori (SCEN_T, calibrate_weights.py) au lieu
d'une cible d'ESS arbitraire. Un nœud dont un scénario touche terre est rejeté (robuste). Même contraintes que
beam_prog.py (≤ BEAM_ORDERS ordres sur [0, H], voile figée la nuit, voiles BEAM_SAILS).

Valeur terminale par scénario : TERMINAL=member (tables data/ens_C/bellman_members, valeur clairvoyante au-delà de
l'horizon) ou gfs (table GFS pour tous : stochastique seulement sur l'horizon du faisceau).
Accélération : vents et valeurs des scénarios échantillonnés une fois sur la grille de la table, dans une fenêtre autour
du bateau, empilés (scénario, temps, [couche,] lat, lon) -> une seule interpolation vectorisée par pas et par action.
Usage : beam_stoch.py CYCLE OBS lat lon énergie angle voile H [programme entré JSON]   -> ligne 'BEAMS {json}'
"""
import os, sys, json, glob, time, datetime as dt
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
os.environ.setdefault('FULL_PACK', '1')
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'stochastic'))
import optimal_route as p, bellman_fine as bf, game_wind
from beam_prog import night_paris, crossed, LAYER_SAIL, TWA, SAILS, MAXO, HZ, DT, E_W, UTC
N = int(os.environ.get('BS_N', '400')); K = int(os.environ.get('BS_K', '0'))  # K>0 : K scénarios (GFS + les plus lourds)
TERM = os.environ.get('TERMINAL', 'member'); WIN = float(os.environ.get('BS_WIN', '14'))


def _uv(ws, wd): return -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))


class Shifted:
    """Champ d'un bulletin plus ancien P appelé en échéances du bulletin courant C : lead_P = lead_C + (C - P)."""
    def __init__(self, F, shift): self.F, self.s = F, shift
    def __call__(self, L, lat, lon): return self.F(L + self.s, lat, lon)


class Stoch:
    def __init__(self, cyc, obs, lat, lon, sails, ens=None, truth=None):
        self.c0 = dt.datetime.strptime(cyc, '%Y%m%d%H').replace(tzinfo=UTC); self.t0 = obs
        self.L0 = (obs - self.c0).total_seconds() / 3600; ens = Path(ens or R / 'data' / f'ens_{cyc}')
        G = dict(np.load(ens / 'bellman_fine.npz'))
        base = game_wind.fields(ens, ['gfs'], cyc)[0]  # vent du jeu connu (GFS précédent puis GFS C), scénario = GFS C
        # Scénarios : GFS C, membres de C, et (point 1, ensemble décalé) membres des LAGGED bulletins précédents recalés
        # dans le temps ; chacun avec sa table de valeur (déjà calculée pour son bulletin), décalée de la même façon.
        scen = [dict(name='gfs', raw=bf.Field(ens / 'gfs3h.npz'), shift=0., table=G)]
        for k in range(0 if os.environ.get('CUR_MEMBERS', '1') == '1' else 1, int(os.environ.get('LAGGED', '2')) + 1):  # CUR_MEMBERS=0 : GEFS du bulletin pas encore là
            P = (self.c0 - dt.timedelta(hours=6 * k)).strftime('%Y%m%d%H'); eP = ens.parent / f'ens_{P}'
            for x in sorted(glob.glob(str(eP / 'gefs*3h.npz'))):
                m = Path(x).name[:-len('3h.npz')]; tf = eP / 'bellman_members' / f'{m}.npz'
                if TERM == 'member' and not tf.exists(): continue
                scen.append(dict(name=m if k == 0 else f'{m}@{P[6:]}', raw=bf.Field(x), shift=6. * k, table=tf if TERM == 'member' else G))
        t = float(os.environ.get('SCEN_T', '0.7'))  # température calibrée a posteriori (calibrate_weights.py, 24-72 h)
        w = calibrated_weights(scen, ens, t, lat, lon)
        if K and K < len(scen):  # on garde GFS + les K-1 scénarios les plus lourds (renormalisés)
            keep = [0] + list(np.argsort(-w[1:])[:K - 1] + 1); scen = [scen[i] for i in keep]; w = w[keep]
        self.w = w / w.sum(); self.names = [x['name'] for x in scen]; self.sails = sails
        WF = [base] + [game_wind.GameWind(base.prev, base.new, Shifted(x['raw'], x['shift']), base.switch, base.known) for x in scen[1:]]
        gl, go = G['lat'], G['lon']; st = float(gl[1] - gl[0])
        i0, i1 = np.searchsorted(gl, lat - WIN), np.searchsorted(gl, lat + WIN) + 1
        j0, j1 = np.searchsorted(go, lon - WIN * 1.3), np.searchsorted(go, lon + WIN * 1.3) + 1
        self.la, self.lo, self.st = gl[i0:i1], go[j0:j1], st
        LA, LO = np.meshgrid(self.la, self.lo, indexing='ij')
        self.tl = np.arange(np.floor(self.L0), self.L0 + HZ + 2)  # échéances horaires
        U = np.empty((len(scen), len(self.tl), *LA.shape), np.float32); V = np.empty_like(U)
        for k, F in enumerate(WF):
            for i, L in enumerate(self.tl):
                u, v = _uv(*F(L, LA, LO)); U[k, i], V[k, i] = u, v
        self.U, self.V = U, V
        Vt = np.empty((len(scen), len(self.tl), 4, *LA.shape), np.float32)
        for k, x in enumerate(scen):
            T = x['table'] if isinstance(x['table'], dict) else dict(np.load(x['table']))
            for i, L in enumerate(self.tl):
                y = L + x['shift'] - float(T['lead0']); kk = int(np.clip(np.floor(y), 0, len(T['values']) - 2)); f = float(np.clip(y - kk, 0, 1))
                g = dict(lat0=T['lat'][0], lon0=T['lon'][0], step=float(T['lat'][1] - T['lat'][0]))
                for j in range(4):
                    Vt[k, i, j] = bf.interp(T['values'][kk, j], LA, LO, **g) * (1 - f) + bf.interp(T['values'][kk + 1, j], LA, LO, **g) * f
        self.Vt = Vt; self.K = len(scen)
        self.n_lagged = sum(1 for n in self.names if '@' in n)

    def _c(self, kidx, L, lat, lon):
        return [kidx, (L - self.tl[0]) * np.ones_like(lat), (lat - self.la[0]) / self.st, (lon - self.lo[0]) / self.st]

    def wind(self, L, lat, lon):
        kidx = np.broadcast_to(np.arange(self.K), lat.shape).astype(float); c = self._c(kidx, L, lat, lon)
        c = [x.ravel() for x in c]
        u = map_coordinates(self.U, c, order=1, mode='nearest', prefilter=False).reshape(lat.shape)
        v = map_coordinates(self.V, c, order=1, mode='nearest', prefilter=False).reshape(lat.shape)
        return np.hypot(u, v), np.degrees(np.arctan2(-u, -v)) % 360

    def value(self, L, lat, lon, layer):
        kidx = np.broadcast_to(np.arange(self.K), lat.shape).astype(float)
        c = self._c(kidx, L, lat, lon); c.insert(2, layer.astype(float)); c = [x.ravel() for x in c]
        return map_coordinates(self.Vt, c, order=1, mode='constant', cval=2000, prefilter=False).reshape(lat.shape)

    def step(self, S, twa, sail, h):
        lat, lon, e = S['lat'], S['lon'], S['e']  # (n, K)
        tk = 1 if twa > 0 else -1; cs = (S['sail'] != sail)[:, None]; ct = (S['tack'] != tk)[:, None]
        ws, _ = self.wind(self.L0 + h, lat, lon)
        used, pen = p.maneuver(ws, e, cs, ct)
        e1 = np.maximum(0, e - used); pl = np.where(cs | ct, np.maximum(S['pen'], pen), S['pen'])
        la, lo = lat, lon; ok = np.ones(lat.shape[0], bool)
        for half in (0, 1):
            w2, d2 = self.wind(self.L0 + h + (half + .5) * DT / 2, la, lo)
            inp = np.minimum(DT / 2, pl)
            la2, lo2 = p.move(la, lo, (d2 - twa) % 360, p.polar(twa, w2, sail) * (DT / 2 - p.PEN_LOSS * inp))
            la2 = np.where(S['done'], la, la2); lo2 = np.where(S['done'], lo, lo2)
            ok &= p.leg_allowed(la, lo, la2, lo2).all(1)
            e1 = np.minimum(100, e1 + (DT / 2 - inp) * 3600 / p.recover_seconds(w2)); la, lo, pl = la2, lo2, pl - inp
        return dict(lat=la, lon=lo, e=e1, pen=pl, sail=np.full(len(la), sail), tack=np.full(len(la), tk), twa=np.full(len(la), float(twa))), ok

    def run(self, lat, lon, e, a0, s0, H, forced=None):
        K = self.K; one = lambda x: np.full((1, K), x, float)
        S = dict(lat=one(lat), lon=one(lon), e=one(e), pen=one(0.), sail=np.array([s0]), tack=np.array([1 if a0 > 0 else -1]),
                 twa=np.array([float(a0)]), n=np.zeros(1, int), done=np.zeros((1, K), bool), arr=one(np.inf), emin=np.array([e]))
        hist = [[]]; h = 0.; last = None
        while h < HZ - 1e-9:
            now = self.t0 + dt.timedelta(hours=h); prog = h < H - 1e-9
            if forced is not None and prog:
                a = [x for t, x in forced if t <= h + 1e-9][-1]; acts = [(a, s0)]
            else:
                acts = [(s * a, sl) for a in TWA for s in (-1, 1) for sl in self.sails]
            cand = []
            for twa, sl in acts:
                Nn, ok = self.step(S, float(twa), sl, h)
                chg = (np.abs(S['twa'] - twa) > 1e-6) | (S['sail'] != sl)
                n = S['n'] + (chg & (h > 1e-9) & prog)
                if prog and forced is None:
                    ok &= n <= MAXO
                    if h > 1e-9 and night_paris(now.astimezone(PARIS)): ok &= S['sail'] == sl
                m, f, toff = crossed(S['lat'], S['lon'], Nn['lat'], Nn['lon']); m &= ~S['done']
                arr = np.where(m, h + f * DT + toff, S['arr']); done = S['done'] | m
                lay = 2 * LAYER_SAIL[Nn['sail']][:, None] + (Nn['tack'] > 0)[:, None] + 0 * Nn['lat']
                v = self.value(self.L0 + h + DT, Nn['lat'], Nn['lon'], lay)
                cost = np.where(done, arr, h + DT + v)
                sc = cost @ self.w + E_W * (100 - Nn['e'] @ self.w) / 10
                Nn.update(n=n, done=done, arr=arr, emin=np.minimum(S['emin'], Nn['e'].min(1)), score=np.where(ok, sc, np.inf),
                          par=np.arange(len(sc)), act=np.full(len(sc), float(twa)), asl=np.full(len(sc), sl), cost=cost)
                cand.append(Nn)
            C = {k: np.concatenate([c[k] for c in cand]) for k in cand[0]}
            fin = np.isfinite(C['score'])
            if not fin.any(): break
            ml, mo, me = C['lat'] @ self.w, C['lon'] @ self.w, C['e'] @ self.w
            key = ((np.floor(ml / .03).astype(np.int64) * 100000 + np.floor(mo / .03).astype(np.int64)) * 40 +
                   (2 * LAYER_SAIL[C['sail']] + (C['tack'] > 0)) * 10 + np.floor(me / 25).astype(int)) * 64 + np.minimum(C['n'], 7) * 8 + C['sail']
            o = np.lexsort((C['score'], key)); first = np.r_[True, key[o][1:] != key[o][:-1]]
            sel = o[first]; sel = sel[np.isfinite(C['score'][sel])]; sel = sel[np.argsort(C['score'][sel])][:N]
            hist = [hist[C['par'][i]] + [(h, float(C['act'][i]), int(C['asl'][i]))] for i in sel]
            S = {k: C[k][sel] for k in ('lat', 'lon', 'e', 'pen', 'sail', 'tack', 'twa', 'n', 'done', 'arr', 'emin')}
            last = (float(C['score'][sel[0]]), C['cost'][sel[0]]); h += DT
            if S['done'][0].all(): break
        tr = hist[0]; orders = []
        for i, r in enumerate(tr):
            if r[0] >= H - 1e-9: break
            prev = (float(a0), int(s0)) if i == 0 else (tr[i - 1][1], tr[i - 1][2])
            if i == 0 or (r[1], r[2]) != prev: orders.append([round(r[0] * 60), int(round(r[1])), p.SAIL_NAMES[r[2]]])
        E, cost = last
        return E, orders, cost, float(S['emin'][0])


PARIS = dt.timezone(dt.timedelta(hours=1))


def cap_weights(w, cap):
    """Poids de moyenne 1 avec max <= cap exactement : w' = min(c w, cap), c trouvé par dichotomie (la renormalisation
    itérée dépassait le plafond, 3,12 au lieu de 3 : bogue trouvé par tests/test_routing.py)."""
    w = np.asarray(w, float) / np.mean(w)
    if w.max() <= cap: return w
    lo, hi = 0., 1.
    while np.minimum(w * hi, cap).mean() < 1: hi *= 2
    for _ in range(100):
        c = (lo + hi) / 2
        if np.minimum(w * c, cap).mean() < 1: lo = c
        else: hi = c
    return np.minimum(w * hi, cap) / np.minimum(w * hi, cap).mean()


def calibrated_weights(scen, ens, t, lat0, lon0):
    """Filtre bayésien calibré : w ∝ exp(-t (d² - min d²)), d = écart RMS (vecteur, nds) scénario / GFS du bulletin courant
    sur ses échéances 6..18 h (ce que le jeu va appliquer), dans une boîte autour du bateau ; t calibrée a posteriori
    (calibrate_weights.py). Le contrôle est toujours le plus proche du GFS (même modèle) : normalisation sur les autres.
    Plafond 3x la moyenne. Les membres décalés plus anciens qui contredisent le dernier GFS perdent leur poids d'eux-mêmes.
    Colonne GFS = poids moyen."""
    la, lo = np.meshgrid(np.arange(max(6, lat0 - 12), min(57, lat0 + 12) + .1, 1.),
                         np.arange(max(-74, lon0 - 18), min(4, lon0 + 18) + .1, 1.), indexing='ij')
    g = scen[0]['raw']; leads = list(range(6, 19, 3)); ref = [_uv(*g(L, la, lo)) for L in leads]; d = []
    for x in scen[1:]:
        F = x['raw']; sh = x['shift']
        d.append(np.sqrt(np.mean([np.mean((u - ur) ** 2 + (v - vr) ** 2) for (u, v), (ur, vr) in zip([_uv(*F(L + sh, la, lo)) for L in leads], ref)])))
    if not d: return np.ones(1)
    d = np.array(d); ctl = [i for i, x in enumerate(scen[1:]) if x['name'].startswith('gefsc00')]
    ref_min = np.delete(d, ctl).min() if ctl and len(d) > len(ctl) else d.min()
    w = np.exp(-t * np.maximum(d ** 2 - ref_min ** 2, 0)); w = w / w.mean()
    w = cap_weights(w, 3.)
    w = np.r_[w.mean(), w]
    print('SCEN_T', t, 'scénarios', len(w), 'dont décalés', sum('@' in x['name'] for x in scen), 'ESS', round(w.sum() ** 2 / (w ** 2).sum(), 1), flush=True)
    return w


def main():
    a = sys.argv[1:]; tic = time.time()
    cyc, obs = a[0], dt.datetime.strptime(a[1], '%Y%m%d%H%M').replace(tzinfo=UTC)
    lat, lon, e, a0, s0, H = float(a[2]), float(a[3]), float(a[4]), int(a[5]), int(a[6]), float(a[7])
    cur = json.loads(a[8]) if len(a) > 8 else None
    if s0 not in SAILS: s0 = int(LAYER_SAIL[s0])
    B = Stoch(cyc, obs, lat, lon, sorted(set(SAILS) | {s0}))
    E, orders, cost, emin = B.run(lat, lon, e, a0, s0, H)
    res = dict(E=E, orders=orders, per_scenario=np.round(cost, 2).tolist(), names=B.names, weights=np.round(B.w, 4).tolist(), n_lagged=B.n_lagged,
               min_energy=emin, H=H, horizon=HZ, beam=N, K=B.K, max_orders=MAXO, sails=SAILS, seconds=round(time.time() - tic, 1))
    if cur:
        cur = sorted([(0., a0)] + [(float(t), int(x)) for t, x in cur if float(t) > 0])
        res['ref_E'] = B.run(lat, lon, e, a0, s0, H, forced=cur)[0]
    print('BEAMS ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
