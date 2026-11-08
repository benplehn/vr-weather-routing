"""Routeur à faisceau (programme principal des mails depuis le 7/11).

Pourquoi : le bilan a posteriori (docs/METHOD.md) a montré qu'un routeur GFS déterministe réactualisé à chaque bulletin
avec un faisceau de 24 h arrive à ~2,7 h de l'optimum clairvoyant, alors que l'exploitation à horizon court de la table
Bellman en perdait ~10 ; le stochastique (beam_stoch.py) fait jeu égal pour 4x le calcul et sert de contrôle.

plan() = la seule façon de produire un programme :
  1. faisceau (BEAM_N états, pas BEAM_DT_MIN) sur BEAM_HORIZON h dans le vent du jeu, physique complète (énergie,
     pénalités qui débordent sur les pas suivants, régulateur TWA), coût terminal = table Bellman du bulletin ;
     contraintes sur la fenêtre du programme [0, H] : ≤ BEAM_ORDERS ordres après « maintenant », voile changée seulement
     maintenant ou de jour (NIGHT, Paris) ; au-delà de H : libre (c'est la replanification suivante) ;
  2. candidats : programme du faisceau + programme précédent décalé (démarrage à chaud, BEAM_PREV) ;
  3. affinage continu de chaque candidat (heures à la minute, angles au degré, retrait d'ordres) qui PROPOSE des
     variantes ; le choix se fait sur la continuation (programme jusqu'à H puis petit faisceau jusqu'à BEAM_HORIZON),
     jamais sur la seule table à H (myope : rejeu du 4/11, front manqué) ;
  4. VÉRIFICATION OBLIGATOIRE au pas d'1 minute (terre, zone interdite, marge côtière VERIFY_CLR hors approche de
     l'arrivée) : un programme qui échoue n'est jamais proposé (7/11 : un programme validé au pas de 7 min touchait
     la pointe de Vieux-Fort dans le jeu) ;
  5. le programme entré dans le jeu est évalué par la même fonction (gain comparable).
FINISH_REAL=1 : simulation jusqu'à la vraie ligne d'arrivée (côte ouest, sud de Basse-Terre, Pointe-à-Pitre) au lieu de
la ligne de passage des tables + estimation forfaitaire.

Usage : beam_prog.py CYCLE OBS(YYYYmmddHHMM UTC) lat lon énergie angle voile H [programme entré JSON [[h, angle], ...]]
Sortie : ligne 'BEAM {json}' : orders [[minute, angle signé, voile]], E (h jusqu'à la bouée NO, ou jusqu'à la ligne si
FINISH_REAL), ref_E (programme entré), verified, warnings, min_energy, track du faisceau.
"""
import os, sys, json, datetime as dt
from pathlib import Path
import numpy as np
os.environ.setdefault('FULL_PACK', '1')  # polaire des 7 voiles chargée (indices 0 Jib, 1 Spi inchangés) ; BEAM_SAILS choisit
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'stochastic'))
import optimal_route as p, bellman_fine as bf, game_wind

UTC = dt.timezone.utc; PARIS = dt.timezone(dt.timedelta(hours=1))  # CET (pas d'heure d'été en novembre)
N = int(os.environ.get('BEAM_N', '1500'))               # états gardés par pas
HZ = float(os.environ.get('BEAM_HORIZON', '24'))        # horizon du faisceau (h)
DT = float(os.environ.get('BEAM_DT_MIN', '15')) / 60    # pas de décision du faisceau (h)
MAXO = int(os.environ.get('BEAM_ORDERS', '3'))          # ordres programmables après « maintenant »
NIGHT = os.environ.get('NIGHT', '22:30-10:00')          # absence du skipper (Paris) : pas de changement de voile
TWA = np.r_[np.arange(40, 170, 5), 168.]
SAILS = [int(x) for x in os.environ.get('BEAM_SAILS', '0,1').split(',')]  # 0 Jib 1 Spi 2 Staysail 3 LightJib 4 Code0 5 HeavyGnk 6 LightGnk
LAYER_SAIL = np.array([0, 1, 0, 0, 0, 1, 1])  # tables Jib/Spi seulement : près/reaching -> Jib, portant -> Spi
E_W = .02                                      # h par 10 points d'énergie manquants (pénalités futures plus longues)
FINISH_REAL = os.environ.get('FINISH_REAL', '0') == '1'
FIN_R = float(os.environ.get('FIN_R_NM', '1.5'))      # arrivée = à moins de FIN_R NM du point de la ligne
CLR = float(os.environ.get('BEAM_CLEARANCE', '2'))    # marge côtière du faisceau (NM, p.allowed impose déjà 2)
VERIFY_CLR = float(os.environ.get('VERIFY_CLR', '2'))  # marge exigée à la vérification minute par minute
FINISH_ZONE_NM = 6.                                    # près de la ligne (côtière) la marge tombe à 1 NM
SPD_TAIL = float(os.environ.get('TAIL_SPD', '14'))    # vitesse moyenne estimée après la ligne de passage


def night_paris(t):
    """t (datetime aware, Paris) dans la nuit NIGHT 'HH:MM-HH:MM' ?"""
    a, b = [int(x[:2]) * 60 + int(x[3:]) for x in NIGHT.split('-')]; m = t.hour * 60 + t.minute
    return m >= a or m < b if a > b else a <= m < b


LINE_E_SAFE = -61.92  # bout est de la ligne exigé par plan/vérification (marge : 7/11, un passage à 61,8818 W « validé »
                     # au pas d'1 min n'en était plus un au pas de 5 min, et le bateau continuait dans Basse-Terre)


def crossed(lat0, lon0, lat1, lon1, east=-61.88):
    """Passage de la ligne de fin des tables (16.25 N, de 62.8 W à `east`) : (masque, fraction du pas, valeur de la ligne)."""
    m = (lat0 > 16.25) & (lat1 <= 16.25)
    f = np.where(m, (lat0 - 16.25) / np.maximum(lat0 - lat1, 1e-9), 0); lx = lon0 + f * (lon1 - lon0)
    m &= (lx <= east) & (lx >= -62.8)
    _, toff = p.goal_target(np.full(lx.shape, 16.25), np.clip(lx, -62.8, -61.88))
    return m, f, toff


def tail_left(lat, lon):
    """Heures estimées après la ligne de passage : côte ouest de Basse-Terre -> pointe sud (TAIL_WP) -> ligne d'arrivée."""
    d1 = p.distance(lat, lon, p.TAIL_WP) + float(p.distance(np.array([p.TAIL_WP[0]]), np.array([p.TAIL_WP[1]]), p.FINISH)[0])
    d2 = p.distance(lat, lon, p.FINISH)
    return np.where((lat > p.TAIL_WP[0] + .05) & (lon < -61.70), d1, d2) / SPD_TAIL


def coast_ok(la, lo, la2, lo2, req):
    """Marge côtière req (NM) au point d'arrivée et au milieu du segment, 1 NM seulement près de la ligne d'arrivée."""
    req = np.where(p.distance(la2, lo2, p.FINISH) < FINISH_ZONE_NM, 1., req)
    return (p.clearance(la2, lo2) > req) & (p.clearance((la + la2) / 2, (lo + lo2) / 2) > req)


class Beam:
    """Faisceau depuis l'heure obs. Par défaut vent du jeu (game_wind) et table du bulletin data/ens_CYCLE ; wind (lead C,
    lat, lon) -> (tws, twd) et table (dict bellman_fine) peuvent être injectés (rejeux)."""
    def __init__(self, cyc, obs, sail_set, wind=None, table=None):
        self.c0 = dt.datetime.strptime(cyc, '%Y%m%d%H').replace(tzinfo=UTC); self.t0 = obs
        self.L0 = (obs - self.c0).total_seconds() / 3600
        ens = R / 'data' / f'ens_{cyc}'
        self.F = wind or game_wind.fields(ens, ['gfs'], cyc)[0]
        self.G = table if table is not None else dict(np.load(ens / 'bellman_fine.npz'))
        self.sails = sail_set

    def step(self, S, twa, sail, h, step_h=None):
        """Avance tous les états S d'un pas (DT, ou step_h) à TWA constante (twa signé, + tribord) avec la voile sail.
        Les pénalités (manœuvre, changement de voile) ralentissent à PEN_LOSS et débordent sur les pas suivants."""
        st = step_h or DT
        lat, lon, e = S['lat'], S['lon'], S['e']
        tk = np.full(lat.shape, 1 if twa > 0 else -1); cs = S['sail'] != sail; ct = tk != S['tack']
        ws, _ = self.F(self.L0 + h, lat, lon)
        used, pen = p.maneuver(ws, e, cs, ct)
        e1 = np.maximum(0, e - used); pl = np.where(cs | ct, np.maximum(S['pen'], pen), S['pen'])
        la, lo = lat, lon; ok = np.ones(lat.shape, bool)
        for half in (0, 1):
            w2, d2 = self.F(self.L0 + h + (half + .5) * st / 2, la, lo)
            inp = np.minimum(st / 2, pl)
            la2, lo2 = p.move(la, lo, (d2 - twa) % 360, p.polar(twa, w2, sail) * (st / 2 - p.PEN_LOSS * inp))
            ok &= p.leg_allowed(la, lo, la2, lo2)
            if CLR > 2: ok &= coast_ok(la, lo, la2, lo2, CLR)
            e1 = np.minimum(100, e1 + (st / 2 - inp) * 3600 / p.recover_seconds(w2)); la, lo, pl = la2, lo2, pl - inp
        return dict(lat=la, lon=lo, e=e1, sail=np.full(la.shape, sail), tack=tk, pen=pl, twa=np.full(la.shape, float(twa))), ok

    def _start(self, lat, lon, e, a0, s0):
        return dict(lat=np.array([lat]), lon=np.array([lon]), e=np.array([e]), sail=np.array([s0]), tack=np.array([1 if a0 > 0 else -1]),
                    pen=np.zeros(1), twa=np.array([float(a0)]))

    def run(self, lat, lon, e, a0, s0, H):
        """Faisceau. Retourne (E heures, ordres [[minute, angle, voile]], trace [[minute, lat, lon]], énergie mini)."""
        S = self._start(lat, lon, e, a0, s0); S.update(n=np.zeros(1, int), emin=np.array([e]), ps=np.zeros(1, bool))
        hist = [[]]; best = (np.inf, None, None); h = 0.; last = np.inf
        while h < HZ - 1e-9:
            now = self.t0 + dt.timedelta(hours=h); prog = h < H - 1e-9
            cand = []
            for twa, sl in [(s * a, sl) for a in TWA for s in (-1, 1) for sl in self.sails]:
                Nn, ok = self.step(S, float(twa), sl, h)
                chg = (np.abs(S['twa'] - twa) > 1e-6) | (S['sail'] != sl)
                n = S['n'] + (chg & (h > 1e-9) & prog)
                if prog:
                    ok &= n <= MAXO
                    if h > 1e-9 and night_paris(now.astimezone(PARIS)): ok &= S['sail'] == sl
                m, f, toff = crossed(S['lat'], S['lon'], Nn['lat'], Nn['lon'], LINE_E_SAFE)
                Nn['n'] = n; Nn['emin'] = np.minimum(S['emin'], Nn['e'])
                if FINISH_REAL:  # la ligne de passage n'est qu'une étape ; arrivée = près du point de la ligne d'arrivée
                    Nn['ps'] = S['ps'] | m
                    m = Nn['ps'] & (p.distance(Nn['lat'], Nn['lon'], p.FINISH) < FIN_R); f = np.ones(len(m)); toff = np.zeros(len(m))
                else: Nn['ps'] = S['ps']
                arr = m & ok
                if arr.any():
                    E = np.where(arr, h + f * DT + toff, np.inf); i = int(np.argmin(E))
                    if E[i] < best[0]: best = (float(E[i]), hist[i] + [(h, float(twa), int(sl), float(Nn['lat'][i]), float(Nn['lon'][i]))], float(Nn['emin'][i]))
                v = bf.value(self.G, self.L0 + h + DT, Nn['lat'], Nn['lon'], LAYER_SAIL[Nn['sail']], Nn['tack'])
                if FINISH_REAL: v = np.where(Nn['ps'], tail_left(Nn['lat'], Nn['lon']), v + p.FINISH_TAIL_H)
                Nn['score'] = np.where(ok & ~m, h + DT + v + E_W * (100 - Nn['e']) / 10, np.inf)
                Nn['par'] = np.arange(len(m)); Nn['act'] = np.full(len(m), float(twa)); Nn['as'] = np.full(len(m), sl)
                cand.append(Nn)
            C = {k: np.concatenate([c[k] for c in cand]) for k in cand[0]}
            fin = np.isfinite(C['score'])
            if not fin.any() or C['score'][fin].min() >= best[0]: break
            # un représentant par case 0,03° x couche (voile, amure) x tranche d'énergie x ordres utilisés x ligne passée
            key = ((np.floor(C['lat'] / .03).astype(np.int64) * 100000 + np.floor(C['lon'] / .03).astype(np.int64)) * 40 +
                   (2 * C['sail'] + (C['tack'] > 0)) * 10 + np.floor(C['e'] / 25).astype(int)) * 16 + np.minimum(C['n'], 7) * 2 + C['ps']
            o = np.lexsort((C['score'], key)); first = np.r_[True, key[o][1:] != key[o][:-1]]
            sel = o[first]; sel = sel[np.isfinite(C['score'][sel])]; sel = sel[np.argsort(C['score'][sel])][:N]
            hist = [hist[C['par'][i]] + [(h, float(C['act'][i]), int(C['as'][i]), float(C['lat'][i]), float(C['lon'][i]))] for i in sel]
            S = {k: C[k][sel] for k in ('lat', 'lon', 'e', 'sail', 'tack', 'pen', 'twa', 'n', 'emin', 'ps')}; last = float(C['score'][sel[0]]); h += DT
        if np.isfinite(best[0]): E, tr, emin = best
        else: E, tr, emin = last, hist[0], float(S['emin'][0])  # pas d'arrivée : meilleur état final (temps + valeur)
        orders = []
        for i, r in enumerate(tr):
            if r[0] >= H - 1e-9: break
            prev = (float(a0), int(s0)) if i == 0 else (tr[i - 1][1], tr[i - 1][2])
            if i == 0 or (r[1], r[2]) != prev: orders.append([round(r[0] * 60), int(round(r[1])), p.SAIL_NAMES[r[2]]])
        return E, orders, [[round(r[0] * 60), round(r[3], 4), round(r[4], 4)] for r in tr], emin

    def simulate(self, lat, lon, e, a0, s0, H, prog, step_h=5 / 60, clr=None):
        """Navigue un programme [[minute, angle, voile]] (TWA tenue entre les ordres) jusqu'à H ou l'arrivée.
        Retourne dict(J coût, ok, why, h, lat, lon, emin, min_clr, arrived). clr : marge côtière vérifiée en plus."""
        prog = sorted([[m, a, p.SAIL_NAMES.index(s) if isinstance(s, str) else int(s)] for m, a, s in prog])
        out = dict(ok=True, why='', arrived=False, min_clr=np.inf)
        if sum(1 for m, *_ in prog if m > 0) > MAXO: return dict(out, ok=False, why=f'plus de {MAXO} ordres', J=np.inf)
        S = self._start(lat, lon, e, a0, s0); cur = (float(a0), int(s0)); h = 0.; k = 0; passed = False; emin = e
        while h < H - 1e-9:
            while k < len(prog) and prog[k][0] <= h * 60 + 1e-6:
                m, a, sl = prog[k]
                if m > 0 and sl != cur[1] and night_paris((self.t0 + dt.timedelta(minutes=m)).astimezone(PARIS)):
                    return dict(out, ok=False, why='voile changée la nuit', J=np.inf)
                cur = (float(a), sl); k += 1
            st = max(min(([prog[k][0] / 60] if k < len(prog) else []) + [H, h + step_h]) - h, 1e-3)
            Nn, ok = self.step(S, cur[0], cur[1], h, st)
            if clr is not None:
                far = p.distance(Nn['lat'], Nn['lon'], p.FINISH)[0] > FINISH_ZONE_NM
                if far: out['min_clr'] = min(out['min_clr'], float(p.clearance(Nn['lat'], Nn['lon'])[0]))
                ok &= coast_ok(S['lat'], S['lon'], Nn['lat'], Nn['lon'], clr)
            if not ok[0]:
                t = self.t0 + dt.timedelta(hours=h + st)
                return dict(out, ok=False, why=f"terre/marge à {t.astimezone(PARIS):%H:%M} ({Nn['lat'][0]:.3f}, {Nn['lon'][0]:.3f})", J=np.inf)
            emin = min(emin, float(Nn['e'][0]))
            m_, f, toff = crossed(S['lat'], S['lon'], Nn['lat'], Nn['lon'], LINE_E_SAFE)
            if FINISH_REAL:
                passed = passed or bool(m_[0])
                if passed and p.distance(Nn['lat'], Nn['lon'], p.FINISH)[0] < FIN_R:
                    return dict(out, J=float(h + st), h=h + st, emin=emin, arrived=True, lat=float(Nn['lat'][0]), lon=float(Nn['lon'][0]))
            elif m_[0]:
                return dict(out, J=float(h + f[0] * st + toff[0]), h=h + st, emin=emin, arrived=True, lat=float(Nn['lat'][0]), lon=float(Nn['lon'][0]))
            S = Nn; h += st
        v = bf.value(self.G, self.L0 + h, S['lat'], S['lon'], LAYER_SAIL[S['sail']], S['tack'])[0]
        if FINISH_REAL: v = float(tail_left(S['lat'], S['lon'])[0]) if passed else v + p.FINISH_TAIL_H
        return dict(out, J=float(h + v + E_W * (100 - S['e'][0]) / 10), h=h, emin=emin, lat=float(S['lat'][0]), lon=float(S['lon'][0]), S=S)

    def continuation(self, lat, lon, e, a0, s0, H, prog):
        """Coût d'un programme jugé sur tout l'horizon : le programme jusqu'à H, puis un petit faisceau libre (CONT_N
        états) jusqu'à BEAM_HORIZON. Sert à CHOISIR entre candidats : juger à H avec la seule table relaxée est myope
        (rejeu 4/11 : passage d'un front raté, le bateau filait vers l'est)."""
        global N, HZ
        r = self.simulate(lat, lon, e, a0, s0, H, prog)
        if not r['ok']: return np.inf
        if r['arrived'] or HZ - H < .25: return r['J']
        S = r['S']; B2 = Beam.__new__(Beam); B2.__dict__.update(self.__dict__)
        B2.t0 = self.t0 + dt.timedelta(hours=H); B2.L0 = self.L0 + H
        n0, hz0 = N, HZ; N, HZ = int(os.environ.get('CONT_N', '300')), HZ - H
        try:
            E2 = B2.run(float(S['lat'][0]), float(S['lon'][0]), float(S['e'][0]), float(S['twa'][0]), int(S['sail'][0]), 0.)[0]
        finally:
            N, HZ = n0, hz0
        return float(H + E2 + 0 * r['J'])

    def evaluate(self, lat, lon, e, a0, s0, H, prog):
        return self.simulate(lat, lon, e, a0, s0, H, prog)['J']

    def verify(self, lat, lon, e, a0, s0, H, prog):
        """Contrôle au pas d'1 minute avec la marge VERIFY_CLR (hors approche de l'arrivée)."""
        return self.simulate(lat, lon, e, a0, s0, max(H, 1.), prog, step_h=1 / 60, clr=VERIFY_CLR)

    def refine(self, lat, lon, e, a0, s0, H, prog, passes=4):
        """Descente coordonnée : heure de chaque ordre (±30/15/5/2/1 min), angle (±5/2/1°), retrait d'un ordre."""
        ev = lambda q: self.evaluate(lat, lon, e, a0, s0, H, q)
        best = [list(x) for x in prog]; J = ev(best); n_ev = 1
        for _ in range(passes):
            gain = False
            for i in range(len(best)):
                for kind, deltas in (('t', (30, 15, 5, 2, 1)), ('a', (5, 2, 1))):
                    for d in deltas:
                        for sgn in (-1, 1):
                            q = [list(x) for x in best]
                            if kind == 't':
                                if q[i][0] == 0: continue
                                q[i][0] += sgn * d
                                if not (0 < q[i][0] < H * 60) or any(q[j][0] == q[i][0] for j in range(len(q)) if j != i): continue
                            else:
                                na = q[i][1] + sgn * d
                                if not (35 <= abs(na) <= 175) or np.sign(na) != np.sign(q[i][1]): continue
                                q[i][1] = na
                            Jq = ev(q); n_ev += 1
                            if Jq < J - 1e-4: best, J, gain = sorted(q), Jq, True
            for i in range(1, len(best)):
                q = best[:i] + best[i + 1:]; Jq = ev(q); n_ev += 1
                if Jq < J - 1e-4: best, J, gain = q, Jq, True; break
            if not gain: break
        return best, J, n_ev


def to_minutes(prev, obs):
    """Programme [[utc 'YYYYmmddHHMM', angle, voile], ...] -> [[minute depuis obs, angle, voile]] ; les ordres déjà passés
    sont fusionnés en un seul ordre « maintenant » (le dernier passé est celui en vigueur)."""
    out = {}
    for u, a, sl in sorted(prev):
        m = max(0, round((dt.datetime.strptime(u, '%Y%m%d%H%M').replace(tzinfo=UTC) - obs).total_seconds() / 60))
        out[m] = [m, int(a), sl]
    return [out[k] for k in sorted(out)]


def plan(B, lat, lon, e, a0, s0, H, prev=None, entered=None, refine=True):
    """Programme vérifié depuis l'état (lat, lon, énergie, angle courant, voile). prev : programme précédent en minutes
    (démarrage à chaud) ; entered : programme entré dans le jeu en minutes (évalué pour le gain). Retourne un dict.
    Si aucun candidat ne passe la vérification minute par minute (près des îles, toutes les routes du faisceau rasent la
    marge : rejeu du 7/11, échouage sur Montserrat), on relance avec une marge côtière élargie (+1 puis +2 NM)."""
    global CLR
    base = CLR
    try:
        for extra in (0., 1., 2.):
            CLR = max(base, 2.) + extra if extra else base
            res = _plan(B, lat, lon, e, a0, s0, H, prev, entered, refine)
            res['beam_clearance'] = max(CLR, 2.)
            if res['verified']: break
    finally:
        CLR = base
    return res


def _plan(B, lat, lon, e, a0, s0, H, prev, entered, refine):
    E, orders, tr, emin = B.run(lat, lon, e, a0, s0, H)
    sn = p.SAIL_NAMES[s0]
    cands = {'faisceau': orders or [[0, int(a0), sn]]}
    if prev:
        pv = [list(x) for x in prev]
        if pv[0][0] > 0: pv = [[0, int(a0), sn]] + pv
        cands['précédent'] = pv
    res = dict(beam_E=E, track=tr, H=H, horizon=HZ, beam=N, dt_min=DT * 60, max_orders=MAXO, sails=SAILS, candidates={}, warnings=[])
    # chaque candidat brut et affiné ; l'affinage (myope : jugé à H + table) propose, la continuation (faisceau jusqu'à
    # BEAM_HORIZON) décide ; un candidat qui ne passe pas la vérification minute par minute n'est jamais retenu
    for k, q in list(cands.items()):
        if refine:
            qb, _, n_ev = B.refine(lat, lon, e, a0, s0, H, q)
            if qb != q: cands[k + ' affiné'] = qb
    for k, q in cands.items():
        chk = B.verify(lat, lon, e, a0, s0, H, q)
        J = B.continuation(lat, lon, e, a0, s0, H, q) if chk['ok'] else np.inf
        res['candidates'][k] = dict(J=J, orders=q, verified=chk['ok'], why=chk['why'], min_clr=chk['min_clr'])
    ok = {k: c for k, c in res['candidates'].items() if c['verified'] and np.isfinite(c['J'])}
    if ok: k = min(ok, key=lambda k: ok[k]['J'])
    else:
        k = min(res['candidates'], key=lambda k: res['candidates'][k]['J'])
        res['warnings'].append('AUCUN programme ne passe la vérification minute par minute : ' + res['candidates'][k]['why'])
    c = res['candidates'][k]
    res.update(source=k, E=c['J'], verified=c['verified'], min_clearance=c['min_clr'],
               orders=[[int(m), int(a_), sl if isinstance(sl, str) else p.SAIL_NAMES[sl]] for m, a_, sl in c['orders']])
    res['min_energy'] = B.simulate(lat, lon, e, a0, s0, H, res['orders']).get('emin', emin)
    if entered:
        q = [[0, int(a0), sn]] + [list(x) for x in entered if x[0] > 0]
        r = B.verify(lat, lon, e, a0, s0, H, q); res['ref_E'] = B.continuation(lat, lon, e, a0, s0, H, q) if r['ok'] else np.inf
        if not r['ok']: res['warnings'].append('Programme entré : ' + r['why'])
    return res


def main():
    a = sys.argv[1:]
    cyc, obs = a[0], dt.datetime.strptime(a[1], '%Y%m%d%H%M').replace(tzinfo=UTC)
    lat, lon, e, a0, s0, H = float(a[2]), float(a[3]), float(a[4]), int(a[5]), int(a[6]), float(a[7])
    if s0 not in SAILS: s0 = int(LAYER_SAIL[s0])  # voile du pack expiré -> Jib/Spi
    entered = [[round(float(t) * 60), int(x), p.SAIL_NAMES[s0]] for t, x in json.loads(a[8])] if len(a) > 8 else None
    prev = to_minutes(json.loads(os.environ['BEAM_PREV']), obs) if os.environ.get('BEAM_PREV') else None
    B = Beam(cyc, obs, sorted(set(SAILS) | {s0}))
    res = plan(B, lat, lon, e, a0, s0, H, prev=prev, entered=entered, refine=os.environ.get('REFINE', '1') == '1')
    print('BEAM ' + json.dumps(res, default=float), flush=True)


if __name__ == '__main__':
    main()
