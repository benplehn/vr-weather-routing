"""Rejeu « temps réel » de la course avec la stratégie actuelle : à chaque programme, seulement ce qui était disponible.

Information à l'heure t d'un programme :
  - table Bellman et GFS du dernier bulletin C tel que C + 4 h <= t (temps de téléchargement + calcul) ;
  - vent du jeu connu (game_wind.GameWind) : GFS C-6 jusqu'à C+6 (le jeu applique un bulletin 6 h après son heure),
    puis GFS C jusqu'à C+12, puis prévision C ;
  - vent qui arrive : si le GFS du bulletin suivant C+6 est déjà publié (t >= C+6+3,5 h) mais pas encore sa table,
    il remplace la prévision sur [C+12, C+18) (game_wind.NextWind), comme le calcul rapide de nuit ;
  - stochastique (POLICY=stoch) : membres GEFS du bulletin + ceux des bulletins précédents (ensemble décalé, LAGGED),
    filtre bayésien calibré (beam_stoch.py) ; le programme stochastique est vérifié minute par minute, sinon repli
    sur le programme déterministe.
Programme : stochastic/beam_prog.plan (faisceau + programme précédent + affinage + vérification minute par minute),
contraintes du skipper (3 ordres, voile figée la nuit), calendrier 10:00 / 11:00 / 17:00 / 22:00 Paris.
Exécution : vent vrai du jeu (hindsight/data/gfs3h.npz), pas de 5 min, physique identique.

Usage : replay_live.py POLICY(det|stoch) SRC(mpc|server) TAG [START_IDX]
  SRC=mpc    : tables « grande boîte » reconstruites a posteriori (hindsight/data/mpc, local), course entière ;
  SRC=server : tables et membres réels de chaque bulletin (data/ens_C, serveur, à partir du 3/11).
"""
import os, sys, json, glob, time, datetime as dt
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; R = H.parent
POL, SRC, TAG = sys.argv[1], sys.argv[2], sys.argv[3]; I0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
if SRC == 'mpc':
    os.environ.setdefault('LON_MIN', '-74'); os.environ.setdefault('ROUTING_SCENARIO', str(H / 'scenario.json'))
os.environ.setdefault('FINISH_REAL', '0')
sys.path[:0] = [str(R), str(R / 'stochastic'), str(H)]
import optimal_route as p, bellman_fine as bf, game_wind
import beam_prog as bp
from track import TRACK
T0 = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc); UTC = dt.timezone.utc
TAIL = p.FINISH_TAIL_H
ENDGAME_NM = 400.  # en deçà, planification jusqu'à la vraie ligne d'arrivée (FINISH_REAL)


def cdir(c): return H / 'data/mpc' / c if SRC == 'mpc' else R / 'data' / f'ens_{c}'


CYC = {}
for d in sorted((H / 'data/mpc').glob('20*')) if SRC == 'mpc' else [Path(x) for x in sorted(glob.glob(str(R / 'data/ens_20261*')))]:
    c = d.name[-10:]
    if c.isdigit() and (d / 'bellman_fine.npz').exists(): CYC[(dt.datetime.strptime(c, '%Y%m%d%H').replace(tzinfo=UTC) - T0).total_seconds() / 3600] = c


def ch(c): return (dt.datetime.strptime(c, '%Y%m%d%H').replace(tzinfo=UTC) - T0).total_seconds() / 3600


def gfs(c):
    f = cdir(c) / 'gfs3h.npz'
    return bf.Field(f) if f.exists() else None


def planning_wind(c, t):
    """Vent du jeu tel qu'on le connaît à l'heure t (heures depuis T0), en échéances du bulletin c."""
    prev = gfs((dt.datetime.strptime(c, '%Y%m%d%H') - dt.timedelta(hours=6)).strftime('%Y%m%d%H'))
    new = gfs(c); W = game_wind.GameWind(prev, new, new, 6., 12.); used_next = False
    nc = (dt.datetime.strptime(c, '%Y%m%d%H') + dt.timedelta(hours=6)).strftime('%Y%m%d%H')
    if t >= ch(c) + 6 + 3.5 and gfs(nc) is not None:
        W = game_wind.NextWind(W, gfs(nc), new, 12.); used_next = True
    return W, used_next


truth = bp.Beam('2026110100', T0, [0, 1], wind=bf.Field(H / 'data/gfs3h.npz'), table={})  # F(lead depuis T0)
truth.L0 = 0.


def main():
    u, la, lo, s, tk, _ = TRACK[I0]
    t = (dt.datetime.strptime(u, '%Y%m%d%H%M').replace(tzinfo=UTC) - T0).total_seconds() / 3600
    S = dict(lat=np.array([la]), lon=np.array([lo]), e=np.array([100. if I0 == 0 else 73.]), sail=np.array([s]),
             tack=np.array([tk]), pen=np.zeros(1), twa=np.array([(50. if s == 0 else 140.) * tk]))
    plans = [t]; d = T0
    while (d - T0).total_seconds() / 3600 < 240:
        for hh in (9, 10, 16, 21):
            x = (d + dt.timedelta(hours=hh) - T0).total_seconds() / 3600
            if x > t + .2: plans.append(x)
        d += dt.timedelta(days=1)
    eta = eta_line = None; hist = []; log = []; prev_abs = None; warn = []; aground = passed = False
    for k, tp in enumerate(plans):
        if eta is not None or aground or tp > 238: break
        tn = plans[k + 1]; tic = time.time()
        c = CYC[max(x for x in CYC if x + 4 <= tp + 1e-9)]
        obs = T0 + dt.timedelta(hours=tp)
        a0 = float(S['twa'][0]); s0 = int(S['sail'][0]); lat, lon, e = float(S['lat'][0]), float(S['lon'][0]), float(S['e'][0])
        # fin de course : on planifie jusqu'à la vraie ligne (le jeu continue après la ligne de passage des tables)
        bp.FINISH_REAL = float(p.distance(np.array([lat]), np.array([lon]), p.FINISH)[0]) < ENDGAME_NM
        W, used_next = planning_wind(c, tp)
        G = dict(np.load(cdir(c) / 'bellman_fine.npz'))
        B = bp.Beam(c, obs, [0, 1], wind=W, table=G)
        prev = None
        if prev_abs:  # démarrage à chaud : programme précédent en heures absolues -> minutes depuis obs
            pv = {}
            for ta, a_, sl in prev_abs:
                m = max(0, round((ta - tp) * 60)); pv[m] = [m, a_, sl]
            prev = [pv[x] for x in sorted(pv)]
        res = bp.plan(B, lat, lon, e, a0, s0, tn - tp, prev=prev)
        src = res['source']
        if POL == 'stoch':
            import beam_stoch as bs
            nm = len(glob.glob(str(cdir(c) / 'bellman_members/*.npz'))); bs.TERM = 'member' if nm >= 15 else 'gfs'
            try:
                St = bs.Stoch(c, obs, lat, lon, [0, 1]); Es, orders_s, _, _ = St.run(lat, lon, e, a0, s0, tn - tp)
                chk = B.verify(lat, lon, e, a0, s0, tn - tp, orders_s)
                if chk['ok']: res['orders'], src = orders_s, f'stoch({St.K} scén., {St.n_lagged} décalés)'
                else: warn.append(f'{obs:%d/%m %H:%M} stoch rejeté : {chk["why"]}')
            except Exception as x:
                warn.append(f'{obs:%d/%m %H:%M} stoch échec {x}')
        orders = res['orders']; prev_abs = [(tp + m / 60, a_, sl) for m, a_, sl in orders]
        log.append(dict(t=tp, cycle=c, orders=orders, source=src, verified=res['verified'], next_gfs=used_next, sec=round(time.time() - tic)))
        print(f"{obs + dt.timedelta(hours=1):%d/%m %H:%M} Paris  bulletin {c[6:]}{' +GFS suivant' if used_next else ''}  "
              f"pos {lat:.2f} {lon:.2f} E{e:.0f}  {src}{'' if res['verified'] else ' NON VÉRIFIÉ'}  {orders}  ({time.time() - tic:.0f}s)", flush=True)
        h = tp; ex = sorted([[m, a_, p.SAIL_NAMES.index(sl)] for m, a_, sl in orders])
        while h < tn - 1e-9:
            cur = [o for o in ex if o[0] <= (h - tp) * 60 + 1e-6]
            twa, sl = (cur[-1][1], cur[-1][2]) if cur else (a0, s0)
            Nn, ok = truth.step(S, float(twa), sl, h, 5 / 60)
            if not ok[0]:  # échouage : le rejeu s'arrête (avant, l'exécution traversait l'île)
                warn.append(f'TERRE dans le vent vrai {T0 + dt.timedelta(hours=h + 1):%d/%m %H:%M} Paris ({Nn["lat"][0]:.3f}, {Nn["lon"][0]:.3f})')
                print('  !!', warn[-1], flush=True); aground = True; break
            mm, f, toff = bp.crossed(S['lat'], S['lon'], Nn['lat'], Nn['lon'])
            if mm[0] and eta_line is None: eta_line = h + f[0] * 5 / 60 + toff[0] + TAIL  # comparable aux bornes (ligne + 2,5 h)
            passed = passed or bool(mm[0])
            hist.append((h, float(Nn['lat'][0]), float(Nn['lon'][0]), float(twa), int(sl), float(Nn['e'][0])))
            S = Nn; h += 5 / 60
            if passed and p.distance(S['lat'], S['lon'], p.FINISH)[0] < 1.5: eta = h; break  # vraie ligne d'arrivée
        if aground: break
    out = dict(policy=POL, src=SRC, tag=TAG, start=TRACK[I0][0], eta_h=eta, eta_line_h=eta_line, aground=aground,
               eta_utc=None if eta is None else f'{T0 + dt.timedelta(hours=eta):%Y-%m-%d %H:%M}Z',
               eta_line_utc=None if eta_line is None else f'{T0 + dt.timedelta(hours=eta_line):%Y-%m-%d %H:%M}Z', warnings=warn,
               seconds=[x['sec'] for x in log], track=hist, programs=log)
    (H / 'out').mkdir(exist_ok=True); (H / 'out' / f'live_{TAG}.json').write_text(json.dumps(out, default=float))
    print('ARRIVÉE (vraie ligne)', out['eta_utc'], '| ligne de passage + 2,5 h', out['eta_line_utc'], '| ÉCHOUAGE' if aground else '', '| programmes', len(log), '| calcul moyen', round(np.mean(out['seconds'])), 's | alertes', len(warn))


if __name__ == '__main__':
    main()
