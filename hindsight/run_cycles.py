"""Tables Bellman « routeur GFS déterministe » de chaque bulletin (même grande boîte, LON_MIN -74) pour le rejeu MPC.
hindsight/data/mpc/C/bellman_fine.npz : V depuis C+4 h (heure où le bulletin est disponible) avec le seul GFS C.
Le 01/11 18Z n'existe qu'en f003–f018 (data/gfs_game) : non utilisé (le joueur garde le 12Z jusqu'au 02/11 00Z)."""
import os, sys, json, datetime as dt
from pathlib import Path
R = Path(__file__).resolve().parent.parent
os.environ.update(LON_MIN='-74', BOX_LAT_LO='10', BOX_LAT_HI='52', BOX_LON_HI='-1', FINISH_LOCAL='0')
os.environ.setdefault('FINE_WORKERS', '8')
os.environ['ROUTING_SCENARIO'] = str(R / 'hindsight/scenario.json')
sys.path.insert(0, str(R))
import bellman_fine as bf
T0 = dt.datetime(2026, 11, 1)
SRC = {'2026110106': R / 'data/weather.npz', '2026110112': R / 'data/night_12z/gfs.npz'}
for c in ('2026110200', '2026110206', '2026110212', '2026110218', '2026110300', '2026110306', '2026110312'):
    SRC[c] = R / f'data/ens_{c}/gfs3h.npz'
for d in sorted((R / 'hindsight/data/cycles/data').glob('ens_*')):
    SRC[d.name[4:]] = d / 'gfs3h.npz'
END = 228.  # 09/11 12Z
for c in sorted(SRC, reverse=os.environ.get("REV")=="1"):
    ch = (dt.datetime.strptime(c, '%Y%m%d%H') - T0).total_seconds() / 3600
    d = R / 'hindsight/data/mpc' / c; d.mkdir(parents=True, exist_ok=True)
    if (d / 'bellman_fine.npz').exists(): continue
    if not (d / 'gfs3h.npz').exists(): os.symlink(SRC[c], d / 'gfs3h.npz')
    lead0 = 4. if c != '2026110106' else 7.
    hor = int(min(197 - lead0, END - ch - lead0))
    print(c, 'lead0', lead0, 'horizon', hor, flush=True)
    bf.build(str(d), .25, 1., lead0=lead0, horizon=hor)
print('done')
