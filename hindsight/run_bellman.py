"""Bellman a posteriori (vent vrai du jeu connu) sur tout l'Atlantique, depuis la première observation.

Usage : run_bellman.py TAG [LON_MIN]  -> hindsight/data/TAG/bellman_fine.npz
V[k, couche(voile,amure), lat, lon] = heures restantes jusqu'à la ligne de passage (ouest de Basse-Terre), à l'heure
lead0+k (heures depuis T0 = 01/11 00Z). Même physique que bellman_fine.py (Jib/Spi, pénalités à énergie 85).
"""
import os, sys, json, shutil
from pathlib import Path
R = Path(__file__).resolve().parent.parent
tag = sys.argv[1]; lon_min = sys.argv[2] if len(sys.argv) > 2 else '-74'; STEPG = float(sys.argv[3]) if len(sys.argv) > 3 else .25
os.environ.update(LON_MIN=lon_min, BOX_LAT_LO='10', BOX_LAT_HI='52', BOX_LON_HI='-1', FINISH_LOCAL='0')
os.environ.setdefault('FINE_WORKERS', '7')
scen = R / 'hindsight/scenario.json'
scen.write_text(json.dumps(dict(case_name='hindsight', start_lat=48.959167, start_lon=-2.645278,
                                start_utc='2026-11-01T13:13:00+00:00', forecast_cycle_utc='2026-11-01T06:00:00+00:00',
                                energy=100., sail=0, tack=1, weather_file='weather.npz')))
os.environ['ROUTING_SCENARIO'] = str(scen)
sys.path.insert(0, str(R))
import bellman_fine as bf
d = R / 'hindsight/data' / tag; d.mkdir(parents=True, exist_ok=True)
src = R / 'hindsight/data/gfs3h.npz'
if not (d / 'gfs3h.npz').exists(): os.symlink(src, d / 'gfs3h.npz')
LEAD0 = 13 + 13 / 60  # 01/11 13:13Z
print('WROTE', bf.build(str(d), STEPG, 1., lead0=LEAD0, horizon=int(228 - LEAD0) - 1), flush=True)
