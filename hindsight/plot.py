"""Carte des routes a posteriori + courbe 'ETA parfaite depuis notre position' -> hindsight/out/bilan.png"""
import json, sys, datetime as dt
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from shapely.geometry import shape
H = Path(__file__).resolve().parent; R = H.parent
sys.path.insert(0, str(H))
from track import TRACK
from eval_track import eta, T0
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
ink, ink2, surf = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': ink2, 'xtick.color': ink2, 'ytick.color': ink2})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1.35, 1]), facecolor=surf)
for a in (ax, bx): a.set_facecolor(surf)
for f in json.loads((R / 'data/land.geojson').read_text())['features']:
    g = shape(f['geometry'])
    for poly in getattr(g, 'geoms', [g]):
        x, y = poly.exterior.xy; ax.fill(x, y, color='#dcdbd5', lw=0)
ax.axvspan(-80, -65, color='#e34948', alpha=.06, lw=0); ax.axvline(-65, color='#e34948', lw=1, ls='--')
ax.text(-64.5, 48, 'mur de calcul −65°\n(jusqu\'au 6/11)', color=ink2, ha='left', fontsize=8)
routes = [('route_w74_free_0', 'Parfaite, ordres libres'), ('route_w74_night_0', 'Parfaite, contrainte de nuit'),
          ('live_det_full', 'Notre stratégie rejouée en temps réel (contraintes)')]
for i, (k, lab) in enumerate(routes):
    f = H / 'out' / f'{k}.json'
    if not f.exists(): continue
    r = json.loads(f.read_text()); tr = np.array([x[:3] for x in r['track']])  # (heure, lat, lon) pour sim.py et replay_live.py
    u = r.get('eta_line_utc') or r['eta_utc']; e = u[8:10] + '/11 ' + u[11:16] + 'Z'
    ax.plot(tr[:, 2], tr[:, 1], color=C[i + 1], lw=2, label=f'{lab} — arrivée {e}')
tt = np.array([[t[1], t[2]] for t in TRACK]); ax.plot(tt[:, 1], tt[:, 0], color=C[0], lw=2, label='Notre route réelle')
ax.plot(tt[:, 1], tt[:, 0], 'o', color=C[0], ms=3)
ax.set_xlim(-70, 0); ax.set_ylim(14, 52); ax.set_aspect(1.15); ax.grid(color='#ecebe6', lw=.6)
ax.legend(loc='lower right', bbox_to_anchor=(1, .3), fontsize=8, frameon=True, facecolor=surf, edgecolor='none'); ax.set_title('Routes depuis le départ, vent vrai du jeu', color=ink, loc='left')
ts, es = [], []
for u, la, lo, s, tk, kind in TRACK:
    t, v, e = eta(u, la, lo, s, tk); ts.append(T0 + dt.timedelta(hours=t)); es.append(e - eta(*TRACK[0][:3], TRACK[0][3], TRACK[0][4])[2])
bx.plot(ts, es, color=C[0], lw=2, marker='o', ms=4)
bx.set_title('Retard cumulé sur la route parfaite (h)', color=ink, loc='left')
for d, txt in [(dt.datetime(2026, 11, 3, 2), 'nuit 2→3/11\n+3,3 h'), (dt.datetime(2026, 11, 7, 6), 'nuit 6→7/11\n+3,0 h')]:
    bx.annotate(txt, (d, np.interp((d - T0).total_seconds(), [(x - T0).total_seconds() for x in ts], es)), xytext=(-60, 15),
                textcoords='offset points', fontsize=8, color=ink2, arrowprops=dict(arrowstyle='-', color=ink2, lw=.6))
for k in range(1, 8): bx.axvspan(dt.datetime(2026, 11, k, 21, 30), dt.datetime(2026, 11, k + 1, 9), color='#6250d6', alpha=.06, lw=0)
bx.text(dt.datetime(2026, 11, 1, 22), max(es) * .97, 'bandes = nuits (21:30–09:00 UTC)', fontsize=8, color=ink2)
bx.grid(color='#ecebe6', lw=.6); bx.set_ylim(bottom=-.5)
import matplotlib.dates as md; bx.xaxis.set_major_formatter(md.DateFormatter('%d/%m'))
for a in (ax, bx): a.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); out = H / 'out/bilan.png'; fig.savefig(out, dpi=140, facecolor=surf); print(out)
