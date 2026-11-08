"""Vent « vrai » du jeu reconstitué a posteriori pour toute la course -> hindsight/data/gfs3h.npz (format Field).

Règle du jeu (game_wind.py, vérifiée sur le journal de bord) : le bulletin GFS du cycle C est appliqué de C+6 à C+12,
interpolé dans le temps ENTRE SES PROPRES échéances (f006, f009, f012), puis le jeu BASCULE net sur le bulletin suivant.
Correction du 7/11 : la première version interpolait entre le f009 de C et le f006 du bulletin suivant, donc mélangeait
deux bulletins pendant les 3 h précédant chaque bascule (le jeu ne fait pas ça) ; cela changeait des décisions (front du
4/11). Ici : échantillons horaires C+6 ... C+11 et C+11,999 tirés du seul bulletin C, puis C'+6 du suivant -> bascule
en 0,001 h. Après le dernier bulletin (07/11 12Z), sa prévision jusqu'à f072.

Sources : data/weather.npz (01/11 06Z, pas 6 h), data/night_12z/gfs.npz (01/11 12Z, pas 6 h),
data/gfs_game/gfs_20261101183h.npz (01/11 18Z), data/ens_C/gfs3h.npz (02/11 00Z..03/11 12Z),
hindsight/data/truth_srv.npz (03/11 00Z..07/11 12Z, échéances 6..18 h extraites des data/ens_C/gfs3h.npz du serveur).
"""
from pathlib import Path
import datetime as dt
import numpy as np
R = Path(__file__).resolve().parent.parent
T0 = dt.datetime(2026, 11, 1, 0)


def cyc_hour(c): return (dt.datetime.strptime(c, '%Y%m%d%H') - T0).total_seconds() / 3600


def lead(w, h):
    """(u, v) du fichier w à l'échéance h, interpolé linéairement entre SES échéances."""
    H = list(np.asarray(w['hours'], float))
    if h in H: i = H.index(h); return w['u'][i], w['v'][i]
    i = int(np.searchsorted(H, h)) - 1; f = (h - H[i]) / (H[i + 1] - H[i])
    return w['u'][i] * (1 - f) + w['u'][i + 1] * f, w['v'][i] * (1 - f) + w['v'][i + 1] * f


def main():
    src = {'2026110106': R / 'data/weather.npz', '2026110112': R / 'data/night_12z/gfs.npz',
           '2026110118': R / 'data/gfs_game/gfs_20261101183h.npz'}
    for c in ('2026110200', '2026110206', '2026110212', '2026110218'): src[c] = R / f'data/ens_{c}/gfs3h.npz'
    srv = np.load(R / 'hindsight/data/truth_srv.npz'); srv_c = sorted(k[2:] for k in srv.files if k.startswith('h_'))
    cycles = sorted(set(src) | set(srv_c)); hours, U, V, used = [], [], [], []; lat = lon = None
    for c in cycles:
        if c in srv_c: w = dict(hours=srv['h_' + c], u=srv['u_' + c], v=srv['v_' + c]); la, lo = srv['lat'], srv['lon']
        else: z = np.load(src[c]); w = {k: z[k] for k in z.files}; la, lo = w['lat'], w['lon']
        if lat is None: lat, lon = la, lo
        assert np.allclose(la, lat) and np.allclose(lo, lon), c
        last = c == cycles[-1]
        leads = list(np.arange(6, 12, 1.)) + [11.999] if not last else [x for x in np.arange(6, 73, 1.) if x <= float(np.max(w['hours']))]
        for L in leads:
            u, v = lead(w, float(L)); hours.append(cyc_hour(c) + L); U.append(u.astype(np.float32)); V.append(v.astype(np.float32))
            used.append(f'{c}+f{L:06.3f}')
    hours = np.array(hours); assert np.all(np.diff(hours) > 0), hours
    out = R / 'hindsight/data/gfs3h.npz'
    np.savez(out, hours=hours, lat=lat, lon=lon, u=np.array(U), v=np.array(V), used=np.array(used))
    print('wrote', out, 'hours', hours[0], '->', hours[-1], len(hours), 'pas ; bascule nette à chaque bulletin')


if __name__ == '__main__':
    main()
