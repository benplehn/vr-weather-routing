"""Socle Vendée Globe (point 4) : Bellman par TRONÇON sur une fenêtre glissante, longitude 180° gérée, zones d'exclusion.

- Parcours (course JSON) : portes successives (caps) = méridien à franchir vers l'est, au sud (ou au nord) d'une latitude ;
  zone d'exclusion antarctique (AEZ) = limite nord en latitude par longitude (interpolée) ; d'autres zones (polygones).
- Fenêtre : du bateau à la prochaine porte, en longitudes « déroulées » (continues, sans saut à ±180°) ; les champs de vent
  sont lus en longitude repliée. Grille COARSE puis fine possible (même idée que bellman_c2f).
- Tronçon : la cible est la porte (méridien) ; au-delà de l'horizon de prévision (16 j), la valeur terminale vient de
  terminal() : table climatologique si fournie (CLIM_TABLE, à construire une fois par tronçon avec le même solveur sur des
  vents mensuels moyens ERA5), sinon une estimation (distance orthodromique / vitesse climatologique par bande de latitude).
- Même physique que bellman_fine : polaire, foils/polish, pénalités de manœuvre à énergie typique, Jib/Spi (couches voile×amure).
Usage (test) : window_bellman.py --selftest
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
os.environ.setdefault('ROUTING_SCENARIO', str(R / 'hindsight/scenario.json'))
import optimal_route as p  # polaire, manœuvres, déplacement (indépendants de la boîte Atlantique)
ANGLES = np.array([-160, -150, -140, -135, -130, -125, -120, -115, -105, -90, -75, -60, -50, 50, 60, 75, 90, 105, 115, 120, 125, 130, 135, 140, 150, 160.])
TYPICAL_ENERGY = 85.


def wrap(lon): return (np.asarray(lon) + 180.) % 360. - 180.


def unwrap(lon, ref): return ref + ((np.asarray(lon) - ref + 180.) % 360. - 180.)


class Course:
    def __init__(self, path):
        c = json.loads(Path(path).read_text()); self.c = c
        self.gates = c['gates']; a = np.array(c.get('aez', []), float)
        self.aez = a[np.argsort(a[:, 0])] if len(a) else None
        self.zones = [np.array(z, float) for z in c.get('zones', [])]

    def aez_lat(self, lon):
        """Limite nord de l'AEZ (lat) à la longitude repliée lon ; -90 sans AEZ."""
        if self.aez is None: return np.full(np.shape(lon), -90.)
        x = self.aez[:, 0]; y = self.aez[:, 1]
        return np.interp(wrap(lon), x, y, period=360)


class Window:
    """Fenêtre déroulée [lat0,lat1] x [lon0,lon1] (lon1 peut dépasser 180), masque terre + AEZ + zones, marge côtière."""
    def __init__(self, course, lat0, lat1, lon0, lon1, step=.25, coast_nm=2., mask_step=.05):
        self.course, self.step = course, step
        self.las = np.arange(lat0, lat1 + 1e-9, step); self.los = np.arange(lon0, lon1 + 1e-9, step)
        land = unary_union([shape(f['geometry']) for f in json.loads((R / 'data/land.geojson').read_text())['features']])
        mla = np.arange(lat0 - 1, lat1 + 1 + 1e-9, mask_step); mlo = np.arange(lon0 - 1, lon1 + 1 + 1e-9, mask_step)
        xx, yy = np.meshgrid(mlo, mla); L = contains_xy(land, wrap(xx), yy)
        self.m0 = (mla[0], mlo[0], mask_step); self.clear = distance_transform_edt(~L) * mask_step * 60 * np.cos(np.radians(np.clip(yy, -80, 80))) ** .5
        self.coast = coast_nm

    def allowed(self, lat, lon):
        lat, lon = np.broadcast_arrays(lat, lon); a0, o0, st = self.m0
        cl = map_coordinates(self.clear, [((lat - a0) / st).ravel(), ((lon - o0) / st).ravel()], order=1, cval=0, prefilter=False).reshape(lat.shape)
        ok = (cl > self.coast) & (lat > self.las[0]) & (lat < self.las[-1]) & (lon > self.los[0]) & (lon < self.los[-1])
        ok &= lat > self.course.aez_lat(lon)
        for z in self.course.zones:  # polygones (lat, lon) en longitudes repliées
            ins = np.zeros(lat.shape, bool); n = len(z); lw = wrap(lon)
            for i in range(n):
                (y1, x1), (y2, x2) = z[i], z[(i + 1) % n]
                ins ^= ((y1 > lat) != (y2 > lat)) & (lw < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-12) + x1)
            ok &= ~ins
        return ok

    def leg_allowed(self, la, lo, la2, lo2):
        return self.allowed(la2, lo2) & self.allowed((la + la2) / 2, (lo + lo2) / 2)


def gc_nm(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2); dl = np.radians(lo2 - lo1)
    z = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 3440.065 * 2 * np.arcsin(np.sqrt(np.clip(z, 0, 1)))


def clim_speed(lat):
    """Vitesse climatologique moyenne d'un IMOCA/Ultim VR par bande (nds) — ESTIMATION à remplacer par la table ERA5."""
    a = np.abs(lat); return np.interp(a, [0, 5, 10, 30, 40, 50, 60], [10, 12, 18, 20, 22, 26, 24])


def terminal(lat, lon, gate):
    """Heures estimées jusqu'à la porte (méridien gate['lon'], latitude gate['lat']) : table climatologique si CLIM_TABLE,
    sinon distance orthodromique / vitesse climatologique moyenne entre le point et la porte."""
    ct = os.environ.get('CLIM_TABLE')
    if ct and Path(ct).exists():
        T = np.load(ct); return map_coordinates(T['values'], [(lat - T['lat'][0]) / T['step'], (unwrap(lon, T['lon'][0] + 180) - T['lon'][0]) / T['step']], order=1, cval=2000)
    glat = np.minimum(lat, gate['lat']) if gate.get('side', 'S') == 'S' else np.maximum(lat, gate['lat'])
    d = gc_nm(lat, lon, glat, gate['lon_unwrapped']); return d / clim_speed((lat + glat) / 2)


def build(W, wind, gate, lead0, horizon, dt=1., workers=1, log=print):
    """Valeur V[k, couche, lat, lon] (heures jusqu'à la porte) sur la fenêtre W, vent wind(lead, lat, lon_déroulée)."""
    lo2, la2 = np.meshgrid(W.los, W.las); ok = W.allowed(la2, lo2)
    side_ok = (la2 <= gate['lat']) if gate.get('side', 'S') == 'S' else (la2 >= gate['lat'])
    target = (lo2 >= gate['lon_unwrapped']) & side_ok & ok
    g = dict(lat0=W.las[0], lon0=W.los[0], step=W.step)
    term = np.where(target, 0., terminal(la2, lo2, gate)).astype(np.float32); term[~ok] = 2000
    cur = np.repeat(term[None], 4, 0); n = int(round(horizon / dt)); V = np.empty((n + 1, 4, *la2.shape), np.float32); V[-1] = cur
    tic = time.time()
    for k in range(n - 1, -1, -1):
        ws, wd = wind(lead0 + (k + .5) * dt, la2, lo2)
        best = np.full((4, *la2.shape), 2000., np.float32)
        for a in ANGLES:
            h = (wd - a) % 360; nt = 1 if a > 0 else -1
            for ns in (0, 1):
                sp = p.polar(a, ws, ns); jn = 2 * ns + (nt > 0)
                for ps in (0, 1):
                    for pt in (-1, 1):
                        _, pen = p.maneuver(ws, TYPICAL_ENERGY, ps != ns, pt != nt)
                        ln, on = p.move(la2, lo2, h, sp * (dt - .5 * np.minimum(dt, pen)))
                        v = dt + map_coordinates(cur[jn], [((ln - g['lat0']) / g['step']).ravel(), ((on - g['lon0']) / g['step']).ravel()],
                                                 order=1, cval=2000, prefilter=False).reshape(ln.shape)
                        # franchir la porte pendant le pas = arrivée du tronçon (fraction du pas)
                        cross = (on >= gate['lon_unwrapped']) & ((ln <= gate['lat']) if gate.get('side', 'S') == 'S' else (ln >= gate['lat']))
                        frac = np.clip((gate['lon_unwrapped'] - lo2) / np.maximum(on - lo2, 1e-9), 0, 1) * dt
                        v = np.where(cross, frac, v)
                        v = np.where(W.leg_allowed(la2, lo2, ln, on) | cross, v, 2000)
                        jp = 2 * ps + (pt > 0); np.minimum(best[jp], v, out=best[jp])
        best[:, target] = 0; best[:, ~ok] = 2000; cur = best; V[k] = best
        if k % 48 == 0: log(f'tronçon Bellman {k * dt:.0f} h restantes ({time.time() - tic:.0f}s)')
    return dict(values=V, lat=W.las, lon=W.los, lead0=lead0, dt=dt)


def value(T, lead, lat, lon, layer):
    x = (lead - T['lead0']) / T['dt']; k = int(np.clip(np.floor(x), 0, len(T['values']) - 2)); f = float(np.clip(x - k, 0, 1))
    st = float(T['lat'][1] - T['lat'][0]); c = [[(lat - T['lat'][0]) / st], [(lon - T['lon'][0]) / st]]
    a = map_coordinates(T['values'][k, layer], c, order=1, cval=2000)[0]; b = map_coordinates(T['values'][k + 1, layer], c, order=1, cval=2000)[0]
    return float(a * (1 - f) + b * f)


def selftest():
    """Tronçon Nouvelle-Zélande -> cap Horn sur un vent synthétique d'ouest (25 nds ± 8, ondes mobiles), AEZ factice :
    vérifie le passage de 180°, le respect de l'AEZ et la cohérence de l'ETA avec distance/vitesse."""
    course = Course(R / 'vendee/vendee2028_example.json'); gate = dict(course.gates[2]); lon_ref = 160.
    gate['lon_unwrapped'] = float(unwrap(gate['lon'], lon_ref + 180))
    W = Window(course, -66, -35, lon_ref - 2, gate['lon_unwrapped'] + 2, step=.5)

    def wind(lead, lat, lon):  # vent d'ouest (vient du 270) + perturbations qui se déplacent vers l'est
        ws = 25 + 8 * np.sin(np.radians(3 * (lon - 2 * lead))) * np.cos(np.radians(4 * (lat + 50)))
        wd = 270 + 25 * np.sin(np.radians(2 * (lon - 1.5 * lead)))
        return ws, wd % 360
    tic = time.time(); T = build(W, wind, gate, 0., 300., log=lambda *a: None); t_b = time.time() - tic
    la, lo = -47., 166.
    v = min(value(T, 0., la, lo, j) for j in range(4))
    # route gloutonne pour vérifier 180° et AEZ
    s, tk, path = 1, 1, [(la, lo)]; t = 0.
    while t < 300:
        ws, wd = wind(t + .5, np.array([la]), np.array([lo])); best = None
        for a in ANGLES:
            for ns in (0, 1):
                nt = 1 if a > 0 else -1; _, pen = p.maneuver(ws, TYPICAL_ENERGY, ns != s, nt != tk)
                ln, on = p.move(np.array([la]), np.array([lo]), (wd - a) % 360, p.polar(a, ws, ns) * (1 - .5 * np.minimum(1, pen)))
                if not W.leg_allowed(np.array([la]), np.array([lo]), ln, on)[0]: continue
                c = 1 + value(T, t + 1, ln[0], on[0], 2 * ns + (nt > 0)) if on[0] < gate['lon_unwrapped'] else 0
                if best is None or c < best[0]: best = (c, ln[0], on[0], ns, nt)
        _, la, lo, s, tk = best; t += 1; path.append((la, lo))
        if lo >= gate['lon_unwrapped']: break
    path = np.array(path); aez = course.aez_lat(path[:, 1])
    d = gc_nm(-47, 166, -56.5, gate['lon_unwrapped'])
    print(f'SELFTEST fenêtre {len(W.las)}x{len(W.los)} (0,5°) Bellman 300 h en {t_b:.0f}s ; V(départ) {v:.1f} h ; route gloutonne {t:.0f} h ; '
          f'orthodromie {d:.0f} NM -> {d / v:.1f} nds moyens ; 180° franchi : {bool((path[:, 1] > 180).any())} ; '
          f'marge mini à l\'AEZ {np.min(path[:, 0] - aez):.2f}° (doit être > 0) ; arrivée porte {path[-1, 0]:.1f}N {wrap(path[-1, 1]):.1f}')
    return T, path


if __name__ == '__main__':
    if '--selftest' in sys.argv: selftest()
