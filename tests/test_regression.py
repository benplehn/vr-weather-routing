"""Tests de régression sur des situations réelles de la Route du Rhum (lents, ~1 min chacun ; ignorés si les données
a posteriori hindsight/data ne sont pas présentes). python -m unittest tests.test_regression -v
"""
import os, sys, unittest, datetime as dt
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; H = R / 'hindsight'
os.environ.setdefault('LON_MIN', '-74')
if not (R / 'data/Ultim_BP_XI_current.json').exists(): os.environ.setdefault('POLAR_FILE', str(R / 'tests/fixtures/polar_synthetic.json'))
os.environ.setdefault('ROUTING_SCENARIO', str(H / 'scenario.json'))
sys.path[:0] = [str(R), str(R / 'stochastic'), str(H)]
DATA = (H / 'data/gfs3h.npz').exists() and (H / 'data/mpc/2026110406/bellman_fine.npz').exists()


@unittest.skipUnless(DATA, 'données a posteriori absentes (hindsight/data)')
class Front0411(unittest.TestCase):
    def test_front_is_crossed_not_fled(self):
        """4/11 10Z, 41.92N 40.22W, Jib 71° bâbord, bulletin 06Z, front qui arrive de l'ouest. Garde-fou de bout en bout
        (vent du jeu game_wind -> plan -> exécution dans le vent vrai) : le programme doit descendre au sud-ouest (≈ 43 W
        à 16Z). Leçon du 7/11 : la première reconstitution du vent vrai mélangeait deux bulletins avant chaque bascule et
        faisait croire que ce programme filait vers l'est ; avec la bascule nette du jeu (build_truth.py) il est le bon."""
        import bellman_fine as bf, beam_prog as bp, game_wind
        T0 = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc); t = 3 * 24 + 10.
        prev, new = bf.Field(H / 'data/mpc/2026110400/gfs3h.npz'), bf.Field(H / 'data/mpc/2026110406/gfs3h.npz')
        W = game_wind.GameWind(prev, new, new, 6., 12.)
        B = bp.Beam('2026110406', T0 + dt.timedelta(hours=t), [0, 1], wind=W, table=dict(np.load(H / 'data/mpc/2026110406/bellman_fine.npz')))
        r = bp.plan(B, 41.92, -40.22, 100., -71., 0, 6.)
        truth = bp.Beam('2026110100', T0, [0, 1], wind=bf.Field(H / 'data/gfs3h.npz'), table={}); truth.L0 = 0.
        S = truth._start(41.92, -40.22, 100., -71., 0); h = t; ex = sorted([[m, a, 0 if s == 'Jib' else 1] for m, a, s in r['orders']])
        while h < t + 6:
            cur = [o for o in ex if o[0] <= (h - t) * 60 + 1e-6][-1]
            S, ok = truth.step(S, float(cur[1]), cur[2], h, 5 / 60); h += 5 / 60
        self.assertLess(float(S['lon'][0]), -41.0, f"programme {r['orders']} -> {S['lat'][0]:.2f} {S['lon'][0]:.2f}")


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(DATA and (H / 'data/mpc/2026110706/bellman_fine.npz').exists(), 'données a posteriori absentes')
class LineEdge0711(unittest.TestCase):
    def test_knife_edge_crossing_is_not_an_arrival(self):
        """7/11 11:00 Paris, 17.36N 62.28W : ce programme coupait la ligne de passage à 61,8818 W (bout de ligne 61,88 W),
        la vérification l'acceptait comme arrivée et s'arrêtait ; ensuite le dernier ordre menait sur Basse-Terre. Il doit
        désormais être rejeté (marge de bout de ligne + vérification de la suite)."""
        import bellman_fine as bf, beam_prog as bp, game_wind
        T0 = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc); t = 154.
        W = game_wind.GameWind(bf.Field(H / 'data/mpc/2026110700/gfs3h.npz'), bf.Field(H / 'data/mpc/2026110706/gfs3h.npz'),
                               bf.Field(H / 'data/mpc/2026110706/gfs3h.npz'), 6., 12.)
        B = bp.Beam('2026110706', T0 + dt.timedelta(hours=t), [0, 1], wind=W, table=dict(np.load(H / 'data/mpc/2026110706/bellman_fine.npz')))
        r = B.verify(17.356, -62.282, 86.9, 59., 0, 6., [[0, 64, 'Jib'], [162, -50, 'Jib'], [189, -50, 'Jib'], [225, 55, 'Jib']])
        self.assertFalse(r['ok'], r)


@unittest.skipUnless(DATA and (H / 'data/mpc/2026110706/bellman_fine.npz').exists(), 'données a posteriori absentes')
class Montserrat0711(unittest.TestCase):
    def test_plan_near_islands_is_verified(self):
        """7/11 11:00 Paris, 17.35N 62.27W, Jib 55° tribord, fin de course (FINISH_REAL) : le faisceau ne trouvait que des
        routes rasant Montserrat (aucune vérifiée -> échouage dans le rejeu). Avec la marge élargie, un programme vérifié."""
        import bellman_fine as bf, beam_prog as bp, game_wind
        T0 = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc); t = 154.
        W = game_wind.GameWind(bf.Field(H / 'data/mpc/2026110700/gfs3h.npz'), bf.Field(H / 'data/mpc/2026110706/gfs3h.npz'),
                               bf.Field(H / 'data/mpc/2026110706/gfs3h.npz'), 6., 12.)
        B = bp.Beam('2026110706', T0 + dt.timedelta(hours=t), [0, 1], wind=W, table=dict(np.load(H / 'data/mpc/2026110706/bellman_fine.npz')))
        fr = bp.FINISH_REAL; bp.FINISH_REAL = True
        try: r = bp.plan(B, 17.35, -62.27, 87., 60., 0, 6.)
        finally: bp.FINISH_REAL = fr
        self.assertTrue(r['verified'], r['warnings'])
