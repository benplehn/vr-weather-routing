"""Tests anti-bogues rapides (< 1 min, sans réseau). Chacun correspond à un bogue réellement rencontré ou à une règle du
jeu qu'un changement pourrait casser. Lancer depuis la racine du projet :
    python -m unittest tests.test_routing -v
(tests/check_logic.py reste le contrôle des tables d'un bulletin réel, lancé par auto_update avant chaque mail).
"""
import os, sys, json, unittest, datetime as dt
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'stochastic'), str(R / 'stochastic/auto'), str(R / 'vendee')]
if not (R / 'data/Ultim_BP_XI_current.json').exists(): os.environ.setdefault('POLAR_FILE', str(R / 'tests/fixtures/polar_synthetic.json'))
os.environ.setdefault('ROUTING_SCENARIO', str(R / 'hindsight/scenario.json'))
import optimal_route as p
import beam_prog as bp

UTC = dt.timezone.utc


class Const:
    """Vent synthétique constant (tws nds, twd degrés) appelable comme un champ du jeu : (lead, lat, lon) -> (tws, twd)."""
    def __init__(self, ws=15., wd=90.): self.ws, self.wd = ws, wd
    def __call__(self, L, lat, lon):
        lat = np.asarray(lat, float); return np.full(lat.shape, self.ws), np.full(lat.shape, self.wd)


def flat_table(val=10., lat=(10, 30), lon=(-70, -50)):
    las = np.arange(lat[0], lat[1] + 1e-9, .25); los = np.arange(lon[0], lon[1] + 1e-9, .25)
    return dict(values=np.full((60, 4, len(las), len(los)), val, np.float32), lead0=0., lat=las, lon=los)


def beam(wd=90., obs=dt.datetime(2026, 11, 7, 12, tzinfo=UTC)):
    return bp.Beam('2026110712', obs, [0, 1], wind=Const(15., wd), table=flat_table())


class GameRules(unittest.TestCase):
    def test_night_window(self):
        P = bp.PARIS
        self.assertTrue(bp.night_paris(dt.datetime(2026, 11, 7, 22, 30, tzinfo=P)))
        self.assertTrue(bp.night_paris(dt.datetime(2026, 11, 8, 9, 59, tzinfo=P)))
        self.assertFalse(bp.night_paris(dt.datetime(2026, 11, 8, 10, 0, tzinfo=P)))
        self.assertFalse(bp.night_paris(dt.datetime(2026, 11, 7, 15, 0, tzinfo=P)))

    def test_passage_line(self):
        # ligne 16.25 N entre 62.8 W et 61.88 W (à l'ouest de la bouée NO), franchie vers le sud seulement
        m, f, toff = bp.crossed(np.array([16.3]), np.array([-62.2]), np.array([16.2]), np.array([-62.2]))
        self.assertTrue(m[0]); self.assertAlmostEqual(f[0], .5, places=6)
        m, *_ = bp.crossed(np.array([16.3]), np.array([-61.6]), np.array([16.2]), np.array([-61.6]))
        self.assertFalse(m[0])  # à l'est de la bouée : interdit
        m, *_ = bp.crossed(np.array([16.2]), np.array([-62.2]), np.array([16.3]), np.array([-62.2]))
        self.assertFalse(m[0])  # vers le nord : pas une arrivée

    def test_forbidden_zone_and_land(self):
        self.assertFalse(p.allowed(np.array([16.40]), np.array([-61.60]))[0])  # Grand Cul-de-Sac Marin (zone interdite)
        self.assertFalse(p.allowed(np.array([16.75]), np.array([-62.19]))[0])  # Montserrat
        self.assertTrue(p.allowed(np.array([16.60]), np.array([-62.60]))[0])

    def test_maneuver_costs(self):
        used, pen = p.maneuver(np.array([15.]), np.array([100.]), True, True)
        used2, pen2 = p.maneuver(np.array([15.]), np.array([30.]), True, True)
        self.assertGreater(used[0], 0); self.assertGreater(pen2[0], pen[0])  # énergie basse = pénalité plus longue
        u0, p0 = p.maneuver(np.array([15.]), np.array([100.]), False, False)
        self.assertEqual(u0[0], 0); self.assertEqual(p0[0], 0)


class BeamProgram(unittest.TestCase):
    def setUp(self):
        self.N, self.HZ = bp.N, bp.HZ; bp.N, bp.HZ = 60, 4.

    def tearDown(self):
        bp.N, bp.HZ = self.N, self.HZ

    def test_step_energy_and_penalty(self):
        B = beam(); S = B._start(20., -60., 80., 60., 0)
        Nt, ok = B.step(S, -60., 0, 0., 1 / 60)  # virement : énergie dépensée, pénalité encore en cours après 1 min
        self.assertTrue(ok[0]); self.assertLess(Nt['e'][0], 80); self.assertGreater(Nt['pen'][0], 0)
        Nn, _ = B.step(S, 60., 0, 0., 1 / 60)    # même amure : récupération, pas de pénalité
        self.assertGreater(Nn['e'][0], 80)
        self.assertGreater(p.distance(Nn['lat'], Nn['lon'], np.array([20., -60.]))[0], p.distance(Nt['lat'], Nt['lon'], np.array([20., -60.]))[0])

    def test_order_limit(self):
        B = beam(); q = [[0, 60, 0], [10, -60, 0], [20, 60, 0], [30, -60, 0], [40, 60, 0]]
        self.assertEqual(B.simulate(20., -60., 90., 60., 0, 2., q)['why'], f'plus de {bp.MAXO} ordres')
        r = bp.plan(B, 20., -60., 90., 60., 0, 2.)
        self.assertLessEqual(sum(1 for m, *_ in r['orders'] if m > 0), bp.MAXO)

    def test_no_sail_change_at_night(self):
        obs = dt.datetime(2026, 11, 7, 22, tzinfo=UTC)  # 23:00 Paris
        B = beam(obs=obs)
        r = B.simulate(20., -60., 90., 60., 0, 2., [[0, 60, 0], [30, 60, 'Spi']])
        self.assertFalse(r['ok']); self.assertIn('nuit', r['why'])
        self.assertTrue(B.simulate(20., -60., 90., 60., 0, 2., [[0, 60, 'Spi'], [30, 60, 'Spi']])['ok'])  # maintenant : permis

    def test_verify_catches_land(self):
        # 7/11 : un programme validé au pas de 7 min touchait Vieux-Fort. Cap droit sur Montserrat par vent de nord.
        B = bp.Beam('2026110712', dt.datetime(2026, 11, 7, 12, tzinfo=UTC), [0, 1], wind=Const(15., 0.), table=flat_table())
        # vent du nord (0°), TWA 160 tribord => cap 200° environ : depuis 17.05N 62.10W on file sur Montserrat
        r = B.verify(17.05, -62.10, 90., 160., 1, 3., [[0, 160, 'Spi']])
        self.assertFalse(r['ok']); self.assertIn('terre', r['why'])

    def test_plan_never_returns_unverified_when_a_safe_one_exists(self):
        B = beam(); r = bp.plan(B, 20., -60., 90., 60., 0, 2.)
        self.assertTrue(r['verified']); self.assertFalse(r['warnings'])

    def test_to_minutes_merges_past_orders(self):
        obs = dt.datetime(2026, 11, 7, 12, tzinfo=UTC)
        q = bp.to_minutes([['202611071100', 50, 'Jib'], ['202611071130', -55, 'Jib'], ['202611071300', 60, 'Jib']], obs)
        self.assertEqual(q, [[0, -55, 'Jib'], [60, 60, 'Jib']])

    def test_pack_sail_layer(self):
        # les voiles du pack lisent la couche Jib (près) ou Spi (portant) des tables Jib/Spi
        self.assertEqual(list(bp.LAYER_SAIL), [0, 1, 0, 0, 0, 1, 1])


class Stochastic(unittest.TestCase):
    def test_calibrated_weights_cap_and_control(self):
        import beam_stoch as bs

        class F:  # scénario = vent constant décalé de off nds
            def __init__(self, off): self.off = off
            def __call__(self, L, lat, lon):
                lat = np.asarray(lat, float); return np.full(lat.shape, 15. + self.off), np.full(lat.shape, 90.)
        scen = [dict(name='gfs', raw=F(0), shift=0.), dict(name='gefsc00', raw=F(.1), shift=0.)] + \
               [dict(name=f'gefsp{i:02d}', raw=F(1 + i * .5), shift=0.) for i in range(1, 11)] + \
               [dict(name=f'gefsp{i:02d}@0306', raw=F(4 + i), shift=6.) for i in range(1, 6)]
        w = bs.calibrated_weights(scen, None, .7, 20., -60.)
        self.assertEqual(len(w), len(scen))
        self.assertLessEqual(w[1:].max(), 3 * w[1:].mean() + 1e-9)   # plafond
        self.assertLess(w[1] / w[1:].sum(), .5)                       # le contrôle ne prend pas tout
        self.assertAlmostEqual(w[0], w[1:].mean(), places=9)           # colonne GFS = poids moyen
        self.assertLess(w[-1], w[2])                                   # membre décalé en désaccord : poids faible


class Mail(unittest.TestCase):
    def test_unverified_program_is_flagged(self):
        import auto_update as au
        now = dt.datetime(2026, 11, 7, 21, tzinfo=UTC)
        B = dict(E=2.5, ref_E=3., orders=[[0, 55, 'Jib'], [60, -55, 'Jib']], min_energy=70, horizon=24, max_orders=3,
                 verified=False, warnings=['AUCUN programme ne passe la vérification minute par minute : terre'])
        s, b = au.beam_mail('12Z', now, 16.8, -62.1, 85, B)
        self.assertTrue(s.startswith('A VERIFIER')); self.assertIn('PAS passé', '\n'.join(b))

    def test_strategy_divergence(self):
        import auto_update as au
        now = dt.datetime(2026, 11, 7, 21, tzinfo=UTC)
        BM = dict(orders=[[0, 130, 'Spi'], [300, -125, 'Spi']])
        same = dict(orders=[[0, 135, 'Spi'], [280, -120, 'Spi']], K=40, n_lagged=9)
        diff = dict(orders=[[0, 135, 'Spi'], [200, 130, 'Spi']], K=40, n_lagged=9)
        self.assertIn('Même stratégie', au.stoch_line(now, same, BM)); self.assertIn('DIFFÉRENTE', au.stoch_line(now, diff, BM))


class Vendee(unittest.TestCase):
    def test_longitude_wrap(self):
        import window_bellman as wb
        self.assertAlmostEqual(float(wb.unwrap(-170, 175)), 190.)
        self.assertAlmostEqual(float(wb.wrap(190)), -170.)

    def test_aez_periodic(self):
        import window_bellman as wb
        c = wb.Course(R / 'vendee/vendee2028_example.json')
        self.assertAlmostEqual(float(c.aez_lat(180.)), float(c.aez_lat(-180.)))
        self.assertAlmostEqual(float(c.aez_lat(190.)), float(c.aez_lat(-170.)))


class Guards(unittest.TestCase):
    def test_member_tables_never_start_from_a_corridor_table(self):
        # une table GFS « couloir » (bellman_c2f) a des murs hors de la route GFS : les membres repartent de zéro
        self.assertIn("'corridor' not in np.load(gfs_table).files", (R / 'bellman_fine.py').read_text())

    def test_west_limit_wide_enough(self):
        # bilan 7/11 : le mur à -65 W coûtait 2,7 h ; la route optimale passait à 66-67 W
        self.assertLessEqual(p.LON_MIN, -74)


if __name__ == '__main__':
    unittest.main()
