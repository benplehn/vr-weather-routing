"""Core model tests: geometry, polar, game-wind timing, scenario weights, Vendée Globe window. Fast, no network, no data.
    python -m unittest tests.test_core -v
"""
import os, sys, unittest
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(R), str(R / 'stochastic'), str(R / 'vendee')]
if not (R / 'data/Ultim_BP_XI_current.json').exists(): os.environ.setdefault('POLAR_FILE', str(R / 'tests/fixtures/polar_synthetic.json'))
os.environ.setdefault('ROUTING_SCENARIO', str(R / 'hindsight/scenario.json'))
import optimal_route as p


class Const:
    def __init__(self, ws, wd): self.ws, self.wd = ws, wd
    def __call__(self, L, lat, lon):
        lat = np.asarray(lat, float); return np.full(lat.shape, float(self.ws)), np.full(lat.shape, float(self.wd))


class Geometry(unittest.TestCase):
    def test_move_and_distance_are_consistent(self):
        la, lo = p.move(np.array([20.]), np.array([-40.]), 0., 60.)        # 60 NM due north = 1° of latitude
        self.assertAlmostEqual(float(la[0]), 21., places=6); self.assertAlmostEqual(float(lo[0]), -40., places=9)
        la, lo = p.move(np.array([20.]), np.array([-40.]), 90., 30.)
        self.assertAlmostEqual(float(p.distance(la, lo, np.array([20., -40.]))[0]), 30., delta=.05)

    def test_true_wind_angle_sign(self):
        # wind from 90°, heading 0° -> wind on the starboard side (positive TWA, starboard tack)
        self.assertGreater(float(p.angle(90., 0.)), 0); self.assertLess(float(p.angle(270., 0.)), 0)


class Polar(unittest.TestCase):
    def test_symmetric_in_tack(self):
        for a in (50., 90., 140.):
            self.assertAlmostEqual(float(p.polar(a, 15., 0)), float(p.polar(-a, 15., 0)))

    def test_no_speed_head_to_wind_and_increasing_wind(self):
        self.assertLess(float(p.polar(10., 15., 0)), float(p.polar(90., 15., 0)))
        self.assertLessEqual(float(p.polar(90., 8., 0)), float(p.polar(90., 16., 0)))


class GameWind(unittest.TestCase):
    def test_bulletin_switch_and_blend(self):
        import game_wind
        prev, new, scen = Const(10, 0), Const(20, 90), Const(30, 180)
        W = game_wind.GameWind(prev, new, scen, switch=6., known=12., blend=6.)
        z = np.zeros(1)
        self.assertEqual(float(W(3., z, z)[0][0]), 10.)    # before C+6 the game still blows the previous bulletin
        self.assertEqual(float(W(8., z, z)[1][0]), 90.)    # C+6..C+12: this bulletin, known exactly
        self.assertEqual(float(W(20., z, z)[0][0]), 30.)   # after the blend: the scenario
        ws, _ = W(15., z, z); self.assertTrue(10. < float(ws[0]) < 30.)

    def test_next_bulletin_replaces_scenario(self):
        import game_wind
        base = game_wind.GameWind(Const(10, 0), Const(20, 90), Const(30, 180), 6., 12.)
        W = game_wind.NextWind(base, Const(25, 45), Const(30, 180), 12.)
        z = np.zeros(1)
        self.assertEqual(float(W(14., z, z)[1][0]), 45.)    # the next GFS, once published, is used on [C+12, C+18)


class Weights(unittest.TestCase):
    def test_cap_is_exact_and_mean_is_one(self):
        import beam_stoch as bs
        w = bs.cap_weights(np.array([50., 1, 1, 1, 1, 1, 1, 1, 1, 1]), 3.)
        self.assertAlmostEqual(w.mean(), 1., places=9); self.assertLessEqual(w.max(), 3. + 1e-9)
        self.assertTrue(np.allclose(bs.cap_weights(np.ones(5), 3.), 1.))


class Beam(unittest.TestCase):
    def test_line_end_margin(self):
        import beam_prog as bp
        m, *_ = bp.crossed(np.array([16.3]), np.array([-61.89]), np.array([16.2]), np.array([-61.89]))
        self.assertTrue(m[0])                                    # within the table's line
        m, *_ = bp.crossed(np.array([16.3]), np.array([-61.89]), np.array([16.2]), np.array([-61.89]), bp.LINE_E_SAFE)
        self.assertFalse(m[0])                                   # but too close to its end for a planned arrival


class VendeeWindow(unittest.TestCase):
    def test_aez_blocks_the_south(self):
        import window_bellman as wb
        c = wb.Course(R / 'vendee/vendee2028_example.json')
        W = wb.Window(c, -66, -40, 170, 200, step=1.)
        self.assertTrue(W.allowed(np.array([-50.]), np.array([185.]))[0])
        self.assertFalse(W.allowed(np.array([-63.]), np.array([185.]))[0])  # south of the exclusion limit


if __name__ == '__main__':
    unittest.main()
