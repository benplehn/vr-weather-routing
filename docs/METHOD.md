# Method, evaluation and results

## 1. Problem
A Virtual Regatta boat is steered by a true-wind-angle (TWA) autopilot. The player chooses, at a few moments, a TWA and a
sail; between programs the boat holds its TWA. The objective is the arrival time. Constraints of the human player used
throughout: programs computed at 10:00, 11:00, 17:00 and 22:00 Paris time; at most three programmed orders after "now";
no sail change between 22:30 and 10:00 (asleep).

**Game model** (`optimal_route.py`, validated against in-game screenshots: wind within ±0.2 kn / ±1°, polar speed within
0.1–0.2 kn, a 9 h night replayed within 3–7 NM and the same energy):
- speed = polar(TWA, TWS, sail) × foils (+4 % in the 70–170° / 11–40 kn band) × polish (×1.003);
- tack/gybe and sail change: energy spent, penalty time during which speed is halved, longer when energy is low;
  energy recovered with time (`maneuver`, `recover_seconds`);
- **game wind**: the deterministic GFS of cycle C is applied from C+6 h to C+12 h, interpolated between its own lead
  times, then the game switches to the next bulletin (`stochastic/game_wind.py`);
- land: Natural Earth 1:50m with a 2 NM clearance, the forbidden zone of the Grand Cul-de-Sac Marin, the finish leaving
  the north-west Basse-Terre mark to port.

## 2. Algorithms
**Bellman value.** V(t, x, sail, tack) = min over (TWA, sail) of [Δt + V(t+Δt, x + v Δt)], semi-Lagrangian on a 0.25°
grid, 26 TWAs, 1 h step, penalties at a typical energy. Variants: per GEFS member, scenario tree (Ward clustering of the
members, information revealed at the next bulletins), non-anticipative, gated (north/south homotopy classes).
*Coarse-to-fine* (`bellman_c2f.py`): 1° table, greedy routes from the boat for every sail/tack, a 300 NM corridor, 0.25°
recursion on the corridor only.

**Beam search planner** (`stochastic/beam_prog.py`). States (position, energy, sail, tack, pending penalty, orders used)
are propagated every 15 min over 24 h for 54 TWAs × 2 sails; one representative per 0.03° cell × sail/tack × energy band ×
orders used; ranking by elapsed time + value + energy term; 1500 states. Then:
1. candidates: the beam program and the previous program shifted in time (warm start);
2. local refinement (order times ±1–30 min, angles ±1–5°, order removal) proposes variants;
3. **selection on a 24 h continuation** (the candidate up to the program horizon, then a small free beam) — selecting on
   the relaxed table at the end of the program window is myopic (see §5);
4. **minute-by-minute verification** (land, zone, coastal margin); if nothing passes, the beam is rerun with a wider
   coastal margin. The program entered in the game is evaluated by the same functions.

**Stochastic beam** (`stochastic/beam_stoch.py`). Each node is a program common to all scenarios; its state is the boat in
every scenario; cost = weighted mean of (time + value of that scenario); a node touching land in any scenario is dropped.
Scenarios: GFS, GEFS members of the bulletin and of the previous ones (lagged ensemble, time-shifted, their tables
reused). Weights: Bayesian filter w ∝ exp(−t (d² − min d²)), d = RMS vector difference between a scenario and the new GFS
over its first 6–18 h, weights capped at 3× the mean, temperature t calibrated on the race (§4).

**Round-the-world** (`vendee/window_bellman.py`). Same recursion on a window from the boat to the next cape, longitudes
unwrapped across 180°, exclusion zone as a latitude limit per longitude, terminal value beyond the forecast horizon from
a climatological table (or great-circle distance / climatological speed by latitude band).

## 3. Evaluation protocol (hindsight)
1. **Game wind reconstruction** (`hindsight/build_truth.py`): for each of the 26 bulletins from 1 Nov 06Z to 7 Nov 12Z,
   its hourly wind on [C+6, C+12) from its own lead times, switching at C+12.
2. **Clairvoyant optimum** (`hindsight/run_bellman.py`, `hindsight/sim.py`): Bellman on the reconstructed wind over the
   North Atlantic (10–52°N, 74°W–1°W), then beam search with the full physics from the first observation; free orders, or
   the night constraint.
3. **Loss along the sailed track** (`hindsight/eval_track.py`): ETA = t + V(t, x) at every observed position; increments
   locate where time was lost.
4. **Real-time replay** (`hindsight/replay_live.py`): at each program time, only what was available — the table and GFS
   of the last bulletin with C + 4 h ≤ t, the game wind (previous GFS until C+6, then C), the next GFS once published,
   the GEFS members of the bulletin and the previous ones; the program is executed in the reconstructed wind every 5 min.

## 4. Results (Route du Rhum 2026)
| Policy | Arrival (passage line + 2.5 h) |
|---|---|
| Clairvoyant optimum, free orders | 07/11 13:37 UTC |
| Clairvoyant optimum, night constraint | 07/11 16:04 UTC |
| Clairvoyant optimum, west limit at 65°W (grid used during the race until 6 Nov) | 07/11 16:44 / 18:11 UTC |
| This router, real-time replay from the start | **07/11 17:26 UTC** |
| Route sailed during the race | ≥ 08/11 04:25 UTC |

**Robustness of the optimum**: beam 3 000 → 13:35, 10 000 → 13:35, 0.125° table → 13:33 (earlier wind reconstruction);
finer 10 min steps with the same beam: 14:13 (deeper pruning).

**Where the sailed route lost time**: +3.3 h in the night of 2–3 Nov (ridge crossed too far north), ~0.4 h per bulletin
from 3 to 6 Nov, +3 h in the night of 6–7 Nov, +2 h in the final beat. The route family was right; the losses were
tactical and caused by (i) the 65°W limit of the computation grid, (ii) short-horizon exploitation of relaxed tables.

**Deterministic vs stochastic** (server replay from 3 Nov 08:51 UTC, real tables and members):

| Program driver | Arrival | Compute / program |
|---|---|---|
| Deterministic beam | 07/11 22:58 UTC | 154 s |
| Stochastic beam (lagged ensemble, calibrated filter, 24 scenarios) | 08/11 08:25 UTC | 260 s |

The whole difference comes from one decision (4 Nov, a front: the stochastic program stays ahead of it, the deterministic
one crosses it). On this race the stochastic layer brought no gain; it is kept as a strategic control only.

**Bayesian filter calibration** (`stochastic/calibrate_weights.py`, log-score of the true wind under the weighted mixture):
early agreement with the new GFS predicts later skill (mean rank correlation 0.41 at 12–48 h, 0.30 at 24–72 h); the
calibrated filter gains +0.145 (12–48 h, cross-validated) and +0.056 (24–72 h) over equal weights, against +0.045 and
+0.031 for the previous fixed-ESS rule. The ensemble mean was closer to the true game wind than the deterministic GFS in
11 of 12 bulletins at 12–48 h.

**Coarse-to-fine**: identical values along the whole track and identical optimal route, 29 s instead of 67 s (corridor =
28 % of the nodes on the Atlantic; smaller fraction on larger windows).

## 5. Defects found by the evaluation (each covered by a regression test)
| Defect | Effect | Fix |
|---|---|---|
| Computation grid limited at 65°W | optimal route at 66–67°W out of reach (≈ 2.7 h) | limit 74°W, test |
| Candidates judged on the relaxed table at the end of the program window | front of 4 Nov fled instead of crossed | selection on a 24 h continuation |
| Coastal check every ~7 min with a 2 NM margin | a validated program touched Vieux-Fort in the game | minute-by-minute verification |
| Arrival accepted at the very end of the passage line | the held order then led onto Basse-Terre | end margin + verification after the line |
| No verifiable route near Montserrat | grounding in the replay | rerun with a wider margin |
| Scenario weight cap enforced by iterated renormalisation | cap exceeded (3.12 > 3) | exact capped normalisation |
| Hindsight wind blending two bulletins before each switch | reversed a decision in the evaluation | hard switch as in the game |

## 6. Limitations and open questions
- No direct comparison with other routers (e.g. Zezo) under the same execution constraints; the expected edge is the
  constrained execution (~2.4 h between free and constrained optima), not the route family.
- The planning horizon (24 h) is a choice, not an optimum; the remaining gap to the clairvoyant bound (1.4 h) caps any gain.
- The stochastic layer may matter more on longer, more uncertain races (Vendée Globe); this is not demonstrated.
- After the last downloaded bulletin the reconstructed wind is a forecast; the last miles around Basse-Terre are
  approximated (2.5 h common tail) in the comparisons.
