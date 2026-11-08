#!/bin/sh
# Cron (host time zone Europe/Paris; the four slots are C+3h15 UTC after each GFS cycle):
#   */10 * * * *          cd ROUTING_HOME && flock -n /tmp/routage_ultim.lock .venv/bin/python stochastic/auto/auto_update.py >> stochastic/auto/cron.log 2>&1
#   15 4,10,16,22 * * *   ROUTING_HOME/stochastic/auto/prefetch.sh
#   20 4,10,16,22 * * *   ROUTING_HOME/stochastic/auto/quick.sh
# Prefetch at every bulletin (~3 h 15 after its cycle): download the next cycle (last full cycle + 6 h) while NOAA
# publishes it (forecast-hour order, fetch_ensemble.py --follow), then run the full update as soon as it is complete
# instead of waiting for the */10 detection. Runs alongside the quick update (quick.sh), which does not need these files.
cd "$(dirname "$0")/../.." || exit 1
[ -f config.env ] && . ./config.env  # NOTIFY_EMAIL, PARALLEL (see config.env.example)
L=$(.venv/bin/python -c "import json;print(json.load(open('stochastic/auto/state.json'))['last_cycle'])")
N=$(date -u -d "${L%??} ${L#????????}:00 6 hours" +%Y%m%d%H)
echo "$(date -u +%H:%MZ) prefetch: $N" >> stochastic/auto/cron.log
.venv/bin/python stochastic/fetch_ensemble.py $N ${GEFS_H:-120} --follow > stochastic/auto/prefetch_$N.log 2>&1 &
F=$!
# The GFS is complete ~20 min before the GEFS: build its table meanwhile (angle-parallel), the members need it first.
n=0; until [ -f data/ens_$N/gfs3h.npz ] || ! kill -0 $F 2>/dev/null; do sleep 20; done
[ -f data/ens_$N/gfs3h.npz ] && [ ! -f data/ens_$N/bellman_fine.npz ] && FULL_PACK=1 FINE_WORKERS=12 OMP_NUM_THREADS=1 .venv/bin/python bellman_fine.py data/ens_$N > stochastic/auto/prefetch_gfs_$N.log 2>&1
wait $F
echo "$(date -u +%H:%MZ) prefetch: $N downloaded ($(grep -c READY stochastic/auto/prefetch_$N.log) ready)" >> stochastic/auto/cron.log
/usr/bin/flock -w 3600 /tmp/routage_ultim.lock env NOTIFY_EMAIL="${NOTIFY_EMAIL:-}" NOTIFY_ALWAYS=1 PARALLEL="${PARALLEL:-8}" .venv/bin/python stochastic/auto/auto_update.py >> stochastic/auto/cron.log 2>&1
