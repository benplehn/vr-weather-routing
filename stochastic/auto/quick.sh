#!/bin/sh
# Quick update at every bulletin: as soon as the next GFS (last full cycle + 6 h) is out to f018, re-plan on the tables
# of the last full cycle (its GEFS members weighted by the new GFS, Bayesian) with the new GFS as game wind.
# Result ~25 min after the GFS, before the game applies it; the full run (new GEFS, new tables) follows ~2 h later.
cd "$(dirname "$0")/../.." || exit 1
[ -f config.env ] && . ./config.env  # NOTIFY_EMAIL, PARALLEL (see config.env.example)
L=$(.venv/bin/python -c "import json;print(json.load(open('stochastic/auto/state.json'))['last_cycle'])")
N=$(date -u -d "${L%??} ${L#????????}:00 6 hours" +%Y%m%d%H)
U="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.${N%??}/${N#????????}/atmos/gfs.t${N#????????}z.pgrb2.0p25.f018.idx"
n=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' "$U")" = 200 ]; do n=$((n+1)); [ $n -gt 60 ] && exit 1; sleep 60; done
echo "$(date -u +%H:%MZ) quick: $N f018 available (tables $L)" >> stochastic/auto/cron.log
/usr/bin/flock -w 1800 /tmp/routage_ultim.lock env NOTIFY_EMAIL="${NOTIFY_EMAIL:-}" NOTIFY_ALWAYS=1 PARALLEL="${PARALLEL:-8}" .venv/bin/python stochastic/auto/auto_update.py --quick >> stochastic/auto/cron.log 2>&1
