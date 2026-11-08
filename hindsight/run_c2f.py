"""Test du grossier->fin sur le cas a posteriori (même boîte et vent que run_bellman.py w74)."""
import os, sys, json, time
from pathlib import Path
R = Path(__file__).resolve().parent.parent
os.environ.update(LON_MIN='-74', BOX_LAT_LO='10', BOX_LAT_HI='52', BOX_LON_HI='-1', FINISH_LOCAL='0')
os.environ.setdefault('FINE_WORKERS', '7'); os.environ['ROUTING_SCENARIO'] = str(R / 'hindsight/scenario.json')
sys.path.insert(0, str(R))
import bellman_c2f as c
LEAD0 = 13 + 13 / 60
print('WROTE', c.build(R / 'hindsight/data/c2f', LEAD0, [(48.959167, -2.645278)], horizon=int(228 - LEAD0) - 1))
