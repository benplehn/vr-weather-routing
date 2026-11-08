# Gated value tables of one route family only (fast path once the skipper has decided the family).
# Usage: build_family_gates.py ENS_DIR GATE_JSON SIDE(N|S) [WORKERS]  (GATE_JSON = gate.json of a previous cycle)
import os,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'stochastic'))
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'));os.environ.setdefault('FULL_PACK','1')
import strategy as st
ens=Path(sys.argv[1]);g=json.loads(Path(sys.argv[2]).read_text());gate={k:g[k] for k in ('lon','lat','half')}
mem=[f.name[:-len('3h.npz')] for f in sorted(ens.glob('gefs*3h.npz'))]
st.build_gates(ens,mem,gate,int(sys.argv[4]) if len(sys.argv)>4 else 16,sys.argv[3])
print('built',sys.argv[3],len(mem),gate,flush=True)
