"""Non-anticipative (non-clairvoyant) ensemble Bellman value.

V(t,x,s) = min_a mean_j [ dt + V(t+dt, x + v_j(a) dt, s') ]   over the GEFS members j (equal weights),
i.e. the expectation is inside the min: one decision for all scenarios, the scenario is never revealed.
This is the pessimistic bound (real forecasts are refreshed every 6 h), whereas per-member tables
(bellman_fine.py ... MEMBER, min inside the expectation) are the optimistic, clairvoyant bound.
Used as terminal value, the feedback phase of optimize_pack becomes non-anticipative too: each member
reacts to its own current wind (observed) but not to its future.
Members up to the last GEFS lead (blended into GFS over 6 h like bellman_fine.Blend); before that lead the
terminal is the GFS fine table. Usage: bellman_na.py ens_dir [workers] -> ens_dir/bellman_na.npz
"""
import os,sys,time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'))
import optimal_route as p
import bellman_fine as bf
STEP=.25;LAS=np.arange(p.LAT_LO,p.LAT_HI+1e-9,STEP);LOS=np.arange(p.LON_MIN,p.LON_HI+1e-9,STEP);G=dict(lat0=LAS[0],lon0=LOS[0],step=STEP)
LO,LA=np.meshgrid(LOS,LAS);OK=p.allowed(LA,LO);TARGET,TOFF=p.goal_target(LA,LO)
_W=None
def _init(ens_dir):
 global _W
 gfs=bf.Field(Path(ens_dir)/'gfs3h.npz')
 _W=[bf.Blend(bf.Field(f),gfs) for f in sorted(Path(ens_dir).glob('gefs*3h.npz'))]
def _step(job):
 # Partial min over a subset of angles for one time step; returns best (4, lat, lon).
 lead,angles,cur=job;dt=1.
 W=[w(lead+.5*dt,LA,LO) for w in _W]
 best=np.full((4,*LA.shape),2000.,np.float32)
 for a in angles:
  nt=1 if a>0 else -1
  for ns in (0,1):
   jn=int(bf.layer(ns,nt))
   for ps in (0,1):
    for pt in (-1,1):
     acc=np.zeros(LA.shape)
     for ws,wd in W:
      _,pen=p.maneuver(ws,bf.TYPICAL_ENERGY,ps!=ns,pt!=nt)
      ln,on=p.move(LA,LO,(wd-a)%360,p.polar(a,ws,ns)*(dt-.5*np.minimum(dt,pen)))
      acc+=np.where(p.leg_allowed(LA,LO,ln,on),dt+bf.interp(cur[jn],ln,on,**G),2000)
     jp=int(bf.layer(ps,pt));np.minimum(best[jp],(acc/len(W)).astype(np.float32),out=best[jp])
 return best
def build(ens_dir,workers=7):
 ens_dir=Path(ens_dir);Gf=dict(np.load(ens_dir/'bellman_fine.npz'));lead0=float(Gf['lead0'])
 m0=bf.Field(next(iter(sorted(ens_dir.glob('gefs*3h.npz')))));end=int(m0.h[-1]) # last member lead
 n=int(end-lead0);V=np.empty((n+1,4,*LA.shape),np.float32)
 cur=Gf['values'][n].astype(np.float32);V[n]=cur # GFS fine value at the last member lead
 chunks=np.array_split(bf.ANGLES,workers);tic=time.time()
 with Pool(workers,_init,(str(ens_dir),)) as pool:
  for k in range(n-1,-1,-1):
   parts=pool.map(_step,[(lead0+k,c,cur) for c in chunks])
   best=np.min(parts,axis=0);best[:,TARGET]=TOFF[TARGET];best[:,~OK]=2000;cur=best;V[k]=best
   if k%12==0:print('NA Bellman lead',lead0+k,'elapsed',round(time.time()-tic),'s',flush=True)
 out=ens_dir/'bellman_na.npz';np.savez(out,values=V,lead0=lead0,lat=LAS,lon=LOS,typical_energy=bf.TYPICAL_ENERGY,dt=1.,members=len(list(ens_dir.glob('gefs*3h.npz'))))
 return out
if __name__=='__main__':print('WROTE',build(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 7))
