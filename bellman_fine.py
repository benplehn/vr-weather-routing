"""Fine Bellman value: 0.25 deg grid, 1 h step, state (sail, tack) with VR-like maneuver penalties.

V[k, s, k_tack, lat, lon] = hours to the approximate finish (17N 61W r25NM) from lead L0+k.
Layer index j = 2*sail + (tack>0). Energy is not a state: penalties use a fixed typical energy.
Usage: bellman_fine.py ens_dir [step_deg dt_h [member]]  (default .25 1) -> ens_dir/bellman_fine[_step_dt].npz
With member (e.g. gefsp07): wind of that GEFS member while it exists (blended into GFS over its last 6 h),
GFS after; only the first KEEP_HOURS hourly values are stored -> ens_dir/bellman_members/MEMBER.npz
Values are stored hourly whatever dt is; the grid is stored in the file and used by value().
"""
import os,sys,time,json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parent
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'))
import optimal_route as p
STEP=.25;LAS=np.arange(p.LAT_LO,p.LAT_HI+1e-9,STEP);LOS=np.arange(p.LON_MIN,p.LON_HI+1e-9,STEP)
ANGLES=np.array([-160,-150,-140,-135,-130,-125,-120,-115,-105,-90,-75,-60,-50,50,60,75,90,105,115,120,125,130,135,140,150,160.])
TYPICAL_ENERGY=85.

class Field:
 """Fast wind lookup on a regular lat/lon grid with linear time interpolation (lead hours)."""
 def __init__(self,path):
  w=np.load(path);self.h=w['hours'].astype(float);self.lat0=float(w['lat'][0]);self.lon0=float(w['lon'][0])
  self.d=float(w['lat'][1]-w['lat'][0]);self.u=w['u'];self.v=w['v'];self._t=None
 def at(self,lead):
  if self._t!=lead:
   i=int(np.clip(np.searchsorted(self.h,lead)-1,0,len(self.h)-2));f=float(np.clip((lead-self.h[i])/(self.h[i+1]-self.h[i]),0,1))
   self._u=self.u[i]*(1-f)+self.u[i+1]*f;self._v=self.v[i]*(1-f)+self.v[i+1]*f;self._t=lead
  return self._u,self._v
 def __call__(self,lead,lat,lon):
  u,v=self.at(lead);lat,lon=np.broadcast_arrays(lat,lon)
  c=[((lat-self.lat0)/self.d).ravel(),((lon-self.lon0)/self.d).ravel()]
  uu=map_coordinates(u,c,order=1,mode='nearest',prefilter=False).reshape(lat.shape)
  vv=map_coordinates(v,c,order=1,mode='nearest',prefilter=False).reshape(lat.shape)
  return np.hypot(uu,vv)*1.943844492,np.degrees(np.arctan2(-uu,-vv))%360

class Blend:
 """GEFS member wind up to its last lead, linearly blended into GFS over the last 6 h of the member."""
 def __init__(self,member,gfs):self.m=member;self.g=gfs;self.end=float(member.h[-1])
 def __call__(self,lead,lat,lon):
  if lead<=self.end-6:return self.m(lead,lat,lon)
  if lead>=self.end:return self.g(lead,lat,lon)
  f=(lead-(self.end-6))/6;a,b=self.m(lead,lat,lon),self.g(lead,lat,lon)
  u=(1-f)*np.array(_uv(*a))+f*np.array(_uv(*b))
  return np.hypot(*u),np.degrees(np.arctan2(-u[0],-u[1]))%360
def _uv(ws,wd):return -ws*np.sin(np.radians(wd)),-ws*np.cos(np.radians(wd))
KEEP_HOURS=60

def interp(a,lat,lon,fill=2000.,lat0=LAS[0],lon0=LOS[0],step=STEP):
 lat,lon=np.broadcast_arrays(lat,lon)
 return map_coordinates(a,[((lat-lat0)/step).ravel(),((lon-lon0)/step).ravel()],order=1,mode='constant',cval=fill,prefilter=False).reshape(lat.shape)

def layer(sail,tack):return 2*np.asarray(sail)+(np.asarray(tack)>0)

def value(G,lead,lat,lon,sail,tack):
 """Remaining hours at lead time for arrays lat/lon and (sail,tack) arrays of same shape."""
 L0=float(G['lead0']);t=lead-L0;k=int(np.clip(np.floor(t),0,len(G['values'])-2));f=float(np.clip(t-k,0,1))
 g=dict(lat0=float(G['lat'][0]),lon0=float(G['lon'][0]),step=float(G['lat'][1]-G['lat'][0]))
 lat,lon,j=np.broadcast_arrays(lat,lon,layer(sail,tack));out=np.empty(lat.shape)
 for jj in range(4):
  m=j==jj
  if m.any():out[m]=interp(G['values'][k,jj],lat[m],lon[m],**g)*(1-f)+interp(G['values'][k+1,jj],lat[m],lon[m],**g)*f
 return out

def load_members(ens_dir,names,G):
 """Per-column value tables for names (gfs -> G, member -> ens_dir/bellman_members/member.npz), or None if absent."""
 d=Path(ens_dir)/'bellman_members'
 if not all((d/f'{n}.npz').exists() for n in names if n!='gfs'):return None
 return [G if n=='gfs' else dict(np.load(d/f'{n}.npz')) for n in names]

def value_members(Gs,lead,lat,lon,sail,tack):
 """value() with column j of axis 1 (scenario axis) read from Gs[j]."""
 lat,lon,sail,tack=np.broadcast_arrays(lat,lon,sail,tack);out=np.empty(lat.shape)
 for j,G in enumerate(Gs):out[:,j]=value(G,lead,lat[:,j],lon[:,j],sail[:,j],tack[:,j])
 return out

_P={} # per-step context shared with the worker processes (fork)
def _partial(job):
 # Partial min over a subset of angles for one backward step (same update as the serial loop in build()).
 angles,cur=job;la,lo,g,dt,ws,wd=(_P[k] for k in ('la','lo','g','dt','ws','wd'))
 best=np.full((4,*la.shape),2000.,np.float32)
 for a in angles:
  h=(wd-a)%360;nt=1 if a>0 else -1
  for ns in (0,1):
   sp=p.polar(a,ws,ns);jn=int(layer(ns,nt))
   for ps in (0,1):
    for pt in (-1,1):
     _,pen=p.maneuver(ws,TYPICAL_ENERGY,ps!=ns,pt!=nt)
     ln,on=p.move(la,lo,h,sp*(dt-.5*np.minimum(dt,pen)))
     val=np.where(p.leg_allowed(la,lo,ln,on),dt+interp(cur[jn],ln,on,**g),2000)
     jp=int(layer(ps,pt));np.minimum(best[jp],val,out=best[jp])
 return best
def _partial_ws(job):
 ch,cur,ws,wd=job;_P.update(ws=ws,wd=wd);return _partial((ch,cur))
def build(ens_dir,step=STEP,dt=1.,lead0=6.,horizon=None,member=None):
 # FINE_HORIZON (h after lead0, default 180). 90 tested 6/11: wrong values for slow points (terminal reached), kept 180.
 if horizon is None:horizon=int(os.environ.get('FINE_HORIZON','180'))
 W=Field(Path(ens_dir)/'gfs3h.npz')
 out=Path(ens_dir)/('bellman_fine.npz' if (step,dt)==(STEP,1.) else f'bellman_fine_{step:g}_{dt:g}.npz')
 if member:
  W=Blend(Field(Path(ens_dir)/f'{member}3h.npz'),W);out=Path(ens_dir)/'bellman_members'/f'{member}.npz'
  if os.environ.get('GATE_OUT'):out=Path(os.environ['GATE_OUT'])/f'{member}.npz'
  out.parent.mkdir(parents=True,exist_ok=True)
 las=np.arange(p.LAT_LO,p.LAT_HI+1e-9,step);los=np.arange(p.LON_MIN,p.LON_HI+1e-9,step);g=dict(lat0=las[0],lon0=los[0],step=step)
 lo,la=np.meshgrid(los,las);ok=p.allowed(la,lo);target,toff=p.goal_target(la,lo,step)
 # GATE (env JSON {"lon","lat","side"}): homotopy-class constraint for stochastic/strategy.py. A wall 2*half deg wide
 # (wider than one hour of sailing) along meridian lon closes lat<lat (side N: the route must pass north of the gate)
 # or lat>=lat (side S). Output then goes to GATE_OUT/MEMBER.npz.
 gate=json.loads(os.environ['GATE']) if os.environ.get('GATE') else None
 if gate:ok=ok&~((np.abs(lo-gate['lon'])<=gate.get('half',.75))&((la<gate['lat']) if gate['side']=='N' else (la>=gate['lat'])))
 n=int(round(horizon/dt));per=int(round(1/dt))
 V=np.empty((horizon+1,4,*la.shape),np.float32)
 # Value on the passage line: fine Bellman of the end of the course (finish_local.py, same scenario) when present,
 # else the fixed estimate of optimal_route.goal_target. Per layer (sail, tack).
 FL=Path(ens_dir)/'finish_local'/f"{member or 'gfs'}.npz"
 loc=dict(np.load(FL)) if os.environ.get('FINISH_LOCAL','1')=='1' and FL.exists() else None
 nt_=int(target.sum());lt,ot=la[target],lo[target]
 tval=lambda L:(np.repeat(toff[target][None],4,0) if loc is None else np.array([value(loc,L,lt,ot,np.full(nt_,j//2),np.full(nt_,1 if j%2 else -1)) for j in range(4)],np.float32))
 print('finish values from',FL if loc is not None else 'fixed estimate',flush=True)
 term=np.where(target,toff,200+p.distance(la,lo)/20);term[~ok]=2000
 V[-1]=term;cur=np.repeat(term[None],4,0).astype(np.float32);cur[:,target]=tval(lead0+horizon);k0=n
 gfs_table=Path(ens_dir)/'bellman_fine.npz'
 r=int(round(step/STEP))
 if member and gfs_table.exists() and dt==1. and abs(step-r*STEP)<1e-9 and 'corridor' not in np.load(gfs_table).files and np.load(gfs_table)['values'].shape[-1]==(len(los)-1)*r+1: # a corridor (bellman_c2f) table has walls off the GFS route: members start from scratch
  # Past the member's last lead the wind is GFS, so the member value equals the GFS table: start there (same result,
  # ~35 % faster); a coarser grid (multiple of STEP) takes every r-th node. With a gate, walls are applied from there on.
  Gf=np.load(gfs_table);k0=min(int(W.end-lead0),n,len(Gf['values'])-1);cur=Gf['values'][k0][:,::r,::r].astype(np.float32);cur[:,~ok]=2000
 tic=time.time();NW=int(os.environ.get('FINE_WORKERS','1'));pool=None
 if NW>1: # FINE_WORKERS>1: each backward step split over angle chunks in worker processes (the GFS table, alone)
  import multiprocessing as mp;_P.update(la=la,lo=lo,g=g,dt=dt);pool=mp.get_context('fork').Pool(NW)
 for k in range(k0-1,-1,-1):
  ws,wd=W(lead0+(k+.5)*dt,la,lo)
  if pool is not None:
   # Workers are forked once; the step's wind is sent with the job (they cannot see later updates of _P).
   parts=pool.map(_partial_ws,[(ch,cur,ws,wd) for ch in np.array_split(ANGLES,NW)]);best=np.min(parts,axis=0)
   best[:,target]=tval(lead0+k*dt);best[:,~ok]=2000;cur=best
   if k%per==0:V[k//per]=best
   if k%(24*per)==0:print('fine Bellman',step,dt,k*dt,'h remaining',round(time.time()-tic,1),'s',flush=True)
   continue
  best=np.full((4,*la.shape),2000.,np.float32)
  for a in ANGLES:
   h=(wd-a)%360;nt=1 if a>0 else -1
   for ns in (0,1):
    sp=p.polar(a,ws,ns);jn=int(layer(ns,nt))
    for ps in (0,1):
     for pt in (-1,1):
      _,pen=p.maneuver(ws,TYPICAL_ENERGY,ps!=ns,pt!=nt)
      ln,on=p.move(la,lo,h,sp*(dt-.5*np.minimum(dt,pen)))
      val=dt+interp(cur[jn],ln,on,**g)
      val=np.where(p.leg_allowed(la,lo,ln,on),val,2000)
      jp=int(layer(ps,pt));np.minimum(best[jp],val,out=best[jp])
  best[:,target]=tval(lead0+k*dt);best[:,~ok]=2000;cur=best
  if k%per==0:V[k//per]=best
  if k%(24*per)==0:print('fine Bellman',step,dt,k*dt,'h remaining',round(time.time()-tic,1),'s',flush=True)
 if pool is not None:pool.close();pool.join()
 if member:V=V[:KEEP_HOURS+1]
 np.savez(out,values=V,lead0=lead0,lat=las,lon=los,typical_energy=TYPICAL_ENERGY,dt=dt)
 return out
if __name__=='__main__':
 a=sys.argv[1:];print('WROTE',build(a[0],*(float(x) for x in a[1:3]),member=a[3] if len(a)>3 else None))
