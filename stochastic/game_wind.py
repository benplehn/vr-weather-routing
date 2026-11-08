# Game wind (point 1). In Virtual Regatta the "true" wind is the deterministic GFS of the bulletin the game has
# applied, not the real atmosphere. So, seen from a new cycle C:
#   lead < SWITCH            : previous cycle's GFS (still applied in the game; SWITCH = GAME_DELAY h after C, default 6)
#   SWITCH <= lead < KNOWN   : GFS of C (applied, known exactly; KNOWN = SWITCH + 6, when the next, unknown cycle lands)
#   KNOWN <= lead < KNOWN+6  : linear blend into the scenario (GEFS member j, or GFS for column 0)
#   lead >= KNOWN + 6        : scenario
# Uncertainty therefore only starts with the next bulletin, as in the game. Usage in optimize/refine: GAME_WIND=1.
import os,sys,datetime,urllib.request,urllib.parse,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import bellman_fine as bf
DELAY=float(os.environ.get('GAME_DELAY','6'))
def prev_cycle(cyc):return (datetime.datetime.strptime(cyc,'%Y%m%d%H')-datetime.timedelta(hours=6)).strftime('%Y%m%d%H')
def fetch_prev(cyc):
 # GFS 0.25 10 m wind of a cycle, f003..f018 every 3 h, ensemble box -> data/gfs_game/gfs_CYC3h.npz (unless data/ens_CYC exists).
 ens=R/'data'/f'ens_{cyc}'/'gfs3h.npz'
 if ens.exists():return ens
 import eccodes as ec
 D=R/'data'/'gfs_game';D.mkdir(exist_ok=True);out=D/f'gfs_{cyc}3h.npz'
 if out.exists():return out
 day,hh=cyc[:8],cyc[8:];U=[];V=[];hours=list(range(3,19,3))
 for h in hours:
  name=f'gfs.t{hh}z.pgrb2.0p25.f{h:03}';p=D/f'{cyc}_{name}'
  if not p.exists():
   q=urllib.parse.urlencode(dict(file=name,dir=f'/gfs.{day}/{hh}/atmos',lev_10_m_above_ground='on',var_UGRD='on',var_VGRD='on',subregion='',leftlon=-75,rightlon=5,toplat=58,bottomlat=5))
   for a in range(4):
    try:
     data=urllib.request.urlopen('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'+q,timeout=40).read()
     if not data.startswith(b'GRIB'):raise ValueError(data[:80])
     p.write_bytes(data);break
    except Exception:
     if a==3:raise
     time.sleep(5*(a+1))
  fields={}
  with p.open('rb') as f:
   while (g:=ec.codes_grib_new_from_file(f)) is not None:
    short=ec.codes_get(g,'shortName');lat=ec.codes_get_array(g,'latitudes');lon=(ec.codes_get_array(g,'longitudes')+180)%360-180;vals=ec.codes_get_values(g)
    las=np.unique(lat);los=np.unique(lon);a=np.empty((len(las),len(los)));a[np.searchsorted(las,lat),np.searchsorted(los,lon)]=vals
    fields[short]=a;ec.codes_release(g)
  U.append(fields['10u']);V.append(fields['10v'])
 np.savez_compressed(out,hours=hours,lat=las,lon=los,u=np.array(U,np.float32),v=np.array(V,np.float32));return out
def _uv(ws,wd):return -ws*np.sin(np.radians(wd)),-ws*np.cos(np.radians(wd))
class GameWind:
 """Callable (lead in hours after the new cycle, lat, lon) -> (tws kn, twd deg) for one scenario."""
 def __init__(self,prev,new,scenario,switch,known,blend=6.):
  self.prev,self.new,self.sc,self.switch,self.known,self.blend=prev,new,scenario,switch,known,blend
 def __call__(self,lead,lat,lon):
  if lead<self.switch and self.prev is not None:return self.prev(lead+6,lat,lon) # previous cycle is 6 h older
  if lead<self.known:return self.new(lead,lat,lon)
  if lead>=self.known+self.blend:return self.sc(lead,lat,lon)
  f=(lead-self.known)/self.blend;a,b=self.new(lead,lat,lon),self.sc(lead,lat,lon)
  u=(1-f)*np.array(_uv(*a))+f*np.array(_uv(*b));return np.hypot(*u),np.degrees(np.arctan2(-u[0],-u[1]))%360
class NextWind:
 """Quick night update (GAME_NEXT=next cycle): the next deterministic GFS (first 18 h, out ~3 h20 after its cycle)
 replaces the scenarios while the game applies it ([known, known+6) after cycle C), then 6 h blend into the scenario."""
 def __init__(self,base,nxt,scenario,known,blend=6.):self.base,self.nxt,self.sc,self.known,self.blend=base,nxt,scenario,known,blend
 def __call__(self,lead,lat,lon):
  if lead<self.known:return self.base(lead,lat,lon)
  if lead<self.known+6:return self.nxt(lead-6,lat,lon) # next cycle is 6 h younger
  if lead>=self.known+6+self.blend:return self.sc(lead,lat,lon)
  f=(lead-self.known-6)/self.blend;a,b=self.nxt(lead-6,lat,lon),self.sc(lead,lat,lon)
  u=(1-f)*np.array(_uv(*a))+f*np.array(_uv(*b));return np.hypot(*u),np.degrees(np.arctan2(-u[0],-u[1]))%360
def scenario_weights(ens,names,cyc):
 """Bayesian update of the scenarios with the GFS the game applies (the next one, GAME_NEXT, in quick runs; this cycle's otherwise): weight of member j proportional to
 exp(-t (d_j^2 - min d^2)), d_j = RMS vector wind difference (kn) to the next GFS over the route box, leads 12..24 h after
 cycle C (next GFS f006..f018), temperature set by a target effective sample size (SCEN_ESS).
 Ones when GAME_NEXT is unset or SCEN_WEIGHTS=0. Returns (weights over names[1:], effective sample size)."""
 n=len(names)-1
 if os.environ.get('SCEN_WEIGHTS','1')=='0' or names[1:]==['gfs']*n:return np.ones(n),float(n)
 # Box of what is left of the course (was 25-50 N / 60-20 W, the start of the race: the boat has left it).
 ens=Path(ens);la,lo=np.meshgrid(np.arange(14,40.1,1.),np.arange(-65,-39.9,1.),indexing='ij');d=[]
 if os.environ.get('GAME_NEXT'): # quick: the next GFS, which the game applies from C+12 h (its f006..f018)
  nxt=bf.Field(fetch_prev(os.environ['GAME_NEXT']));leads=list(range(12,25,3));ref=[_uv(*nxt(L-6,la,lo)) for L in leads]
 else: # full run: this cycle's GFS, which the game applies from C+6 h (f006..f018): the scenarios that agree are weighted up
  g=bf.Field(ens/'gfs3h.npz');leads=list(range(6,19,3));ref=[_uv(*g(L,la,lo)) for L in leads]
 for nm in names[1:]:
  F=bf.Field(ens/f'{nm}3h.npz');e=0.
  for L,(ur,vr) in zip(leads,ref):
   u,v=_uv(*F(L,la,lo));e+=np.mean((u-ur)**2+(v-vr)**2)
  d.append(np.sqrt(e/len(leads)))
 # The grid points are not independent, so the likelihood temperature is set by the target effective sample size
 # (SCEN_ESS, fraction of members, default 0.5): the update is clear but no scenario is thrown away.
 d=np.array(d);x=(d**2-d.min()**2);target=float(os.environ.get('SCEN_ESS','0.5'))*n
 W=lambda t:np.exp(-x*t);E=lambda w:w.sum()**2/(w**2).sum();lo,hi=0.,1.
 while E(W(hi))>target and hi<1e6:hi*=2
 for _ in range(60):
  mid=(lo+hi)/2
  if E(W(mid))>target:lo=mid
  else:hi=mid
 w=W(lo);w=w/w.mean();w=np.minimum(w,3.);w=w/w.mean();ess=E(w) # cap: the control run is always closest to GFS
 print('SCEN_WEIGHTS d',np.round(d,1).tolist(),'w',np.round(w,2).tolist(),'ESS',round(ess,1),flush=True)
 return w,ess
def fields(ens,names,cyc):
 """One GameWind per column of `names` ('gfs' then GEFS members), as used by optimize_pack/refine_pack."""
 ens=Path(ens);gfs=bf.Field(ens/'gfs3h.npz')
 try:prev=bf.Field(fetch_prev(prev_cycle(cyc)))
 except Exception as e:print('GAME_WIND: previous cycle unavailable, using new GFS from the start:',e,flush=True);prev=None
 switch=DELAY;known=DELAY+6
 W=[GameWind(prev,gfs,gfs if n=='gfs' else bf.Field(ens/f'{n}3h.npz'),switch,known) for n in names]
 if os.environ.get('GAME_NEXT'):
  nxt=bf.Field(fetch_prev(os.environ['GAME_NEXT']));print('GAME_NEXT',os.environ['GAME_NEXT'],flush=True)
  W=[NextWind(w,nxt,w.sc,known) for w in W]
 return W
