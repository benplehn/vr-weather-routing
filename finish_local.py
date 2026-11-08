"""Fine Bellman of the end of the course, per scenario (NW Basse-Terre mark to the finish line).

Domain 15.70-16.50 N, 62.90-61.40 W, 1' grid (~1 NM), 10 min steps, layers (sail Jib/Spi, tack), maneuver penalties at
the typical energy, coastline of data/land.geojson with a 0.5 NM margin, the game's forbidden zone (optimal_route.ZONE).
Target: within 1.5 NM of the finish (16°06'12"N 61°32'31"W). From the passage line west of the mark the boat goes down
the west coast, round the south of Basse-Terre and up to the line, in the scenario's wind (Basse-Terre lee as the model
sees it). V[k, layer, lat, lon] = hours to the finish from lead lead0+k (hourly), read by bellman_fine.py as the value of
its passage line (instead of a fixed 20 kn estimate).
Usage: finish_local.py ENS_DIR NAME [LEAD_FROM]   (NAME = gfs or a GEFS member) -> ENS_DIR/finish_local/NAME.npz
"""
import os,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parent
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'));os.environ.setdefault('FULL_PACK','1')
import shapely
import optimal_route as p
import bellman_fine as bf
D=1/60;LAS=np.arange(15.70,16.50+1e-9,D);LOS=np.arange(-62.90,-61.40+1e-9,D);DT=1/6;FIN_R=1.5
def grid():
 lo,la=np.meshgrid(LOS,LAS)
 ok=~shapely.contains_xy(p.LAND.buffer(.5/60),lo,la)&~p.in_zone(la,lo)
 return la,lo,ok
def build(ens,name,lead_from=12.,span=120.):
 ens=Path(ens);gfs=bf.Field(ens/'gfs3h.npz');W=gfs if name=='gfs' else bf.Blend(bf.Field(ens/f'{name}3h.npz'),gfs)
 lead_to=min(float(gfs.h[-1]),lead_from+span)
 la,lo,ok=grid();okf=ok.astype(np.float32);fin=p.distance(la,lo,p.FINISH)<FIN_R
 g=dict(lat0=LAS[0],lon0=LOS[0],step=D)
 okat=lambda a,b:map_coordinates(okf,[(a-LAS[0])/D,(b-LOS[0])/D],order=0,mode='constant',cval=0,prefilter=False)>.5
 n=int(round((lead_to-lead_from)/DT));per=int(round(1/DT))
 term=np.where(fin,0,p.distance(la,lo,p.FINISH)/12.).astype(np.float32);term[~ok]=2000
 cur=np.repeat(term[None],4,0);V=np.empty((n//per+1,4,*la.shape),np.float32);V[-1]=cur;tic=time.time()
 for k in range(n-1,-1,-1):
  ws,wd=W(lead_from+(k+.5)*DT,la,lo);best=np.full((4,*la.shape),2000.,np.float32)
  for a in bf.ANGLES:
   h=(wd-a)%360;nt=1 if a>0 else -1
   for ns in (0,1):
    sp=p.polar(a,ws,ns);jn=int(bf.layer(ns,nt))
    for ps in (0,1):
     for pt in (-1,1):
      _,pen=p.maneuver(ws,bf.TYPICAL_ENERGY,ps!=ns,pt!=nt)
      ln,on=p.move(la,lo,h,sp*(DT-p.PEN_LOSS*np.minimum(DT,pen)))
      val=DT+bf.interp(cur[jn],ln,on,**g)
      val=np.where(okat(ln,on)&okat((la+ln)/2,(lo+on)/2),val,2000)
      jp=int(bf.layer(ps,pt));np.minimum(best[jp],val,out=best[jp])
  best[:,fin]=0;best[:,~ok]=2000;cur=best
  if k%per==0:V[k//per]=best
 out=ens/'finish_local'/f'{name}.npz';out.parent.mkdir(exist_ok=True)
 np.savez(out,values=V,lead0=lead_from,lat=LAS,lon=LOS,dt=1.)
 print('finish_local',name,'leads',lead_from,lead_to,'s',round(time.time()-tic,1),flush=True)
 return out
_OK=[]
def descend(G,W,lead,lat,lon,s,t,hmax=24.):
 """Greedy 10-min descent of a finish table G (npz dict) in wind W(lead,lat[],lon[]); path (lead,lat,lon,sail,tack)
 from a point of the passage line to the finish (for the map)."""
 if not _OK:_OK.append(grid()[2].astype(np.float32))
 okat=lambda a,b:map_coordinates(_OK[0],[(a-LAS[0])/D,(b-LOS[0])/D],order=0,mode='constant',cval=0,prefilter=False)>.5
 A=np.repeat(bf.ANGLES,2);NS=np.tile([0,1],len(bf.ANGLES));NT=np.where(A>0,1,-1);s,t=int(s),int(t)
 path=[(lead,lat,lon,s,t)]
 for _ in range(int(hmax/DT)):
  if float(p.distance(np.array([lat]),np.array([lon]),p.FINISH)[0])<FIN_R:break
  ws,wd=W(lead+DT/2,np.array([lat]),np.array([lon]));ws,wd=float(ws[0]),float(wd[0])
  _,pen=p.maneuver(np.full(A.shape,ws),bf.TYPICAL_ENERGY,NS!=s,NT!=t)
  ln,on=p.move(np.full(A.shape,lat),np.full(A.shape,lon),(wd-A)%360,p.polar(A,np.full(A.shape,ws),NS)*(DT-p.PEN_LOSS*np.minimum(DT,pen)))
  c=np.where(okat(ln,on)&okat((lat+ln)/2,(lon+on)/2),DT+bf.value(G,lead+DT,ln,on,NS,NT),np.inf)
  if not np.isfinite(c).any():break
  i=int(np.argmin(c));lat,lon,s,t=float(ln[i]),float(on[i]),int(NS[i]),int(NT[i]);lead+=DT;path.append((lead,lat,lon,s,t))
 return path
if __name__=='__main__':build(sys.argv[1],sys.argv[2],float(sys.argv[3]) if len(sys.argv)>3 else 12.)
