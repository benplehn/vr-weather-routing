"""ETA a posteriori (vent vrai connu, route parfaite depuis ce point) le long de notre trace. Usage: eval_track.py TAG"""
import numpy as np,datetime as dt,sys
from pathlib import Path
from scipy.ndimage import map_coordinates
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
from track import TRACK,BOATS
TAIL=2.5 # fin de parcours commune après la ligne de passage (optimal_route.FINISH_TAIL_H)
G=dict(np.load(H/'data'/(sys.argv[1] if len(sys.argv)>1 else 'w74')/'bellman_fine.npz'))
T0=dt.datetime(2026,11,1)
def val(t,lat,lon,s,tk):
    L0=float(G['lead0']);x=t-L0;k=int(np.clip(np.floor(x),0,len(G['values'])-2));f=x-k
    j=2*s+(tk>0);c=[[(lat-G['lat'][0])/.25],[(lon-G['lon'][0])/.25]]
    a=map_coordinates(G['values'][k,j],c,order=1,cval=2000)[0];b=map_coordinates(G['values'][k+1,j],c,order=1,cval=2000)[0]
    return float(a*(1-f)+b*f)
def hrs(u):return (dt.datetime.strptime(u,'%Y%m%d%H%M')-T0).total_seconds()/3600
def date(e):return (T0+dt.timedelta(hours=float(e))).strftime('%d/%m %H:%MZ')
def eta(u,la,lo,s=None,tk=None):
    t=hrs(u);v=val(t,la,lo,s,tk) if s is not None else min(val(t,la,lo,a,b) for a in (0,1) for b in (-1,1))
    return t,v,t+v+TAIL
if __name__=='__main__':
    prev=None;rows=[]
    print('UTC           lat     lon     V(h)  ETA parfaite   perte segment')
    for u,la,lo,s,tk,kind in TRACK:
        t,v,e=eta(u,la,lo,s,tk)
        print(u,f'{la:6.2f} {lo:7.2f} {v:6.1f}  {date(e)}  {"" if prev is None else f"{e-prev:+5.1f} h"}  {kind}');prev=e
    for u,la,lo,n in BOATS:
        t,v,e=eta(u,la,lo);print(n,u,la,lo,'V',round(v,1),'ETA',date(e))
