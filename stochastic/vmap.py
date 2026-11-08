# Bellman value map (for the mail): expected hours to the finish line from every point at the program start, weighted
# mean over the GEFS scenarios (Bayesian weights), each scenario at its best sail/tack. Free 0.25 deg member tables when
# present, else the decided family's gated tables (fast path). Other boats (env VMAP_BOATS json [[lat,lon,name],...]) are
# placed on the map with their expected time and the gap to us (same instant, same weather: a pure position comparison).
# Usage: vmap.py CYCLE OBS LAT LON [OUT_PNG]  -> prints VMAP_PNG path and BOAT name hours delta lines.
import os,sys,json,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'stochastic')]
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'));os.environ.setdefault('FULL_PACK','1')
import numpy as np
import optimal_route as p,bellman_fine as bf,game_wind
UTC=datetime.timezone.utc;PARIS=datetime.timezone(datetime.timedelta(hours=1))
def tables(ens,mem):
 ok=(ens/'bellman_members').exists() and all((ens/'bellman_members'/f'{m}.npz').exists() for m in mem)
 if ok:
  Ts=[dict(np.load(ens/'bellman_members'/f'{m}.npz')) for m in mem]
  if float(Ts[0]['lon'][0])<=p.LON_MIN+.1:return Ts,'tables fines 0,25°' # (tables built before the 6/11 west extension: gated ones)
 import strategy as st
 S=json.loads((R/'stochastic/auto/state.json').read_text());g,T=st.load_gates(ens,mem)
 return T[S.get('family','N')],'tables route '+('nord' if S.get('family','N')=='N' else 'sud')+' 0,5°'
def expected(Ts,w,lead,la,lo):
 out=np.zeros(la.shape)
 for wj,G in zip(w,Ts):
  v=np.min([bf.value(G,lead,la,lo,np.full(la.shape,s),np.full(la.shape,t)) for s in (0,1) for t in (-1,1)],axis=0)
  out+=wj*v
 return out
def advance(Ts,w,W,lead_b,lead,lat,lon):
 """A boat reported at lead_b, moved to lead along the ideal route (greedy descent of the weighted expected value in the
 GFS wind, hourly, last part interpolated): its estimated position now, and the path for the map."""
 import strategy as st
 h=lead-lead_b
 if h<=.05:return lat,lon,[(lat,lon)]
 cost=lambda L,la,lo,s,t:expected_st(Ts,w,L,la,lo,s,t)
 v=[(expected_st(Ts,w,lead_b,np.array([lat]),np.array([lon]),np.array([s]),np.array([t]))[0],s,t) for s in (0,1) for t in (-1,1)]
 _,s0,t0=min(v);P=st.descend(cost,W,lead_b,lat,lon,s0,t0,int(np.ceil(h)))
 pts=[(x[1],x[2]) for x in P]
 if len(P)>int(h)+1: # interpolate the last (partial) hour
  f=h-int(h);a,b=P[int(h)],P[int(h)+1];pts=pts[:int(h)+1]+[(a[1]+f*(b[1]-a[1]),a[2]+f*(b[2]-a[2]))]
 return pts[-1][0],pts[-1][1],pts
def expected_st(Ts,w,lead,la,lo,s,t):
 return sum(wj*bf.value(G,lead,la,lo,s,t) for wj,G in zip(w,Ts))
def main():
 cyc,obs,lat,lon=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
 out=Path(sys.argv[5]) if len(sys.argv)>5 else R/'stochastic/auto'/f'vmap_{cyc}.png'
 ens=R/'data'/f'ens_{cyc}';names=['gfs']+[f.name[:-len('3h.npz')] for f in sorted(ens.glob('gefs*3h.npz'))];mem=names[1:]
 w=np.asarray(game_wind.scenario_weights(ens,names,cyc)[0],float);w/=w.sum()
 c0=datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC);t0=datetime.datetime.strptime(obs,'%Y%m%d%H%M').replace(tzinfo=UTC)
 lead=(t0-c0).total_seconds()/3600;Ts,src=tables(ens,mem)
 la,lo=np.meshgrid(np.arange(14,min(36,lat+6),.2),np.arange(p.LON_MIN,min(-44,lon+8),.2),indexing='ij')
 V=expected(Ts,w,lead,la,lo);land=~p.allowed(la,lo);V=np.where((V<400)&~land,V,np.nan);from scipy.ndimage import binary_dilation
 hole=np.isnan(V)&~binary_dilation(land,iterations=4)&(lo>p.LON_MIN+.6) # not near the coast nor the domain edge
 me=float(expected(Ts,w,lead,np.array([lat]),np.array([lon]))[0])
 boats=json.loads(os.environ.get('VMAP_BOATS','[]'));bu=os.environ.get('VMAP_BOATS_UTC');moved={}
 if boats and bu: # positions reported earlier: estimated now by following the ideal route since the report
  lb=(datetime.datetime.strptime(bu,'%Y%m%d%H%M').replace(tzinfo=UTC)-c0).total_seconds()/3600;W=bf.Field(ens/'gfs3h.npz')
  Tf=None # the gated tables force the route through our gate: other boats only move on the free (unconstrained) tables
  if (ens/'bellman_members').exists():
   try:Tf=[dict(np.load(ens/'bellman_members'/f'{m}.npz')) for m in mem]
   except Exception:Tf=None
  for b in boats:
   try:
    if Tf and float(Tf[0]['lon'][0])+.1<b[1]<2 and lead-lb>.05:
     la1,lo1,pts=advance(Tf,w,W,lb,lead,b[0],b[1]);moved[b[2]]=(b[0],b[1],pts);print('BOAT_MOVED',b[2],f'{la1:.3f}',f'{lo1:.3f}',f'{lead-lb:.2f}h',flush=True);b[0],b[1]=la1,lo1
   except Exception as x:print('BOAT_MOVE_FAILED',b[2],x,flush=True)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,8),dpi=110);ax.set_facecolor('#c9c2a7')
 top=float(np.ceil((me+24)/6)*6);lv=np.arange(0,top+1e-6,2);cf=ax.contourf(lo,la,np.minimum(V,top),levels=lv,cmap='viridis_r')
 cs=ax.contour(lo,la,V,levels=np.arange(6,top+1e-6,6),colors='white',linewidths=.8)
 ax.clabel(cs,fmt='%d h',fontsize=8)
 if hole.any(): # areas the tables leave out (with the gated route-nord tables: the wall south of the gate, east of it)
  ax.contourf(lo,la,hole.astype(float),levels=[.5,1.5],colors='none',hatches=['///'])
  ax.contour(lo,la,hole.astype(float),levels=[.5],colors='#555',linewidths=.6);fig.colorbar(cf,ax=ax,label="heures jusqu'à la ligne d'arrivée (moyenne pondérée)")
 ax.plot(lon,lat,'*',color='gold',mec='k',ms=18,zorder=5);leg=[f'★ toi : {me:.1f} h']
 for b in boats:
  v=float(expected(Ts,w,lead,np.array([b[0]]),np.array([b[1]]))[0]);ok=v<400 and p.LON_MIN<b[1]<2
  if not ok and p.LON_MIN<b[1]<2 and (ens/'bellman_members').exists(): # e.g. next to the gate wall of the family tables
   try:
    Tf=[dict(np.load(ens/'bellman_members'/f'{m}.npz')) for m in mem];v=float(expected(Tf,w,lead,np.array([b[0]]),np.array([b[1]]))[0]);ok=v<400
   except Exception:pass
  print('BOAT',b[2],f'{v:.2f}' if ok else 'hors-domaine',f'{v-me:+.2f}' if ok else '',flush=True)
  if not ok: # west of the computed domain (65 W): value at 64.5 W, same latitude (rough estimate)
   ve=float(expected(Ts,w,lead,np.array([b[0]]),np.array([p.LON_MIN+.5]))[0]);print('BOAT_EDGE',b[2],f'{ve:.2f}',f'{ve-me:+.2f}',flush=True)
  x=min(max(b[1],p.LON_MIN+.2),-44.2);ax.plot(x,b[0],'o',color='#d33',mec='k',ms=13,zorder=5)
  if b[2] in moved:
   q=np.array(moved[b[2]][2]);ax.plot(q[:,1],q[:,0],':',color='#d33',lw=1.5,zorder=4);ax.plot(moved[b[2]][1],moved[b[2]][0],'o',mfc='none',mec='#d33',ms=7,zorder=4)
  k=len(leg);ax.annotate(str(k),(x,b[0]),fontsize=9,fontweight='bold',color='white',ha='center',va='center',zorder=6)
  leg.append(f"{k} {b[2]}{' (estimé)' if b[2] in moved else ''} : {v:.1f} h, {abs(v-me):.1f} h {'devant' if v<me else 'derrière'} toi" if ok else
   f"{k} {b[2]} ({abs(b[1]):.1f}°W, hors carte) : ≈ {ve:.1f} h ({ve-me:+.1f} h vs toi)")
 ax.plot(p.FINISH[1],p.FINISH[0],'s',color='k',ms=7);ax.plot(p.MARK_NW[1],p.MARK_NW[0],'^',color='orange',mec='k',ms=8)
 if hole.any():leg.append('hachures : zone hors des tables (mur de la porte route nord)')
 ax.text(.99,.99,'\n'.join(leg),transform=ax.transAxes,fontsize=9.5,va='top',ha='right',zorder=7,bbox=dict(fc='white',ec='#888',alpha=.92,pad=4))
 ax.set_xlabel('longitude');ax.set_ylabel('latitude');ax.set_aspect(1/np.cos(np.radians(25)))
 ax.set_title(f"Carte de Bellman — heures jusqu'à l'arrivée, {t0.astimezone(PARIS):%d/%m %Hh%M} Paris\nbulletin {cyc[8:]}Z, {len(mem)} scénarios pondérés ({src}). "
  "Même courbe = même heure d'arrivée."+(f"\nconcurrents : relevé {datetime.datetime.strptime(bu,'%Y%m%d%H%M').replace(tzinfo=UTC).astimezone(PARIS):%Hh%M} (cercle vide), "
  "position estimée maintenant en suivant la route idéale (pointillés)" if moved else ''),fontsize=9)
 fig.tight_layout();fig.savefig(out);plt.close(fig);print('VMAP_PNG',out,'ME',f'{me:.2f}',flush=True)
if __name__=='__main__':main()
