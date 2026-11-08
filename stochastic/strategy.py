# Strategic decision (homotopy classes, Manchester / Lantoine style): discrete choice of the route family, then the
# continuous optimum inside each family.
#  1. Families: in every GEFS scenario the optimal route to the finish is traced (greedy descent of that member's
#     Bellman table, member game wind); the routes are split in two families by where they cross the reference
#     meridians (2-means). The gate = the meridian where they separate most, at the mid latitude.
#  2. Loss matrix: per scenario j and family f, V[f,j] = time to go with the route forced through the gate on side f
#     (gated member tables, bellman_fine.py GATE, 0.5 deg). Loss[f,j] = V[f,j] - min_f V[f,j]; the family chosen
#     minimizes the weighted expected time (weights = Bayesian scenario weights when a newer GFS is known).
#  3. Value of waiting = min_f E[V_f] - E[min_f V_f] (upper bound: the scenario fully known before committing).
#     Point of no return: along the waiting route (greedy descent of E[min_f V_f], GFS game wind), the first time when
#     following it costs more than TOL h to one family: D_f(t) = E[V_f](x(t),t) + (t-t0) - E[V_f](x0,t0).
#  4. The chosen program is checked: D_f at its end (position after H h) > TOL = the program closes family f.
#  5. Rank: the fleet (deterministic GFS router) takes the family of the GFS route; per family, probability of gaining
#     at least GOAL_H h on it (scenario by scenario).
#  6. Map (PNG): routes of every scenario coloured by family, gate, waiting route, boat.
# Usage: strategy.py CYCLE OBS LAT LON SAIL TACK [--build] [--program LATH LONH SAILH TACKH H]
#  --build: (re)compute the gated tables of CYCLE for the current gate (~6 min on 14 workers) when missing.
# Out: stochastic/auto/strategy_CYCLE.json/.png and STRATEGY lines (mail), STRATEGY_MAP path.
import os,sys,json,datetime
from pathlib import Path
from multiprocessing import Pool
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'stochastic'))
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'));os.environ.setdefault('FULL_PACK','1') # same boat model as the tables
import optimal_route as p
import bellman_fine as bf
import game_wind
UTC=datetime.timezone.utc;PARIS=datetime.timezone(datetime.timedelta(hours=1))
MERIDIANS=(-45.,-50.,-55.,-58.);TOL=float(os.environ.get('STRAT_TOL','0.3'));GOAL=float(os.environ.get('GOAL_H','0.3'))
GSTEP=.5;GHALF=.75;HMAX=48
FL={}
def near_line(la,lo):return la<16.45 and -62.9<=lo<=-61.8 # reached the passage line west of the NW Basse-Terre mark
def fine_leg(ens,name,W,P):
 """Finish leg (fine local table of the scenario, finish_local.py) appended to a route that reached the passage line."""
 x=P[-1];f=Path(ens)/'finish_local'/f'{name}.npz'
 if not near_line(x[1],x[2]) or not f.exists() or os.environ.get('FINISH_LOCAL')=='0':return P
 import finish_local as fl
 if name not in FL:FL[name]=dict(np.load(f))
 return P+fl.descend(FL[name],W,*x)[1:]
def sub(P,n=3):
 """Hourly part thinned to every n h for the map, finish leg (10 min steps) kept whole."""
 k=next((i for i in range(1,len(P)) if P[i][0]-P[i-1][0]<.9),len(P));return P[:k:n]+P[k-1:]
def candidates(la,lo,s,t,ws,wd):
 """All (angle, sail) moves for one hour from one point: next lat/lon arrays, new sail/tack, allowed mask."""
 A=np.repeat(bf.ANGLES,2);NS=np.tile([0,1],len(bf.ANGLES));NT=np.where(A>0,1,-1)
 _,pen=p.maneuver(np.full(A.shape,float(ws)),bf.TYPICAL_ENERGY,NS!=s,NT!=t)
 ln,on=p.move(np.full(A.shape,la),np.full(A.shape,lo),(wd-A)%360,p.polar(A,np.full(A.shape,float(ws)),NS)*(1-.5*np.minimum(1,pen)))
 return ln,on,NS,NT,p.leg_allowed(np.full(A.shape,la),np.full(A.shape,lo),ln,on)
def descend(cost,W,lead,lat,lon,sail,tack,hmax,stop=None):
 """Greedy descent of cost(lead, lat[], lon[], sail[], tack[]) with wind W; path of (lead, lat, lon, sail, tack)."""
 la,lo,s,t=lat,lon,int(sail),int(tack);path=[(lead,la,lo,s,t)]
 for k in range(hmax):
  if stop and stop(lead+k,la,lo,s,t):break
  ws,wd=W(lead+k+.5,np.array([la]),np.array([lo]));ln,on,NS,NT,ok=candidates(la,lo,s,t,float(ws[0]),float(wd[0]))
  if not ok.any():break
  c=np.where(ok,1+cost(lead+k+1,ln,on,NS,NT),np.inf);i=int(np.argmin(c))
  la,lo,s,t=float(ln[i]),float(on[i]),int(NS[i]),int(NT[i]);path.append((lead+k+1,la,lo,s,t))
 return path
def crossing(path,m):
 """Latitude where the route crosses meridian m (linear), None if it does not."""
 for a,b in zip(path,path[1:]):
  if (a[2]-m)*(b[2]-m)<=0 and a[2]!=b[2]:return a[1]+(b[1]-a[1])*(m-a[2])/(b[2]-a[2])
 return None
def families(rows):
 """2-means on the crossing latitudes; gate = meridian of largest separation, mid latitude; None if no real split."""
 X=np.array([[x if x is not None else np.nan for x in r['cross']] for r in rows],float)
 # Meridians already behind the boat (no route crosses them) are dropped: one all-NaN column made every distance NaN,
 # all labels 0 and a false "single family" (bug of 05/11, boat past 45 W).
 idx=np.where(~np.isnan(X).all(0))[0]
 if len(idx)==0:return np.zeros(len(rows),int),None,None
 X=X[:,idx];X=np.where(np.isnan(X),np.nanmean(X,axis=0),X)
 mi=int(np.argmax(np.ptp(X,axis=0)));c=np.array([X[np.argmax(X[:,mi])],X[np.argmin(X[:,mi])]])
 for _ in range(30):
  lab=np.argmin(((X[:,None,:]-c[None])**2).sum(-1),axis=1);c=np.array([X[lab==k].mean(0) if (lab==k).any() else c[k] for k in (0,1)])
 mi=int(np.argmax(np.abs(c[0]-c[1])))
 if abs(c[0,mi]-c[1,mi])<=3 or len(set(lab))<2:return lab*0,None,c
 return lab,dict(lon=MERIDIANS[idx[mi]],lat=round(float(c[0,mi]+c[1,mi])/2*2)/2,half=GHALF),c # lab 0 = north
def _build(job):
 ens,member,side,gate,out=job
 os.environ['GATE']=json.dumps({**gate,'side':side});os.environ['GATE_OUT']=str(out);os.environ['OMP_NUM_THREADS']='1'
 bf.build(ens,GSTEP,1.,member=member);return member,side
def build_gates(ens,names,gate,workers,sides='NS'):
 d=Path(ens)/'bellman_gates';jobs=[]
 for side in sides:
  for n in names:
   if not (d/side/f'{n}.npz').exists():jobs.append((str(ens),n,side,gate,d/side))
 with Pool(workers,maxtasksperchild=1) as pool:
  for m,sd in pool.imap_unordered(_build,jobs):pass
 (d/'gate.json').write_text(json.dumps(dict(gate,members=names,sides=sides)))
def load_gates(ens,names):
 d=Path(ens)/'bellman_gates'
 if not (d/'gate.json').exists():return None,None
 g=json.loads((d/'gate.json').read_text())
 sides=g.get('sides','NS')
 if g.get('members')!=names or not all((d/s/f'{n}.npz').exists() for s in sides for n in names):return None,None
 return g,{s:[dict(np.load(d/s/f'{n}.npz')) for n in names] for s in sides}
def Vg(T,lead,la,lo,s,t):
 """Gated values, shape (members, points)."""
 la,lo,s,t=(np.atleast_1d(np.asarray(x,float)) for x in (la,lo,s,t))
 return np.array([bf.value(G,lead,la,lo,s.astype(int),t.astype(int)) for G in T])
def draw(path_png,rows,lab,gate,hedge,start,marks,title,hedge_label="route d'attente (options ouvertes)",track=None,wts=None,tlabel=None):
 # wts (family decided): rows are the decided family's route of each scenario, line width/opacity = Bayesian weight.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 la,lo=np.meshgrid(np.arange(10,55.01,.25),np.arange(-70,0.01,.25),indexing='ij')
 fig,(ax,az)=plt.subplots(1,2,figsize=(14,6.4),dpi=110,gridspec_kw=dict(width_ratios=[2.3,1]));ax.set_facecolor('#dbe9f5')
 # Real land only (clearance 0); the routing domain limit (lon -65 / lat 10-55) is drawn apart, not as land.
 ax.contourf(lo,la,(p.clearance(la,lo)<=0).astype(float),levels=[.5,1.5],colors=['#c9c2a7'])
 ax.axvspan(p.LON_MIN-6,p.LON_MIN,color='#eeeeee',hatch='//',alpha=.6,lw=0)
 col={0:'#1f5fbf',1:'#d9480f'}
 wm=max(wts) if wts is not None else 1.
 for i,(r,k) in enumerate(zip(rows,lab)):
  P=np.array([(x[1],x[2]) for x in r['path']])
  if wts is None:ax.plot(P[:,1],P[:,0],color=col[int(k)],lw=.9,alpha=.55)
  else:ax.plot(P[:,1],P[:,0],color='#1f5fbf',lw=.4+2.6*wts[i]/wm,alpha=.2+.6*wts[i]/wm)
 if gate:
  ax.plot([gate['lon']]*2,[10,gate['lat']],color='k',lw=1,ls=':');ax.plot([gate['lon']]*2,[gate['lat'],55],color='k',lw=1,ls=':')
  ax.plot(gate['lon'],gate['lat'],'kD',ms=6);ax.annotate(f" porte {gate['lat']:.1f}°N {abs(gate['lon']):.0f}°W",(gate['lon'],gate['lat']),fontsize=8)
 if hedge:
  H=np.array([(x[1],x[2]) for x in hedge]);ax.plot(H[:,1],H[:,0],'k-',lw=2.4,label=hedge_label)
  if tlabel: # passage times along the expected route, every 12 h
   L0_=hedge[0][0]
   for x in hedge:
    if x[0]>L0_+1 and abs((x[0]-L0_)%12)<1/12+1e-6 or abs((x[0]-L0_)%12-12)<1/12:
     ax.plot(x[2],x[1],'ko',ms=4);ax.annotate(tlabel(x[0]),(x[2],x[1]),xytext=(-62,-3),textcoords='offset points',fontsize=7.5,bbox=dict(fc='white',ec='none',alpha=.75,pad=1))
 if track: # the programmed orders, simulated in the game wind (GFS), with the order times
  P=np.array([(x[0],x[1]) for x in track if not x[2]]);ax.plot(P[:,1],P[:,0],'-',color='#c2187a',lw=2.6,label='ton programme (vent GFS)')
  for x in track:
   if x[2]:
    k=sum(1 for y in track if y[2] and y is not x and track.index(y)<track.index(x)) # alternate label sides
    ax.plot(x[1],x[0],'o',color='#c2187a',ms=5);ax.annotate(x[2],(x[1],x[0]),xytext=(8,8) if k%2==0 else (8,-14),textcoords='offset points',fontsize=8,color='#c2187a',bbox=dict(fc='white',ec='none',alpha=.7,pad=1))
 ax.plot(start[1],start[0],'*',color='gold',mec='k',ms=16,label='toi')
 for n,a,o in marks:ax.plot(o,a,'o',color='#444',ms=5);ax.annotate(' '+n,(o,a),fontsize=8)
 if wts is None:ax.plot([],[],color=col[0],label='famille nord (route optimale du scénario)');ax.plot([],[],color=col[1],label='famille sud')
 else:
  ax.plot([],[],color='#1f5fbf',lw=2.6,alpha=.8,label='route nord d\'un scénario crédible (poids fort)');ax.plot([],[],color='#1f5fbf',lw=.5,alpha=.3,label='scénario peu crédible (poids faible)')
  ax.text(.1,.99,"Filtre bayésien : chaque scénario météo GEFS est pondéré selon son accord\navec le vent GFS que le jeu va appliquer. Trait épais = scénario crédible.\nNoir = route nord moyenne pondérée (heures de passage), magenta = ton programme.",
   transform=ax.transAxes,va='top',fontsize=7.5,bbox=dict(fc='white',ec='#999',alpha=.85))
 if hasattr(p,'MARK_NW'): # real finish: NW Basse-Terre mark passed to the west, forbidden zone, line
  Z=np.vstack([p.ZONE,p.ZONE[:1]]);ax.fill(Z[:,1],Z[:,0],color='#d33',alpha=.35,lw=0,label='zone interdite')
  ax.plot(p.MARK_NW[1],p.MARK_NW[0],'^',color='orange',mec='k',ms=8,label='bouée NO Basse-Terre (à laisser à l\'est)')
  ax.plot([-61.88,-62.8],[16.25,16.25],'-',color='green',lw=3,label='ligne de passage du calcul (ouest de la bouée, libre)')
  ax.plot(p.FINISH[1],p.FINISH[0],'s',color='k',ms=6,label='ligne d\'arrivée (~2,5 h après le passage)')
 else:ax.plot(float(p.GOAL[1]),float(p.GOAL[0]),'P',color='green',ms=10,label="arrivée (point d'approche)")
 ax.set_xlim(p.LON_MIN-1,min(-25,start[1]+5));ax.set_ylim(14,min(45,start[0]+3)) # zoomed on what is left of the course
 ax.text(p.LON_MIN-.5,ax.get_ylim()[1]-.3,'hors zone de calcul',fontsize=7,color='#666',rotation=90,va='top',ha='center',clip_on=True);ax.set_xlabel('longitude');ax.set_ylabel('latitude')
 ax.grid(alpha=.3);ax.legend(loc='lower right',fontsize=7.5);ax.set_title(title,fontsize=10)
 zoom(az,rows,wts,hedge,tlabel);fig.tight_layout();fig.savefig(path_png);plt.close(fig)
def zoom(az,rows,wts,hedge,tlabel):
 """Right panel: the end of the course (Antigua to the finish line), real coast, forbidden zone, mark, finish legs."""
 az.set_facecolor('#dbe9f5')
 for g in getattr(p.LAND,'geoms',[p.LAND]):
  if g.bounds[2]>-63.5 and g.bounds[0]<-60.8 and g.bounds[3]>15.5 and g.bounds[1]<17.5:
   x,y=g.exterior.xy;az.fill(x,y,color='#c9c2a7',ec='#8a8166',lw=.5)
 wm=max(wts) if wts is not None else 1.
 for i,r in enumerate(rows):
  P=np.array([(x[1],x[2]) for x in r['path']]);a=wts[i]/wm if wts is not None else .5
  az.plot(P[:,1],P[:,0],color='#1f5fbf',lw=.4+2.*a,alpha=.2+.6*a)
 if hasattr(p,'MARK_NW'):
  Z=np.vstack([p.ZONE,p.ZONE[:1]]);az.fill(Z[:,1],Z[:,0],color='#d33',alpha=.35,lw=0)
  az.plot([-61.88,-62.8],[16.25,16.25],'-',color='green',lw=2.5,alpha=.7)
  az.plot(p.MARK_NW[1],p.MARK_NW[0],'^',color='orange',mec='k',ms=9);az.annotate(' bouée',(p.MARK_NW[1],p.MARK_NW[0]),fontsize=8)
  az.plot(p.FINISH[1],p.FINISH[0],'s',color='k',ms=7);az.annotate(' arrivée',(p.FINISH[1],p.FINISH[0]),fontsize=8)
 if hedge:
  H=np.array([(x[1],x[2]) for x in hedge]);az.plot(H[:,1],H[:,0],'k-',lw=2.2)
  if tlabel and len(hedge)>1:
   x=hedge[-1];az.annotate('arrivée attendue\n'+tlabel(x[0])+' Paris',(x[2],x[1]),xytext=(12,-30),textcoords='offset points',fontsize=8,
    arrowprops=dict(arrowstyle='-',color='k',lw=.6),bbox=dict(fc='white',ec='#999',alpha=.85,pad=2))
 az.set_xlim(-62.9,-61.0);az.set_ylim(15.75,17.3);az.set_aspect(1/np.cos(np.radians(16.5)))
 az.grid(alpha=.3);az.set_title("Zoom arrivée : bouée par l'ouest, tour du sud de Basse-Terre",fontsize=9);az.tick_params(labelsize=7)
def main():
 a=sys.argv[1:];cyc,obs=a[0],a[1];lat,lon=float(a[2]),float(a[3]);sail,tack=int(a[4]),int(a[5]);build='--build' in a
 prog=[float(x) for x in a[a.index('--program')+1:a.index('--program')+6]] if '--program' in a else None
 ens=R/'data'/f'ens_{cyc}';names=['gfs']+[f.name[:-len('3h.npz')] for f in sorted(ens.glob('gefs*3h.npz'))];mem=names[1:]
 fast=os.environ.get('STRAT_FAST')=='1' # family decided, fast mail: map from the gated tables only (no free member tables yet)
 G=dict(np.load(ens/'bellman_fine.npz'));GM=None if fast else bf.load_members(ens,names,G)[1:];WF=game_wind.fields(ens,names,cyc);Wg,Wm=WF[0],WF[1:]
 w=np.asarray(game_wind.scenario_weights(ens,names,cyc)[0],float);w=w/w.sum()
 c0=datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC);t0=datetime.datetime.strptime(obs,'%Y%m%d%H%M').replace(tzinfo=UTC)
 lead=(t0-c0).total_seconds()/3600;loc=lambda L:(c0+datetime.timedelta(hours=L)).astimezone(PARIS).strftime('%d/%m %Hh%M')
 # 1. Optimal route of every scenario, and of the GFS (the fleet).
 one=lambda T:(lambda L,la_,lo_,s_,t_:bf.value(T,L,la_,lo_,s_,t_))
 thr=.05 if (ens/'finish_local').exists() else 1.5 # real finish: stop on the passage line, then the fine finish leg
 stop=lambda T:(lambda L,la_,lo_,s_,t_:near_line(la_,lo_) or float(bf.value(T,L,np.array([la_]),np.array([lo_]),np.array([s_]),np.array([t_]))[0])<=thr)
 rows=[]
 for j,n in enumerate([] if fast else mem):
  P=descend(one(GM[j]),Wm[j],lead,lat,lon,sail,tack,150,stop(GM[j]))
  rows.append(dict(member=n,eta=P[-1][0]-lead+float(bf.value(GM[j],P[-1][0],np.array([P[-1][1]]),np.array([P[-1][2]]),np.array([P[-1][3]]),np.array([P[-1][4]]))[0]),
   cross=[crossing(P,m) for m in MERIDIANS],path=P[::3]+[P[-1]]))
 Pg=None if fast else descend(one(G),Wg,lead,lat,lon,sail,tack,150,stop(G));fleet_cross=None if fast else [crossing(Pg,m) for m in MERIDIANS]
 lab,gate,cent=(np.zeros(len(mem),int),None,None) if fast else families(rows);lines=[];out=dict(cycle=cyc,obs=obs,start=[lat,lon],weights=w.tolist(),rows=rows,gate=gate)
 if gate is None and not fast:
  lines.append(f"Options stratégiques : une seule famille de routes dans tous les scénarios, pas de choix nord/sud à faire.")
 # Gated tables: those of this cycle when their gate is close to today's (same members), else built (--build).
 g_old,T=load_gates(ens,mem)
 if gate and (T is None or abs(g_old['lon']-gate['lon'])>.1 or abs(g_old['lat']-gate['lat'])>1.5) and build:
  import shutil;shutil.rmtree(ens/'bellman_gates',ignore_errors=True);build_gates(ens,mem,gate,int(os.environ.get('PARALLEL','14')),os.environ.get('GATE_SIDES','NS'));g_old,T=load_gates(ens,mem)
 if gate is None and T is not None:gate=g_old
 if gate is not None and T is None:
  share=float(w[lab==0].sum())
  lines.append(f"Options stratégiques : nord (passage {gate['lat']:.0f}°N+ à {abs(gate['lon']):.0f}°W) optimal dans {share:.0%} des scénarios, sud dans {1-share:.0%} ; coût de chaque choix au prochain calcul complet.")
 if gate is not None and T is not None and len(T)<2:
  gate={k:g_old[k] for k in ('lon','lat','half')};hedge=None
  lines.append("Comparaison nord/sud non recalculée : la route est actée et l'autre option est fermée (tables de la famille retenue seules, calcul plus rapide).")
 elif gate is not None and T is not None:
  gate={k:g_old[k] for k in ('lon','lat','half')};side=lambda x:0 if x is not None and x>=gate['lat'] else 1
  gi=MERIDIANS.index(gate['lon']) # a route that does not reach the gate meridian keeps its 2-means family
  lab=np.array([side(r['cross'][gi]) if r['cross'][gi] is not None else int(k) for r,k in zip(rows,lab)]);ff=side(fleet_cross[gi]) if fleet_cross[gi] is not None else 0
  # 2. Loss matrix at the boat.
  V=np.array([Vg(T[s],lead,lat,lon,sail,tack)[:,0] for s in 'NS']);best=V.min(0);L=V-best
  E=V@w;fs=int(np.argmin(E));Pbest=[float(w[V.argmin(0)==k].sum()) for k in (0,1)]
  q95=[float(np.quantile(L[k],.95)) for k in (0,1)];nm=['nord','sud']
  # 3. Value of waiting and point of no return along the waiting route (descent of E[min_f V_f]).
  VOI=float(E.min()-best@w)
  qopen=lambda L_,la_,lo_,s_,t_:w@np.minimum(Vg(T['N'],L_,la_,lo_,s_,t_),Vg(T['S'],L_,la_,lo_,s_,t_))
  hedge=descend(qopen,Wg,lead,lat,lon,sail,tack,HMAX)
  dl={}
  for k,sd in enumerate('NS'):
   for x in hedge:
    if (w@Vg(T[sd],x[0],x[1],x[2],x[3],x[4])[:,0])+(x[0]-lead)-E[k]>TOL:dl[k]=x[0];break
  dead=min(dl.values()) if dl else None
  # Next full results (bulletin + ~6.5 h) before the point of no return.
  nxt=[c0+datetime.timedelta(hours=h) for h in range(6,7*24,6)];nxt=[c for c in nxt if c+datetime.timedelta(hours=6.5)>t0]
  before=[c for c in nxt if dead is not None and (c-c0).total_seconds()/3600+6.5<dead][:3]
  # 4. Does the chosen program close a family?
  closes=[]
  if prog:
   la1,lo1,s1,t1,H=prog
   # Compared with the waiting route at the same time (same tables, same grid): the program closes family f when it
   # costs f more than TOL beyond what keeping both options open costs it (the absolute D_f carries grid bias).
   xh=min(hedge,key=lambda x:abs(x[0]-(lead+H)))
   for k,sd in enumerate('NS'):
    Dk=float(w@Vg(T[sd],lead+H,la1,lo1,s1,t1)[:,0])-float(w@Vg(T[sd],xh[0],xh[1],xh[2],xh[3],xh[4])[:,0])-(xh[0]-lead-H)
    if Dk>TOL:closes.append((k,Dk))
  # 5. Rank: fleet in family ff, scenario by scenario.
  gain=[V[ff]-V[k] for k in (0,1)];pgain=[float(w@(g>=GOAL)) for g in gain]
  out.update(gate=gate,V=V.tolist(),expected=E.tolist(),choice=fs,p_best=Pbest,loss_q95=q95,value_of_waiting=VOI,
   deadline_lead={str(k):v for k,v in dl.items()},fleet_family=ff,p_gain_fleet=pgain,hedge=hedge,closes=closes)
  g_txt=f"porte {gate['lat']:.1f}°N à {abs(gate['lon']):.0f}°W"
  verdict=f"**{nm[fs]}** recommandé" if abs(E[0]-E[1])>=TOL else "**pas encore de choix net**, garder les deux options ouvertes"
  lines.append(f"**Choix stratégique** ({g_txt}, {len(mem)} scénarios) : {verdict}. "
   f"{nm[fs].capitalize()} plus rapide de {abs(E[0]-E[1]):.1f} h en moyenne (nord = passage de la porte au-dessus de {gate['lat']:.0f}°N, sud = en dessous) ; "
   f"nord meilleur dans {Pbest[0]:.0%} des scénarios, sud dans {Pbest[1]:.0%}.")
  lines.append(f"Risque du choix : si {nm[fs]} se révèle mauvais, perte {L[fs][L[fs]>0].mean() if (L[fs]>0).any() else 0:.1f} h en moyenne dans ces scénarios (pire 5 % : {q95[fs]:.1f} h) ; "
   f"choisir {nm[1-fs]} perdrait jusqu'à {q95[1-fs]:.1f} h.")
  if dead is None:lines.append(f"Point de non-retour : les deux options restent ouvertes au-delà de {HMAX} h ; valeur d'attendre la météo exacte ≤ {VOI:.2f} h.")
  else:
   b=', '.join(f"{c:%H}Z (vers {(c+datetime.timedelta(hours=6.5)).astimezone(PARIS):%Hh%M})" for c in before) or 'aucun'
   lines.append(f"Point de non-retour : décision à prendre avant le {loc(dead)} (au-delà, l'option {nm[min(dl,key=dl.get)]} coûte >{TOL:.1f} h). "
    f"Valeur d'attendre la météo exacte ≤ {VOI:.2f} h ; bulletins disponibles avant : {b}.")
  if prog:lines.append("Programme choisi : garde les deux options ouvertes." if not closes else
   "Programme choisi : "+' ; '.join(f"ferme l'option {nm[k]} (coût +{d:.1f} h si elle s'avère la bonne)" if d<50 else f"engage le passage {nm[1-k]} (l'option {nm[k]} n'est plus jouable après le programme)" for k,d in closes)+'.')
  lines.append(f"Classement : la flotte (routeur GFS) part {nm[ff]} ; choisir {nm[1-ff]} donne {pgain[1-ff]:.0%} de chances de lui prendre ≥{GOAL:.1f} h "
   f"(et {float(w@(gain[1-ff]<=-GOAL)):.0%} d'en perdre autant) ; choisir {nm[ff]} = rester avec elle.")
 else:hedge=None
 S=json.loads((R/'stochastic'/'auto'/'state.json').read_text());marks=[(m[0],m[1],m[2]) for m in S.get('marks',[])]
 hl="route d'attente (options ouvertes)";fam=S.get('family')
 if fam in ('N','S') and T is not None and gate is not None: # family decided by the skipper: show its expected route
  hedge=fine_leg(ens,'gfs',Wg,descend(lambda L_,la_,lo_,s_,t_:w@Vg(T[fam],L_,la_,lo_,s_,t_),Wg,lead,lat,lon,sail,tack,150,
   lambda L_,la_,lo_,s_,t_:near_line(la_,lo_) or float(w@Vg(T[fam],L_,np.array([la_]),np.array([lo_]),np.array([s_]),np.array([t_]))[:,0])<=thr))
  tl=loc
  hl=f"route {'nord' if fam=='N' else 'sud'} attendue (moyenne pondérée des scénarios)"
  # The map shows the decided family only: each scenario's route inside it (gated tables), width = Bayesian weight.
  rows=[];lab=np.zeros(len(mem),int);fw=w
  for j,n in enumerate(mem):rows.append(dict(member=n,path=sub(fine_leg(ens,n,Wm[j],descend(one(T[fam][j]),Wm[j],lead,lat,lon,sail,tack,150,stop(T[fam][j]))))))
  lines.insert(0,f"**Route {'nord' if fam=='N' else 'sud'} actée** : programme et carte optimisés pour passer {'au-dessus' if fam=='N' else 'en dessous'} de {gate['lat']:.1f}°N à {abs(gate['lon']):.0f}°W.")
 track=json.loads(os.environ['STRAT_TRACK']) if os.environ.get('STRAT_TRACK') else None
 fw=locals().get('fw');tl=locals().get('tl')
 png=R/'stochastic'/'auto'/f"strategy_{cyc}{os.environ.get('STRAT_SFX','')}.png"
 ttl=(f"Route nord : routes des {len(mem)} scénarios pondérés — bulletin {cyc[8:]}Z, {loc(lead)} Paris" if fw is not None else f"Routes optimales des {len(mem)} scénarios — bulletin {cyc[8:]}Z, {loc(lead)} Paris")
 draw(png,rows,lab,gate,hedge,(lat,lon),marks,ttl,hl,track,fw,tl)
 (R/'stochastic'/'auto'/f"strategy_{cyc}{os.environ.get('STRAT_SFX','')}.json").write_text(json.dumps(out,default=float))
 for l in lines:print('STRATEGY',l,flush=True)
 print('STRATEGY_MAP',png,flush=True)
if __name__=='__main__':main()
