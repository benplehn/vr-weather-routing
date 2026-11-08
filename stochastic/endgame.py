"""End-of-race stochastic optimal control by direct simulation (7/11, skipper: "clear and correct, no assumption that
spoils the result, no north/south, but stochastic").

Problem. Choose a program u = (current angle/sail, then up to K orders (time, signed TWA, sail)) entered in the game,
held after its last order, minimizing the EXPECTED time to the finish over the scenarios, subject to: no scenario
touches land (or the forbidden zone). Each program is sailed to the finish line in every scenario; there is no value
table, no route family, no horizon artifact (the table values near the islands, 0.25 deg cells, read 300-1000 h next to
Antigua and drove the optimizer east of it on 7/11).

Model (all from optimal_route / game_wind, nothing added):
 - wind: game_wind.fields: what the game blows (previous GFS until it switches, then this GFS, then each GEFS member
   after the next bulletin): 32 scenarios (GFS + 31 GEFS), equal weights;
 - speed: p.polar(TWA, TWS, sail); sails after the pack: Jib (0) and Spi (1);
 - tack / gybe / sail change: p.maneuver (energy spent, penalty time at PEN_LOSS of the speed, longer with low energy),
   energy recovered at p.recover_seconds;
 - land: p.allowed + EG_CLEARANCE NM (3) from the coast, forbidden zone, at every step and mid-step; one scenario
   closer than that rejects the program;
 - finish: crossing 16.25 N between 62.8 W and 61.88 W (west of the NW Basse-Terre mark), plus the common rest of the
   course FINISH_TAIL_H and the extra distance of a wide crossing at TAIL_KN (p.goal_target).
Open loop over the whole remaining race (the program does not use information the skipper will only get later): a
conservative bound, re-planned at every bulletin and every capture (receding horizon).

Search: structured enumeration (alternating tacks, order times on a grid, angles from a set), then coordinate descent
on every order time (+-10..60 min), angle (+-1..6 deg) and sail, on all scenarios. The program entered in the game is
always evaluated with the same code (gain, land check).

Usage: endgame.py CYCLE OBS_UTC(YYYYmmddHHMM) LAT LON ENERGY ANGLE SAIL [PROGRAM_JSON] -> prints RESULT {json}
PROGRAM_JSON: entered orders [[minutes_from_obs, angle], ...] (sail kept).
"""
import os,sys,json,datetime,itertools
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'stochastic'),str(R)]
os.environ['FULL_PACK']='1';os.environ.setdefault('ROUTING_SCENARIO',str(R/'data'/'morning_00z'/'scenario.json'))
import optimal_route as p,game_wind
UTC=datetime.timezone.utc
SAILS=(0,1) # pack sails ended 3/11
LINE_LAT,LINE_W,LINE_E=16.25,-62.8,-61.88
TMAX=float(os.environ.get('EG_TMAX','30'))
# Land clearance (NM) for the endgame, above p.allowed's 2 NM: the game's projection and ours differ by 1-3 NM after a
# few hours (7/11: our 22:50 tack cleared Montserrat in the GFS column, the game showed a collision on its south coast).
CLR=float(os.environ.get('EG_CLEARANCE','3'))
def clear(lat,lon):return p.allowed(lat,lon)&(p.clearance(lat,lon)>CLR)
D0=float(p.distance(np.array([LINE_LAT]),np.array([LINE_E]),p.TAIL_WP)[0])

def load(cyc):
 ens=R/'data'/f'ens_{cyc}'
 names=['gfs']+[f.name[:-len('3h.npz')] for f in sorted(ens.glob('gefs*3h.npz'))]
 return names,game_wind.fields(ens,names,cyc)

def simulate(W,L0,lat,lon,energy,sail0,tack0,times,angles,sails,dt=1/12,track=False):
 """Programs (n): times (n,K) hours from start (inf = unused), angles (n,K+1) signed TWA, sails (n,K+1).
 Returns per scenario and program: finish hours (inf on land), land flag, min energy; and tracks if asked."""
 n,K1=angles.shape;S=len(W)
 fin=np.full((S,n),np.inf);land=np.zeros((S,n),bool);emin=np.full((S,n),100.);trk=[]
 for j,F in enumerate(W):
  la=np.full(n,float(lat));lo=np.full(n,float(lon));e=np.full(n,float(energy));pen=np.zeros(n)
  sa=np.full(n,sail0);ta=np.full(n,tack0);alive=np.ones(n,bool);T=0.;tj=[]
  while T<TMAX and alive.any():
   i=np.where(alive)[0];k=(times[i]<=T+1e-9).sum(1);a=angles[i,k];s=sails[i,k];nt=np.where(a>0,1,-1)
   ws,wd=F(L0+T,la[i],lo[i])
   cs=s!=sa[i];ct=nt!=ta[i]
   used,ph=p.maneuver(ws,e[i],cs,ct);e[i]=np.maximum(0,e[i]-used);pen[i]+=ph
   slow=np.minimum(dt,pen[i]);v=p.polar(a,ws,s)
   ln,on=p.move(la[i],lo[i],(wd-a)%360,v*(dt-p.PEN_LOSS*slow))
   ok=np.asarray(clear(ln,on)&clear((la[i]+ln)/2,(lo[i]+on)/2),bool)
   bad=i[~ok];land[j,bad]=True;alive[bad]=False
   cr=ok&(la[i]>=LINE_LAT)&(ln<LINE_LAT)
   f=(la[i]-LINE_LAT)/np.maximum(la[i]-ln,1e-9);xl=lo[i]+f*(on-lo[i])
   inl=cr&(xl>=LINE_W)&(xl<=LINE_E)
   g=i[inl]
   if len(g):
    extra=np.maximum(0,p.distance(np.full(len(g),LINE_LAT),xl[inl],p.TAIL_WP)-D0)/p.TAIL_KN
    fin[j,g]=T+f[inl]*dt+p.FINISH_TAIL_H+extra;alive[g]=False
   e[i]=np.minimum(100,e[i]+(dt-slow)*3600/p.recover_seconds(ws));pen[i]=np.maximum(0,pen[i]-dt)
   emin[j,i]=np.minimum(emin[j,i],e[i])
   la[i],lo[i],sa[i],ta[i]=ln,on,s,nt
   if track:tj.append((T,la.copy(),lo.copy()))
   T+=dt
  # not finished within TMAX: TMAX plus the straight-line rest at 10 kn (never better than finishing)
  o=np.isinf(fin[j])&~land[j];fin[j,o]=TMAX+p.distance(la[o],lo[o],p.MARK_NW)/10+p.FINISH_TAIL_H
  trk.append(tj)
 return (fin,land,emin,trk) if track else (fin,land,emin)

def score(fin,land):
 """Expected time (equal weights); a program on land in any scenario is infeasible (inf)."""
 m=fin.mean(0);m[land.any(0)]=np.inf;return m

def best_sail(W,L0,lat,lon,a):
 ws,_=W[0](L0,np.array([lat]),np.array([lon]))
 return max(SAILS,key=lambda s:float(p.polar(np.array([abs(a)]),ws,np.array([s]))[0]))

def enumerate_programs(a0,s0,H=14.,K=3):
 """Alternating-tack programs: hold a0 until t1, then (-a2), (a3), (-a4)... sails: Jib below 110 deg, Spi above (the
 refinement tries the other sail on every order). Plus 'hold a0' and same-tack angle changes."""
 sg=1 if a0>0 else -1
 T1=list(np.arange(0,H/2+1e-9,1/3));D=[.5,1,1.5,2,3,4,5,6];AN=[45,50,55,60,70,85,100,120,140,150]
 sail=lambda a:0 if abs(a)<110 else 1
 rows=[([],[a0],[s0])]
 for t1 in T1:
  for b in AN:rows.append(([t1],[a0,sg*b],[s0,sail(b)]))
  for b in AN:rows.append(([t1],[a0,-sg*b],[s0,sail(b)]))
 for t1,d2 in itertools.product(T1,D):
  for b,c in itertools.product(AN[:6],AN[:6]):rows.append(([t1,t1+d2],[a0,-sg*b,sg*c],[s0,0,0]))
 if K>=3:
  for t1,d2,d3 in itertools.product(T1[::2],D[:6],D):
   for b,c,d in itertools.product((45,50,55,60),(45,50,55,60),(45,50,55,60)):
    rows.append(([t1,t1+d2,t1+d2+d3],[a0,-sg*b,sg*c,-sg*d],[s0,0,0,0]))
 return rows

def pack(rows,K):
 n=len(rows);Ti=np.full((n,K),np.inf);An=np.zeros((n,K+1));Sa=np.zeros((n,K+1),int)
 for r,(t,a,s) in enumerate(rows):
  m=len(t);Ti[r,:m]=t;An[r,:m+1]=a;An[r,m+1:]=a[-1];Sa[r,:m+1]=s;Sa[r,m+1:]=s[-1]
 return Ti,An,Sa

def evaluate(W,L0,st,rows,K,dt=1/12,batch=4000):
 out_f=[];out_l=[];out_e=[]
 for b in range(0,len(rows),batch):
  Ti,An,Sa=pack(rows[b:b+batch],K);f,l,e=simulate(W,L0,*st,Ti,An,Sa,dt=dt);out_f.append(f);out_l.append(l);out_e.append(e)
 return np.concatenate(out_f,1),np.concatenate(out_l,1),np.concatenate(out_e,1)

def neighbours(row):
 t,a,s=row;out=[]
 for k in range(len(t)):
  for d in (-1,-2/3,-1/3,-1/6,1/6,1/3,2/3,1):
   nt=list(t);nt[k]=t[k]+d
   if nt[k]>=0 and all(x<y for x,y in zip(nt,nt[1:])):out.append((nt,a,s))
 for k in range(len(a)):
  for d in (-6,-3,-1,1,3,6):
   na=list(a);na[k]=a[k]+np.sign(a[k])*d
   if 35<abs(na[k])<175:out.append((t,na,s))
  if k>0:
   ns=list(s);ns[k]=1-s[k];out.append((t,a,ns))
 if t:out.append((t[:-1],a[:-1],s[:-1])) # drop the last order
 return out

ROB=float(os.environ.get('EG_ROBUST_MIN','10'))/60
def shifted(row):
 t,a,s=row;out=[]
 for k in range(len(t)):
  for d in (-ROB,ROB):
   nt=list(t);nt[k]=max(0.,t[k]+d)
   if all(x<y for x,y in zip(nt,nt[1:])):out.append((nt,a,s))
 return out
def robust_eval(W,L0,st,rows,K,dt=1/12):
 """Expected time of each program; inf when the program, or one of its orders shifted alone by +-EG_ROBUST_MIN (10)
 minutes, comes within EG_CLEARANCE of land in any scenario (7/11: the optimum threaded the islands so closely that a
 10 min shift of any order touched land: not a program a skipper can enter)."""
 f,l,e=evaluate(W,L0,st,rows,K,dt);J=score(f,l)
 ok=np.where(np.isfinite(J))[0];ext=[];own=[]
 for i in ok:
  for q in shifted(rows[i]):ext.append(q);own.append(i)
 if ext:
  f2,l2,_=evaluate(W,L0,st,ext,K,dt);bad=l2.any(0)
  for i,b in zip(own,bad):
   if b:J[i]=np.inf
 return J,f,l,e
def key(r):return (tuple(round(x*60) for x in r[0]),tuple(round(x) for x in r[1]),tuple(r[2]))

def optimize(W,L0,st,a0,s0,cur=None,K=3,iters=6,log=print):
 rows=enumerate_programs(a0,s0,K=K)
 if cur:rows.append(cur)
 f,l,_=evaluate(W,L0,st,rows,K);J=score(f,l);log(f'coarse: {len(rows)} programs x {len(W)} scenarios, best {np.nanmin(J):.2f} h')
 seen={key(r):J[i] for i,r in enumerate(rows)}
 top=[rows[i] for i in np.argsort(J)[:300] if np.isfinite(J[i])] # robust check of the best 300 coarse programs
 Jr,_,_,_=robust_eval(W,L0,st,top,K);rob=set()
 for r,j in zip(top,Jr):
  seen[key(r)]=j
  if np.isfinite(j):rob.add(key(r))
 elite=[top[i] for i in np.argsort(Jr)[:12] if np.isfinite(Jr[i])];log(f'robust: {len(rob)}/{len(top)}, best {np.min(Jr):.2f} h')
 for it in range(iters):
  cand=[];
  for r in elite:
   for q in neighbours(r):
    if key(q) not in seen:cand.append(q);seen[key(q)]=None
  if not cand:break
  Jc,_,_,_=robust_eval(W,L0,st,cand,K)
  for q,j in zip(cand,Jc):
   seen[key(q)]=j
   if np.isfinite(j):rob.add(key(q))
  pool=sorted(((v,k) for k,v in seen.items() if v is not None and np.isfinite(v) and k in rob))[:12]
  bymap={key(r):r for r in elite+cand};elite=[bymap[k] for v,k in pool if k in bymap]
  log(f'refine {it+1}: {len(cand)} new, best {pool[0][0]:.2f} h')
 return elite

def margins(W,L0,st,row,K,base):
 """Per order: shift alone by -60..+60 min; window where the loss stays under 0.1 h and land is never touched."""
 t,a,s=row;res=[]
 for k in range(len(t)):
  sh=[-60,-40,-20,-10,10,20,40,60];rows=[]
  for m in sh:
   nt=list(t);nt[k]=t[k]+m/60
   rows.append((nt,a,s) if nt[k]>=0 and all(x<y for x,y in zip(nt,nt[1:])) else (t,a,s))
  f,l,_=evaluate(W,L0,st,rows,K,dt=1/30);J=score(f,l);d=J-base
  ok={m for m,x in zip(sh,d) if x<=.1} # inf (land) is never ok
  wl=wh=0
  for m in (-10,-20,-40,-60):
   if m not in ok:break
   wl=m
  for m in (10,20,40,60):
   if m not in ok:break
   wh=m
  res.append(dict(order=k,window=[wl,wh],land=[m for m,x in zip(sh,l.any(0)) if x],loss={m:(None if not np.isfinite(x) else round(float(x),2)) for m,x in zip(sh,d)}))
 return res

def track_map(path,tr,trc,t0,lat,lon):
 """PNG: islands, chosen program (GFS column, hourly labels Paris time), entered program, finish line."""
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from shapely.geometry import box
 la=[x[1] for x in tr+trc]+[lat,LINE_LAT];lo=[x[2] for x in tr+trc]+[lon,LINE_W,LINE_E]
 bb=(min(lo)-.3,min(la)-.3,max(lo)+.3,max(la)+.3);fig,ax=plt.subplots(figsize=(8,8))
 g=p.LAND.intersection(box(*bb))
 for poly in getattr(g,'geoms',[g]):
  if hasattr(poly,'exterior'):x,y=poly.exterior.xy;ax.fill(x,y,color='#8a8',lw=.5,ec='#454')
 ax.plot([LINE_W,LINE_E],[LINE_LAT]*2,'k--',lw=1);ax.plot(p.MARK_NW[1],p.MARK_NW[0],'r^')
 ax.plot([x[2] for x in trc],[x[1] for x in trc],color='#999',lw=1.5,label='programme entré')
 ax.plot([x[2] for x in tr],[x[1] for x in tr],color='#c0306a',lw=2.2,label='programme proposé')
 P=datetime.timezone(datetime.timedelta(hours=1))
 for T,y,x in tr:
  if abs(T-round(T))<.13:ax.annotate((t0+datetime.timedelta(hours=T)).astimezone(P).strftime('%Hh'),(x,y),fontsize=7,xytext=(3,3),textcoords='offset points')
 ax.plot(lon,lat,'ko');ax.set_xlim(bb[0],bb[2]);ax.set_ylim(bb[1],bb[3]);ax.set_aspect(1/np.cos(np.radians(np.mean(bb[1::2]))))
 ax.legend(loc='upper right');ax.set_title('Fin de course : route du programme (vent GFS du jeu)');fig.savefig(path,dpi=110,bbox_inches='tight');plt.close(fig)

def main():
 cyc,obs,lat,lon,e,a0,s0=sys.argv[1:8];cur_orders=json.loads(sys.argv[8]) if len(sys.argv)>8 else []
 lat,lon,e,a0,s0=float(lat),float(lon),float(e),int(float(a0)),int(s0)
 t0=datetime.datetime.strptime(obs,'%Y%m%d%H%M').replace(tzinfo=UTC);c0=datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC)
 L0=(t0-c0).total_seconds()/3600;names,W=load(cyc);st=(lat,lon,e,s0,1 if a0>0 else -1);K=3
 log=lambda *x:print(*x,flush=True)
 cur=([m/60 for m,_ in cur_orders if m>0],[a0]+[a for m,a in cur_orders if m>0],[s0]*(1+len([m for m,_ in cur_orders if m>0])))
 Kc=max(K,len(cur[0]))
 elite=optimize(W,L0,st,a0,s0,cur if len(cur[0])<=K else None,K=Kc,log=log)
 # final evaluation at fine time step: elite + entered program
 rows=elite[:8]+[cur];J,f,l,emin=robust_eval(W,L0,st,rows,Kc,dt=1/30)
 if not np.isfinite(J[:-1]).any():J[:-1]=score(f,l)[:-1] # never empty-handed: fall back to nominal feasibility (flagged)
 i=int(np.argmin(J[:-1]));best=rows[i];fb=f[:,i]
 fc,lc=f[:,-1],l[:,-1]
 Ti,An,Sa=pack([best,cur],Kc);_,_,_,trk=simulate(W[:1],L0,*st,Ti,An,Sa,dt=1/12,track=True)
 tr=[(round(T,3),round(float(la[0]),4),round(float(lo[0]),4)) for T,la,lo in trk[0][::3]]
 trc=[(round(T,3),round(float(la[1]),4),round(float(lo[1]),4)) for T,la,lo in trk[0][::3]]
 # land on the entered program: first scenario/time it hits (GFS column first)
 cur_land=None
 if lc.any():
  j=int(np.argmax(lc));Ti2,An2,Sa2=pack([cur],Kc)
  _,_,_,tk=simulate([W[j]],L0,*st,Ti2,An2,Sa2,dt=1/12,track=True);Tl,la,lo=tk[0][-1];cur_land=dict(scenario=names[j],n=int(lc.sum()),hours=round(Tl,2),lat=round(float(la[0]),3),lon=round(float(lo[0]),3))
 mg=margins(W,L0,st,best,Kc,J[i])
 if os.environ.get('EG_MAP'):track_map(os.environ['EG_MAP'],tr,trc,t0,lat,lon)
 res=dict(cycle=cyc,obs=obs,n=len(W),names=names,best=dict(times=best[0],angles=[int(x) for x in best[1]],sails=[int(x) for x in best[2]]),
  expected=float(J[i]),p10=float(np.percentile(fb,10)),p90=float(np.percentile(fb,90)),land=int(l[:,i].sum()),min_energy=float(emin[:,i].min()),
  current=dict(times=cur[0],angles=cur[1],expected=float(fc.mean()) if not lc.any() else None,land=int(lc.sum()),land_at=cur_land,p90=float(np.percentile(fc,90))),
  better=int((fb<fc).sum()),margins=mg,track=tr,track_current=trc,
  alternatives=[dict(times=r[0],angles=[int(x) for x in r[1]],sails=[int(x) for x in r[2]],expected=float(J[k])) for k,r in enumerate(rows[:-1]) if k!=i][:4])
 print('RESULT '+json.dumps(res,default=float),flush=True)
if __name__=='__main__':main()
