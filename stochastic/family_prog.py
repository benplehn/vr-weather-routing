# Program optimized inside one route family (gated Bellman tables of strategy.py), stochastic over the GEFS scenarios.
# A program = up to 4 regulator orders (time, signed TWA; + = tribord), Spi. Each program is simulated in every scenario's
# game wind (10 min steps, gybe penalty as in the tables); cost_j = elapsed + V_family_j(end); minimizes the weighted mean
# (Bayesian scenario weights). Coarse grid (trib a1 -> gybe bab a2 -> gybe trib a3, 30 min) then local refinement
# (order times +-10/20 min, angles +-1/2/5 deg, one extra angle change), scenarios in parallel (fork Pool).
# Usage: family_prog.py CYCLE OBS LAT LON ANGLE FAMILY HOURS [REF "HHMM:ANGLE,..." (Paris times)]
# Out: lines PROG/REF/TRACK on stdout; TRACK = json [[lat,lon,label],...] of the best program in the GFS game wind.
import os,sys,datetime,json
from pathlib import Path
from multiprocessing import get_context
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'stochastic'))
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'));os.environ.setdefault('FULL_PACK','1')
import numpy as np
import strategy as st, bellman_fine as bf, game_wind, optimal_route as p
cyc,obs,lat,lon,ang0,fam,HRS=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),sys.argv[6],float(sys.argv[7])
ens=R/'data'/f'ens_{cyc}';names=['gfs']+[f.name[:-6] for f in sorted(ens.glob('gefs*3h.npz'))];mem=names[1:]
WF=game_wind.fields(ens,names,cyc);w=np.asarray(game_wind.scenario_weights(ens,names,cyc)[0],float);w/=w.sum()
g,T=st.load_gates(ens,mem);assert T is not None,'gated tables missing'
# Value beyond the horizon from two models when the free member tables exist (full run): the decided family's gated tables
# and the free tables (no wall). Their mean is model-robust: on 5/11 12Z they disagreed by ~5 min (keep 140 vs 145).
GM=None
if os.environ.get('FAMILY_FREE','1')=='1' and (ens/'bellman_fine.npz').exists():
 _ml=bf.load_members(ens,names,dict(np.load(ens/'bellman_fine.npz')));GM=_ml[1:] if _ml else None
 if GM and float(GM[0]['lon'][0])>float(T[fam][0]['lon'][0])+.1: # free tables on a narrower domain (before 6/11 west extension)
  print('free tables narrower than the gated ones: not used',flush=True);GM=None
print('MODEL',fam+(' + free (mean)' if GM is not None else ' only'),flush=True)
c0=datetime.datetime.strptime(cyc,'%Y%m%d%H');t0=datetime.datetime.strptime(obs,'%Y%m%d%H%M')
L0=(t0-c0).total_seconds()/3600;END=L0+HRS;DT=1/6;K=int(os.environ.get('FP_K','3'))
E0=float(os.environ.get('FP_ENERGY',bf.TYPICAL_ENERGY));REGEN_MIN=float(os.environ.get('FP_REGEN_MIN','14'));OTHER='S' if fam=='N' else 'N' # regen: 1 point per ~14 min (game)
paris=lambda L:(c0+datetime.timedelta(hours=L+1)).strftime('%H:%M')
def lead_of(hhmm):
 d=c0.replace(hour=int(hhmm[:2]),minute=int(hhmm[2:]))-datetime.timedelta(hours=1)
 L=(d-c0).total_seconds()/3600
 return L+24 if L<L0-1 else L
def pad(orders):
 o=sorted(orders)[:K];return o+[(99.,o[-1][1] if o else ang0)]*(K-len(o))
# END = when the skipper will next enter orders (HOURS, from auto_update.user_gap: ~3 h by day, 9:00 after a night). The
# programmed orders are all before END; after the last one the angle is held until END (what the game does), then the
# table values the position (optimal play from the next update on). Before 6/11 HOURS was a fixed 8-14 h: a held angle
# for many hours the skipper would never really hold made the answer depend on that number (8 h gybe, 12 h no gybe).
# FP_TAIL_MODE=policy: table value from last order + TAIL h instead (study only).
TAIL=float(os.environ.get('FP_TAIL','1.5'));MIN_H=float(os.environ.get('FP_MIN_H','3'))
def ends(Ot):
 if os.environ.get('FP_TAIL_MODE')!='policy':return np.full(len(Ot),END)
 last=np.where(Ot<END-1e-9,Ot,-np.inf).max(1);last=np.where(np.isfinite(last),last,L0)
 e=np.clip(last+TAIL,L0+MIN_H,END);return L0+np.ceil((e-L0)/DT-1e-6)*DT
def sim(Ot,Oa,W,Vt=None,track=False):
 N=len(Ot);la=np.full(N,lat);lo=np.full(N,lon);a=np.full(N,ang0);rem=np.zeros(N);L=L0;ok=np.ones(N,bool);tr=[]
 en=np.full(N,E0) # boat energy: gybe timer and cost from the real energy (6/11: at 50 a gybe lasts ~1.7x the 85 one)
 Ep=np.full(N,END) if track else ends(Ot);Lend=float(Ep.max())
 while L<Lend-1e-9:
  act=L<Ep-1e-9
  na=a.copy()
  for k in range(K):na=np.where(Ot[:,k]<=L+1e-9,Oa[:,k],na)
  ws,wd=W(L+DT/2,la,lo);gy=np.sign(na)!=np.sign(a)
  used,pen=p.maneuver(ws,en,np.zeros(N,bool),gy);rem=rem+np.where(gy,pen,0)
  en=np.clip(np.where(gy&act,en-used,en)+DT*60/REGEN_MIN,0,100)
  lost=.5*np.minimum(DT,rem);rem-=np.minimum(DT,rem)
  ln,on=p.move(la,lo,(wd-na)%360,p.polar(na,ws,np.ones(N))*(DT-lost))
  ok&=p.leg_allowed(la,lo,ln,on)|~act;la,lo,a=np.where(act,ln,la),np.where(act,on,lo),np.where(act,na,a);L+=DT
  if track:tr.append((float(la[0]),float(lo[0])))
 if track:return tr
 out=np.full(N,np.inf);t_=(a>0).astype(int)*2-1
 for e in np.unique(np.round(Ep,6)):
  m=np.abs(Ep-e)<1e-6;out[m]=(e-L0)+Vt(e,la[m],lo[m],t_[m])
 return np.where(ok,out,np.inf)
def _job(args):
 j,Ot,Oa,fam_=args
 V=lambda L,la,lo,t:bf.value(T[fam_][j],L,la,lo,np.ones(len(la),int),t)
 if GM is not None and fam_==fam:
  wf=float(os.environ.get('FAMILY_FREE_W','0.5')) # weight of the free 0.25 deg tables (gated ones: 0.5 deg, coarser)
  Vn=V;V=lambda L,la,lo,t:(1-wf)*Vn(L,la,lo,t)+wf*bf.value(GM[j],L,la,lo,np.ones(len(la),int),t)
 return sim(Ot,Oa,WF[1+j],V)
POOL=get_context('fork').Pool(int(os.environ.get('PARALLEL','16')))
def evaluate(progs,fam_=fam):
 O=[pad(x) for x in progs];Ot=np.array([[t for t,_ in o] for o in O]);Oa=np.array([[a for _,a in o] for o in O])
 C=np.array(POOL.map(_job,[(j,Ot,Oa,fam_) for j in range(len(mem))]));return C
def key(pr):return tuple((round(t,3),a) for t,a in sorted(pr))
A=[130,135,140,145,150];GY=np.arange(np.ceil((L0+.5)*2)/2,END-.4,.5);cand={}
# Signs relative to the current tack (s0): the first angle keeps the tack (no immediate gybe), then optional gybes.
s0=1. if ang0>0 else -1.
for a1 in A:
 base=[(L0,s0*a1)] if s0*a1!=ang0 else []
 cand[key(base)]=base
 for g1 in GY:
  for a2 in A:
   x=base+[(g1,s0*a2)];cand[key(x)]=x # angle change on the same tack
   x=base+[(g1,-s0*a2)];cand[key(x)]=x
   for g2 in GY[GY>=g1+1]:
    for a3 in A:x=base+[(g1,-s0*a2),(g2,s0*a3)];cand[key(x)]=x
progs=list(cand.values());print('coarse programs',len(progs),flush=True)
C=evaluate(progs);E=w@C;seen={key(x):(E[i],C[:,i]) for i,x in enumerate(progs)}
def neighbours(pr):
 out=[]
 for k,(t,a) in enumerate(pr):
  for dt in (-1/3,-1/6,1/6,1/3):
   if L0<=t+dt<END:q=list(pr);q[k]=(t+dt,a);out.append(q)
  for da in (-5,-2,-1,1,2,5):
   b=abs(a)+da
   if 120<=b<=160:q=list(pr);q[k]=(t,np.sign(a)*b);out.append(q)
 if len(pr)<K: # one extra angle change on the current tack
  for t in np.arange(np.ceil(L0*3)/3,END-.3,1/3):
   cur=ang0
   for tt,aa in sorted(pr):
    if tt<=t:cur=aa
   for da in (-5,5):
    b=abs(cur)+da
    if 120<=b<=160:out.append(list(pr)+[(t,np.sign(cur)*b)])
 return out
for rnd in range(4):
 top=sorted(seen.items(),key=lambda kv:kv[1][0])[:12];new={}
 for k_,_ in top:
  for q in neighbours([tuple(x) for x in k_]):
   kk=key(q)
   if kk not in seen and kk not in new:new[kk]=q
 if not new:break
 qs=list(new.values());Cq=evaluate(qs);Eq=w@Cq
 for i,q in enumerate(qs):seen[key(q)]=(Eq[i],Cq[:,i])
 print('refine',rnd,len(qs),'best',round(min(v[0] for v in seen.values()),3),flush=True)
rank=sorted(seen.items(),key=lambda kv:kv[1][0])
# No zigzag: among programs within ZZ_TOL h of the best, keep the one with the fewest gybes (then the fastest).
ngy=lambda k_:int(sum(np.sign(a)!=np.sign(b) for a,b in zip([ang0]+[x[1] for x in k_],[x[1] for x in k_])))
ZZ=float(os.environ.get('ZZ_TOL','0.1'));E0=rank[0][1][0]
rank=sorted(rank,key=lambda kv:(kv[1][0]>E0+ZZ,ngy(kv[0]) if kv[1][0]<=E0+ZZ else 0,kv[1][0]))
bk,(bE,bC)=rank[0]
def desc(k_):
 s=[f"maintenant {abs(ang0):.0f}° {'tribord' if ang0>0 else 'bâbord'}"]
 for t,a in k_:s.append(f"{paris(t)} {abs(a):.0f}° {'tribord' if a>0 else 'bâbord'}")
 return ' | '.join(s)
best=[list(x) for x in bk]
Co=evaluate([best],OTHER)[:,0] if OTHER in T else np.full(len(mem),np.inf)
print('FAMILY',fam,'delay',game_wind.DELAY,'start',paris(L0),'end',paris(END),flush=True)
print('PROG',json.dumps(dict(desc=desc(bk),orders=[(paris(t),float(a)) for t,a in bk],orders_utc=[((c0+datetime.timedelta(hours=t)).strftime('%Y%m%d%H%M'),float(a)) for t,a in bk],E=float(bE),other=float(w@Co))),flush=True)
for k_,(e,c) in rank[1:6]:print(f"ALT {e-bE:+.3f} h  better/equal in {int((c<=bC+.05).sum())}/{len(mem)}  {desc(k_)}",flush=True)
# Margins of each order (time shifted alone).
for k,(t,a) in enumerate(bk):
 sh=[];res=[]
 for dt in (-1,-.5,.5,1):
  q=[list(x) for x in bk];q[k][0]=t+dt
  if L0<=q[k][0]<END:sh.append(dt);res.append(q)
 Cm=evaluate(res);print('MARGIN',paris(t),' '.join(f"{dt:+.1f}h:{w@Cm[:,i]-bE:+.2f}" for i,dt in enumerate(sh)),flush=True)
if len(sys.argv)>8:
 ref=[(lead_of(x.split(':')[0]),float(x.split(':')[1])) for x in sys.argv[8].split(',')]
 Cr=evaluate([ref]);Cro=evaluate([ref],OTHER) if OTHER in T else np.full((len(mem),1),np.inf)
 print(f"REF {w@Cr[:,0]-bE:+.3f} h (better/equal for best in {int((bC<=Cr[:,0]+.05).sum())}/{len(mem)}; other family {float(w@Cro[:,0]):.2f})  {desc(tuple(ref))}",flush=True)
O=pad(bk);tr=sim(np.array([[t for t,_ in O]]),np.array([[a for _,a in O]]),WF[0],track=True)
pts=[[lat,lon,'']]+[[x[0],x[1],''] for x in tr[5::6]]
for t,a in bk:
 i=int(round((t-L0)/DT))-1
 if 0<=i<len(tr):pts.append([tr[i][0],tr[i][1],f"{paris(t)} {abs(a):.0f}° {'trib' if a>0 else 'bâb'}"])
pts.sort(key=lambda x:0);print('TRACK',json.dumps([[float(a),float(b),c] for a,b,c in [[lat,lon,'']]+[[x[0],x[1],''] for x in tr]][::3]+[[x[0],x[1],x[2]] for x in pts if x[2]]),flush=True)
POOL.close()
