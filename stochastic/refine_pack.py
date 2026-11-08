# Continuous refinement of the best full-pack programs (shooting / local search on the stochastic objective).
# Usage: refine_pack.py CYCLE OBS_UTC lat lon energy angle sail tag SEED_RESULT [H] [K]
# Starts from the best programs of SEED_RESULT (optimize_pack output) and moves switch times to the minute,
# angles to the degree, sails, and adds/removes switches (up to K commands), keeping a move only when the
# mean/CVaR objective over the 31 GEFS members improves. Same dynamics, feedback phase and terminal value as optimize_pack.
import os,sys,json,datetime,glob
os.environ['FULL_PACK']='1'
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'))
import optimal_route as p
import bellman_fine as bf
from planner import risk
# Risk attitude (point 3): weight of the CVaR tail in the objective; 0 = mean only (offensive), .25 = prudent.
TAIL_W=float(os.environ.get('TAIL_WEIGHT','0.25'))
CYC,OBS=sys.argv[1],sys.argv[2];ENS=R/'data'/f'ens_{CYC}'
LEAD0=(datetime.datetime.strptime(OBS,'%Y%m%d%H%M')-datetime.datetime.strptime(CYC,'%Y%m%d%H')).total_seconds()/3600
START=(float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]));A0=float(sys.argv[6]);SAIL0=int(sys.argv[7]);TAG=sys.argv[8]
SEED=json.loads(Path(sys.argv[9]).read_text())
H=float(sys.argv[10]) if len(sys.argv)>10 else 12.;K=int(sys.argv[11]) if len(sys.argv)>11 else 3;FEED=12.;TACK0=1 if A0>0 else -1
G=dict(np.load(os.environ.get('BELLMAN',ENS/'bellman_fine.npz')))
names=['gfs']+sorted(Path(x).name[:-len('3h.npz')] for x in glob.glob(str(ENS/'gefs*3h.npz')))
if os.environ.get('DET')=='1':names=['gfs','gfs'] # deterministic GFS router (the fleet), backtest baseline
F=[bf.Field(ENS/f'{n}3h.npz') for n in names];M=len(F);L=p.PEN_LOSS
if os.environ.get('GAME_WIND')=='1': # point 1: game wind = previous GFS, then this GFS, scenarios only after the next bulletin
 import game_wind;F=game_wind.fields(ENS,names,CYC);print('GAME_WIND delay',game_wind.DELAY,flush=True)
WEIGHTS=np.ones(len(names)-1)
if os.environ.get('GAME_WIND')=='1':WEIGHTS=game_wind.scenario_weights(ENS,names,CYC)[0] # quick night update: members weighted by the next GFS
ONE=os.environ.get('ONE_SAIL')=='1'
# Pack end (hours after OBS, env PACK_END): pack sails unavailable afterwards; programmed pack sail -> PACK_FALLBACK.
PACK_END=float(os.environ.get('PACK_END','1e9'));PACK_FALLBACK=int(os.environ.get('PACK_FALLBACK','1'))
AUTO=os.environ.get('AUTO_SAIL')=='1'
def auto_sail(a,ws,t):
 # Game auto sail: fastest available sail at the current TWA/TWS (Jib/Spi only once the pack has ended).
 av=[0,1] if t>=PACK_END else SAILS
 sp=np.stack([p.polar(a,ws,x) for x in av]);return np.array(av)[np.argmax(sp,axis=0)]
SAILS=[1,4,5,6,0] # Spi, Code0, HeavyGnk, LightGnk, Jib
print('MEMBERS',M-1,'lead0',round(LEAD0,3),flush=True)
def wind(t,lat,lon):
 ws=np.empty(lat.shape);wd=np.empty(lat.shape)
 for j,f in enumerate(F):ws[:,j],wd[:,j]=f(LEAD0+t,lat[:,j],lon[:,j])
 return ws,wd
# Terminal value per scenario (member wind up to its last lead, then GFS) when the member tables exist, else GFS only.
# TERMINAL=na: non-anticipative ensemble table (bellman_na.py, one decision for all scenarios) for every member.
TERM=os.environ.get('TERMINAL','member')
if TERM=='na':G=dict(np.load(ENS/'bellman_na.npz'))
GM=bf.load_members(ENS,names,G) if TERM=='member' and os.environ.get('MEMBER_VALUE','1')=='1' else None
print('TERMINAL','per-member' if GM else ('non-anticipative' if TERM=='na' else 'gfs-only'),flush=True)
# TERMINAL=tree: scenario tree (bellman_tree.py), information revealed at the next bulletins.
TREE=None
# TERMINAL=na48: one root node over all members up to 48 h (bellman_tree.py TREE_SPLITS=[[48,0]]), fast non-anticipative bound.
if TERM in ('tree','na48'):
 import bellman_tree;TREE=bellman_tree.load(ENS,'bellman_na48.npz' if TERM=='na48' else 'bellman_tree.npz');GT=bf.load_members(ENS,names,G)[1:];print('TERMINAL tree splits',TREE['splits'].tolist(),flush=True)
def V(t,lat,lon,sail,tack):
 s=np.where(np.asarray(sail)==0,0,1)
 if TREE is not None:return bellman_tree.value_tree(TREE,GT,LEAD0+t,lat,lon,s,tack)
 return bf.value_members(GM,LEAD0+t,lat,lon,s,tack) if GM else bf.value(G,LEAD0+t,lat,lon,s,tack)
def evaluate(programs):
 N=len(programs);times=np.full((N,K),np.inf);angles=np.zeros((N,K));sails=np.zeros((N,K),int)
 for i,q in enumerate(programs):
  n=len(q['times']);times[i,:n]=q['times'];angles[i,:n]=q['angles'];sails[i,:n]=q['sails']
 la=np.full((N,M),START[0]);lo=np.full((N,M),START[1]);en=np.full((N,M),START[2]);ta=np.full((N,M),TACK0);sa=np.full((N,M),SAIL0)
 pend=np.zeros((N,M));valid=np.ones((N,M),bool);minimum=en.copy();dt=1/60;r=np.arange(N)
 for t in np.arange(0,H-1e-8,dt):
  idx=np.sum(times<=t+1e-8,axis=1)-1;a=angles[r,idx][:,None];s=np.broadcast_to(sails[r,idx][:,None],la.shape)
  if t>=PACK_END:s=np.where((s==0)|(s==1),s,PACK_FALLBACK)
  ws,wd=wind(t,la,lo);nt=np.where(a>=0,1,-1)
  if AUTO:s=auto_sail(a,ws,t)
  spent,pen=p.maneuver(ws,en,s!=sa,nt!=ta);pend=pend+pen;slow=np.minimum(dt,pend)
  ln,on=p.move(la,lo,(wd-a)%360,p.polar(a,ws,s)*(dt-L*slow))
  en=np.minimum(100,np.maximum(0,en-spent)+(dt-slow)*3600/p.recover_seconds(ws))
  valid&=p.leg_allowed(la,lo,ln,on);la,lo=ln,on;ta=np.broadcast_to(nt,la.shape).copy();sa=s.copy();pend=np.maximum(0,pend-dt);minimum=np.minimum(minimum,en)
 ll,oo,ee,tt,pp,ss=la,lo,en,ta,pend,sa
 A=np.repeat(bf.ANGLES,len(SAILS));S=np.tile(SAILS,len(bf.ANGLES))
 for t in np.arange(H,H+FEED,.5):
  ws,wd=wind(t,ll,oo);heading=(wd[:,:,None]-A)%360;newt=np.where(A>=0,1,-1)
  spent,pen=p.maneuver(ws[:,:,None],ee[:,:,None],S!=ss[:,:,None],newt!=tt[:,:,None]);slow=np.minimum(.5,pp[:,:,None]+pen)
  ln,on=p.move(ll[:,:,None],oo[:,:,None],heading,p.polar(A,ws[:,:,None],S)*(.5-L*slow))
  en2=np.minimum(100,np.maximum(0,ee[:,:,None]-spent)+(.5-slow)*3600/p.recover_seconds(ws[:,:,None]))
  cost=V(t+.5,ln,on,np.broadcast_to(S,ln.shape),np.broadcast_to(newt,ln.shape))+.0005*(100-en2)
  cost[~p.leg_allowed(ll[:,:,None],oo[:,:,None],ln,on)]=np.inf
  if t>=PACK_END:cost[...,~np.isin(S,[0,1])]=np.inf
  ix=np.argmin(cost,axis=2);aa=A[ix];ns=S[ix]
  for sub in range(6):
   w_,d_=wind(t+sub/12,ll,oo);nt=np.where(aa>=0,1,-1)
   debit,penalty=p.maneuver(w_,ee,ns!=ss,nt!=tt);pp=pp+penalty;slow=np.minimum(1/12,pp)
   ll,oo=p.move(ll,oo,(d_-aa)%360,p.polar(aa,w_,ns)*(1/12-L*slow))
   ee=np.minimum(100,np.maximum(0,ee-debit)+(1/12-slow)*3600/p.recover_seconds(w_));ss=ns;tt=nt;pp=np.maximum(0,pp-1/12)
 final=H+FEED+V(H+FEED,ll,oo,ss,tt)+.0005*(100-ee)+L*pp;final[~valid]=np.inf
 out=[]
 for k in range(N):
  m=risk(final[k,1:],WEIGHTS,alpha=.8,tail_weight=TAIL_W) if np.all(np.isfinite(final[k])) else dict(objective=np.inf,mean=np.inf,cvar=np.inf)
  out.append(dict(**m,nominal_gfs=float(final[k,0]),min_energy=float(minimum[k].min()),members=[float(x) for x in final[k,1:]],gfs_pos_H=[float(la[k,0]),float(lo[k,0])]))
 return out
def key(q):return (tuple(round(t*60) for t in q['times']),tuple(q['angles']),tuple(q['sails']))
# Night lock: the sail cannot be programmed, so an order whose local (Paris, UTC+1) time falls in NIGHT_LOCK keeps the
# previous sail; the user changes sail by hand only when awake. NIGHT_LOCK='' disables it.
_nl=os.environ.get('NIGHT_LOCK','22:30-09:00');_OBS_LOCAL=datetime.datetime.strptime(OBS,'%Y%m%d%H%M')+datetime.timedelta(hours=1)
def is_night(t):
 if not _nl:return False
 a,b=[int(x[:2])*60+int(x[3:]) for x in _nl.split('-')];m=(_OBS_LOCAL+datetime.timedelta(hours=t));m=m.hour*60+m.minute
 return (m>=a or m<b) if a>b else (a<=m<b)
def clean(q):
 # Program in whole minutes and degrees; drop switches that change nothing or fall outside (0,H).
 t=[0.]+[round(x*60)/60 for x in q['times'][1:]];a=[int(round(x)) for x in q['angles']];s=list(q['sails'])
 o=sorted(range(len(t)),key=lambda i:t[i]);T,Aa,Ss=[],[],[]
 for i in o:
  if i and not (0<t[i]<H):continue
  if T and (Aa[-1],Ss[-1])==(a[i],s[i]):continue
  if T and abs(t[i]-T[-1])<1e-9:T.pop();Aa.pop();Ss.pop()
  T.append(t[i]);Aa.append(a[i]);Ss.append(s[i])
 if ONE: Ss=[Ss[0]]*len(Ss) # sail cannot be programmed: the sail chosen now holds for the whole program
 for i in range(1,len(T)):
  if is_night(T[i]):Ss[i]=Ss[i-1]
 Ss=[x if (T[i]<PACK_END or x in (0,1)) else PACK_FALLBACK for i,x in enumerate(Ss)] # pack sails gone (also now)
 return dict(times=T,angles=Aa,sails=Ss)
def neighbours(q,dt_min,da):
 n=len(q['times']);out=[]
 for i in range(1,n):
  for d in (-dt_min,dt_min):
   z=json.loads(json.dumps(q));z['times'][i]+=d/60;out.append(z)
 for i in range(n):
  for d in (-da,da):
   z=json.loads(json.dumps(q));z['angles'][i]+=TACK0*d if z['angles'][i]*TACK0>0 else -TACK0*d;out.append(z)
  for s in SAILS:
   if s!=q['sails'][i]:
    z=json.loads(json.dumps(q))
    if ONE:z['sails']=[s]*n
    else:z['sails'][i]=s
    out.append(z)
 if n<K: # add a switch in the middle of each segment (angle step 5 either side) or at a few fixed times after the last one
  ends=q['times'][1:]+[H]
  for i in range(n):
   for tm in {(q['times'][i]+ends[i])/2,q['times'][i]+(ends[i]-q['times'][i])/4,q['times'][i]+3*(ends[i]-q['times'][i])/4}:
    for d in (-5,5):
     z=json.loads(json.dumps(q));z['times'].insert(i+1,tm);z['angles'].insert(i+1,z['angles'][i]+(TACK0 if z['angles'][i]*TACK0>0 else -TACK0)*d);z['sails'].insert(i+1,z['sails'][i]);out.append(z)
 for i in range(1,n): # gybe at switch i (this and later segments on the other tack)
  z=json.loads(json.dumps(q));z['angles']=z['angles'][:i]+[-x for x in z['angles'][i:]];out.append(z)
 if n<K: # gybe added at a few times in each segment, at 120/135/150 on the new tack
  ends=q['times'][1:]+[H]
  for i in range(n):
   for tm in np.linspace(q['times'][i],ends[i],5)[1:-1]:
    for a in (120,135,150):
     z=json.loads(json.dumps(q));z['times'].insert(i+1,float(tm));z['angles'].insert(i+1,-np.sign(z['angles'][i])*a);z['sails'].insert(i+1,z['sails'][i]);out.append(z)
 for i in range(1,n): # remove a switch
  z=json.loads(json.dumps(q));[z[k].pop(i) for k in ('times','angles','sails')];out.append(z)
 return out
seed=[dict(times=x['times'],angles=x['angles'],sails=x['sails']) for x in SEED['records'] if x['family'].startswith('current')]
seed+=[dict(times=x['times'],angles=x['angles'],sails=x['sails']) for x in SEED['records'] if x['family']!='gybe_now' and not x['family'].startswith('current')][:6]
# Gybe seeds: each seed plus one gybe at every hour, new tack at 120..160.
seed+=[dict(times=q['times']+[float(t)],angles=q['angles']+[-TACK0*a],sails=q['sails']+[q['sails'][-1]]) for q in seed[:3] if len(q['times'])<K for t in np.arange(1,H,1.) for a in (120,135,150,160)]
# EVAL_ONLY=1: score the seed records as given (consensus between terminals), no search; MARGINS=1 adds margins for record 0.
EVAL=os.environ.get('EVAL_ONLY')=='1'
if EVAL:seed=[dict(times=x['times'],angles=x['angles'],sails=x['sails']) for x in SEED['records']]
pool=[clean(q) for q in seed];seen={}
# Safety margin (point 5): a program is scored on its nominal objective plus ROBUST_W times the worst loss when all
# its switch times are shifted by +-ROBUST_DT minutes (late execution, weather event earlier/later than forecast).
# Programs sitting next to a cliff (gybe 1 h too early = +10 h) are thus pushed to the flat side.
# The penalty is capped (ROBUST_CAP h): programmed orders run on time and the front timing is already in the scenarios,
# so a cliff must not override the scenario mean (4/11: a gybe 30 min early = +16 h made it drop a program better in 27/27).
ROB_DT=float(os.environ.get('ROBUST_DT','30'));ROB_W=float(os.environ.get('ROBUST_W','0.5'));ROB_CAP=float(os.environ.get('ROBUST_CAP','0.3'));TOL=float(os.environ.get('MARGIN_TOL','0.15'))
def shifted(q,s_min,idx=None):
 z=json.loads(json.dumps(q))
 for i in (range(1,len(z['times'])) if idx is None else [idx]):z['times'][i]+=s_min/60
 if idx is not None and not all(a<b for a,b in zip(z['times'],z['times'][1:])):return None
 return clean(z)
def nom(q):return seen[key(q)][1]['objective']
def J(q):
 if ROB_DT<=0 or len(q['times'])<2:return nom(q)
 return nom(q)+min(ROB_CAP,ROB_W*max(0.,max(nom(shifted(q,d)) for d in (-ROB_DT,ROB_DT))-nom(q)))
def score(progs):
 allp=list(progs)
 if ROB_DT>0:allp+=[shifted(q,d) for q in progs if len(q['times'])>1 for d in (-ROB_DT,ROB_DT)]
 new={key(q):q for q in allp if key(q) not in seen}
 if new:
  for q,r in zip(new.values(),evaluate(list(new.values()))):seen[key(q)]=(q,r)
 return [J(q) for q in progs]
score(pool);best=sorted(pool,key=J)[:3]
print('SEED',[round(J(q),3) for q in best],'robust dt',ROB_DT,'w',ROB_W,flush=True)
for dt_min,da in ([] if EVAL else [(40,4),(20,2),(10,1),(5,1),(2,1)]):
 improved=True;it=0
 while improved and it<6:
  improved=False;it+=1;cand=[]
  for q in best:cand+=[clean(z) for z in neighbours(q,dt_min,da)]
  score(cand)
  allq=sorted({key(q):q for q in best+cand}.values(),key=J)
  nb=allq[:3]
  if J(nb[0])<J(best[0])-1e-4:improved=True
  best=nb
  print('step',dt_min,da,'it',it,'evaluated',len(seen),'best',round(J(best[0]),3),key(best[0]),flush=True)
# Explicit margins for the chosen program: loss when one switch (or all of them) is moved by -120..+120 min.
b0=pool[0] if EVAL else best[0];shifts=list(range(-120,121,10));margins=[]
for idx in ([] if EVAL and os.environ.get('MARGINS')!='1' else list(range(1,len(b0['times'])))+[None]):
 qs=[(m,shifted(b0,m,idx)) for m in shifts];qs=[(m,q) for m,q in qs if q is not None and len(q['times'])==len(b0['times'])]
 score([q for _,q in qs])
 loss={m:nom(q)-nom(b0) for m,q in qs}
 lo=0
 while lo-10 in loss and loss[lo-10]<=TOL:lo-=10
 hi=0
 while hi+10 in loss and loss[hi+10]<=TOL:hi+=10
 margins.append(dict(switch='all' if idx is None else idx,minute=None if idx is None else round(b0['times'][idx]*60),window=[lo,hi],
  early_60=loss.get(-60),late_60=loss.get(60),loss={str(m):round(v,3) for m,v in loss.items()}))
ranked=sorted(seen.values(),key=lambda x:J(x[0]) if key(x[0]) in seen and all(key(z) in seen for z in ([shifted(x[0],d) for d in (-ROB_DT,ROB_DT)] if ROB_DT>0 and len(x[0]['times'])>1 else [])) else x[1]['objective']+99)
if EVAL:ranked=[seen[key(q)] for q in pool]
records=[dict(**q,sail_names=[p.SAIL_NAMES[s] for s in q['sails']],minutes=[round(t*60) for t in q['times']],robust=J(q),**r) for q,r in ranked[:40]]
seed_best=min(J(clean(q)) for q in seed)
s0=clean(seed[0])
# Leave the fleet (point 3): the fleet follows a deterministic GFS router, i.e. the program with the best GFS
# outcome. Among programs within FLEET_TIE h (robust) of the best, pick the one ending farthest from that route at H.
TIE=float(os.environ.get('FLEET_TIE','0'))
herdq=min(seen.values(),key=lambda x:x[1]['nominal_gfs']);herd=herdq[1]['gfs_pos_H']
def gap(r):return float(p.distance(np.array([r['gfs_pos_H'][0]]),np.array([r['gfs_pos_H'][1]]),(herd[0],herd[1]))[0])
for r in records:r['fleet_gap_nm']=gap(r)
if TIE>0:
 close=[r for r in records if r['robust']<=records[0]['robust']+TIE]
 pick=max(close,key=lambda r:r['fleet_gap_nm'])
 if pick is not records[0]:print('FLEET pick',pick['minutes'],pick['angles'],'gap',round(pick['fleet_gap_nm']),'vs',round(records[0]['fleet_gap_nm']),flush=True)
 records.remove(pick);records.insert(0,pick)
res=dict(seed0=dict(seen[key(s0)][1],robust=J(s0)),seed0_program=s0,cycle=CYC,observation_utc=OBS,start=START,members=M-1,evaluated=len(seen),robust_dt=ROB_DT,robust_w=ROB_W,
 seed_best=seed_best,refined_best=records[0]['robust'],herd_program={k:herdq[0][k] for k in ('times','angles','sails')},weights=[float(x) for x in WEIGHTS],margins=margins,records=records)
(R/f'stochastic/result_{TAG}.json').write_text(json.dumps(res,indent=1,default=float));print('DONE gain',round(seed_best-records[0]['robust'],3),flush=True)
