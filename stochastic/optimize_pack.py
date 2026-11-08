# Full pack (7 sails, pro winch, furler, jacket, pug) stochastic program choice: commands = (time, TWA, sail).
# Usage: optimize_pack.py CYCLE OBS_UTC lat lon energy angle sail tag [H]
# Terminal value: Jib/Spi fine Bellman of the cycle (pack assumed gone after ~24 h); sail state mapped to Spi/Jib layer.
import os,sys,json,itertools,datetime,glob
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
H=float(sys.argv[9]) if len(sys.argv)>9 else 12.;FEED=12.;TACK0=1 if A0>0 else -1
G=dict(np.load(os.environ.get('BELLMAN',ENS/'bellman_fine.npz')))
names=['gfs']+sorted(Path(x).name[:-len('3h.npz')] for x in glob.glob(str(ENS/'gefs*3h.npz')))
if os.environ.get('DET')=='1':names=['gfs','gfs'] # deterministic GFS router (the fleet), backtest baseline
F=[bf.Field(ENS/f'{n}3h.npz') for n in names];M=len(F);L=p.PEN_LOSS
if os.environ.get('GAME_WIND')=='1': # point 1: game wind = previous GFS, then this GFS, scenarios only after the next bulletin
 import game_wind;F=game_wind.fields(ENS,names,CYC);print('GAME_WIND delay',game_wind.DELAY,flush=True)
WEIGHTS=np.ones(len(names)-1)
if os.environ.get('GAME_WIND')=='1':WEIGHTS=game_wind.scenario_weights(ENS,names,CYC)[0] # quick night update: members weighted by the next GFS
print('MEMBERS',M-1,'lead0',round(LEAD0,3),'pen_loss',L,flush=True)
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
if TERM=='tree':
 import bellman_tree;TREE=bellman_tree.load(ENS);GT=bf.load_members(ENS,names,G)[1:];print('TERMINAL tree splits',TREE['splits'].tolist(),flush=True)
def V(t,lat,lon,sail,tack):
 s=np.where(np.asarray(sail)==0,0,1)
 if TREE is not None:return bellman_tree.value_tree(TREE,GT,LEAD0+t,lat,lon,s,tack)
 return bf.value_members(GM,LEAD0+t,lat,lon,s,tack) if GM else bf.value(G,LEAD0+t,lat,lon,s,tack)
# Pack end (hours after OBS, env PACK_END): pack sails unavailable afterwards; programmed pack sail -> PACK_FALLBACK.
PACK_END=float(os.environ.get('PACK_END','1e9'));PACK_FALLBACK=int(os.environ.get('PACK_FALLBACK','1'))
AUTO=os.environ.get('AUTO_SAIL')=='1'
def auto_sail(a,ws,t):
 # Game auto sail: fastest available sail at the current TWA/TWS (Jib/Spi only once the pack has ended).
 av=[0,1] if t>=PACK_END else SAILS
 sp=np.stack([p.polar(a,ws,x) for x in av]);return np.array(av)[np.argmax(sp,axis=0)]
SAILS=[1,4,5,6,0] if PACK_END>0 else [1,0] # Spi, Code0, HeavyGnk, LightGnk, Jib; Spi/Jib only once the pack has ended (5x fewer programs)
T=np.round(np.arange(0,H+1e-9,1/3),6);programs=[]
for a,s in itertools.product([100,105,110,115,120,125,130,135,140,145,150,155],SAILS):
 programs.append(dict(family='constant',times=[0.],angles=[TACK0*a],sails=[s]))
for x,s1,t1,y,s2 in itertools.product([95,100,105,110,115,120,125,130,135,140,145,150],SAILS,T[1:],[100,105,110,115,120,125,130,135,140,145,150],SAILS):
 if (x,s1)!=(y,s2):programs.append(dict(family='change',times=[0.,float(t1)],angles=[TACK0*x,TACK0*y],sails=[s1,s2]))
for a,s in itertools.product([110,120,130,140,150],SAILS):
 programs.append(dict(family='gybe_now',times=[0.],angles=[-TACK0*a],sails=[s]))
if os.environ.get('AUTO_SAIL')=='1':programs=[q for q in programs if set(q['sails'])=={1}] # sail chosen by the game
# Night lock: the sail cannot be programmed, so an order whose local (Paris, UTC+1) time falls in NIGHT_LOCK keeps the
# previous sail; the user changes sail by hand only when awake. NIGHT_LOCK='' disables it.
_nl=os.environ.get('NIGHT_LOCK','22:30-09:00');_OBS_LOCAL=datetime.datetime.strptime(OBS,'%Y%m%d%H%M')+datetime.timedelta(hours=1)
def is_night(t):
 if not _nl:return False
 a,b=[int(x[:2])*60+int(x[3:]) for x in _nl.split('-')];m=(_OBS_LOCAL+datetime.timedelta(hours=t));m=m.hour*60+m.minute
 return (m>=a or m<b) if a>b else (a<=m<b)
programs=[q for q in programs if not (len(q['times'])>1 and is_night(q['times'][1]) and q['sails'][1]!=q['sails'][0])]
if os.environ.get('ONE_SAIL')=='1':programs=[q for q in programs if len(set(q['sails']))==1] # sail cannot be programmed
N=len(programs);K=2;times=np.full((N,K),np.inf);angles=np.zeros((N,K));sails=np.zeros((N,K),int)
for i,q in enumerate(programs):
 n=len(q['times']);times[i,:n]=q['times'];angles[i,:n]=q['angles'];sails[i,:n]=q['sails']
print('PROGRAMS',N,flush=True)
def simulate(ix):
 # Programs ix over [0, H] in every scenario (2 min step), score = H + terminal value; one chunk per worker process.
 n=len(ix);tm,an,sl=times[ix],angles[ix],sails[ix];r=np.arange(n)
 la=np.full((n,M),START[0]);lo=np.full((n,M),START[1]);en=np.full((n,M),START[2]);ta=np.full((n,M),TACK0);sa=np.full((n,M),SAIL0)
 pend=np.zeros((n,M));valid=np.ones((n,M),bool);minimum=en.copy();dt=1/30
 for t in np.arange(0,H-1e-8,dt):
  idx=np.sum(tm<=t+1e-8,axis=1)-1;a=an[r,idx][:,None];s=np.broadcast_to(sl[r,idx][:,None],la.shape)
  if t>=PACK_END:s=np.where((s==0)|(s==1),s,PACK_FALLBACK)
  ws,wd=wind(t,la,lo);nt=np.where(a>=0,1,-1)
  if AUTO:s=auto_sail(a,ws,t)
  spent,pen=p.maneuver(ws,en,s!=sa,nt!=ta);pend=pend+pen;slow=np.minimum(dt,pend)
  ln,on=p.move(la,lo,(wd-a)%360,p.polar(a,ws,s)*(dt-L*slow))
  en=np.minimum(100,np.maximum(0,en-spent)+(dt-slow)*3600/p.recover_seconds(ws))
  valid&=p.leg_allowed(la,lo,ln,on);la,lo=ln,on;ta=np.broadcast_to(nt,la.shape).copy();sa=s.copy();pend=np.maximum(0,pend-dt);minimum=np.minimum(minimum,en)
 score=H+V(H,la,lo,sa,ta)+.0005*(100-en)+L*pend;score[~valid]=np.inf
 return la,lo,en,ta,sa,pend,minimum,score
# Parallel over program chunks (OPT_WORKERS processes, default all cores; fork shares the wind and value tables).
import multiprocessing as mp
NW=int(os.environ.get('OPT_WORKERS',os.cpu_count() or 1));chunks=[c for c in np.array_split(np.arange(N),max(1,2*NW)) if len(c)]
if NW>1:
 with mp.get_context('fork').Pool(NW) as pool:parts=pool.map(simulate,chunks)
else:parts=[simulate(c) for c in chunks]
la,lo,en,ta,sa,pend,minimum,score=(np.concatenate([q[k] for q in parts]) for k in range(8))
print('SIMULATED',N,'programs on',NW,'workers',flush=True)
w=WEIGHTS;rank=np.array([risk(row[1:],w,alpha=.8,tail_weight=TAIL_W)['objective'] for row in score])
selected=[]
for fam in ['constant','change','gybe_now']:
 ids=[i for i in np.argsort(rank) if programs[i]['family']==fam and np.isfinite(rank[i])];selected+=ids[:15]
# Feedback H -> H+FEED over (angle, sail) with own-member current wind.
ll=la[selected];oo=lo[selected];ee=en[selected];tt=ta[selected];pp=pend[selected];ss=sa[selected]
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
final=H+FEED+V(H+FEED,ll,oo,ss,tt)+.0005*(100-ee)+L*pp
records=[]
for k,i in enumerate(selected):
 m=risk(final[k,1:],w,alpha=.8,tail_weight=TAIL_W)
 records.append(dict(**programs[i],sail_names=[p.SAIL_NAMES[x] for x in programs[i]['sails']],**m,nominal_gfs=float(final[k,0]),ens_spread=float(np.std(final[k,1:])),min_energy=float(minimum[i].min()),rank12=float(rank[i])))
records.sort(key=lambda q:q['objective'])
# Reference: Spi 115 constant (no pack benefit) for the gain estimate.
ref=[i for i,q in enumerate(programs) if q['family']=='constant' and q['angles']==[TACK0*115] and q['sails']==[1]]
res=dict(full_pack=True,cycle=CYC,observation_utc=OBS,start=START,members=M-1,candidates=N,records=records,
 rank12_spi115=float(rank[ref[0]]) if ref else None,best_rank12=float(np.min(rank)),
 limits='terminal value from Jib/Spi table at +24 h; pack penalties per ITYC model; sails restricted to Spi/Code0/HeavyGnk/LightGnk/Jib')
(R/f'stochastic/result_{TAG}.json').write_text(json.dumps(res,indent=1));print('DONE',flush=True)
