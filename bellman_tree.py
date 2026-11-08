"""Scenario-tree Bellman value (information revealed at the next bulletins).

Between the clairvoyant bound (bellman_fine.py MEMBER: each scenario known from the start) and the
non-anticipative bound (bellman_na.py: never known), the scenario is revealed progressively, bulletin after bulletin:
the GEFS members are clustered hierarchically (Ward, on their wind over the route area), the tree is cut into
SPLITS = [(lead, clusters), ...] e.g. 12 h -> 3 groups, 24 h -> 9, 48 h -> every member alone.
Inside a node one decision serves all its members (non-anticipative, mean over members); at the node's end the
value is the weighted mean of its children (we learn which child we are in). After the last split each member
uses its own table (bellman_members/MEMBER.npz). Wind = game wind (stochastic/game_wind.py) when available.
Usage: bellman_tree.py ens_dir CYCLE [workers] -> ens_dir/bellman_tree.npz
"""
import os,sys,time,json
from pathlib import Path
import numpy as np
from multiprocessing import Pool
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));sys.path.insert(0,str(R/'stochastic'))
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'))
import optimal_route as p
import bellman_fine as bf
from bellman_na import LA,LO,G,OK,TARGET,LAS,LOS
SPLITS=json.loads(os.environ.get('TREE_SPLITS','[[12,3],[24,9],[48,0]]')) # 0 = every member alone
_W=None
def _init(ens_dir,cyc):
 global _W
 names=['gfs']+[f.name[:-len('3h.npz')] for f in sorted(Path(ens_dir).glob('gefs*3h.npz'))]
 try:
  import game_wind;_W=game_wind.fields(ens_dir,names,cyc)[1:]
 except Exception as e:
  print('tree: game wind unavailable, member wind',e,flush=True)
  gfs=bf.Field(Path(ens_dir)/'gfs3h.npz');_W=[bf.Blend(bf.Field(Path(ens_dir)/f'{n}3h.npz'),gfs) for n in names[1:]]
def _step(job):
 # Partial min over a subset of angles for one node and one time step, mean over the node's members.
 lead,angles,cur,members=job;dt=1.
 W=[_W[j](lead+.5*dt,LA,LO) for j in members]
 best=np.full((4,*LA.shape),2000.,np.float32)
 for a in angles:
  nt=1 if a>0 else -1
  for ns in (0,1):
   jn=int(bf.layer(ns,nt))
   for ps in (0,1):
    for pt in (-1,1):
     acc=np.zeros(LA.shape)
     for ws,wd in W:
      _,pen=p.maneuver(ws,bf.TYPICAL_ENERGY,ps!=ns,pt!=nt)
      ln,on=p.move(LA,LO,(wd-a)%360,p.polar(a,ws,ns)*(dt-.5*np.minimum(dt,pen)))
      acc+=np.where(p.leg_allowed(LA,LO,ln,on),dt+bf.interp(cur[jn],ln,on,**G),2000)
     jp=int(bf.layer(ps,pt));np.minimum(best[jp],(acc/len(W)).astype(np.float32),out=best[jp])
 return best
def clusters(ens_dir,lead0):
 """Nested member clusters for each split (Ward on u,v over the route box, leads 12..120 every 12 h)."""
 from scipy.cluster.hierarchy import linkage,fcluster
 files=sorted(Path(ens_dir).glob('gefs*3h.npz'));X=[]
 la,lo=np.meshgrid(np.arange(LAS[0],min(50,LAS[-1])+.1,2.5),np.arange(LOS[0],min(-10,LOS[-1])+.1,2.5),indexing="ij")
 for f in files:
  F=bf.Field(f);v=[]
  for L in range(12,121,12):
   ws,wd=F(L,la,lo);v+=[(-ws*np.sin(np.radians(wd))).ravel(),(-ws*np.cos(np.radians(wd))).ravel()]
  X.append(np.concatenate(v))
 Z=linkage(np.array(X),'ward')
 return [np.unique(fcluster(Z,k if k>0 else len(files),'maxclust'),return_inverse=True)[1] for _,k in SPLITS] # labels 0..n-1, nested
def build(ens_dir,cyc,workers=7):
 ens_dir=Path(ens_dir);Gf=dict(np.load(ens_dir/'bellman_fine.npz'));lead0=float(Gf['lead0'])
 names=[f.name[:-len('3h.npz')] for f in sorted(ens_dir.glob('gefs*3h.npz'))];M=len(names)
 labels=clusters(ens_dir,lead0)
 last=SPLITS[-1][0];GM=[dict(np.load(ens_dir/'bellman_members'/f'{n}.npz')) for n in names]
 # Value at the last split for each member = its own (clairvoyant) table.
 val={j:GM[j]['values'][int(last-lead0)].astype(np.float32) for j in range(M)};lev_lab=list(range(M))
 out={'labels':np.array(labels),'splits':np.array(SPLITS),'lead0':lead0,'lat':LAS,'lon':LOS,'members':np.array(names)}
 chunks=np.array_split(bf.ANGLES,workers);tic=time.time()
 bounds=[lead0]+[s for s,_ in SPLITS]
 with Pool(workers,_init,(str(ens_dir),cyc)) as pool:
  for lev in range(len(SPLITS)-1,-1,-1):
   a,b=bounds[lev],bounds[lev+1];lab=labels[lev-1] if lev>0 else np.zeros(M,int)
   nodes=sorted(set(lab));V=np.empty((len(nodes),int(b-a)+1,4,*LA.shape),np.float32)
   for c in nodes:
    mem=[j for j in range(M) if lab[j]==c]
    cur=np.mean([val[j] for j in mem],axis=0).astype(np.float32);V[c,-1]=cur # mean of the children = mean of members
    for k in range(int(b-a)-1,-1,-1):
     parts=pool.map(_step,[(a+k,ch,cur,mem) for ch in chunks])
     best=np.min(parts,axis=0);best[:,TARGET]=0;best[:,~OK]=2000;cur=best;V[c,k]=best
    for j in mem:val[j]=V[c,0]
    print('tree level',lev,'node',c,'members',len(mem),'leads',a,b,'elapsed',round(time.time()-tic),'s',flush=True)
   out[f'level{lev}']=V;out[f'level{lev}_lead0']=a;out[f'level{lev}_labels']=lab
 path=ens_dir/os.environ.get('TREE_OUT','bellman_tree.npz');np.savez(path,**out);return path # TREE_OUT=bellman_na48.npz with TREE_SPLITS=[[48,0]]: non-anticipative 48 h
def load(ens_dir,name='bellman_tree.npz'):
 return dict(np.load(Path(ens_dir)/name))
def value_tree(T,GM,lead,lat,lon,sail,tack):
 """Column j of axis 1 = GEFS member j-1 (column 0 = GFS, uses the root/its mean like member 0). Node by lead."""
 lat,lon,sail,tack=np.broadcast_arrays(lat,lon,sail,tack);out=np.empty(lat.shape)
 splits=[s for s,_ in T['splits']];last=splits[-1]
 for col in range(lat.shape[1]):
  j=max(col-1,0)
  if lead>=last:G_=GM[j]
  else:
   lev=int(np.searchsorted(splits,lead,side='right'));node=int(T[f'level{lev}_labels'][j])
   G_=dict(values=T[f'level{lev}'][node],lead0=T[f'level{lev}_lead0'],lat=T['lat'],lon=T['lon'])
  out[:,col]=bf.value(G_,lead,lat[:,col],lon[:,col],sail[:,col],tack[:,col])
 return out
if __name__=='__main__':print('WROTE',build(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 7))
