"""Logic checks of the routing chain (no pytest needed): python tests/check_logic.py [CYCLE] [--fast]

Each check covers a part where a silent error already happened or would change the orders:
  finish   passage line exists on every grid step (6/11 0:13: none on the 0.5 deg grid -> 122 h announced), the forbidden
           zone and the mark are where the game puts them, the finish is reachable in the fine table.
  boat     polar / maneuver penalty (a gybe costs time, nothing changed costs nothing; standard winch 50 %).
  mail     gybe count from a bâbord start, CHANGER / RIEN A CHANGER threshold, plausibility guard (sanity).
  tables   on the last cycle: every value table (GFS, members, gated family) finite, not constant, decreasing towards the
           finish, plausible value on the passage line.
  program  family_prog.py from the state position: no gybe/back-gybe zigzag, plausible time, sane margins (~20 s; --fast skips).
Exit code 1 if any check fails (a failing check names what is wrong).
"""
import os,sys,json,subprocess,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'stochastic'),str(R/'stochastic'/'auto')]
os.environ.setdefault('ROUTING_SCENARIO',str(R/'data/morning_00z/scenario.json'));os.environ.setdefault('FULL_PACK','1')
import numpy as np
import optimal_route as p
import bellman_fine as bf
FAIL=[];OK=[]
def check(name,cond,msg=''):
 (OK if cond else FAIL).append(name+('' if cond else f' : {msg}'))
 print(('ok   ' if cond else 'FAIL ')+name+('' if cond else f' : {msg}'),flush=True)

def t_finish():
 for st in (.25,.5): # grids in use: value tables 0.25 deg, gated family tables 0.5 deg
  las=np.arange(10,55+1e-9,st);los=np.arange(p.LON_MIN,2+1e-9,st);lo,la=np.meshgrid(los,las);m,toff=p.goal_target(la,lo,st)
  check(f'passage line on the {st} deg grid',m.sum()>=2,f'{int(m.sum())} nodes')
  check(f'passage line west of the mark ({st} deg)',bool(m.any()) and (lo[m]<p.MARK_NW[1]).all() and (np.abs(la[m]-16.25)<=st/2+1e-6).all())
 check('mark: sea just west of it is allowed',bool(p.allowed(np.array([p.MARK_NW[0]]),np.array([p.MARK_NW[1]-.1]))[0]))
 check('forbidden zone between the islands is forbidden',bool(p.in_zone(np.array([16.40]),np.array([-61.60]))[0]) and not p.allowed(np.array([16.40]),np.array([-61.60]))[0])
 check('open sea west of Guadeloupe is not in the zone',not p.in_zone(np.array([16.30]),np.array([-62.20]))[0])
 import finish_local as fl
 la,lo,ok=fl.grid();fin=p.distance(la,lo,p.FINISH)<fl.FIN_R
 check('finish reachable in the fine grid (sea cells within 1.5 NM)',bool((fin&ok).any()))
 check('fine grid covers the passage line',fl.LAS[0]<=16.0 and fl.LAS[-1]>=16.5-1e-9 and fl.LOS[0]<=-62.8)

def t_boat():
 v=float(p.polar(np.array([140.]),np.array([20.]),np.array([1]))[0]);check('polar Spi 140 deg 20 kn plausible',15<v<50,f'{v:.1f} kn')
 v1=float(p.polar(np.array([140.]),np.array([12.]),np.array([1]))[0]);check('polar: more wind, more speed (12 -> 20 kn)',v>v1)
 _,pen=p.maneuver(np.array([20.]),85.,np.array([False]),np.array([True]));check('gybe costs time',float(pen[0])>0)
 _,pen0=p.maneuver(np.array([20.]),85.,np.array([False]),np.array([False]));check('no maneuver costs nothing',float(pen0[0])==0)
 check('standard winch: 50 % speed lost during a penalty',abs(p.PEN_LOSS-.5)<1e-9 or os.environ.get('PACK_GEAR')=='1',f'PEN_LOSS {p.PEN_LOSS}')

def t_mail():
 import auto_update as a
 now=datetime.datetime(2026,11,6,8,0,tzinfo=datetime.timezone.utc)
 P=dict(orders=[['12:00',-130.],['13:19',-120.]],E=46.8);lat,lon=27.3,-59.5
 s,b=a.short_mail('t',now,lat,lon,-135.,'N',P,[],None,31)
 check('mail: same tack from bâbord = 0 gybe',any(l=='Empannages : 0.' for l in b),[l for l in b if 'Empann' in l])
 check('mail: no program entered -> CHANGER',' CHANGER' in s and 'A VERIFIER' not in s,s)
 s,b=a.short_mail('t',now,lat,lon,-135.,'N',P,[],.01,31);check('mail: gain 0.6 min -> RIEN A CHANGER','RIEN A CHANGER' in s,s)
 s,b=a.short_mail('t',now,lat,lon,-135.,'N',P,[],.05,31);check('mail: gain 3 min -> CHANGER (threshold 2 min)',' CHANGER' in s and 'RIEN' not in s,s)
 P2=dict(orders=[['08:30',140.],['09:30',-132.]],E=49.);s,b=a.short_mail('t',now,lat,lon,-137.,'N',P2,[],None,31)
 check('mail: gybe and back = 2 gybes',any(l=='Empannages : 2.' for l in b))
 check('guard: 122 h for ~700 NM is flagged',bool(a.sanity(dict(E=121.77),[],31.3,-56.1)))
 check('guard: 46.8 h for ~690 NM passes',not a.sanity(dict(E=46.8),[],lat,lon),a.sanity(dict(E=46.8),[],lat,lon))
 check('guard: +202 h margin is flagged',bool(a.sanity(dict(E=49.),['08:30 -1.0h:+202.69 +0.5h:+0.27'],28.1,-58.5)))
 P_=a.PARIS;g=lambda h,m:a.user_gap(datetime.datetime(2026,11,6,h,m,tzinfo=P_))
 check('next update: 3 h by day',abs(g(9,40)-3)<1e-6 and abs(g(16,45)-3)<1e-6)
 check('next update: evening program lasts until 9:00',abs(g(22,45)-10.25)<1e-6 and abs(g(20,30)-12.5)<1e-6 and abs(g(0,10)-8.8333)<1e-3)
 check('guard: night margin +19.5 h with an 11 h hold passes',not a.sanity(dict(E=31.),['23:30 -1.0h:+19.48 +0.5h:+4.40'],26.45,-65.27,11.1))
 check('guard: night +202 h still flagged',bool(a.sanity(dict(E=31.),['23:30 -1.0h:+202.0'],26.45,-65.27,11.1)))
 check('family program off once past its gate (7/11: 120 Spi all night instead of the Jib)',not a.family_active(dict(family='N',family_gate=dict(lon=-58.,lat=24.)),-63.2) and a.family_active(dict(family='N',family_gate=dict(lon=-58.,lat=24.)),-50.))
 check('no family once the gate is passed (state family empty)',not a.family_active(dict(family='',family_gate=dict(lon=-58.,lat=24.)),-50.))
 fc=a.finish_cap('2026110700',datetime.datetime(2026,11,7,9,45,tzinfo=datetime.timezone.utc),20.45,-62.97) if (R/'data/ens_2026110700/bellman_fine.npz').exists() else 12.
 check('horizon capped before the finish (7/11: 14 h with ~18 h to go incl. calm approach)',3<=fc<=17,f'{fc:.1f} h')
 if (R/'data/ens_2026110700/gfs3h.npz').exists():
  g=a.sail_guard('2026110700',datetime.datetime(2026,11,7,9,45,tzinfo=datetime.timezone.utc),20.45,-62.97,dict(angles=[68,60,140],sail_names=['Spi','Jib','Spi']))
  check('sail guard: Spi at 68 deg replaced (7/11 mail), Jib at 60 and Spi at 140 kept',g==['Jib','Jib','Spi'],str(g))
 check('guard: normal margins pass',not a.sanity(dict(E=49.),['12:00 -1.0h:+0.16 +0.5h:+1.82'],28.1,-58.5))
 n_=datetime.datetime.now(a.UTC).replace(second=0,microsecond=0);fm=lambda m:(n_+datetime.timedelta(minutes=m)).strftime('%Y%m%d%H%M')
 ps=lambda pr:round((a.program_start({'program':pr})-n_).total_seconds()/60)
 check('program start: an entered order within the next 20 min is re-optimized (start before it)',
  ps([])==20 and ps([[fm(10),-140]])==8 and ps([[fm(3),-140]])==5 and ps([[fm(40),-140]])==20,[ps([]),ps([[fm(10),-140]]),ps([[fm(3),-140]])])

def t_tables(cyc,S):
 ens=R/'data'/f'ens_{cyc}';c0=datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=datetime.timezone.utc)
 lead=max(6.,(datetime.datetime.now(datetime.timezone.utc)-c0).total_seconds()/3600)
 boat=(float(S['lat']),float(S['lon']))
 path=[boat,(22.,-61.),(18.5,-62.),(16.25,-62.25)] # boat -> Caribbean -> Antigua latitude -> passage line
 if boat[0]<23.5 or boat[1]<-62.5:path=[boat,((boat[0]+18.5)/2,(boat[1]-62.)/2),(18.5,-62.),(16.25,-62.25)] # 7/11: boat already west/south of (22,-61), which then lies off the route (false alarm)
 # Table box (optimal_route LAT_LO/LAT_HI/LON_HI, cropped 6/11): boat, other boats and finish well inside (a route
 # leaving the box would hit a 2000 h wall). Margin 3 deg (~6 h of sailing) on the open sides.
 pts=[boat]+[(b_[0],b_[1]) for b_ in S.get('boats',[])]+[tuple(p.FINISH)]
 check('table box contains boat, other boats and finish with 3 deg margin',all(p.LAT_LO+3<=x<=p.LAT_HI-3 and y<=p.LON_HI-3 for x,y in pts),
  f'box {p.LAT_LO}-{p.LAT_HI}N to {p.LON_HI}E, points {pts}')
 mem=sorted(f.name[:-len('3h.npz')] for f in ens.glob('gefs*3h.npz'))
 pick=[m for m in (mem[0],mem[len(mem)//2],mem[-1])] if mem else []
 tabs=[('gfs',ens/'bellman_fine.npz')]+[(f'member {m}',ens/'bellman_members'/f'{m}.npz') for m in pick]
 fam=S.get('family')
 if fam in ('N','S'):tabs+=[(f'gate {fam} {m}',ens/'bellman_gates'/fam/f'{m}.npz') for m in pick]
 for name,f in tabs:
  if not f.exists():
   if name.startswith('member') and not any((ens/'bellman_members').glob('*.npz')):continue # fast path: member tables come later
   check(f'{name} table exists ({cyc})',False,str(f));continue
  G=dict(np.load(f))
  for L in (lead,lead+12):
   v=[float(bf.value(G,L,np.array([a]),np.array([o]),np.array([1]),np.array([-1 if o<-61.9 else 1]))[0]) for a,o in path]
   ok=all(np.isfinite(v)) and v[0]<300 and all(x>y for x,y in zip(v,v[1:]))
   check(f'{name} lead {L:.0f}: V decreases towards the finish',ok,' > '.join(f'{x:.1f}' for x in v))
   check(f'{name} lead {L:.0f}: V on the passage line plausible (0.5-20 h)',.5<=v[-1]<=20,f'{v[-1]:.2f} h')
  la,lo=np.meshgrid(np.arange(18,32,1.),np.arange(-62,-50,1.),indexing='ij');m=p.allowed(la,lo)
  V=bf.value(G,lead,la[m],lo[m],np.ones(int(m.sum()),int),np.ones(int(m.sum()),int));V=V[V<1000]
  check(f'{name}: V not constant over the Atlantic',V.size>10 and V.std()>2,f'std {V.std():.2f}' if V.size else 'no value')
  # West edge of the domain must not attract the routes (6/11: routes hugged 64.5 W because the domain stopped at 65 W
  # while the fleet's west option went to 66 W). V must not keep decreasing towards the edge.
  lat_e=np.arange(18,30.1,1.);lon0=float(G['lon'][0]);g0=np.full(lat_e.shape,lon0+.5);g1=g0+1.5;ok_e=p.allowed(lat_e,g0)&p.allowed(lat_e,g1)
  if ok_e.any():
   v0=bf.value(G,lead,lat_e[ok_e],g0[ok_e],np.ones(int(ok_e.sum()),int),np.ones(int(ok_e.sum()),int))
   v1=bf.value(G,lead,lat_e[ok_e],g1[ok_e],np.ones(int(ok_e.sum()),int),np.ones(int(ok_e.sum()),int))
   bad=[f'{a:.0f}N' for a,x,y in zip(lat_e[ok_e],v0,v1) if x<y-.3 and x<400]
   check(f'{name}: west edge of the domain ({lon0:g}) not attracting the route',not bad,'V still decreasing at the edge at '+', '.join(bad))
 fl=ens/'finish_local'
 if fl.exists():
  n=len(list(fl.glob('*.npz')));check('finish tables: one per scenario + GFS',n==len(mem)+1,f'{n} for {len(mem)+1}')
  import finish_local as F
  G=dict(np.load(fl/'gfs.npz'));W=bf.Field(ens/'gfs3h.npz');P=F.descend(G,W,lead+24,16.25,-62.25,1,1)
  d=float(p.distance(np.array([P[-1][1]]),np.array([P[-1][2]]),p.FINISH)[0])
  check('fine finish: from the line to the finish',d<F.FIN_R+.5 and 1<P[-1][0]-P[0][0]<12,f'end {d:.1f} NM from finish after {P[-1][0]-P[0][0]:.1f} h')

def t_program(cyc,S):
 import auto_update as a
 now=datetime.datetime.now(datetime.timezone.utc).replace(second=0,microsecond=0)+datetime.timedelta(minutes=10)
 lat,lon,e,a0,s0=a.dead_reckon(S,cyc,now) if S.get('obs_utc') else (S['lat'],S['lon'],100,S['angle'],1)
 o=subprocess.run([sys.executable,str(R/'stochastic/family_prog.py'),cyc,now.strftime('%Y%m%d%H%M'),f'{lat:.5f}',f'{lon:.5f}',str(int(a0)),S['family'],f'{a.user_gap(now):.2f}'],
  cwd=R,env={**os.environ,'OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=900)
 L={k:[l[len(k)+1:] for l in o.stdout.splitlines() if l.startswith(k+' ')] for k in ('PROG','MARGIN')}
 if not L['PROG']:check('family_prog runs',False,o.stderr[-500:]);return
 P=json.loads(L['PROG'][0]);angs=[a0]+[x for _,x in P['orders']]
 hm=lambda t:int(t[:2])*60+int(t[3:]);ts=[now.astimezone(datetime.timezone(datetime.timedelta(hours=1))).strftime('%H:%M')]+[t for t,_ in P['orders']]
 gyt=[hm(ts[i+1]) for i in range(len(angs)-1) if (angs[i]>0)!=(angs[i+1]>0)]
 zz=[(x,y) for x,y in zip(gyt,gyt[1:]) if 0<=(y-x)%1440<60]
 check('program: no gybe then gybe back within 1 h (zigzag)',not zz,P['desc'])
 bad=a.sanity(P,L['MARGIN'],lat,lon,a.user_gap(now));check('program: plausibility guard',not bad,'; '.join(bad))
 end=now+datetime.timedelta(hours=a.user_gap(now));endp=end.astimezone(datetime.timezone(datetime.timedelta(hours=1))).strftime('%H:%M')
 check('program: at most 3 programmed orders',len(P['orders'])<=3,P['desc'])
 check('program: every order before the next update (no order the next mail will replace)',
  all((hm(t)-hm(ts[0])+10)%1440<=(hm(endp)-hm(ts[0])+10)%1440 for t,_ in P['orders']),f"next update {endp}: {P['desc']}")
 print('     program',P['desc'],f"E {P['E']:.2f} h")

if __name__=='__main__':
 args=[x for x in sys.argv[1:] if not x.startswith('--')]
 S=json.loads((R/'stochastic/auto/state.json').read_text());cyc=args[0] if args else S['last_cycle']
 for t in (t_finish,t_boat,t_mail):t()
 if (R/'data'/f'ens_{cyc}').exists():t_tables(cyc,S)
 if '--fast' not in sys.argv and S.get('family') in ('N','S') and (R/'data'/f'ens_{cyc}'/'bellman_gates'/'gate.json').exists():t_program(cyc,S)
 print(f'\n{len(OK)} ok, {len(FAIL)} FAIL'+(':\n- '+'\n- '.join(FAIL) if FAIL else ''))
 sys.exit(1 if FAIL else 0)
