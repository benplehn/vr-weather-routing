# Automatic update at each new bulletin (point 2).
# Usage: auto_update.py [--force CYCLE] [--no-notify]
# 1. Finds the newest cycle fully on NOMADS (GFS f198 + GEFS p30 f120); exits if it is not newer than state.json last_cycle.
# 2. Downloads it, builds the GFS fine value and one value table per GEFS member (6 in parallel, ~30 min).
# 3. Dead-reckons the boat from the last observation with the program entered in the game (new GFS, auto sail).
# 4. Runs optimize_pack + refine_pack (auto sail, pack end, per-member terminal value) from that estimated state, up to
#    night_until_local next morning (or 12 h), the entered program being evaluated as seed 0.
# 5. Writes stochastic/auto/REPORT_CYCLE.md (+ .json); macOS notification only if the gain exceeds GAIN_NOTIFY hours
#    or the night is flagged risky. Never touches the game.
import os,sys,json,subprocess,datetime,urllib.request,re
from pathlib import Path
import numpy as np
os.environ['FULL_PACK']='1'
A=Path(__file__).resolve().parent;R=A.parents[1];sys.path.insert(0,str(R))
PY=sys.executable;STATE=A/'state.json';GAIN_NOTIFY=float(os.environ.get('GAIN_NOTIFY','0.3'))
UTC=datetime.timezone.utc;PARIS=datetime.timezone(datetime.timedelta(hours=1)) # CET (no DST in November)
def log(*a):
 line=datetime.datetime.now(UTC).strftime('%H:%MZ ')+' '.join(str(x) for x in a);print(line,flush=True)
 with open(A/'auto_update.log','a') as f:f.write(line+'\n')
def listing(url):
 try:return urllib.request.urlopen(url,timeout=30).read().decode()
 except Exception:return ''
def complete(cyc):
 day,hh=cyc[:8],cyc[8:]
 g=listing(f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{day}/{hh}/atmos/')
 e=listing(f'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{day}/{hh}/atmos/pgrb2ap5/')
 return f'gfs.t{hh}z.pgrb2.0p25.f198' in g and f'gep30.t{hh}z.pgrb2a.0p50.f120' in e
def newest():
 t=datetime.datetime.now(UTC).replace(minute=0,second=0,microsecond=0)
 t-=datetime.timedelta(hours=t.hour%6)
 for k in range(4):
  c=(t-datetime.timedelta(hours=6*k)).strftime('%Y%m%d%H')
  if complete(c):return c
def run(cmd,env=None,out=None):
 log('RUN',' '.join(cmd[1:3]))
 with open(out or os.devnull,'w') as f:subprocess.run(cmd,cwd=R,env={**os.environ,**(env or {})},stdout=f,stderr=subprocess.STDOUT,check=True)
def gfs_table_cmd(ens):
 # C2F=1 : table GFS grossier (1°) puis fin (0,25°) dans le couloir des routes depuis la dernière position observée
 # (bellman_c2f.py ; validé 7/11 sur la course : valeurs et route identiques, 2,3x plus rapide). Sinon table pleine.
 if os.environ.get('C2F','1')=='1':
  S=json.loads(STATE.read_text());return [PY,'bellman_c2f.py',str(ens),'6',f"{S['lat']:.4f}",f"{S['lon']:.4f}"]
 return [PY,'bellman_fine.py',str(ens)]
def tables(cyc):
 ens=R/'data'/f'ens_{cyc}'
 if not (ens/'bellman_fine.npz').exists():run(gfs_table_cmd(ens),{'OMP_NUM_THREADS':'1','FINE_WORKERS':os.environ.get('PARALLEL','6')}) # GFS table alone: angle-parallel
 members=[p.name[:-len('3h.npz')] for p in sorted(ens.glob('gefs*3h.npz'))]
 todo=[m for m in members if not (ens/'bellman_members'/f'{m}.npz').exists()]
 if todo:
  log('member tables',len(todo))
  subprocess.run(['xargs','-P',os.environ.get('PARALLEL','6'),'-I{}',PY,'bellman_fine.py',str(ens),'.25','1','{}'],input='\n'.join(todo).encode(),cwd=R,
   env={**os.environ,'OMP_NUM_THREADS':'1'},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
def finish_tables(cyc):
 # Fine end-of-course tables (finish_local.py: mark, west coast, south of Basse-Terre, line), one per column, built
 # before any value table of the cycle (bellman_fine.py reads them as the value of its passage line). ~6 min on 16 cores.
 ens=R/'data'/f'ens_{cyc}';names=['gfs']+[x.name[:-len('3h.npz')] for x in sorted(ens.glob('gefs*3h.npz'))]
 todo=[n for n in names if not (ens/'finish_local'/f'{n}.npz').exists()]
 if not todo or os.environ.get('FINISH_LOCAL','1')!='1':return
 L=max(6.,(datetime.datetime.now(UTC)-datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC)).total_seconds()/3600)
 log('finish tables',len(todo),'from lead',round(L,1))
 for attempt in (1,2): # one retry of the missing tables before giving up (the skipper wants the fine finish)
  todo=[n for n in names if not (ens/'finish_local'/f'{n}.npz').exists()]
  if not todo:break
  r=subprocess.run(['xargs','-P',os.environ.get('PARALLEL','6'),'-I{}',PY,'finish_local.py',str(ens),'{}',f'{L:.1f}'],input='\n'.join(todo).encode(),cwd=R,
   env={**os.environ,'OMP_NUM_THREADS':'1'},stdout=open(A/f'finish_{cyc}.log','a'),stderr=subprocess.STDOUT)
  if r.returncode:log('finish tables attempt',attempt,'exit',r.returncode)
 try:
  miss=[n for n in names if not (ens/'finish_local'/f'{n}.npz').exists()]
  if miss:raise RuntimeError(f'missing {miss}')
 except Exception as x: # all or nothing: the value tables of one cycle must share the same finish model
  log('finish tables failed',x);import shutil;shutil.rmtree(ens/'finish_local',ignore_errors=True)
 # The GFS table built during the prefetch (before the finish tables) used the fixed finish estimate: rebuild it now
 # (~2 min, angle-parallel), the gated and member tables start from it past the member horizon.
 G=ens/'bellman_fine.npz';F=ens/'finish_local'/'gfs.npz'
 if G.exists() and F.exists() and G.stat().st_mtime<F.stat().st_mtime:
  log('gfs table rebuilt with the fine finish');G.unlink()
  try:run(gfs_table_cmd(ens),{'OMP_NUM_THREADS':'1','FINE_WORKERS':os.environ.get('PARALLEL','6')})
  except Exception as x:log('gfs table rebuild failed (tables() retries)',x)
def t_utc(s):return datetime.datetime.strptime(s,'%Y%m%d%H%M').replace(tzinfo=UTC)
def dead_reckon(S,cyc,now):
 # Boat from the last observation to now with the entered program, new GFS, auto sail (pack sails until pack end).
 import optimal_route as p,bellman_fine as bf
 c0=datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC)
 sys.path.insert(0,str(R/'stochastic'));import game_wind
 F=game_wind.fields(R/'data'/f'ens_{cyc}',['gfs'],cyc)[0] # what the game actually blew: previous GFS, then this one
 t0=t_utc(S['obs_utc']);pend_end=t_utc(S['pack_end_utc']);prog=[(t_utc(t),a) for t,a in S['program']]
 la,lo,e=np.array([S['lat']]),np.array([S['lon']]),float(S['energy']);ta=1 if S['angle']>0 else -1;sa=S['sail'] if (t0<pend_end or S['sail'] in (0,1)) else 1;pend=0.;dt=1/60
 t=t0;a=S['angle']
 while t<now:
  for tp,ap in prog:
   if tp<=t:a=ap
  ws,wd=F((t-c0).total_seconds()/3600,la,lo);nt=1 if a>0 else -1
  if os.environ.get('AUTO_SAIL','0')=='1':
   av=[1,4,5,6,0] if t<pend_end else [1,0];s=av[int(np.argmax([p.polar(np.array([a]),ws,x)[0] for x in av]))]
  else:s=sa if (t<pend_end or sa in (0,1)) else 1 # manual sail: kept; pack sail falls back to Spi at pack end
  sp,pen=p.maneuver(ws,np.array([e]),s!=sa,nt!=ta);pend+=pen[0];slow=min(dt,pend)
  la,lo=p.move(la,lo,(wd-a)%360,p.polar(a,ws,s)*(dt-p.PEN_LOSS*slow))
  e=min(100,max(0,e-sp[0])+(dt-slow)*3600/p.recover_seconds(ws)[0]);ta,sa=nt,s;pend=max(0,pend-dt);t+=datetime.timedelta(hours=dt)
 return float(la[0]),float(lo[0]),e,a,sa
def night_risk(cyc,obs,lat,lon,H,best):
 # Risky night: a maneuver planned during the night, or calm (<6 kn) on the nominal track in >25 % of members.
 import optimal_route as p,bellman_fine as bf
 ens=R/'data'/f'ens_{cyc}';L0=(obs-datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC)).total_seconds()/3600
 night=[m for m,a,b in zip(best['minutes'],best['angles'],[None]+best['angles'][:-1]) if b is not None and np.sign(a)!=np.sign(b)
  and (obs+datetime.timedelta(minutes=m)).astimezone(PARIS).hour in (23,0,1,2,3,4,5,6,7,8)]
 calm=0;files=sorted(ens.glob('gefs*3h.npz'))
 for f in files:
  F=bf.Field(f)
  if any(F(L0+t,np.array([lat]),np.array([lon]))[0][0]<6 for t in np.arange(0,H,1.)):calm+=1
 return dict(night_gybes=night,calm_fraction=calm/len(files),risky=bool(night) or calm/len(files)>.25)
CHANGE_H=float(os.environ.get('CHANGE_H','0.03')) # change the entered program only when it gains more than this (h)
def vmap(S,cyc,obs,lat,lon):
 # Bellman heat map (stochastic/vmap.py): expected hours to the finish from every point, the boat and the other boats
 # reported in the state ('boats' [[lat,lon,name]], 'boats_utc', shown when reported less than 12 h before obs, moved
 # to now along the ideal route by vmap.py).
 try:
  b=S.get('boats') or [];age=(t_utc(obs)-t_utc(S['boats_utc'])).total_seconds()/3600 if S.get('boats_utc') else 99
  o=subprocess.run([PY,'stochastic/vmap.py',cyc,obs,f'{lat:.5f}',f'{lon:.5f}',str(A/f'vmap_{cyc}.png')],cwd=R,capture_output=True,text=True,timeout=600,
   env={**os.environ,'OMP_NUM_THREADS':'4','VMAP_BOATS':json.dumps(b if age<12 else []),'VMAP_BOATS_UTC':S.get('boats_utc','')}).stdout
  return next((l.split()[1] for l in o.splitlines() if l.startswith('VMAP_PNG ')),None)
 except Exception as x:log('vmap failed',x);return None
def user_gap(start):
 # Hours from the program start to the skipper's next update (programs cover exactly that, the angle is held after the
 # last order): USER_GAP_H (3) by day; when that ends in the night (23:00-08:00 Paris), until 9:00 the next morning.
 loc=start.astimezone(PARIS);end=loc+datetime.timedelta(hours=float(os.environ.get('USER_GAP_H','3')))
 night=lambda d:d.hour>=23 or d.hour<8
 if night(loc) or night(end):
  m=loc.replace(hour=9,minute=0,second=0,microsecond=0)
  if m<=loc:m+=datetime.timedelta(days=1)
  end=m
 return max(1.,(end-loc).total_seconds()/3600)
def selftest(cyc):
 # Logic checks of the cycle's tables (tests/check_logic.py --fast, ~10 s): FAIL lines, empty = OK. Run before mailing.
 try:
  o=subprocess.run([PY,'tests/check_logic.py',cyc,'--fast'],cwd=R,capture_output=True,text=True,timeout=300,env={**os.environ,'OMP_NUM_THREADS':'1'})
  bad=[l[5:] for l in o.stdout.splitlines() if l.startswith('FAIL ')]
  if o.returncode and not bad:bad=['tests interrompus : '+(o.stderr.strip().splitlines() or ['?'])[-1]]
  # tests anti-bogues du code (règles du jeu, vérification minute par minute, poids, mails), ~3 s
  u=subprocess.run([PY,'-m','unittest','tests.test_routing'],cwd=R,capture_output=True,text=True,timeout=300,env={**os.environ,'OMP_NUM_THREADS':'1'})
  if u.returncode:bad+=['tests du code en échec : '+' ; '.join(l for l in u.stderr.splitlines() if l.startswith(('FAIL:','ERROR:')))[:400]]
 except Exception as x:bad=[f'tests non exécutés : {x}']
 if bad:log('SELFTEST',cyc,bad)
 return bad
def sanity(P,margins,lat,lon,hrs=3.):
 """Automatic plausibility check of a program before it is mailed (list of problems, empty = OK). Catches broken value
 tables (6/11 0:13: no finish on the 0.5 deg grid, 122 h announced instead of ~50 h) and model walls in the margins."""
 import optimal_route as p
 bad=[];E=float(P["E"]);d=float(p.distance(np.array([lat]),np.array([lon]),np.array(p.FINISH))[0])
 v=d/E if E>0 else float('inf')
 if not np.isfinite(E) or not 9<=v<=40:bad.append(f"temps restant {E:.1f} h pour {d:.0f} NM à vol d'oiseau ({v:.1f} nds de moyenne) : invraisemblable")
 for m in margins:
  hh,*xs=m.split()
  for x in xs:
   k,c=x.split('h:')
   if not np.isfinite(float(c)) or float(c)>max(12.,2*hrs):bad.append(f"décaler l'ordre de {hh} de {k} h coûterait {c} h : valeur aberrante (mur ou terre dans le modèle)");break
 return bad
def short_mail(lab,now,lat,lon,a0,fam,P,margins,ref_d,n,control=None,risk=None):
 # Short mail (the skipper's view): what to enter or "nothing to change", sensitive order, gybes, night, control.
 # The full computation and the detailed report are unchanged (REPORT_*.md on the server).
 nm='nord' if fam=='N' else 'sud';sd=lambda a:f"{abs(a):.0f}° {'tribord' if a>0 else 'bâbord'}"
 orders=[f"maintenant : {sd(a0)}"]+[f"{t} : {sd(a)}" for t,a in P['orders']]
 angs=[a0]+[a for _,a in P['orders']];gy=sum(1 for x,y in zip(angs,angs[1:]) if (x>0)!=(y>0))
 change=ref_d is None or ref_d>CHANGE_H
 subj=f"Routage {lab} : "+(("CHANGER - "+' ; '.join(orders)) if change else "RIEN A CHANGER")
 b=[f"Route {nm} — bulletin {lab}, {n} scénarios. Position estimée {lat:.2f} N {abs(lon):.2f} W à {now.astimezone(PARIS):%H:%M}.",'']
 if change:b+=["À ENTRER (Spi, cadenas bleu) :"]+[f"- {x}" for x in orders]+([f"Gain vs ton programme actuel : {ref_d*60:.0f} min."] if ref_d is not None else [])
 else:b+=[f"GARDE TON PROGRAMME ACTUEL (le meilleur ne gagne que {max(ref_d,0)*60:.0f} min).","Optimum pour info : "+' ; '.join(orders)]
 sens=[]
 for m in margins:
  hh,*xs=m.split();c=[float(x.split('h:')[1]) for x in xs if x.split('h:')[0] in ('-0.5','+0.5')]
  if c and max(c)>0.05:sens.append(f"{hh} (±30 min → jusqu'à +{max(c)*60:.0f} min)")
 b+=['',f"Empannages : {gy}.","Ordre sensible à l'heure : "+(', '.join(sens) if sens else "aucun (±30 min sans perte notable).")]
 if risk is not None:
  why=([f"empannage de nuit prévu par l'optimiseur sans contrainte à {', '.join(map(str,risk['night_gybes']))} (revu au programme du soir)"] if risk.get('night_gybes') else [])+([f"calme <6 nds dans {risk['calm_fraction']:.0%} des scénarios"] if risk['calm_fraction']>.25 else [])
  b.append("Nuit : "+("À SURVEILLER : "+' ; '.join(why) if why else f"pas de risque (calme <6 nds dans {risk['calm_fraction']:.0%} des scénarios)")+".")
 if control:b.append(control)
 b.append(f"Temps restant espéré jusqu'à la ligne d'arrivée : {P['E']:.1f} h." if os.environ.get('FINISH_FINE')=='1' else
  f"Temps restant espéré jusqu'à la bouée NO de Basse-Terre : {P['E']:.1f} h (+ ~2,5 h jusqu'à la ligne).")
 bad=sanity(P,margins,lat,lon,user_gap(now))
 if bad:
  log('SANITY',lab,bad);subj='A VERIFIER - '+subj
  b=["⚠ CONTRÔLE AUTOMATIQUE : résultat suspect, ne pas appliquer sans vérification."]+[f"- {x}" for x in bad]+['']+b
 return subj,b
def family_gate_file(S):
 # Gate of the skipper's decision (state 'family_gate'), kept fixed while the family is decided: re-deriving it from each
 # bulletin's route families moved it (24 -> 25.5 N on 5/11 12Z) and produced a gybe/back-gybe only to clear the new wall.
 g=S.get('family_gate');f=A/'family_gate.json'
 if g:f.write_text(json.dumps(g));return f
 prev=R/'data'/f"ens_{S['last_cycle']}"/'bellman_gates'/'gate.json'
 return prev if prev.exists() else None
def program_start(S):
 # Start of the computed program: now + LEAD_MIN (time to read the mail and enter it), but before the first order already
 # entered in the game that would fall in that window, so that order is re-optimized too (6/11: an 18:30 gybe was taken
 # as done by a program starting at 18:34). Never less than 5 min from now.
 t0=datetime.datetime.now(UTC).replace(second=0,microsecond=0);st=t0+datetime.timedelta(minutes=int(os.environ.get('LEAD_MIN','20')))
 pend=[t_utc(t) for t,a in S.get('program',[]) if t0<t_utc(t)<=st]
 if pend:st=max(t0+datetime.timedelta(minutes=5),min(pend)-datetime.timedelta(minutes=2))
 return st
def finish_cap(cyc,now,lat,lon,margin=3.):
 # Horizon cap near the finish: the optimizer must stop before the line (the value table takes over there), i.e. the
 # expected time to go from the GFS table minus margin h, never under 3 h.
 import bellman_fine as bf
 G=dict(np.load(R/'data'/f'ens_{cyc}'/'bellman_fine.npz'));L=(now-datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC)).total_seconds()/3600
 v=min(float(bf.value(G,L,np.array([lat]),np.array([lon]),np.array([s]),np.array([t]))[0]) for s in (0,1) for t in (-1,1))
 return max(3.,v-margin) if np.isfinite(v) and v<400 else 14.
def sail_guard(cyc,now,lat,lon,best):
 # 7/11: the mail said "maintenant 68 deg, Spi" (Spi at 68 deg = 2 kn, Jib 28 kn). Each order's sail is checked with the
 # polar at its TWA in the GFS wind at the boat; a sail slower than 70 % of the best allowed one is replaced (logged).
 import optimal_route as p
 sys.path.insert(0,str(R/'stochastic'));import game_wind
 F=game_wind.fields(R/'data'/f'ens_{cyc}',['gfs'],cyc)[0];L=(now-datetime.datetime.strptime(cyc,'%Y%m%d%H').replace(tzinfo=UTC)).total_seconds()/3600
 ws,_=F(L,np.array([lat]),np.array([lon]));out=[]
 for a,n in zip(best['angles'],best['sail_names']):
  sp={k:float(p.polar(np.array([abs(float(a))]),ws,np.array([i]))[0]) for i,k in enumerate(p.SAIL_NAMES) if i in (0,1) or k==n}
  b=max(sp,key=sp.get)
  if sp.get(n,0)<.7*sp[b]:log('SAIL_GUARD',f'{abs(a):.0f} deg: {n} {sp.get(n,0):.1f} kn -> {b} {sp[b]:.1f} kn');n=b
  out.append(n)
 return out
def family_active(S,lon):
 # The decided route family only matters before its gate. Past it (7/11: boat at 63 W, gate 58 W) family_prog, which
 # searches the Spi at 130-150 deg only, kept the night on 120 Spi while the unconstrained optimizer wanted the Jib at
 # 77-100 deg (+3 to +10 h): past the gate the unconstrained program (all sails and angles) is the program.
 g=S.get('family_gate') or {};fam=S.get('family')
 if fam not in ('N','S'):return False
 return not ('lon' in g and lon<float(g['lon'])-1)
def fast_family(S,cyc,redo=False):
 # Route family decided (state 'family'): the family program first, ~15 min after the download. Gated tables of that
 # family only (gate of the previous full cycle), dead reckoning, family_prog.py (31 scenarios), mail. The full run
 # (member tables, tree, unconstrained control, map) follows and sends the complete mail.
 fam=S['family'];ens=R/'data'/f'ens_{cyc}';nm='nord' if fam=='N' else 'sud'
 if not family_active(S,float(S['lon'])):log('fast family: gate passed, unconstrained program only');return
 if (ens/'bellman_gates'/'gate.json').exists() and not redo:return # redo (--fast-redo CYCLE): new capture, gates already built
 prev=family_gate_file(S)
 if prev is None and not redo:log('fast family: no gate');return
 try:
  if not redo:run([PY,'stochastic/build_family_gates.py',str(ens),str(prev),fam,os.environ.get('PARALLEL','6')],{'OMP_NUM_THREADS':'1'},A/f'fastgates_{cyc}.log')
  now=program_start(S)
  lat,lon,e,a0,s0=dead_reckon(S,cyc,now);obs=now.strftime('%Y%m%d%H%M')
  past=[(t,a) for t,a in S['program'] if t_utc(S['obs_utc'])<t_utc(t)<=now] # orders assumed entered since the capture
  cap=f"{t_utc(S['obs_utc']).astimezone(PARIS):%H:%M}"
  hyp=(f"Position estimée depuis ta capture de {cap}"+(", EN SUPPOSANT que tu as entré : "+', '.join(f"{t_utc(t).astimezone(PARIS):%H:%M} {abs(a):.0f}° {'trib' if a>0 else 'bâb'}" for t,a in past)
   +" (ordres gardés tels quels, NON réoptimisés : le programme démarre après eux). Si ce n'est pas le cas, ce programme est faux : envoie une capture." if past else " (aucun ordre supposé depuis)."))
  ref=','.join(f"{t_utc(t).astimezone(PARIS):%H%M}:{a}" for t,a in S['program'] if t_utc(t)>now)
  o=subprocess.run([PY,'stochastic/family_prog.py',cyc,obs,f'{lat:.5f}',f'{lon:.5f}',str(int(a0)),fam,f'{user_gap(now):.2f}']+([ref] if ref else []),cwd=R,
   env={**os.environ,'OMP_NUM_THREADS':'1','FP_ENERGY':f'{e:.0f}'},capture_output=True,text=True,timeout=1800).stdout
  L={k:[l[len(k)+1:] for l in o.splitlines() if l.startswith(k+' ')] for k in ('PROG','MARGIN','REF','TRACK')}
  P=json.loads(L['PROG'][0]);sd=lambda a:f"{abs(a):.0f}° {'tribord' if a>0 else 'bâbord'}";n=len(list(ens.glob('gefs*3h.npz')))
  orders=[f"{t} : {sd(a)}, Spi" for t,a in P['orders']]
  rep=[f"# Route {nm} — programme rapide, bulletin {cyc[8:]}Z complet ({n} scénarios GEFS)",
   f"Programme calculé pour démarrer le {now.astimezone(PARIS):%d/%m à %H:%M} Paris, position estimée à cette heure {lat:.2f} N {abs(lon):.2f} W "
   f"(depuis ta dernière capture et les ordres entrés), énergie {e:.0f}.",'',
   f"**ROUTE {nm.upper()} ACTÉE — programme à entrer** (optimisé dans la famille {nm}, {n} scénarios, Spi) :",f"- maintenant : {sd(a0)}, Spi"]+[f'- {x}' for x in orders]
  rep+=[f"Ordre de {m.split()[0]} décalé seul : "+', '.join(f"{x.split('h:')[0]} h → {x.split('h:')[1]} h" for x in m.split()[1:]) for m in L['MARGIN']]
  if L['REF']:
   d=float(L['REF'][0].split()[0]);rep.append(f"Programme actuellement entré dans le jeu : {d:+.2f} h vs ce programme"+(" → écart négligeable, tu peux le garder." if d<=0.05 else " → à remplacer."))
  rep+=['',hyp]
  rep+=[f"Temps restant espéré route {nm} : {P['E']:.2f} h.",'',"Mail complet (carte, contrôle par l'optimiseur sans contrainte, classement) dans environ 1 h."]
  (A/f'REPORT_{cyc}f.md').write_text('\n'.join(rep)+'\n')
  rd=float(L['REF'][0].split()[0]) if L['REF'] else None
  subj,b=short_mail(f"{cyc[8:]}Z "+(f"recalcul capture {cap}" if redo else "rapide"),now,lat,lon,a0,fam,P,L['MARGIN'],rd,n);b=b[:3]+['',hyp]+b[3:]
  st=selftest(cyc)
  if st:
   if not subj.startswith('A VERIFIER'):subj='A VERIFIER - '+subj
   b=["⚠ TESTS DE LOGIQUE EN ÉCHEC (calcul suspect) :"]+[f"- {x}" for x in st]+['']+b
  mp=None # map of the decided family (gated tables only: scenario routes weighted, expected route, finish zoom)
  try:
   so=subprocess.run([PY,'stochastic/strategy.py',cyc,obs,f'{lat:.5f}',f'{lon:.5f}','1',str(1 if a0>0 else -1)],cwd=R,timeout=900,capture_output=True,text=True,
    env={**os.environ,'OMP_NUM_THREADS':'1','STRAT_FAST':'1','STRAT_SFX':'f','GATE_SIDES':fam,**({'STRAT_TRACK':L['TRACK'][0]} if L['TRACK'] else {})}).stdout
   mp=next((l[len('STRATEGY_MAP '):].strip() for l in so.splitlines() if l.startswith('STRATEGY_MAP ')),None)
   if not mp:log('fast family: no map')
  except Exception as x:log('fast family map failed',x)
  if '--no-notify' not in sys.argv:notify(subj,b[2],'\n'.join(b+['',"Cartes jointes : routes nord des 31 scénarios pondérées (route attendue, zoom arrivée) et carte de Bellman (heures restantes depuis chaque point).","Contrôle complet (optimiseur sans contrainte) dans le mail suivant (~1 h)."]),attach=[mp,vmap(S,cyc,obs,lat,lon)])
  log('fast family: mail sent',P['orders'])
 except Exception as x:log('fast family failed',x)
NAMES={'tree':'arbre','na48':'non clairvoyant 48 h','cv':'clairvoyant'}
SAILS_FR=dict(Jib='Foc (Jib)',Spi='Spi',Code0='Code 0',HeavyGnk='Spi lourd',LightGnk='Spi léger',Staysail='Trinquette',LightJib='Foc léger')
def notify(title,msg,body='',attach=None):
 # E-mail (NOTIFY_EMAIL, via the host's configured `mail`, GNU mailutils: -A attachment) when set, else a macOS notification.
 if os.environ.get('MAIL_DRY')=='1':(A/'mail_dry.txt').write_text(title+'\n\n'+(body or msg)+'\n');return # test: mail text only
 to=os.environ.get('NOTIFY_EMAIL')
 att=[x for x in (attach if isinstance(attach,(list,tuple)) else [attach]) if x and Path(x).exists()]
 if to:subprocess.run(['mail','-s',title]+[y for x in att for y in ('-A',str(x))]+[to],input=(body or msg).encode(),check=False)
 else:subprocess.run(['osascript','-e',f'display notification {json.dumps(msg)} with title {json.dumps(title)}'],check=False)
ENDGAME_NM=float(os.environ.get('ENDGAME_NM','500'))
def endgame(S,cyc,lab):
 """End of race (boat within ENDGAME_NM of the NW Basse-Terre mark): stochastic optimal control by direct simulation
 of every program to the finish line in all scenarios (stochastic/endgame.py). Replaces optimize/refine on value
 tables, route families and north/south strategy (7/11: table values near the islands drove the program east of
 Antigua; skipper: 'clear and correct, no assumption, stochastic'). Returns True when it handled the cycle."""
 import optimal_route as p
 now=program_start(S);lat,lon,e,a0,s0=dead_reckon(S,cyc,now)
 if os.environ.get('ENDGAME','1')!='1' or float(p.distance(np.array([lat]),np.array([lon]),p.MARK_NW)[0])>ENDGAME_NM:return False
 obs=now.strftime('%Y%m%d%H%M');log('endgame from',obs,round(lat,3),round(lon,3),'energy',round(e))
 cur=[[round((t_utc(t)-now).total_seconds()/60),a] for t,a in S.get('program',[]) if t_utc(t)>now]
 png=A/f'endgame_{cyc}.png'
 o=subprocess.run([PY,'stochastic/endgame.py',cyc,obs,f'{lat:.5f}',f'{lon:.5f}',f'{e:.0f}',str(int(a0)),str(int(s0)),json.dumps(cur)],cwd=R,
  env={**os.environ,'OMP_NUM_THREADS':'4','EG_MAP':str(png)},capture_output=True,text=True,timeout=3600)
 (A/f'endgame_{cyc}.log').write_text(o.stdout+o.stderr)
 r=json.loads(next(l[7:] for l in o.stdout.splitlines() if l.startswith('RESULT ')))
 subj,body=endgame_mail(r,now,lat,lon,e,lab)
 (A/f'REPORT_{cyc}.md').write_text(body+'\n');(A/f'REPORT_{cyc}.json').write_text(json.dumps(r,indent=1))
 if '--no-notify' not in sys.argv:notify(subj,subj,body,png)
 S2=json.loads(STATE.read_text());S2['last_cycle']=max(cyc,S2.get('last_cycle',''));STATE.write_text(json.dumps(S2,indent=1))
 log('endgame done',round(r['expected'],2),'h; entered',r['current']['expected'],'land',r['current']['land'])
 return True
def endgame_mail(r,now,lat,lon,e,lab):
 """Mail text of an endgame result (pure function of the result: tested in tests/check_logic.py)."""
 loc=lambda h:(now+datetime.timedelta(hours=h)).astimezone(PARIS).strftime('%H:%M')
 side=lambda x:f"{abs(int(x))}° {'tribord' if x>0 else 'bâbord'}"
 b=r['best'];c=r['current'];n=r['n']
 orders=[f"maintenant : {side(b['angles'][0])}, {SAILS_FR.get(['Jib','Spi'][b['sails'][0]],'')}"]+[f"{loc(t)} : {side(a)}, {SAILS_FR.get(['Jib','Spi'][s],'')}" for t,a,s in zip(b['times'],b['angles'][1:],b['sails'][1:])]
 gain=None if c['expected'] is None else c['expected']-r['expected']
 change=c['land']>0 or gain is None or gain>CHANGE_EG
 if c['land']:
  la=c.get('land_at') or {};cl=f"⚠ TON PROGRAMME ACTUEL TOUCHE LA TERRE dans {c['land']}/{n} scénarios (premier contact {loc(la.get('hours',0))}, {la.get('lat',0):.2f} N {abs(la.get('lon',0)):.2f} W)."
 else:cl=f"Programme entré dans le jeu : arrivée {loc(c['expected'])} ({c['expected']:.2f} h), aucune terre ; le nouveau programme gagne {gain:+.2f} h (meilleur dans {r['better']}/{n} scénarios)."
 head='CHANGER' if change else 'garder'
 L=[f"# Fin de course — bulletin {lab} : simulation directe jusqu'à la ligne, {n} scénarios",
  f"Départ du programme {now.astimezone(PARIS):%d/%m %H:%M} Paris, position estimée {lat:.2f} N {abs(lon):.2f} W, énergie {e:.0f}.",'',cl,'',
  (f"**{head.upper()} — programme à entrer** (Foc = Jib) :" if change else "**Garder le programme en cours.** Programme optimal pour info (gain < "+f"{CHANGE_EG*60:.0f} min) :")]
 L+=[f"- {o}" for o in orders]
 L+=['',f"Arrivée espérée {loc(r['expected'])} ({r['expected']:.2f} h) ; 10 % des scénarios avant {loc(r['p10'])}, 90 % avant {loc(r['p90'])}.",
  f"Terre : 0/{n} scénarios à moins de 3 milles des côtes (ni avec un ordre décalé de ±10 min). Énergie mini {r['min_energy']:.0f}."]
 for m,t in zip(r['margins'],b['times']):
  w=m['window'];land=[int(x) for x in m['land']]
  L.append(f"Ordre de {loc(t)} : sans perte >6 min de {w[0]} à +{w[1]} min"+(f" ; ⚠ terre si décalé de {', '.join(f'{x:+d}' for x in land)} min" if land else '')+'.')
 L+=['',"Méthode : chaque programme (jusqu'à 3 ordres, puis tenu) est navigué jusqu'à la ligne dans le vent du jeu de chaque scénario (GFS appliqué, puis 31 membres GEFS), avec virements/empannages, changements de voile et énergie du jeu ; terre interdite dans tous les scénarios ; on minimise le temps d'arrivée moyen. Pas de table de valeur, pas de nord/sud.",
  "Carte jointe : route du programme proposé (rose, heures de Paris) et du programme entré (gris)."]
 subj=f"Routage {lab} : {head} " + (f"({gain:+.1f} h)" if gain is not None else "(TERRE)") + f" — arrivée ~{loc(r['expected'])} — " + ' ; '.join(orders[:2])
 return subj,'\n'.join(L)
CHANGE_EG=float(os.environ.get('CHANGE_EG','0.15'))
# Beam MPC router (stochastic/beam_prog.py, bilan a posteriori 7/11 : faisceau 24 h sur le GFS du bulletin, réactualisé à
# chaque bulletin = 2,7 h de l'optimum clairvoyant sur la course ; les programmes courts sur table en perdaient ~10).
# BEAM=1 (défaut) : calculé à chaque mise à jour ; BEAM_MAIN=1 (défaut) : c'est le programme du mail, l'optimiseur
# stochastique (arbre de scénarios) devient le contrôle. Une famille nord/sud actée garde la priorité (faisceau en contrôle).
BEAM_MAIN=os.environ.get('BEAM_MAIN','1')=='1'
def beam_program(S,cyc,now,lat,lon,e,a0,s0,H,sfx=''):
 """Programme du routeur à faisceau depuis l'état estimé (None si échec, journalisé). Le programme entré est évalué aussi."""
 if os.environ.get('BEAM','1')!='1':return None
 cur=[[round((t_utc(t)-now).total_seconds()/3600,3),a] for t,a in S.get('program',[]) if t_utc(t)>now]
 # Démarrage à chaud (point 3) : le dernier programme du faisceau (heures UTC absolues), s'il a moins de 18 h.
 lastf=A/'beam_last.json';prev={}
 try:
  L=json.loads(lastf.read_text())
  if (now-t_utc(L['obs'])).total_seconds()<18*3600:prev={'BEAM_PREV':json.dumps(L['program'])}
 except Exception:pass
 try:
  o=subprocess.run([PY,'stochastic/beam_prog.py',cyc,now.strftime('%Y%m%d%H%M'),f'{lat:.5f}',f'{lon:.5f}',f'{e:.0f}',str(int(a0)),str(int(s0)),f'{H:.2f}',json.dumps(cur)],
   cwd=R,env={**os.environ,**prev,'OMP_NUM_THREADS':'4','NIGHT':os.environ.get('NIGHT','22:30-'+S.get('night_until_local','10:00'))},capture_output=True,text=True,timeout=1800)
  (A/f'beam_{cyc}{sfx}.log').write_text(o.stdout+o.stderr)
  r=json.loads(next(l[5:] for l in o.stdout.splitlines() if l.startswith('BEAM ')))
  (A/f'beam_{cyc}{sfx}.json').write_text(json.dumps(r,indent=1))
  lastf.write_text(json.dumps(dict(obs=now.strftime('%Y%m%d%H%M'),program=[[(now+datetime.timedelta(minutes=m)).strftime('%Y%m%d%H%M'),a,sl] for m,a,sl in r['orders']])));log('beam',round(r['E'],2),'h; entered',round(r.get('ref_E',float('nan')),2),r['orders'])
  return r
 except Exception as x:log('beam failed',x);return None
def stoch_program(cyc,now,lat,lon,e,a0,s0,H,sfx=''):
 """Contrôle stratégique : faisceau STOCHASTIQUE (beam_stoch.py) sur les membres du bulletin + ceux des LAGGED bulletins
 précédents (ensemble décalé, tables déjà calculées), filtre bayésien calibré (SCEN_T). Rejeu A/B 7/11 : à égalité avec le
 faisceau déterministe sur la course, 4x plus cher -> contrôle (désaccord = signal stratégique), pas programme principal."""
 if os.environ.get('BEAM_STOCH','1')!='1':return None
 try:
  o=subprocess.run([PY,'stochastic/beam_stoch.py',cyc,now.strftime('%Y%m%d%H%M'),f'{lat:.5f}',f'{lon:.5f}',f'{e:.0f}',str(int(a0)),str(int(s0)),f'{H:.2f}'],
   cwd=R,env={**os.environ,'OMP_NUM_THREADS':'4','BS_N':os.environ.get('BS_N','300'),'BS_K':os.environ.get('BS_K','40'),'LAGGED':os.environ.get('LAGGED','1')},capture_output=True,text=True,timeout=3600)
  (A/f'stoch_{cyc}{sfx}.log').write_text(o.stdout+o.stderr)
  r=json.loads(next(l[6:] for l in o.stdout.splitlines() if l.startswith('BEAMS ')))
  (A/f'stoch_{cyc}{sfx}.json').write_text(json.dumps(r,indent=1));log('stoch',round(r['E'],2),'h',r['K'],'scénarios',r['n_lagged'],'décalés',r['orders'])
  return r
 except Exception as x:log('stoch failed',x);return None
def stoch_line(now,ST,BM):
 if not ST:return None
 loc=lambda m:(now+datetime.timedelta(minutes=m)).astimezone(PARIS).strftime('%H:%M')
 o=' ; '.join(f"{'maintenant' if m==0 else loc(m)} {abs(a)}° {'trib' if a>0 else 'bâb'} {s}" for m,a,s in ST['orders'])
 same=BM is not None and BM['orders'] and ST['orders'] and (BM['orders'][0][1]>0)==(ST['orders'][0][1]>0) and sum(1 for x,y in zip(ST['orders'],ST['orders'][1:]) if (x[1]>0)!=(y[1]>0))==sum(1 for x,y in zip(BM['orders'],BM['orders'][1:]) if (x[1]>0)!=(y[1]>0))
 return (f"Contrôle stratégique (faisceau stochastique, {ST['K']} scénarios GEFS dont {ST['n_lagged']} des bulletins précédents, filtre calibré) : {o}. "
  +("Même stratégie (amures et empannages)." if same else "⚠ STRATÉGIE DIFFÉRENTE (amure ou nombre d'empannages) : regarder avant d'entrer."))
def beam_mail(lab,now,lat,lon,e,B,control=None):
 """Mail court du routeur à faisceau (fonction pure du résultat). Retourne (sujet, lignes)."""
 loc=lambda m:(now+datetime.timedelta(minutes=m)).astimezone(PARIS).strftime('%H:%M')
 sd=lambda a:f"{abs(a):.0f}° {'tribord' if a>0 else 'bâbord'}"
 orders=[f"{'maintenant' if m==0 else loc(m)} : {sd(a)}, {SAILS_FR.get(s,s)}" for m,a,s in B['orders']]
 gain=B.get('ref_E',float('inf'))-B['E'];change=not np.isfinite(gain) or gain>CHANGE_H
 subj=f"Routage {lab} : "+(("CHANGER - "+' ; '.join(orders)) if change else "RIEN A CHANGER")
 b=[f"Routeur à faisceau (GFS {lab}, {B['horizon']:.0f} h, {B['max_orders']} ordres max, voile figée la nuit). Position estimée {lat:.2f} N {abs(lon):.2f} W à {now.astimezone(PARIS):%H:%M}, énergie {e:.0f}.",'']
 if change:b+=["À ENTRER (cadenas bleu) :"]+[f"- {x}" for x in orders]+([f"Gain vs ton programme actuel : {gain*60:.0f} min."] if np.isfinite(gain) else [])
 else:b+=[f"GARDE TON PROGRAMME ACTUEL (le meilleur ne gagne que {max(gain,0)*60:.0f} min).","Optimum pour info : "+' ; '.join(orders)]
 gy=sum(1 for x,y in zip(B['orders'],B['orders'][1:]) if (x[1]>0)!=(y[1]>0))
 b+=['',f"Empannages/virements : {gy}. Énergie mini {B['min_energy']:.0f}.",f"Temps restant estimé jusqu'à la bouée NO de Basse-Terre : {B['E']:.1f} h (+ ~2,5 h jusqu'à la ligne)."]
 if control:b.append(control)
 b+=[f"⚠ {w}" for w in B.get('warnings',[])]
 if B.get('verified') is False:subj='A VERIFIER - '+subj;b.insert(0,"⚠ Le programme n'a PAS passé la vérification minute par minute (terre ou marge côtière) : ne pas l'entrer sans contrôle.")
 bad=sanity(dict(E=B['E']),[],lat,lon)
 if bad:subj='A VERIFIER - '+subj;b=["⚠ CONTRÔLE AUTOMATIQUE : résultat suspect, ne pas appliquer sans vérification."]+[f"- {x}" for x in bad]+['']+b
 return subj,b
def main():
 S=json.loads(STATE.read_text())
 if '--fast-redo' in sys.argv: # manual: program + maps + mail now from the state's (new) capture, on the cycle's gated tables
  fast_family(S,sys.argv[sys.argv.index('--fast-redo')+1],redo=True);return
 force='--force' in sys.argv;quick='--quick' in sys.argv
 cyc=sys.argv[sys.argv.index('--force')+1] if force else newest()
 # --quick: night update on the tables of the last full cycle, game wind from the next GFS (GAME_NEXT, first 18 h).
 if quick:
  cyc=S['last_cycle'];nxt=(datetime.datetime.strptime(cyc,'%Y%m%d%H')+datetime.timedelta(hours=6)).strftime('%Y%m%d%H')
  os.environ['GAME_NEXT']=nxt;os.environ['SKIP_FETCH']='1';log('quick',cyc,'game wind',nxt)
 if not cyc:log('no complete cycle');return
 if not (force or quick) and cyc<=S['last_cycle']:log('no new cycle',cyc);return
 log('cycle',cyc)
 if os.environ.get('SKIP_FETCH')!='1':run([PY,'stochastic/fetch_ensemble.py',cyc,os.environ.get('GEFS_H','120')],out=A/f'fetch_{cyc}.log') # SKIP_FETCH=1: rerun on the same members
 if endgame(json.loads(STATE.read_text()),cyc,f"{cyc[8:]}Z"+(f" + GFS {os.environ['GAME_NEXT'][8:]}Z (rapide)" if quick else '')):return # end of race: direct simulation only
 if not quick:finish_tables(cyc)
 if (R/'data'/f'ens_{cyc}'/'finish_local'/'gfs.npz').exists():os.environ['FINISH_FINE']='1'
 if S.get('family') in ('N','S') and not quick:fast_family(S,cyc) # decided family: its program mailed first
 tables(cyc)
 # Scenario tree (information revealed at the next bulletins): primary terminal value when TREE=1 (default).
 use_tree=os.environ.get('TREE','1')=='1'
 if use_tree and not (R/'data'/f'ens_{cyc}'/'bellman_tree.npz').exists():
  run([PY,'bellman_tree.py',str(R/'data'/f'ens_{cyc}'),cyc,os.environ.get('PARALLEL','6')],{'OMP_NUM_THREADS':'1'},A/f'tree_{cyc}.log')
 # Fast non-anticipative bound: one root node over all members up to 48 h (same code as the tree, no split before 48 h).
 na48=use_tree and not S.get('family') # control model only useful while the route family is open
 if na48 and not (R/'data'/f'ens_{cyc}'/'bellman_na48.npz').exists():
  run([PY,'bellman_tree.py',str(R/'data'/f'ens_{cyc}'),cyc,os.environ.get('PARALLEL','6')],{'OMP_NUM_THREADS':'1','TREE_SPLITS':'[[48,0]]','TREE_OUT':'bellman_na48.npz'},A/f'na48_{cyc}.log')
 S=json.loads(STATE.read_text()) # re-read: capture/program entered during the long table build (6/11: mail from a 4 h old position)
 now=program_start(S)
 lat,lon,e,a0,s0=dead_reckon(S,cyc,now);obs=now.strftime('%Y%m%d%H%M')
 hh,mm=map(int,S.get('night_until_local','10:00').split(':'))
 end=now.astimezone(PARIS).replace(hour=hh,minute=mm)
 if end<=now.astimezone(PARIS):end+=datetime.timedelta(days=1)
 # Receding horizon (MPC): the program must cover at least until the next full result (next cycle + RESULT_DELAY h,
 # about 6.5 h after the bulletin) plus 1 h, so that the boat never sails on an unchecked last order before re-planning.
 dl=float(os.environ.get('RESULT_DELAY','6.5'));c=now.replace(hour=now.hour//6*6,minute=0)
 while c+datetime.timedelta(hours=dl)<=now+datetime.timedelta(minutes=30):c+=datetime.timedelta(hours=6)
 Hn=(c+datetime.timedelta(hours=dl+1)-now).total_seconds()/3600
 H=min(14.,max(4.,Hn,user_gap(now)+1)) # until the skipper's next update (7/11: the night end '10:00' was used by day -> 14 h, 4 orders held 6 h along the islands)
 H=min(H,finish_cap(cyc,now,lat,lon)) # 7/11: 14 h horizon with ~11 h to go: sailed on past the line (26 h, '181 deg')
 H=float(os.environ.get('HORIZON',H)) # manual override (urgent re-plan)
 log('horizon',round(H,2),'h, next full result',c.strftime('%d%HZ'))
 BM=beam_program(S,cyc,now,lat,lon,e,a0,s0,H,'q' if quick else '')
 ST=stoch_program(cyc,now,lat,lon,e,a0,s0,H,'q' if quick else '') if not quick else None # quick (nuit) : pas le temps
 pack_end=max(0.,(t_utc(S['pack_end_utc'])-now).total_seconds()/3600)
 env={'TERMINAL':'tree' if use_tree else 'member','TAIL_WEIGHT':os.environ.get('TAIL_WEIGHT','0'),'FLEET_TIE':os.environ.get('FLEET_TIE','0.15'),'GAME_WIND':os.environ.get('GAME_WIND','1'),'AUTO_SAIL':os.environ.get('AUTO_SAIL','0'),'PACK_END':f'{pack_end:.3f}','OMP_NUM_THREADS':'1'}
 tag=f'auto_{cyc}'+('q' if quick else '');sfx='q' if quick else ''
 lab=f"{cyc[8:]}Z"+(f" + GFS {os.environ['GAME_NEXT'][8:]}Z (rapide)" if quick else '')
 args=[cyc,obs,f'{lat:.5f}',f'{lon:.5f}',f'{e:.0f}',str(int(a0)),str(int(s0))]
 run([PY,'stochastic/optimize_pack.py',*args,f'pack_{tag}',f'{H:.2f}'],env,A/f'optimize_{cyc}.log')
 # Seed 0 = the program entered in the game, re-timed from now.
 cur=[(max(0.,(t_utc(t)-now).total_seconds()/3600),ang) for t,ang in S['program']]
 cur=[(0.,int(a0))]+[(t,ang) for t,ang in cur if t>0]
 seed=json.loads((R/'stochastic'/f'result_pack_{tag}.json').read_text())
 # Current sail kept (no change) until the pack ends, then Spi; seeds also the same program with each sail now.
 keep=[int(s0) if (t<pack_end or s0 in (0,1)) else 1 for t,_ in cur]
 seed['records']=[dict(family='current',times=[t for t,_ in cur],angles=[x for _,x in cur],sails=keep)]+[dict(family='current_sail',times=[t for t,_ in cur],angles=[x for _,x in cur],sails=[sv]+keep[1:]) for sv in ((1,0,4,5,6) if pack_end>0 else (1,0)) if sv!=keep[0]]+seed['records']
 sp=A/f'seed_{cyc}.json';sp.write_text(json.dumps(seed))
 # Three models in parallel (one process each): scenario tree (information at the next bulletins), non-anticipative
 # 48 h (never revealed) and clairvoyant (revealed at once). Then every model scores the best programs of all three and
 # the program kept is the one whose worst regret over the three models is smallest (decision robust to the model).
 terms=({'tree':'tree','na48':'na48','cv':'member'} if na48 else {'tree':'tree','cv':'member'}) if use_tree else {'cv':'member'}
 def par(jobs):
  ps=[subprocess.Popen(c,cwd=R,env={**os.environ,**e},stdout=open(o,'w'),stderr=subprocess.STDOUT) for c,e,o in jobs]
  for q in ps:
   if q.wait():raise RuntimeError(f'job failed {q.args}')
 res=lambda t:json.loads((R/'stochastic'/f'result_{t}.json').read_text())
 par([([PY,'stochastic/refine_pack.py',*args,f'refine_{tag}_{k}',str(sp),f'{H:.2f}','4'],{**env,'TERMINAL':v},A/f'refine_{k}_{cyc}.log') for k,v in terms.items()])
 R3={k:res(f'refine_{tag}_{k}') for k in terms}
 # Candidate 1 = the fleet program (best GFS-only program, what a deterministic router gives the fleet), for the rank objective.
 cand=[R3[next(iter(terms))]['seed0_program'],R3[next(iter(terms))]['herd_program']];keys=set()
 for k in terms:cand+=[dict(times=x['times'],angles=x['angles'],sails=x['sails']) for x in R3[k]['records'][:4]]
 uniq=[]
 for q in cand:
  kk=(tuple(round(t*60) for t in q['times']),tuple(q['angles']),tuple(q['sails']))
  if kk not in keys:keys.add(kk);uniq.append(q)
  if q is cand[1]:ih=[(tuple(round(t*60) for t in x['times']),tuple(x['angles']),tuple(x['sails'])) for x in uniq].index(kk)
 cp=A/f'cand_{cyc}.json';cp.write_text(json.dumps(dict(records=[dict(family='cand',**q) for q in uniq])))
 par([([PY,'stochastic/refine_pack.py',*args,f'eval_{tag}_{k}',str(cp),f'{H:.2f}','4'],{**env,'TERMINAL':v,'EVAL_ONLY':'1','FLEET_TIE':'0'},A/f'eval_{k}_{cyc}.log') for k,v in terms.items()])
 E3={k:res(f'eval_{tag}_{k}')['records'] for k in terms}
 J={k:np.array([x['robust'] for x in E3[k]]) for k in terms};reg={k:J[k]-J[k].min() for k in terms}
 worst=np.max([reg[k] for k in terms],axis=0);main=next(iter(terms))
 i=int(min(range(len(worst)),key=lambda i:(round(worst[i],2),J[main][i])))
 # Rank objective: the race is won on the fleet, not on the clock. Per scenario, gain vs the fleet program = fleet time
 # minus ours; among programs within RANK_TOL h (worst regret and main robust objective) of the best, keep the one with
 # the highest (weighted) probability of gaining at least GOAL_H on the fleet, then the farthest from the GFS route.
 tol=float(os.environ.get('RANK_TOL','0.2'));goal=float(os.environ.get('GOAL_H','0.3'))
 wts=np.array(res(f'eval_{tag}_{main}').get('weights') or np.ones(len(E3[main][0]['members'])));wts=wts/wts.sum()
 fleet=np.array(E3[main][ih]['members']);dfl=[fleet-np.array(x['members']) for x in E3[main]]
 pbeat=[float(wts@(d>=goal)) for d in dfl];gfl=[float(wts@d) for d in dfl]
 close=[j for j in range(len(worst)) if worst[j]<=worst[i]+tol and J[main][j]<=J[main][i]+tol]
 itime=i;i=max(close,key=lambda j:(round(pbeat[j],2),E3[main][j].get('fleet_gap_nm',0)))
 rank_line=(f"Objectif classement (gagner ≥{goal:.1f} h sur la flotte) : probabilité {pbeat[i]:.0%} pour le programme choisi"
  f" (programme en cours {pbeat[0]:.0%} ; meilleur temps pur {pbeat[itime]:.0%}) ; gain moyen sur la flotte {gfl[i]:+.2f} h.")
 mp=A/f'chosen_{cyc}.json';mp.write_text(json.dumps(dict(records=[dict(family='chosen',**uniq[i])])))
 run([PY,'stochastic/refine_pack.py',*args,f'margin_{tag}',str(mp),f'{H:.2f}','4'],{**env,'TERMINAL':terms[main],'EVAL_ONLY':'1','MARGINS':'1','FLEET_TIE':'0'},A/f'margin_{cyc}.log')
 best=dict(E3[main][i]);r=dict(seed0=E3[main][0],seed0_program=uniq[0],margins=res(f'margin_{tag}')['margins'])
 gain=J[main][0]-J[main][i] # robust objective (safety margin, point 5), main model
 consensus={k:float(reg[k][i]) for k in terms};current_reg={k:float(reg[k][0]) for k in terms}
 risk=night_risk(cyc,now,lat,lon,H,best)
 local=lambda m:(now+datetime.timedelta(minutes=m)).astimezone(PARIS).strftime('%H:%M')
 side=lambda x:f"{abs(x)}° {'tribord' if x>0 else 'bâbord'}"
 best['sail_names']=sail_guard(cyc,now,lat,lon,best)
 orders=[f"{'maintenant' if m==0 else local(m)} : {side(x)}, {SAILS_FR.get(n,n)}" for m,x,n in zip(best['minutes'],best['angles'],best['sail_names'])]
 if len(best['members'])==len(r['seed0']['members']):better=int((np.array(best['members'])<np.array(r['seed0']['members'])).sum())
 else:better=-1
 mapf=[None]
 def strategy_lines(build):
  # Strategic decision (stochastic/strategy.py): route families of the scenarios, cost of each choice (gated tables),
  # point of no return, whether the chosen program closes a family, rank per family, map (attached to the mail).
  try:
   cmd=[PY,'stochastic/strategy.py',cyc,obs,f'{lat:.5f}',f'{lon:.5f}',str(int(best['sails'][0])),str(1 if best['angles'][0]>0 else -1),
    '--program',f"{best['gfs_pos_H'][0]:.5f}",f"{best['gfs_pos_H'][1]:.5f}",str(int(best['sails'][-1])),str(1 if best['angles'][-1]>0 else -1),f'{H:.2f}']+(['--build'] if build else [])
   o=subprocess.run(cmd,cwd=R,env={**os.environ,'OMP_NUM_THREADS':'1','STRAT_SFX':sfx,'GATE_SIDES':S.get('family') if S.get('family') in ('N','S') else 'NS','PARALLEL':os.environ.get('STRAT_PARALLEL',os.environ.get('PARALLEL','14'))},
    capture_output=True,text=True,timeout=3600).stdout
   for l in o.splitlines():
    if l.startswith('STRATEGY_MAP '):mapf[0]=l[len('STRATEGY_MAP '):].strip()
   L=[l[len('STRATEGY '):] for l in o.splitlines() if l.startswith('STRATEGY ')]
   if S.get('no_strategy'):L=[x for x in L if not re.search(r'nord|sud|famille|stratégique',x,re.I)] # 7/11: skipper: no more north/south question
   return L
  except Exception as x:log('strategy failed',x);return []
 FI={}
 def family_lines():
  # Route family decided by the skipper (state 'family' N/S): program optimized inside that family on the gated tables
  # of this cycle (stochastic/family_prog.py), with the unconstrained program as reference. Returns (lines, track json).
  fam=S.get('family')
  if fam not in ('N','S') or not (R/'data'/f'ens_{cyc}'/'bellman_gates'/'gate.json').exists():return [],None
  if not family_active(S,lon):return [],None # gate passed: the unconstrained program is the headline
  ref=','.join(f"{t_utc(t).astimezone(PARIS):%H%M}:{a}" for t,a in S['program'] if t_utc(t)>now) # program entered in the game
  hrs=user_gap(now);nm='nord' if fam=='N' else 'sud'
  try:
   o=subprocess.run([PY,'stochastic/family_prog.py',cyc,obs,f'{lat:.5f}',f'{lon:.5f}',str(int(a0)),fam,f'{hrs:.2f}']+([ref] if ref else []),cwd=R,
    env={**os.environ,'OMP_NUM_THREADS':'1','FP_ENERGY':f'{e:.0f}'},capture_output=True,text=True,timeout=1800).stdout
   L={k:[l[len(k)+1:] for l in o.splitlines() if l.startswith(k+' ')] for k in ('PROG','ALT','MARGIN','REF','TRACK')}
   P=json.loads(L['PROG'][0]);sd=lambda a:f"{abs(a):.0f}° {'tribord' if a>0 else 'bâbord'}"
   out=[f"**ROUTE {nm.upper()} ACTÉE — programme à entrer** (optimisé dans la famille {nm}, {len(best['members'])} scénarios, Spi) :",
    f"- maintenant : {sd(a0)}, Spi"]+[f"- {t} : {sd(a)}, Spi" for t,a in P['orders']]
   out+=[f"Ordre de {m.split()[0]} décalé seul : "+', '.join(f"{x.split('h:')[0]} h → {x.split('h:')[1]} h" for x in m.split()[1:]) for m in L['MARGIN']]
   if L['REF']:out.append(f"Programme entré dans le jeu, évalué sur la route {nm} : {L['REF'][0].split(' (')[0].strip()} vs ce programme.")
   FI.update(P=P,margins=L['MARGIN'],ref=float(L['REF'][0].split()[0]) if L['REF'] else None)
   out.append(f"Temps restant espéré route {nm} : {P['E']:.2f} h"+(f" (si l'autre option s'avérait la bonne : {P['other']:.2f} h)." if P['other']<500 else " (l'autre option n'est plus jouable avec ce programme)."))
   return out,(L['TRACK'][0] if L['TRACK'] else None)
  except Exception as x:log('family program failed',x);return [],None
 def report():
  fl,track=family_lines()
  if track:os.environ['STRAT_TRACK']=track
  na_line=f'Contrôle des {len(terms)} modèles (perte vs le meilleur de chaque modèle, programme choisi / programme en cours) : '+' ; '.join(f"{NAMES[k]} {consensus[k]:.2f} / {current_reg[k]:.2f} h" for k in consensus)+'.'
  rep=[f'# Mise à jour automatique — bulletin {lab}',f'Programme calculé pour démarrer le {now.astimezone(PARIS):%d/%m à %H:%M} Paris (calcul lancé 20 min avant), position estimée à cette heure {lat:.2f} N {abs(lon):.2f} W, énergie {e:.0f} (estimation, pas de capture).',
   '',f'Programme en cours : objectif {r["seed0"]["objective"]:.2f} h ; meilleur : {best["objective"]:.2f} h ; **gain {gain:+.2f} h** (meilleur dans {better}/{len(best["members"])} scénarios).','',
   f"**Nouveau programme** (cadenas bleu ; voile à changer à la main à l'heure indiquée ; énergie mini {best['min_energy']:.0f}) :" if gain>GAIN_NOTIFY else 'Gain faible : **garder le programme en cours**. Programme optimal pour info :']+[f'- {o}' for o in orders]+['',
   na_line,f"Nuit à risque : {'OUI' if risk['risky'] else 'non'} (empannages de nuit : {', '.join(local(m) for m in risk['night_gybes']) or 'aucun'} ; scénarios avec calme <6 nds sur la position : {risk['calm_fraction']:.0%})."]
  if fl: # decided family first; the unconstrained optimum stays below for information
   rep=rep[:2]+['']+fl+['','## Programme sans contrainte de famille (pour info)']+rep[2:]
  if BM:
   bs,bb=beam_mail(lab,now,lat,lon,e,BM)
   head='## Routeur à faisceau — programme principal' if BEAM_MAIN and not fl else '## Routeur à faisceau (contrôle)'
   sl=stoch_line(now,ST,BM);bb=bb+([sl] if sl else [])
   rep=rep[:2]+['',head]+bb+['','## Optimiseur stochastique (arbre de scénarios)'+(' — contrôle' if BEAM_MAIN and not fl else '')]+rep[2:]
  rep.append(f"Écart à la route « flotte » (routeur GFS) au bout de {H:.0f} h : {best.get('fleet_gap_nm',0):.0f} NM.")
  rep.append(rank_line)
  rep+=strategy_lines(False)
  for m in r.get('margins',[]):
   if m['switch']=='all':rep.append(f"Tous les ordres décalés ensemble : sans perte notable de {m['window'][0]} à +{m['window'][1]} min.")
   else:
    w=f"{local(m['minute']+m['window'][0])}–{local(m['minute']+m['window'][1])}"
    ex=' ; '.join(x for x in [f"1 h trop tôt {m['early_60']:+.1f} h" if m['early_60'] is not None else '',f"1 h trop tard {m['late_60']:+.1f} h" if m['late_60'] is not None else ''] if x)
    rep.append(f"Ordre de {local(m['minute'])} : marge {w}"+(f" ({ex})" if ex else '')+'.')
  (A/f'REPORT_{cyc}{sfx}.md').write_text('\n'.join(rep)+'\n')
  (A/f'REPORT_{cyc}{sfx}.json').write_text(json.dumps(dict(cycle=cyc,now=obs,estimate=[lat,lon,e],gain=gain,better=better,best=best,current=r['seed0_program'],risk=risk,beam=BM,stoch=ST),indent=1,default=float))
  oc=' ; '.join(o for o in orders)
  if BM and BEAM_MAIN and not fl and '--no-notify' not in sys.argv:
   bg=BM.get('ref_E',float('inf'))-BM['E']
   diverge=bool(ST) and 'DIFFÉRENTE' in (stoch_line(now,ST,BM) or '')
   if bg>CHANGE_H or risk['risky'] or diverge or os.environ.get('NOTIFY_ALWAYS')=='1':
    ctl=f"Contrôle (optimiseur stochastique, {len(best['members'])} scénarios) : {oc} ({gain:+.2f} h vs ton programme, objectif robuste)."
    sl=stoch_line(now,ST,BM);ctl=ctl+('\n'+sl if sl else '')
    subj,b=beam_mail(lab,now,lat,lon,e,BM,ctl)
    if risk['risky']:subj+=' - NUIT A RISQUE'
    st=selftest(cyc)
    if st:
     if not subj.startswith('A VERIFIER'):subj='A VERIFIER - '+subj
     b=["⚠ TESTS DE LOGIQUE EN ÉCHEC (calcul suspect) :"]+[f"- {x}" for x in st]+['']+b
    notify(subj,b[2],'\n'.join(b+['',f"Rapport détaillé : stochastic/auto/REPORT_{cyc}{sfx}.md sur le serveur."]),[mapf[0],vmap(S,cyc,obs,lat,lon)])
   return
  if fl and '--no-notify' not in sys.argv:
   oc=' ; '.join(o.replace(', Spi','') for o in orders)
   ctl=(f"Contrôle (optimiseur sans contrainte) : son optimum = {oc}. Pas d'écart chiffré : il compare au programme entré tenu {H:.0f} h ({gain:+.0f} h), pas à celui-ci → comparer les ordres." if not np.isfinite(gain) or abs(gain)>12
    else f"Contrôle (optimiseur sans contrainte, arbre de scénarios) : d'accord ({max(gain,0)*60:.0f} min d'écart)." if gain<=CHANGE_H
    else f"Contrôle (optimiseur sans contrainte) : son optimum ({oc}) gagne {gain*60:.0f} min, sous le seuil d'alerte de {GAIN_NOTIFY*60:.0f} min → garder." if gain<=GAIN_NOTIFY
    else f"Contrôle (optimiseur sans contrainte) : ALERTE, son optimum ({oc}) gagne {gain*60:.0f} min → à regarder.")
   if BM:ctl+=f" Routeur à faisceau (sans famille) : {' ; '.join(f'{m} min {a}° {s}' for m,a,s in BM['orders'])}, {BM['E']:.1f} h restantes."
   subj,b=short_mail(lab,now,lat,lon,a0,S['family'],FI['P'],FI['margins'],FI['ref'],len(best['members']),ctl,risk)
   st=selftest(cyc)
   if st:
    if not subj.startswith('A VERIFIER'):subj='A VERIFIER - '+subj
    b=["⚠ TESTS DE LOGIQUE EN ÉCHEC (calcul suspect) :"]+[f"- {x}" for x in st]+['']+b
   notify(subj,b[2],'\n'.join(b+['',f"Cartes jointes : routes des scénarios et carte de Bellman. Rapport détaillé : stochastic/auto/REPORT_{cyc}{sfx}.md sur le serveur."]),[mapf[0],vmap(S,cyc,obs,lat,lon)])
  elif '--no-notify' not in sys.argv and (gain>GAIN_NOTIFY or risk['risky'] or os.environ.get('NOTIFY_ALWAYS')=='1'):
   act='CHANGER' if gain>GAIN_NOTIFY else 'garder'
   notify(f"Routage {lab} : {act} ({gain:+.1f} h){' - NUIT A RISQUE' if risk['risky'] else ''}",'; '.join(orders),'\n'.join(rep),mapf[0])
 # Family decided and full cycle: build this cycle's gated tables first, so that the single mail carries the family program.
 if S.get('family') in ('N','S') and not quick and not (R/'data'/f'ens_{cyc}'/'bellman_gates'/'gate.json').exists():
  gf=family_gate_file(S);log('strategy: building gated tables (family decided)',gf)
  if gf:run([PY,'stochastic/build_family_gates.py',str(R/'data'/f'ens_{cyc}'),str(gf),S['family'],os.environ.get('PARALLEL','6')],{'OMP_NUM_THREADS':'1'},A/f'fastgates_{cyc}.log')
  else:strategy_lines(True)
 report()
 # Full cycle with a north/south split but no gated tables yet: build them now (~6 min) and send the strategic decision.
 sj=A/f'strategy_{cyc}{sfx}.json'
 if not quick and not S.get('family') and not S.get('no_strategy') and sj.exists() and json.loads(sj.read_text()).get('gate') and 'V' not in json.loads(sj.read_text()):
  log('strategy: building gated tables');sl=strategy_lines(True)
  if sl and '--no-notify' not in sys.argv:
   notify(f"Routage {lab} : décision stratégique nord/sud",sl[0],'\n'.join([f'# Décision stratégique — bulletin {lab}',f'Position {lat:.2f} N {abs(lon):.2f} W, {now.astimezone(PARIS):%d/%m %H:%M} Paris.','']+[f'- {x}' for x in sl]+['','Carte jointe : routes optimales de chaque scénario (bleu nord, orange sud), porte, route d\'attente.']),mapf[0])
 # Re-read the state: a capture may have been saved while this run was going (it must not be overwritten).
 if not quick:S2=json.loads(STATE.read_text());S2['last_cycle']=cyc;STATE.write_text(json.dumps(S2,indent=1))
 log('done gain',round(gain,2),'risky',risk['risky'])
if __name__=='__main__':main()
