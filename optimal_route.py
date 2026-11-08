"""Experimental weather routing. All times relative to 2026-11-01 13:13 UTC.

Global Bellman relaxation supplies a heuristic; a hybrid beam DP retains sail,
tack and stamina. This is approximate dynamic programming, not a global proof.
"""
from pathlib import Path
import json,time,os,datetime,hashlib
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import shape, LineString
from shapely.ops import unary_union
from shapely import contains_xy

ROOT=Path(__file__).parent
DATA=ROOT/'data'
DEFAULT_SCENARIO=dict(case_name='',start_lat=48+57/60+33/3600,start_lon=-(2+38/60+43/3600),start_utc='2026-11-01T13:13:00+00:00',forecast_cycle_utc='2026-11-01T06:00:00+00:00',energy=100.,sail=0,tack=1,goal_lat=17.,goal_lon=-61.,weather_file='weather.npz')
SCENARIO=dict(DEFAULT_SCENARIO)
if os.environ.get('ROUTING_SCENARIO'):
 SCENARIO.update(json.loads(Path(os.environ['ROUTING_SCENARIO']).read_text()))
OUT=ROOT/'optimization'/SCENARIO['case_name']
OUT.mkdir(parents=True, exist_ok=True)  # run outputs (land masks, scenario fingerprint), not versioned
START=np.array([SCENARIO['start_lat'],SCENARIO['start_lon']])
# Target of the tables (see below), set here and not in SCENARIO so that the scenario fingerprint is unchanged.
GOAL=np.array([float(x) for x in os.environ.get('GOAL','16.25,-62.0').split(',')])
# Finish of the course (in game, 5/11): leave the NW Basse-Terre mark (16°22'03"N 61°46'51"W) passing west of it, down the
# west coast, round the south of Basse-Terre, up to the line (16°06'12"N 61°32'31"W). The tables end on a passage line
# west of the mark (latitude 16°15'N, from 61°53'W out to 62°48'W): the skipper is free to pass the mark wide. The value on
# that line is the extra time of the rest of the course from the crossing point (distance to the SW tip of Basse-Terre at
# TAIL_KN), so passing far out is paid for; the rest of the course from the closest crossing (~45 NM, ~2.5 h) is common.
MARK_NW=np.array([16.3675,-61.7808]);FINISH=np.array([16.1033,-61.5419]);TAIL_WP=np.array([15.90,-61.80]);FINISH_TAIL_H=2.5
TAIL_KN=float(os.environ.get('TAIL_KN','20'));GOAL_R=12.
def goal_target(lat,lon,step=.25):
 """(mask of the passage-line nodes, value there in hours) for the value tables. The line row is the grid row(s) within
 half a step of 16.25 N (a 0.5 deg grid has no 16.25 row: rows 16.0 and 16.5, both west of the mark, take it)."""
 lat,lon=np.broadcast_arrays(np.asarray(lat,float),np.asarray(lon,float))
 m=(np.abs(lat-16.25)<max(.13,step/2+1e-6))&(lon<=-61.88)&(lon>=-62.8)&allowed(lat,lon)
 d=distance(lat,lon,TAIL_WP);d0=float(distance(np.array([16.25]),np.array([-61.88]),TAIL_WP)[0])
 return m,np.where(m,np.maximum(0,d-d0)/TAIL_KN,0.)
# Forbidden zone of the game between Basse-Terre and Grande-Terre (Grand Cul-de-Sac Marin), from the in-game screenshots.
ZONE=np.array([(16.380,-61.765),(16.494,-61.469),(16.40,-61.45),(16.24,-61.50),(16.24,-61.66),(16.33,-61.72)])
def in_zone(lat,lon,Z=ZONE):
 lat,lon=np.broadcast_arrays(np.asarray(lat,float),np.asarray(lon,float));inside=np.zeros(lat.shape,bool);n=len(Z)
 for i in range(n):
  (y1,x1),(y2,x2)=Z[i],Z[(i+1)%n]
  c=((y1>lat)!=(y2>lat))&(lon<(x2-x1)*(lat-y1)/(y2-y1+1e-12)+x1);inside^=c
 return inside
INITIAL_ENERGY,INITIAL_SAIL,INITIAL_TACK=SCENARIO['energy'],SCENARIO['sail'],SCENARIO['tack']
START_LEAD=(datetime.datetime.fromisoformat(SCENARIO['start_utc'])-datetime.datetime.fromisoformat(SCENARIO['forecast_cycle_utc'])).total_seconds()/3600
# Boat polar (Virtual Regatta Ultim, JSON format of the ITYC export: twa, tws, sail[].speed[twa][tws]). Not distributed with
# this repository (third-party data): see README. POLAR_FILE overrides the path; tests use tests/fixtures/polar_synthetic.json.
POLAR_FILE=Path(os.environ.get('POLAR_FILE',DATA/'Ultim_BP_XI_current.json'))
if not POLAR_FILE.exists():
 raise FileNotFoundError(f'Polar file {POLAR_FILE} not found: download it (README, "Data") or set POLAR_FILE.')
P=json.loads(POLAR_FILE.read_text())
# Legacy single-bulletin weather file (6-hourly GFS), only used by wind()/global_bellman(); the routers read their own fields.
WEATHER_FILE=DATA/SCENARIO['weather_file']
if WEATHER_FILE.exists():
 with np.load(WEATHER_FILE) as weather_file:
  W={key:weather_file[key] for key in weather_file.files}
else:
 W={'lat':np.array([5.]),'lon':np.array([-75.]),'hours':np.array([6.,12.])}
fingerprint=hashlib.sha256(json.dumps(SCENARIO,sort_keys=True).encode()+(WEATHER_FILE.read_bytes() if WEATHER_FILE.exists() else b'')+POLAR_FILE.read_bytes()).hexdigest()
manifest=OUT/'scenario.json'
if manifest.exists() and json.loads(manifest.read_text()).get('fingerprint')!=fingerprint:
 raise RuntimeError('Scenario/data changed: use a fresh case_name to prevent reuse of stale Bellman values.')
_tmp=manifest.with_suffix(f'.{os.getpid()}.tmp');_tmp.write_text(json.dumps(dict(SCENARIO,fingerprint=fingerprint),indent=2));os.replace(_tmp,manifest) # atomic: parallel imports read it
LAT0,LON0=W['lat'][0],W['lon'][0]
TWAS=np.array(P['twa']);TWSS=np.array(P['tws'])
# FULL_PACK=1: all 7 sails (0 Jib,1 Spi,2 Staysail,3 LightJib,4 Code0,5 HeavyGnk,6 LightGnk), pro winch,
# furler, jacket, pug, per ITYC penalty/energy model (data/ityc_main.js: Vf, Uf, If, Qf).
FULL_PACK=os.environ.get('FULL_PACK')=='1'
SAIL_NAMES=[s['name'] for s in P['sail']]
POLAR=np.array([s['speed'] for s in (P['sail'] if FULL_PACK else P['sail'][:2])])
# Pack gear (pro winch, furler, jacket, pug) ended with the pack on 3/11 12:25Z: standard maneuvers (50 % speed during a
# penalty, confirmed in game 5/11 by the skipper). PACK_GEAR=1 restores the pack model (backtests of the pack period).
PACK_GEAR=os.environ.get('PACK_GEAR','0')=='1'
PEN_LOSS=.3 if PACK_GEAR else .5 # fraction of speed lost during a penalty (winch pro ratio .7, std .5)
LAND=unary_union([shape(f['geometry']) for f in json.loads((DATA/'land.geojson').read_text())['features']])

def interp2(a,lat,lon,lat0=LAT0,lon0=LON0,step=.25,fill=0.):
 lat,lon=np.broadcast_arrays(lat,lon)
 return map_coordinates(a,[((lat-lat0)/step).ravel(),((lon-lon0)/step).ravel()],order=1,mode='constant',cval=fill,prefilter=False).reshape(lat.shape)

def wind(t,lat,lon):
 z=(START_LEAD+np.asarray(t)-6)/6
 # Time is scalar in the vectorized router.
 i=int(np.clip(np.floor(z),0,len(W['hours'])-2));f=float(np.clip(z-i,0,1))
 u=interp2(W['u'][i],lat,lon)*(1-f)+interp2(W['u'][i+1],lat,lon)*f
 v=interp2(W['v'][i],lat,lon)*(1-f)+interp2(W['v'][i+1],lat,lon)*f
 return np.hypot(u,v)*1.943844492, np.degrees(np.arctan2(-u,-v))%360

def polar(twa,tws,sail):
 a,w,s=np.broadcast_arrays(np.abs(twa),tws,sail)
 ai=np.clip(np.searchsorted(TWAS,a,side='right')-1,0,len(TWAS)-2)
 wi=np.clip(np.searchsorted(TWSS,w,side='right')-1,0,len(TWSS)-2)
 af=np.clip((a-TWAS[ai])/(TWAS[ai+1]-TWAS[ai]),0,1)
 wf=np.clip((w-TWSS[wi])/(TWSS[wi+1]-TWSS[wi]),0,1)
 s=s.astype(int)
 q=(POLAR[s,ai,wi]*(1-af)*(1-wf)+POLAR[s,ai+1,wi]*af*(1-wf)+POLAR[s,ai,wi+1]*(1-af)*wf+POLAR[s,ai+1,wi+1]*af*wf)
 ct=np.minimum(np.clip((a-70)/10,0,1),np.clip((170-a)/10,0,1))
 cw=np.minimum(np.clip((w-11)/5,0,1),np.clip((40-w)/5,0,1))
 return q*(1+.04*ct*cw)*1.003

def angle(wd,heading):return (wd-heading+180)%360-180

def move(lat,lon,heading,dist):
 h=np.radians(heading)
 dlat=dist*np.cos(h)/60
 return lat+dlat,lon+dist*np.sin(h)/(60*np.cos(np.radians(lat+dlat*.5)))

def distance(lat,lon,goal=GOAL):
 p1,p2=np.radians(lat),np.radians(goal[0]);dl=np.radians(lon-goal[1])
 z=np.sin((p1-p2)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
 return 3440.065*2*np.arcsin(np.sqrt(np.clip(z,0,1)))

# West limit of the routing domain (6/11: -65 -> -74, the west option of the fleet was outside; GEFS fields reach -75).
LON_MIN=float(os.environ.get('LON_MIN','-74'))
# Box of the Bellman tables (bellman_fine/bellman_na/tree, gates, members). 6/11 evening, ~34 h from the finish: cropped
# from the whole North Atlantic (10-55 N, to 2 E: built for the start) to the end-of-race area, ~4x less computation.
# Outside the box the tables read 2000 (wall), so it must contain every route still worth considering (test: box edges).
LAT_LO=float(os.environ.get('BOX_LAT_LO','12'));LAT_HI=float(os.environ.get('BOX_LAT_HI','38'));LON_HI=float(os.environ.get('BOX_LON_HI','-40'))
MASK_STEP=.04;MLAT0=9.;MLON0=LON_MIN-2
MASK_LAT=np.arange(9,56.001,MASK_STEP);MASK_LON=np.arange(MLON0,3.001,MASK_STEP)
MASK_PATH=OUT/('land_mask.npz' if MLON0==-67 else f'land_mask_{-MLON0:g}w.npz')
if MASK_PATH.exists():
 _m=np.load(MASK_PATH);LANDMASK=_m['mask'];CLEARANCE=_m['clearance']
else:
 xx,yy=np.meshgrid(MASK_LON,MASK_LAT)
 LANDMASK=contains_xy(LAND,xx,yy)
 CLEARANCE=distance_transform_edt(~LANDMASK)*MASK_STEP*60*.7
 np.savez_compressed(MASK_PATH,mask=LANDMASK,clearance=CLEARANCE)

def clearance(lat,lon):
 lat,lon=np.broadcast_arrays(lat,lon)
 return map_coordinates(CLEARANCE,[((lat-MLAT0)/MASK_STEP).ravel(),((lon-MLON0)/MASK_STEP).ravel()],order=1,mode='constant',cval=0,prefilter=False).reshape(lat.shape)

def allowed(lat,lon,channel=False):
 ok=(lat>10)&(lat<55)&(lon>LON_MIN)&(lon<2)&(clearance(lat,lon)>2.)&~in_zone(lat,lon)
 if channel:
  # Conservative operational corridor from the screenshots; NOT authoritative VR polygons.
  ok &= (lon<=-6)|((lat>=48.93)&(lat<=49.5))
 return ok

def leg_allowed(lat,lon,lat2,lon2,channel=False):
 return allowed(lat2,lon2,channel)&allowed((lat+lat2)/2,(lon+lon2)/2,channel)

def maneuver(tws,energy,change_sail,change_tack,after_debit=False):
 factor=np.interp(tws,[0,10,20,30],[1,1,1.2,1.8] if PACK_GEAR else [1,1.2,1.5,2])
 furl=.8 if PACK_GEAR else 1.
 used=(20*furl*change_sail+10*change_tack)*1.5*factor
 c=(1-np.cos(np.clip((tws-10)/20,0,1)*np.pi))*.5
 # Combined tack+sail change costs the longer timer (ITYC "Tack / Gybe + sail change").
 base=np.maximum((300+180*c)*furl*change_sail,(180+120*c)*change_tack)
 e=np.maximum(0,energy-used) if after_debit else energy
 penalty_hours=base*np.maximum(.5,2-.015*e)/3600
 return used,penalty_hours

def recover_seconds(tws):
 # Smooth approximation to official 5--15 min / point bounds. Sensitivity uses constant 15 min.
 return (300+600*(1-np.cos(np.clip(tws/30,0,1)*np.pi))*.5)*(.8 if PACK_GEAR else 1.)

def propagate(t,lat,lon,energy,old_sail,old_tack,heading,new_sail,dt,conservative=False):
 ws,wd=wind(t,lat,lon);a=angle(wd,heading);new_tack=np.where(a>=0,1,-1)
 cs=(new_sail!=old_sail);ct=(new_tack!=old_tack)
 spent,penalty=maneuver(ws,energy,cs,ct,conservative)
 speed=polar(a,ws,new_sail)
 # Lower stamina is permitted; penalty continues into this interval. Search dt >= .5 h.
 eff=dt-.5*np.minimum(dt,penalty)
 la,lo=move(lat,lon,heading,speed*eff)
 # Midpoint wind corrects the displacement, retaining maneuver trigger at leg start.
 mw,mwd=wind(t+dt/2,(lat+la)/2,(lon+lo)/2)
 ma=angle(mwd,heading);sp=polar(ma,mw,new_sail)
 la,lo=move(lat,lon,heading,sp*eff)
 rec=900 if conservative else recover_seconds(mw)
 e=np.minimum(100,np.maximum(0,energy-spent)+np.maximum(0,dt-penalty)*3600/rec)
 valid=(np.abs(a)>35)&(np.abs(a)<175)&(np.sign(ma)==new_tack)&(penalty<=dt)
 return la,lo,e,new_tack,sp,penalty,spent,valid

def global_bellman(step=.5,horizon=180):
 if START_LEAD<W['hours'][0] or START_LEAD+horizon>W['hours'][-1]:
  raise ValueError('Requested horizon is outside the loaded forecast validity interval.')
 path=OUT/f'bellman_global_{step:g}.npz'
 if path.exists():return dict(np.load(path))
 las=np.arange(10,55+step/2,step);los=np.arange(LON_MIN,2+step/2,step)
 lo,la=np.meshgrid(los,las);ok=allowed(la,lo)
 target,toff=goal_target(la,lo)
 values=np.empty((horizon+1,*la.shape),dtype=np.float32)
 policies=np.empty((horizon,*la.shape),dtype=np.int8)
 values[-1]=200+distance(la,lo)/20;values[-1][target]=toff[target];values[-1][~ok]=2000
 # Both sides of the wind, with denser controls around downwind optimum.
 angles=np.array([-160,-150,-140,-135,-130,-125,-115,-105,-90,-75,-60,-50,50,60,75,90,105,115,125,130,135,140,150,160.])
 tic=time.time()
 for k in range(horizon-1,-1,-1):
  ws,wd=wind(k+.5,la,lo)
  best=np.full(la.shape,2000.,np.float32);policy=np.zeros(la.shape,np.int8)
  for j,a in enumerate(angles):
   h=(wd-a)%360;sp=np.maximum(polar(a,ws,0),polar(a,ws,1))
   ln,on=move(la,lo,h,sp)
   val=1+interp2(values[k+1],ln,on,las[0],los[0],step,2000)
   val=np.where(leg_allowed(la,lo,ln,on),val,2000)
   improve=val<best;best[improve]=val[improve];policy[improve]=j
  best[target]=toff[target];best[~ok]=2000;values[k]=best;policies[k]=policy
  if k%24==0:print('global Bellman',k,'remaining, elapsed',round(time.time()-tic,1),flush=True)
 np.savez_compressed(path,lat=las,lon=los,values=values,policies=policies,angles=angles)
 return dict(np.load(path))

def value(g,t,lat,lon):
 k=int(np.clip(np.floor(t),0,len(g['values'])-2));f=np.clip(t-k,0,1)
 step=g['lat'][1]-g['lat'][0]
 return interp2(g['values'][k],lat,lon,g['lat'][0],g['lon'][0],step,2000)*(1-f)+interp2(g['values'][k+1],lat,lon,g['lat'][0],g['lon'][0],step,2000)*f

def hybrid_dp(g,horizon=36,beam=3500,dt=.5,conservative=False):
 """Approximate forward Bellman recursion with grid aggregation and a bounded beam.
 Keep one representative per (position cell,sail,tack,energy bin). V_global
 ranks representatives; the relaxed bound is a heuristic, not an admissible proof.
 """
 name=f'hybrid_{horizon:g}_{beam}_{int(conservative)}';file=OUT/(name+'.json')
 labels=np.array([[START[0],START[1],INITIAL_ENERGY,INITIAL_SAIL,INITIAL_TACK]])
 layers=[];times=np.arange(0,horizon,dt);tic=time.time()
 offsets=np.array([-160,-150,-140,-135,-130,-125,-120,-115,-105,-90,-75,-60,-50,50,60,75,90,105,115,120,125,130,135,140,150,160.])
 for step,t in enumerate(times):
  la,lo,en,os,ot=labels.T;ws,wd=wind(t,la,lo)
  # At the first two hours retain small cap differences explicitly.
  hdg=(wd[:,None]-offsets[None,:])%360
  if t<2:hdg=np.concatenate([hdg,np.tile(np.arange(265,301,5),(len(labels),1))],axis=1)
  nh=hdg.shape[1]
  hdg=np.repeat(hdg,2,axis=1);ns=np.tile([0,1],(len(labels),nh))
  parent=np.repeat(np.arange(len(labels)),hdg.shape[1])
  h=hdg.ravel();s=ns.ravel();la=la[parent];lo=lo[parent];en=en[parent];os=os[parent];ot=ot[parent]
  ln,on,ee,nt,sp,pen,used,valid=propagate(t,la,lo,en,os,ot,h,s,dt,conservative)
  valid &= leg_allowed(la,lo,ln,on,channel=True)
  # Keep a broad Atlantic corridor; no direct route to a coast is allowed.
  ids=np.flatnonzero(valid)
  if not len(ids):raise RuntimeError('No feasible candidate at '+str(t))
  ln,on,ee,nt,h,s,parent,sp,pen,used=[x[ids] for x in [ln,on,ee,nt,h,s,parent,sp,pen,used]]
  potential=value(g,t+dt,ln,on)
  # Near-coast coarse value contamination: first hours use westward progress;
  # the corridor constrains the branch until the Atlantic opens west of 6 W.
  if t<5:potential=np.where(on>-6,145+(on+6)*3+np.abs(ln-49.1)*.25,potential)
  score=potential+(100-ee)*.0005
  cell=.06 if t<12 else .12
  keys=np.stack([np.rint(ln/cell),np.rint(on/cell),s,nt,np.floor(ee/10)],axis=1).astype(np.int32)
  order=np.argsort(score);_,unique=np.unique(keys[order],axis=0,return_index=True)
  keep=order[unique];keep=keep[np.argsort(score[keep])[:beam]]
  labels=np.stack([ln[keep],on[keep],ee[keep],s[keep],nt[keep]],axis=1)
  layers.append(dict(labels=labels.copy(),parent=parent[keep],heading=h[keep],speed=sp[keep],penalty=pen[keep],spent=used[keep],score=score[keep]))
  if step%12==0:print(name,'hour',round(t+dt,1),'states',len(labels),'best remaining',round(float(np.min(score[keep])),2),'elapsed',round(time.time()-tic,1),flush=True)
 j=int(np.argmin(layers[-1]['score']));rows=[]
 for k in range(len(layers)-1,-1,-1):
  l=layers[k];x=l['labels'][j];rows.append(dict(hours=float((k+1)*dt),lat=float(x[0]),lon=float(x[1]),energy=float(x[2]),sail=int(x[3]),tack=int(x[4]),heading=float(l['heading'][j]),speed=float(l['speed'][j]),penalty_hours=float(l['penalty'][j]),spent=float(l['spent'][j])))
  j=int(l['parent'][j])
 rows.reverse();rows.insert(0,dict(hours=0.,lat=float(START[0]),lon=float(START[1]),energy=float(INITIAL_ENERGY),sail=int(INITIAL_SAIL),tack=int(INITIAL_TACK),heading=285.,speed=20.6,penalty_hours=0.,spent=0.))
 output={'method':'hybrid approximate forward Bellman','horizon':horizon,'beam':beam,'dt':dt,'conservative':conservative,'remaining_value':float(np.min(layers[-1]['score'])),'rows':rows}
 file.write_text(json.dumps(output,indent=2))
 return output

def audit(rows):
 hits=[]
 for a,b in zip(rows,rows[1:]):
  if LineString([(a['lon'],a['lat']),(b['lon'],b['lat'])]).intersects(LAND):hits.append((a['hours'],b['hours']))
 return {'land_crossings':hits,'min_energy':min(q['energy'] for q in rows),'sail_changes':sum(a['sail']!=b['sail'] for a,b in zip(rows,rows[1:])),'tack_changes':sum(a['tack']!=b['tack'] for a,b in zip(rows,rows[1:]))}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--global-only',action='store_true');ap.add_argument('--horizon',type=float,default=36);ap.add_argument('--beam',type=int,default=3500);ap.add_argument('--conservative',action='store_true');a=ap.parse_args()
 g=global_bellman()
 if not a.global_only:
  r=hybrid_dp(g,a.horizon,a.beam,conservative=a.conservative);print('AUDIT',audit(r['rows']),flush=True)
