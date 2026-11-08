# Generic NOAA fetch: GFS 0.25 deterministic + full GEFS (c00,p01..p30) 0.5, 10 m wind, 3-hourly.
# Usage: fetch_ensemble.py YYYYMMDDHH [GEFS_H] [--follow]  -> data/ens_YYYYMMDDHH/{gfs,gefsc00,gefsp01..}3h.npz
# --follow: start while the cycle is still being published; files are requested in forecast-hour order and retried
# every 60 s until FOLLOW_MAX s (default 3 h). Each model/member is converted (READY) as soon as its hours are all in.
import sys,urllib.request,urllib.parse,concurrent.futures,time
from pathlib import Path
import numpy as np
import eccodes as ec
CYC=sys.argv[1];DAY,HH=CYC[:8],CYC[8:]
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'/f'ens_{CYC}';D.mkdir(exist_ok=True)
MEMBERS=['c00']+[f'p{k:02}' for k in range(1,31)]
GFS_H=list(range(3,int(__import__('os').environ.get('GFS_H','198'))+1,3)) # GFS forecast hours (198: the 180 h tables need it)
# GEFS horizon (h) optional 2nd arg: 48 is enough for the 24 h program+feedback; 120 for multi-day strategy.
GEFS_H=list(range(3,(int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 48)+1,3))
FOLLOW='--follow' in sys.argv;T0=time.time();FOLLOW_MAX=float(__import__('os').environ.get('FOLLOW_MAX','10800'))
def fname(model,member,h):
 return f'gfs.t{HH}z.pgrb2.0p25.f{h:03}' if model=='gfs' else f'ge{member}.t{HH}z.pgrb2a.0p50.f{h:03}'
def fetch(job):
 model,member,h=job;name=fname(model,member,h)
 if model=='gfs':folder=f'/gfs.{DAY}/{HH}/atmos';flt='filter_gfs_0p25.pl'
 else:folder=f'/gefs.{DAY}/{HH}/atmos/pgrb2ap5';flt='filter_gefs_atmos_0p50a.pl'
 p=D/name;err=None;attempt=0
 while True:
  if p.exists():return job
  try:
   query=urllib.parse.urlencode(dict(file=name,dir=folder,lev_10_m_above_ground='on',var_UGRD='on',var_VGRD='on',subregion='',leftlon=-75,rightlon=5,toplat=58,bottomlat=5))
   data=urllib.request.urlopen('https://nomads.ncep.noaa.gov/cgi-bin/'+flt+'?'+query,timeout=40).read()
   if not data.startswith(b'GRIB'): raise ValueError(str(job)+' not GRIB '+repr(data[:80]))
   tmp=p.with_suffix(p.suffix+f'.part{id(job)}');tmp.write_bytes(data);tmp.replace(p);return job # atomic: concurrent fetchers never see a partial file
  except Exception as e:
   err=e;attempt+=1
   if FOLLOW and time.time()-T0<FOLLOW_MAX:time.sleep(60);continue # not published yet
   if attempt>=4:raise err
   time.sleep(5*attempt)
def build(model,member):
 hours=GFS_H if model=='gfs' else GEFS_H;U=[];V=[]
 for h in hours:
  fields={}
  with (D/fname(model,member,h)).open('rb') as f:
   while (g:=ec.codes_grib_new_from_file(f)) is not None:
    short=ec.codes_get(g,'shortName');lat=ec.codes_get_array(g,'latitudes');lon=(ec.codes_get_array(g,'longitudes')+180)%360-180;vals=ec.codes_get_values(g)
    if ec.codes_get(g,'dataDate')!=int(DAY) or ec.codes_get(g,'dataTime')!=int(HH)*100:raise ValueError('forecast timestamp mismatch')
    las=np.unique(lat);los=np.unique(lon);a=np.empty((len(las),len(los)))
    a[np.searchsorted(las,lat),np.searchsorted(los,lon)]=vals
    fields[short]=a;ec.codes_release(g)
  U.append(fields['10u']);V.append(fields['10v'])
 np.savez_compressed(D/(model+member+'3h.npz'),hours=hours,lat=las,lon=los,u=np.array(U,np.float32),v=np.array(V,np.float32))
if __name__=='__main__':
 # Forecast-hour order (= NOMADS publication order), GFS first at equal hour; convert each model/member when complete.
 jobs=sorted([('gfs','',h) for h in GFS_H]+[('gefs',m,h) for m in MEMBERS for h in GEFS_H],key=lambda j:(j[2],j[0]!='gfs',j[1]))
 left={}
 for j in jobs:left[(j[0],j[1])]=left.get((j[0],j[1]),0)+1
 failures=[];done=set()
 def ready(key):
  model,member=key
  if (D/(model+member+'3h.npz')).exists() and key not in done:done.add(key);print('READY',model+member,flush=True);return
  try:build(model,member);done.add(key);print('READY',model+member,flush=True)
  except Exception as e:print('SKIP',model+member,e,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  pending={pool.submit(fetch,j):j for j in jobs}
  for k,f in enumerate(concurrent.futures.as_completed(pending)):
   j=pending[f]
   try:f.result();left[(j[0],j[1])]-=1
   except Exception as e:failures.append(str(e))
   if left[(j[0],j[1])]==0:ready((j[0],j[1]))
   if (k+1)%100==0:print('Downloaded',k+1,'/',len(jobs),'errors',len(failures),round(time.time()-T0),'s',flush=True)
 print('FAILURES',len(failures),failures[:5],flush=True)
 for key in [('gfs','')]+[('gefs',m) for m in MEMBERS]:
  if key not in done and left[key]==0:ready(key)
