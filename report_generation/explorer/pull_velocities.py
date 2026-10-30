import sys, io, datetime as dt, pandas as pd, requests, time
from concurrent.futures import ThreadPoolExecutor
out=sys.argv[1]
COLS=['game_pk','at_bat_number','pitch_number','pitcher','pitch_type','game_date','release_pos_x','release_pos_y','release_pos_z','vx0','vy0','vz0','ax','ay','az','release_speed','release_extension','plate_x','plate_z']
U="https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&game_date_gt={d}&game_date_lt={d}&player_type=pitcher"
def get(d):
    for k in range(4):
        try:
            r=requests.get(U.format(d=d),timeout=180); r.raise_for_status()
            t=r.content.decode('utf-8-sig')
            if len(t)<50: return None
            df=pd.read_csv(io.StringIO(t),usecols=COLS,low_memory=False)
            return df
        except Exception as e:
            print(d,'retry',e,flush=True); time.sleep(2**k)
    print(d,'FAILED',flush=True)
days=[str(dt.date(2025,3,20)+dt.timedelta(i)) for i in range(196)]
with ThreadPoolExecutor(4) as ex: res=list(ex.map(get,days))
df=pd.concat([r for r in res if r is not None]); print(len(df)); df.to_csv(out,index=False)
