"""Join the Statcast fit to the repo's pitches, average per pitcher x pitch type, write data.json,
and compare the page's tunnel metrics under the old and new trajectory models."""
import sys, json, numpy as np, pandas as pd
fitcsv, repo, outjson = sys.argv[1:4]
YF=17/12
sc=pd.read_csv(fitcsv)
rp=pd.concat([pd.read_csv(f'{repo}/datamining/outputs/pitches_2025_{m:02d}.csv') for m in range(3,11)])
k=['game_pk','at_bat_number','pitch_number']
df=rp.merge(sc[k+['rx','ry','rz','rvx','rvy','rvz','ax','ay','az','tf','vx0','vy0','vz0']].rename(columns={'ax':'fax','ay':'fay','az':'faz'}),on=k,how='left')
print('repo pitches',len(rp),'matched',df.rx.notna().sum())
df=df.dropna(subset=['rx','pitch_type'])
sw={'swinging_strike','swinging_strike_blocked','foul_tip','foul','foul_bunt','hit_into_play','missed_bunt','bunt_foul_tip'}
wh={'swinging_strike','swinging_strike_blocked','foul_tip','missed_bunt'}
df['swing']=df.description.isin(sw); df['whiff']=df.description.isin(wh); df['csw']=df.description.isin(wh|{'called_strike'})
P=['rx','ry','rz','rvx','rvy','rvz','fax','fay','faz']
O=['release_pos_x_ft','release_pos_y_ft','release_pos_z_ft','ax_ft_s2','ay_ft_s2','az_ft_s2','release_speed_mph','plate_x_ft','plate_z_ft']
g=df.groupby(['pitcher','pitch_type']).agg(n=('rx','size'),sw=('swing','sum'),wh=('whiff','sum'),csw=('csw','mean'),**{c:(c,'mean') for c in P+O}).reset_index()
tot=g.groupby('pitcher').n.transform('sum'); g['usage']=g.n/tot
g=g[(g.n>=25)&(tot>=300)].copy()
a=0.5*g.fay; b=g.rvy; c=g.ry-YF
g['tf']=(-b-np.sqrt(b*b-4*a*c))/(2*a)
print(len(g),'rows',g.pitcher.nunique(),'pitchers')
out={}
for pid,grp in g.groupby('pitcher'):
    out[int(pid)]=[[r.pitch_type,int(r.n),round(r.usage,3),round(r.rx,3),round(r.ry,3),round(r.rz,3),round(r.rvx,3),round(r.rvy,3),round(r.rvz,3),
      round(r.fax,2),round(r.fay,2),round(r.faz,2),round(r.tf,4),round(r.release_speed_mph,1),round(r.plate_x_ft,3),round(r.plate_z_ft,3),int(r.sw),int(r.wh),round(r.csw,3)]
      for r in grp.sort_values('usage',ascending=False).itertuples()]
s=json.dumps(out,separators=(',',':')); open(outjson,'w').write(s); print('json bytes',len(s))

# ---- page-metric comparison (mirrors traj()/metrics() in the page) ----
IN=1/12; EYE={'R':np.array([-35.8*IN,-28.3*IN,65*IN]),'L':np.array([35.8*IN,-28.3*IN,65*IN])}
def told(r):
    x0,y0,z0,ax,ay,az,v,xf,zf=[getattr(r,c) for c in O]
    tf=abs(y0)/(0.95*v*1.46667)
    vx=(xf-x0-.5*ax*tf*tf)/tf; vy=(0-y0-.5*ay*tf*tf)/tf; vz=(zf-z0-.5*az*tf*tf)/tf
    return tf,lambda t:np.array([x0+vx*t+.5*ax*t*t,y0+vy*t+.5*ay*t*t,z0+vz*t+.5*az*t*t])
def toldf(r):
    # old model, but with the decision clock measured to the front of the plate like the new one
    tf,at=told(r); vy=(0-r.release_pos_y_ft-.5*r.ay_ft_s2*tf*tf)/tf; a=.5*r.ay_ft_s2; c=r.release_pos_y_ft-YF
    return (-vy-np.sqrt(vy*vy-4*a*c))/(2*a),at
def tnew(r):
    return r.tf,lambda t:np.array([r.rx+r.rvx*t+.5*r.fax*t*t,r.ry+r.rvy*t+.5*r.fay*t*t,r.rz+r.rvz*t+.5*r.faz*t*t])
def ang(a,b): return np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)))
rows=[]
for pid,grp in g.groupby('pitcher'):
    rs=list(grp.itertuples()); fb=[r for r in rs if r.pitch_type in('FF','SI','FC')] or rs
    ref=max(fb,key=lambda r:r.usage)
    for side in 'RL':
        e=EYE[side]; res={}
        for lab,T in (('old',told),('oldf',toldf),('new',tnew)):
            tf,at=T(ref); td=max(0.02,tf-0.175); pf=at(td); res[lab]={}
            for r in rs:
                if r is ref: continue
                tf2,at2=T(r); p=at2(min(td,tf2))
                res[lab][r.pitch_type]=(ang(p-e,pf-e),np.linalg.norm(p-pf)*12,tf)
        for pt in res['new']:
            rows.append(dict(pid=pid,side=side,pt=pt,a_old=res['old'][pt][0],a_new=res['new'][pt][0],a_oldf=res['oldf'][pt][0],g_old=res['old'][pt][1],g_new=res['new'][pt][1],tf_old=res['old'][pt][2],tf_new=res['new'][pt][2]))
m=pd.DataFrame(rows)
m['da']=m.a_new-m.a_old; m['dg']=m.g_new-m.g_old
print('ref flight time, old (to y=0) vs new (to front of plate), ms:', round(m.tf_old.median()*1000,1), round(m.tf_new.median()*1000,1))
print('angle at decision, deg: median new %.2f; |new-old| median %.3f p90 %.3f max %.3f; median rel %.1f%%'%(m.a_new.median(),m.da.abs().median(),m.da.abs().quantile(.9),m.da.abs().max(),(m.da.abs()/m.a_new).median()*100))
print('gap at decision, in: |new-old| median %.2f p90 %.2f'%(m.dg.abs().median(),m.dg.abs().quantile(.9)))
# leaderboard churn: best-tunnel rank per side (min angle among non-ref pitches with usage) — compare top 25 sets
for side in 'RL':
    s=m[m.side==side].groupby('pid')[['a_old','a_new']].min()
    top_o=set(s.a_old.nsmallest(25).index); top_n=set(s.a_new.nsmallest(25).index)
    print(side,'top-25 overlap',len(top_o&top_n),'spearman',round(s.a_old.rank().corr(s.a_new.rank()),4))
m.to_csv(outjson.replace('.json','_cmp.csv'),index=False)
m['daf']=m.a_new-m.a_oldf
print('same clock (front of plate): |new-old| median %.3f p90 %.3f; median rel %.1f%%; new>old share %.2f'%(m.daf.abs().median(),m.daf.abs().quantile(.9),(m.daf.abs()/m.a_new).median()*100,(m.daf>0).mean()))
for side in 'RL':
    s=m[m.side==side].groupby('pid')[['a_oldf','a_new']].min()
    print(side,'same clock: top-25 overlap',len(set(s.a_oldf.nsmallest(25).index)&set(s.a_new.nsmallest(25).index)),'top-10',len(set(s.a_oldf.nsmallest(10).index)&set(s.a_new.nsmallest(10).index)),'rank corr',round(s.a_oldf.rank().corr(s.a_new.rank()),4))
