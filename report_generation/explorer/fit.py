"""Rebuild each pitch from Statcast's 9-parameter fit and compare with the old guessed-flight-time model."""
import sys, json, numpy as np, pandas as pd
sc=pd.read_csv(sys.argv[1]); repo=sys.argv[2]
YF=17/12  # front of plate, where plate_x/plate_z are reported
sc=sc.dropna(subset=['vx0','vy0','vz0','ax','ay','az','release_pos_y','plate_x','plate_z','release_speed'])
# time from y=50 to the front of the plate: 50 + vy0 t + ay t^2/2 = YF
def t_at(y, vy0, ay):
    a=0.5*ay; b=vy0; c=50-y
    return (-b-np.sqrt(b*b-4*a*c))/(2*a)
tp=t_at(YF,sc.vy0,sc.ay)          # >0
tr=t_at(sc.release_pos_y,sc.vy0,sc.ay)  # <0, release is before y=50
# x,z at y=50 from the plate location (plate_x/z are the fit evaluated at the front of the plate)
x50=sc.plate_x-sc.vx0*tp-0.5*sc.ax*tp**2
z50=sc.plate_z-sc.vz0*tp-0.5*sc.az*tp**2
# state at release
sc['rx']=x50+sc.vx0*tr+0.5*sc.ax*tr**2
sc['rz']=z50+sc.vz0*tr+0.5*sc.az*tr**2
sc['ry']=sc.release_pos_y
sc['rvx']=sc.vx0+sc.ax*tr; sc['rvy']=sc.vy0+sc.ay*tr; sc['rvz']=sc.vz0+sc.az*tr
sc['tf']=tp-tr
sc['v_fit']=np.sqrt(sc.rvx**2+sc.rvy**2+sc.rvz**2)/1.46667
print('pitches',len(sc))
print('release pos check (fit vs reported), in: x med %.2f p90 %.2f; z med %.2f p90 %.2f'%(
  (abs(sc.rx-sc.release_pos_x)*12).median(),(abs(sc.rx-sc.release_pos_x)*12).quantile(.9),
  (abs(sc.rz-sc.release_pos_z)*12).median(),(abs(sc.rz-sc.release_pos_z)*12).quantile(.9)))
print('speed check mph: med |fit-reported| %.2f'%abs(sc.v_fit-sc.release_speed).median())

# old model, per pitch: starts at reported release point, ends at y=0 at plate_x/z, tf guessed
def old(x0,y0,z0,ax,ay,az,v,xf,zf):
    tf=np.abs(y0)/(0.95*v*1.46667)
    vx=(xf-x0-0.5*ax*tf*tf)/tf; vy=(0-y0-0.5*ay*tf*tf)/tf; vz=(zf-z0-0.5*az*tf*tf)/tf
    return tf,vx,vy,vz
otf,ovx,ovy,ovz=old(sc.release_pos_x,sc.release_pos_y,sc.release_pos_z,sc.ax,sc.ay,sc.az,sc.release_speed,sc.plate_x,sc.plate_z)
# old model's time to the front of the plate (y=YF)
a=0.5*sc.ay; b=ovy; c=sc.release_pos_y-YF
otp=(-b-np.sqrt(b*b-4*a*c))/(2*a)
def pos_new(t): return np.stack([sc.rx+sc.rvx*t+0.5*sc.ax*t*t, sc.ry+sc.rvy*t+0.5*sc.ay*t*t, sc.rz+sc.rvz*t+0.5*sc.az*t*t])
def pos_old(t): return np.stack([sc.release_pos_x+ovx*t+0.5*sc.ax*t*t, sc.release_pos_y+ovy*t+0.5*sc.ay*t*t, sc.release_pos_z+ovz*t+0.5*sc.az*t*t])
res={}
res['flight_ms_new_med']=float(sc.tf.median()*1000)
d=(otp-sc.tf)*1000
res['flight_err_ms']={'median':float(d.median()),'mean_abs':float(d.abs().mean()),'p90_abs':float(d.abs().quantile(.9))}
for lab,dt in [('decision_175ms',0.175)]:
    e=np.linalg.norm(pos_old(otp-dt)-pos_new(sc.tf-dt),axis=0)*12
    res[f'pos_err_in_at_{lab}_before_plate']={'median':float(np.median(e)),'p90':float(np.quantile(e,.9))}
    e=np.linalg.norm(pos_old(sc.tf-dt)-pos_new(sc.tf-dt),axis=0)*12
    res[f'pos_err_in_same_clock_{lab}']={'median':float(np.median(e)),'p90':float(np.quantile(e,.9))}
res['by_type_flight_err_ms']=d.groupby(sc.pitch_type).median().round(1).dropna().to_dict()
print(json.dumps(res,indent=1))
sc.to_csv(sys.argv[3],index=False)
