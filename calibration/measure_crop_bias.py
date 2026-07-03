"""Calibration v2: subtle cut-edge on smooth dark desk.

Fronts: SIFT to render -> projected render rect anchors the cut search;
        cut found per scan by 2-segment changepoint (CUSUM/SSE) on
        band-averaged profiles; borders computed in render space.
Backs:  gold frame rectangle via tophat+Hough, refined by peak centroid;
        cut = changepoint scanning outward from the frame.
"""
import sys, json, os
import numpy as np, cv2
sys.path.insert(0, "/sessions/wizardly-lucid-bell/ca/src")
from centering.render_match import match_to_render
from centering import geometry as G

CARD_W, CARD_H = 63.5, 88.9

def profile(gray, p, n_dir, offs, band, e_dir):
    """band-averaged profile at point p along n_dir over offsets offs."""
    xs = p[0] + offs[:,None]*n_dir[0] + np.arange(-(band//2),band//2+1)[None,:]*e_dir[0]
    ys = p[1] + offs[:,None]*n_dir[1] + np.arange(-(band//2),band//2+1)[None,:]*e_dir[1]
    v = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    return v.mean(axis=1)

def changepoint(prof):
    """best 2-segment split; returns (idx, quality). prof ordered out->in."""
    n=len(prof)
    c1=np.cumsum(prof); c2=np.cumsum(prof**2)
    best=None; bcost=None
    ks=np.arange(8,n-8)
    m1=c1[ks-1]/ks
    v1=c2[ks-1]-ks*m1**2
    m2=(c1[-1]-c1[ks-1])/(n-ks)
    v2=(c2[-1]-c2[ks-1])-(n-ks)*m2**2
    cost=v1+v2
    i=int(np.argmin(cost)); k=int(ks[i])
    sep=abs(m1[i]-m2[i])
    resid=np.sqrt(cost[i]/n)
    q=sep/max(resid,1e-3)
    # sub-pixel: crossing of mid-level near k
    mid=0.5*(m1[i]+m2[i])
    lo,hi=max(0,k-4),min(n-1,k+4)
    seg=prof[lo:hi+1]
    s=np.sign(seg-mid)
    idx=None
    for j in range(len(seg)-1):
        if s[j]!=s[j+1] and s[j]!=0:
            f=(mid-seg[j])/(seg[j+1]-seg[j])
            idx=lo+j+f
            break
    if idx is None: idx=float(k)
    return idx, q, sep

def cut_scan(gray, anchor_line, side, u_positions, win_out_mm, win_in_mm, ppm,
             band=12, min_prom=10.0, plateau_mm=0.35):
    """Hybrid cut-edge detector.
    1) specular ridge (bright cardboard edge): prominence peak, centroid.
    2) else plateau-exit knee: border plateau level L; first sustained
       departure; sub-pixel via crossing of transition slope with L.
    Window: anchor-win_in (inside) .. anchor+win_out (outward)."""
    sign=-1 if side in ("left","top") else 1
    horiz = side in ("top","bottom")
    e_dir=(1.0,0.0) if horiz else (0.0,1.0)
    n_dir=(0.0,sign*1.0) if horiz else (sign*1.0,0.0)
    step=0.3
    offs=np.arange(-win_in_mm*ppm, win_out_mm*ppm, step)
    us,vs,methods=[],[],[]
    H,W=gray.shape
    k0=max(6,int((plateau_mm+win_in_mm)*ppm/step))
    for u in u_positions:
        v0=anchor_line(u)
        p=(u,v0) if horiz else (v0,u)
        xe=p[0]+offs[-1]*n_dir[0]; ye=p[1]+offs[-1]*n_dir[1]
        xb=p[0]+offs[0]*n_dir[0];  yb=p[1]+offs[0]*n_dir[1]
        if not (5<xe<W-5 and 5<ye<H-5 and 5<xb<W-5 and 5<yb<H-5): continue
        prof=profile(gray,p,n_dir,offs,band,e_dir)
        n=len(prof)
        off=None
        # --- ridge attempt ---
        pk=int(np.argmax(prof[k0:]))+k0
        if 6<=pk<=n-7:
            lbase=np.median(prof[:max(4,pk-6)])
            rbase=np.median(prof[min(n-4,pk+6):])
            prom=prof[pk]-max(lbase,rbase)
            if prom>=min_prom:
                half=max(lbase,rbase)+0.4*prom
                a=pk
                while a>0 and prof[a-1]>half: a-=1
                b=pk
                while b<n-1 and prof[b+1]>half: b+=1
                if (b-a)*step<=1.2*ppm and a>2 and b<n-3:
                    w=np.clip(prof[a:b+1]-half,0,None)
                    if w.sum()>0:
                        off=(offs[a:b+1]*w).sum()/w.sum()
                        methods.append("ridge")
        # --- knee fallback ---
        if off is None:
            L=np.median(prof[:k0]); noise=max(np.std(prof[:k0]),1.0)
            thr=max(6.0,4.5*noise)
            dep=np.abs(prof-L)>thr
            idx=None; run=0
            for i in range(k0,n):
                run=run+1 if dep[i] else 0
                if run>=6: idx=i-5; break
            if idx is None or idx<3 or idx>n-8: continue
            j0,j1=max(0,idx-2),min(n,idx+7)
            A=np.polyfit(offs[j0:j1],prof[j0:j1],1)
            if abs(A[0])<1e-4: continue
            off=(L-A[1])/A[0]
            if not (offs[max(0,idx-4)]-2*step<=off<=offs[min(n-1,idx+8)]+2*step):
                off=float(offs[idx])
            methods.append("knee")
        us.append(float(u)); vs.append(float(v0+sign*off))
    return np.array(us),np.array(vs)

def fit_line(us,vs,orient):
    ln=G.FittedLine.fit(orient,np.asarray(us),np.asarray(vs))
    return ln

def front_cal(photo, render_file):
    bgr=cv2.imread(photo); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    H,W=gray.shape
    r=cv2.imread(render_file,0).astype(np.float32)
    Hr,Wr=r.shape
    Hpr,ninl,med=match_to_render(gray,r,photo_mask=None)
    Hrp=np.linalg.inv(Hpr)
    rect=np.array([[0,0],[Wr,0],[Wr,Hr],[0,Hr]],np.float64)
    rp=G.transform_points(Hrp,rect)  # render corners in photo
    ppm=np.linalg.norm(rp[1]-rp[0])/61.5  # approx (render ~61.5mm wide)
    lines={}
    for side,(a,b) in {"top":(rp[0],rp[1]),"right":(rp[1],rp[2]),
                       "bottom":(rp[3],rp[2]),"left":(rp[0],rp[3])}.items():
        horiz = side in ("top","bottom")
        n=90
        ts=np.linspace(0.12,0.88,n)
        pts=a[None,:]+ts[:,None]*(b-a)[None,:]
        us=pts[:,0] if horiz else pts[:,1]
        vv=pts[:,1] if horiz else pts[:,0]
        co=np.polyfit(us,vv,1)
        anchor=lambda u,co=co: co[0]*u+co[1]
        u_ok,v_ok=cut_scan(gray,anchor,side,us,win_out_mm=3.2,win_in_mm=0.4,ppm=ppm)
        if len(u_ok)<25: raise RuntimeError(f"front {side}: {len(u_ok)} scans")
        lines[side]=fit_line(u_ok,v_ok,"h" if horiz else "v")
    # map cut lines to render space
    rl={}
    for side,ln in lines.items():
        pr=G.transform_points(Hpr,ln.points(60))
        orient="v" if side in("left","right") else "h"
        u,v=(pr[:,1],pr[:,0]) if orient=="v" else (pr[:,0],pr[:,1])
        rl[side]=G.FittedLine.fit(orient,u,v)
    x_l=float(rl["left"].v_at(Hr/2)); x_r=float(rl["right"].v_at(Hr/2))
    y_t=float(rl["top"].v_at(Wr/2)); y_b=float(rl["bottom"].v_at(Wr/2))
    s=(x_r-x_l)/CARD_W
    hchk=(y_b-y_t)/s
    # physical borders to render frame anchors (left/right 60px, top 66, band 1907)
    out={"n_inl":ninl,"med_reproj":med,"s_render_ppm":s,"height_check_mm":hchk,
      "rms_px":{k:lines[k].rms for k in lines},
      "offsets_mm":{"left":-x_l/s,"right":(x_r-Wr)/s,"top":-y_t/s,"bottom":(y_b-Hr)/s},
      "shift_x_mm":((Wr/2)-0.5*(x_l+x_r))/s, "shift_y_mm":((Hr/2)-0.5*(y_t+y_b))/s,
      "phys_border_mm":{"left":(60-x_l)/s,"right":(x_r-(Wr-60))/s,
                        "top":(66-y_t)/s,"bottom":(y_b-1907)/s}}
    return out

def back_frame_lines(gray):
    """gold frame rectangle via tophat + Hough + peak refinement."""
    H,W=gray.shape
    g8=cv2.convertScaleAbs(gray)
    top=cv2.morphologyEx(g8,cv2.MORPH_TOPHAT,np.ones((9,9),np.uint8))
    bw=(top>18).astype(np.uint8)*255
    ls=cv2.HoughLinesP(bw,1,np.pi/360,150,minLineLength=int(0.25*W),maxLineGap=30)
    if ls is None: raise RuntimeError("no hough lines")
    hs,vs=[],[]
    for l in ls[:,0]:
        x1,y1,x2,y2=l
        if abs(y2-y1)<0.05*abs(x2-x1): hs.append(((y1+y2)/2,(x1,x2)))
        elif abs(x2-x1)<0.05*abs(y2-y1): vs.append(((x1+x2)/2,(y1,y2)))
    if not hs or not vs: raise RuntimeError("no axis lines")
    hpos=np.array([h[0] for h in hs]); vpos=np.array([v[0] for v in vs])
    sides={}
    sides["top"]=hpos.min(); sides["bottom"]=hpos.max()
    sides["left"]=vpos.min(); sides["right"]=vpos.max()
    return sides

def refine_peak_line(gray, side, approx_v, ppm, n=90, band=3, half_mm=0.8):
    H,W=gray.shape
    horiz = side in ("top","bottom")
    span = W if horiz else H
    us=np.linspace(0.18*span,0.82*span,n)
    hw=half_mm*ppm
    offs=np.arange(-hw,hw,0.4)
    e_dir=(1.0,0.0) if horiz else (0.0,1.0)
    n_dir=(0.0,1.0) if horiz else (1.0,0.0)
    ok_u,ok_v=[],[]
    for u in us:
        p=(u,approx_v) if horiz else (approx_v,u)
        prof=profile(gray,p,n_dir,offs,band,e_dir)
        base=np.median(prof); exc=prof-base
        pk=int(np.argmax(exc))
        if exc[pk]<20: continue
        a,b=max(0,pk-4),min(len(prof),pk+5)
        w=np.clip(exc[a:b],0,None)
        pos=(offs[a:b]*w).sum()/w.sum()
        ok_u.append(float(u)); ok_v.append(float(approx_v+pos))
    if len(ok_u)<25: raise RuntimeError(f"frame {side}: {len(ok_u)}")
    return fit_line(ok_u,ok_v,"h" if horiz else "v")

def back_cal(photo):
    bgr=cv2.imread(photo); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    H,W=gray.shape
    sides=back_frame_lines(gray)
    ppm0=(sides["right"]-sides["left"])/58.2
    frame={s:refine_peak_line(gray,s,sides[s],ppm0) for s in ("left","right","top","bottom")}
    # cut: changepoint outward from frame, window 1.0..4.2mm
    cuts={}
    for side in ("left","right","top","bottom"):
        horiz=side in ("top","bottom")
        span=W if horiz else H
        us=np.linspace(0.15*span,0.85*span,90)
        ln=frame[side]
        anchor=lambda u,ln=ln: float(ln.v_at(u))
        sign=1
        u_ok,v_ok=cut_scan(gray,anchor,side,us,win_out_mm=4.2,win_in_mm=-1.0,ppm=ppm0)
        if len(u_ok)<20: raise RuntimeError(f"back cut {side}: {len(u_ok)}")
        cuts[side]=fit_line(u_ok,v_ok,"h" if horiz else "v")
    xl=float(cuts["left"].v_at(H/2)); xr=float(cuts["right"].v_at(H/2))
    yt=float(cuts["top"].v_at(W/2)); yb=float(cuts["bottom"].v_at(W/2))
    ppm=(xr-xl)/CARD_W
    borders={ "left":(float(frame["left"].v_at(H/2))-xl)/ppm,
              "right":(xr-float(frame["right"].v_at(H/2)))/ppm,
              "top":(float(frame["top"].v_at(W/2))-yt)/ppm,
              "bottom":(yb-float(frame["bottom"].v_at(W/2)))/ppm }
    return {"ppm":ppm,"height_check_mm":(yb-yt)/ppm,
            "cut_rms_px":{k:cuts[k].rms for k in cuts},
            "frame_rms_px":{k:frame[k].rms for k in frame},
            "borders_mm":borders,
            "ratio_tb": 100*borders["top"]/(borders["top"]+borders["bottom"]),
            "ratio_lr": 100*borders["left"]/(borders["left"]+borders["right"])}

if __name__=="__main__":
    mode,photo=sys.argv[1],sys.argv[2]
    if mode=="front":
        print(json.dumps(front_cal(photo,sys.argv[3]),indent=1,default=float))
    else:
        print(json.dumps(back_cal(photo),indent=1,default=float))

# ---- v3: back via SIFT transfer from reference back photo ----
from centering.imgio import load_photo
from centering.locate import coarse_locate
from centering import edges as LE
from centering.games.lorcana import LORCANA

REF_BACK="/sessions/wizardly-lucid-bell/mnt/uploads/IMG_6331.HEIC"

def ref_back_frame():
    rgb,gray,inp=load_photo(REF_BACK)
    H,W=gray.shape
    coarse,ppm0=coarse_locate(gray,CARD_W,CARD_H)
    ys,ye=coarse["top"].pos,coarse["bottom"].pos
    xs,xe=coarse["left"].pos,coarse["right"].pos
    rows=np.linspace(ys+0.15*(ye-ys),ys+0.85*(ye-ys),55)
    cols=np.linspace(xs+0.15*(xe-xs),xs+0.85*(xe-xs),55)
    lines={}
    for side in ("left","right","top","bottom"):
        us=rows if side in ("left","right") else cols
        u,v,d=LE.texture_scan(gray,side,coarse[side].pos,us,
                              search_out_px=6.0*ppm0,search_in_px=3.0*ppm0)
        lines[side]=G.FittedLine.fit("v" if side in ("left","right") else "h",u,v)
    ppm=float(np.median(lines["right"].v_at(rows)-lines["left"].v_at(rows)))/CARD_W
    fl={}
    for side in ("left","right","top","bottom"):
        us=rows if side in ("left","right") else cols
        fu,fv,fd=LE.frame_peak_scan(gray,side,lines[side],us,ppm,
                                    min_peak=45.0,search_mm=(0.5,6.0))
        fl[side]=G.FittedLine.fit("v" if side in ("left","right") else "h",fu,fv)
    return gray, fl

def back_cal2(photo, ref_gray, ref_frames):
    gray=cv2.cvtColor(cv2.imread(photo),cv2.COLOR_BGR2GRAY).astype(np.float32)
    H,W=gray.shape
    Hpr,ninl,med=match_to_render(gray,ref_gray,photo_mask=None)  # photo->ref
    Hrp=np.linalg.inv(Hpr)
    frame={}
    ppm0=None
    # transfer frame lines ref->photo, refine
    tp={}
    for side,fl in ref_frames.items():
        pts=G.transform_points(Hrp,fl.points(60))
        horiz=side in ("top","bottom")
        u,v=(pts[:,0],pts[:,1]) if horiz else (pts[:,1],pts[:,0])
        co=np.polyfit(u,v,1)
        tp[side]=co
    ppm0=abs((tp["right"][1]+tp["right"][0]*H/2)-(tp["left"][1]+tp["left"][0]*H/2))/58.7
    for side,co in tp.items():
        approx_mid=co[1]+co[0]*((W if side in ("top","bottom") else H)/2)
        frame[side]=refine_peak_line(gray,side,approx_mid,ppm0,half_mm=0.6)
    cuts={}
    for side in ("left","right","top","bottom"):
        horiz=side in ("top","bottom")
        span=W if horiz else H
        us=np.linspace(0.15*span,0.85*span,90)
        ln=frame[side]
        anchor=lambda u,ln=ln: float(ln.v_at(u))
        u_ok,v_ok=cut_scan(gray,anchor,side,us,win_out_mm=4.2,win_in_mm=-1.0,
                           ppm=ppm0)
        if len(u_ok)<20: raise RuntimeError(f"back cut {side}: {len(u_ok)}")
        cuts[side]=fit_line(u_ok,v_ok,"h" if horiz else "v")
    xl=float(cuts["left"].v_at(H/2)); xr=float(cuts["right"].v_at(H/2))
    yt=float(cuts["top"].v_at(W/2)); yb=float(cuts["bottom"].v_at(W/2))
    ppm=(xr-xl)/CARD_W
    borders={"left":(float(frame["left"].v_at(H/2))-xl)/ppm,
             "right":(xr-float(frame["right"].v_at(H/2)))/ppm,
             "top":(float(frame["top"].v_at(W/2))-yt)/ppm,
             "bottom":(yb-float(frame["bottom"].v_at(W/2)))/ppm}
    return {"n_inl":ninl,"med_reproj":med,"ppm":ppm,
            "height_check_mm":(yb-yt)/ppm,
            "cut_rms_px":{k:round(cuts[k].rms,2) for k in cuts},
            "frame_rms_px":{k:round(frame[k].rms,2) for k in frame},
            "borders_mm":borders,
            "ratio_tb":100*borders["top"]/(borders["top"]+borders["bottom"]),
            "ratio_lr":100*borders["left"]/(borders["left"]+borders["right"])}
