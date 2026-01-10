#@title Two-fencer tracker: ROI-only for first frame, global keypoint matching after
!pip -q install ultralytics==8.3.35 opencv-python-headless==4.10.0.84

import cv2, numpy as np, os
from ultralytics import YOLO
from typing import List, Optional, Tuple, Dict

# -----------------------------
# Pose drawing / skeleton
# -----------------------------
SKELETON = [(a-1, b-1) for a,b in [
    [16,14],[14,12],[17,15],[15,13],[12,13],
    [6,12],[7,13],[6,7],
    [6,8],[8,10],[7,9],[9,11],
    [12,14],[14,16],[13,15],[15,17]
]]
COLORS = [(255,64,64), (64,255,64)]
LABELS = ["Fencer 0 (Left)", "Fencer 1 (Right)"]

def draw_pose(img, kpts_xy, color, label=None):
    if kpts_xy is None: return img
    for a,b in SKELETON:
        xa,ya = int(kpts_xy[a,0]), int(kpts_xy[a,1])
        xb,yb = int(kpts_xy[b,0]), int(kpts_xy[b,1])
        if xa>0 and ya>0 and xb>0 and yb>0:
            cv2.line(img,(xa,ya),(xb,yb),color,2)
    for p in kpts_xy:
        x,y = int(p[0]), int(p[1])
        if x>0 and y>0: cv2.circle(img,(x,y),4,color,-1)
    if label:
        c = centroid_from_kpts(kpts_xy)
        if c:
            x,y = int(c[0]), int(c[1])
            (tw,th),bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            pad = 6
            cv2.rectangle(img,(x,y-th-2*pad),(x+tw+2*pad,y+bl),color,-1)
            cv2.putText(img,label,(x+pad,y-pad),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv2.LINE_AA)
    return img

# -----------------------------
# Geometry helpers
# -----------------------------
def valid_mask(kpts_xy: np.ndarray) -> np.ndarray:
    return (kpts_xy[:,0]>0) & (kpts_xy[:,1]>0)

def centroid_from_kpts(kpts_xy: np.ndarray) -> Optional[Tuple[float,float]]:
    m = valid_mask(kpts_xy)
    if not np.any(m): return None
    pts = kpts_xy[m]
    return float(pts[:,0].mean()), float(pts[:,1].mean())

def bbox_from_kpts(kpts_xy: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    m = valid_mask(kpts_xy)
    if not np.any(m): return None
    pts = kpts_xy[m]
    x1,y1 = np.min(pts[:,0]), np.min(pts[:,1])
    x2,y2 = np.max(pts[:,0]), np.max(pts[:,1])
    return int(x1), int(y1), int(x2), int(y2)

def bbox_center(b): x1,y1,x2,y2 = b; return (0.5*(x1+x2), 0.5*(y1+y2))
def bbox_area(b): x1,y1,x2,y2 = b; return max(0,x2-x1)*max(0,y2-y1)

def iou(b1,b2)->float:
    if b1 is None or b2 is None: return 0.0
    ax1,ay1,ax2,ay2 = b1; bx1,by1,bx2,by2 = b2
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    union = bbox_area(b1)+bbox_area(b2)-inter + 1e-6
    return float(inter/union)

def torso_scale(k: np.ndarray) -> float:
    idxs = [(5,11),(6,12)]  # shoulder-hip pairs (0-based)
    d=[]
    for a,b in idxs:
        pa,pb = k[a],k[b]
        if pa[0]>0 and pa[1]>0 and pb[0]>0 and pb[1]>0:
            d.append(np.hypot(pa[0]-pb[0], pa[1]-pb[1]))
    if not d:
        b = bbox_from_kpts(k)
        if b is None: return 100.0
        x1,y1,x2,y2 = b
        return float(max(50.0, np.hypot(x2-x1,y2-y1)))
    return float(max(40.0, np.median(d)))

# -----------------------------
# Matching costs
# -----------------------------
def robust_kpt_distance(k1: np.ndarray, k2: np.ndarray) -> float:
    m = valid_mask(k1) & valid_mask(k2)
    if not np.any(m): return 1e3
    s = 0.5*(torso_scale(k1)+torso_scale(k2))
    d = np.linalg.norm(k1[m]-k2[m], axis=1)/(s+1e-6)
    med = np.median(d)
    clipped = np.clip(d,0,3.0)
    return float(0.6*med + 0.4*clipped.mean())

def composite_cost(k_det: np.ndarray, k_prev: np.ndarray,
                   prev_bbox: Optional[Tuple[int,int,int,int]],
                   pred_center: Optional[Tuple[float,float]],
                   frame_diag: float) -> float:
    k_cost = robust_kpt_distance(k_det, k_prev)
    b_det = bbox_from_kpts(k_det)
    c_det = bbox_center(b_det) if b_det else centroid_from_kpts(k_det)
    b_prev = prev_bbox
    c_prev = bbox_center(b_prev) if b_prev else centroid_from_kpts(k_prev)
    center_dist = 0.0
    if c_det and c_prev:
        center_dist = np.hypot(c_det[0]-c_prev[0], c_det[1]-c_prev[1])/(frame_diag+1e-6)
    iou_term = 1.0 - iou(b_det, b_prev)
    motion_term = 0.0
    if pred_center and c_det:
        motion_term = np.hypot(c_det[0]-pred_center[0], c_det[1]-pred_center[1])/(frame_diag+1e-6)
    α,β,γ,δ = 0.65, 0.20, 0.10, 0.05
    return float(α*k_cost + β*center_dist + γ*iou_term + δ*motion_term)

# -----------------------------
# Track classes
# -----------------------------
class Track:
    def __init__(self, tid:int, kpts: np.ndarray, frame_idx:int):
        self.id = tid
        self.kpts = kpts
        self.bbox = bbox_from_kpts(kpts)
        self.centroid = centroid_from_kpts(kpts)
        self.prev_centroid = None
        self.last_seen = frame_idx
    def predict_center(self):
        if self.centroid is None or self.prev_centroid is None: return self.centroid
        dx = self.centroid[0]-self.prev_centroid[0]
        dy = self.centroid[1]-self.prev_centroid[1]
        return (self.centroid[0]+dx, self.centroid[1]+dy)

# -----------------------------
# Tracker (ROI used only at init)
# -----------------------------
class TwoFencerKPTracker:
    """
    Frame 0: pick two fencers using middle-left/middle-right & edge-safe constraints.
    After that: track globally (no ROIs), matching by keypoint-level composite cost.
    """
    def __init__(self, w:int, h:int,
                 max_miss:int=25,
                 edge_margin_frac:float=0.06,     # reject near left/right edges
                 top_margin_frac:float=0.12,      # reject near top
                 bottom_margin_frac:float=0.20,   # reject near bottom (fisheye giant zone)
                 hcenter_sigma_frac:float=0.22):  # horizontal preference for 1/4 and 3/4 at init
        self.w,self.h = w,h
        self.frame_diag = float(np.hypot(w,h))
        self.max_miss = max_miss
        self.edge_margin = int(edge_margin_frac*w)
        self.y_top = int(top_margin_frac*h)
        self.y_bottom = int((1.0-bottom_margin_frac)*h)
        self.h_sigma = hcenter_sigma_frac*w
        self.tracks: Dict[int, Optional[Track]] = {0:None,1:None}
        self.frame_idx = -1
        self.initialized = False
        # gates
        self.cost_gate = 0.95    # if best match cost > gate, mark as miss
        self.swap_gate = 0.10    # avoid jittery identity swaps

    def _edge_safe(self, c: Tuple[float,float]) -> bool:
        x,y = c
        if x < self.edge_margin or x > (self.w - self.edge_margin): return False
        if y < self.y_top or y > self.y_bottom: return False
        return True

    def _init_score(self, k: np.ndarray, side: str) -> float:
        b = bbox_from_kpts(k)
        if b is None: return -1e6
        cx,cy = bbox_center(b)
        if not self._edge_safe((cx,cy)): return -1e6
        if side=="left" and cx >= self.w/2: return -1e6
        if side=="right" and cx <= self.w/2: return -1e6
        area = float(bbox_area(b))
        # prefer horizontally near quarter/three-quarter
        target_x = 0.25*self.w if side=="left" else 0.75*self.w
        hx = np.exp(-((cx-target_x)**2)/(2.0*self.h_sigma**2 + 1e-6))
        return area * (0.5 + 0.5*hx)

    def _pick_init(self, det_kpts: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        left_best, left_score = None, -1e9
        right_best, right_score = None, -1e9
        for k in det_kpts:
            sL = self._init_score(k,"left")
            if sL > left_score: left_score, left_best = sL, k
            sR = self._init_score(k,"right")
            if sR > right_score: right_score, right_best = sR, k
        # Ensure two distinct picks (if same, split by x)
        if left_best is not None and right_best is not None and np.all(left_best==right_best):
            b = bbox_from_kpts(left_best)
            if b:
                cx,_ = bbox_center(b)
                # pick an alternative on the opposite half
                alt = None; alt_score = -1e9
                for k in det_kpts:
                    if np.all(k==left_best): continue
                    s = self._init_score(k, "right" if cx<self.w/2 else "left")
                    if s > alt_score: alt_score, alt = s, k
                if cx < self.w/2: right_best = alt
                else: left_best = alt
        return left_best, right_best

    def initialize(self, det_kpts: List[np.ndarray]):
        L,R = self._pick_init(det_kpts)
        if L is None or R is None: return False
        self.tracks[0] = Track(0, L, self.frame_idx)
        self.tracks[1] = Track(1, R, self.frame_idx)
        self.initialized = True
        return True

    def update(self, det_kpts: List[np.ndarray]):
        self.frame_idx += 1
        if not self.initialized:
            self.initialize(det_kpts)
            return

        T0, T1 = self.tracks[0], self.tracks[1]
        if len(det_kpts)==0: return

        # Build full cost lists (no ROI now)
        pred0 = T0.predict_center() if T0 else None
        pred1 = T1.predict_center() if T1 else None
        costs0 = [composite_cost(k, T0.kpts, T0.bbox, pred0, self.frame_diag) for k in det_kpts] if T0 else []
        costs1 = [composite_cost(k, T1.kpts, T1.bbox, pred1, self.frame_diag) for k in det_kpts] if T1 else []

        # Choose best pair (i != j) globally (O(N^2))
        best_pair, best_total = None, 1e9
        n = len(det_kpts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                total = costs0[i] + costs1[j]
                if total < best_total:
                    best_pair, best_total = (i, j), total  # <-- FIXED order

        # Also compute the swapped assignment
        best_pair_swapped, best_total_swapped = None, 1e9
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                total = costs1[i] + costs0[j]
                if total < best_total_swapped:
                    best_pair_swapped, best_total_swapped = (i, j), total  # <-- FIXED order


        choose_normal = best_total + self.swap_gate < best_total_swapped

        # Apply gates per track; if too large, mark as miss for that track
        if choose_normal and best_pair is not None:
            i,j = best_pair
            c0, c1 = costs0[i], costs1[j]
            # Update T0
            if c0 <= self.cost_gate:
                T0.prev_centroid = T0.centroid
                T0.kpts = det_kpts[i]
                T0.bbox = bbox_from_kpts(T0.kpts)
                T0.centroid = centroid_from_kpts(T0.kpts)
                T0.last_seen = self.frame_idx
            # Update T1
            if c1 <= self.cost_gate:
                T1.prev_centroid = T1.centroid
                T1.kpts = det_kpts[j]
                T1.bbox = bbox_from_kpts(T1.kpts)
                T1.centroid = centroid_from_kpts(T1.kpts)
                T1.last_seen = self.frame_idx

        elif (not choose_normal) and best_pair_swapped is not None:
            i,j = best_pair_swapped
            c1, c0 = costs1[i], costs0[j]
            # Update T1 with i
            if c1 <= self.cost_gate:
                T1.prev_centroid = T1.centroid
                T1.kpts = det_kpts[i]
                T1.bbox = bbox_from_kpts(T1.kpts)
                T1.centroid = centroid_from_kpts(T1.kpts)
                T1.last_seen = self.frame_idx
            # Update T0 with j
            if c0 <= self.cost_gate:
                T0.prev_centroid = T0.centroid
                T0.kpts = det_kpts[j]
                T0.bbox = bbox_from_kpts(T0.kpts)
                T0.centroid = centroid_from_kpts(T0.kpts)
                T0.last_seen = self.frame_idx

        # Re-init if a track is gone too long and we have 2+ detections to re-pick
        for tid in (0,1):
            tr = self.tracks[tid]
            if tr and (self.frame_idx - tr.last_seen > self.max_miss):
                self.initialized = False
                self.initialize(det_kpts)
                break

# -----------------------------
# Runner
# -----------------------------
def run_fencer_tracking(
    video_path: str,
    output_video_path: str = "fencer_keypoints_overlay.mp4",
    model_path: str = "yolo11x-pose.pt",
    max_miss: int = 25,
    edge_margin_frac: float = 0.06,
    top_margin_frac: float = 0.12,
    bottom_margin_frac: float = 0.20,
    hcenter_sigma_frac: float = 0.22,
    verbose: bool = True
):
    if not os.path.exists(video_path): raise FileNotFoundError(video_path)
    print("Loading YOLO pose model...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total} frames @ {fps:.2f} FPS, {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w,h))
    if not out.isOpened():
        print("Falling back to MJPG/AVI writer...")
        out = cv2.VideoWriter("fencer_keypoints_overlay.avi",
                              cv2.VideoWriter_fourcc(*"MJPG"), fps, (w,h))

    tracker = TwoFencerKPTracker(
        w,h,max_miss=max_miss,
        edge_margin_frac=edge_margin_frac,
        top_margin_frac=top_margin_frac,
        bottom_margin_frac=bottom_margin_frac,
        hcenter_sigma_frac=hcenter_sigma_frac
    )

    per_frame=[]
    frame_idx=-1
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        r = model(frame, verbose=False)
        det_kpts=[]
        if len(r)>0 and r[0].keypoints is not None:
            k = r[0].keypoints.xy
            k = k.cpu().numpy() if hasattr(k,'cpu') else np.array(k)
            for i in range(k.shape[0]):
                det_kpts.append(k[i])

        tracker.update(det_kpts)

        canvas = frame.copy()
        t0,t1 = tracker.tracks[0], tracker.tracks[1]
        if t0 and t0.kpts is not None: draw_pose(canvas, t0.kpts, COLORS[0], LABELS[0])
        if t1 and t1.kpts is not None: draw_pose(canvas, t1.kpts, COLORS[1], LABELS[1])

        # HUD
        txt = f"Frame {frame_idx}/{total-1} | Det: {len(det_kpts)}"
        cv2.putText(canvas, txt, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,10,10), 3, cv2.LINE_AA)
        cv2.putText(canvas, txt, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 1, cv2.LINE_AA)

        out.write(canvas)
        per_frame.append({
            "frame_idx": frame_idx,
            "kpts_0": t0.kpts if t0 else None,
            "kpts_1": t1.kpts if t1 else None,
            "num_people": len(det_kpts)
        })

        if verbose and frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total} frames...")

    cap.release(); out.release()
    print("Done. Video saved to:", output_video_path if output_video_path.endswith(".mp4") else "fencer_keypoints_overlay.avi")
    both = sum(1 for pf in per_frame if (pf["kpts_0"] is not None and pf["kpts_1"] is not None))
    print(f"Frames with both fencers present: {both}/{len(per_frame)}")
    return per_frame

# Example:
video_path = "20251014_202890_phrase05.avi"
run_fencer_tracking(video_path, output_video_path="fencer_keypoints_overlay.mp4")
