# auto_slam_grid_explore01.py
# EKF + ArUco SLAM + grid exploration + fruit detection with semantic targets.
# Driving, planning, and avoidance are unchanged.

import sys, os, time, math, argparse, json
import numpy as np
import pygame
import cv2
from collections import deque

# ---- Your stack (unchanged) ----
sys.path.insert(0, "slam")
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure
import util.DatasetHandler as dh

# ---- YOLO detector (from operate.py) ----
from YOLO.detector import Detector

# ---------------- Tunables you may tweak quickly ----------------
HEADING_TOL_RAD          = np.deg2rad(20.0)
RECHECK_TURN_RAD         = np.deg2rad(20.0)
DIST_TOL_M               = 0.05
TURN_TICK                = 30
DRIVE_TICK               = 24
SAVE_INTERVAL_S          = 4.0

# Spins
SPIN_SECS_MIN            = 5.0
SPIN_ANGLE_MIN           = 2*np.pi
SPIN_TICK_SLOW           = 15
MICRO_SPIN_EVERY_M       = 0.60
MICRO_SPIN_SECS          = 2.0
MICRO_SPIN_ANGLE         = np.pi

# Status printing
PRINT_EVERY_S            = 1.5
PRINT_LM_TABLE_EVERY_S   = 2.5
PRINT_FRUIT_TABLE_EVERY_S= 2.0

# Avoidance (reactive)
AVOID_RADIUS_M           = 0.18
EMERGENCY_RADIUS_M       = 0.14
EMERGENCY_TURN_DEG       = 90.0
EMERGENCY_BACKOFF_S      = 1.5
EMERGENCY_COOLDOWN_S     = 2.0
EMERGENCY_DRIVE_FORWARD_S = 2.0
MAX_REPLAN_SEGMENTS      = 3

# Stuck watchdog
STUCK_WINDOW_S           = 2.2
STUCK_MIN_PROGRESS_M     = 0.04

# Planning grid defaults
DEFAULT_GRID_RES         = 0.15
BOUNDARY_KEEP_M          = 0.02
MARKER_KEEP_M            = 0.12
SKIP_WAYPOINT_IF_WITHIN  = 0.10

# Fruit detection defaults
DEFAULT_YOLO_MODEL       = "YOLO/model/best.pt"
CENTER_THRESHOLD         = 0.30    # 30 percent of half-width/height
MIN_BBOX_SIZE_RATIO      = 0.05    # 5 percent of image
MAX_BBOX_SIZE_RATIO      = 0.30    # 30 percent of image
MERGE_DIST_THRESHOLD_M   = 0.12
MIN_DETECTIONS_TO_MERGE  = 2
FRUIT_SAVE_INTERVAL_S    = 4.0     # save targets.json cadence

# ---- Confidence model (2D, χ² 95 percent ellipse) ----
CHI2_95 = 5.991

def compute_landmark_confidences(ekf, r_ref=0.10, k=CHI2_95):
    n = getattr(ekf, "number_landmarks", lambda: 0)()
    if n == 0:
        return {}
    tags = getattr(ekf, "taglist", list(range(1, n+1)))
    conf = {}
    A_ref = math.pi * k * (r_ref ** 2)
    for i in range(n):
        idx = 3 + 2 * i
        Sigma = ekf.P[idx:idx+2, idx:idx+2]
        det = float(np.linalg.det(Sigma))
        det = max(det, 0.0)
        area = math.pi * k * math.sqrt(det)
        c = math.exp(-area / max(1e-9, A_ref))
        key = f"aruco{int(tags[i]):02d}"
        conf[key] = c
    return conf

def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def bearing_to(p_from, p_to): return math.atan2(p_to[1]-p_from[1], p_to[0]-p_from[0])
def distance(a, b): return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def load_ekf(calib_dir, ip):
    K = np.loadtxt(os.path.join(calib_dir, "intrinsic.txt"), delimiter=',')
    D = np.loadtxt(os.path.join(calib_dir, "distCoeffs.txt"), delimiter=',')
    S = np.loadtxt(os.path.join(calib_dir, "scale.txt"), delimiter=',')
    if ip == 'localhost':
        S /= 2.0
    B = np.loadtxt(os.path.join(calib_dir, "baseline.txt"), delimiter=',')
    return EKF(Robot(B, S, K, D))

# ---- simple geometry ----
def segment_point_distance(p0, p1, q):
    p0 = np.array(p0); p1 = np.array(p1); q = np.array(q)
    v = p1 - p0
    if np.allclose(v, 0):
        return float(np.linalg.norm(q - p0))
    t = np.clip(((q - p0) @ v) / (v @ v), 0.0, 1.0)
    closest = p0 + t * v
    return float(np.linalg.norm(q - closest))

def path_blocking_landmark(ekf, p0, p1, radius):
    if ekf.number_landmarks() == 0:
        return None, None, float('inf'), float('inf')
    lms = ekf.markers
    tags = getattr(ekf, "taglist", list(range(1, lms.shape[1]+1)))
    best = (None, None, float('inf'), float('inf'))
    for i in range(lms.shape[1]):
        lm = (float(lms[0,i]), float(lms[1,i]))
        d_path = segment_point_distance(p0, p1, lm)
        d_robot = float(np.hypot(lm[0]-p0[0], lm[1]-p0[1]))
        if (d_path < radius) or (d_robot < radius):
            if (d_path < best[3]) or (abs(d_path - best[3]) < 1e-6 and d_robot < best[2]):
                best = (tags[i] if i < len(tags) else None, lm, d_robot, d_path)
    return best

def plan_detour_perp(robot_pose, target_xy, obstacle_xy, side_offset=0.18):
    px, py, th = robot_pose
    tx, ty = target_xy
    ox, oy = obstacle_xy
    r_to_t = np.array([tx - px, ty - py])
    r_to_o = np.array([ox - px, oy - py])
    side = +1.0 if (r_to_t[0]*r_to_o[1] - r_to_t[1]*r_to_o[0]) >= 0 else -1.0
    dir_ro = r_to_o / (np.linalg.norm(r_to_o) + 1e-9)
    perp = np.array([-dir_ro[1], dir_ro[0]])
    skirt = np.array([ox, oy]) + side * side_offset * perp
    return float(skirt[0]), float(skirt[1])

# ---------------- Grid generation (lawnmower) ----------------
def generate_lawnmower_grid(arena_w, arena_h, spacing, margin=0.05, start=(0.0,0.0)):
    xmin, xmax = -arena_w/2 + margin, arena_w/2 - margin
    ymin, ymax = -arena_h/2 + margin, arena_h/2 - margin
    ys = np.arange(ymin, ymax + 1e-6, spacing)
    xs = np.arange(xmin, xmax + 1e-6, spacing)
    wps = []
    flip = False
    for y in ys:
        if not flip:
            for x in xs: wps.append((float(x), float(y)))
        else:
            for x in xs[::-1]: wps.append((float(x), float(y)))
        flip = not flip
    wps.sort(key=lambda p: np.hypot(p[0]-start[0], p[1]-start[1]))
    return wps

# ---------------- A* lattice planner ----------------
def build_occupancy(arena_w, arena_h, res, markers, boundary_keep=BOUNDARY_KEEP_M, marker_keep=MARKER_KEEP_M):
    nx = int(round(arena_w / res)) + 1
    ny = int(round(arena_h / res)) + 1
    occ = np.zeros((ny, nx), dtype=np.uint8)

    rad_x = int(math.ceil(boundary_keep / res))
    rad_y = int(math.ceil(boundary_keep / res))
    occ[:rad_y, :] = 1
    occ[-rad_y:, :] = 1
    occ[:, :rad_x] = 1
    occ[:, -rad_x:] = 1

    if markers is not None and markers.shape[1] > 0:
        rr = int(math.ceil(marker_keep / res))
        yy, xx = np.indices(occ.shape)
        def to_idx(x, y):
            j = int(round((x + arena_w/2) / res))
            i = int(round((y + arena_h/2) / res))
            return i, j
        for k in range(markers.shape[1]):
            mx, my = float(markers[0,k]), float(markers[1,k])
            i, j = to_idx(mx, my)
            i0 = max(0, i-rr); i1 = min(ny-1, i+rr)
            j0 = max(0, j-rr); j1 = min(nx-1, j+rr)
            suby = yy[i0:i1+1, j0:j1+1]
            subx = xx[i0:i1+1, j0:j1+1]
            d2 = (subx - j)**2 + (suby - i)**2
            occ[i0:i1+1, j0:j1+1][d2 <= rr*rr] = 1
    return occ, nx, ny

def world_to_ij(x, y, arena_w, arena_h, res):
    j = int(round((x + arena_w/2) / res))
    i = int(round((y + arena_h/2) / res))
    return i, j

def ij_to_world(i, j, arena_w, arena_h, res):
    x = j * res - arena_w/2
    y = i * res - arena_h/2
    return float(x), float(y)

def astar(occ, arena_w, arena_h, res, start_xy, goal_xy):
    si, sj = world_to_ij(start_xy[0], start_xy[1], arena_w, arena_h, res)
    gi, gj = world_to_ij(goal_xy[0], goal_xy[1], arena_w, arena_h, res)
    ny, nx = occ.shape
    if not(0 <= si < ny and 0 <= sj < nx and 0 <= gi < ny and 0 <= gj < nx):
        return None
    if occ[si, sj] or occ[gi, gj]:
        return None

    NBR = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    cst = [1,1,1,1, math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)]

    def h(i,j):  return ((i-gi)**2 + (j-gj)**2)**0.5

    openset = {(si,sj)}
    g = { (si,sj): 0.0 }
    f = { (si,sj): h(si,sj) }
    parent = {}
    while openset:
        ci, cj = min(openset, key=lambda ij: f.get(ij, 1e18))
        if (ci, cj) == (gi, gj):
            path = []
            cur = (ci, cj)
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append((si, sj))
            path = path[::-1]
            return [ij_to_world(i, j, arena_w, arena_h, res) for (i,j) in path]

        openset.remove((ci, cj))
        for k, (di, dj) in enumerate(NBR):
            ni, nj = ci+di, cj+dj
            if not (0 <= ni < ny and 0 <= nj < nx): continue
            if occ[ni, nj]: continue
            tentative = g[(ci, cj)] + cst[k]
            if tentative < g.get((ni,nj), 1e18):
                parent[(ni,nj)] = (ci,cj)
                g[(ni,nj)] = tentative
                f[(ni,nj)] = tentative + h(ni,nj)
                openset.add((ni,nj))
    return None

# ---------------- Fruit utilities from operate.py (adapted) ----------------
FRUIT_CLASS_BASES = ['orange','lemon','pear','tomato','capsicum','potato','pumpkin','garlic']
VARIANT_TO_BASE = {
    'capsicum_red':'capsicum','capsicum_green':'capsicum','capsicum_yellow':'capsicum',
    'pear_green':'pear','pear_yellow':'pear',
    'pumpkin_green':'pumpkin','pumpkin_orange':'pumpkin'
}

def normalise_label(label: str) -> str:
    if label in VARIANT_TO_BASE:
        return VARIANT_TO_BASE[label]
    for b in FRUIT_CLASS_BASES:
        if label.startswith(b):
            return b
    return label

def fruit_dims_m():
    # height used for distance via pinhole (z); x,y are placeholders
    return {
        'orange':   [0.0756, 0.0767, 0.0729],
        'lemon':    [0.0540, 0.0740, 0.0536],
        'pear':     [0.0704, 0.0757, 0.1043],
        'tomato':   [0.0678, 0.0700, 0.0617],
        'capsicum': [0.0749, 0.0745, 0.0931],
        'potato':   [0.0677, 0.0940, 0.0566],
        'pumpkin':  [0.0848, 0.0823, 0.0867],
        'garlic':   [0.0578, 0.0645, 0.0747],
    }

def bbox_is_centered_and_sized(xywh, center_thr=CENTER_THRESHOLD,
                               min_ratio=MIN_BBOX_SIZE_RATIO,
                               max_ratio=MAX_BBOX_SIZE_RATIO):
    x, y, w, h = [float(v) for v in xywh]
    img_w, img_h = 320.0, 240.0
    cx = x + w/2.0; cy = y + h/2.0
    dx = abs(cx - img_w/2.0) / (img_w/2.0)
    dy = abs(cy - img_h/2.0) / (img_h/2.0)
    size_ratio = (w*h) / (img_w*img_h)
    centered = (dx < center_thr and dy < center_thr)
    sized = (size_ratio >= min_ratio and size_ratio <= max_ratio)
    return centered and sized

def detection_confidence(bbox_area, image_area, distance, optimal_distance=0.4):
    size_ratio = bbox_area / image_area
    if size_ratio < 0.05:
        size_score = size_ratio / 0.05
    elif size_ratio > 0.30:
        size_score = max(0, 1 - (size_ratio - 0.30) / 0.20)
    else:
        size_score = 1.0
    distance_score = np.exp(-((distance - optimal_distance) ** 2) / (2 * 0.15 ** 2))
    return (size_score * 0.6 + distance_score * 0.4)

# ---------------- Main class ----------------
class AutoGridSLAM:
    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.display.set_mode((400, 300))
        self.clock = pygame.time.Clock()

        # Robot I/O
        self.ppi = PenguinPi(args.ip, args.port)
        try:
            self.ppi.set_velocity([0,0], tick=0, turning_tick=0)
        except Exception as e:
            print(f"[FATAL] Could not reach robot at http://{args.ip}:{args.port} — {e}")
            raise

        # SLAM bits
        self.ekf = load_ekf(args.calib_dir, args.ip)
        self.det = aruco.aruco_detector(self.ekf.robot, marker_length=args.marker_size)
        self.out = dh.OutputWriter('lab_output')

        self.last_cmd_time = time.time()
        self.last_sent_lr = (0.0, 0.0)
        self.last_saved = time.time()
        self.last_print = 0.0
        self._last_lm_print = 0.0
        self._avoid_cooldown_until = 0.0

        # arena/grid
        self.arena_w = args.arena_w
        self.arena_h = args.arena_h
        self.grid_spacing = args.grid
        self.boundary_keep = args.boundary_keep
        self.marker_keep = args.marker_keep
        self.plan_res = args.plan_res

        # waypoints
        self.grid_wps = None
        self.wp_index = 0

        # path following
        self.active_path = []
        self.path_cursor = 0
        self.completed_paths = 0
        self.dist_since_spin = 0.0
        self.prev_pose_for_dist = None

        # render surface
        self.render_res = (520, 520)
        self.conf_ref_radius = args.conf_ref_radius

        # ---- Fruit detection state (no driving changes) ----
        self.fruit_enabled = (args.yolo_model != "" and args.yolo_model is not None)
        if self.fruit_enabled:
            print(f"[FRUIT] Loading YOLO model: {args.yolo_model}")
            self.fruit_det = Detector(args.yolo_model)
        else:
            self.fruit_det = None

        self.auto_detection = args.auto_detection
        self.center_thr = args.center_threshold
        self.min_bbox_ratio = args.min_bbox_ratio
        self.max_bbox_ratio = args.max_bbox_ratio

        # history and merged semantic targets
        self.detection_history = {}  # key -> {x,y,confidence,timestamp,range,bearing,class}
        self.merge_threshold = args.merge_dist_threshold
        self.min_detections = args.min_detections_to_merge
        self._last_fruit_print = 0.0
        self._last_targets_save = 0.0
        self.semantic_targets = {}  # class_i -> {x,y,uncertainty,confidence,n_detections}

    # -------- low-level motion --------
    def turn(self, ccw=True, slow=False):
        tick = SPIN_TICK_SLOW if slow else TURN_TICK
        l, r = self.ppi.set_velocity([0, +1 if ccw else -1], turning_tick=tick)
        return l, r

    def drive_fwd(self, speed_scale=1.0):
        tick = int(DRIVE_TICK * speed_scale)
        l, r = self.ppi.set_velocity([+1, 0], tick=tick)
        return l, r

    def stop(self):
        l, r = self.ppi.set_velocity([0, 0], tick=0, turning_tick=0)
        self.last_sent_lr = (0.0, 0.0)
        return l, r

    # -------- EKF plumbing --------
    def predict_from_last(self):
        now = time.time()
        dt = now - self.last_cmd_time
        if dt <= 0: dt = 1e-3
        l, r = self.last_sent_lr
        if self.args.ip != 'localhost':
            r = -r
        self.ekf.predict(measure.Drive(l, r, dt))
        self.last_cmd_time = now

    def sense_and_update(self):
        # single camera grab for both ArUco and YOLO
        img = self.ppi.get_image()

        # ArUco → EKF
        meas, _ = self.det.detect_marker_positions(img)
        if meas:
            self.ekf.add_landmarks(meas)
            self.ekf.update(meas)

        # Fruit detection side channel
        if self.fruit_det is not None:
            self.detect_and_update_fruits(img)

        # landmark table print
        if (time.time() - self._last_lm_print) >= PRINT_LM_TABLE_EVERY_S:
            if self.ekf.number_landmarks() > 0:
                tags = getattr(self.ekf, "taglist", [])
                lms = self.ekf.markers
                confs = compute_landmark_confidences(self.ekf, r_ref=self.conf_ref_radius)
                rows = []
                for i in range(lms.shape[1]):
                    tag = tags[i] if i < len(tags) else i+1
                    key = f"aruco{int(tag):02d}"
                    c = 100.0 * confs.get(key, 0.0)
                    rows.append(f"  {key}: ({lms[0,i]:+.3f}, {lms[1,i]:+.3f}) m | conf={c:.0f}%")
                print("[LM] EKF landmark table:\n" + "\n".join(rows))
            else:
                print("[LM] EKF landmark table: (none yet)")
            self._last_lm_print = time.time()

        # periodic fruit print and save
        if (time.time() - self._last_fruit_print) >= PRINT_FRUIT_TABLE_EVERY_S:
            self.print_fruits()
            self._last_fruit_print = time.time()
        if (time.time() - self._last_targets_save) >= FRUIT_SAVE_INTERVAL_S:
            self.save_targets()
            self._last_targets_save = time.time()

    def pose(self):
        x, y, th = self.ekf.robot.state.flatten()
        return float(x), float(y), float(wrap_to_pi(th))

    def print_pose_unc(self):
        x, y, th = self.pose()
        C = self.ekf.P[0:3,0:3]
        sx = float(np.sqrt(max(C[0,0], 0.0)))
        sy = float(np.sqrt(max(C[1,1], 0.0)))
        sth = float(np.sqrt(max(C[2,2], 0.0)))
        print(f"Pose x={x:.3f}, y={y:.3f}, theta={np.degrees(th):.1f} deg | sig_x={sx:.03f}, sig_y={sy:.03f}, sig_theta={np.degrees(sth):.1f} deg")

    def maybe_save_map(self, force=False):
        if force or (time.time() - self.last_saved) >= SAVE_INTERVAL_S:
            self.out.write_map(self.ekf)
            self.save_targets()
            self.last_saved = time.time()

    def pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: raise KeyboardInterrupt

    # -------- Fruit detection pipeline --------
    def estimate_pose_robust(self, bbox_xywh, class_name):
        K = self.ekf.robot.camera_matrix
        fy = float(K[1][1]); cx = float(K[0][2])

        x_c, y_c, w, h = [float(v) for v in bbox_xywh]
        if h <= 1e-6:
            return None

        dims = fruit_dims_m()
        base = normalise_label(class_name)
        if base not in dims:
            return None
        true_h = float(dims[base][2])

        dist = max(0.0, true_h * fy / h)
        bearing = np.arctan2((x_c - cx), fy)

        rx, ry, rtheta = [float(v) for v in self.ekf.robot.state[0:3]]
        world_theta = rtheta + bearing
        x_w = rx + dist * math.cos(world_theta)
        y_w = ry + dist * math.sin(world_theta)

        bbox_area = w * h
        img_area = 240.0 * 320.0
        conf = detection_confidence(bbox_area, img_area, dist)

        dist_unc = dist * 0.10
        ang_unc = (1.0 / max(conf, 1e-6)) * 0.05
        sigma = float(np.sqrt(dist_unc**2 + ang_unc**2))

        return {
            'class': base,
            'position': np.array([[x_w],[y_w]], dtype=float),
            'range': dist,
            'bearing': float(bearing),
            'confidence': float(conf),
            'uncertainty': float(sigma),
            'covariance': np.array([[sigma*sigma, 0.0],[0.0, sigma*sigma]], dtype=float)
        }

    def detect_and_update_fruits(self, rgb_img):
        # YOLO expects BGR; robot returns RGB
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        bboxes, _vis = self.fruit_det.detect_single_image(bgr)  # list of [label, np.asarray([x,y,w,h])]
        if not bboxes:
            return

        now = time.time()
        for label, xywh in bboxes:
            # optional centering gate
            if self.auto_detection:
                if not bbox_is_centered_and_sized(xywh, self.center_thr, self.min_bbox_ratio, self.max_bbox_ratio):
                    continue

            est = self.estimate_pose_robust(xywh, label)
            if est is None:
                continue

            # store in history with a simple growing key
            key = f"{est['class']}_{int(now*1000)}"
            self.detection_history[key] = {
                "x": float(est['position'][0,0]),
                "y": float(est['position'][1,0]),
                "confidence": float(est['confidence']),
                "uncertainty": float(est['uncertainty']),
                "timestamp": now,
                "range": float(est['range']),
                "bearing": float(est['bearing']),
                "class": est['class']
            }

        # prune old
        cutoff = now - 10.0
        self.detection_history = {k:v for k,v in self.detection_history.items() if v["timestamp"] > cutoff}

        # merge to semantic targets
        self.semantic_targets = self.merge_estimations_robust(self.detection_history,
                                                              distance_threshold=self.merge_threshold,
                                                              min_detections=self.min_detections)

    def merge_estimations_robust(self, target_pose_dict, distance_threshold=0.12, min_detections=2,
                                outlier_std_threshold=2.5):
        # dynamic range scaling for merge threshold
        BASE = max(0.08, distance_threshold)
        K_RANGE = 0.15  # meters of extra merge radius per meter of range
        MIN_CONF_TO_START = 0.12  # ignore extremely weak singletons

        # group by class
        grouped = {}
        for _, p in target_pose_dict.items():
            base = p["class"]
            grouped.setdefault(base, []).append({
                "x": float(p["x"]), "y": float(p["y"]),
                "conf": float(p["confidence"]), "range": float(p["range"])
            })

        class_clusters = {}
        for cls, pts in grouped.items():
            if not pts:
                continue

            # greedy clustering with dynamic threshold
            clusters = []
            for pt in sorted(pts, key=lambda q: -q["conf"]):
                placed = False
                for c in clusters:
                    cx = c["wx"] / max(c["ws"], 1e-9)
                    cy = c["wy"] / max(c["ws"], 1e-9)
                    # dynamic eps grows with current average range of the cluster
                    avg_r = c["wr"] / max(len(c["pts"]), 1)
                    eps = BASE + K_RANGE * avg_r
                    if np.hypot(pt["x"] - cx, pt["y"] - cy) <= eps:
                        c["pts"].append(pt)
                        c["wx"] += pt["x"] * pt["conf"]
                        c["wy"] += pt["y"] * pt["conf"]
                        c["ws"] += pt["conf"]
                        c["wr"] += pt["range"]
                        placed = True
                        break
                if not placed:
                    clusters.append({
                        "pts": [pt],
                        "wx": pt["x"] * pt["conf"], "wy": pt["y"] * pt["conf"],
                        "ws": pt["conf"], "wr": pt["range"]
                    })

            # consolidate clusters that ended up close after the first pass
            merged_again = []
            for c in clusters:
                cx = c["wx"] / max(c["ws"], 1e-9)
                cy = c["wy"] / max(c["ws"], 1e-9)
                rbar = c["wr"] / max(len(c["pts"]), 1)
                eps = BASE + K_RANGE * rbar
                absorbed = False
                for d in merged_again:
                    dx = d["wx"] / max(d["ws"], 1e-9)
                    dy = d["wy"] / max(d["ws"], 1e-9)
                    if np.hypot(cx - dx, cy - dy) <= eps:
                        d["pts"].extend(c["pts"])
                        d["wx"] += c["wx"]; d["wy"] += c["wy"]; d["ws"] += c["ws"]; d["wr"] += c["wr"]
                        absorbed = True
                        break
                if not absorbed:
                    merged_again.append(c)

            # robust averaging and filtering
            refined = []
            for c in merged_again:
                pts = c["pts"]; n = len(pts)
                xs = np.array([p["x"] for p in pts]); ys = np.array([p["y"] for p in pts]); cs = np.array([p["conf"] for p in pts])
                if n >= 3:
                    mx = np.median(xs); my = np.median(ys)
                    madx = np.median(np.abs(xs - mx)); mady = np.median(np.abs(ys - my))
                    zx = 0.6745 * (xs - mx) / (madx if madx > 0 else 1.0)
                    zy = 0.6745 * (ys - my) / (mady if mady > 0 else 1.0)
                    keep = np.sqrt(zx * zx + zy * zy) < outlier_std_threshold
                    if np.sum(keep) >= min_detections:
                        xs, ys, cs = xs[keep], ys[keep], cs[keep]
                        n = len(xs)

                # gate very weak one-off long range clusters
                if n < min_detections and float(np.mean(cs)) < MIN_CONF_TO_START:
                    continue

                w = cs / max(np.sum(cs), 1e-9)
                x_final = float(np.sum(xs * w)); y_final = float(np.sum(ys * w))
                std_x = float(np.std(xs)); std_y = float(np.std(ys))
                uncertainty = float(np.sqrt(std_x * std_x + std_y * std_y))
                avg_conf = float(np.mean(cs))
                refined.append({"x": x_final, "y": y_final, "uncertainty": uncertainty, "confidence": avg_conf, "n": int(n)})

            class_clusters[cls] = refined

        # build final dict
        targets = {}
        for cls, clusters in class_clusters.items():
            clusters.sort(key=lambda c: (-c["confidence"], c["n"]))
            for i, c in enumerate(clusters):
                targets[f"{cls}_{i}"] = {
                    "x": c["x"], "y": c["y"], "uncertainty": c["uncertainty"],
                    "confidence": c["confidence"], "n_detections": int(c["n"])
                }
        return targets


    def print_fruits(self):
        if not self.semantic_targets:
            print("[FRUIT] None detected or not enough evidence yet")
            return
        header = "[FRUIT]  {:10} {:>8} {:>8} {:>7} {:>6} {:>3}".format("name", "x[m]", "y[m]", "unc[m]", "conf", "n")
        print(header)
        for name in sorted(self.semantic_targets.keys()):
            d = self.semantic_targets[name]
            print("          {name:10} {x:+8.3f} {y:+8.3f} {unc:7.3f} {conf:6.2f} {n:3d}".format(
                name=name, x=d["x"], y=d["y"], unc=d["uncertainty"], conf=d["confidence"], n=d["n_detections"]))


    def save_targets(self):
        try:
            os.makedirs("lab_output", exist_ok=True)
            # flatten to just x,y for each target
            flat = {k: {"x": float(v["x"]), "y": float(v["y"])}
                    for k, v in self.semantic_targets.items()}
            with open("lab_output/targets.txt", "w", encoding="utf-8") as f:
                json.dump(flat, f, indent=4)
        except Exception as e:
            print(f"[FRUIT] Save targets.txt failed: {e}")


    # -------- spin routines --------
    def spin_relocalize(self, ccw=True, min_secs=SPIN_SECS_MIN, min_angle=SPIN_ANGLE_MIN, watchdog_s=12.0):
        print("[AUTO] Relocalization spin...")
        t0 = time.time()
        _, _, last_th = self.pose()
        accum = 0.0
        while True:
            self.pump_events()
            self.last_sent_lr = self.turn(ccw=ccw, slow=True)
            self.predict_from_last()
            self.sense_and_update()

            _, _, th = self.pose()
            dth = wrap_to_pi(th - last_th)
            accum += abs(dth)
            last_th = th

            if time.time() - self.last_print >= PRINT_EVERY_S:
                self.print_pose_unc()
                self.last_print = time.time()

            if (time.time() - t0) >= min_secs and accum >= min_angle: break
            if (time.time() - t0) >= watchdog_s:
                print("[AUTO] spin watchdog → continue")
                break
            self.maybe_save_map()
            self.clock.tick(30)

        self.last_sent_lr = self.stop()
        self.predict_from_last()
        self.sense_and_update()
        self.maybe_save_map()
        print("[AUTO] Spin done.")

    def micro_spin(self):
        t0 = time.time()
        _, _, last_th = self.pose()
        accum = 0.0
        while (time.time() - t0) < MICRO_SPIN_SECS and accum < MICRO_SPIN_ANGLE:
            self.pump_events()
            self.last_sent_lr = self.turn(ccw=True, slow=True)
            self.predict_from_last()
            self.sense_and_update()
            _, _, th = self.pose()
            accum += abs(wrap_to_pi(th - last_th))
            last_th = th
            self.clock.tick(30)
        self.stop()

    # -------- planning helpers --------
    def build_occ_and_plan(self, start_xy, goal_xy):
        markers = self.ekf.markers.copy() if self.ekf.number_landmarks() > 0 else np.zeros((2,0))
        occ, nx, ny = build_occupancy(
            self.arena_w, self.arena_h, self.plan_res, markers,
            boundary_keep=self.boundary_keep, marker_keep=self.marker_keep
        )
        path = astar(occ, self.arena_w, self.arena_h, self.plan_res, start_xy, goal_xy)
        return occ, path

    def save_path_snapshot(self, occ, path, start_xy, goal_xy, fname):
        w, h = self.render_res
        surf = pygame.Surface(self.render_res)
        surf.fill((235,235,235))
        ny, nx = occ.shape
        cell_w = w / nx
        cell_h = h / ny
        for i in range(ny):
            for j in range(nx):
                if occ[i,j]:
                    pygame.draw.rect(surf, (90,90,90), pygame.Rect(j*cell_w, i*cell_h, cell_w, cell_h))
        if path and len(path) > 1:
            def to_px(x,y):
                i, j = world_to_ij(x,y,self.arena_w,self.arena_h,self.plan_res)
                return int(j*cell_w+cell_w/2), int(i*cell_h+cell_h/2)
            pts = [to_px(*p) for p in path]
            pygame.draw.lines(surf, (0,120,255), False, pts, 3)
        def to_px(x,y):
            i, j = world_to_ij(x,y,self.arena_w,self.arena_h,self.plan_res)
            return int(j*cell_w+cell_w/2), int(i*cell_h+cell_h/2)
        pygame.draw.circle(surf, (0,180,0), to_px(*start_xy), 6)
        pygame.draw.circle(surf, (220,0,0), to_px(*goal_xy), 6)
        os.makedirs("lab_output", exist_ok=True)
        pygame.image.save(surf, fname)

    # -------- main exploration logic --------
    def generate_and_filter_grid(self):
        wps = generate_lawnmower_grid(self.arena_w, self.arena_h, self.grid_spacing, margin=self.boundary_keep)
        if self.ekf.number_landmarks() == 0:
            self.grid_wps = wps
            return
        kept = []
        for p in wps:
            too_close = False
            for k in range(self.ekf.markers.shape[1]):
                mx, my = float(self.ekf.markers[0,k]), float(self.ekf.markers[1,k])
                if np.hypot(p[0]-mx, p[1]-my) < SKIP_WAYPOINT_IF_WITHIN:
                    too_close = True; break
            if not too_close:
                kept.append(p)
        self.grid_wps = kept

    def follow_path_segmentwise(self, goal_xy):
        x, y, th = self.pose()
        start_xy = (x, y)
        occ, path = self.build_occ_and_plan(start_xy, goal_xy)
        trial = 0
        while path is None and trial < MAX_REPLAN_SEGMENTS:
            tag, lm, _, _ = path_blocking_landmark(self.ekf, start_xy, goal_xy, max(self.marker_keep, AVOID_RADIUS_M))
            if lm is None:
                side_xy = (x + 0.20*np.cos(th + np.pi/2), y + 0.20*np.sin(th + np.pi/2))
            else:
                side_xy = plan_detour_perp((x,y,th), goal_xy, lm, side_offset=max(0.18, self.marker_keep*1.2))
            _, p1 = self.build_occ_and_plan(start_xy, side_xy)
            _, p2 = self.build_occ_and_plan(side_xy, goal_xy)
            if p1 and p2:
                path = p1[:-1] + p2
                break
            trial += 1

        snap_name = f"lab_output/path_{self.completed_paths:02d}.png"
        self.save_path_snapshot(occ, path, start_xy, goal_xy, snap_name)
        print(f"[PLAN] Saved path snapshot: {snap_name}")

        if not path:
            print("[PLAN] No valid path; skipping this waypoint.")
            return False

        self.active_path = path
        self.path_cursor = 0
        if self.prev_pose_for_dist is None:
            self.prev_pose_for_dist = (x,y)
        while self.path_cursor < len(self.active_path):
            self.pump_events()
            self.predict_from_last()
            self.sense_and_update()

            px, py, th = self.pose()
            subgoal = self.active_path[self.path_cursor]
            if distance((px,py), subgoal) <= max(DIST_TOL_M, 0.8*self.plan_res):
                self.path_cursor += 1
                continue

            if time.time() >= self._avoid_cooldown_until:
                tag_blk, lm_blk, d_robot_blk, _ = path_blocking_landmark(
                    self.ekf, (px,py), subgoal, EMERGENCY_RADIUS_M
                )
                if lm_blk is not None and d_robot_blk < EMERGENCY_RADIUS_M:
                    print(f"[AVOID] EMERGENCY near landmark at {lm_blk} (d={d_robot_blk:.2f})")
                    self.stop()
                    l, r = self.ppi.set_velocity([-1, 0], tick=DRIVE_TICK)
                    rr = -r if self.args.ip != 'localhost' else r
                    self.ekf.predict(measure.Drive(l, rr, EMERGENCY_BACKOFF_S))
                    time.sleep(EMERGENCY_BACKOFF_S)
                    self.stop()
                    px, py, th = self.pose()
                    ox, oy = lm_blk
                    r_to_o = np.array([ox-px, oy-py])
                    r_to_t = np.array([subgoal[0]-px, subgoal[1]-py])
                    side = +1.0 if (r_to_t[0]*r_to_o[1] - r_to_t[1]*r_to_o[0]) >= 0 else -1.0
                    t = max(0.15, 0.6 * (EMERGENCY_TURN_DEG/90.0))
                    l, r = self.ppi.set_velocity([0, +1 if (side<0) else -1], turning_tick=TURN_TICK)
                    rr = -r if self.args.ip != 'localhost' else r
                    self.ekf.predict(measure.Drive(l, rr, t))
                    time.sleep(t)
                    self.stop()
                    l, r = self.ppi.set_velocity([+1, 0], tick=DRIVE_TICK)
                    rr = -r if self.args.ip != 'localhost' else r
                    self.ekf.predict(measure.Drive(l, rr, EMERGENCY_DRIVE_FORWARD_S))
                    time.sleep(EMERGENCY_DRIVE_FORWARD_S)
                    self.stop()
                    self._avoid_cooldown_until = time.time() + EMERGENCY_COOLDOWN_S
                    return self.follow_path_segmentwise(goal_xy)

            des = bearing_to((px,py), subgoal)
            err = wrap_to_pi(des - th)
            if abs(err) > HEADING_TOL_RAD:
                self.last_sent_lr = self.turn(ccw=(err > 0))
            else:
                self.last_sent_lr = self.drive_fwd()
                if abs(err) > RECHECK_TURN_RAD:
                    self.stop()

            moved = distance(self.prev_pose_for_dist, (px,py))
            if moved >= MICRO_SPIN_EVERY_M:
                self.stop()
                self.micro_spin()
                self.prev_pose_for_dist = (px,py)

            if time.time() - self.last_print >= PRINT_EVERY_S:
                self.print_pose_unc()
                self.last_print = time.time()

            self.maybe_save_map()
            self.clock.tick(30)

        self.stop()
        self.completed_paths += 1
        return True

    # -------- run --------
    def run(self):
        print("[AUTO] Initial spin to seed landmarks ...")
        self.spin_relocalize(ccw=True)

        self.generate_and_filter_grid()
        if not self.grid_wps or len(self.grid_wps) == 0:
            self.grid_wps = generate_lawnmower_grid(self.arena_w, self.arena_h, self.grid_spacing, margin=self.boundary_keep)

        print(f"[AUTO] Grid has {len(self.grid_wps)} candidate waypoints.")

        for idx, wp in enumerate(self.grid_wps, 1):
            skip = False
            for k in range(self.ekf.markers.shape[1]):
                mx, my = float(self.ekf.markers[0,k]), float(self.ekf.markers[1,k])
                if np.hypot(wp[0]-mx, wp[1]-my) < SKIP_WAYPOINT_IF_WITHIN:
                    skip = True; break
            if skip:
                print(f"[AUTO] WP {idx}: too close to a marker → skip")
                continue

            print(f"[AUTO] → WP {idx}/{len(self.grid_wps)} {wp}")
            ok = self.follow_path_segmentwise(wp)
            self.micro_spin()

        print("[AUTO] Returning to origin (0,0)...")
        self.follow_path_segmentwise((0.0, 0.0))
        self.spin_relocalize(ccw=True, min_secs=3.0, min_angle=np.pi)

        self.stop()
        self.predict_from_last()
        self.maybe_save_map(force=True)
        print("[AUTO] Finished. Map saved to lab_output/slam.txt and fruits to lab_output/targets.json")

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", type=str, default="192.168.50.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--calib_dir", type=str, default="calibration/param/")
    # Arena
    ap.add_argument("--arena_w", type=float, default=2.20, help="Arena width (m)")
    ap.add_argument("--arena_h", type=float, default=2.20, help="Arena height (m)")
    ap.add_argument("--grid", type=float, default=0.30, help="Waypoint spacing (m)")
    # Planner lattice & keep-outs
    ap.add_argument("--plan_res", type=float, default=DEFAULT_GRID_RES, help="Planner grid resolution (m/cell)")
    ap.add_argument("--boundary_keep", type=float, default=BOUNDARY_KEEP_M, help="Keep-out near boundary (m)")
    ap.add_argument("--marker_keep", type=float, default=MARKER_KEEP_M, help="Keep-out around markers (m)")
    ap.add_argument("--marker_size", type=float, default=0.07, help="ArUco marker size (m) for detection")
    ap.add_argument("--conf_ref_radius", type=float, default=0.10, help="Reference radius for LM confidence")
    # Fruit detection args
    ap.add_argument("--yolo_model", type=str, default=DEFAULT_YOLO_MODEL, help="Path to YOLO model, empty to disable")
    ap.add_argument("--auto_detection", action="store_true", help="Enable center-and-size gate for stable detects")
    ap.add_argument("--center_threshold", type=float, default=CENTER_THRESHOLD)
    ap.add_argument("--min_bbox_ratio", type=float, default=MIN_BBOX_SIZE_RATIO)
    ap.add_argument("--max_bbox_ratio", type=float, default=MAX_BBOX_SIZE_RATIO)
    ap.add_argument("--merge_dist_threshold", type=float, default=MERGE_DIST_THRESHOLD_M)
    ap.add_argument("--min_detections_to_merge", type=int, default=MIN_DETECTIONS_TO_MERGE)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    bot = None
    try:
        bot = AutoGridSLAM(args)
        bot.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping robot and exiting...")
        if bot is not None:
            try: bot.stop()
            except: pass
    except Exception as e:
        print(f"[ERROR] {e}")
        if bot is not None:
            try: bot.stop()
            except: pass

# Example:
# python auto_slam_grid_explore01.py --ip 192.168.50.1 --port 8080 --arena_w 2.2 --arena_h 2.2 --grid 0.30 --plan_res 0.05 --boundary_keep 0.05 --marker_keep 0.12 --conf_ref_radius 0.10 --yolo_model YOLO/model/best.pt --auto_detection
