# M4 - Autonomous fruit searching with A* path planning to standoff (≤ 0.3 m)
# Keeps your original localization + low-level commands/state machine

import sys, os, cv2, numpy as np, json, argparse, time, pygame
sys.path.insert(0, "slam")
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure
import util.DatasetHandler as dh

import pygame
from Helper import *
from math import hypot, atan2, pi

# ---- A* grid planner bits ----
from path_planning_astar import (
    build_costmap_fixed_squares,
    plan_leg_astar,
    smooth_polyline,
    make_collision_checked_path,
    world_to_grid, grid_to_world,
    visualize_costmap_detailed,          # <-- add
    visualize_plan_with_robot_footprint  # <-- optional, for per-leg preview
)

ARENA_SIZE = 2.4       # m (square)
GRID_RES   = 0.02      # m/cell (finer = more precise)
ROBOT_R    = 0.09      # m
SAFETY_M   = 0.05      # m
STANDOFF_R = 0.2      # m  (<= 0.30 m from fruit)
HOLD_SECS  = 5.0


# for localisation 
SIGMA_POS_BAD = 0.45   # m (σx+σy)
SIGMA_TH_BAD  = np.deg2rad(50)  # rad


# --- spin relocalization tuning ---
SPIN_STEP_DEG        = 30.0     # turn this many degrees per step
SPIN_BURST_FRAMES    = 3        # frames to grab while stationary at each stop
SPIN_BURST_DT        = 0.05     # seconds between those frames (~camera frame time)
SPIN_SLOW_TURN_TICK  = 15       # slower turning "tick" while stepping (smaller = slower)


# ---------- helpers from your original ----------
def wrap_to_pi(a): return (a + np.pi) % (2*np.pi) - np.pi
def smallest_angle_diff(target, current): return wrap_to_pi(target - current)

def read_true_map(fname):
    with open(fname, 'r') as fd:
        gt = json.load(fd)
        fruit_list, fruit_true_pos = [], []
        aruco_true_pos = np.empty([10, 2])
        for key in gt:
            x, y = float(gt[key]['x']), float(gt[key]['y'])
            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9] = [x, y]
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id] = [x, y]
            else:
                fruit_list.append(key[:-2])
                fruit_true_pos = np.vstack([fruit_true_pos, [x, y]]) if len(fruit_true_pos) else np.array([[x, y]])
        return fruit_list, fruit_true_pos, aruco_true_pos

def read_search_list(fname='search_list.txt'):
    order = []
    with open(fname, 'r') as fd:
        for line in fd:
            line = line.strip()
            if line: order.append(line)
    return order

def targets_from_search_list(search_list, fruit_list, fruit_true_pos):
    name_to_positions, all_named = {}, []
    for name, (x, y) in zip(fruit_list, fruit_true_pos):
        pos = (float(x), float(y))
        all_named.append((name, pos))
        name_to_positions.setdefault(name, []).append(pos)

    import math
    name_to_closest = {name: min(lst, key=lambda p: math.hypot(p[0], p[1]))
                       for name, lst in name_to_positions.items()}

    targets_xy, target_names, used = [], [], set()
    print("\nSearch order (closest instance for each name):")
    k = 1
    for nm in search_list:
        if nm in name_to_closest:
            pos = name_to_closest[nm]
            targets_xy.append(pos); target_names.append(nm); used.add(nm)
            print(f" {k}) {nm} at [{pos[0]:.3f}, {pos[1]:.3f}]"); k += 1

    distractors = []
    for nm, pos in all_named:
        if nm not in used and pos not in distractors:
            distractors.append(pos)

    if distractors:
        print("\nDistractors (not in search list):")
        for i, p in enumerate(distractors, 1):
            print(f"  {i}) at [{p[0]:.3f}, {p[1]:.3f}]")
    else:
        print("\nNo distractors.")

    return targets_xy, target_names, distractors

def init_ekf(calib_dir, ip):
    K = np.loadtxt(os.path.join(calib_dir, "intrinsic.txt"), delimiter=',')
    D = np.loadtxt(os.path.join(calib_dir, "distCoeffs.txt"), delimiter=',')
    S = np.loadtxt(os.path.join(calib_dir, "scale.txt"), delimiter=',')
    if ip == 'localhost': S /= 2
    B = np.loadtxt(os.path.join(calib_dir, "baseline.txt"), delimiter=',')
    return EKF(Robot(B, S, K, D))

def load_map_to_ekf(ekf, aruco_true_pos):
    ekf.markers = aruco_true_pos.T
    ekf.taglist = list(range(1, 11))
    n = ekf.number_landmarks()
    if n > 0:
        total_size = 3 + 2 * n
        new_P = np.zeros((total_size, total_size))
        new_P[0:3, 0:3] = ekf.P[0:3, 0:3]
        new_P[3:, 3:] = 0.001 * np.eye(2 * n)
        ekf.P = new_P
    print(f"Loaded {n} known landmarks")

# ---------- planning utilities ----------
def build_planning_grid(aruco_xy, distractor_xy, target_xy):
    # Costmap with square inflation (fits physical cubes/markers nicely)
    fruit_points_m = np.array(target_xy, dtype=np.float64) if len(target_xy) > 0 else None
    cost, occ, meta = build_costmap_fixed_squares(
        size_m=ARENA_SIZE,
        aruco_points_m=aruco_xy,
        fruit_points_m=fruit_points_m,
        res=GRID_RES,
        robot_radius_m=ROBOT_R,
        safety_margin_m=SAFETY_M,
        aruco_size_m=0.082,
        fruit_size_m=0.06
    )
    return cost, occ, meta

def plan_standoff_path(costmap, meta, start_xy, fruit_xy, radius=STANDOFF_R, nsamp=24,
                       smooth_lam=0.25, smooth_iters=20, downsample_step=8):
    """Pick best free standoff point around fruit, plan A*, smooth + downsample."""
    import math
    candidates = []
    for th in np.linspace(0.0, 2.0*math.pi, nsamp, endpoint=False):
        cx = float(fruit_xy[0] + radius * np.cos(th))
        cy = float(fruit_xy[1] + radius * np.sin(th))
        gx, gy = world_to_grid(cx, cy, meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
        if costmap[gy, gx] >= 255:
            continue
        poly, gcost = plan_leg_astar(costmap, meta, start_xy, (cx, cy))
        if poly:
            candidates.append((gcost, (cx, cy), poly))
    if not candidates:
        return None, None

    candidates.sort(key=lambda t: t[0])
    best_cost, standoff_xy, raw_poly = candidates[0]

    # smooth + safety refinement
    path = smooth_polyline(raw_poly, lam=smooth_lam, iters=smooth_iters)
    safe = make_collision_checked_path(costmap, meta, path, max_allowed_cost=254, sample_step_cells=0.5)
    if safe is None: safe = path

    # downsample for fewer stops (still fine for your TURN/DRIVE FSM)
    sampled = safe[::max(1, int(downsample_step))]
    if sampled[-1] != safe[-1]:
        sampled.append(safe[-1])

    return sampled, standoff_xy

# ---------- system ----------
class LocalizationSystem:
    """Your original system with a small extension to consume planned paths."""

    def __init__(self, args, aruco_true_pos, waypoints,
                 target_names=None, hold_secs=HOLD_SECS):
        self.args = args
        self.ppi = PenguinPi(args.ip, args.port)
        self.ekf = init_ekf(args.calib_dir, args.ip)
        load_map_to_ekf(self.ekf, aruco_true_pos)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.08)

        # localization state
        self.ekf_on = True
        self.last_print_time = time.time()
        self.print_interval = 2.0
        self.control_clock = time.time()

        # path-following state (list of per-leg waypoint polylines)
        self.legs = [waypoints] if (waypoints and len(waypoints)>0 and isinstance(waypoints[0], (list, tuple)) and isinstance(waypoints[0][0], (int,float))) else waypoints
        self.leg_idx = 0
        self.wp_idx = 0
        self.state = 'TURN_TO_TARGET' if self.total_wps() > 0 else 'DONE'
        self.spin_accum = 0.0
        self.last_theta = None

        # for UI
        self.current_fruit_name = None
        self.target_names = target_names or []
        self.hold_secs = hold_secs
        self.holding_until = None

        # flags
        self.finished = False

        if self.state != 'DONE':
            print(f"[AUTO] Loaded {self.total_wps()} waypoint(s) across {len(self.legs)} leg(s). Starting...")

    # --- legs/waypoints helpers ---
    def total_wps(self):
        if not self.legs: return 0
        return sum(len(leg) for leg in self.legs)

    def current_wp(self):
        if self.leg_idx >= len(self.legs): return None
        leg = self.legs[self.leg_idx]
        if self.wp_idx >= len(leg): return None
        return leg[self.wp_idx]

    def advance_wp(self):
        self.wp_idx += 1
        leg = self.legs[self.leg_idx]
        if self.wp_idx >= len(leg):
            # finished this leg -> hold if requested, then move to next leg
            self.wp_idx = 0
            self.leg_idx += 1
            if self.leg_idx < len(self.legs):
                # announce next fruit name if provided
                if self.leg_idx < len(self.target_names):
                    self.current_fruit_name = self.target_names[self.leg_idx]
                print(f"[AUTO] Next leg {self.leg_idx+1}/{len(self.legs)}")
            else:
                self.state = 'DONE'
                print("[AUTO] All legs finished.")
                self.finished = True

    # --- localization plumbing ---
    def get_robot_pose(self):
        x, y, th = self.ekf.robot.state.flatten()
        return np.array([x, y, wrap_to_pi(th)])
    
    # helper that RETURNS (σx, σy, σθ) instead of printing
    def get_uncertainty(self):
        C = self.ekf.P[0:3, 0:3]
        sx  = float(np.sqrt(max(C[0,0], 0.0)))
        sy  = float(np.sqrt(max(C[1,1], 0.0)))
        sth = float(np.sqrt(max(C[2,2], 0.0)))
        return sx, sy, sth

    # inside LocalizationSystem
    def print_robot_uncertainty(self):
        # top-left 3x3 is robot pose covariance
        C = self.ekf.P[0:3, 0:3]
        sx = float(np.sqrt(max(C[0,0], 0)))       # m (std dev x)
        sy = float(np.sqrt(max(C[1,1], 0)))       # m (std dev y)
        sth = float(np.sqrt(max(C[2,2], 0)))      # rad (std dev heading)
        print(f"Uncertainty: σx={sx:.03f} m, σy={sy:.03f} m, σθ={np.degrees(sth):.1f}°")


    def print_robot_pose(self):
        x, y, th = self.get_robot_pose()
        print(f"                                              Robot Pose: x={x:.3f} m, y={y:.3f} m, θ={np.degrees(th):.1f}°")

    def take_pic(self): return self.ppi.get_image()

    def update_localization(self, img):
        measurements, _ = self.aruco_det.detect_marker_positions(img)

        # TO DEBUG HOW MANY THINGS IT SEES 
        # print(f"[OBS] tags_seen={len(measurements)}")
        if len(measurements) > 0:
            ids = [m.tag for m in measurements]
            dists = [float(np.linalg.norm(m.position)) for m in measurements]
            # print(f"[OBS] ids={ids}  dists={['%.2f'%d for d in dists]}")


        if self.ekf_on and measurements:
            self.ekf.update(measurements)

    # --- low-level motion (as in your M4 script) ---
    def send_turn(self, direction):
        return self.ppi.set_velocity([0, int(np.sign(direction))], turning_tick=20)
    

    # slow, single-tick turn command; direction = +1 (ccw) or -1 (cw) - for reloclising
    def send_turn_slow(self, direction):
        # use a smaller turning_tick to rotate more slowly, improving detector SNR
        return self.ppi.set_velocity([0, int(np.sign(direction))], turning_tick=SPIN_SLOW_TURN_TICK)

    def send_drive(self, direction):
        return self.ppi.set_velocity([int(np.sign(direction)), 0], tick=30)

    def send_stop(self): return self.ppi.set_velocity([0, 0])

    def control_predict_from_last_cmd(self, lv, rv):
        dt = time.time() - self.control_clock
        self.control_clock = time.time()
        if self.args.ip != 'localhost':
            rv = -rv
        drive_meas = measure.Drive(lv, rv, dt)
        if self.ekf_on:
            self.ekf.predict(drive_meas)

    # --- auto waypoint follower (unchanged logic, just fed with planned WPs) ---
    def auto_step(self):
        # Hold timer (at fruit standoff)
        if self.holding_until is not None:
            if time.time() >= self.holding_until:
                print("[AUTO] Hold complete. Continuing…")
                self.holding_until = None
                # proceed to next leg if we just finished a leg
                self.advance_wp()
                if self.state == 'DONE':
                    self.send_stop(); return
                self.state = 'TURN_TO_TARGET'
            else:
                self.send_stop()
                return

        # End condition
        if self.state == 'DONE' or self.leg_idx >= len(self.legs):
            if not self.finished:
                self.send_stop(); print("[AUTO] Done with all waypoints."); self.finished = True
            return
        
        if self.state in ('TURN_TO_TARGET', 'DRIVE_TO_TARGET'):
            sx, sy, sth = self.get_uncertainty()   # see helper below
            if (sx + sy) > SIGMA_POS_BAD or sth > SIGMA_TH_BAD:
                print("[AUTO] Pose uncertainty high → pausing to re-localize…")
                self.send_stop()
                # grab a few stationary frames; this collapses covariance if tags are visible
                for _ in range(6):                # ~0.3 s total (6 * 0.05)
                    img = self.take_pic()
                    self.update_localization(img)
                    time.sleep(0.05)
                # Optional: escalate to a slow spin if still bad
                sx, sy, sth = self.get_uncertainty()
                if (sx + sy) > SIGMA_POS_BAD or sth > SIGMA_TH_BAD:
                    self.state = 'SPIN_RELOCALIZE'
                    self.spin_accum = 0.0
                    self.step_accum = 0.0
                    self.spin_pause_until = None
                    self.last_theta = self.get_robot_pose()[2]
                    return
    # === end uncertainty gate ===


        # Get pose and current waypoint
        x, y, th = self.get_robot_pose()
        wp = self.current_wp()
        if wp is None:
            # finished leg -> start 5 s hold here (standoff reached)
            fruit_name = self.current_fruit_name or "fruit"
            print(f"[AUTO] Arrived at {fruit_name}. Holding {self.hold_secs:.1f}s…")
            self.holding_until = time.time() + self.hold_secs
            self.send_stop()
            return

        tx, ty = wp
        dx, dy = tx - x, ty - y
        distance = np.hypot(dx, dy)
        target_heading = np.arctan2(dy, dx)
        heading_err = wrap_to_pi(target_heading - th)

        HEADING_TOL = np.deg2rad(0.8)
        DIST_TOL    = 0.03  # 2 cm per WP is tight but OK with downsampled WPs
        FULL_TURN   = 2 * np.pi

        if self.state == 'TURN_TO_TARGET':
            if abs(heading_err) > HEADING_TOL:
                turn_dir = 1 if heading_err > 0 else -1
                lv, rv = self.send_turn(turn_dir)
                self.control_predict_from_last_cmd(lv, rv)
            else:
                self.send_stop()
                self.state = 'DRIVE_TO_TARGET'
                print(f"[AUTO] Driving to wp L{self.leg_idx+1}#{self.wp_idx+1}, dist={distance:.2f} m")

        elif self.state == 'DRIVE_TO_TARGET':
            if distance > DIST_TOL:
                if abs(heading_err) > np.deg2rad(15):  # re-align if drifted
                    self.send_stop()
                    self.state = 'TURN_TO_TARGET'
                    print("[AUTO] Heading drift detected, re-aligning…")
                else:
                    lv, rv = self.send_drive(+1)
                    self.control_predict_from_last_cmd(lv, rv)
            else:
                self.send_stop()
                # last waypoint of leg? If yes, spin relocalize and then HOLD.
                leg = self.legs[self.leg_idx]
                at_leg_end = (self.wp_idx == len(leg)-1)
                if at_leg_end:
                    self.state = 'SPIN_RELOCALIZE'
                    self.spin_accum = 0.0
                    self.last_theta = th
                    print(f"[AUTO] Final wp for this fruit reached. Spinning to re-localize…")
                else:
                    # advance to next waypoint immediately
                    self.wp_idx += 1
                    self.state = 'TURN_TO_TARGET'

        elif self.state == 'SPIN_RELOCALIZE':
            FULL_TURN = 2 * np.pi

            # lazy-init step state the first time we enter this state
            if not hasattr(self, "spin_pause_until"):
                self.spin_pause_until = None
            if not hasattr(self, "spin_target_heading"):
                self.spin_target_heading = None
            if not hasattr(self, "spin_dir"):
                self.spin_dir = +1  # default CCW

            x, y, th = self.get_robot_pose()

            # Are we currently in a stationary "burst" (take frames) phase?
            now = time.time()
            if self.spin_pause_until is not None and now < self.spin_pause_until:
                # still pausing: just keep taking frames to collapse covariance
                img = self.take_pic()
                self.update_localization(img)
                time.sleep(SPIN_BURST_DT)
                return
            elif self.spin_pause_until is not None and now >= self.spin_pause_until:
                # pause finished -> clear and prepare next step
                self.spin_pause_until = None
                self.spin_target_heading = None  # next step will pick a new target

            # If we don't have a step target yet, compute one and start a short turn
            if self.spin_target_heading is None:
                # choose direction that generally increases accumulated angle
                self.spin_dir = +1  # or decide dynamically if you want
                step_rad = np.deg2rad(SPIN_STEP_DEG) * self.spin_dir
                self.spin_target_heading = wrap_to_pi(th + step_rad)

            # Check if we've reached the step target heading
            heading_err = wrap_to_pi(self.spin_target_heading - th)
            if abs(heading_err) > np.deg2rad(1.5):  # need more turn to hit the step target
                lv, rv = self.send_turn_slow(self.spin_dir)
                self.control_predict_from_last_cmd(lv, rv)
                # accumulate absolute angle turned
                if self.last_theta is None:
                    self.last_theta = th
                th_new = self.get_robot_pose()[2]
                dth = wrap_to_pi(th_new - self.last_theta)
                self.spin_accum += abs(dth)
                self.last_theta = th_new
                return
            else:
                # Step reached -> STOP and take a burst of frames
                self.send_stop()

                # Take multiple frames while still; each update helps collapse P
                for _ in range(SPIN_BURST_FRAMES):
                    img = self.take_pic()
                    self.update_localization(img)
                    time.sleep(SPIN_BURST_DT)

                # schedule a short cooldown (optional—keeps the loop simple)
                self.spin_pause_until = time.time() + (SPIN_BURST_FRAMES * SPIN_BURST_DT * 0.2)

                # Check if we've completed a full revolution
                if self.spin_accum >= FULL_TURN:
                    # end spin and go to HOLD (end of leg)
                    fruit_name = self.current_fruit_name or "fruit"
                    print(f"[AUTO] Reached {fruit_name} standoff (≤0.30 m). Holding {self.hold_secs:.1f}s…")
                    self.holding_until = time.time() + self.hold_secs
                    self.state = 'HOLD'
                    # reset spin helpers for next time
                    self.spin_accum = 0.0
                    self.last_theta = None
                    self.spin_target_heading = None
                    self.spin_pause_until = None
                    return

                # otherwise, continue to the next step on the next loop
                return

    # --- one frame ---
    def step_one_frame(self, img):
        self.update_localization(img)
        self.auto_step()

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser("Fruit searching with A* planning to standoff (keeps M4 control)")
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt')
    parser.add_argument("--search", type=str, default="search_list.txt")
    parser.add_argument("--ip", type=str, default='192.168.50.1')
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--show_map", action="store_true", help="Show costmap/occupancy preview and continue")
    parser.add_argument("--preview_each_leg", action="store_true", help="Show planned legs with robot footprint")
    parser.add_argument("--dry_run", action="store_true", help="Only visualize and exit (no robot)")

    args, _ = parser.parse_known_args()

    # Load GT map + list
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list(args.search)
    targets_xy, target_names, distractors_xy = targets_from_search_list(search_list, fruit_list, fruit_true_pos)

    # Build one planning grid (AruCo + distractors are hard obstacles, targets lightly inflated)
    costmap, occ, meta = build_planning_grid(aruco_true_pos, distractors_xy, targets_xy)
    if args.show_map:
        # obstacles = ArUco + distractors (as Nx2 array)
        obs_list = [aruco_true_pos]
        if len(distractors_xy) > 0:
            obs_list.append(np.array(distractors_xy, dtype=np.float64))
        obstacles_xy = np.vstack(obs_list)

        t_xy = np.array(targets_xy, dtype=np.float64) if len(targets_xy) > 0 else None

        visualize_costmap_detailed(
            costmap, occ, meta,
            obstacle_points_m=obstacles_xy,
            target_points=t_xy,
            title="A* Costmap (inflated obstacles + targets)"
        )

    if args.dry_run:
        print("[DRY RUN] Visualization done. Exiting before robot starts.")
        return

    # Init pygame (keeps your timing + ESC quit)
    pygame.init(); 
    # pygame.display.set_mode((300, 200)); 
    clock = pygame.time.Clock()

    # Plan a path (leg) to each fruit standoff, chaining legs
    legs = []
    leg_names = []
    # For the first leg start from origin; once live, the EKF start pose will be close to (0,0)
    start_xy = (0.0, 0.0)

    # We’ll re-baseline start_xy again after EKF starts (below) for better first leg
    for nm, fruit_xy in zip(target_names, targets_xy):
        path, standoff_xy = plan_standoff_path(costmap, meta, start_xy, fruit_xy,
                                               radius=STANDOFF_R, nsamp=24,
                                               smooth_lam=0.25, smooth_iters=25, downsample_step=6)
        if path is None:
            print(f"[PLAN] No reachable standoff for {nm} at {fruit_xy}. Skipping.")
            continue
        legs.append(path)
        leg_names.append(nm)
        start_xy = standoff_xy  # chain planner previews

    # Now start the live system and replace the FIRST leg start with EKF pose
    loc_system = LocalizationSystem(args, aruco_true_pos, legs, target_names=leg_names, hold_secs=HOLD_SECS)
    if len(leg_names) > 0:
        loc_system.current_fruit_name = leg_names[0]
        # print(f"[UI] Heading to first fruit: {leg_names[0]}")

    print(f"[AUTO] Planned {len(legs)} legs (standoff={STANDOFF_R:.2f} m, hold={HOLD_SECS:.1f} s).")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        img = loc_system.take_pic()
        loc_system.step_one_frame(img)

        # HUD: pose + current target every 2s (keeps your original cadence)
        if time.time() - loc_system.last_print_time >= loc_system.print_interval:
            loc_system.print_robot_pose()

            loc_system.print_robot_pose()
            loc_system.print_robot_uncertainty()   # <-- add this line

            # also show where we're heading next
            wp = loc_system.current_wp()
            if wp is not None:
                x, y, th = loc_system.get_robot_pose()
                dx, dy = wp[0]-x, wp[1]-y
                dist = (dx*dx + dy*dy) ** 0.5
                target_label = loc_system.current_fruit_name or f"Leg {loc_system.leg_idx+1}"
                # print(f"[UI] Target: {target_label} | Next WP: ({wp[0]:.2f},{wp[1]:.2f}) | d={dist:.2f} m")
            loc_system.last_print_time = time.time()

        if loc_system.finished: break
        clock.tick(40)

    loc_system.send_stop()
    pygame.quit()
    print("Shut down")

if __name__ == "__main__":
    main()


# python M4_fruit_searching_planner.py --ip 192.168.50.1 --map M4_true_map_full.txt --search search_list.txt
