# test.py — localization + grid A* planning with standoff and dry-run
# ------------------------------------------------------------------

import sys, os, time, json
import numpy as np
import argparse
import matplotlib.pyplot as plt

# --- Robot I/O and helpers (only used when not in --dry_run) ---
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure
from Helper import get_distance_robot_to_goal, get_angle_robot_to_goal

# --- SLAM core (camera, kinematics, EKF) ---
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# --- Grid planner & viz ---
from path_planning_astar import (
    build_costmap_fixed, plan_leg_astar, smooth_polyline,
    visualize_costmap_detailed, visualize_plan_over_costmap,
    world_to_grid
)

# ---------------------------
# Defaults (tweakable)
# ---------------------------
ARENA_SIZE = 2.4      # m (square, centered at origin)
RES        = 0.03     # m / cell
ROBOT_R    = 0.08    # robot radius (m)
MARGIN     = 0.03     # extra safety (m)
STANDOFF_R = 0.3    # m from fruit
HOLD_SECS  = 3.0      # s pause at standoff


# ---------------------------
# Map / search helpers
# ---------------------------
def read_true_map(fname: str):
    """Return (fruit_list, fruit_true_pos Nx2, aruco_true_pos 10x2)."""
    with open(fname, 'r') as fd:
        gt = json.load(fd)

    fruit_list, fruit_pos = [], []
    aruco_pos = np.empty((10, 2), dtype=np.float64)

    for key, v in gt.items():
        x = float(np.round(v['x'], 3))
        y = float(np.round(v['y'], 3))
        if key.startswith('aruco'):
            if key.startswith('aruco10'):
                aruco_pos[9, 0] = x; aruco_pos[9, 1] = y
            else:
                idx = int(key[5]) - 1
                aruco_pos[idx, 0] = x; aruco_pos[idx, 1] = y
        else:
            fruit_list.append(key[:-2])
            fruit_pos.append([x, y])

    return fruit_list, np.array(fruit_pos, dtype=np.float64), aruco_pos


def read_search_list(path='search_list.txt'):
    order = []
    with open(path, 'r') as fd:
        for line in fd:
            order.append(line.strip())
    return order


def targets_from_search_list(search_list, fruit_list, fruit_true_pos):
    """Closest occurrence of each fruit in search_list; distractors = others."""
    import math
    name_to_positions, all_named = {}, []
    for name, (x, y) in zip(fruit_list, fruit_true_pos):
        pos = (float(x), float(y))
        all_named.append((name, pos))
        name_to_positions.setdefault(name, []).append(pos)

    name_to_closest = {
        name: min(lst, key=lambda p: math.hypot(p[0], p[1]))
        for name, lst in name_to_positions.items()
    }

    targets_xy, used = [], set()
    print("\nSearch order (closest to origin selected):")
    k = 1
    for name in search_list:
        if name in name_to_closest:
            pos = name_to_closest[name]
            targets_xy.append(pos)
            used.add(name)
            print(f" {k}) {name} at [{pos[0]:.3f}, {pos[1]:.3f}]")
            k += 1

    distractors = []
    for nm, pos in all_named:
        if nm not in used and pos not in distractors:
            distractors.append(pos)

    if distractors:
        print("\nDistractors (not in search list):")
        for i, (nm, pos) in enumerate([(None, p) for p in distractors], 1):
            print(f"  {i}) at [{pos[0]:.3f}, {pos[1]:.3f}]")
    else:
        print("\nNo distractors.")

    return targets_xy, distractors


# ---------------------------
# Localization-only EKF (robot pose)
# ---------------------------
class LocalizationEKF:
    """EKF that estimates only robot [x,y,theta] using known landmarks."""
    def __init__(self, robot, known_landmarks):
        self.robot = robot
        self.known_landmarks = known_landmarks
        self.P = np.eye(3) * 0.01

    def get_state_vector(self):
        return self.robot.state.copy()

    def predict(self, drive_meas):
        print("predicting state")
        F = self.robot.derivative_drive(drive_meas)
        self.robot.drive(drive_meas)
        Q = self.robot.covariance_drive(drive_meas)
        self.P = F @ self.P @ F.T + Q

    def update(self, measurements):
        if not measurements:
            return
        valid = [m for m in measurements if m.tag in self.known_landmarks]
        if not valid:
            return

        print("valid updating using camera")
        # Stack z and R
        z = np.concatenate([m.position.reshape(-1, 1) for m in valid], axis=0)
        R = np.zeros((2*len(valid), 2*len(valid)))
        for i, m in enumerate(valid):
            R[2*i:2*i+2, 2*i:2*i+2] = m.covariance

        zhat_list, H_rows = [], []
        xy = self.robot.state[0:2, :]
        th = self.robot.state[2, 0]
        c, s = np.cos(th), np.sin(th)
        Rot = np.array([[c, -s], [s, c]])
        DRot = np.array([[-s, -c], [c, -s]])

        for m in valid:
            lm = np.array(self.known_landmarks[m.tag]).reshape(2, 1)
            zhat = Rot.T @ (lm - xy)
            zhat_list.append(zhat)

            H = np.zeros((2, 3))
            H[:, 0:2] = -Rot.T
            H[:, 2:3] = DRot.T @ (lm - xy)
            H_rows.append(H)

        zhat = np.vstack(zhat_list)
        H = np.vstack(H_rows)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innov = z - zhat
        self.robot.state = self.robot.state + (K @ innov).reshape(3, 1)
        self.P = (np.eye(3) - K @ H) @ self.P


class ArucoLocalization:
    def __init__(self, robot, known_landmarks, marker_length=0.07):
        self.ekf = LocalizationEKF(robot, known_landmarks)
        self.det = aruco.aruco_detector(robot, marker_length=marker_length)
        self.on = True

    def step(self, rgb, drive_meas):
        if self.on:
            self.ekf.predict(drive_meas)
        meas, img = self.det.detect_marker_positions(rgb)
        if self.on and meas:
            self.ekf.update(meas)
        s = self.ekf.get_state_vector()
        return (float(s[0,0]), float(s[1,0]), float(s[2,0])), img

    def get_pose(self):
        s = self.ekf.get_state_vector()
        return float(s[0,0]), float(s[1,0]), float(s[2,0])


# ---------------------------
# Motion helpers (guarded for dry-run)
# ---------------------------
def execute_motion_with_localization(ppi, localizer, motion_command, duration, ip="192.168.50.1"):
    """If ppi is None (dry-run), do nothing and return pose."""
    if ppi is None:  # dry-run
        return localizer.get_pose()

    fwd, turn = motion_command
    fwd = 1 if fwd > 0 else (-1 if fwd < 0 else 0)
    turn = 1 if turn > 0 else (-1 if turn < 0 else 0)

    # print(f"Executing motion [{fwd}, {turn}] for {duration:.2f}s")
    start = time.time()
    last = start
    while (time.time() - start) < duration:
        lv, rv = ppi.set_velocity([fwd, turn], tick=40, turning_tick=40)
        now = time.time()
        dt = max(now - last, 1e-3)
        last = now
        drive_meas = measure.Drive(lv, -rv if ip != "localhost" else rv, dt)
        img = ppi.get_image()
        try:
            pose, _ = localizer.step(img, drive_meas)
        except Exception as e:
            print(f"[Localization error] {e}")
        time.sleep(0.1)

    ppi.set_velocity([0, 0])
    time.sleep(0.1)
    return localizer.get_pose()


def drive_to_point_with_localization(ppi, waypoint, robot_pose, localizer,
                                     stop_within=None, hold_secs=0.0):
    """Turn-then-go to waypoint; optionally stop short and hold."""
    # Dry-run: pretend we arrive (teleport for preview)
    if ppi is None:
        x, y = float(waypoint[0]), float(waypoint[1])
        th = float(robot_pose[2]) if len(robot_pose) == 3 else 0.0
        return np.array([x, y, th], dtype=float)

    current = np.array(robot_pose, dtype=float)
    wp = np.array(waypoint, dtype=float)

    scale = float(np.mean(np.loadtxt("calibration/param/scale.txt", delimiter=',')))
    baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))

    dist = float(np.squeeze(get_distance_robot_to_goal(current, wp)))
    head = float(np.squeeze(get_angle_robot_to_goal(current, wp)))
    head = (head + np.pi) % (2*np.pi) - np.pi

    if stop_within is not None and dist > stop_within:
        dist = max(0.0, dist - stop_within)

    wheel = 40
    drive_t = dist / (wheel * scale + 1e-9)
    ang_rate = (2.0 * wheel * scale) / (baseline + 1e-9)
    turn_t = abs(head) / (ang_rate + 1e-9)
    turn_dir = 1 if head >= 0 else -1

    print(f"Distance(clamped): {dist:.2f} m, Heading: {head:.3f} rad")
    print(f"Turn time: {turn_t:.2f}s, Drive time: {drive_t:.2f}s")

    if turn_t > 0.05:
        execute_motion_with_localization(ppi, localizer, [0, turn_dir], turn_t, args.ip)

    if drive_t > 0.05:
        execute_motion_with_localization(ppi, localizer, [1, 0], drive_t, args.ip)

    # Final stationary localization
    ppi.set_velocity([0, 0]); time.sleep(0.5)
    img = ppi.get_image()
    drive_meas = measure.Drive(0, 0, 0.001)
    final_pose, _ = localizer.step(img, drive_meas)

    if hold_secs > 0:
        print(f"Holding for {hold_secs:.1f}s at standoff...")
        ppi.set_velocity([0, 0])
        time.sleep(hold_secs)

    return np.array(final_pose, dtype=float)


# ---------------------------
# Planning helpers
# ---------------------------
def plan_to_standoff(costmap, meta, start_xy, target_xy, radius=0.15, n_samples=24):
    """
    Pick a free, reachable point on a circle of 'radius' around the target and
    plan A* to it. Returns (path_xy, standoff_xy, cost) or (None, None, inf).
    """
    import math
    candidates = []
    for th in np.linspace(0.0, 2.0*math.pi, n_samples, endpoint=False):
        cx = float(target_xy[0] + radius * math.cos(th))
        cy = float(target_xy[1] + radius * math.sin(th))
        gx, gy = world_to_grid(cx, cy, meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
        if costmap[gy, gx] >= 255:
            continue
        poly, gcost = plan_leg_astar(costmap, meta, start_xy, (cx, cy))
        if poly:
            candidates.append((gcost, (cx, cy), poly))

    if not candidates:
        return None, None, float('inf')

    candidates.sort(key=lambda t: t[0])
    best_cost, best_xy, best_path = candidates[0]
    return best_path, best_xy, best_cost


def follow_path_with_localization(ppi, localizer, path_xy, skip=3,
                                  stop_within=None, hold_secs=0.0):
    """Follow a polyline; on the LAST point apply stop_within/hold."""
    if not path_xy or len(path_xy) < 2:
        return
    sampled = path_xy[::max(1, int(skip))]
    if sampled[-1] != path_xy[-1]:
        sampled.append(path_xy[-1])

    pose = localizer.get_pose() if ppi is not None else (sampled[0][0], sampled[0][1], 0.0)
    for i, wp in enumerate(sampled):
        last = (i == len(sampled) - 1)
        pose = drive_to_point_with_localization(
            ppi, wp, pose, localizer,
            stop_within=(stop_within if last else None),
            hold_secs=(hold_secs if last else 0.0)
        )


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("A* + EKF navigation with standoff (dry-run supported)")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")

    parser.add_argument("--map", type=str, default="M3_prac_map_full.txt")
    parser.add_argument("--search", type=str, default="search_list.txt")

    parser.add_argument("--arena_size", type=float, default=ARENA_SIZE)
    parser.add_argument("--res", type=float, default=RES)
    parser.add_argument("--robot_r", type=float, default=ROBOT_R)
    parser.add_argument("--margin", type=float, default=MARGIN)

    parser.add_argument("--standoff", type=float, default=STANDOFF_R, help="Target stand-off distance (m)")
    parser.add_argument("--hold_secs", type=float, default=HOLD_SECS, help="Hold at standoff (s)")

    parser.add_argument("--smooth_lam", type=float, default=0.25)
    parser.add_argument("--smooth_iters", type=int, default=25)
    parser.add_argument("--skip", type=int, default=3)

    parser.add_argument("--show_map", action="store_true")
    parser.add_argument("--show_each_leg", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Plan + visualize only (no robot I/O)")
    args, _ = parser.parse_known_args()

    # ---- Load ground truth & search order ----
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list(args.search)
    targets_xy, distractors_xy = targets_from_search_list(search_list, fruit_list, fruit_true_pos)

    # Obstacles: ArUco + distractors
    obstacles = [aruco_true_pos]
    if len(distractors_xy) > 0:
        obstacles.append(np.array(distractors_xy, dtype=np.float64))
    obstacle_points_xy = np.vstack(obstacles)

    # ---- Build planning grid (once) ----
    # Add the 8 cm cube half-diagonal (~0.0566 m) so we never clip corners
    CUBE_HALF_DIAG = (2**0.5) * 0.04
    CUBE_HALF_DIAG = (2**0.5) * 0.06
    costmap, occ, meta = build_costmap_fixed(
        size_m=args.arena_size,
        obstacle_points_m=np.array(obstacle_points_xy, dtype=np.float64),
        res=args.res,
        robot_radius_m=args.robot_r,
        safety_margin_m=(args.margin + CUBE_HALF_DIAG),
    )
    print(f"[INFO] Grid: {meta['W']}x{meta['H']} @ {meta['res']:.3f} m/px")

    if args.show_map:
        visualize_costmap_detailed(
            costmap, occ, meta,
            np.array(obstacle_points_xy, dtype=np.float64),
            np.array(targets_xy, dtype=np.float64),
            title="A* costmap"
        )

    # ---- DRY RUN (no robot connection or localization) ----
    if args.dry_run:
        current_xy = (0.0, 0.0)  # assume origin for preview, or change as you like
        for i, fruit_xy in enumerate(targets_xy, start=1):
            print(f"\n=== Leg {i}/{len(targets_xy)}: start={current_xy} -> fruit={fruit_xy} (standoff {args.standoff:.2f}m) ===")
            raw_path, standoff_xy, cost = plan_to_standoff(costmap, meta, current_xy, fruit_xy,
                                                           radius=args.standoff, n_samples=24)
            if not raw_path:
                print("  [No reachable standoff] Skipping.")
                continue
            path = smooth_polyline(raw_path, lam=args.smooth_lam, iters=args.smooth_iters)
            sampled = path[::max(1, args.skip)]
            if sampled[-1] != path[-1]:
                sampled.append(path[-1])
            print(f"  waypoints: {len(sampled)}  cost: {cost:.2f}  standoff={standoff_xy}")

            if args.show_each_leg:
                visualize_plan_over_costmap(costmap, occ, meta, path, current_xy, standoff_xy,
                                            title=f"Leg {i} to standoff")

            current_xy = standoff_xy  # assume perfect arrival for preview
        sys.exit(0)

    # ---- LIVE RUN (connect to robot, localize, drive) ----
    # Connect to robot
    ppi = PenguinPi(args.ip, args.port)

    # Calibration & Robot instance
    K = np.loadtxt(f"{args.calib_dir}intrinsic.txt", delimiter=',')
    D = np.loadtxt(f"{args.calib_dir}distCoeffs.txt", delimiter=',')
    scale_arr = np.loadtxt(f"{args.calib_dir}scale.txt", delimiter=',')
    baseline = float(np.squeeze(np.loadtxt(f"{args.calib_dir}baseline.txt", delimiter=',')))
    scale = float(np.mean(scale_arr))
    # if args.ip == 'localhost':
    #     scale /= 2.0
    robot = Robot(baseline, scale, K, D)

    # Known landmarks dict for localization
    known_landmarks = {}
    for i in range(len(aruco_true_pos)):
        tag = i + 1
        idx = 9 if tag == 10 else i
        known_landmarks[tag] = [float(aruco_true_pos[idx, 0]), float(aruco_true_pos[idx, 1])]

    # Localization
    localizer = ArucoLocalization(robot, known_landmarks)

    # Start pose for planning
    sx, sy, _ = localizer.get_pose()
    current_xy = (float(sx), float(sy))

    try:
        for i, fruit_xy in enumerate(targets_xy, start=1):
            print(f"\n=== Leg {i}/{len(targets_xy)}: start={current_xy} -> fruit={fruit_xy} (standoff {args.standoff:.2f}m) ===")
            raw_path, standoff_xy, cost = plan_to_standoff(costmap, meta, current_xy, fruit_xy,
                                                           radius=args.standoff, n_samples=24)
            if not raw_path:
                print("  [No reachable standoff] Skipping.")
                continue

            path = smooth_polyline(raw_path, lam=args.smooth_lam, iters=args.smooth_iters)
            follow_path_with_localization(
                ppi, localizer, raw_path, skip=args.skip,   # using the raw path 
                stop_within=args.standoff, hold_secs=args.hold_secs
            )

            # Update start for next leg
            nx, ny, _ = localizer.get_pose()
            current_xy = (float(nx), float(ny))

        print("\n[INFO] All targets processed.")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        try:
            ppi.set_velocity([0, 0])
        except Exception:
            pass

# dry 
# python test5.py --dry_run --show_map --show_each_leg


# live
# python test5.py --ip 192.168.50.1 --show_map