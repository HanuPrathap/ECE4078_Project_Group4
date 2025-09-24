# M4 - Autonomous fruit searching
# Loads a ground truth map, prints target fruit positions,
# and supports manual waypoints OR auto-drive to first N fruits (via RRT or straight).
# Odometry is updated after each move so subsequent waypoints are accurate.

import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import math, random

# --- SLAM components (kept ready, not used in this odom-only demo) ---
sys.path.insert(0, os.path.join(os.getcwd(), "slam"))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# --- Util & robot API ---
sys.path.insert(0, os.path.join(os.getcwd(), "util"))
from pibot import PenguinPi
import measure as measure

from Helper import *  # expects get_distance_robot_to_goal, get_angle_robot_to_goal, etc.

# ---------- Map helpers ----------
def read_true_map(fname):
    """Return (fruit_list, fruit_true_pos, aruco_true_pos) from a ground-truth json map."""
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)
            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)
        return fruit_list, fruit_true_pos, aruco_true_pos

def read_search_list():
    """Read the search order of the target fruits from search_list.txt."""
    search_list = []
    with open('search_list.txt', 'r') as fd:
        for fruit in fd.readlines():
            search_list.append(fruit.strip())
    return search_list

def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(
                    n_fruit, fruit,
                    np.round(fruit_true_pos[i][0], 1),
                    np.round(fruit_true_pos[i][1], 1)
                ))
        n_fruit += 1

def get_fruit_xy_from_map(gt_dict, fruit_name):
    """Return (x,y) for the first map entry whose key starts with '<fruit_name>_'."""
    prefix = fruit_name + "_"
    for k, v in gt_dict.items():
        if k.startswith(prefix):
            return float(v["x"]), float(v["y"])
    return None  # not found

# ---------- Motion helpers ----------
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def drive_to_point(ppi, waypoint, robot_pose):
    """
    Turn-in-place to face waypoint, then drive straight.
    Updates and returns odometry pose consistent with the commanded motion.
    """
    current_robot_pose = np.array(robot_pose, dtype=float)
    waypoint = np.array(waypoint, dtype=float)

    wheel_vel = 30  # ticks/s (safe default)

    # calibration -> scalars
    scale_arr = np.loadtxt("calibration/param/scale.txt", delimiter=',')
    scale = float(np.mean(scale_arr))  # m/tick
    baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))  # m

    # plan
    distance_to_waypoint = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint  = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint  = wrap_to_pi(heading_to_waypoint)

    drive_time = distance_to_waypoint / (wheel_vel * scale)
    turn_dir   = 1 if heading_to_waypoint >= 0 else -1
    turn_time  = (2.0 * abs(heading_to_waypoint) * scale * wheel_vel) / baseline

    # execute (timed)
    print(f"Turning {heading_to_waypoint:+.3f} rad for {turn_time:.2f}s")
    ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=turn_time)

    print(f"Driving {distance_to_waypoint:.2f} m for {drive_time:.2f}s")
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    # --- odometry update (matches commands) ---
    x, y, th = current_robot_pose
    th = wrap_to_pi(th + heading_to_waypoint)  # apply the commanded rotation
    d  = wheel_vel * scale * drive_time        # equals distance_to_waypoint
    x = x + d * np.cos(th)
    y = y + d * np.sin(th)

    new_pose = np.array([x, y, th], dtype=float)
    print(f"Pose (odometry): x={x:.3f}, y={y:.3f}, th={th:.3f}")
    return new_pose

def compute_approach_point(robot_xy, fruit_xy, stop_center_radius=0.15):
    """
    Return a waypoint that is 'stop_center_radius' meters from the fruit along
    the line from the robot to the fruit. 0.15 m ensures the entire robot
    (~0.10 m radius) sits inside the 0.25 m scoring circle.
    """
    rx, ry = robot_xy
    fx, fy = fruit_xy
    v = np.array([fx - rx, fy - ry], dtype=float)
    d = float(np.linalg.norm(v)) + 1e-9
    if d <= stop_center_radius:
        return (rx, ry)
    u = v / d
    gx = fx - stop_center_radius * u[0]
    gy = fy - stop_center_radius * u[1]
    return (gx, gy)

# ===================== Bounds helpers (manual override) =====================
def parse_bounds_arg(s):
    """Parse 'xmin,ymin,xmax,ymax' into a tuple of floats."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bounds must be 'xmin,ymin,xmax,ymax'")
    xmin, ymin, xmax, ymax = map(float, parts)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("bounds must satisfy xmin<xmax and ymin<ymax")
    return (xmin, ymin, xmax, ymax)

def resolve_bounds(gt_raw, arena_size, bounds_str):
    """
    Choose bounds in priority:
      1) bounds_str -> explicit (xmin,ymin,xmax,ymax)
      2) arena_size -> square centered at origin (±arena_size/2)
      3) auto-from-map -> min/max of objects ±0.05
    """
    if bounds_str:
        return parse_bounds_arg(bounds_str)
    if arena_size and arena_size > 0:
        S = float(arena_size) * 0.5
        return (-S, -S, +S, +S)
    # auto-from-map
    xs = [float(v["x"]) for v in gt_raw.values()]
    ys = [float(v["y"]) for v in gt_raw.values()]
    pad = 0.05
    return (min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad)

# ===================== RRT PLANNER (drop-in) =====================
class _Disc:
    __slots__ = ("x","y","r","name")
    def __init__(self, x, y, r, name=""):
        self.x, self.y, self.r, self.name = float(x), float(y), float(r), name

def _segment_circle_intersect(p, q, c, r):
    (x1,y1),(x2,y2) = p,q
    (cx,cy) = c
    dx, dy = x2-x1, y2-y1
    if dx==0 and dy==0:
        return (x1-cx)**2 + (y1-cy)**2 <= r*r
    t = ((cx-x1)*dx + (cy-y1)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    px = x1 + t*dx; py = y1 + t*dy
    return (px-cx)**2 + (py-cy)**2 <= r*r

def _edge_collision_free(p, q, obstacles, bounds):
    (xmin,ymin,xmax,ymax) = bounds
    for x,y in (p,q):
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
    for obs in obstacles:
        if _segment_circle_intersect(p, q, (obs.x,obs.y), obs.r):
            return False
    return True

def _sample_free(bounds, goal=None, goal_sample_rate=0.10):
    (xmin,ymin,xmax,ymax) = bounds
    if goal and random.random() < goal_sample_rate:
        return goal
    return (random.uniform(xmin, xmax), random.uniform(ymin, ymax))

def _nearest(tree_xy, s):
    sx, sy = s
    best, bi = 1e9, 0
    for i,(x,y) in enumerate(tree_xy):
        d = (x-sx)**2 + (y-sy)**2
        if d < best: best, bi = d, i
    return bi

def _steer(p, s, step):
    (px,py),(sx,sy) = p,s
    vx, vy = sx-px, sy-py
    d = math.hypot(vx,vy) + 1e-12
    if d <= step: return (sx,sy)
    u = step / d
    return (px + u*vx, py + u*vy)

def rrt_plan(start, goal, obstacles, bounds,
             step=0.15, iters=3000, goal_tol=0.12, goal_sample_rate=0.10):
    """Basic 2D RRT from start->goal avoiding circular obstacles. Returns [start,...,goal] or []."""
    tree_xy = [tuple(start)]
    parent  = {0: -1}

    for _ in range(iters):
        s = _sample_free(bounds, goal=tuple(goal), goal_sample_rate=goal_sample_rate)
        ni = _nearest(tree_xy, s)
        new = _steer(tree_xy[ni], s, step)
        if not _edge_collision_free(tree_xy[ni], new, obstacles, bounds):
            continue
        tree_xy.append(new); parent[len(tree_xy)-1] = ni

        if math.hypot(new[0]-goal[0], new[1]-goal[1]) <= goal_tol:
            if _edge_collision_free(new, goal, obstacles, bounds):
                tree_xy.append(tuple(goal)); parent[len(tree_xy)-1] = len(tree_xy)-2
                path = []
                cur = len(tree_xy)-1
                while cur != -1:
                    path.append(tree_xy[cur]); cur = parent[cur]
                path.reverse()
                return path
    return []  # no path

def path_shortcut(path, obstacles, bounds, tries=200):
    """Randomly shortcut the polyline if straight segments are free."""
    if len(path) < 3: return path
    pts = list(path)
    n = len(pts)
    for _ in range(tries):
        i = random.randint(0, n-3)
        j = random.randint(i+2, n-1)
        if _edge_collision_free(pts[i], pts[j], obstacles, bounds):
            pts = pts[:i+1] + pts[j:]
            n  = len(pts)
            if n < 3: break
    return pts

def build_obstacles_and_bounds(gt_dict, target_set, current_target_name, bounds,
                               inflate=0.15, base_marker=0.14, base_fruit=0.20):
    """
    From the ground-truth dict + shopping list:
      - ArUcos: always obstacles (radius base_marker + inflate)
      - Fruits not in target_set: obstacles (base_fruit + inflate)
      - Current target fruit: not an obstacle
    'bounds' is provided by resolve_bounds (manual override or auto-from-map).
    """
    obs = []
    for name, v in gt_dict.items():
        x, y = float(v["x"]), float(v["y"])
        if name.startswith("aruco"):
            obs.append(_Disc(x, y, base_marker + inflate, name))
        else:
            fruit_type = name.split("_")[0].lower()
            if fruit_type == current_target_name.lower():
                continue  # allow the current target
            if fruit_type not in (target_set or set()):
                obs.append(_Disc(x, y, base_fruit + inflate, name))
    return obs, bounds

# ===================== unified entrypoint: manual OR auto =====================
def run_manual(ppi, robot_pose):
    """Interactive mode: type x,y and the robot drives there."""
    while True:
        try:
            x = float(input("X coordinate of the waypoint: "))
            y = float(input("Y coordinate of the waypoint: "))
        except ValueError:
            print("Please enter numbers."); continue
        waypoint = [x, y]
        robot_pose = drive_to_point(ppi, waypoint, robot_pose)
        print(f"Finished driving to waypoint: {waypoint}; New robot pose (odom): {robot_pose}")
        ppi.set_velocity([0, 0])
        if input("Add a new waypoint? [Y/N] ").strip().upper() == 'N':
            break
    return robot_pose

def run_auto(ppi, robot_pose, args):
    """Auto mode: plan and drive to the first N fruits in search_list.txt."""
    # Load GT map and search list
    with open(args.map, "r") as f:
        gt_raw = json.load(f)
    search_list = read_search_list()
    fruit_list, fruit_true_pos, _ = read_true_map(args.map)
    print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)

    # Parameters
    STOP_CENTER_RADIUS = args.approach  # e.g., 0.15 m
    HOLD_SECONDS = args.hold            # e.g., 3.0 s
    targets = search_list[:max(0, args.count)]

    # Resolve planning bounds (manual override or auto)
    bounds = resolve_bounds(gt_raw, args.arena_size, args.bounds)
    print(f"Planning bounds: {bounds}")

    # For obstacle building
    target_set = {s.strip().lower() for s in search_list}

    for idx, fruit in enumerate(targets, 1):
        fruit_xy = get_fruit_xy_from_map(gt_raw, fruit)
        if fruit_xy is None:
            print(f"[{idx}/{len(targets)}] {fruit}: not found in map — skipping.")
            continue

        print(f"[{idx}/{len(targets)}] Target: {fruit} at {fruit_xy}")

        # Approach goal that leaves robot center within 0.25 m scoring circle
        rx, ry, _ = robot_pose
        goal_xy = compute_approach_point((rx, ry), fruit_xy, stop_center_radius=STOP_CENTER_RADIUS)
        print(f"Approach waypoint: {goal_xy} (stop {STOP_CENTER_RADIUS:.2f} m from target)")

        path = None
        if args.planner == "rrt":
            # Build obstacles and bounds (exclude current target as obstacle)
            obstacles, bnds = build_obstacles_and_bounds(
                gt_raw, target_set, fruit, bounds,
                inflate=0.15, base_marker=0.14, base_fruit=0.20
            )
            # Plan with RRT
            start_xy = (rx, ry)
            path = rrt_plan(start_xy, goal_xy, obstacles, bnds,
                            step=0.15, iters=args.iters, goal_tol=0.12, goal_sample_rate=0.15)
            if path:
                path = path_shortcut(path, obstacles, bnds, tries=200)
                print(f"Planned {len(path)} waypoints.")
            else:
                print("RRT failed; falling back to straight approach.")

        # Follow path (if any); otherwise straight to goal
        if path and len(path) >= 2:
            for wp in path[1:]:
                robot_pose = drive_to_point(ppi, wp, robot_pose)
        else:
            robot_pose = drive_to_point(ppi, goal_xy, robot_pose)

        # Stop & hold
        ppi.set_velocity([0, 0])
        print(f"Holding at {fruit} for {HOLD_SECONDS:.1f} s…")
        time.sleep(HOLD_SECONDS)

    print("Auto-drive complete.")
    ppi.set_velocity([0, 0])
    return robot_pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching (manual or auto)")
    parser.add_argument("--map",  type=str, default="M3_prac_map_full.txt")
    parser.add_argument("--ip",   metavar="", type=str, default="192.168.50.1")
    parser.add_argument("--port", metavar="", type=int, default=8080)
    # mode & planner
    parser.add_argument("--mode", choices=["manual","auto"], default="auto",
                        help="manual: type waypoints; auto: plan to fruits")
    parser.add_argument("--planner", choices=["rrt","straight"], default="rrt",
                        help="auto mode: use RRT or straight-line")
    parser.add_argument("--count", type=int, default=2, help="how many fruits from search_list to visit")
    parser.add_argument("--hold",  type=float, default=3.0, help="seconds to hold at each fruit")
    parser.add_argument("--approach", type=float, default=0.15,
                        help="stop center radius from fruit center (m)")
    parser.add_argument("--iters", type=int, default=3000, help="RRT iterations")
    # NEW: manual bounds override
    parser.add_argument("--arena_size", type=float, default=0.0,
                        help="square arena size in meters (centered at 0,0). 0 = auto-from-map.")
    parser.add_argument("--bounds", type=str, default="",
                        help="explicit bounds 'xmin,ymin,xmax,ymax' (overrides --arena_size)")

    args, _ = parser.parse_known_args()

    # Connect robot
    ppi = PenguinPi(args.ip, args.port)

    # Start odom pose
    robot_pose = [0.0, 0.0, 0.0]

    # Run
    if args.mode == "manual":
        _ = run_manual(ppi, robot_pose)
    else:
        _ = run_auto(ppi, robot_pose, args)
# ===================== end unified entrypoint =====================
