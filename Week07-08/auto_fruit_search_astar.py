# auto_fruit_search_astar.py
# End-to-end: parse map & search_list -> build grid -> A* leg-by-leg
# --dry_run prints & visualizes paths WITHOUT connecting to the robot.

import sys, os, time, json, math
import numpy as np
import argparse
import matplotlib.pyplot as plt

# ---- Optional project imports (only used when not in dry_run) ----
def _safe_robot_imports():
    sys.path.insert(0, "{}/util".format(os.getcwd()))
    from util.pibot import PenguinPi
    from util.measure import Drive
    import util.DatasetHandler as dh
    import util.measure as measure

    sys.path.insert(0, "{}/slam".format(os.getcwd()))
    from slam.ekf import EKF
    from slam.robot import Robot
    import slam.aruco_detector as aruco
    from Helper import get_distance_robot_to_goal, get_angle_robot_to_goal
    return PenguinPi, Drive, dh, measure, EKF, Robot, aruco, get_distance_robot_to_goal, get_angle_robot_to_goal

from path_planning_astar import (
    build_costmap_fixed, plan_leg_astar, smooth_polyline,
    visualize_costmap_detailed, visualize_plan_over_costmap
)

# ---------------------------
# Params
# ---------------------------
ARENA_SIZE = 3.0
RES        = 0.03
ROBOT_R    = 0.075
MARGIN     = 0.02

# ---------------------------
# Map / search helpers
# ---------------------------

def read_true_map(fname: str):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
    fruit_list = []
    fruit_true_pos = []
    aruco_true_pos = np.empty((10, 2), dtype=np.float64)
    for key in gt_dict:
        x = float(np.round(gt_dict[key]['x'], 1))
        y = float(np.round(gt_dict[key]['y'], 1))
        if key.startswith('aruco'):
            if key.startswith('aruco10'):
                aruco_true_pos[9, 0] = x; aruco_true_pos[9, 1] = y
            else:
                marker_id = int(key[5]) - 1
                aruco_true_pos[marker_id, 0] = x
                aruco_true_pos[marker_id, 1] = y
        else:
            fruit_list.append(key[:-2])
            fruit_true_pos.append([x, y])
    return fruit_list, np.array(fruit_true_pos, dtype=np.float64), aruco_true_pos

def read_search_list(path='search_list.txt'):
    order = []
    with open(path, 'r') as fd:
        for line in fd:
            order.append(line.strip())
    return order

def targets_from_search_list(search_list, fruit_list, fruit_true_pos):
    name_to_pos = {}
    all_named_pos = []
    for name, (x, y) in zip(fruit_list, fruit_true_pos):
        pos = (float(x), float(y))
        all_named_pos.append((name, pos))
        name_to_pos.setdefault(name, []).append(pos)

    name_to_closest = {}
    for name, lst in name_to_pos.items():
        name_to_closest[name] = min(lst, key=lambda p: math.hypot(p[0], p[1]))

    targets_xy, used = [], set()
    print("Search order (closest selected):")
    k = 1
    for name in search_list:
        if name in name_to_closest:
            pos = name_to_closest[name]
            targets_xy.append(pos)
            used.add(name)
            print(f"{k}) {name} at [{pos[0]:.1f}, {pos[1]:.1f}]")
            k += 1

    distractors = []
    for nm, pos in all_named_pos:
        if nm not in used and pos not in distractors:
            distractors.append(pos)

    return targets_xy, distractors

# ---------------------------
# Pretty printing & export
# ---------------------------

def pretty_print_path(i, start_xy, goal_xy, path_xy, cost):
    N = 0 if path_xy is None else len(path_xy)
    print(f"\nLeg {i}: {start_xy} -> {goal_xy}")
    if N == 0:
        print("  [No path]")
        return
    print(f"  waypoints: {N}   cost: {cost:.2f}")
    if N <= 20:
        for k, (x, y) in enumerate(path_xy):
            print(f"    {k:02d}: ({x:.3f}, {y:.3f})")
    else:
        # compact print for long paths
        for k in range(0, N, max(1, N//10)):
            x, y = path_xy[k]
            print(f"    {k:02d}: ({x:.3f}, {y:.3f})")
        x, y = path_xy[-1]
        print(f"    {N-1:02d}: ({x:.3f}, {y:.3f}) [END]")

def export_paths_csv(paths_by_leg, fname="planned_paths.csv"):
    # paths_by_leg: list of dicts with keys: leg, start, goal, cost, path(list[(x,y)])
    import csv
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["leg","start_x","start_y","goal_x","goal_y","cost","index_in_leg","x","y"])
        for rec in paths_by_leg:
            leg = rec["leg"]; sx, sy = rec["start"]; gx, gy = rec["goal"]; cost = rec["cost"]
            for idx, (x,y) in enumerate(rec["path"]):
                w.writerow([leg, sx, sy, gx, gy, cost, idx, x, y])
    print(f"[EXPORT] CSV saved: {fname}")

def export_paths_json(paths_by_leg, fname="planned_paths.json"):
    import json
    # Convert numpy floats to regular floats
    clean = []
    for rec in paths_by_leg:
        clean.append({
            "leg": rec["leg"],
            "start": [float(rec["start"][0]), float(rec["start"][1])],
            "goal":  [float(rec["goal"][0]), float(rec["goal"][1])],
            "cost":  float(rec["cost"]),
            "path":  [[float(x), float(y)] for (x,y) in rec["path"]]
        })
    with open(fname, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[EXPORT] JSON saved: {fname}")

# -------- Waypoint progress logging helpers --------

def _pose_tuple(robot):
    """Return (x, y, theta) as simple floats from robot.state."""
    s = robot.state.flatten()
    return float(s[0]), float(s[1]), float(s[2])

def _heading_to(target_xy, from_pose):
    """Angle from pose to target in world frame."""
    tx, ty = target_xy
    x, y, th = from_pose
    return math.atan2(ty - y, tx - x)

def _wrap_to_pi(a):
    a = (a + math.pi) % (2*math.pi) - math.pi
    return a

def _dist(a_xy, b_xy):
    ax, ay = a_xy; bx, by = b_xy
    return math.hypot(ax - bx, ay - by)

def _fmt_pose(p):
    x, y, th = p
    return f"(x={x:+.3f}, y={y:+.3f}, th={th:+.2f}rad)"

def log_waypoint_progress(leg_idx, wp_idx, total_wps, waypoint_xy, pose_before, pose_after):
    """Pretty print a single waypoint step comparison."""
    d_before = _dist(waypoint_xy, (pose_before[0], pose_before[1]))
    d_after  = _dist(waypoint_xy, (pose_after[0],  pose_after[1]))

    th_goal_before = _heading_to(waypoint_xy, pose_before)
    th_goal_after  = _heading_to(waypoint_xy, pose_after)

    hdg_err_before = _wrap_to_pi(th_goal_before - pose_before[2])
    hdg_err_after  = _wrap_to_pi(th_goal_after  - pose_after[2])

    print(
        f"[leg {leg_idx}] wp {wp_idx+1}/{total_wps}  "
        f"plan=({waypoint_xy[0]:+.3f},{waypoint_xy[1]:+.3f}) | "
        f"before={_fmt_pose(pose_before)}  ->  after={_fmt_pose(pose_after)} | "
        f"dist: {d_before:.3f}m -> {d_after:.3f}m | "
        f"hdg_err: {hdg_err_before:+.2f}rad -> {hdg_err_after:+.2f}rad"
    )


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching with A* (dry-run supported)")
    parser.add_argument("--map", type=str, default="M3_prac_map_full.txt")
    parser.add_argument("--search", type=str, default="search_list.txt")

    parser.add_argument("--arena_size", type=float, default=ARENA_SIZE)
    parser.add_argument("--res", type=float, default=RES)
    parser.add_argument("--robot_r", type=float, default=ROBOT_R)
    parser.add_argument("--margin", type=float, default=MARGIN)

    parser.add_argument("--smooth_lam", type=float, default=0.2)
    parser.add_argument("--smooth_iters", type=int, default=20)
    parser.add_argument("--skip", type=int, default=3)

    parser.add_argument("--show_map", action="store_true", help="Show costmap only")
    parser.add_argument("--show_each_leg", action="store_true", help="Show per-leg path overlays")
    parser.add_argument("--export_csv", type=str, default="", help="Optional: export all legs to CSV at this path")
    parser.add_argument("--export_json", type=str, default="", help="Optional: export all legs to JSON at this path")
    parser.add_argument("--dry_run", action="store_true", help="Plan + print + visualize ONLY (no robot connection)")
    args, _ = parser.parse_known_args()

    # ---- Load ground truth & search order ----
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list(args.search)

    targets_xy, distractors_xy = targets_from_search_list(search_list, fruit_list, fruit_true_pos)
    if len(targets_xy) == 0:
        print("[ERROR] No targets from search_list found in map.")
        sys.exit(1)

    # Obstacles: ArUco + distractor fruits
    obstacles = [aruco_true_pos]
    if len(distractors_xy) > 0:
        obstacles.append(np.array(distractors_xy, dtype=np.float64))
    obstacle_points_xy = np.vstack(obstacles)

    # Build planning grid (once)
    costmap, occ, meta = build_costmap_fixed(
        size_m=args.arena_size,
        obstacle_points_m=np.array(obstacle_points_xy, dtype=np.float64),
        res=args.res,
        robot_radius_m=args.robot_r,
        safety_margin_m=args.margin
    )

    print(f"[INFO] Grid: {meta['W']}x{meta['H']} @ {meta['res']:.3f} m/px")
    if args.show_map:
        visualize_costmap_detailed(costmap, occ, meta, obstacle_points_xy, targets_xy, title="A* costmap")

    # Initial pose for planning previews (assume origin if no robot)
    current_xy = (0.0, 0.0)

    # Collect all plans for export/visualization
    plans = []

    # ---- DRY RUN: just compute/print/visualize ----
    if args.dry_run:
        for i, goal_xy in enumerate(targets_xy, start=1):
            raw_path, cost = plan_leg_astar(costmap, meta, current_xy, goal_xy)
            if raw_path is None or len(raw_path) == 0:
                pretty_print_path(i, current_xy, goal_xy, None, math.inf)
                continue
            path = smooth_polyline(raw_path, lam=args.smooth_lam, iters=args.smooth_iters)
            pretty_print_path(i, current_xy, goal_xy, path, cost)
            plans.append(dict(leg=i, start=current_xy, goal=goal_xy, cost=cost, path=path))
            if args.show_each_leg:
                visualize_plan_over_costmap(costmap, occ, meta, path, current_xy, goal_xy, title=f"Leg {i}")
            current_xy = goal_xy  # assume perfect arrival for the preview

        if args.export_csv:
            export_paths_csv(plans, args.export_csv)
        if args.export_json:
            export_paths_json(plans, args.export_json)
        sys.exit(0)

    # ---- LIVE RUN (connect to robot) ----
    PenguinPi, Drive, dh, measure, EKF, Robot, aruco, get_dist, get_angle = _safe_robot_imports()

    # Connect to robot
    ppi = PenguinPi("192.168.50.1", 8080)  # you can add --ip/--port if you want

    # Calibration & Robot instance
    scale = np.loadtxt("calibration/param/scale.txt", delimiter=',')
    K     = np.loadtxt("calibration/param/intrinsic.txt", delimiter=',')
    D     = np.loadtxt("calibration/param/distCoeffs.txt", delimiter=',')
    baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))
    robot = Robot(baseline, float(np.mean(scale)), K, D)

    try:
        # start from actual robot state
        current_xy = (float(robot.state[0,0]), float(robot.state[1,0]))

        from Helper import get_distance_robot_to_goal, get_angle_robot_to_goal

        def drive_to_point(ppi, waypoint_xy, robot_pose, is_final_target=False, target_threshold=0.30):
            wp = np.array(waypoint_xy, dtype=np.float64)
            pose = np.array(robot_pose, dtype=np.float64)
            scale_arr  = np.loadtxt("calibration/param/scale.txt", delimiter=',')
            scale      = float(np.mean(scale_arr))
            baseline   = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))
            wheel_tick = 30
            dist = float(get_distance_robot_to_goal(pose, np.hstack([wp, 0.0])))
            ang  = float(get_angle_robot_to_goal(pose, np.hstack([wp, 0.0])))
            if is_final_target and dist <= target_threshold: return
            turn_time = abs((2.0 * ang * scale * wheel_tick) / baseline)
            turn_dir  = 1 if ang >= 0 else -1
            ppi.set_velocity([0, turn_dir], turning_tick=wheel_tick, time=turn_time)
            drive_time = dist / (wheel_tick * scale)
            ppi.set_velocity([1, 0], tick=wheel_tick, time=drive_time)

        def follow_path_with_drive_to_point(ppi, robot, path_xy, is_final_path=False, skip=3, leg_idx=1):
            """
            Follow a polyline and log robot pose vs planned waypoints.
            - If ppi is None (dry run), we still print the plan and current pose.
            - If live, we drive (turn-then-go) and then print before/after.
            """
            if not path_xy:
                return

            # Downsample
            sampled = path_xy[::max(1, skip)]
            if sampled[-1] != path_xy[-1]:
                sampled.append(path_xy[-1])

            total = len(sampled)
            for i, wp in enumerate(sampled):
                # Pose BEFORE motion
                pose_before = _pose_tuple(robot)

                # Decide if this waypoint is the end of a "final" leg
                final_here = is_final_path and (i == total - 1)

                if ppi is None:
                    # DRY RUN: do not drive; pose_after == pose_before
                    pose_after = pose_before
                    log_waypoint_progress(leg_idx, i, total, (wp[0], wp[1]), pose_before, pose_after)
                    continue

                # LIVE RUN: command the robot (turn-then-go)
                # You already had a drive_to_point earlier; we inline a minimal version
                # that uses your existing calibration files.
                scale_arr  = np.loadtxt("calibration/param/scale.txt", delimiter=",")
                scale      = float(np.mean(scale_arr))
                baseline   = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))
                wheel_tick = 30

                # compute distance and heading to waypoint using your Helper math
                from Helper import get_distance_robot_to_goal, get_angle_robot_to_goal
                pose_np = np.array([pose_before[0], pose_before[1], pose_before[2]])
                wp_np   = np.array([wp[0], wp[1], 0.0])

                dist = float(get_distance_robot_to_goal(pose_np, wp_np))
                ang  = float(get_angle_robot_to_goal(pose_np,   wp_np))

                # turn
                turn_time = abs((2.0 * ang * scale * wheel_tick) / baseline)
                turn_dir  = 1 if ang >= 0 else -1
                ppi.set_velocity([0, turn_dir], turning_tick=wheel_tick, time=turn_time)

                # drive straight (simple time-based)
                drive_time = dist / (wheel_tick * scale)
                ppi.set_velocity([1, 0], tick=wheel_tick, time=drive_time)

                # small settle
                time.sleep(0.05)

                # Pose AFTER motion (read robot.state)
                pose_after = _pose_tuple(robot)

                # Log comparison
                log_waypoint_progress(leg_idx, i, total, (wp[0], wp[1]), pose_before, pose_after)

                # tiny pause so prints flush
                time.sleep(0.01)


        for i, goal_xy in enumerate(targets_xy, start=1):
            print(f"\n=== Planning leg {i}/{len(targets_xy)}: to {goal_xy} ===")
            raw_path, cost = plan_leg_astar(costmap, meta, current_xy, goal_xy)
            if raw_path is None or len(raw_path) == 0:
                print(f"[WARN] No path to {goal_xy}. Skipping.")
                continue
            path = smooth_polyline(raw_path, lam=args.smooth_lam, iters=args.smooth_iters)
            pretty_print_path(i, current_xy, goal_xy, path, cost)
            follow_path_with_drive_to_point(ppi, robot, path, is_final_path=True, skip=args.skip)
            current_xy = (float(robot.state[0,0]), float(robot.state[1,0]))

        print("\n[INFO] All targets processed.")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        try: ppi.set_velocity([0, 0])
        except: pass
