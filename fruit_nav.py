# fruit_nav.py
import math, heapq, numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import argparse, time, os, sys, json, csv


import cv2  # for optional cost softening
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi  # access the robot
import util.DatasetHandler as dh   # save/load functions
import util.measure as measure     # measurements
import pygame                     # python package for GUI
import shutil                     # python package for file operations


sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco


from YOLO.detector import Detector

from Helper import * 

from path_planning import *

# ---------------------------
# Bounds & conversions
# ---------------------------
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def world_bounds_fixed_2x2(size):
    half = size/2
    return -half, -half, half, half

def world_to_grid(x, y, xmin, ymin, res, W=None, H=None):
    gx = int(math.floor((x - xmin) / res))
    gy = int(math.floor((y - ymin) / res))
    if W is not None and H is not None:
        gx = clamp(gx, 0, W-1)
        gy = clamp(gy, 0, H-1)
    return gx, gy

def grid_to_world(gx, gy, xmin, ymin, res):
    return (xmin + (gx + 0.5)*res, ymin + (gy + 0.5)*res)

# ---------------------------
# Costmap building
# ---------------------------
def rasterize_points_as_obstacles(occ, points_g, inflate_cells):
    H, W = occ.shape
    for (gx, gy) in points_g:
        if 0 <= gx < W and 0 <= gy < H:
            for dx in range(-inflate_cells, inflate_cells+1):
                for dy in range(-inflate_cells, inflate_cells+1):
                    col = gx + dx
                    row = gy + dy
                    if 0 <= col < W and 0 <= row < H:
                        if dx*dx + dy*dy <= inflate_cells*inflate_cells:
                            occ[row, col] = 1
    return occ

def build_costmap_fixed_2x2(
    size,
    obstacle_points_m: np.ndarray,
    res: float = 0.01,
    robot_radius: float = 0.075,
    safety_margin: float = 0.02
):
    xmin, ymin, xmax, ymax = world_bounds_fixed_2x2(size)
    W = int(math.ceil((xmax - xmin) / res))
    H = int(math.ceil((ymax - ymin) / res))
    occ = np.zeros((H, W), dtype=np.uint8)

    inflate_radius = robot_radius + safety_margin
    inflate_cells = max(1, int(round(inflate_radius / res)))

    obstacles_g = [world_to_grid(x, y, xmin, ymin, res) for x, y in obstacle_points_m]
    rasterize_points_as_obstacles(occ, obstacles_g, inflate_cells)

    border_thickness = inflate_cells
    occ[:border_thickness, :] = 1
    occ[-border_thickness:, :] = 1
    occ[:, :border_thickness] = 1
    occ[:, -border_thickness:] = 1

    cost = np.where(occ == 1, 255, 1).astype(np.uint8)
    meta = dict(xmin=xmin, ymin=ymin, res=res, W=W, H=H)
    return cost, occ, meta

def soften_cost(cost, k=2):
    """Lightly blur free-space costs to discourage wall-hugging."""
    C = cost.astype(np.float32).copy()
    ker = np.array([[1,1,1],[1,2,1],[1,1,1]], dtype=np.float32); ker /= ker.sum()
    for _ in range(k):
        C = cv2.filter2D(C, -1, ker)
    C[cost >= 255] = 255
    C = np.clip(C, 1, 255).astype(np.uint8)
    return C

def visualize_costmap_detailed(cost, occ, meta, obstacle_points_m, target_points=None, title="Costmap Visualization"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    xmax = meta['xmin'] + meta['W'] * meta['res']
    ymax = meta['ymin'] + meta['H'] * meta['res']
    extent = [meta['xmin'], xmax, meta['ymin'], ymax]
    x_grid_lines = np.arange(meta['xmin'], xmax + meta['res']/2, meta['res'])
    y_grid_lines = np.arange(meta['ymin'], ymax + meta['res']/2, meta['res'])

    ax1 = axes[0]
    ax1.imshow(occ, cmap='RdYlBu_r', origin='lower', extent=extent)
    ax1.set_title('Occupancy Grid\n(Blue=Free, Red=Occupied)')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    for x in x_grid_lines: ax1.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    for y in y_grid_lines: ax1.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
    if len(obstacle_points_m) > 0:
        ax1.scatter(obstacle_points_m[:, 0], obstacle_points_m[:, 1], c='black', s=20, marker='x', alpha=0.7, label='Original Obstacles')
    if target_points is not None and len(target_points) > 0:
        targets_array = np.array(target_points)
        ax1.scatter(targets_array[:, 0], targets_array[:, 1], c='green', s=100, marker='*', label='Targets', zorder=5)
        for i, (tx, ty) in enumerate(targets_array):
            ax1.annotate(f'{i+1}', (tx, ty), xytext=(5, 5), textcoords='offset points', fontsize=8, color='white', weight='bold')
    ax1.legend(); ax1.set_aspect('equal')

    ax2 = axes[1]
    im = ax2.imshow(cost, cmap='viridis', origin='lower', extent=extent)
    ax2.set_title('Cost Grid\n(Dark=Low Cost, Bright=High Cost)')
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    for x in x_grid_lines: ax2.axvline(x, color='white', alpha=0.3, linewidth=0.5)
    for y in y_grid_lines: ax2.axhline(y, color='white', alpha=0.3, linewidth=0.5)
    plt.colorbar(im, ax=ax2, label='Cost Value')
    if len(obstacle_points_m) > 0:
        ax2.scatter(obstacle_points_m[:, 0], obstacle_points_m[:, 1], c='red', s=20, marker='x', alpha=0.7, label='Original Obstacles')
    if target_points is not None and len(target_points) > 0:
        targets_array = np.array(target_points)
        ax2.scatter(targets_array[:, 0], targets_array[:, 1], c='white', s=100, marker='*', label='Targets', zorder=5)
        for i, (tx, ty) in enumerate(targets_array):
            ax2.annotate(f'{i+1}', (tx, ty), xytext=(5, 5), textcoords='offset points', fontsize=8, color='black', weight='bold')
    ax2.legend(); ax2.set_aspect('equal')

    plt.tight_layout(); plt.suptitle(title, y=1.02); plt.show()

# ---------------------------
# A* + LOS + smoothing
# ---------------------------
def a_star(costmap: np.ndarray, start_g, goal_g):
    H, W = costmap.shape
    def blocked(p):
        x, y = p
        return not (0 <= x < W and 0 <= y < H) or (costmap[y, x] >= 255)
    if blocked(start_g) or blocked(goal_g):
        return None, math.inf

    nbrs = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(1,-1),(-1,1),(1,1)]

    def octile(a, b):
        ax, ay = a; bx, by = b
        dx, dy = abs(ax-bx), abs(ay-by)
        D = 1.0; D2 = math.sqrt(2.0)
        return D*(dx+dy) + (D2-2*D)*min(dx, dy)

    g = {start_g: 0.0}
    came = {}
    eps = 1e-3
    pq = []
    heapq.heappush(pq, (octile(start_g, goal_g), 0.0, start_g))
    closed = set()

    while pq:
        fcur, gcur, u = heapq.heappop(pq)
        if u in closed:
            continue
        if u == goal_g:
            path = [u]
            while u in came:
                u = came[u]
                path.append(u)
            return list(reversed(path)), gcur
        closed.add(u)
        ux, uy = u
        for dx, dy in nbrs:
            vx, vy = ux+dx, uy+dy
            v = (vx, vy)
            if not (0 <= vx < W and 0 <= vy < H): continue
            if costmap[vy, vx] >= 255: continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            move_cost = step * float(costmap[vy, vx])
            gv = gcur + move_cost
            if gv < g.get(v, math.inf):
                g[v] = gv
                came[v] = u
                hv = octile(v, goal_g)
                f = gv + hv + eps*hv
                heapq.heappush(pq, (f, gv, v))
    return None, math.inf

def plan_leg_astar(costmap, meta, start_xy, goal_xy):
    sg = world_to_grid(start_xy[0], start_xy[1], meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
    gg = world_to_grid(goal_xy[0],  goal_xy[1],  meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
    gpath, gcost = a_star(costmap, sg, gg)
    if gpath is None:
        return None, math.inf
    poly = [grid_to_world(px, py, meta["xmin"], meta["ymin"], meta["res"]) for (px, py) in gpath]
    return poly, gcost

def los_free(occ, p0, p1, meta):
    (x0, y0), (x1, y1) = p0, p1
    g0 = world_to_grid(x0, y0, meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
    g1 = world_to_grid(x1, y1, meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
    x0g, y0g = g0; x1g, y1g = g1
    dx = abs(x1g - x0g); dy = abs(y1g - y0g)
    sx = 1 if x0g < x1g else -1
    sy = 1 if y0g < y1g else -1
    err = dx - dy; x, y = x0g, y0g
    H, W = occ.shape
    while True:
        if not (0 <= x < W and 0 <= y < H) or occ[y, x] == 1:
            return False
        if x == x1g and y == y1g:
            return True
        e2 = 2*err
        if e2 > -dy: err -= dy; x += sx
        if e2 <  dx: err += dx; y += sy

def string_pull_los(path_xy, occ, meta):
    if not path_xy or len(path_xy) < 3:
        return path_xy
    out = [path_xy[0]]
    i = 0
    while i < len(path_xy)-1:
        j = i + 1
        while j < len(path_xy) and los_free(occ, out[-1], path_xy[j], meta):
            j += 1
        out.append(path_xy[j-1])
        i = j - 1
    return out

def smooth_polyline(poly, lam=0.1, iters=8):
    if not poly or len(poly) < 4:
        return poly
    P = np.array(poly, dtype=np.float64)
    Q = P.copy()
    for _ in range(iters):
        Q[1:-1] = (1-lam)*Q[1:-1] + lam*0.5*(Q[0:-2] + Q[2:])
    return [tuple(x) for x in Q]

# ---------------------------
# Debug plotting & sim
# ---------------------------
def plot_path_on_costmap(cost, meta, path_xy, traj_xy=None, targets=None, title="Plan & Trajectory"):
    xmax = meta['xmin'] + meta['W'] * meta['res']
    ymax = meta['ymin'] + meta['H'] * meta['res']
    extent = [meta['xmin'], xmax, meta['ymin'], ymax]
    plt.figure(figsize=(7,6))
    plt.imshow(cost, cmap='Greys', origin='lower', extent=extent)
    if path_xy:
        px, py = zip(*path_xy)
        plt.plot(px, py, linewidth=2, label='Planned path')
        plt.scatter(px[0], py[0], marker='o', s=40, label='Start')
        plt.scatter(px[-1], py[-1], marker='x', s=60, label='Goal')
    if traj_xy:
        tx, ty = zip(*traj_xy)
        plt.plot(tx, ty, linestyle='--', linewidth=1.5, label='Simulated follow')
    if targets:
        tx, ty = zip(*targets)
        plt.scatter(tx, ty, marker='*', s=100, label='Targets')
    plt.gca().set_aspect('equal'); plt.legend(); plt.title(title); plt.tight_layout(); plt.show()

def simulate_follow(path_xy, step=0.02):
    if not path_xy or len(path_xy) < 2:
        return path_xy
    traj = [path_xy[0]]
    for i in range(1, len(path_xy)):
        x0, y0 = traj[-1]
        x1, y1 = path_xy[i]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            continue
        n = max(1, int(dist/step))
        for k in range(1, n+1):
            t = k/n
            traj.append((x0 + t*dx, y0 + t*dy))
    return traj

# ---------------------------
# MONASH helpers (your I/O)
# ---------------------------
def read_true_map(fname):
    # Use your provided absolute path (kept as-is)
    with open("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/M3_prac_map_full.txt", 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)
            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x; aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x; aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)
        return fruit_list, fruit_true_pos, aruco_true_pos

def read_search_list():
    search_list = []
    with open('C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/search_list.txt', 'r') as fd:
        fruits = fd.readlines()
        for fruit in fruits:
            search_list.append(fruit.strip())
    return search_list

def targets_from_search_list1(search_list, fruit_list, fruit_true_pos):
    name_to_positions = {}
    all_fruit_positions = []
    for name, (x, y) in zip(fruit_list, fruit_true_pos):
        pos = (float(x), float(y))
        all_fruit_positions.append((name, pos))
        name_to_positions.setdefault(name, []).append(pos)
    name_to_closest_pos = {name: min(positions, key=lambda p: math.hypot(p[0], p[1]))
                           for name, positions in name_to_positions.items()}
    targets_xy, target_names_used = [], set()
    print("Search order (closest to origin selected):")
    n_fruit = 1
    for name in search_list:
        if name in name_to_closest_pos:
            closest_pos = name_to_closest_pos[name]
            targets_xy.append(closest_pos); target_names_used.add(name)
            print(f"{n_fruit}) {name} at [{round(closest_pos[0],1)}, {round(closest_pos[1],1)}]")
            n_fruit += 1
    distractor_xy = []
    for name, pos in all_fruit_positions:
        if name not in target_names_used and pos not in distractor_xy:
            distractor_xy.append(pos)
    return targets_xy, distractor_xy

# ---------------------------
# Robot glue (drive + logging)
# ---------------------------
# Imports that rely on your project structure
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure

sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

def get_robot_pose(robot):
    # Expect robot.state = [x, y, theta]
    return robot.state

def drive_to_target_with_feedback(ppi, robot, target_xy, threshold, wheel_vel, scale):
    max_attempts = 100
    for attempt in range(max_attempts):
        pose = get_robot_pose(robot)
        current_xy = np.array([pose[0], pose[1]])
        dist = float(np.linalg.norm(np.array(target_xy) - current_xy))
        # print(f"[{time.time():.1f}] pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) → d={dist:.2f} m")
        if dist <= threshold:
            print(f"Reached target within {threshold} m.")
            ppi.set_velocity([0, 0]); break
        ppi.set_velocity([1, 0], tick=wheel_vel, time=0.2)
        time.sleep(0.1)
    else:
        print("Max attempts reached; stopping.")
        ppi.set_velocity([0, 0])

def drive_to_point(ppi, waypoint, robot_pose, is_final_target=False, target_threshold=0.3):
    current_robot_pose = np.array(robot_pose)
    waypoint = np.array(waypoint)
    fileS = "calibration/param/scale.txt"
    fileB = "calibration/param/baseline.txt"
    scale_arr = np.loadtxt(fileS, delimiter=',')
    scale = float(np.mean(scale_arr))
    baseline = float(np.squeeze(np.loadtxt(fileB, delimiter=',')))
    wheel_vel = 30

    # Helper functions expected in your Helper.py
    from Helper import get_distance_robot_to_goal, get_angle_robot_to_goal
    distance_to_waypoint = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint  = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))

    print(f"Dist {distance_to_waypoint:.2f} m | Heading {heading_to_waypoint:.2f} rad")
    if is_final_target and distance_to_waypoint <= target_threshold:
        print(f"Already within {target_threshold} m; no move.")
        return

    turn_time = abs((2.0 * heading_to_waypoint * scale * wheel_vel) / baseline)
    turn_dir = 1 if heading_to_waypoint >= 0 else -1
    print(f"Turn {turn_time:.2f}s (dir {turn_dir})")
    ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=turn_time)

    if is_final_target:
        print(f"Final approach until {target_threshold} m")
        drive_to_target_with_feedback(ppi, robot, waypoint, target_threshold, wheel_vel, scale)
    else:
        drive_time = distance_to_waypoint / (wheel_vel * scale)
        print(f"Drive {drive_time:.2f}s")
        ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

def follow_path_with_drive_to_point(ppi, robot, path_xy, is_final_path=False, skip=3, log_csv="run_log.csv"):
    if not path_xy:
        return
    # logger
    logf = open(log_csv, "a", newline="")
    writer = csv.writer(logf); writer.writerow(["t", "rx", "ry", "rtheta"])

    sampled = path_xy[::max(1, skip)]
    if sampled[-1] != path_xy[-1]:
        sampled.append(path_xy[-1])

    for i, wp in enumerate(sampled):
        pose = get_robot_pose(robot)
        tnow = time.time()
        writer.writerow([tnow, float(pose[0]), float(pose[1]), float(pose[2])])
        # print(f"[{tnow:.1f}] state x={pose[0]:.3f}, y={pose[1]:.3f}, th={pose[2]:.3f} | next→ {wp}")
        is_final_target = is_final_path and (i == len(sampled) - 1)
        drive_to_point(ppi, [wp[0], wp[1]], pose, is_final_target=is_final_target, target_threshold=0.30)
        time.sleep(0.02)

    logf.close()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    ARENA_SIZE = 3.0       # meters (square, centered at origin)
    RES        = 0.01      # meters per cell
    ROBOT_R    = 0.075     # robot radius (m)
    MARGIN     = 0.01      # safety margin (m)

    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M3_prac_map_full.txt')
    parser.add_argument("--ip", type=str, default='192.168.50.1')
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--arena_size", type=float, default=ARENA_SIZE)
    parser.add_argument("--res", type=float, default=RES)
    parser.add_argument("--robot_r", type=float, default=ROBOT_R)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--smooth_lam", type=float, default=0.1)
    parser.add_argument("--smooth_iters", type=int, default=8)
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--plot", type=int, default=0)      # 1 = show plots
    parser.add_argument("--sim_only", type=int, default=0)  # 1 = plan + simulate, don't move
    parser.add_argument("--soften", type=int, default=1)    # 1 = soften costs
    args, _ = parser.parse_known_args()

    # --- Connect to robot (only if not sim-only) ---
    ppi = None; robot = None
    if not args.sim_only:
        ppi = PenguinPi(args.ip, args.port)
        wheels_scale = np.loadtxt("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/calibration/param/scale.txt", delimiter=",")
        camera_matrix = np.loadtxt("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/calibration/param/intrinsic.txt", delimiter=",")
        camera_dist   = np.loadtxt("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/calibration/param/distCoeffs.txt", delimiter=",")
        robot = Robot(ppi, wheels_scale, camera_matrix, camera_dist)

    # --- Load ground truth & search order ---
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    target_points_xy, distraction_points_xy = targets_from_search_list1(search_list, fruits_list, fruits_true_pos)
    if len(target_points_xy) == 0:
        print("[ERROR] No targets found that match search_list."); sys.exit(1)

    # --- Obstacles: ArUco + distractors ---
    obstacles_list = [aruco_true_pos]
    if len(distraction_points_xy) > 0:
        obstacles_list.append(np.array(distraction_points_xy))
    obstacle_points_xy = np.vstack(obstacles_list)
    print(f"[INFO] Total obstacle points: {len(obstacle_points_xy)}")

    # --- Costmap ---
    costmap, occ, meta = build_costmap_fixed_2x2(
        size=args.arena_size,
        obstacle_points_m=np.array(obstacle_points_xy, dtype=np.float64),
        res=args.res,
        robot_radius=args.robot_r,
        safety_margin=args.margin,
    )
    if args.soften:
        costmap = soften_cost(costmap, k=2)
    print(f"[INFO] Costmap: {meta['W']}x{meta['H']} @ {meta['res']} m/cell")

    if args.plot:
        visualize_costmap_detailed(costmap, occ, meta, obstacle_points_xy, target_points_xy)

    # --- Start pose ---
    if args.sim_only:
        current_xy = (0.0, 0.0)
    else:
        # Use robot EKF pose if available
        current_xy = (float(robot.state[0]), float(robot.state[1]))

    # --- Plan/Execute legs ---
    try:
        for k, goal_xy in enumerate(target_points_xy, start=1):
            print(f"\n=== Target {k}/{len(target_points_xy)}: {goal_xy} ===")
            raw_leg, leg_cost = plan_leg_astar(costmap, meta, current_xy, goal_xy)
            if raw_leg is None or len(raw_leg) == 0:
                print(f"[WARN] No path to {goal_xy}. Skipping.")
                continue

            leg_los  = string_pull_los(raw_leg, occ, meta)
            leg_path = smooth_polyline(leg_los, lam=args.smooth_lam, iters=args.smooth_iters)
            print(f"[INFO] Path points: raw={len(raw_leg)}, los={len(leg_los)}, smooth={len(leg_path)}, cost={leg_cost:.1f}")

            if args.plot:
                sim_traj = simulate_follow(leg_path, step=0.02)
                plot_path_on_costmap(costmap, meta, leg_path, traj_xy=sim_traj, title=f"Leg to {tuple(np.round(goal_xy,2))}")

            if not args.sim_only:
                follow_path_with_drive_to_point(ppi, robot, leg_path, is_final_path=True, skip=args.skip, log_csv="run_log.csv")
                # Update current pose from EKF/robot
                if hasattr(robot, 'state') and robot.state is not None and len(robot.state) >= 2:
                    current_xy = (float(robot.state[0]), float(robot.state[1]))
                else:
                    current_xy = (goal_xy[0], goal_xy[1])  # fallback
            else:
                current_xy = (goal_xy[0], goal_xy[1])

        print("\n[INFO] All targets processed.")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received.")
    finally:
        if ppi is not None:
            ppi.set_velocity([0, 0])
