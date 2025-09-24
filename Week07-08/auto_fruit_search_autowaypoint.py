# M4 - Autonomous fruit searching
# loads a ground truth map, prints target fruit positions,
# and auto-drives to the first two fruits (turn-then-straight).
# Odometry is updated after each move so subsequent waypoints are accurate.

# NOTE, the rotation is still off, needs adjustment.

import sys, os
import cv2
import numpy as np
import json
import argparse
import time

# --- Arena configuration ---
ARENA_SIZE_M = 2.4   # or 2.4 on the real day
WORLD_X_MIN, WORLD_X_MAX = -ARENA_SIZE_M / 2, ARENA_SIZE_M / 2
WORLD_Y_MIN, WORLD_Y_MAX = -ARENA_SIZE_M / 2, ARENA_SIZE_M / 2


# --- SLAM components (kept ready, not used in this odom-only demo) ---
sys.path.insert(0, os.path.join(os.getcwd(), "slam"))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# --- Util & robot API ---
sys.path.insert(0, os.path.join(os.getcwd(), "util"))
from pibot import PenguinPi
import measure as measure

from Helper import *  # get_distance_robot_to_goal, get_angle_robot_to_goal, etc.

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
    ang_rate = (2.0 * wheel_vel * scale) / baseline  # rad/s
    turn_time = abs(heading_to_waypoint) / ang_rate


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
    (≈0.10 m radius) sits inside the 0.25 m scoring circle.
    """
    rx, ry = robot_xy
    fx, fy = fruit_xy
    v = np.array([fx - rx, fy - ry], dtype=float)
    d = float(np.linalg.norm(v)) + 1e-9
    if d <= stop_center_radius:
        # already inside the scoring radius—just stop here
        return (rx, ry)
    u = v / d
    gx = fx - stop_center_radius * u[0]
    gy = fy - stop_center_radius * u[1]
    return (gx, gy)

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map",  type=str, default='M3_prac_map_full.txt')
    parser.add_argument("--ip",   metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    # Robot connection
    ppi = PenguinPi(args.ip, args.port)

    # Calibration (for SLAM later if you want)
    K  = np.loadtxt("calibration/param/intrinsic.txt",  delimiter=',')
    D  = np.loadtxt("calibration/param/distCoeffs.txt", delimiter=',')  # fix if typo
    S  = float(np.mean(np.loadtxt("calibration/param/scale.txt", delimiter=',')))
    B  = float(np.loadtxt("calibration/param/baseline.txt", delimiter=','))

    # Robot + EKF objects (not actively used in this odom-only demo)
    robot = Robot(B, S, K, D)
    ekf   = EKF(robot)

    # Map & shopping list
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # -------- AUTO-DRIVE TO FIRST TWO FRUITS --------
    with open(args.map, "r") as f:
        gt_raw = json.load(f)

    robot_pose = [0.0, 0.0, 0.0]          # odometry pose
    targets = search_list[:5]             # lemon, tomato (from your example)
    STOP_CENTER_RADIUS = 0.15
    HOLD_SECONDS = 3.0

    for idx, fruit in enumerate(targets, 1):
        fruit_xy = get_fruit_xy_from_map(gt_raw, fruit)
        if fruit_xy is None:
            print(f"[{idx}/2] {fruit}: not found in map — skipping.")
            continue

        print(f"[{idx}/2] Target: {fruit} at {fruit_xy}")

        # Compute approach waypoint that leaves robot center inside 0.25 m scoring circle
        rx, ry, _ = robot_pose
        goal_xy = compute_approach_point((rx, ry), fruit_xy, stop_center_radius=STOP_CENTER_RADIUS)
        print(f"Approach waypoint for {fruit}: {goal_xy}  (stop {STOP_CENTER_RADIUS:.2f} m from target center)")

        # Drive & hold
        robot_pose = drive_to_point(ppi, goal_xy, robot_pose)
        ppi.set_velocity([0, 0])
        print(f"Holding at {fruit} for {HOLD_SECONDS:.1f} s…")
        time.sleep(HOLD_SECONDS)

    print("Auto-drive to first two fruits complete.")
    ppi.set_velocity([0, 0])