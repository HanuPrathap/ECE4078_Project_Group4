# test.py - Modified for localization-only (no landmark estimation)

import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import matplotlib.pyplot as plt


# --- Robot I/O and helpers ---
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure
from Helper import get_distance_robot_to_goal, get_angle_robot_to_goal

# --- SLAM core (your existing modules) ---
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

from path_planning_astar import *

# Global variables ---------------------------------------------------------------------------------
ARENA_SIZE = 3          # meters (square, centered at origin)
RES        = 0.01           # meters per cell (5 cm)
ROBOT_R    = 0.075           # robot radius (m)
MARGIN     = 0.01           # safety margin (m)



def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    # open the ground truth map 
    with open("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/M3_prac_map_full.txt", 'r') as fd:
        gt_dict = json.load(fd)
        # create empty lists 
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 3)
            y = np.round(gt_dict[key]['y'], 3)

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
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def targets_from_search_list(search_list, fruit_list, fruit_true_pos):
    """
    Build targets (x,y) in the same order as search_list,
    using the closest occurrence to origin (0,0) of each fruit in fruit_list.
    Also return distractor fruits (fruits not in search_list) as obstacles.
    Prints the selected targets in search order.
    
    Returns:
        targets_xy: list of (x,y) tuples for fruits in search_list
        distractor_xy: list of (x,y) tuples for fruits NOT in search_list (to use as obstacles)
    """
    import math
    
    # Group all positions by fruit name
    name_to_positions = {}
    all_fruit_positions = []  # Keep track of all fruit positions
    
    # Collect all positions for each fruit name
    for name, (x, y) in zip(fruit_list, fruit_true_pos):
        pos = (float(x), float(y))
        all_fruit_positions.append((name, pos))
        
        if name not in name_to_positions:
            name_to_positions[name] = []
        name_to_positions[name].append(pos)

    # For each fruit name, find the position closest to origin
    name_to_closest_pos = {}
    for name, positions in name_to_positions.items():
        closest_pos = min(positions, key=lambda pos: math.sqrt(pos[0]**2 + pos[1]**2))
        name_to_closest_pos[name] = closest_pos

    # Build targets list in the order of search_list, using closest positions
    targets_xy = []
    target_names_used = set()  # Track which fruits are actually targets
    print("\n")

    print("Search order (closest to origin selected):")
    n_fruit = 1
    
    for name in search_list:
        if name in name_to_closest_pos:
            closest_pos = name_to_closest_pos[name]
            targets_xy.append(closest_pos)
            target_names_used.add(name)
            
            # Print the selected target with rounded coordinates
            print('{}) {} at [{}, {}]'.format(n_fruit,
                                              name,
                                              round(closest_pos[0], 3),
                                              round(closest_pos[1], 3)))
            n_fruit += 1
    print("\n")

    # Create distractor list - fruits that exist in the map but are NOT in search_list
    distractor_xy = []
    distractor_info = []  # Keep track of name and position pairs

    for name, pos in all_fruit_positions:
        if name not in target_names_used:
            # Only add if this position hasn't already been added to distractors
            if pos not in distractor_xy:
                distractor_xy.append(pos)
                distractor_info.append((name, pos))

    # Print distractor fruits
    if distractor_info:
        print("Distractor fruits (not in search list):")
        for i, (name, pos) in enumerate(distractor_info, 1):
            print(f"{i}) {name} at [{pos[0]:.1f}, {pos[1]:.1f}]")
        print()
    else:
        print("No distractor fruits found.")

    return targets_xy, distractor_xy


class LocalizationEKF:
    """
    Modified EKF for localization only - landmarks are known and fixed.
    Only estimates robot pose [x, y, theta].
    """
    def __init__(self, robot, known_landmarks):
        """
        robot: Robot instance with calibration
        known_landmarks: dict {tag_id: [x, y]} of known marker positions
        """
        self.robot = robot
        self.known_landmarks = known_landmarks  # {tag_id: [x, y]}
        
        # State is only robot pose [x, y, theta] - no landmark estimation
        self.P = np.eye(3) * 0.01  # Small initial uncertainty
        
    def get_state_vector(self):
        """Return only robot state [x, y, theta]"""
        return self.robot.state.copy()
    
    def predict(self, drive_meas):
        """Prediction step - only for robot pose"""
        # Get motion jacobian (3x3 for robot pose only)
        F = self.robot.derivative_drive(drive_meas)
        
        # Update robot pose using motion model
        self.robot.drive(drive_meas)
        
        # Update covariance for robot pose only
        Q = self.robot.covariance_drive(drive_meas)
        self.P = F @ self.P @ F.T + Q
    
    def update(self, measurements):
        """Update step - only update robot pose based on known landmarks"""
        if not measurements:
            print("no imaage to update using camera")
            return
            
        # Filter measurements to only include known landmarks
        valid_measurements = []
        for lm in measurements:
            if lm.tag in self.known_landmarks:
                valid_measurements.append(lm)
        
        if not valid_measurements:
            return
            
        print(f"Updating with {len(valid_measurements)} known landmarks")
        
        # Stack measurements
        z = np.concatenate([lm.position.reshape(-1,1) for lm in valid_measurements], axis=0)
        R = np.zeros((2*len(valid_measurements), 2*len(valid_measurements)))
        for i, lm in enumerate(valid_measurements):
            R[2*i:2*i+2, 2*i:2*i+2] = lm.covariance
        
        # Create measurement prediction using known landmark positions
        z_hat_list = []
        H_rows = []
        
        for lm in valid_measurements:
            # Get known landmark position
            lm_true_pos = np.array(self.known_landmarks[lm.tag]).reshape(2, 1)
            
            # Predict measurement from current robot pose to this known landmark
            robot_xy = self.robot.state[0:2, :]
            th = self.robot.state[2, 0]
            Rot_theta = np.array([[np.cos(th), -np.sin(th)],
                                 [np.sin(th), np.cos(th)]])
            
            # Expected measurement in robot frame
            z_hat_lm = Rot_theta.T @ (lm_true_pos - robot_xy)
            z_hat_list.append(z_hat_lm)
            
            # Jacobian for this measurement (2x3 - only w.r.t. robot pose)
            DRot_theta = np.array([[-np.sin(th), -np.cos(th)],
                                  [np.cos(th), -np.sin(th)]])
            
            H_lm = np.zeros((2, 3))
            H_lm[:, 0:2] = -Rot_theta.T  # derivative w.r.t. robot x,y
            H_lm[:, 2:3] = DRot_theta.T @ (lm_true_pos - robot_xy)  # derivative w.r.t. robot theta
            
            H_rows.append(H_lm)
        
        # Stack predictions and jacobians
        z_hat = np.vstack(z_hat_list)
        H = np.vstack(H_rows)
        
        # EKF update equations
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update robot state
        innovation = z - z_hat
        x_update = K @ innovation
        self.robot.state = self.robot.state + x_update.reshape(3, 1)
        
        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P
        
        print(f"Robot pose updated: x={self.robot.state[0,0]:.3f}, y={self.robot.state[1,0]:.3f}, th={self.robot.state[2,0]:.3f}")


class ArucoLocalization:
    """
    Wrapper for localization-only ArUco processing
    """
    def __init__(self, robot, known_landmarks, marker_length=0.07):
        self.known_landmarks = known_landmarks
        self.ekf = LocalizationEKF(robot, known_landmarks)
        self.det = aruco.aruco_detector(robot, marker_length=marker_length)
        self.ekf_on = True
    
    def step(self, rgb_image, drive_meas):
        """Single step: predict with odometry, update with camera if markers seen"""
        # Always predict with odometry
        if self.ekf_on:
            print("predicting wiht odemetry")
            self.ekf.predict(drive_meas)
        
        # Detect markers and update if any known ones are seen
        measurements, aruco_img = self.det.detect_marker_positions(rgb_image)

        # Check if any markers were detected
        if not measurements:
            print("no markers detected")
        else:
            print(f"detected {len(measurements)} markers")
        
        if self.ekf_on and measurements:
            self.ekf.update(measurements)
        
        return self.get_pose(), aruco_img
    
    def get_pose(self):
        state = self.ekf.get_state_vector()
        return float(state[0, 0]), float(state[1, 0]), float(state[2, 0])


def execute_motion_with_localization(ppi, localizer, motion_command, duration, ip="192.168.50.1"):
    """
    Execute motion while running localization updates
    """
    fwd, turn = motion_command
    fwd = 1 if fwd > 0 else (-1 if fwd < 0 else 0)
    turn = 1 if turn > 0 else (-1 if turn < 0 else 0)
    
    print(f"Executing motion [{fwd}, {turn}] for {duration:.2f}s")
    
    start_time = time.time()
    last_time = start_time
    
    while (time.time() - start_time) < duration:
        # Issue motion command
        lv, rv = ppi.set_velocity([fwd, turn], tick=40, turning_tick=40)
        
        # Calculate dt
        now = time.time()
        dt = max(now - last_time, 1e-3)
        last_time = now
        
        # Create Drive measurement
        if ip == "localhost":
            drive_meas = measure.Drive(lv, rv, dt)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)  # Right wheel reversed for physical robot
        
        # Get camera image and run localization
        img = ppi.get_image()
        try:
            pose, aruco_img = localizer.step(img, drive_meas)
        except Exception as e:
            print(f"Localization error: {e}")
        
        time.sleep(0.1)  # Control loop frequency
    
    # Stop robot
    ppi.set_velocity([0, 0])
    time.sleep(0.1)
    
    return localizer.get_pose()


def drive_to_point_with_localization(ppi, waypoint, robot_pose, localizer):
    """Drive to waypoint with continuous localization"""
    current_robot_pose = np.array(robot_pose, dtype=float)
    waypoint = np.array(waypoint, dtype=float)
    
    # Load calibration
    scale = float(np.mean(np.loadtxt("calibration/param/scale.txt", delimiter=',')))
    baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))
    
    # Calculate motion parameters
    distance = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    heading = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))
    heading = (heading + np.pi) % (2*np.pi) - np.pi  # wrap to [-pi, pi]
    
    wheel_vel = 40
    drive_time = distance / (wheel_vel * scale + 1e-9)
    ang_rate = (2.0 * wheel_vel * scale) / (baseline + 1e-9)
    turn_time = abs(heading) / (ang_rate + 1e-9)
    turn_dir = 1 if heading >= 0 else -1
    
    print(f"Distance: {distance:.2f}m, Heading: {heading:.3f}rad")
    print(f"Turn time: {turn_time:.2f}s, Drive time: {drive_time:.2f}s")
    
    # Execute turn
    if turn_time > 0.05:
        print(f"Turning...")
        pose = execute_motion_with_localization(ppi, localizer, [0, turn_dir], turn_time, args.ip)
        print(f"After turn: {pose}")
    
    # Execute drive
    if drive_time > 0.05:
        print(f"Driving...")
        pose = execute_motion_with_localization(ppi, localizer, [1, 0], drive_time, args.ip)
        print(f"After drive: {pose}")
    
    # Final localization update (stop and look around for markers)
    print("Final localization check...")
    ppi.set_velocity([0, 0])
    time.sleep(0.5)
    
    print("getting image to use for localisation")
    img = ppi.get_image()
    drive_meas = measure.Drive(0, 0, 0.001)  # Zero motion
    final_pose, _ = localizer.step(img, drive_meas)
    
    print(f"Final pose AFTER LOCALISATION: {final_pose}")
    return np.array(final_pose, dtype=float)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default="192.168.50.1")
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")

    parser.add_argument("--arena_size", type=float, default=ARENA_SIZE)
    parser.add_argument("--res", type=float, default=RES)
    parser.add_argument("--robot_r", type=float, default=ROBOT_R)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--smooth_lam", type=float, default=0.1)
    parser.add_argument("--smooth_iters", type=int, default=10)
    parser.add_argument("--skip", type=int, default=3)

    args, _ = parser.parse_known_args()
    
    # Initialize robot
    ppi = PenguinPi(args.ip, args.port)
    
    # Load calibration
    camera_matrix = np.loadtxt(f"{args.calib_dir}intrinsic.txt", delimiter=',')
    dist_coeffs = np.loadtxt(f"{args.calib_dir}distCoeffs.txt", delimiter=',')
    scale = np.loadtxt(f"{args.calib_dir}scale.txt", delimiter=',')
    baseline = np.loadtxt(f"{args.calib_dir}baseline.txt", delimiter=',')
    
    if args.ip == 'localhost':
        scale /= 2.0 
    
    # create robot class
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map("M3_prac_map_full.txt")

    search_list = read_search_list()


    target_points_xy, distraction_points_xy = targets_from_search_list(search_list, fruits_list, fruits_true_pos) # test the distractoin list - chcanged thsi to print and get cosest target 


    known_landmarks = {}
    for i in range(len(aruco_true_pos)):
        marker_id = i + 1  # Markers are numbered 1-10, array indices are 0-9
        if marker_id == 10:
            # Handle marker 10 (which was stored at index 9)
            known_landmarks[10] = [float(aruco_true_pos[9, 0]), float(aruco_true_pos[9, 1])]
        else:
            # Markers 1-9 are stored at indices 0-8
            known_landmarks[marker_id] = [float(aruco_true_pos[i, 0]), float(aruco_true_pos[i, 1])]

    print("Known landmarks:")
    for tag_id, pos in known_landmarks.items():
        print(f"Marker {tag_id}: [{pos[0]:.4f}, {pos[1]:.4f}]")


    # After you create known_landmarks and distractor_xy
    obstacles_list = []

    # Add ArUco marker positions (extract [x,y] from dictionary values)
    for marker_pos in known_landmarks.values():
        obstacles_list.append(tuple(marker_pos))  # Convert [x,y] to (x,y)

    # Add distractor fruit positions 
    obstacles_list.extend(distraction_points_xy)

    numpy_obstacles = np.array(obstacles_list)
    numpy_targets = np.array(target_points_xy)

    print("\n")

    print(f"Total obstacles: {len(obstacles_list)}")
    print("Obstacles list:", obstacles_list)

    print("\n")

    # --- Build planning grid once (fixed bounds) --- only need to do it once 
    costmap, occ, meta = build_costmap_fixed(
        size_m=args.arena_size,
        obstacle_points_m=np.array(numpy_obstacles, dtype=np.float64),
        res=args.res,
        robot_radius_m=args.robot_r,
        safety_margin_m=args.margin,

    )

    print(f"[INFO] Costmap built: {meta['W']}x{meta['H']} cells @ {meta['res']} m/cell")

    # view  cost map 
    visualize_costmap_detailed(costmap, occ, meta, numpy_obstacles, numpy_targets)
        
    # Initialize localization system
    localizer = ArucoLocalization(robot, known_landmarks)
    
    # Waypoints to navigate
    waypoints = [
        [0.5, 0.0],
        [-0.5, 0.0], 
        [0.0, 0.5], 
        [0.0, -0.5],
        [0.0, 0.0]
    ]
    
    print("\nWaypoints:")
    for i, wp in enumerate(waypoints, 1):
        print(f"  {i}. [{wp[0]:.2f}, {wp[1]:.2f}]")
    
    # Navigate through waypoints
    robot_pose = localizer.get_pose()
    print(f"\nStarting from: {robot_pose}")
    
    for i, waypoint in enumerate(waypoints):
        print(f"\n--- Waypoint {i+1}/{len(waypoints)}: {waypoint} ---")
        print(f"Current pose: {robot_pose}")
        robot_pose = drive_to_point_with_localization(ppi, waypoint, robot_pose, localizer)
        print(f"Reached pose: {robot_pose}")
    
    print("\nNavigation complete!")