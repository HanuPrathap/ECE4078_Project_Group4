# M3 - Autonomous fruit searching


# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import matplotlib.pyplot as plt

# IMPORT --------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------



# Global variables ---------------------------------------------------------------------------------
ARENA_SIZE = 3          # meters (square, centered at origin)
RES        = 0.01           # meters per cell (5 cm)
ROBOT_R    = 0.075           # robot radius (m)
MARGIN     = 0.01           # safety margin (m)


# --------------------------------------------------------------------------------------------------


# MONASH FUNCTIONS ---------------------------------------------------------------------------------

# reads in ground truth map - depends on which level 
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

# this function gets the search.txt and converts it into a list so we can use it to go to target fruits
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

# --------------------------------------------------------------------------------------------------

#  gets the current 
def targets_from_search_list1(search_list, fruit_list, fruit_true_pos):
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
                                              round(closest_pos[0], 1),
                                              round(closest_pos[1], 1)))
            n_fruit += 1

    # Create distractor list - fruits that exist in the map but are NOT in search_list
    distractor_xy = []
    for name, pos in all_fruit_positions:
        if name not in target_names_used:
            # Only add if this position hasn't already been added to distractors
            if pos not in distractor_xy:
                distractor_xy.append(pos)

    return targets_xy, distractor_xy


#  samples border so that we can add it to the list of obstacles
# def sample_arena_border(ARENA_SIZE, spacing=0.05):
#     """
#     Sample a square arena border centered at (0,0) as obstacle points.
#     Returns: list of (x,y) points around the boundary.
#     """
#     half = ARENA_SIZE / 2.0

#     # sample points from the dimensions of the arena
#     xs = np.arange(-half, half + 1e-9, spacing)
#     ys = np.arange(-half, half + 1e-9, spacing)

#     top    = np.stack([xs, np.full_like(xs,  half)], axis=1) # x varies y = +half
#     bottom = np.stack([xs, np.full_like(xs, -half)], axis=1) # x varies y = -half
#     left   = np.stack([np.full_like(ys, -half), ys], axis=1) # x = -half y varies 
#     right  = np.stack([np.full_like(ys,  half), ys], axis=1)
#     border = np.vstack([top, bottom, left, right])
#     # remove duplicate rows
#     border = np.unique(border, axis=0)
#     # kept as a n by 2 numpy array 
#     return border


# monash code that just prints search list and their coordinates - dont use 
def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

"""

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically

"""


def drive_to_target_with_feedback(target_xy, threshold, wheel_vel, scale):
    """
    Drive forward until within threshold distance of target using position feedback
    """
    max_attempts = 100
    attempt = 0
    
    while attempt < max_attempts:
        # Get current robot pose (you'll need to implement this properly with your EKF)
        current_pose = get_robot_pose(robot)  # This should return [x, y, theta]
        current_xy = np.array([current_pose[0], current_pose[1]])
        
        # Calculate current distance to target
        distance = np.linalg.norm(target_xy - current_xy)
        
        print(f"Attempt {attempt}: distance to target = {distance:.2f}m")
        
        if distance <= threshold:
            print(f"Reached target! Within {threshold}m threshold.")
            ppi.set_velocity([0, 0])  # Stop
            break
            
        # Drive forward for a short duration (0.2 seconds)
        ppi.set_velocity([1, 0], tick=wheel_vel, time=0.2)
        time.sleep(0.1)  # Brief pause for pose update
        
        attempt += 1
    
    if attempt >= max_attempts:
        print("Warning: Max attempts reached, stopping drive")
        ppi.set_velocity([0, 0])


def drive_to_point(waypoint, robot_pose, is_final_target=False, target_threshold=0.3):
    """
    Drive to a waypoint with option for final target approach within threshold
    
    @param waypoint: [x, y] target position
    @param robot_pose: [x, y, theta] current robot pose  
    @param is_final_target: if True, stop when within target_threshold distance
    @param target_threshold: minimum distance to target (meters) for final approach
    """
    # Convert to numpy arrays for Helper functions
    current_robot_pose = np.array(robot_pose)
    waypoint = np.array(waypoint)

    # Load calibration parameters
    fileS = "calibration/param/scale.txt"
    fileB = "calibration/param/baseline.txt"
    scale_arr = np.loadtxt(fileS, delimiter=',')
    scale = float(np.mean(scale_arr))
    baseline = float(np.squeeze(np.loadtxt(fileB, delimiter=',')))
    
    wheel_vel = 30  # tick

    # Get initial distance and heading
    distance_to_waypoint = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))
    
    print(f"Initial distance: {distance_to_waypoint:.2f}m, heading: {heading_to_waypoint:.2f} rad")
    
    # If we're already close enough to a final target, don't move
    if is_final_target and distance_to_waypoint <= target_threshold:
        print(f"Already within {target_threshold}m of target. No movement needed.")
        return

    # Step 1: Turn to face the waypoint (always do this precisely)
    turn_time = abs((2.0 * heading_to_waypoint * scale * wheel_vel) / baseline)
    turn_dir = 1 if heading_to_waypoint >= 0 else -1
    
    print(f"Turning for {turn_time:.2f} seconds")
    ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=turn_time)
    
    # Step 2: Drive forward with different strategies
    if is_final_target:
        # For final targets: drive until within threshold using feedback
        print(f"Final approach: driving until within {target_threshold}m of target")
        drive_to_target_with_feedback(waypoint, target_threshold, wheel_vel, scale)
    else:
        # For intermediate waypoints: use timed drive (existing behavior)
        drive_time = distance_to_waypoint / (wheel_vel * scale)
        print(f"Intermediate waypoint: driving for {drive_time:.2f} seconds")
        ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    print(f"Movement complete. Target: [{float(waypoint[0]):.2f}, {float(waypoint[1]):.2f}]")


def get_robot_pose(robot): # pass in the robot instance 
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    robot_pose = robot.state  # This should return [x, y, theta]

    ####################################################

    return robot_pose

# recursively call the drive to point function 
def follow_path_with_drive_to_point(ppi, robot, path_xy, is_final_path=False, skip=3):
    """
    Follow a polyline by calling drive_to_point for sampled points.
    
    @param is_final_path: True if this path leads to a final target fruit
    """
    if not path_xy:
        return
        
    # Downsample the path
    sampled = path_xy[::max(1, skip)]
    if sampled[-1] != path_xy[-1]:
        sampled.append(path_xy[-1])

    for i, wp in enumerate(sampled):
        robot_pose = get_robot_pose(robot)
        
        # Only the very last waypoint of a final path gets the special treatment
        is_final_target = is_final_path and (i == len(sampled) - 1)
        
        drive_to_point([wp[0], wp[1]], robot_pose, 
                      is_final_target=is_final_target, 
                      target_threshold=0.3)
        time.sleep(0.01)


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M3_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    # FIXED: Added missing arguments that are used in the code
    parser.add_argument("--arena_size", type=float, default=ARENA_SIZE)
    parser.add_argument("--res", type=float, default=RES)
    parser.add_argument("--robot_r", type=float, default=ROBOT_R)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--smooth_lam", type=float, default=0.1)
    parser.add_argument("--smooth_iters", type=int, default=10)
    parser.add_argument("--skip", type=int, default=3)
    args, _ = parser.parse_known_args()

    # --- Connect to robot ---
    ppi = PenguinPi(args.ip, args.port)

    # --- Load calibration parameters ---
    wheels_scale = np.loadtxt("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/calibration/param/scale.txt", delimiter=",")
    camera_matrix = np.loadtxt("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/calibration/param/intrinsic.txt", delimiter=",")
    camera_dist   = np.loadtxt("C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/calibration/param/distCoeffs.txt", delimiter=",")

    # --- Create Robot instance (your Robot expects ppi + calibration) ---
    robot = Robot(ppi, wheels_scale, camera_matrix, camera_dist)

    # --- Load ground truth & search order ---
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    # print_target_fruits_pos(search_list, fruits_list, fruits_true_pos) # dont use this fucntion to print - clean up 

    # --- Build target list in search order - if duplicate then closest occurence 
    target_points_xy, distraction_points_xy = targets_from_search_list1(search_list, fruits_list, fruits_true_pos) # test the distractoin list - chcanged thsi to print and get cosest target 
   
    if len(target_points_xy) == 0:
        print("[ERROR] No targets found that match search_list. Exiting.")
        ppi.set_velocity([0, 0])
        sys.exit(1)

    # # --- Assemble obstacles (arena border; add more if you have them) ---
    # obstacle_points_xy = sample_arena_border(ARENA_SIZE, spacing=RES/2.0)
    # If you want to treat ArUco markers as obstacles too, uncomment:
    # obstacles_list = [obstacle_points_xy, aruco_true_pos] 
    obstacles_list = [aruco_true_pos] 

    if len(distraction_points_xy) > 0:
        # Convert distraction points to numpy array if not empty
        distraction_array = np.array(distraction_points_xy)
        obstacles_list.append(distraction_array)

    # Combine all obstacles
    obstacle_points_xy = np.vstack(obstacles_list)
    print(f"[INFO] Total obstacle points: {len(obstacle_points_xy)}")


    # --- Build planning grid once (fixed bounds) --- only need to do it once 
    costmap, occ, meta = build_costmap_fixed_2x2(
        size=args.arena_size,
        obstacle_points_m=np.array(obstacle_points_xy, dtype=np.float64),
        res=args.res,
        robot_radius=args.robot_r,
        safety_margin=args.margin,
    )

    print(f"[INFO] Costmap built: {meta['W']}x{meta['H']} cells @ {meta['res']} m/cell")

    # view  cost map 
    visualize_costmap_detailed(costmap, occ, meta, obstacle_points_xy, target_points_xy)

    # --- Autonomous run: plan & drive one leg at a time in fixed order ---
    try:
        current_xy = robot.state 
        # FIXED: Extract only x,y coordinates as a tuple
        if hasattr(robot, 'state') and robot.state is not None:
            if len(robot.state) >= 2:
                current_xy = (float(robot.state[0]), float(robot.state[1]))

        for k, goal_xy in enumerate(target_points_xy, start=1):
            print(f"\n=== Target {k}/{len(target_points_xy)}: {goal_xy} ===")

            # Plan one leg
            raw_leg, leg_cost = plan_leg_dijkstra(costmap, meta, current_xy, goal_xy)
            if raw_leg is None or len(raw_leg) == 0:
                print(f"[WARN] No path to {goal_xy}. Skipping.")
                continue

            # Smooth (optional but recommended)
            leg_path = smooth_polyline(raw_leg, lam=args.smooth_lam, iters=args.smooth_iters)
            print(f"[INFO] Path points: raw={len(raw_leg)}, smooth={len(leg_path)}, cost={leg_cost:.1f}")

            # Follow path using your existing drive_to_point (sampled)
            follow_path_with_drive_to_point(ppi, robot, leg_path, skip=args.skip)

            # Update current position (ideally from EKF; here we assume we reached the goal)
            current_xy = (goal_xy[0], goal_xy[1])
            current_xy = robot.state # double check this TODO
            # FIXED: Extract only x,y coordinates as a tuple
            if hasattr(robot, 'state') and robot.state is not None:
                if len(robot.state) >= 2:
                    current_xy = (float(robot.state[0]), float(robot.state[1]))

        print("\n[INFO] All targets processed. Stopping.")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Stopping.")
    finally:
        ppi.set_velocity([0, 0])
# ============================ END MAIN ============================



# working main code from before ###################################################

    # ppi = PenguinPi(args.ip,args.port)

    # # --- Load calibration parameters ---
    # wheels_scale = np.loadtxt("calibration/param/scale.txt", delimiter=',')
    # camera_matrix = np.loadtxt("calibration/param/intrinsic.txt", delimiter=',')
    # camera_dist = np.loadtxt("calibration/param/distCoeffs.txt", delimiter=',')

    # # --- Create Robot instance with all required arguments ---
    # robot = Robot(ppi, wheels_scale, camera_matrix, camera_dist)


    # # read in the true map
    # fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    # search_list = read_search_list()
    # print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # waypoint = [0.0,0.0]
    # robot_pose = [0.0,0.0,0.0]

    # # The following is only a skeleton code for semi-auto navigation
    # while True:
    #     # enter the waypoints
    #     # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
    #     x,y = 0.0,0.0
    #     x = input("X coordinate of the waypoint: ")
    #     try:
    #         x = float(x)
    #     except ValueError:
    #         print("Please enter a number.")
    #         continue
    #     y = input("Y coordinate of the waypoint: ")
    #     try:
    #         y = float(y)
    #     except ValueError:
    #         print("Please enter a number.")
    #         continue

    #     # estimate the robot's pose
    #     robot_pose = get_robot_pose(robot)
    #     # print out the robot pose 
    #     print(f"the robots position is x: {robot_pose[0]}, y: {robot_pose[1]}, theta: {robot_pose[2]}")

    #     # robot drives to the waypoint
    #     waypoint = [x,y]
    #     drive_to_point(waypoint,robot_pose)
    #     print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

    #     # exit
    #     ppi.set_velocity([0, 0])
    #     uInput = input("Add a new waypoint? [Y/N]")
    #     if uInput == 'N':
    #         break

 