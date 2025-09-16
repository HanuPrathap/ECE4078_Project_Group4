# M4 - Autonomous fruit searching

# this file loads a ground truth map 
# prints target fruits position in a specified order 
# provides a skeletonn for naviagtion where you enter way points and the robot drives to them 



# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

# import SLAM components
sys.path.insert(0, "C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
sys.path.insert(0, "C:/Users/gurvi/Desktop/ECE4078/Project_G04/ECE4078_Project_Group4/Week07-08/util".format(os.getcwd()))
from pibot import PenguinPi
import measure as measure


# import Helper functions - jas
from Helper import * 

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
    with open(fname, 'r') as fd:
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
    
# fruit_list = ["orange", "apple", "potato"]  of all the fruit in the map including distractor fruit - python list
# fruit_true_pos = (n_targets, 2) storing x, y - numpy array
# aruco_true_pos = (10,2) for markers 1-10 - numpy array 


# this function gets the search.txt and converts it into a list so we can use it to go to target fruits
def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


# this function takes in the search list we made and the fruit list and pose and prints out search fruits target pose
# TODO - jas: we can return a list of only the target fruit pose from this and use that for our code 
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


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically

def drive_to_point(waypoint, robot_pose):
    # for now this does not avoid obtacles only straight line motion and rotation on the spot 

    # Convert to numpy arrays for Helper functions
    current_robot_pose = np.array(robot_pose)
    waypoint = np.array(waypoint)


    # TODO - jas: create varibles used to navigate to waypoints 
    threshold_distance = 0.25 # 0.25 meters 
    threshold_angle = 0.05  # Tighter threshold # doesnt matter for now 
    max_iterations = 1000
    iteration = 0

    # TODO - jas: might need to change this
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    # Params
    wheel_vel = 30  # tick

    # Load calibration and coerce to scalars
    fileS = "calibration/param/scale.txt"
    fileB = "calibration/param/baseline.txt"
    scale_arr = np.loadtxt(fileS, delimiter=',')
    # If scale.txt has two values (L/R), take the mean to get a scalar
    scale = float(np.mean(scale_arr))
    baseline = float(np.squeeze(np.loadtxt(fileB, delimiter=',')))

    # Get scalar distance and heading
    distance_to_waypoint = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))
    print(f"the angle to turn to face waypoint {heading_to_waypoint} rads")

    # Compute times as scalars
    drive_time = distance_to_waypoint / (wheel_vel * scale)
    turn_time = (2.0 * heading_to_waypoint * scale * wheel_vel) / baseline
    turn_dir = 1 if heading_to_waypoint >= 0 else -1

    # Turn
    print(f"Turning for {turn_time:.2f} seconds")
    ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=turn_time)

    # Drive straight
    print(f"Driving for {drive_time:.2f} seconds")
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    print(f"Arrived at [{float(waypoint[0]):.2f}, {float(waypoint[1]):.2f}]")


def get_robot_pose(robot): # pass in the robot instance 
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    robot_pose = robot.state  # This should return [x, y, theta]

    ####################################################

    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M3_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()


    ppi = PenguinPi(args.ip,args.port)

    # --- Load calibration parameters ---
    wheels_scale = np.loadtxt("calibration/param/scale.txt", delimiter=',')
    camera_matrix = np.loadtxt("calibration/param/intrinsic.txt", delimiter=',')
    camera_dist = np.loadtxt("calibration/param/distCoeffs.txt", delimiter=',')

    # --- Create Robot instance with all required arguments ---
    robot = Robot(ppi, wheels_scale, camera_matrix, camera_dist)


    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # The following is only a skeleton code for semi-auto navigation
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        robot_pose = get_robot_pose(robot)
        # print out the robot pose 
        print(f"the robots position is x: {robot_pose[0]}, y: {robot_pose[1]}, theta: {robot_pose[2]}")

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break

        # # Simple test for drive_to_point
        # # Example: drive from (0,0,0) to (1, 0.5)
        # test_robot_pose = [0.0, 0.0, 0.0]  # [x, y, theta]
        # test_waypoint = [1.0, 1.0]         # [x, y]
        # print("Testing drive_to_point with:")
        # print(f"  Start pose: {test_robot_pose}")
        # print(f"  Waypoint:   {test_waypoint}")
        # drive_to_point(test_waypoint, test_robot_pose)
