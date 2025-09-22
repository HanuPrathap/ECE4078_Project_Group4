# M4 - Autonomous fruit searching

# this file loads a ground truth map 
# prints target fruits position in a specified order 
# provides a skeletonn for naviagtion where you enter way points and the robot drives to them 

# RUN python Week07-08/gui.py image_to_map_generator/generated_ground_truth_maps/final_output.txt


ARENA_SIZE_M = 3   # need to change this 2.4 on the actual day
WORLD_X_MIN, WORLD_X_MAX = -ARENA_SIZE_M / 2, ARENA_SIZE_M / 2
WORLD_Y_MIN, WORLD_Y_MAX = -ARENA_SIZE_M / 2, ARENA_SIZE_M / 2

PAD = 20
PANEL_W = 720
PANEL_H = 720
MAP_ORIGIN_PX = (PAD, PAD)


CAM_W, CAM_H = 320, 240  # PiBot camera size
MAP_W, MAP_H = PANEL_W, PANEL_H

# Window width = map width + padding + camera width + padding
WIN_W = MAP_W + PAD + CAM_W + 2*PAD
WIN_H = max(CAM_H + 2*PAD, MAP_H + 2*PAD)


GOAL_TOL_M = 0.2  # 20 cm tolerance

# basic python packages
import math
import pygame # for GUI
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import matplotlib.pyplot as plt #  for GUI
from dataclasses import dataclass


# import SLAM components
sys.path.insert(0, "C:/Users/jessi/Documents/2025/ECE4078/ECE4078_Project_Group4/Week07-08/slam") # need to modify this assuming Jas will run during demo

from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
sys.path.insert(0, "C:/Users/jessi/Documents/2025/ECE4078/ECE4078_Project_Group4/Week07-08/util") # need to modify this assuming Jas will run during demo


from util.pibot import PenguinPi
import util.measure as measure


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
            x = (np.round(gt_dict[key]['x'], 1))
            y = (np.round(gt_dict[key]['y'], 1))

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
    fname = "C:/Users/jessi/Documents/2025/ECE4078/ECE4078_Project_Group4/Week07-08/search_list.txt"  # also need to change this for Jas
    with open(fname, 'r') as fd:
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

    # TODO 
    threshold_distance = 0.25 # 0.25 meters 
    threshold_angle = 0.05  # Tighter threshold # doesnt matter for now 
    max_iterations = 1000
    iteration = 0

    # TODO 
    # imports camera / wheel calibration parameters 
    fileS = "Week07-08/calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "Week07-08/calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    # Params
    wheel_vel = 30  # tick

    # Load calibration and coerce to scalars
    fileS = "Week07-08/calibration/param/scale.txt"
    fileB = "Week07-08/calibration/param/baseline.txt"
    scale_arr = np.loadtxt(fileS, delimiter=',')
    # If scale.txt has two values (L/R), take the mean to get a scalar
    scale = float(np.mean(scale_arr))
    baseline = float(np.squeeze(np.loadtxt(fileB, delimiter=',')))

    # Get scalar distance and heading
    distance_to_waypoint = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    
    heading_to_waypoint = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))
    print(f"the angle to turn to face waypoint {heading_to_waypoint} rads")

    # Compute times as scalars
    drive_time = abs(distance_to_waypoint / (wheel_vel * scale)  )
    drive_time = min(drive_time, 19.0)  # stay under 20s

    turn_time = abs((2.0 * heading_to_waypoint * scale * wheel_vel) / baseline)
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


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    th: float = 0.0  # radians


def rotate_ccw_90(x,y):
    return y, -x


def rotate_cw_90_px(u, v):
    """Rotate a pixel coordinate 90° clockwise around the center of the map panel."""
    cx = MAP_ORIGIN_PX[0] + PANEL_W / 2
    cy = MAP_ORIGIN_PX[1] + PANEL_H / 2
    
    # Translate so center is origin
    dx = u - cx
    dy = v - cy
    
    # Rotate 90° clockwise
    rx = dy
    ry = dx
    
    # Translate back
    return int(rx + cx), int(ry + cy)


def px_to_world(u: int, v: int):
    x0, y0 = MAP_ORIGIN_PX
    if not (x0 <= u <= x0 + PANEL_W and y0 <= v <= y0 + PANEL_H):
        raise ValueError("Click outside map panel")
    su = (u - x0) / PANEL_W
    sv = (v - y0) / PANEL_H
    x = WORLD_X_MIN + su * (WORLD_X_MAX - WORLD_X_MIN)
    y = WORLD_Y_MAX - sv * (WORLD_Y_MAX - WORLD_Y_MIN)
    return float(x), float(y)


def world_to_px(x: float, y: float, origin_px=MAP_ORIGIN_PX):
    x0, y0 = origin_px
    x0, y0 = rotate_ccw_90(x0,y0)
    su = (x - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN)
    sv = (WORLD_Y_MAX - y) / (WORLD_Y_MAX - WORLD_Y_MIN)
    u = int(x0 + su * PANEL_W)
    v = int(y0 + sv * PANEL_H)
    return u, v


def draw_map(screen, origin_px=MAP_ORIGIN_PX):
    x0, y0 = origin_px
    pygame.draw.rect(screen, (30, 30, 30), (x0, y0, PANEL_W, PANEL_H))
    pygame.draw.rect(screen, (90, 90, 90), (x0, y0, PANEL_W, PANEL_H), 2)
    for gx in np.linspace(WORLD_X_MIN, WORLD_X_MAX, 6):
        u, _ = world_to_px(gx, 0.0, origin_px=origin_px)
        pygame.draw.line(screen, (55, 55, 55), (u, MAP_ORIGIN_PX[1]), (u, MAP_ORIGIN_PX[1]+PANEL_H))
    for gy in np.linspace(WORLD_Y_MIN, WORLD_Y_MAX, 6):
        _, v = world_to_px(0.0, gy, origin_px=origin_px)
        pygame.draw.line(screen, (55, 55, 55), (MAP_ORIGIN_PX[0], v), (MAP_ORIGIN_PX[0]+PANEL_W, v))

def draw_goal(screen, goal_xy, origin_px=MAP_ORIGIN_PX):
    if goal_xy is None:
        return
    u, v = world_to_px(*goal_xy, origin_px = origin_px )
    # Draw the goal as a small yellow circle
    pygame.draw.circle(screen, (255, 200, 0), (u, v), 6)
    # Draw acceptance radius
    rad_px = int(GOAL_TOL_M / (WORLD_X_MAX - WORLD_X_MIN) * PANEL_W)
    pygame.draw.circle(screen, (255, 200, 0), (u, v), rad_px, 1)

def draw_groundtruth_map(screen, fruits_true_pos, fruit_list, aruco_true_pos, origin_px = MAP_ORIGIN_PX):
    """
    Draws fruits and ArUco markers on the 480x480 map.
    """
    # Draw fruits
    for i, pos in enumerate(fruits_true_pos):
        u, v = world_to_px(pos[0], pos[1], origin_px = origin_px)
        u,v = rotate_cw_90_px(u,v)
        #u,v = rotate_ccw_90(u,v)
        pygame.draw.circle(screen, (255, 165, 0), (u, v), 6)  # orange dot for fruits
        # optional: label
        font = pygame.font.SysFont("consolas", 14)
        label = font.render(fruit_list[i], True, (255, 255, 255))
        screen.blit(label, (u+5, v-5))

    # Draw ArUco markers
    for i, pos in enumerate(aruco_true_pos):
        u, v = world_to_px(pos[0], pos[1], origin_px = origin_px)
        # u,v = rotate_ccw_90(u,v)
        u,v = rotate_cw_90_px(u,v)
        pygame.draw.rect(screen, (0, 255, 0), (u-4, v-4, 8, 8))  # green square for markers
        font = pygame.font.SysFont("consolas", 12)
        label = font.render(f"A{i+1}", True, (0,255,0))
        screen.blit(label, (u+5, v-5))

def draw_robot(screen, pose: Pose, origin_px = MAP_ORIGIN_PX):
    thet_rot = pose.th + math.pi/2
    u, v = world_to_px(pose.x, pose.y, origin_px = origin_px)
    pygame.draw.circle(screen, (0, 180, 255), (u, v), 8)  # blue dot
    hx = u + int(14 * math.cos(thet_rot))
    hy = v - int(14 * math.sin(thet_rot))
    pygame.draw.line(screen, (0, 255, 180), (u, v), (hx, hy), 2)  # heading line

def draw_pibot_camera(pibot_frame, screen, pos):
    # --- Draw PiBot camera POV ---
    if pibot_frame is not None:
        cam_view = cv2.resize(pibot_frame, (CAM_W, CAM_H))
        cam_view = cv2.cvtColor(cam_view, cv2.COLOR_BGR2RGB)
        cam_surf = pygame.surfarray.make_surface(cam_view.swapaxes(0,1))
        screen.blit(cam_surf, pos)

        # Caption
        font = pygame.font.SysFont("consolas", 16)
        label = font.render("PiBot Cam", True, (220, 220, 220))
        screen.blit(label, (pos[0], pos[1]-20))

def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M3_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()


    ppi = PenguinPi(args.ip,args.port)

    # --- Load calibration parameters ---
    wheels_scale = np.loadtxt("Week07-08/calibration/param/scale.txt", delimiter=',')
    camera_matrix = np.loadtxt("Week07-08/calibration/param/intrinsic.txt", delimiter=',')
    camera_dist = np.loadtxt("Week07-08/calibration/param/distCoeffs.txt", delimiter=',')

    # --- Create Robot instance with all required arguments ---
    robot = Robot(ppi, wheels_scale, camera_matrix, camera_dist)
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("M3 Fruit Search - Click-to-Go")
    clock = pygame.time.Clock()
    robot_pose = Pose(0.0, 0.0, 0.0) 

    goal_xy = None
    busy = False
    last_msg = ""

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # emergency stop
                    ppi.set_velocity([0,0])
                    goal_xy = None
                    busy = False
                elif event.key == pygame.K_r:
                    robot_pose = Pose(0.0, 0.0, 0.0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not busy:
                mx, my = event.pos
                try:
                    gx, gy = px_to_world(mx, my)
                    goal_xy = (gx, gy) # flip for inversion
                    last_msg = f"New goal: ({gx:.2f}, {gy:.2f})"
                except ValueError:
                    pass

        # drive to goal if set
        if goal_xy is not None and not busy:
            drive_to_point([goal_xy[0], goal_xy[1]], [robot_pose.x, robot_pose.y, robot_pose.th])
            robot_pose.x, robot_pose.y = goal_xy
            goal_xy = None

        # Draw map + robot
    
        screen.fill((10,10,12))
        map_pos = (PAD, PAD)
        draw_map(screen, origin_px = map_pos)  # grid + background
        
        draw_groundtruth_map(screen, fruits_true_pos, fruits_list, aruco_true_pos, origin_px = map_pos)
        if goal_xy:
            draw_goal(screen, goal_xy, origin_px = map_pos)
        camera_pos = (MAP_W + 2*PAD, PAD)
        
        pibot_frame = ppi.get_camera_frame()  # refresh pibot camera
        draw_pibot_camera(pibot_frame, screen, pos = camera_pos)

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


    # TODOS
    ## change parser arg -- DONE
    ## change GUI so starts with robot front facing north --> DONE
    ## change waypoints so it is meters based not pixel based
    # add piBot camera to GUI