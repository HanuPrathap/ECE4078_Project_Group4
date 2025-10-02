# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import pygame

# import SLAM components
sys.path.insert(0, "slam")
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure
import util.DatasetHandler as dh

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for"""
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

def init_ekf(datadir, ip):
    """Initialize EKF with calibration parameters for localization-only mode"""
    fileK = "calibration/param/intrinsic.txt"
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    if ip == 'localhost':
        scale /= 2
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    return EKF(robot)

def load_map_to_ekf(ekf, aruco_true_pos):
    """Load the known map into EKF for localization-only mode"""
    # Set markers from the true map
    ekf.markers = aruco_true_pos.T  # Convert to 2xN format
    ekf.taglist = list(range(1, 11))  # Markers 1 through 10
    
    # Initialize covariance for landmarks as very small (known positions)
    n_landmarks = ekf.number_landmarks()
    if n_landmarks > 0:
        # First, we need to resize the covariance matrix to include all landmarks
        total_state_size = 3 + 2 * n_landmarks  # 3 for robot + 2 per landmark
        
        # Create a new covariance matrix of the correct size
        new_P = np.zeros((total_state_size, total_state_size))
        
        # Copy the existing robot covariance (first 3x3 block)
        new_P[0:3, 0:3] = ekf.P[0:3, 0:3]
        
        # Set small covariance for known landmarks
        landmark_cov = 0.001 * np.eye(2 * n_landmarks)
        new_P[3:, 3:] = landmark_cov
        
        # Replace the old covariance matrix
        ekf.P = new_P
    
    print(f"Loaded {n_landmarks} known landmarks for localization")

class LocalizationSystem:
    def __init__(self, args, aruco_true_pos):
        self.ppi = PenguinPi(args.ip, args.port)
        self.ekf = init_ekf(args.calib_dir, args.ip)
        
        # Load known map FIRST, before creating aruco_detector
        load_map_to_ekf(self.ekf, aruco_true_pos)
        
        # Now create the detector with the updated EKF
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)
        
        # Localization state
        self.ekf_on = True
        self.last_print_time = time.time()
        self.print_interval = 2.0  # Print position every 2 seconds
        self.control_clock = time.time()
        
        # For teleoperation recording
        self.data = dh.DatasetWriter('teleop_record') if args.save_data else None
        
        # Control command (like operate.py)
        self.command = {'motion': [0, 0]}  # [forward, turn]
        
    def get_robot_pose(self):
        """Get current robot pose from EKF"""
        return self.ekf.robot.state.flatten()  # [x, y, theta]
    
    def print_robot_pose(self):
        """Print robot pose in a readable format"""
        pose = self.get_robot_pose()
        print(f"Robot Pose: x={pose[0]:.3f}m, y={pose[1]:.3f}m, θ={np.degrees(pose[2]):.1f}°")
    
    def update_localization(self):
        """Perform one localization update cycle"""
        # Get camera image
        img = self.ppi.get_image()
        
        # Detect ARUCO markers
        measurements, aruco_img = self.aruco_det.detect_marker_positions(img)
        
        if self.ekf_on and measurements:
            # Update EKF with marker measurements (localization only - no new landmarks)
            self.ekf.update(measurements)
        
        return img, measurements
    
    def control(self):
        """Apply control commands to robot (like operate.py)"""
        lv, rv = self.ppi.set_velocity(self.command['motion'])
        
        # Record teleoperation data if saving
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        
        dt = time.time() - self.control_clock
        
        # Create drive measurement for EKF prediction
        if args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)  # right wheel reversed on physical robot
        
        self.control_clock = time.time()
        return drive_meas
    
    def update_keyboard(self):
        """Handle keyboard input (like operate.py)"""
        for event in pygame.event.get():
            # Movement commands
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [1, 0]  # Forward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-1, 0]  # Backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 1]   # Turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -1]  # Turn right
            # Stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]   # Stop
            # Quit
            elif event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

def main_loop():
    parser = argparse.ArgumentParser("Fruit searching with localization")
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true', help="Save teleoperation data")
    parser.add_argument("--mode", type=str, choices=['teleop', 'auto'], default='teleop', 
                       help="Operation mode: teleop or auto")
    args, _ = parser.parse_known_args()

    # Initialize pygame PROPERLY (like operate.py)
    pygame.font.init()
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab - Localization Mode')
    canvas.fill((0, 0, 0))
    pygame.display.update()

    # Read true map and initialize localization system
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    loc_system = LocalizationSystem(args, aruco_true_pos)
    
    print("Localization System Ready!")
    print("Controls:")
    print("  UP ARROW: Move forward")
    print("  DOWN ARROW: Move backward") 
    print("  LEFT ARROW: Turn left")
    print("  RIGHT ARROW: Turn right")
    print("  SPACE: Stop")
    print("  ESC: Quit")
    print("Robot pose will be printed every 2 seconds")
    
    # Main loop (like operate.py)
    running = True
    
    while running:
        # Handle keyboard input
        quit_signal = loc_system.update_keyboard()
        if quit_signal:
            running = False
            break
        
        # Take picture and update localization
        loc_system.take_pic()
        
        # Apply control and get drive measurement
        drive_meas = loc_system.control()
        
        # Update SLAM with drive measurement
        if loc_system.ekf_on:
            loc_system.ekf.predict(drive_meas)
        
        # Update localization with camera measurements
        img, measurements = loc_system.update_localization()
        
        # Print robot pose periodically
        current_time = time.time()
        if current_time - loc_system.last_print_time >= loc_system.print_interval:
            loc_system.print_robot_pose()
            loc_system.last_print_time = current_time
        
        # Small delay to prevent excessive CPU usage
        time.sleep(0.05)
    
    # Cleanup
    loc_system.ppi.set_velocity([0, 0])
    pygame.quit()
    print("Localization system shut down")

if __name__ == "__main__":
    main_loop()