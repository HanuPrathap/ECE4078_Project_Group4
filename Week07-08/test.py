import sys, os
import numpy as np
import time

# Util & robot I/O
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure     # Drive(lv, rv, dt)

# SLAM core
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# Import helper functions
from Helper import * 


# add a function to initiliase the ekf
def init_ekf(calib_dir: str, ip: str)-> EKF:
    # same file names as in your Operate.init_ekf
    fileK = f"{calib_dir}intrinsic.txt"
    fileD = f"{calib_dir}distCoeffs.txt"
    fileS = f"{calib_dir}scale.txt"
    fileB = f"{calib_dir}baseline.txt"

    camera_matrix = np.loadtxt(fileK, delimiter=',')
    dist_coeffs   = np.loadtxt(fileD, delimiter=',')
    scale         = np.loadtxt(fileS, delimiter=',')
    if ip == 'localhost':
        # simulation encoder ticks/scale are different
        scale /= 2.0
    baseline     = np.loadtxt(fileB, delimiter=',')

    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    return EKF(robot)


class DriveFeeder:
    def __init__(self, pibot: PenguinPi, ip: str):
        self.pibot = pibot
        self.ip = ip
        self._t_last = time.time()

    def send_and_measure(self, motion_cmd):
        lv, rv = self.pibot.set_velocity(motion_cmd)
        t = time.time()
        dt = t - self._t_last
        self._t_last = t
        dt = max(dt, 1e-3)
        if self.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        return drive_meas


class ArucoSLAM:
    def __init__(self, ekf: EKF, marker_length: float = 0.07):
        self.ekf = ekf
        self.det = aruco.aruco_detector(self.ekf.robot, marker_length=marker_length)
        self.ekf_on = True
        self._need_recover = False

    def step(self, rgb_image, drive_meas):
        lms, aruco_img = self.det.detect_marker_positions(rgb_image)
        if self._need_recover:
            ok = self.ekf.recover_from_pause(lms)
            self.ekf_on = bool(ok)
            self._need_recover = False
        if self.ekf_on:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
        return get_ekf_pose(self.ekf), aruco_img


def get_ekf_pose(ekf: EKF):
    mu = np.array(ekf.get_state_vector()).reshape(-1)
    return float(mu[0]), float(mu[1]), float(mu[2])


def wrap_to_pi(a):
    """Wrap angle to [-π, π]"""
    return (a + np.pi) % (2*np.pi) - np.pi


def drive_to_point(ppi, waypoint, robot_pose):
    # ppi input
    # robot pose input 
    # goal point

    current_robot_pose = np.array(robot_pose, dtype=float)
    waypoint = np.array(waypoint, dtype=float)

    wheel_vel = 30  # ticks/s

    # Load calibration parameters
    scale = float(np.mean(np.loadtxt("calibration/param/scale.txt", delimiter=',')))
    baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))

    # Calculate distance and angle to waypoint
    distance_to_waypoint = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint = wrap_to_pi(heading_to_waypoint)

    # Compute movement times
    drive_time = distance_to_waypoint / (wheel_vel * scale)
    ang_rate = (2.0 * wheel_vel * scale) / baseline  # rad/s
    turn_time = abs(heading_to_waypoint) / ang_rate
    turn_dir = 1 if heading_to_waypoint >= 0 else -1

    print(f"Distance to point: {distance_to_waypoint:.2f}m, Angle to point: {heading_to_waypoint:.3f}rad")

    # Execute turn
    print(f"Turning for {turn_time:.2f}s")
    ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=turn_time)
    time.sleep(turn_time + 0.1)

    # Execute drive
    print(f"Driving for {drive_time:.2f}s")
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    time.sleep(3) # sleep for 3 seconds after every way point - change this later

    # Stop
    ppi.set_velocity([0, 0])

    # --- Odometry update (matches commands) ---
    x, y, th = current_robot_pose
    th = wrap_to_pi(th + heading_to_waypoint)  # apply the commanded rotation
    d = wheel_vel * scale * drive_time        # equals distance_to_waypoint
    x = x + d * np.cos(th)
    y = y + d * np.sin(th)

    new_pose = np.array([x, y, th], dtype=float)
    print(f"New odometry pose mathematically calculated: x={x:.3f}, y={y:.3f}, th={th:.3f}")

    return new_pose

def drive_to_waypoints(waypoints, ppi):
    """
    Drive through all waypoints in sequence using odometry updates
    """
    # Start from origin
    robot_pose = np.array([0.0, 0.0, 0.0], dtype=float)
    print(f"Starting from: {robot_pose}")
    
    for i, waypoint in enumerate(waypoints):
        print(f"\nNavigating to waypoint {i + 1}/{len(waypoints)}: {waypoint}")
        print(f"Current pose: {robot_pose}")
        
        # Drive to the waypoint and get updated odometry
        robot_pose = drive_to_point(ppi, waypoint, robot_pose)
    
    print("\nAll waypoints completed!")

# Main execution
if __name__ == "__main__":
    # Robot connection
    ppi = PenguinPi('192.168.50.1', 8080)

    # Hardcoded list of waypoints
    waypoints = [
        [0.5, 0.0],
        [-0.5, 0.0], 
        [0.0, 0.5],
        [0.0, -0.5],
        [0.0, 0.0]
    ]

    print("Starting navigation with waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"Waypoint {i+1}: [{wp[0]:.2f}, {wp[1]:.2f}]")

    # Drive to all waypoints with odometry updates
    drive_to_waypoints(waypoints, ppi)

    print("Navigation complete!")