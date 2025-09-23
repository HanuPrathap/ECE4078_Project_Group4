# test.py
import sys, os, time
import numpy as np
import cv2

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


# -----------------------------
# EKF / SLAM glue
# -----------------------------
def init_ekf(calib_dir: str, ip: str) -> EKF:
    """
    Mirror of your working Operate.init_ekf().
    Loads camera + wheel calibration, builds Robot, then EKF(robot).
    """
    fileK = f"{calib_dir}intrinsic.txt"
    fileD = f"{calib_dir}distCoeffs.txt"
    fileS = f"{calib_dir}scale.txt"
    fileB = f"{calib_dir}baseline.txt"

    camera_matrix = np.loadtxt(fileK, delimiter=',')
    dist_coeffs   = np.loadtxt(fileD, delimiter=',')
    scale         = np.loadtxt(fileS, delimiter=',')
    if ip == 'localhost':
        # simulator scale is different
        scale /= 2.0
    baseline     = np.loadtxt(fileB, delimiter=',')

    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    return EKF(robot)


class ArucoSLAM:
    """
    Thin wrapper around your existing flow:
      detect_marker_positions(img) -> (lms, dbg_img)
      ekf.predict(drive) -> ekf.add_landmarks(lms) -> ekf.update(lms)
    """
    def __init__(self, ekf: EKF, marker_length: float = 0.07):
        self.ekf = ekf
        self.det = aruco.aruco_detector(self.ekf.robot, marker_length=marker_length)
        self.ekf_on = True
        self._need_recover = False

    def set_pause(self, pause: bool):
        self.ekf_on = (not pause)

    def request_recover(self):
        self._need_recover = True

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
    """
    Returns (x, y, theta) from EKF. Uses get_state_vector() which your EKF exposes.
    """
    mu = np.array(ekf.get_state_vector()).reshape(-1)
    return float(mu[0]), float(mu[1]), float(mu[2])


def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi




# -----------------------------
# Your original time-based planner, now returning EKF pose
# -----------------------------
def drive_to_point(ppi, waypoint, robot_pose, ekf):
    """
    Keep your time-based motion (turn_time, drive_time), but execute it in slices
    and fuse with SLAM. Returns the EKF pose after this waypoint.
    """


    current_robot_pose = np.array(robot_pose, dtype=float)
    waypoint = np.array(waypoint, dtype=float)

    wheel_vel = 40  # ticks/s (adjust to your robot)

    # Load calibration parameters
    scale = float(np.mean(np.loadtxt("calibration/param/scale.txt", delimiter=',')))
    baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))

    # Distance + heading
    distance_to_waypoint = float(np.squeeze(get_distance_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint  = float(np.squeeze(get_angle_robot_to_goal(current_robot_pose, waypoint)))
    heading_to_waypoint  = wrap_to_pi(heading_to_waypoint)

    # Convert to times (your original logic)
    drive_time = distance_to_waypoint / (wheel_vel * scale + 1e-9)
    ang_rate   = (2.0 * wheel_vel * scale) / (baseline + 1e-9)  # rad/s
    turn_time  = abs(heading_to_waypoint) / (ang_rate + 1e-9)
    turn_dir   = 1 if heading_to_waypoint >= 0 else -1

    print(f"Distance: {distance_to_waypoint:.2f} m, Heading: {heading_to_waypoint:.3f} rad")
    print(f"Turning for {turn_time:.2f} s, then driving for {drive_time:.2f} s")


    # Execute turn
    print(f"Turning for {turn_time:.2f}s")
    ppi.set_velocity([0, turn_dir], turning_tick=wheel_vel, time=turn_time)
    time.sleep(turn_time + 0.1)

    # Execute drive
    print(f"Driving for {drive_time:.2f}s")
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    time.sleep(3) # sleep for 3 seconds after every way point - change this later

    # Return EKF pose (single source of truth)
    x, y, th = get_ekf_pose(ekf)
    print(f"EKF pose after segment: x={x:.3f}, y={y:.3f}, th={th:.3f}")
    return np.array([x, y, th], dtype=float)


def drive_to_waypoints(waypoints, ppi, ekf, slam, ip):
    """
    Drive through all waypoints using time-based segments,
    while SLAM keeps the pose accurate; pass EKF pose forward each hop.
    """
    robot_pose = get_ekf_pose(ekf) # getting pose from  ekf
    print(f"Starting from: {robot_pose}")

    for i, waypoint in enumerate(waypoints):
        print(f"\nNavigating to waypoint {i + 1}/{len(waypoints)}: {waypoint}")
        print(f"Current pose (before): {robot_pose}")
        robot_pose = drive_to_point(ppi, waypoint, robot_pose, ekf)
        print(f"Current pose (after):  {robot_pose}")

    print("\nAll waypoints completed!")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default="192.168.50.1")
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--self_test", action="store_true", help="Do 1s forward motion test at start")
    args, _ = parser.parse_known_args()

    # Setup
    ip = args.ip
    ppi = PenguinPi(ip, args.port)

    # Optional: 1-second self-test to confirm motion path works
    if args.self_test:
        print("[self-test] forward 1s @ tick=40")
        ppi.set_velocity([1, 0], tick=40, turning_tick=40)
        time.sleep(1.0)
        ppi.set_velocity([0, 0]); time.sleep(0.3)

    # EKF/SLAM init
    ekf  = init_ekf(args.calib_dir, ip)
    slam = ArucoSLAM(ekf, marker_length=0.07)

    # Waypoints
    waypoints = [
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
        [0.0, -0.5],
        [0.0, 0.0]
    ]

    print("Starting navigation with waypoints:")
    for i, wp in enumerate(waypoints, 1):
        print(f"  {i}. [{wp[0]:.2f}, {wp[1]:.2f}]")

    # Run
    drive_to_waypoints(waypoints, ppi, ekf, slam, ip)

    print("Navigation complete!")
