
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



def execute_motion_with_slam(ppi, slam, motion_command, duration, scale, baseline):
    """
    Execute motion while continuously running SLAM updates
    
    Args:
        ppi: PenguinPi robot interface
        slam: ArucoSLAM instance
        motion_command: [forward, turn] where forward=1 for forward, turn=-1/0/1 for left/none/right
        duration: how long to execute the motion
        scale: wheel scale parameter  
        baseline: robot baseline parameter
    """
    print(f"Executing motion {motion_command} for {duration:.2f}s with SLAM")
    
    # Start the motion and get actual wheel velocities
    if motion_command[0] == 1:  # Forward motion
        lv, rv = ppi.set_velocity([1, motion_command[1]], tick=40)
    else:  # Pure rotation  
        lv, rv = ppi.set_velocity([0, motion_command[1]], turning_tick=40)
    
    # Run SLAM loop during motion
    start_time = time.time()
    last_time = start_time
    
    while (time.time() - start_time) < duration:
        try:
            # Get camera image
            img = ppi.get_image()
            
            # Calculate dt for this iteration
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)  # Cap dt to avoid huge jumps
            if dt < 0.01:  # Skip if too small
                time.sleep(0.01)
                continue
            last_time = current_time
            
            # Create Drive measurement object (like in Operate.py)
            # Note: For physical robot, right wheel is already negated in set_velocity
            drive_meas = measure.Drive(lv, rv, dt)
            
            # Run SLAM step
            pose, aruco_img = slam.step(img, drive_meas)
            
            # Small sleep to prevent overwhelming the system
            time.sleep(0.05)
            
        except Exception as e:
            print(f"SLAM update error: {e}")
            time.sleep(0.05)
    
    # Stop the robot
    ppi.set_velocity([0, 0])
    time.sleep(0.1)
    
    return get_ekf_pose(slam.ekf)# test.py


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

def execute_motion_with_slam(ppi, slam, motion_command, duration, scale, baseline, ip="192.168.50.1", tick=40, slice_dt=0.10):
    """
    Execute a time-based motion while continuously running SLAM updates in small slices.
    motion_command: [forward, turn] with values in {-1,0,1}.
      forward: 1=forward, 0=stop
      turn:    -1=left, 0=none, 1=right  (same convention as your teleop)
    """
    fwd, turn = motion_command
    # Quantize to {-1,0,1}
    fwd  = 1 if fwd  > 0 else (-1 if fwd  < 0 else 0)
    turn = 1 if turn > 0 else (-1 if turn < 0 else 0)

    print(f"Executing motion [{fwd}, {turn}] for {duration:.2f}s with SLAM")

    t0 = time.time()
    t_last = t0

    while (time.time() - t0) < duration:
        # 1) Re-issue the command every slice (no 'time=' here)
        #    Use tick for straight, turning_tick for rotation; both ok together.
        lv, rv = ppi.set_velocity([fwd, turn], tick=tick, turning_tick=tick)

        # 2) dt for this slice
        now = time.time()
        dt = max(now - t_last, 1e-3)
        t_last = now

        # 3) Build the Drive() properly (this is what SLAM expects)
        #    Real robot has the right wheel reversed sign vs sim, same as Operate.control()
        if ip == "localhost":
            drive_meas = measure.Drive(lv,  rv, dt)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)

        # 4) Grab a frame and step SLAM (predict + update)
        img = ppi.get_image()
        try:
            _pose, _aruco_img = slam.step(img, drive_meas)
        except Exception as e:
            # If you still see left_speed/right_speed errors, it means drive_meas isn't a measure.Drive instance.
            print(f"[warn] SLAM step failed: {e}")

        # 5) Pace the loop
        time.sleep(slice_dt)

    # stop at the end of this segment
    ppi.set_velocity([0, 0])
    time.sleep(0.1)

    return get_ekf_pose(slam.ekf)


# -----------------------------
# Updated motion planner with SLAM integration
# -----------------------------
def drive_to_point(ppi, waypoint, robot_pose, ekf, slam):
    """
    Drive to waypoint while running SLAM continuously
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

    # Execute turn with SLAM
    if turn_time > 0.05:  # Only turn if significant rotation needed
        print(f"Turning for {turn_time:.2f}s")
        pose = execute_motion_with_slam(ppi, slam, [0, turn_dir], turn_time, scale, baseline)
        print(f"Pose after turn: x={pose[0]:.3f}, y={pose[1]:.3f}, th={pose[2]:.3f}")

    # Execute drive with SLAM
    if drive_time > 0.05:  # Only drive if significant distance needed
        print(f"Driving for {drive_time:.2f}s")
        pose = execute_motion_with_slam(ppi, slam, [1, 0], drive_time, scale, baseline)
        print(f"Pose after drive: x={pose[0]:.3f}, y={pose[1]:.3f}, th={pose[2]:.3f}")

    # Final pose from EKF
    final_pose = get_ekf_pose(ekf)
    print(f"EKF pose after segment: x={final_pose[0]:.3f}, y={final_pose[1]:.3f}, th={final_pose[2]:.3f}")
    
    # Brief pause between waypoints
    time.sleep(1.0)
    
    return np.array(final_pose, dtype=float)


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
        robot_pose = drive_to_point(ppi, waypoint, robot_pose, ekf, slam)
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
        scale = float(np.mean(np.loadtxt("calibration/param/scale.txt", delimiter=',')))
        baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))
        execute_motion_with_slam(ppi, slam, [1, 0], 1.0, scale, baseline)

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