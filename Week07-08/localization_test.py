# localization_test.py - Simple robot localization testing
# Drive to hardcoded waypoints and test EKF localization accuracy

import sys, os, time, json
import numpy as np
import argparse

# --- Robot I/O and helpers ---
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure
from Helper import get_distance_robot_to_goal, get_angle_robot_to_goal

# --- SLAM core (camera, kinematics, EKF) ---
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# ---------------------------
# Enhanced Localization with Logging
# ---------------------------
class TestableLocalizationEKF:
    """EKF with detailed logging for testing"""
    def __init__(self, robot, known_landmarks):
        self.robot = robot
        self.known_landmarks = known_landmarks
        self.P = np.eye(3) * 0.01
        self.update_count = 0

    # gets the state vector from the robot class
    def get_state_vector(self):
        return self.robot.state.copy()

    # predicts based on odemtry and co variances takes in drive meas 
    def predict(self, drive_meas):
        print(f"[EKF] Prediction step - Drive: left={drive_meas.left_speed:.3f}, right={drive_meas.right_speed:.3f}, dt={drive_meas.dt:.3f}")
        F = self.robot.derivative_drive(drive_meas)
        self.robot.drive(drive_meas)
        Q = self.robot.covariance_drive(drive_meas)
        self.P = F @ self.P @ F.T + Q
        
        s = self.robot.state
        print(f"[EKF] After prediction: x={s[0,0]:.4f}, y={s[1,0]:.4f}, theta={s[2,0]:.4f}")

    def update(self, measurements):
        if not measurements:
            print("[EKF] No measurements received")
            return
            
        valid = [m for m in measurements if m.tag in self.known_landmarks]
        if not valid:
            print(f"[EKF] No valid measurements (received {len(measurements)} total)")
            return

        self.update_count += 1
        print(f"\n[EKF UPDATE #{self.update_count}] Using {len(valid)} ArUco markers:")
        
        for m in valid:
            landmark_pos = self.known_landmarks[m.tag]

            # --- robust: flatten/reshape measurement to 1D float array of length 2
            measured_pos = np.asarray(m.position, dtype=float).reshape(-1)
            if measured_pos.shape[0] != 2:
                raise ValueError(f"Unexpected measurement shape for tag {m.tag}: {measured_pos.shape}")

            print(f"  ArUco {m.tag}: Expected at ({landmark_pos[0]:.3f}, {landmark_pos[1]:.3f})")
            print(f"             Measured at ({measured_pos[0]:.3f}, {measured_pos[1]:.3f}) relative to robot")

            # Calculate expected distance for validation
            robot_pos = self.robot.state[0:2, 0]
            expected_global = np.array(landmark_pos).reshape(2, 1)
            dist_to_landmark = np.linalg.norm(expected_global.squeeze() - robot_pos)
            measured_dist = np.linalg.norm(measured_pos)
            print(f"             Expected distance: {dist_to_landmark:.3f}m, Measured: {measured_dist:.3f}m")


        # Store pose before update
        s_before = self.robot.state.copy()
        
        # Stack z and R
        z = np.concatenate([m.position.reshape(-1, 1) for m in valid], axis=0)
        R = np.zeros((2*len(valid), 2*len(valid)))
        for i, m in enumerate(valid):
            R[2*i:2*i+2, 2*i:2*i+2] = m.covariance

        zhat_list, H_rows = [], []
        xy = self.robot.state[0:2, :]
        th = self.robot.state[2, 0]
        c, s = np.cos(th), np.sin(th)
        Rot = np.array([[c, -s], [s, c]])
        DRot = np.array([[-s, -c], [c, -s]])

        for m in valid:
            lm = np.array(self.known_landmarks[m.tag]).reshape(2, 1)
            zhat = Rot.T @ (lm - xy)
            zhat_list.append(zhat)

            H = np.zeros((2, 3))
            H[:, 0:2] = -Rot.T
            H[:, 2:3] = DRot.T @ (lm - xy)
            H_rows.append(H)

        zhat = np.vstack(zhat_list)
        H = np.vstack(H_rows)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innov = z - zhat
        
        print(f"[EKF] Innovation magnitude: {np.linalg.norm(innov):.4f}")
        
        self.robot.state = self.robot.state + (K @ innov).reshape(3, 1)
        self.P = (np.eye(3) - K @ H) @ self.P
        
        # Show the correction
        s_after = self.robot.state
        dx = s_after[0,0] - s_before[0,0]
        dy = s_after[1,0] - s_before[1,0]
        dtheta = s_after[2,0] - s_before[2,0]
        
        print(f"[EKF] POSE CORRECTION: dx={dx:.4f}m, dy={dy:.4f}m, dtheta={dtheta:.4f}rad")
        print(f"[EKF] UPDATED POSE: x={s_after[0,0]:.4f}, y={s_after[1,0]:.4f}, theta={s_after[2,0]:.4f}")
        print(f"[EKF] Covariance trace: {np.trace(self.P):.6f}")


class TestableArucoLocalization:
    def __init__(self, robot, known_landmarks, marker_length=0.07):
        self.ekf = TestableLocalizationEKF(robot, known_landmarks)
        self.det = aruco.aruco_detector(robot, marker_length=marker_length)
        self.on = True
        self.step_count = 0

    def step(self, rgb, drive_meas, pause_on_update=2.0):
        self.step_count += 1
        print(f"\n[LOCALIZATION STEP {self.step_count}]")
        
        if self.on and drive_meas is not None:
            self.ekf.predict(drive_meas)
            
        meas, img = self.det.detect_marker_positions(rgb)
        
        if meas:
            print(f"[ArUco Detection] Found {len(meas)} markers: {[m.tag for m in meas]}")
            if self.on:
                # --- BEFORE update: snapshot & print pose, then pause ---
                s_before = self.ekf.get_state_vector()
                print(f"[LOCALISATION] Marker(s) visible — pausing {pause_on_update:.1f}s and printing pose BEFORE update:")
                print(f"    BEFORE: x={s_before[0,0]:.4f}, y={s_before[1,0]:.4f}, theta={s_before[2,0]:.4f} rad")
                time.sleep(pause_on_update)

                # --- UPDATE using the ArUco measurements ---
                self.ekf.update(meas)

                # --- AFTER update: print pose and delta for easy comparison ---
                s_after = self.ekf.get_state_vector()
                dx = float(s_after[0,0] - s_before[0,0])
                dy = float(s_after[1,0] - s_before[1,0])
                dth = float(s_after[2,0] - s_before[2,0])
                print("[LOCALISATION] AFTER update:")
                print(f"    AFTER : x={s_after[0,0]:.4f}, y={s_after[1,0]:.4f}, theta={s_after[2,0]:.4f} rad")
                print(f"    CHANGE: dx={dx:.4f} m, dy={dy:.4f} m, dtheta={dth:.4f} rad")
        else:
            print("[ArUco Detection] No markers detected")

            
        s = self.ekf.get_state_vector()
        current_pose = (float(s[0,0]), float(s[1,0]), float(s[2,0]))
        print(f"[CURRENT ROBOT POSE] x={current_pose[0]:.4f}m, y={current_pose[1]:.4f}m, theta={current_pose[2]:.4f}rad")
        
        return current_pose, img

    def get_pose(self):
        s = self.ekf.get_state_vector()
        return float(s[0,0]), float(s[1,0]), float(s[2,0])


# ---------------------------
# Map loading (from your code)
# ---------------------------
def read_true_map(fname: str):
    """Load map and extract fruit positions as waypoints"""
    with open(fname, 'r') as fd:
        gt = json.load(fd)

    fruit_list, fruit_pos = [], []
    aruco_pos = np.empty((10, 2), dtype=np.float64)

    for key, v in gt.items():
        x = float(np.round(v['x'], 3))
        y = float(np.round(v['y'], 3))
        if key.startswith('aruco'):
            if key.startswith('aruco10'):
                aruco_pos[9, 0] = x; aruco_pos[9, 1] = y
            else:
                idx = int(key[5]) - 1
                aruco_pos[idx, 0] = x; aruco_pos[idx, 1] = y
        else:
            fruit_list.append(key[:-2])
            fruit_pos.append([x, y])

    return fruit_list, np.array(fruit_pos, dtype=np.float64), aruco_pos


# ---------------------------
# Simple Turn-and-Go Navigation
# ---------------------------
def simple_drive_to_point(ppi, localizer, waypoint, current_pose, 
                         pause_duration=3.0, ip="192.168.50.1", 
                         scale=None, baseline=None):
    """Simple turn-then-drive navigation with localization"""
    
    if scale is None or baseline is None:
        try:
            scale = float(np.mean(np.loadtxt("calibration/param/scale.txt", delimiter=',')))
            baseline = float(np.squeeze(np.loadtxt("calibration/param/baseline.txt", delimiter=',')))
        except Exception as e:
            raise RuntimeError(f"Calibration params not loaded: {e}")

    current = np.array(current_pose, dtype=float)
    wp = np.array(waypoint, dtype=float)
    
    print(f"\n{'='*60}")
    print(f"[NAVIGATION] Driving from ({current[0]:.3f}, {current[1]:.3f}) to waypoint ({wp[0]:.3f}, {wp[1]:.3f})")
    print(f"{'='*60}")

    # Calculate required turn and distance
    dist = float(np.squeeze(get_distance_robot_to_goal(current, wp)))
    head = float(np.squeeze(get_angle_robot_to_goal(current, wp)))
    head = (head + np.pi) % (2*np.pi) - np.pi  # Normalize to [-pi, pi]

    print(f"[NAVIGATION] Required distance: {dist:.3f}m, heading change: {head:.3f}rad ({np.degrees(head):.1f}°)")

    wheel = 40  # PWM ticks
    drive_t = dist / (wheel * scale + 1e-9)
    ang_rate = (2.0 * wheel * scale) / (baseline + 1e-9)
    turn_t = abs(head) / (ang_rate + 1e-9)
    turn_dir = 1 if head >= 0 else -1

    print(f"[NAVIGATION] Turn time: {turn_t:.2f}s, Drive time: {drive_t:.2f}s")

    # PHASE 1: Turn in place
    if turn_t > 0.05:
        print(f"\n[TURN PHASE] Turning {'RIGHT' if turn_dir > 0 else 'LEFT'} for {turn_t:.2f}s")
        start_time = time.time()
        last_time = start_time
        
        while (time.time() - start_time) < turn_t:
            lv, rv = ppi.set_velocity([0, turn_dir], tick=40, turning_tick=40)
            now = time.time()
            dt = max(now - last_time, 1e-3)
            last_time = now
            
            drive_meas = measure.Drive(lv, -rv if ip != "localhost" else rv, dt)
            img = ppi.get_image()
            
            try:
                pose, _ = localizer.step(img, drive_meas, pause_on_update=pause_duration)
            except Exception as e:
                print(f"[Localization Error] {e}")
                
            time.sleep(0.1)
        
        ppi.set_velocity([0, 0])
        time.sleep(0.3)
        print("[TURN PHASE] Turn completed")

    # PHASE 2: Drive forward
    if drive_t > 0.05:
        print(f"\n[DRIVE PHASE] Driving forward for {drive_t:.2f}s")
        start_time = time.time()
        last_time = start_time
        
        while (time.time() - start_time) < drive_t:
            lv, rv = ppi.set_velocity([1, 0], tick=40, turning_tick=40)
            now = time.time()
            dt = max(now - last_time, 1e-3)
            last_time = now
            
            drive_meas = measure.Drive(lv, -rv if ip != "localhost" else rv, dt)
            img = ppi.get_image()
            
            try:
                pose, _ = localizer.step(img, drive_meas, pause_on_update=pause_duration)
            except Exception as e:
                print(f"[Localization Error] {e}")
                
            time.sleep(0.1)
        
        ppi.set_velocity([0, 0])
        time.sleep(0.5)
        print("[DRIVE PHASE] Drive completed")

    # PHASE 3: Final stationary localization
    print(f"\n[FINAL LOCALIZATION] Taking final pose measurement...")
    img = ppi.get_image()
    drive_meas = measure.Drive(0, 0, 0.001)
    final_pose, _ = localizer.step(img, drive_meas, pause_on_update=pause_duration)
    
    print(f"[WAYPOINT REACHED] Final pose: ({final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f})")
    print(f"[WAYPOINT REACHED] Target was:  ({waypoint[0]:.4f}, {waypoint[1]:.4f})")
    
    error_dist = np.sqrt((final_pose[0] - waypoint[0])**2 + (final_pose[1] - waypoint[1])**2)
    print(f"[WAYPOINT REACHED] Position error: {error_dist:.4f}m")
    
    return final_pose


# ---------------------------
# Main Test Script
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simple Localization Test")
    parser.add_argument("--ip", type=str, default="192.168.50.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--map", type=str, default="M3_prac_map_full.txt")
    parser.add_argument("--pause_duration", type=float, default=3.0, 
                       help="How long to pause when ArUco markers are detected (s)")
    parser.add_argument("--max_waypoints", type=int, default=3,
                       help="Maximum number of fruit waypoints to visit")
    
    args = parser.parse_args()

    print("="*80)
    print("ROBOT LOCALIZATION TEST - Simple Turn-and-Go Navigation")
    print("="*80)

    # Load map and extract fruit positions as waypoints
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map(args.map)
    
    print(f"\nLoaded map with {len(fruit_list)} fruits and {len(aruco_true_pos)} ArUco markers")
    print("\nFruit positions (will be used as waypoints):")
    for i, (name, pos) in enumerate(zip(fruit_list, fruit_true_pos)):
        if i < args.max_waypoints:
            print(f"  {i+1}. {name}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    print("\nArUco marker positions (for localization):")
    for i, pos in enumerate(aruco_true_pos):
        print(f"  ArUco {i+1}: ({pos[0]:.3f}, {pos[1]:.3f})")

    # Connect to robot
    print(f"\nConnecting to robot at {args.ip}:{args.port}...")
    ppi = PenguinPi(args.ip, args.port)

    # Load calibration
    print("Loading calibration parameters...")
    K = np.loadtxt(f"{args.calib_dir}intrinsic.txt", delimiter=',')
    D = np.loadtxt(f"{args.calib_dir}distCoeffs.txt", delimiter=',')
    scale_arr = np.loadtxt(f"{args.calib_dir}scale.txt", delimiter=',')
    baseline = float(np.squeeze(np.loadtxt(f"{args.calib_dir}baseline.txt", delimiter=',')))
    scale = float(np.mean(scale_arr))
    
    robot = Robot(baseline, scale, K, D)
    print(f"Calibration: scale={scale:.6f}, baseline={baseline:.6f}")

    # Setup known landmarks for localization
    known_landmarks = {}
    for i in range(len(aruco_true_pos)):
        tag = i + 1
        idx = 9 if tag == 10 else i
        known_landmarks[tag] = [float(aruco_true_pos[idx, 0]), float(aruco_true_pos[idx, 1])]

    print(f"Known landmarks: {list(known_landmarks.keys())}")

    # Initialize localization
    localizer = TestableArucoLocalization(robot, known_landmarks)

    # Get initial pose
    print(f"\n" + "="*60)
    print("GETTING INITIAL ROBOT POSE")
    print("="*60)
    img = ppi.get_image()
    initial_pose, _ = localizer.step(img, None, pause_on_update=args.pause_duration)
    
    print(f"[INITIAL POSE] x={initial_pose[0]:.4f}m, y={initial_pose[1]:.4f}m, theta={initial_pose[2]:.4f}rad")

    # Navigate to each fruit waypoint
    current_pose = initial_pose
    waypoints = fruit_true_pos[:args.max_waypoints]  # Limit number of waypoints
    
    try:
        for i, waypoint in enumerate(waypoints):
            print(f"\n" + "="*60)
            print(f"DRIVING TO WAYPOINT {i+1}/{len(waypoints)}: {fruit_list[i]}")
            print("="*60)
            
            current_pose = simple_drive_to_point(
                ppi, localizer, waypoint, current_pose,
                pause_duration=args.pause_duration,
                ip=args.ip, scale=scale, baseline=baseline
            )
            
            print(f"\n[WAYPOINT {i+1} COMPLETED] Pausing 2s before next waypoint...")
            time.sleep(2.0)

        print(f"\n" + "="*80)
        print("ALL WAYPOINTS COMPLETED - LOCALIZATION TEST FINISHED")
        print("="*80)
        print(f"Final robot pose: ({current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f})")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test stopped by user")
        
    finally:
        print("\n[CLEANUP] Stopping robot...")
        try:
            ppi.set_velocity([0, 0])
        except Exception:
            pass
        print("Test completed.")

# Run with: python localization_test.py --ip 192.168.50.1 --pause_duration 3.0 --max_waypoints 3