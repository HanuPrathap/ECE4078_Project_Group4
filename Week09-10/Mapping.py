# import necessary functions
import numpy as np
import time
import json
import os
import math
import cv2
from operate import Operate
import argparse
from slam.ekf import EKF
from slam.robot import Robot




"""
Integration: The script imports your Operate class and works with your existing SLAM and YOLO systems
TargetPoseEst01.py: The script calls your target estimation script automatically between Bug 0 and Bug 1 phases
Navigation Control: The navigate_to_position() method uses a simple proportional controller - you may need to tune the gains based on your robot's dynamics
File Dependencies: Expects lab_output/targets.txt to exist (created by your target estimation script)
Safety: Includes position/heading tolerances and max speed limits

The script will systematically explore your arena, then intelligently revisit areas where object detections had low confidence, automatically improving the quality of your target map!

"""""




# bug 0 algorithm
# gets dimensions of map
# generate a list of waypoints as [East, NorthEast, North, NorthWest, West, Southwest, South]
# drive to these way points to get initial locations

# bug 1 algorithm
# waypoints are any aruco markers or fruits whose confidence level is below a parameter confidence threshold

# run bug 0 algorithm once
# main loop
# while true
# run targetposest.py to get confidence levels for each item
# run bug 1



#!/usr/bin/env python3
"""
Bug 0/Bug 1 Navigation Algorithm for SLAM-based exploration
Combines systematic grid exploration with adaptive revisiting of low-confidence detections
"""

import numpy as np
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Import from your existing codebase
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import util.measure as measure

# ===============================================================
# GLOBAL PARAMETERS — default configuration for your arena setup
# ===============================================================

# Arena dimensions (metres)
MAP_BOUNDS = (-1.2, 1.2, -1.2, 1.2)   # (x_min, x_max, y_min, y_max)
GRID_SPACING = 0.4                    # metres between Bug 0 waypoints (~6×6 grid)
CONFIDENCE_THRESHOLD = 0.6            # minimum confidence to stop revisiting
LAB_DIR = "lab_output"                # directory for SLAM and target outputs
SLAM_FILE = "slam.txt"                # SLAM map file name
TARGETS_FILE = "targets.txt"          # Target estimation output file
FRUIT_CONF_MIN = 0.0                  # min confidence to include fruit in GT map
ROUND_TO = 6                          # decimal rounding for GT map
# ===============================================================


# --------------------------
# Tunables for scaling
# --------------------------
CHI2_95 = 5.991  # 95% of chi-square with 2 dof

def _exp_safe(x):
    return float(np.exp(np.clip(x, -50, 50)))

# --------------------------
# SLAM reader: slam.txt
# --------------------------
def load_slam_confidence(
    slam_path: str,
    ref_radius_m: float = 0.10,   # r_ref in meters for A0 = pi*k*r_ref^2
    k: float = CHI2_95
) -> dict:
    """
    Parse the SLAM file with structure:
      {
        "taglist": [id1, id2, ...],
        "map": [[x1..xN], [y1..yN]],
        "covariance": [[...], ...]  # (2N x 2N), order [x1,y1,x2,y2,...]
      }
    Returns: {"aruco<ID>": confidence_float, ...}
    """
    with open(slam_path, "r") as f:
        data = json.load(f)

    taglist = data["taglist"]
    cov = np.array(data["covariance"], dtype=float)
    N = len(taglist)

    if cov.shape != (2*N, 2*N):
        raise ValueError(f"Covariance shape {cov.shape} does not match 2N x 2N for N={N}")

    # Reference area (95% ellipse of a circle with radius ref_radius_m)
    A0 = math.pi * k * (ref_radius_m ** 2)

    out = {}
    for i, tag in enumerate(taglist):
        # 2x2 block for this landmark: indices (2i,2i+1)
        i0, i1 = 2*i, 2*i + 1
        Sigma = cov[i0:i1+1, i0:i1+1]

        # Numerical safety
        det = float(np.linalg.det(Sigma))
        if det < 0:  # small negative due to rounding
            det = 0.0

        # 95% ellipse area
        A95 = math.pi * k * math.sqrt(det)  # == pi*k*sqrt(lambda1*lambda2)

        # Confidence in [0,1]
        conf = _exp_safe(- A95 / A0)
        conf = float(np.clip(conf, 0.0, 1.0))
        out[f"aruco{int(tag)}"] = conf

    return out

# --------------------------
# Targets reader: targets.txt
# --------------------------
def load_targets_confidence(
    targets_path: str,
    sigma0_m: float = 0.08   # spread scale for damping
) -> dict:
    """
    Parse the targets file with structure like:
      {
        "orange_0": {"x":..,"y":..,"uncertainty":..,"confidence":..,"n_detections":..},
        ...
      }
    Returns: {"orange_0": confidence_float, ...}
    """
    with open(targets_path, "r") as f:
        data = json.load(f)

    out = {}
    for k, v in data.items():
        base = float(v.get("confidence", 0.0))
        unc  = float(v.get("uncertainty", 0.0))
        # Penalize spatial spread softly
        penal = _exp_safe(- (unc / max(1e-9, sigma0_m)) ** 2)
        conf = base * penal
        out[k] = float(np.clip(conf, 0.0, 1.0))
    return out

# --------------------------
# Merge helper
# --------------------------
def load_confidence_merged(
    lab_output_dir: str,
    slam_filename: str = "slam.txt",
    targets_filename: str = "targets.txt",
    ref_radius_m: float = 0.10,
    sigma0_m: float = 0.08
) -> dict:
    """
    Reads lab_output/<slam_filename> and lab_output/<targets_filename>
    and returns one dict mapping names -> confidence.
    """
    slam_path = os.path.join(lab_output_dir, slam_filename)
    tgt_path  = os.path.join(lab_output_dir, targets_filename)

    slam_conf = load_slam_confidence(slam_path, ref_radius_m=ref_radius_m)
    tgt_conf  = load_targets_confidence(tgt_path, sigma0_m=sigma0_m)

    merged = {**slam_conf, **tgt_conf}
    return merged



def create_gt_map(
    slam_path: str,
    targets_path: str,
    out_path: str = None,
    fruit_conf_min: float = 0.0,   # keep fruits with confidence >= this
    round_to: int | None = None,   # e.g., 6 to round to 6 decimals; None = no rounding
) -> dict:
    """
    Build a unified ground-truth-style map from:
      - SLAM file (with keys: taglist, map: [[x...],[y...]])
      - Targets file (with keys per fruit, each having x,y and confidence)

    Output dict structure:
      {
        "aruco<ID>_0": {"x": <float>, "y": <float>},
        "<fruit_key>": {"x": <float>, "y": <float>},
        ...
      }

    Writes to `gt_map_generated.txt` beside the inputs (unless out_path is provided).
    Returns the dict as well.
    """
    # ---------- read inputs ----------
    with open(slam_path, "r") as f:
        slam = json.load(f)
    with open(targets_path, "r") as f:
        targets = json.load(f)

    taglist = slam["taglist"]
    xs = slam["map"][0]  # x for each aruco in taglist order
    ys = slam["map"][1]  # y for each aruco in taglist order

    # ---------- assemble output ----------
    gt = {}

    # ArUco: "aruco<ID>_0" -> {x, y}
    for i, tag in enumerate(taglist):
        x_val = xs[i]
        y_val = ys[i]
        if round_to is not None:
            x_val = round(x_val, round_to)
            y_val = round(y_val, round_to)
        gt[f"aruco{int(tag)}_0"] = {"x": x_val, "y": y_val}

    # Fruits: keep as-is from targets (optionally filter by confidence)
    for k, v in targets.items():
        # Expecting v to have x,y and confidence (your robust merger format)
        if "x" not in v or "y" not in v:
            continue
        if v.get("confidence", 1.0) < fruit_conf_min:
            continue
        x_val = float(v["x"])
        y_val = float(v["y"])
        if round_to is not None:
            x_val = round(x_val, round_to)
            y_val = round(y_val, round_to)
        gt[k] = {"x": x_val, "y": y_val}

    # ---------- write file ----------
    if out_path is None:
        # default: write next to slam_path in the same folder
        out_dir = os.path.dirname(os.path.abspath(slam_path))
        out_path = os.path.join(out_dir, "gt_map_generated.txt")

    with open(out_path, "w") as fo:
        json.dump(gt, fo, indent=None, separators=(",", ": "))

    print(f"✓ Wrote ground truth map → {out_path}")
    return gt


# Convenience wrapper that takes a lab_output directory
def create_gt_map_from_lab(
    lab_output_dir: str,
    slam_filename: str = "slam.txt",
    targets_filename: str = "targets.txt",
    out_filename: str = "gt_map_generated.txt",
    fruit_conf_min: float = 0.0,
    round_to: int | None = None,
) -> dict:
    slam_path = os.path.join(lab_output_dir, slam_filename)
    targets_path = os.path.join(lab_output_dir, targets_filename)
    out_path = os.path.join(lab_output_dir, out_filename)
    return create_gt_map(
        slam_path, targets_path, out_path=out_path,
        fruit_conf_min=fruit_conf_min, round_to=round_to
    )


class BugNavigator:
    def __init__(self, map_bounds=(-1.5, 1.5, -1.5, 1.5), grid_spacing=0.5, 
                 confidence_threshold=0.6, min_detections=2):
        """
        Initialize Bug Navigation system
        
        Args:
            map_bounds: (x_min, x_max, y_min, y_max) in meters
            grid_spacing: distance between waypoints in meters
            confidence_threshold: minimum confidence to skip revisiting
            min_detections: minimum detections required for high confidence
        """
        self.map_bounds = map_bounds
        self.grid_spacing = grid_spacing
        self.confidence_threshold = confidence_threshold
        self.min_detections = min_detections
        
        # Track visited locations and low-confidence targets
        self.visited_waypoints = []
        self.low_confidence_targets = []
        
        # Current state
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        
    def generate_bug0_waypoints(self):
        """
        Generate systematic grid waypoints for Bug 0 exploration
        Pattern: East, NorthEast, North, NorthWest, West, Southwest, South, SouthEast, origin
        """
        x_min, x_max, y_min, y_max = self.map_bounds
        
        # Generate grid points
        x_coords = np.arange(x_min, x_max + self.grid_spacing, self.grid_spacing)
        y_coords = np.arange(y_min, y_max + self.grid_spacing, self.grid_spacing)
        
        waypoints = []
        
        # Spiral pattern from center outward
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Start from center, move in expanding squares
        for x in x_coords:
            for y in y_coords:
                waypoints.append((x, y))
        
        # Sort by distance from origin (closer first for initial exploration)
        waypoints.sort(key=lambda p: np.sqrt(p[0]**2 + p[1]**2))
        
        if (0.0, 0.0) in waypoints: # add origin as last waypoint
            waypoints.remove((0.0, 0.0))
        waypoints.append((0.0, 0.0))

        self.waypoints = waypoints
        print(f"Generated {len(waypoints)} Bug 0 waypoints")
        return waypoints
    
    def generate_bug1_waypoints(self, target_estimates_file='lab_output/targets.txt'):
        """
        Generate waypoints for low-confidence targets (Bug 1)
        
        Args:
            target_estimates_file: path to targets.txt with confidence data
        """
        if not os.path.exists(target_estimates_file):
            print(f"Warning: {target_estimates_file} not found, skipping Bug 1")
            return []
        
        with open(target_estimates_file, 'r') as f:
            targets = json.load(f)
        
        low_conf_waypoints = []
        
        for target_name, data in targets.items():
            confidence = data.get('confidence', 1.0)
            n_detections = data.get('n_detections', 1)
            uncertainty = data.get('uncertainty', 0.0)
            
            # Add to revisit list if:
            # 1. Low confidence, OR
            # 2. Few detections, OR
            # 3. High uncertainty
            if (confidence < self.confidence_threshold or 
                n_detections < self.min_detections or
                uncertainty > 0.15):
                
                x, y = data['x'], data['y']
                low_conf_waypoints.append({
                    'name': target_name,
                    'position': (x, y),
                    'confidence': confidence,
                    'n_detections': n_detections,
                    'reason': self._get_revisit_reason(confidence, n_detections, uncertainty)
                })
                
                print(f"  Low confidence target: {target_name}")
                print(f"    Position: ({x:.3f}, {y:.3f})")
                print(f"    Reason: {low_conf_waypoints[-1]['reason']}")
        
        self.low_confidence_targets = low_conf_waypoints
        return low_conf_waypoints
    
    def _get_revisit_reason(self, confidence, n_detections, uncertainty):
        """Helper to explain why target needs revisiting"""
        reasons = []
        if confidence < self.confidence_threshold:
            reasons.append(f"low confidence ({confidence:.2f})")
        if n_detections < self.min_detections:
            reasons.append(f"few detections ({n_detections})")
        if uncertainty > 0.15:
            reasons.append(f"high uncertainty ({uncertainty:.3f}m)")
        return ", ".join(reasons)
    
    def plan_approach_waypoints(self, target_pos, approach_distance=0.3, num_angles=4):
        """
        Generate multiple approach waypoints around a target for better coverage
        
        Args:
            target_pos: (x, y) target position
            approach_distance: distance from target to take photos (meters)
            num_angles: number of approach angles
        """
        tx, ty = target_pos
        waypoints = []
        
        for i in range(num_angles):
            angle = 2 * np.pi * i / num_angles
            wx = tx + approach_distance * np.cos(angle)
            wy = ty + approach_distance * np.sin(angle)
            waypoints.append((wx, wy, angle + np.pi))  # Face towards target
        
        return waypoints
    
    def compute_path_cost(self, start, goal):
        """Simple Euclidean distance cost"""
        return np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
    
    def order_waypoints_greedy(self, waypoints, start_pos=(0, 0)):
        """
        Order waypoints using greedy nearest-neighbor heuristic
        """
        if not waypoints:
            return []
        
        ordered = []
        remaining = waypoints.copy()
        current = start_pos
        
        while remaining:
            # Find nearest waypoint
            distances = [self.compute_path_cost(current, wp[:2]) for wp in remaining]
            nearest_idx = np.argmin(distances)
            
            ordered.append(remaining[nearest_idx])
            current = remaining[nearest_idx][:2]
            remaining.pop(nearest_idx)
        
        return ordered
    
    def run_target_estimation(self):
        """
        Run target pose estimation script
        """
        print("\n" + "="*60)
        print("Running target pose estimation...")
        print("="*60)
        
        try:
            result = subprocess.run(
                ['python3', 'TargetPoseEst01.py'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("Target estimation completed successfully")
                return True
            else:
                print(f"Target estimation failed with error:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Target estimation timed out")
            return False
        except Exception as e:
            print(f"Error running target estimation: {e}")
            return False
    
    def execute_navigation(self, operate_instance):
        """
        Main navigation loop – Bug 0 runs once, then Bug 1 repeats until all confidences are satisfactory.
        """
        print("\n" + "="*60)
        print("STARTING BUG ALGORITHM NAVIGATION")
        print("="*60)

        # --------------------------
        # 1. Bug 0 — run once
        # --------------------------
        print("\nPhase 1: BUG 0 — Systematic Exploration")
        print("-"*60)

        bug0_waypoints = self.generate_bug0_waypoints()

        for i, (x, y) in enumerate(bug0_waypoints):
            print(f"\nWaypoint {i+1}/{len(bug0_waypoints)}: ({x:.2f}, {y:.2f})")
            self.navigate_to_position(operate_instance, x, y)
            operate_instance.take_pic()
            operate_instance.command['save_image'] = True
            operate_instance.save_image()
            self.visited_waypoints.append((x, y))
            time.sleep(0.5)

        print("\nBug 0 complete — initial exploration done!")

        # --------------------------
        # 2. Main adaptive loop
        # --------------------------
        iteration = 0
        max_iterations = 5  # prevent infinite loops

        while iteration < max_iterations:
            iteration += 1
            print("\n" + "="*60)
            print(f"BUG 1 CYCLE {iteration}")
            print("="*60)

            # Run target pose estimation (creates lab_output/targets.txt)
            if not self.run_target_estimation():
                print("Target estimation failed — skipping this iteration")
                continue

            # Load updated target confidences
            try:
                targets_conf = load_targets_confidence('lab_output/targets.txt')
            except Exception as e:
                print(f"Error loading targets.txt: {e}")
                break

            # Identify low-confidence targets
            low_conf_targets = [
                k for k, v in targets_conf.items()
                if v < self.confidence_threshold
            ]

            if not low_conf_targets:
                print("\nAll target confidences satisfactory!")
                break

            print(f"\n{len(low_conf_targets)} targets below confidence threshold:")
            for name in low_conf_targets:
                print(f"  - {name} ({targets_conf[name]:.2f})")

            # Run Bug 1 on low-confidence targets
            bug1_waypoints = self.generate_bug1_waypoints()
            if not bug1_waypoints:
                print("No Bug 1 waypoints found, exiting loop.")
                break

            for target_data in bug1_waypoints:
                target_pos = target_data['position']
                target_name = target_data['name']
                print(f"\nRevisiting target: {target_name}")
                approach_points = self.plan_approach_waypoints(target_pos)

                for j, (x, y, heading) in enumerate(approach_points):
                    print(f"  Approach {j+1}: ({x:.2f}, {y:.2f})")
                    self.navigate_to_position(operate_instance, x, y, heading)
                    operate_instance.take_pic()
                    operate_instance.command['save_image'] = True
                    operate_instance.save_image()
                    time.sleep(0.5)

        print("\nAll targets sufficiently confident or max iterations reached.")
        print("="*60)
        print("NAVIGATION COMPLETE")
        print("="*60)

        operate_instance.ekf.save_map("slam_map_final.txt")
        print("Final SLAM map saved to slam_map_final.txt")


    
    def navigate_to_position(self, operate_instance, x_goal, y_goal, heading_goal=None):
        """
        Navigate robot to goal position using simple proportional controller
        
        Args:
            operate_instance: Operate class instance
            x_goal, y_goal: target position in meters
            heading_goal: optional target heading in radians
        """
        position_tolerance = 0.1  # 10cm
        heading_tolerance = 0.1  # ~5 degrees
        max_speed = 0.3  # m/s
        
        # Get current pose from EKF
        current_pose = operate_instance.ekf.robot.state
        
        while True:
            # Update current position
            current_pose = operate_instance.ekf.robot.state
            x_curr, y_curr, theta_curr = current_pose[0, 0], current_pose[1, 0], current_pose[2, 0]
            
            # Compute error
            dx = x_goal - x_curr
            dy = y_goal - y_curr
            distance = np.sqrt(dx**2 + dy**2)
            
            # Check if reached
            if distance < position_tolerance:
                if heading_goal is None:
                    operate_instance.command['motion'] = [0, 0]
                    break
                else:
                    # Adjust heading
                    heading_error = self._normalize_angle(heading_goal - theta_curr)
                    if abs(heading_error) < heading_tolerance:
                        operate_instance.command['motion'] = [0, 0]
                        break
            
            # Compute desired heading
            desired_theta = np.arctan2(dy, dx)
            heading_error = self._normalize_angle(desired_theta - theta_curr)
            
            # Proportional controller
            linear_speed = min(max_speed, 2.0 * distance)
            angular_speed = 3.0 * heading_error
            
            # Convert to wheel commands (simplified)
            # This assumes your operate.py accepts [forward, turn] commands
            operate_instance.command['motion'] = [linear_speed, angular_speed]
            
            # Update SLAM
            drive_meas = operate_instance.control()
            operate_instance.take_pic()
            operate_instance.update_slam(drive_meas)
            
            time.sleep(0.05)  # 20Hz control loop
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main():
    """
    Main entry point — uses global parameters for consistent setup.
    """
    import pprint

    # Initialize Operate instance
    from operate import Operate
    operate = Operate({
        "ip": "192.168.50.1",
        "port": 8080,
        "calib_dir": "calibration/param/",
        "yolo_model": "YOLO/model/best.pt"
    })

    # Initialize Bug Navigator using global config
    navigator = BugNavigator(
        map_bounds=MAP_BOUNDS,
        grid_spacing=GRID_SPACING,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # Start SLAM
    print("Starting SLAM...")
    operate.ekf_on = True
    operate.notification = "SLAM is running - Bug Algorithm Active"

    # Execute Bug 0 + Bug 1 navigation loop
    try:
        navigator.execute_navigation(operate)
    except KeyboardInterrupt:
        print("\nNavigation interrupted by user")
    finally:
        operate.command['motion'] = [0, 0]
        operate.control()
        operate.ekf.save_map("slam_map_final.txt")
        print("Final map saved")

        # === Create ground truth map and save ===
        script_dir = os.getcwd()
        slam_path = os.path.join(LAB_DIR, SLAM_FILE)
        targets_path = os.path.join(LAB_DIR, TARGETS_FILE)
        out_path = os.path.join(script_dir, "gt_map_generated.txt")

        gt_map = create_gt_map(
            slam_path,
            targets_path,
            out_path=out_path,
            fruit_conf_min=FRUIT_CONF_MIN,
            round_to=ROUND_TO,
        )

        print("\n=== Ground Truth Map Generated ===")
        pprint.pprint(gt_map)




if __name__ == "__main__":
    main()


# TO DO
# implement follow-wall algorithm
