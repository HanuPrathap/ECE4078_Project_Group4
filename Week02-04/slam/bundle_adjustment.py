# bundle_adjustment.py
import numpy as np
import cv2
from scipy.optimize import least_squares
import math

class BundleAdjustment:
    def __init__(self, ekf, min_markers=2, lambda_init=1e-3):
        """
        Bundle adjustment for SLAM optimization
        
        Args:
            ekf: Your existing EKF instance
            min_markers: Minimum markers required in an image for recording
            lambda_init: Initial lambda for Levenberg-Marquardt
        """
        self.ekf = ekf
        self.min_markers = min_markers
        self.lambda_val = lambda_init
        
        # Storage for bundle adjustment data
        self.recorded_images = []  # List of image data
        self.image_poses = []      # Initial poses from EKF [x, y, theta]
        self.observations = []     # Marker observations for each image
        self.unique_markers = set() # Set of all observed marker IDs
        
    def record_observation(self, measurements, current_pose):
        """
        Record an observation if it contains enough markers
        
        Args:
            measurements: List of marker measurements from aruco_detector
            current_pose: Current robot pose [x, y, theta] from EKF
        """
        if len(measurements) < self.min_markers:
            return False
            
        # Store the observation
        obs_data = {
            'pose': current_pose.copy(),
            'markers': {}
        }
        
        for lm in measurements:
            # Only consider markers with ID <= 10 (as per your aruco_detector)
            if lm.tag <= 10:
                obs_data['markers'][lm.tag] = {
                    'tvec': lm.position.flatten(),  # [z, -x] from your coordinate system
                    'covariance': lm.covariance
                }
                self.unique_markers.add(lm.tag)
        
        if len(obs_data['markers']) >= self.min_markers:
            self.observations.append(obs_data)
            return True
        return False
    
    def optimize_poses_and_markers(self, max_iterations=50, tolerance=1e-6):
        """
        Perform bundle adjustment optimization
        
        Returns:
            optimized_poses: Updated robot poses
            optimized_markers: Updated marker positions
            final_error: Final optimization error
        """
        if len(self.observations) < 2:
            print("Not enough observations for bundle adjustment")
            return None, None, None
            
        # Initialize parameters
        initial_params = self._pack_parameters()
        
        # Perform optimization using Levenberg-Marquardt
        result = least_squares(
            self._compute_residuals,
            initial_params,
            jac=self._compute_jacobian,
            method='lm',
            max_nfev=max_iterations,
            ftol=tolerance,
            xtol=tolerance
        )
        
        if result.success:
            optimized_params = result.x
            poses, markers = self._unpack_parameters(optimized_params)
            return poses, markers, result.cost
        else:
            print(f"Bundle adjustment failed: {result.message}")
            return None, None, None
    
    def _pack_parameters(self):
        """Pack poses and marker positions into a single parameter vector"""
        params = []
        
        # Add robot poses [x, y, theta] for each observation
        for obs in self.observations:
            params.extend(obs['pose'].flatten())
        
        # Add marker positions [x, y] for each unique marker
        unique_marker_list = sorted(list(self.unique_markers))
        for marker_id in unique_marker_list:
            if marker_id in self.ekf.taglist:
                # Get current marker position from EKF
                marker_idx = self.ekf.taglist.index(marker_id)
                marker_pos = self.ekf.markers[:, marker_idx]
                params.extend(marker_pos)
            else:
                # Initialize with zeros if marker not in EKF yet
                params.extend([0.0, 0.0])
        
        return np.array(params)
    
    def _unpack_parameters(self, params):
        """Unpack parameter vector into poses and marker positions"""
        n_observations = len(self.observations)
        n_markers = len(self.unique_markers)
        
        # Extract poses
        poses = params[:n_observations * 3].reshape(-1, 3)
        
        # Extract marker positions
        marker_params = params[n_observations * 3:].reshape(-1, 2)
        unique_marker_list = sorted(list(self.unique_markers))
        markers = {}
        for i, marker_id in enumerate(unique_marker_list):
            markers[marker_id] = marker_params[i]
            
        return poses, markers
    
    def _compute_residuals(self, params):
        """Compute residuals for all observations"""
        poses, markers = self._unpack_parameters(params)
        residuals = []
        
        for i, obs in enumerate(self.observations):
            robot_pose = poses[i]  # [x, y, theta]
            robot_x, robot_y, robot_theta = robot_pose
            
            # Rotation matrix for robot pose
            cos_th = np.cos(robot_theta)
            sin_th = np.sin(robot_theta)
            R = np.array([[cos_th, -sin_th],
                         [sin_th, cos_th]])
            robot_pos = np.array([robot_x, robot_y])
            
            for marker_id, marker_obs in obs['markers'].items():
                if marker_id in markers:
                    # Predicted observation (what we should see)
                    marker_world = markers[marker_id]
                    marker_relative = R.T @ (marker_world - robot_pos)
                    predicted_tvec = np.array([marker_relative[1], -marker_relative[0]])  # [z, -x]
                    
                    # Actual observation
                    observed_tvec = marker_obs['tvec']
                    
                    # Compute residual
                    residual = observed_tvec - predicted_tvec
                    residuals.extend(residual)
        
        return np.array(residuals)
    
    def _compute_jacobian(self, params):
        """Compute Jacobian matrix for optimization"""
        poses, markers = self._unpack_parameters(params)
        n_observations = len(self.observations)
        n_markers = len(self.unique_markers)
        unique_marker_list = sorted(list(self.unique_markers))
        
        # Count total residuals
        total_residuals = 0
        for obs in self.observations:
            total_residuals += len(obs['markers']) * 2
        
        # Initialize Jacobian
        n_params = n_observations * 3 + n_markers * 2
        jacobian = np.zeros((total_residuals, n_params))
        
        residual_idx = 0
        for i, obs in enumerate(self.observations):
            robot_pose = poses[i]
            robot_x, robot_y, robot_theta = robot_pose
            
            cos_th = np.cos(robot_theta)
            sin_th = np.sin(robot_theta)
            
            for marker_id, marker_obs in obs['markers'].items():
                if marker_id in markers:
                    marker_world = markers[marker_id]
                    marker_x, marker_y = marker_world
                    
                    # Compute derivatives w.r.t. robot pose
                    pose_param_idx = i * 3
                    
                    # Derivatives w.r.t. robot_x
                    jacobian[residual_idx, pose_param_idx] = -cos_th      # d(z)/d(robot_x)
                    jacobian[residual_idx + 1, pose_param_idx] = sin_th   # d(-x)/d(robot_x)
                    
                    # Derivatives w.r.t. robot_y
                    jacobian[residual_idx, pose_param_idx + 1] = -sin_th  # d(z)/d(robot_y)
                    jacobian[residual_idx + 1, pose_param_idx + 1] = -cos_th  # d(-x)/d(robot_y)
                    
                    # Derivatives w.r.t. robot_theta
                    dx = marker_x - robot_x
                    dy = marker_y - robot_y
                    jacobian[residual_idx, pose_param_idx + 2] = -sin_th * dx + cos_th * dy
                    jacobian[residual_idx + 1, pose_param_idx + 2] = -cos_th * dx - sin_th * dy
                    
                    # Derivatives w.r.t. marker position
                    marker_param_idx = n_observations * 3 + unique_marker_list.index(marker_id) * 2
                    
                    jacobian[residual_idx, marker_param_idx] = cos_th      # d(z)/d(marker_x)
                    jacobian[residual_idx + 1, marker_param_idx] = -sin_th # d(-x)/d(marker_x)
                    
                    jacobian[residual_idx, marker_param_idx + 1] = sin_th   # d(z)/d(marker_y)
                    jacobian[residual_idx + 1, marker_param_idx + 1] = cos_th # d(-x)/d(marker_y)
                    
                    residual_idx += 2
        
        return jacobian
    
    def update_ekf_with_optimization(self, optimized_poses, optimized_markers):
        """Update EKF state with optimized results"""
        if optimized_poses is None or optimized_markers is None:
            return False
        
        # Update robot pose with the most recent optimized pose
        if len(optimized_poses) > 0:
            latest_pose = optimized_poses[-1]
            self.ekf.robot.state = latest_pose.reshape(-1, 1)
        
        # Update marker positions
        for marker_id, marker_pos in optimized_markers.items():
            if marker_id in self.ekf.taglist:
                marker_idx = self.ekf.taglist.index(marker_id)
                self.ekf.markers[:, marker_idx] = marker_pos
        
        # Optionally reduce covariance after optimization
        self.ekf.P *= 0.8  # Reduce uncertainty after optimization
        
        return True
    
    def should_run_optimization(self):
        """Determine if we should run bundle adjustment"""
        # Run optimization if we have enough observations and some loop closure potential
        if len(self.observations) < 3:
            return False
            
        # Check if we have overlapping marker observations (potential loop closure)
        all_markers = set()
        overlap_count = 0
        for obs in self.observations:
            current_markers = set(obs['markers'].keys())
            if all_markers & current_markers:  # Intersection
                overlap_count += 1
            all_markers.update(current_markers)
        
        return overlap_count >= 2
    
    def clear_old_observations(self, max_observations=20):
        """Keep only recent observations to manage memory"""
        if len(self.observations) > max_observations:
            self.observations = self.observations[-max_observations:]


# Integration with existing EKF class
class EnhancedEKF:
    """Wrapper to integrate bundle adjustment with your existing EKF"""
    
    def __init__(self, robot):
        self.ekf = EKF(robot)  # Your existing EKF
        self.bundle_adjustment = BundleAdjustment(self.ekf)
        self.optimization_counter = 0
        self.optimization_interval = 10  # Run optimization every N observations
        
    def predict(self, raw_drive_meas):
        """Same as your EKF predict"""
        self.ekf.predict(raw_drive_meas)
    
    def update(self, measurements):
        """Enhanced update with bundle adjustment"""
        # Standard EKF update
        self.ekf.add_landmarks(measurements)
        self.ekf.update(measurements)
        
        # Record observation for bundle adjustment
        current_pose = self.ekf.robot.state.flatten()
        if self.bundle_adjustment.record_observation(measurements, current_pose):
            self.optimization_counter += 1
            
            # Periodically run bundle adjustment
            if (self.optimization_counter % self.optimization_interval == 0 and 
                self.bundle_adjustment.should_run_optimization()):
                
                print("Running bundle adjustment optimization...")
                poses, markers, error = self.bundle_adjustment.optimize_poses_and_markers()
                
                if poses is not None:
                    print(f"Bundle adjustment completed with error: {error}")
                    self.bundle_adjustment.update_ekf_with_optimization(poses, markers)
                    
                    # Clear old observations to manage memory
                    self.bundle_adjustment.clear_old_observations()
    
    def __getattr__(self, name):
        """Delegate other methods to the original EKF"""
        return getattr(self.ekf, name)


# Usage in your main SLAM loop:
"""
# Replace your EKF initialization with:
ekf = EnhancedEKF(robot)

# Your existing SLAM loop remains the same:
while operating:
    # Get drive measurements
    ekf.predict(drive_measurements)
    
    # Get marker detections
    measurements, img_marked = aruco_detector.detect_marker_positions(img)
    
    # Update - now includes bundle adjustment
    ekf.update(measurements)
    
    # Save map as usual
    ekf.save_map()
"""