# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from YOLO.detector import Detector
from scipy import stats
import time


# list of target fruits and vegs types 
TARGET_TYPES = ['orange', 'lemon', 'pear', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']

# Removed GUI class - now using terminal output for live updates


def normalise_label(label):
    """Strip color prefix from labels"""
    base_labels = ['orange', 'lemon', 'pear', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']
    for base in base_labels:
        if label.startswith(base):
            return base
    return label


def compute_detection_confidence(bbox_area, image_area, distance, optimal_distance=0.4):
    """
    Compute confidence score for a detection based on:
    - Bounding box size (larger = more reliable)
    - Distance (closer to optimal = better)
    
    Returns confidence score between 0 and 1
    """
    # Normalize bbox size (prefer 5-30% of image area)
    size_ratio = bbox_area / image_area
    if size_ratio < 0.05:  # Too small
        size_score = size_ratio / 0.05
    elif size_ratio > 0.30:  # Too large (too close)
        size_score = max(0, 1 - (size_ratio - 0.30) / 0.20)
    else:  # Optimal range
        size_score = 1.0
    
    # Distance score (optimal around 40cm)
    distance_score = np.exp(-((distance - optimal_distance) ** 2) / (2 * 0.15 ** 2))
    
    # Combined confidence
    confidence = (size_score * 0.6 + distance_score * 0.4)
    return confidence


def estimate_pose(camera_matrix, obj_info, robot_pose, image_shape=(240, 320)):
    """
    Estimate the pose of a target based on size and location of its bounding box
    and the corresponding robot pose.
    
    Returns: (target_pose dict, confidence score)
    """
    focal_length = camera_matrix[0][0]
    
    target_dimensions_dict = {'orange': [0.0756, 0.0767, 0.0729], 'lemon': [0.054,0.074, 0.0536], 
                              'pear': [0.0704, 0.07565, 0.10425], 'tomato': [0.0678, 0.07, 0.0617], 
                              'capsicum': [0.07487, 0.07447, 0.0931], 'potato': [0.0677, 0.094, 0.0566], 
                              'pumpkin_orange': [0.08475, 0.08225, 0.07815], 'garlic': [0.0578, 0.0645, 0.0747],
                              'apple': [0.0678, 0.07, 0.0617], 'capsicum_green': [0.07487, 0.07447, 0.0931],
                              'capsicum_yellow': [0.07487, 0.07447, 0.0931], 'pear_yellow' : [0.0704, 0.07565, 0.10425],
                              'pear_green' : [0.0704, 0.07565, 0.10425], 'capsicum_red': [0.07487, 0.07447, 0.0931],
                              'pumpkin_green': [0.08475, 0.08225, 0.09515]}
    
    target_class = obj_info[0]
    target_box = obj_info[1]  # [x, y, width, height]
    true_height = target_dimensions_dict[target_class][2]
    
    # Compute pose
    pixel_height = target_box[3]
    pixel_width = target_box[2]
    pixel_center = target_box[0]
    
    distance = true_height / pixel_height * focal_length
    
    image_width = image_shape[1]
    x_shift = image_width / 2 - pixel_center
    theta = np.arctan(x_shift / focal_length)
    
    # Relative position (robot frame)
    distance_obj = distance / np.cos(theta)
    x_relative = distance_obj * np.cos(theta)
    y_relative = distance_obj * np.sin(theta)
    
    # Transform to world frame
    delta_x_world = x_relative * np.cos(robot_pose[2]) - y_relative * np.sin(robot_pose[2])
    delta_y_world = x_relative * np.sin(robot_pose[2]) + y_relative * np.cos(robot_pose[2])
    
    target_pose = {
        'y': (robot_pose[1] + delta_y_world)[0],
        'x': (robot_pose[0] + delta_x_world)[0]
    }
    
    # Apply offset for cube center correction
    offset_distance = 0.055  # 5.5 cm
    dx = target_pose['x'] - robot_pose[0][0]
    dy = target_pose['y'] - robot_pose[1][0]
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx_offset = offset_distance * dx / norm
        dy_offset = offset_distance * dy / norm
        target_pose['x'] += dx_offset
        target_pose['y'] += dy_offset
    
    # Compute confidence score
    bbox_area = pixel_width * pixel_height
    image_area = image_shape[0] * image_shape[1]
    confidence = compute_detection_confidence(bbox_area, image_area, distance)
    
    return target_pose, confidence


def merge_estimations_robust(target_pose_dict, distance_threshold=0.12, min_detections=2, outlier_std_threshold=2.0):
    """
    Advanced merging with outlier rejection, weighted averaging, and uncertainty estimation.
    
    Parameters:
    - target_pose_dict: dict of {label: {"x": x, "y": y, "confidence": c}}
    - distance_threshold: max distance for clustering (meters)
    - min_detections: minimum detections to keep a cluster
    - outlier_std_threshold: standard deviations for outlier rejection
    
    Returns:
    - target_est: refined positions with uncertainty metrics
    """
    
    # Group detections by base class
    grouped = {}
    for key, pose_data in target_pose_dict.items():
        base = key.split("_")[0]
        grouped.setdefault(base, []).append({
            "x": float(pose_data["x"]),
            "y": float(pose_data["y"]),
            "confidence": float(pose_data.get("confidence", 1.0))
        })
    
    # Process each class
    class_clusters = {}
    for cls, poses in grouped.items():
        if len(poses) == 0:
            continue
            
        clusters = []
        
        for p in poses:
            px, py, conf = p["x"], p["y"], p["confidence"]
            placed = False
            
            # Try to place in existing cluster
            for c in clusters:
                cx, cy = c["weighted_x"] / c["weight_sum"], c["weighted_y"] / c["weight_sum"]
                dist = np.hypot(px - cx, py - cy)
                
                if dist <= distance_threshold:
                    # Add to cluster
                    c["positions"].append({"x": px, "y": py, "conf": conf})
                    c["weighted_x"] += px * conf
                    c["weighted_y"] += py * conf
                    c["weight_sum"] += conf
                    placed = True
                    break
            
            if not placed:
                # Create new cluster
                clusters.append({
                    "positions": [{"x": px, "y": py, "conf": conf}],
                    "weighted_x": px * conf,
                    "weighted_y": py * conf,
                    "weight_sum": conf
                })
        
        # Refine each cluster with outlier rejection
        refined_clusters = []
        for c in clusters:
            positions = c["positions"]
            n = len(positions)
            
            # Skip if too few detections
            if n < min_detections:
                print(f"Warning: {cls} cluster with only {n} detection(s) - skipping")
                continue
            
            # Extract coordinates
            xs = np.array([p["x"] for p in positions])
            ys = np.array([p["y"] for p in positions])
            confs = np.array([p["conf"] for p in positions])
            
            # Outlier rejection using modified Z-score (more robust than standard deviation)
            if n >= 3:  # Need at least 3 points for outlier detection
                # Compute median absolute deviation (MAD) - more robust than std
                median_x = np.median(xs)
                median_y = np.median(ys)
                mad_x = np.median(np.abs(xs - median_x))
                mad_y = np.median(np.abs(ys - median_y))
                
                # Modified z-scores
                if mad_x > 0:
                    z_scores_x = 0.6745 * (xs - median_x) / mad_x
                else:
                    z_scores_x = np.zeros_like(xs)
                    
                if mad_y > 0:
                    z_scores_y = 0.6745 * (ys - median_y) / mad_y
                else:
                    z_scores_y = np.zeros_like(ys)
                
                # Combined z-score
                z_scores = np.sqrt(z_scores_x**2 + z_scores_y**2)
                
                # Keep only inliers
                inlier_mask = z_scores < outlier_std_threshold
                
                if np.sum(inlier_mask) >= min_detections:
                    xs = xs[inlier_mask]
                    ys = ys[inlier_mask]
                    confs = confs[inlier_mask]
                    n_removed = n - len(xs)
                    if n_removed > 0:
                        print(f"  Removed {n_removed} outlier(s) from {cls}")
            
            # Weighted average with normalized confidence weights
            confs_normalized = confs / np.sum(confs)
            x_final = np.sum(xs * confs_normalized)
            y_final = np.sum(ys * confs_normalized)
            
            # Compute uncertainty metrics
            std_x = np.std(xs)
            std_y = np.std(ys)
            uncertainty = np.sqrt(std_x**2 + std_y**2)
            
            # Average confidence
            avg_confidence = np.mean(confs)
            
            refined_clusters.append({
                "x": x_final,
                "y": y_final,
                "uncertainty": uncertainty,
                "confidence": avg_confidence,
                "n_detections": len(xs)
            })
        
        class_clusters[cls] = refined_clusters
    
    # Flatten to output format
    target_est = {}
    for cls, clusters in class_clusters.items():
        # Sort by confidence (highest first), then by position
        clusters_sorted = sorted(clusters, key=lambda c: (-c["confidence"], c["x"], c["y"]))
        
        for i, c in enumerate(clusters_sorted):
            target_est[f"{cls}_{i}"] = {
                "x": float(c["x"]),
                "y": float(c["y"]),
                "uncertainty": float(c["uncertainty"]),
                "confidence": float(c["confidence"]),
                "n_detections": int(c["n_detections"])
            }
    
    return target_est


# Removed GUI processing function - now using direct terminal output

# main loop
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("LIVE TARGET POSE ESTIMATION MONITOR")
    print("=" * 60)
    
    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    
    # init YOLO model
    model_path = f'{script_dir}/YOLO/model/best.pt'
    yolo = Detector(model_path)
    
    # load image poses
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    print(f"Loaded {len(image_poses)} images. Starting detection...")
    print("-" * 60)
    
    # estimate pose of targets in each image
    target_pose_dict = {}
    detected_type_list = []
    images_processed = 0
    
    for image_path in image_poses.keys():
        print(f"\n🔄 Processing: {os.path.basename(image_path)}")
        
        input_image = cv2.imread(image_path)
        bounding_boxes, bbox_img = yolo.detect_single_image(input_image)
        robot_pose = image_poses[image_path]
        
        image_detections = 0
        for detection in bounding_boxes:
            normalised_label = normalise_label(detection[0])
            occurrence = detected_type_list.count(normalised_label)
            
            # Estimate pose with confidence
            pose, confidence = estimate_pose(camera_matrix, detection, robot_pose)
            
            target_pose_dict[f'{normalised_label}_{occurrence}'] = {
                "x": pose["x"],
                "y": pose["y"],
                "confidence": confidence
            }
            
            detected_type_list.append(normalised_label)
            image_detections += 1
            
            print(f"  ✅ Detected {normalised_label}_{occurrence}: ({pose['x']:.3f}, {pose['y']:.3f}) [conf: {confidence:.3f}]")
        
        images_processed += 1
        
        # Print live results after each image
        if target_pose_dict:
            print(f"\n📊 LIVE RESULTS ({images_processed} images processed, {len(target_pose_dict)} total detections):")
            print("-" * 40)
            
            # Create live results for display
            live_results = {}
            for key, data in target_pose_dict.items():
                live_results[key] = {
                    "x": data["x"],
                    "y": data["y"],
                    "confidence": data["confidence"],
                    "uncertainty": 0.0,  # Will be calculated during merging
                    "n_detections": 1    # Single detection for now
                }
            
            # Print formatted JSON
            formatted_json = json.dumps(live_results, indent=4, sort_keys=True)
            print(formatted_json)
            print("-" * 40)
    
    print(f"\n🔄 Merging estimations with outlier rejection...")
    
    # Merge with robust estimation
    target_est = merge_estimations_robust(
        target_pose_dict,
        distance_threshold=0.15,  # 15cm clustering threshold (more lenient)
        min_detections=1,  # Allow single detections
        outlier_std_threshold=3.0  # Less aggressive outlier removal
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎯 FINAL TARGET ESTIMATES")
    print("=" * 60)
    
    for target_name, data in target_est.items():
        print(f"{target_name}:")
        print(f"  Position: ({data['x']:.4f}, {data['y']:.4f})")
        print(f"  Uncertainty: {data['uncertainty']:.4f} m")
        print(f"  Confidence: {data['confidence']:.3f}")
        print(f"  Detections: {data['n_detections']}")
        print()
    
    # Save with metadata
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)
    
    # Also save simplified version (just x, y for backwards compatibility)
    target_est_simple = {k: {"x": v["x"], "y": v["y"]} for k, v in target_est.items()}
    with open(f'{script_dir}/lab_output/targets_simple.txt', 'w') as fo:
        json.dump(target_est_simple, fo, indent=4)
    
    print('✅ Estimations saved!')
    print(f"  📁 Full data: lab_output/targets.txt")
    print(f"  📁 Simple format: lab_output/targets_simple.txt")
    print("\n" + "=" * 60)