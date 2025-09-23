import matplotlib.pyplot as plt
import numpy as np
import json
import math

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for"""
    # Use relative path for portability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, fname)
    
    with open(file_path, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 3)
            y = np.round(gt_dict[key]['y'], 3)

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

def read_search_list():
    """Read the search order of the target fruits"""
    search_list = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'search_list.txt')
    
    with open(file_path, 'r') as fd:
        fruits = fd.readlines()
        for fruit in fruits:
            search_list.append(fruit.strip())
    return search_list

def targets_from_search_list(search_list, fruit_list, fruit_true_pos):
    """Build targets and distractors from search list"""
    name_to_positions = {}
    all_fruit_positions = []
    
    for name, (x, y) in zip(fruit_list, fruit_true_pos):
        pos = (float(x), float(y))
        all_fruit_positions.append((name, pos))
        
        if name not in name_to_positions:
            name_to_positions[name] = []
        name_to_positions[name].append(pos)

    name_to_closest_pos = {}
    for name, positions in name_to_positions.items():
        closest_pos = min(positions, key=lambda pos: math.sqrt(pos[0]**2 + pos[1]**2))
        name_to_closest_pos[name] = closest_pos

    targets_xy = []
    target_names = []
    target_names_used = set()
    
    for name in search_list:
        if name in name_to_closest_pos:
            closest_pos = name_to_closest_pos[name]
            targets_xy.append(closest_pos)
            target_names.append(name)
            target_names_used.add(name)

    distractor_xy = []
    distractor_names = []
    
    for name, pos in all_fruit_positions:
        if name not in target_names_used:
            if pos not in distractor_xy:
                distractor_xy.append(pos)
                distractor_names.append(name)

    return targets_xy, target_names, distractor_xy, distractor_names

def visualize_map():
    """Create a comprehensive map visualization"""
    # Load data
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map("M3_prac_map_full.txt")
    search_list = read_search_list()
    targets_xy, target_names, distractor_xy, distractor_names = targets_from_search_list(
        search_list, fruits_list, fruits_true_pos)
    
    # Convert ArUco positions to dictionary
    known_landmarks = {}
    for i in range(len(aruco_true_pos)):
        marker_id = i + 1
        if marker_id == 10:
            known_landmarks[10] = [float(aruco_true_pos[9, 0]), float(aruco_true_pos[9, 1])]
        else:
            known_landmarks[marker_id] = [float(aruco_true_pos[i, 0]), float(aruco_true_pos[i, 1])]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot ArUco markers
    aruco_x = [pos[0] for pos in known_landmarks.values()]
    aruco_y = [pos[1] for pos in known_landmarks.values()]
    ax.scatter(aruco_x, aruco_y, c='blue', marker='s', s=100, 
               label='ArUco Markers', edgecolors='black', linewidth=1, zorder=3)
    
    # Add ArUco marker labels
    for tag_id, pos in known_landmarks.items():
        ax.annotate(f'A{tag_id}', (pos[0], pos[1]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color='blue', weight='bold')
    
    # Plot target fruits (in search order)
    target_x = [pos[0] for pos in targets_xy]
    target_y = [pos[1] for pos in targets_xy]
    ax.scatter(target_x, target_y, c='red', marker='o', s=120,
               label='Target Fruits', edgecolors='darkred', linewidth=2, zorder=4)
    
    # Add target labels with search order numbers
    for i, ((x, y), name) in enumerate(zip(targets_xy, target_names)):
        ax.annotate(f'{i+1}:{name}', (x, y), 
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, color='darkred', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Draw path between targets (search order)
    if len(targets_xy) > 1:
        # Add robot starting position (0,0) to beginning of path
        path_x = [0] + target_x
        path_y = [0] + target_y
        ax.plot(path_x, path_y, 'r--', linewidth=2, alpha=0.7, 
                label='Search Path', zorder=2)
        
        # Add arrows to show direction
        for i in range(len(path_x)-1):
            dx = path_x[i+1] - path_x[i]
            dy = path_y[i+1] - path_y[i]
            ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                       xytext=(path_x[i], path_y[i]),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Plot distractor fruits
    if distractor_xy:
        distractor_x = [pos[0] for pos in distractor_xy]
        distractor_y = [pos[1] for pos in distractor_xy]
        ax.scatter(distractor_x, distractor_y, c='orange', marker='x', s=80,
                   label='Distractor Fruits', linewidth=3, zorder=3)
        
        # Add distractor labels
        for (x, y), name in zip(distractor_xy, distractor_names):
            ax.annotate(name, (x, y), 
                       xytext=(5, -15), textcoords='offset points',
                       fontsize=8, color='orange', style='italic')
    
    # Plot robot starting position
    ax.scatter([0], [0], c='green', marker='^', s=200,
               label='Robot Start', edgecolors='darkgreen', linewidth=2, zorder=5)
    ax.annotate('START', (0, 0), 
               xytext=(10, 10), textcoords='offset points',
               fontsize=10, color='darkgreen', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    # Combine obstacles for visualization
    obstacles_list = [(pos[0], pos[1]) for pos in known_landmarks.values()] + distractor_xy
    
    # Set up the plot
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    ax.set_title('Robot Navigation Map\nTargets, Obstacles, and Search Path', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')
    all_x = aruco_x + target_x + ([0] if not distractor_xy else [0] + [pos[0] for pos in distractor_xy])
    all_y = aruco_y + target_y + ([0] if not distractor_xy else [0] + [pos[1] for pos in distractor_xy])
    
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Add statistics text box
    stats_text = f"""Map Statistics:
    • ArUco Markers: {len(known_landmarks)}
    • Target Fruits: {len(targets_xy)}
    • Distractor Fruits: {len(distractor_xy)}
    • Total Obstacles: {len(obstacles_list)}
    
    Search Order:
    {chr(10).join([f"{i+1}. {name}" for i, name in enumerate(target_names)])}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("MAP VISUALIZATION SUMMARY")
    print("="*50)
    print(f"Total elements plotted:")
    print(f"  - ArUco Markers: {len(known_landmarks)} (blue squares)")
    print(f"  - Target Fruits: {len(targets_xy)} (red circles, numbered)")
    print(f"  - Distractor Fruits: {len(distractor_xy)} (orange X's)")
    print(f"  - Robot Start: (0, 0) (green triangle)")
    print(f"  - Search Path: Red dashed line with arrows")
    print(f"\nTotal obstacles for path planning: {len(obstacles_list)}")
    print("="*50)

if __name__ == "__main__":
    import os
    visualize_map()