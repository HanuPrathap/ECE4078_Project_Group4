
# inputs to path planning 

# fruit_list = ["orange", "apple", "potato"]  of all the fruit in the map including distractor fruit - python list
# fruit_true_pos = (n_targets, 2) storing x, y - numpy array
# aruco_true_pos = (10,2) for markers 1-10 - numpy array 





# path_planning_streaming.py
# Streaming grid planner for a 2x2 m arena — computes one leg at a time.
# Supports fixed-order mode (exactly one plan per leg) and greedy-on-demand mode.

import math, heapq, numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


# global variables 
size = 2.4



# ---------------------------
# Bounds & conversions
# ---------------------------

def world_bounds_fixed_2x2(size):
    """
    Return fixed bounds for a 2x2 m arena centered at the origin.
    Outputs:
        (xmin, ymin, xmax, ymax)
    """
    # 2x2 arena centered at origin
    # TODO - jas: change this up to accept inputs so we have the same dimensions 
    half = size/2
    return -half, -half, half, half

# input real wold coordinates to get graph indcies
def world_to_grid(x, y, xmin, ymin, res):
    """
    Convert world coordinates into grid indices.
    Inputs:
        x, y: world position in meters
        xmin, ymin: world origin of the grid
        res: resolution (meters per cell)
    Outputs:
        (gx, gy): integer grid coordinates
    """
    gx = int((x - xmin) / res)
    gy = int((y - ymin) / res)
    return gx, gy

# return real coordintes if we have graph coordinates 
def grid_to_world(gx, gy, xmin, ymin, res):
    """
    Convert grid indices back into world coordinates.
    Inputs:
        gx, gy: grid indices
        xmin, ymin: grid origin in meters
        res: resolution (meters per cell)
    Outputs:
        (x, y): world coordinates in meters
    """
    x = xmin + (gx + 0.5) * res
    y = ymin + (gy + 0.5) * res
    return x, y

# ---------------------------
# Costmap building (build once, reuse)
# ---------------------------

def rasterize_points_as_obstacles(occ, points_g, inflate_cells):
    H, W = occ.shape
    for (gx, gy) in points_g:  # gx=column, gy=row
        if 0 <= gx < W and 0 <= gy < H:
            for dx in range(-inflate_cells, inflate_cells+1):
                for dy in range(-inflate_cells, inflate_cells+1):
                    col = gx + dx  # x-direction = columns
                    row = gy + dy  # y-direction = rows
                    
                    if 0 <= col < W and 0 <= row < H:
                        if dx*dx + dy*dy <= inflate_cells*inflate_cells:
                            occ[row, col] = 1  # Clear: row first, col second
    return occ

# this outputs the occ grid 
def build_costmap_fixed_2x2(
    size,
    obstacle_points_m: np.ndarray,
    res: float = 0.05,
    robot_radius: float = 0.10,
    safety_margin: float = 0.02
):
    """
    Build costmap and occupancy grid from ground-truth arena + obstacles.
    """
    xmin, ymin, xmax, ymax = world_bounds_fixed_2x2(size)
    W = int(math.ceil((xmax - xmin) / res))
    H = int(math.ceil((ymax - ymin) / res))
    occ = np.zeros((H, W), dtype=np.uint8)

    # Calculate inflation parameters
    inflate_radius = robot_radius + safety_margin
    inflate_cells = max(1, int(round(inflate_radius / res)))

    # STEP 1: Add and inflate ONLY the internal obstacles (not borders)
    obstacles_g = [world_to_grid(x, y, xmin, ymin, res) for x, y in obstacle_points_m]
    rasterize_points_as_obstacles(occ, obstacles_g, inflate_cells)

    # STEP 2: Add solid border AFTER inflation (this ensures consistent thickness)
    # Create border mask
    border_thickness = inflate_cells    # Make border same as inflate cell - can turn this TODO
    
    # Apply thick border
    occ[:border_thickness, :] = 1      # Top border
    occ[-border_thickness:, :] = 1     # Bottom border  
    occ[:, :border_thickness] = 1      # Left border
    occ[:, -border_thickness:] = 1     # Right border

    # Convert to cost map
    cost = np.where(occ == 1, 255, 1).astype(np.uint8)
    meta = dict(xmin=xmin, ymin=ymin, res=res, W=W, H=H)
    return cost, occ, meta

#  visulalise the cost map for debugging 
# Option 2: Detailed matplotlib visualization
def visualize_costmap_detailed(cost, occ, meta, obstacle_points_m, target_points=None, title="Costmap Visualization"):
    """
    Create a detailed matplotlib visualization of the costmap with grid lines matching cell resolution.
    Args:
        cost: cost grid (1=free, 255=blocked)
        occ: occupancy grid (0=free, 1=occupied) 
        meta: metadata dict with grid info
        obstacle_points_m: original obstacle points in world coordinates
        target_points: optional list of target points to overlay
        title: plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate world extents
    xmax = meta['xmin'] + meta['W'] * meta['res']
    ymax = meta['ymin'] + meta['H'] * meta['res']
    extent = [meta['xmin'], xmax, meta['ymin'], ymax]
    
    # Create grid line positions (cell boundaries)
    x_grid_lines = np.arange(meta['xmin'], xmax + meta['res']/2, meta['res'])
    y_grid_lines = np.arange(meta['ymin'], ymax + meta['res']/2, meta['res'])
    
    # Plot 1: Occupancy Grid
    ax1 = axes[0]
    ax1.imshow(occ, cmap='RdYlBu_r', origin='lower', extent=extent)
    ax1.set_title('Occupancy Grid\n(Blue=Free, Red=Occupied)')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    
    # Add grid lines at cell boundaries
    for x in x_grid_lines:
        ax1.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    for y in y_grid_lines:
        ax1.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
    
    # Overlay original obstacle points
    if len(obstacle_points_m) > 0:
        ax1.scatter(obstacle_points_m[:, 0], obstacle_points_m[:, 1], 
                   c='black', s=20, marker='x', alpha=0.7, label='Original Obstacles')
    
    # Overlay targets if provided
    if target_points is not None and len(target_points) > 0:
        targets_array = np.array(target_points)
        ax1.scatter(targets_array[:, 0], targets_array[:, 1], 
                   c='green', s=100, marker='*', label='Targets', zorder=5)
        # Number the targets
        for i, (tx, ty) in enumerate(targets_array):
            ax1.annotate(f'{i+1}', (tx, ty), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white', weight='bold')
    
    ax1.legend()
    ax1.set_aspect('equal')  # Ensure square cells look square
    
    # Plot 2: Cost Grid
    ax2 = axes[1]
    im = ax2.imshow(cost, cmap='viridis', origin='lower', extent=extent)
    ax2.set_title('Cost Grid\n(Dark=Low Cost, Bright=High Cost)')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    
    # Add grid lines at cell boundaries
    for x in x_grid_lines:
        ax2.axvline(x, color='white', alpha=0.3, linewidth=0.5)
    for y in y_grid_lines:
        ax2.axhline(y, color='white', alpha=0.3, linewidth=0.5)
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Cost Value')
    
    # Overlay original obstacle points
    if len(obstacle_points_m) > 0:
        ax2.scatter(obstacle_points_m[:, 0], obstacle_points_m[:, 1], 
                   c='red', s=20, marker='x', alpha=0.7, label='Original Obstacles')
    
    # Overlay targets if provided
    if target_points is not None and len(target_points) > 0:
        targets_array = np.array(target_points)
        ax2.scatter(targets_array[:, 0], targets_array[:, 1], 
                   c='white', s=100, marker='*', label='Targets', zorder=5)
        # Number the targets
        for i, (tx, ty) in enumerate(targets_array):
            ax2.annotate(f'{i+1}', (tx, ty), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='black', weight='bold')
    
    ax2.legend()
    ax2.set_aspect('equal')  # Ensure square cells look square
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
# ---------------------------
# Dijkstra for ONE leg
# ---------------------------

def dijkstra(costmap: np.ndarray, start_g, goal_g):
    H, W = costmap.shape

    # tells dickstra wether cell is off limits or not 
    # return true if cell is blocked and false if traversable
    def blocked(p):
        x, y = p
        return not (0 <= x < W and 0 <= y < H) or (costmap[y, x] >= 255)

    # checks if our end opint and start point are valid 
    if blocked(start_g) or blocked(goal_g):
        return None, math.inf
    
    
    nbrs = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(1,-1),(-1,1),(1,1)]
    pq = []
    g = {start_g: 0.0}
    came = {}
    heapq.heappush(pq, (0.0, start_g))
    visited = set()

    while pq:
        gcur, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == goal_g:
            path = [u]
            while u in came:
                u = came[u]
                path.append(u)
            path.reverse()
            return path, gcur

        ux, uy = u
        for dx, dy in nbrs:
            v = (ux + dx, uy + dy)
            if blocked(v):
                continue
            step = 1.4142 if dx and dy else 1.0
            move_cost = step * float(costmap[v[1], v[0]])
            gv = gcur + move_cost
            if gv < g.get(v, math.inf):
                g[v] = gv
                came[v] = u
                heapq.heappush(pq, (gv, v))

    return None, math.inf

def plan_leg_dijkstra(costmap, meta, start_xy: Tuple[float,float], goal_xy: Tuple[float,float]):
    # Compute ONE path (start -> goal). No precomputation for others.
    sx, sy = start_xy; gx, gy = goal_xy
    sg = world_to_grid(sx, sy, meta["xmin"], meta["ymin"], meta["res"])
    gg = world_to_grid(gx, gy, meta["xmin"], meta["ymin"], meta["res"])
    gpath, gcost = dijkstra(costmap, sg, gg)
    if gpath is None:
        return None, math.inf
    # Convert to world
    poly = [grid_to_world(px, py, meta["xmin"], meta["ymin"], meta["res"]) for (px, py) in gpath]
    return poly, gcost

# ---------------------------
# Optional smoothing for a single leg
# ---------------------------

def smooth_polyline(poly, lam=0.4, iters=40):
    if not poly or len(poly) < 4:
        return poly
    P = np.array(poly, dtype=np.float64)
    Q = P.copy()
    for _ in range(iters):
        Q[1:-1] = (1-lam)*Q[1:-1] + lam*0.5*(Q[0:-2] + Q[2:])
    return [tuple(x) for x in Q]

# ---------------------------
# Greedy-on-demand ordering (optional)
# ---------------------------

def choose_next_target_greedy(current_xy, remaining_targets, costmap, meta):
    """
    From current pose, evaluate ONE path to each remaining target and choose the cheapest.
    Returns: (best_index_in_remaining, best_path, best_cost)
    """
    if not remaining_targets:
        return None, None, 0.0
    best_i, best_p, best_c = None, None, math.inf
    for i, tgt in enumerate(remaining_targets):
        pth, cst = plan_leg_dijkstra(costmap, meta, current_xy, tgt)
        if pth is not None and cst < best_c:
            best_i, best_p, best_c = i, pth, cst
    return best_i, best_p, best_c





