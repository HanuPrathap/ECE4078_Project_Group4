# path_planning_astar.py
# Grid + A* planner for a ~2–3 m square arena.
# - Builds a 2D occupancy/cost grid with inflation
# - A* (octile heuristic) with diagonal corner-cut prevention
# - World<->grid helpers and visualization

import math, heapq, numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# ---------------------------
# Bounds & conversions
# ---------------------------

def world_bounds_fixed(size_m: float) -> Tuple[float,float,float,float]:
    half = size_m * 0.5
    return -half, -half, half, half

def world_to_grid(x, y, xmin, ymin, res, W=None, H=None):
    gx = int(math.floor((x - xmin) / res))
    gy = int(math.floor((y - ymin) / res))
    if W is not None and H is not None:
        gx = max(0, min(W-1, gx))
        gy = max(0, min(H-1, gy))
    return gx, gy

def grid_to_world(gx, gy, xmin, ymin, res):
    x = xmin + (gx + 0.5) * res
    y = ymin + (gy + 0.5) * res
    return x, y

# ---------------------------
# Costmap building (inflate obstacles)
# ---------------------------

# does a circle inflation
def _rasterize_points_as_obstacles(occ, points_g, inflate_cells: int):
    H, W = occ.shape
    r2 = inflate_cells * inflate_cells
    for (gx, gy) in points_g:
        if 0 <= gx < W and 0 <= gy < H:
            for dx in range(-inflate_cells, inflate_cells+1):
                for dy in range(-inflate_cells, inflate_cells+1):
                    if dx*dx + dy*dy <= r2:
                        cx, cy = gx + dx, gy + dy
                        if 0 <= cx < W and 0 <= cy < H:
                            occ[cy, cx] = 1
    return occ


def build_costmap_fixed(
    size_m: float,
    obstacle_points_m: np.ndarray,
    res: float = 0.03,
    robot_radius_m: float = 0.10,
    safety_margin_m: float = 0.02,
    target_points_m: Optional[np.ndarray] = None,
):
    """
    Build occupancy & cost maps with circular inflation around obstacles and borders.
    - obstacle_points_m: (N,2) array of obstacles (AruCo + distractors)
    - target_points_m: (M,2) array of targets (light inflation) or None
    """
    xmin, ymin, xmax, ymax = world_bounds_fixed(size_m)
    W = int(math.ceil((xmax - xmin) / res))
    H = int(math.ceil((ymax - ymin) / res))
    occ = np.zeros((H, W), dtype=np.uint8)

    # ==== base obstacles (AruCo, distractors) ===== #
    inflate_r = robot_radius_m + safety_margin_m
    inflate_cells = max(1, int(round(inflate_r / res)))

    # Light inflation for targets (so we don't clip the fruit)
    target_inflate_r = 0.05  # tune if needed
    inflate_cells2 = max(1, int(round(target_inflate_r / res)))

    if obstacle_points_m is not None and len(obstacle_points_m) > 0:
        pts_g = [world_to_grid(float(x), float(y), xmin, ymin, res, W, H)
                 for (x, y) in obstacle_points_m]
        _rasterize_points_as_obstacles(occ, pts_g, inflate_cells)

    # === target obstacles == #
    if target_points_m is not None and len(target_points_m) > 0:
        pts_g = [world_to_grid(float(x), float(y), xmin, ymin, res, W, H)
                 for (x, y) in target_points_m]
        _rasterize_points_as_obstacles(occ, pts_g, inflate_cells2)

    # --- inflate borders ---- #
    b = inflate_cells
    occ[:b, :] = 1; occ[-b:, :] = 1; occ[:, :b] = 1; occ[:, -b:] = 1

    cost = np.where(occ == 1, 255, 1).astype(np.uint8)
    meta = dict(xmin=xmin, ymin=ymin, res=res, W=W, H=H)
    return cost, occ, meta

# does square inflation - check to see if its better 
def _rasterize_points_as_obstacles_square(occ, points_g, half_cells: int):
    H, W = occ.shape
    h = int(max(1, half_cells))
    for (gx, gy) in points_g:
        if 0 <= gx < W and 0 <= gy < H:
            x0 = max(0, gx - h)
            x1 = min(W - 1, gx + h)
            y0 = max(0, gy - h)
            y1 = min(H - 1, gy + h)
            occ[y0:y1+1, x0:x1+1] = 1
    return occ

# builds the cost map for the squares 
def build_costmap_fixed_squares(
    size_m: float,
    aruco_points_m: np.ndarray,           # (Na,2) or None
    fruit_points_m: np.ndarray,           # (Nf,2) or None (targets + distractors)
    res: float = 0.03,
    robot_radius_m: float = 0.10,
    safety_margin_m: float = 0.02,
    aruco_size_m: float = 0.08,           # marker side length (m)
    fruit_size_m: float = 0.06,           # cube side length (m)
):
    """
    Build occupancy/cost maps using axis-aligned SQUARE inflation.
    - Each object gets its own physical half-size plus robot clearance.
    - Borders are inflated by clearance as well.
    """
    xmin = -size_m * 0.5
    ymin = -size_m * 0.5
    xmax = +size_m * 0.5
    ymax = +size_m * 0.5

    W = int(math.ceil((xmax - xmin) / res))
    H = int(math.ceil((ymax - ymin) / res))
    occ = np.zeros((H, W), dtype=np.uint8)

    # robot clearance (added to half-size of each object)
    clearance_m = robot_radius_m + safety_margin_m  

    # convert half-size (m) + clearance to half-width in cells
    aruco_half_cells = int(math.ceil(((aruco_size_m * 0.5) + clearance_m) / res))
    fruit_half_cells = int(math.ceil(((fruit_size_m * 0.5) + clearance_m) / res))

    def world_to_grid_local(x, y):
        gx = int(math.floor((x - xmin) / res))
        gy = int(math.floor((y - ymin) / res))
        gx = max(0, min(W - 1, gx))
        gy = max(0, min(H - 1, gy))
        return gx, gy

    # Rasterize ArUco markers as squares
    if aruco_points_m is not None and len(aruco_points_m) > 0:
        pts_g = [world_to_grid_local(float(x), float(y)) for (x, y) in np.asarray(aruco_points_m)]
        _rasterize_points_as_obstacles_square(occ, pts_g, aruco_half_cells)

    # Rasterize fruits (targets + distractors) as squares (can be a different size)
    if fruit_points_m is not None and len(fruit_points_m) > 0:
        pts_g = [world_to_grid_local(float(x), float(y)) for (x, y) in np.asarray(fruit_points_m)]
        _rasterize_points_as_obstacles_square(occ, pts_g, fruit_half_cells)

    # Inflate borders by clearance (so the robot can't clip walls)
    border_cells = max(1, int(math.ceil(clearance_m / res)))
    occ[:border_cells, :] = 1
    occ[-border_cells:, :] = 1
    occ[:, :border_cells] = 1
    occ[:, -border_cells:] = 1

    cost = np.where(occ == 1, 255, 1).astype(np.uint8)
    meta = dict(xmin=xmin, ymin=ymin, res=res, W=W, H=H)
    return cost, occ, meta





# ---------------------------
# A* for ONE leg
# ---------------------------

_OCT_DIAG = 1.41421356237

def _blocked(costmap: np.ndarray, p) -> bool:
    H, W = costmap.shape
    x, y = p
    return not (0 <= x < W and 0 <= y < H) or (costmap[y, x] >= 255)

def _can_step(costmap: np.ndarray, u, v) -> bool:
    ux, uy = u; vx, vy = v
    dx, dy = vx - ux, vy - uy
    # prevent diagonal corner cutting
    if dx and dy:
        if _blocked(costmap, (ux + dx, uy)) or _blocked(costmap, (ux, uy + dy)):
            return False
    return True

def _octile(dx: int, dy: int) -> float:
    a, b = abs(dx), abs(dy)
    m = min(a, b); M = max(a, b)
    return (M - m) + _OCT_DIAG * m

def astar(costmap: np.ndarray, start_g: Tuple[int,int], goal_g: Tuple[int,int]):
    if _blocked(costmap, start_g) or _blocked(costmap, goal_g):
        return None, math.inf

    free = costmap[costmap < 255]
    min_free = float(np.min(free)) if free.size else 1.0

    nbrs = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(1,-1),(-1,1),(1,1)]
    g = {start_g: 0.0}
    came = {}
    pq = []
    h0 = _octile(goal_g[0]-start_g[0], goal_g[1]-start_g[1]) * min_free
    heapq.heappush(pq, (h0, 0.0, start_g))
    visited = set()

    while pq:
        fcur, gcur, u = heapq.heappop(pq)
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
            if _blocked(costmap, v) or not _can_step(costmap, u, v):
                continue
            step = _OCT_DIAG if (dx and dy) else 1.0
            move_cost = step * float(costmap[v[1], v[0]])
            gv = gcur + move_cost
            if gv < g.get(v, math.inf):
                g[v] = gv
                came[v] = u
                hv = _octile(goal_g[0]-v[0], goal_g[1]-v[1]) * min_free
                fv = gv + hv
                heapq.heappush(pq, (fv, gv, v))

    return None, math.inf

def plan_leg_astar(costmap, meta, start_xy: Tuple[float,float], goal_xy: Tuple[float,float]):
    sx, sy = start_xy; gx, gy = goal_xy
    sg = world_to_grid(sx, sy, meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
    gg = world_to_grid(gx, gy, meta["xmin"], meta["ymin"], meta["res"], meta["W"], meta["H"])
    gpath, gcost = astar(costmap, sg, gg)
    if gpath is None:
        return None, math.inf
    poly = [grid_to_world(px, py, meta["xmin"], meta["ymin"], meta["res"]) for (px, py) in gpath]
    return poly, gcost

# ---------------------------
# Polyline smoothing (optional)
# ---------------------------

def smooth_polyline(poly, lam=0.3, iters=30):
    if not poly or len(poly) < 4:
        return poly
    P = np.array(poly, dtype=np.float64)
    Q = P.copy()
    for _ in range(iters):
        Q[1:-1] = (1-lam)*Q[1:-1] + lam*0.5*(Q[0:-2] + Q[2:])
    return [tuple(x) for x in Q]

# ---------------------------
# Visualization (debug)
# ---------------------------

def visualize_costmap_detailed(cost, occ, meta, obstacle_points_m, target_points=None, title="Costmap"):
    # make layout auto-reserve space for legends/colorbar
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')

    xmax = meta['xmin'] + meta['W'] * meta['res']
    ymax = meta['ymin'] + meta['H'] * meta['res']
    extent = [meta['xmin'], xmax, meta['ymin'], ymax]
    x_grid_lines = np.arange(meta['xmin'], xmax + meta['res']/2, meta['res'])
    y_grid_lines = np.arange(meta['ymin'], ymax + meta['res']/2, meta['res'])

    # --- Left: Occupancy ---
    ax1 = axes[0]
    ax1.imshow(occ, cmap='RdYlBu_r', origin='lower', extent=extent)
    for x in x_grid_lines: ax1.axvline(x, color='gray', alpha=0.2, linewidth=0.5)
    for y in y_grid_lines: ax1.axhline(y, color='gray', alpha=0.2, linewidth=0.5)
    h_obs = None
    if obstacle_points_m is not None and len(obstacle_points_m) > 0:
        obs = np.asarray(obstacle_points_m)
        h_obs = ax1.scatter(obs[:,0], obs[:,1], c='k', s=20, marker='x', label='obstacles', zorder=5)
    ax1.set_aspect('equal'); ax1.set_title('Occupancy'); ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    if h_obs is not None:
        ax1.legend(handles=[h_obs], loc='center right', bbox_to_anchor=(-0.12, 0.5), frameon=True)

    # --- Right: Cost ---
    ax2 = axes[1]
    im = ax2.imshow(cost, cmap='viridis', origin='lower', extent=extent)
    for x in x_grid_lines: ax2.axvline(x, color='w', alpha=0.2, linewidth=0.5)
    for y in y_grid_lines: ax2.axhline(y, color='w', alpha=0.2, linewidth=0.5)
    h_tgt = None
    if target_points is not None and len(target_points) > 0:
        tarr = np.array(target_points)
        h_tgt = ax2.scatter(tarr[:,0], tarr[:,1], s=80, marker='*', label='targets', zorder=6)
    ax2.set_aspect('equal'); ax2.set_title('Cost'); ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    if h_tgt is not None:
        ax2.legend(handles=[h_tgt], loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

    # colorbar below both plots so it doesn't collide with legends
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, fraction=0.05)
    cbar.set_label('cost')

    plt.suptitle(title)
    plt.show()

def visualize_plan_over_costmap(cost, occ, meta, path_xy, start_xy=None, goal_xy=None, title="Planned path"):
    """Quick preview: draw costmap + one polyline path."""
    xmax = meta['xmin'] + meta['W'] * meta['res']
    ymax = meta['ymin'] + meta['H'] * meta['res']
    extent = [meta['xmin'], xmax, meta['ymin'], ymax]

    plt.figure(figsize=(7, 6))
    plt.imshow(cost, cmap='gray_r', origin='lower', extent=extent, alpha=0.8)
    if path_xy and len(path_xy) > 1:
        P = np.array(path_xy)
        plt.plot(P[:,0], P[:,1], linewidth=2)
        plt.scatter(P[0,0], P[0,1], marker='o', s=70, label='start')
        plt.scatter(P[-1,0], P[-1,1], marker='*', s=100, label='goal')
    if start_xy is not None:
        plt.scatter(start_xy[0], start_xy[1], marker='o', s=70)
    if goal_xy is not None:
        plt.scatter(goal_xy[0], goal_xy[1], marker='*', s=100)
    plt.legend(); plt.title(title); plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.gca().set_aspect('equal')
    plt.tight_layout(); plt.show()


import math, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def _sample_polyline_world(path_xy, step_m=0.02):
    """Evenly sample points (≈every step_m meters) along a world-space polyline."""
    if not path_xy or len(path_xy) < 2: return list(path_xy)
    P = np.array(path_xy, dtype=float)
    out = [tuple(P[0])]
    for i in range(len(P)-1):
        a, b = P[i], P[i+1]
        seg = b - a
        L = float(np.hypot(seg[0], seg[1]))
        if L <= 1e-9: continue
        n = max(1, int(math.ceil(L / step_m)))
        for k in range(1, n+1):
            t = k / n
            out.append(tuple(a + t*seg))
    return out

def _circle_rect_collision(cx, cy, r, rx, ry, half):
    """
    Circle(center cx,cy radius r) vs axis-aligned square centered at (rx,ry) with half-side 'half'.
    Collision if distance from circle center to rectangle ≤ r.
    """
    dx = max(abs(cx - rx) - half, 0.0)
    dy = max(abs(cy - ry) - half, 0.0)
    return (dx*dx + dy*dy) <= (r*r)

def _clearance_to_square(cx, cy, r, rx, ry, half):
    """Signed clearance: positive if separated, negative if overlapping."""
    dx = max(abs(cx - rx) - half, 0.0)
    dy = max(abs(cy - ry) - half, 0.0)
    d = math.hypot(dx, dy)
    return d - r

def visualize_plan_with_robot_footprint(
    cost, occ, meta, path_xy,
    start_xy=None, goal_xy=None,
    robot_radius_m=0.10,
    aruco_points=None, fruit_points=None,
    aruco_size_m=0.08, fruit_size_m=0.06,
    sample_step_m=0.02,
    title="Planned path (with robot footprint)"
):
    """
    Draw costmap, squares for ArUco/fruits at their *physical* size,
    the planned path, and translucent robot disks along the path.
    Any disk that intersects a square is shown in RED; others in BLUE.
    """
    xmax = meta['xmin'] + meta['W'] * meta['res']
    ymax = meta['ymin'] + meta['H'] * meta['res']
    extent = [meta['xmin'], xmax, meta['ymin'], ymax]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cost, cmap='gray_r', origin='lower', extent=extent, alpha=0.85)

    # Draw squares for obstacles at *true* geometry (not inflated)
    a_half = aruco_size_m * 0.5
    f_half = fruit_size_m * 0.5
    if aruco_points is not None and len(aruco_points) > 0:
        for (x, y) in np.asarray(aruco_points):
            ax.add_patch(Rectangle((x - a_half, y - a_half), 2*a_half, 2*a_half,
                                   facecolor='dimgray', edgecolor='k', alpha=0.9, zorder=4))
    if fruit_points is not None and len(fruit_points) > 0:
        for (x, y) in np.asarray(fruit_points):
            ax.add_patch(Rectangle((x - f_half, y - f_half), 2*f_half, 2*f_half,
                                   facecolor='gray', edgecolor='k', alpha=0.9, zorder=4))

    # Path and endpoints
    if path_xy and len(path_xy) > 1:
        P = np.array(path_xy)
        ax.plot(P[:,0], P[:,1], linewidth=2, label='path', zorder=6)
        ax.scatter(P[0,0], P[0,1], s=70, marker='o', label='start', zorder=7)
        ax.scatter(P[-1,0], P[-1,1], s=100, marker='*', label='goal', zorder=7)

    if start_xy is not None:
        ax.scatter(start_xy[0], start_xy[1], s=70, marker='o', zorder=7)
    if goal_xy is not None:
        ax.scatter(goal_xy[0], goal_xy[1], s=100, marker='*', zorder=7)

    # Pre-arrays for collision checks
    A = np.asarray(aruco_points) if aruco_points is not None else np.empty((0,2))
    F = np.asarray(fruit_points) if fruit_points is not None else np.empty((0,2))

    # Sample robot footprint along the path
    samples = _sample_polyline_world(path_xy, step_m=sample_step_m) if path_xy else []
    min_clear = float('inf'); collisions = 0

    for (cx, cy) in samples:
        hit = False
        # Check ArUco squares
        for (sx, sy) in A:
            if _circle_rect_collision(cx, cy, robot_radius_m, sx, sy, a_half):
                hit = True; break
        # Check fruits if not already hit
        if not hit:
            for (sx, sy) in F:
                if _circle_rect_collision(cx, cy, robot_radius_m, sx, sy, f_half):
                    hit = True; break

        # track min clearance (positive = free, negative = overlap)
        for (sx, sy) in A:
            min_clear = min(min_clear, _clearance_to_square(cx, cy, robot_radius_m, sx, sy, a_half))
        for (sx, sy) in F:
            min_clear = min(min_clear, _clearance_to_square(cx, cy, robot_radius_m, sx, sy, f_half))

        color = 'tab:red' if hit else 'C0'
        alpha = 0.35 if hit else 0.2
        if hit: collisions += 1
        ax.add_patch(Circle((cx, cy), radius=robot_radius_m, facecolor=color,
                            edgecolor='none', alpha=alpha, zorder=5))

    ax.set_aspect('equal')
    ax.set_title(f"{title}\nfootprint collisions: {collisions} | min clearance: {min_clear:.3f} m")
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.show()



import math, numpy as np

def _segment_min_cost(cost, meta, p0, p1, sample_step_cells=0.5):
    """
    Sample along world-space segment p0->p1 every (sample_step_cells * res).
    Return min cost encountered; 255 means collision with hard obstacle/border.
    """
    res = meta["res"]; xmin = meta["xmin"]; ymin = meta["ymin"]
    W = meta["W"]; H = meta["H"]
    p0 = np.asarray(p0, dtype=float); p1 = np.asarray(p1, dtype=float)
    d = float(np.hypot(*(p1 - p0)))
    if d < 1e-9:
        gx, gy = world_to_grid(p0[0], p0[1], xmin, ymin, res, W, H)
        return int(cost[gy, gx])
    step = max(sample_step_cells * res, 1e-6)
    n = max(1, int(math.ceil(d / step)))
    m = 255
    for k in range(n + 1):
        t = k / n
        x, y = p0 + t * (p1 - p0)
        gx, gy = world_to_grid(x, y, xmin, ymin, res, W, H)
        c = int(cost[gy, gx])
        if c >= 255:
            return 255  # immediate collision
        if c < m:
            m = c
    return m

def _subdivide_until_safe(cost, meta, a, b, max_allowed_cost=254,
                          sample_step_cells=0.5, max_depth=12):
    """
    Return a list of waypoints [a,...,b] where each adjacent segment
    is safe (min cost <= max_allowed_cost). Recursively bisects risky segments.
    Return None if we hit max_depth and still not safe.
    """
    mc = _segment_min_cost(cost, meta, a, b, sample_step_cells)
    if mc <= max_allowed_cost:
        return [tuple(a), tuple(b)]
    if max_depth <= 0:
        return None
    a = np.asarray(a, float); b = np.asarray(b, float)
    mid = 0.5 * (a + b)
    left = _subdivide_until_safe(cost, meta, a, mid, max_allowed_cost,
                                 sample_step_cells, max_depth - 1)
    if left is None:
        return None
    right = _subdivide_until_safe(cost, meta, mid, b, max_allowed_cost,
                                  sample_step_cells, max_depth - 1)
    if right is None:
        return None
    return left[:-1] + right  # stitch (drop duplicate mid)

def make_collision_checked_path(cost, meta, path_xy,
                                max_allowed_cost=254, sample_step_cells=0.5):
    """
    For a polyline path, ensure every segment is cost-safe.
    Returns a refined waypoint list or None if it can't be made safe.
    """
    if not path_xy or len(path_xy) < 2:
        return path_xy
    safe = [tuple(path_xy[0])]
    for i in range(len(path_xy) - 1):
        a = safe[-1]
        b = path_xy[i + 1]
        seg = _subdivide_until_safe(cost, meta, a, b, max_allowed_cost, sample_step_cells)
        if seg is None:
            return None
        safe.extend(seg[1:])  # append, skipping duplicate a
    return safe
