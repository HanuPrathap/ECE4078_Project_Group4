# path_planning_astar.py
# Grid + A* planner for a ~2–3 m square arena.
# - Builds a 2D occupancy/cost grid with inflation
# - A* (octile heuristic) with diagonal corner-cut prevention
# - World<->grid helpers and visualization

import math, heapq, numpy as np
from typing import List, Tuple
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
):
    xmin, ymin, xmax, ymax = world_bounds_fixed(size_m)
    W = int(math.ceil((xmax - xmin) / res))
    H = int(math.ceil((ymax - ymin) / res))
    occ = np.zeros((H, W), dtype=np.uint8)

    inflate_r = robot_radius_m + safety_margin_m
    inflate_cells = max(1, int(round(inflate_r / res)))

    if obstacle_points_m is not None and len(obstacle_points_m) > 0:
        pts_g = [world_to_grid(float(x), float(y), xmin, ymin, res, W, H)
                 for (x, y) in obstacle_points_m]
        _rasterize_points_as_obstacles(occ, pts_g, inflate_cells)

    b = inflate_cells
    occ[:b, :] = 1; occ[-b:, :] = 1; occ[:, :b] = 1; occ[:, -b:] = 1

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
        if u in visited: continue
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
        # place legend just outside the left edge
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
        # place legend just outside the right edge
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
