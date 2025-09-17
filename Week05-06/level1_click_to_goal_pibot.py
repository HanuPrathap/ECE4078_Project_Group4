"""
Level 1: Click-to-Go (PenguinPi)
- 2.5 m x 2.5 m map panel you can click
- Robot starts at (0,0,0) in the middle
- On click: turn in place, then drive straight (timed), using your calibration
- Uses the same set_velocity(...) pattern as your working code

Run:
  python level1_click_to_goal_pibot.py --ip 192.168.50.1 --port 8080

Keys:
  SPACE = emergency stop (clears current goal)
  R     = reset internal pose to (0,0,0)
  ESC   = quit
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pygame

# ---------------------------------------
# Import your project modules
# ---------------------------------------
# Allow "util" and "slam" packages to import like in operate.py
sys.path.insert(0, os.path.join(os.getcwd(), "util"))
sys.path.insert(0, os.path.join(os.getcwd(), "slam"))

from pibot import PenguinPi  # type: ignore

# ---------------------------------------
# Map & UI config
# ---------------------------------------
ARENA_SIZE_M = 2.5
WORLD_X_MIN, WORLD_X_MAX = -ARENA_SIZE_M / 2, ARENA_SIZE_M / 2
WORLD_Y_MIN, WORLD_Y_MAX = -ARENA_SIZE_M / 2, ARENA_SIZE_M / 2

PAD = 20
PANEL_W = 480
PANEL_H = 480
MAP_ORIGIN_PX = (PAD, PAD)

WIN_W = PANEL_W + 2 * PAD
WIN_H = PANEL_H + 100

GOAL_TOL_M = 0.25   # stop within 0.25 m (marking rule)
PICKUP_HOLD_S = 2.0

# Drive tuning (ticks/s) — same units as your repo
DEFAULT_WHEEL_TICK = 30  # safe speed for both turning and straight

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    th: float = 0.0  # radians


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def px_to_world(u: int, v: int):
    x0, y0 = MAP_ORIGIN_PX
    if not (x0 <= u <= x0 + PANEL_W and y0 <= v <= y0 + PANEL_H):
        raise ValueError("Click outside map panel")
    su = (u - x0) / PANEL_W
    sv = (v - y0) / PANEL_H
    x = WORLD_X_MIN + su * (WORLD_X_MAX - WORLD_X_MIN)
    y = WORLD_Y_MAX - sv * (WORLD_Y_MAX - WORLD_Y_MIN)  # flip y
    return float(x), float(y)


def world_to_px(x: float, y: float):
    x0, y0 = MAP_ORIGIN_PX
    su = (x - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN)
    sv = (WORLD_Y_MAX - y) / (WORLD_Y_MAX - WORLD_Y_MIN)
    u = int(x0 + su * PANEL_W)
    v = int(y0 + sv * PANEL_H)
    return u, v


def draw_map(screen):
    # background & border
    pygame.draw.rect(screen, (30, 30, 30), (*MAP_ORIGIN_PX, PANEL_W, PANEL_H))
    pygame.draw.rect(screen, (90, 90, 90), (*MAP_ORIGIN_PX, PANEL_W, PANEL_H), 2)
    # grid @ 0.5 m
    for gx in np.linspace(WORLD_X_MIN, WORLD_X_MAX, 6):
        u, _ = world_to_px(gx, 0.0)
        pygame.draw.line(screen, (55, 55, 55), (u, MAP_ORIGIN_PX[1]), (u, MAP_ORIGIN_PX[1] + PANEL_H))
    for gy in np.linspace(WORLD_Y_MIN, WORLD_Y_MAX, 6):
        _, v = world_to_px(0.0, gy)
        pygame.draw.line(screen, (55, 55, 55), (MAP_ORIGIN_PX[0], v), (MAP_ORIGIN_PX[0] + PANEL_W, v))


def draw_robot(screen, pose: Pose):
    u, v = world_to_px(pose.x, pose.y)
    pygame.draw.circle(screen, (0, 180, 255), (u, v), 8)
    hx = u + int(14 * math.cos(pose.th))
    hy = v - int(14 * math.sin(pose.th))
    pygame.draw.line(screen, (0, 255, 180), (u, v), (hx, hy), 2)


def draw_goal(screen, goal_xy):
    if goal_xy is None:
        return
    u, v = world_to_px(*goal_xy)
    pygame.draw.circle(screen, (255, 200, 0), (u, v), 6)
    # acceptance radius circle
    rad_px = int(GOAL_TOL_M / (WORLD_X_MAX - WORLD_X_MIN) * PANEL_W)
    pygame.draw.circle(screen, (255, 200, 0), (u, v), rad_px, 1)


def draw_status(screen, text):
    font = pygame.font.SysFont("consolas", 18)
    surf = font.render(text, True, (230, 230, 230))
    screen.blit(surf, (PAD, PANEL_H + PAD + 18))


def load_calibration(calib_dir="calibration/param/"):
    """
    Try the standard file names first (scale.txt/baseline.txt).
    Fall back to paramscale.txt/parambaseline.txt if needed (as per your folder listing).
    Returns (scale_scalar, baseline_scalar).
    """
    # primary filenames (match operate.py)
    scale_path = os.path.join(calib_dir, "scale.txt")
    base_path = os.path.join(calib_dir, "baseline.txt")

    if not os.path.isfile(scale_path):
        alt = os.path.join(calib_dir, "paramscale.txt")
        if os.path.isfile(alt):
            scale_path = alt
    if not os.path.isfile(base_path):
        alt = os.path.join(calib_dir, "parambaseline.txt")
        if os.path.isfile(alt):
            base_path = alt

    scale_arr = np.loadtxt(scale_path, delimiter=",")
    # if two values (L/R) are stored, take mean as your other code does
    scale = float(np.mean(scale_arr))
    baseline = float(np.squeeze(np.loadtxt(base_path, delimiter=",")))
    return scale, baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", type=str, default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--calib_dir", type=str, default="calibration/param/")
    ap.add_argument("--tick", type=int, default=DEFAULT_WHEEL_TICK,
                    help="wheel ticks per second for both turning and straight")
    args = ap.parse_args()

    # pygame init
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Level 1: Click-to-Go (PenguinPi)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18)

    # Robot connection (or sim)
    use_robot = args.ip.lower() != "localhost"
    ppi = None
    if use_robot:
        try:
            ppi = PenguinPi(args.ip, args.port)
            status = f"Connected to PenguinPi @ {args.ip}:{args.port}"
        except Exception as e:
            use_robot = False
            status = f"Failed to connect ({e}). Falling back to SIM."
    else:
        status = "SIM mode (localhost) — no hardware commands sent."

    # Calibration
    try:
        scale, baseline = load_calibration(args.calib_dir)
    except Exception as e:
        pygame.quit()
        raise SystemExit(f"Failed to load calibration from {args.calib_dir} ({e})")

    # Internal pose estimate (for planning only)
    pose = Pose(0.0, 0.0, 0.0)
    goal_xy = None
    busy = False  # true while issuing timed commands
    last_msg = ""

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # emergency stop: send zero velocity and clear goal
                    if use_robot and ppi is not None:
                        try:
                            ppi.set_velocity([0, 0])  # immediate stop
                        except Exception:
                            pass
                    goal_xy = None
                    busy = False
                    last_msg = "Emergency stop."
                elif event.key == pygame.K_r:
                    pose = Pose(0.0, 0.0, 0.0)
                    last_msg = "Pose reset to (0,0,0)."

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not busy:
                u, v = event.pos
                try:
                    gx, gy = px_to_world(u, v)
                    goal_xy = (gx, gy)
                    last_msg = f"New goal set: ({gx:+.2f}, {gy:+.2f})"
                except ValueError:
                    pass  # click outside map

        # If we have a goal and we're not already executing a move, plan/execute:
        if goal_xy is not None and not busy:
            dx = goal_xy[0] - pose.x
            dy = goal_xy[1] - pose.y
            rho = math.hypot(dx, dy)

            if rho < GOAL_TOL_M:
                # Pickup hold (no movement) — just visual feedback
                last_msg = f"Reached (within {GOAL_TOL_M:.2f} m). Holding {PICKUP_HOLD_S:.0f}s..."
                busy = True
                hold_t0 = time.time()
                # Busy-wait + draw UI during hold
                while time.time() - hold_t0 < PICKUP_HOLD_S:
                    screen.fill((10, 10, 12))
                    draw_map(screen)
                    draw_goal(screen, goal_xy)
                    draw_robot(screen, pose)
                    draw_status(screen, f"{status} | {last_msg}")
                    pygame.display.flip()
                    clock.tick(60)
                # Done
                goal_xy = None
                busy = False
                last_msg = "Pickup hold complete."
            else:
                # Compute heading to goal, then turn time & drive time using your calibration
                heading = wrap_to_pi(math.atan2(dy, dx) - pose.th)
                turn_dir = 1 if heading >= 0 else -1
                wheel_tick = args.tick

                # times derived from your auto_fruit_search.py pattern:
                # drive_time = distance / (wheel_tick * scale)
                # turn_time  = (2 * heading * scale * wheel_tick) / baseline
                drive_time = rho / (wheel_tick * scale)
                turn_time = (2.0 * abs(heading) * scale * wheel_tick) / baseline

                # Issue timed commands to the robot (or simulate)
                busy = True
                last_msg = f"Turning {math.degrees(heading):+.1f}° for {turn_time:.2f}s..."
                # Turn:
                if use_robot and ppi is not None:
                    ppi.set_velocity([0, turn_dir], turning_tick=wheel_tick, time=turn_time)
                else:
                    # SIM: update internal pose over the turn duration
                    pose.th = wrap_to_pi(pose.th + (turn_dir * (baseline / (2.0 * scale)) / wheel_tick) * (turn_time / (turn_time)) * heading)

                # Update internal pose orientation exactly
                pose.th = wrap_to_pi(pose.th + heading)

                # Drive:
                last_msg = f"Driving {rho:.2f} m for {drive_time:.2f}s..."
                if use_robot and ppi is not None:
                    ppi.set_velocity([1, 0], tick=wheel_tick, time=drive_time)
                else:
                    # SIM straight-line update
                    pose.x += rho * math.cos(pose.th)
                    pose.y += rho * math.sin(pose.th)

                # Arrived (we consider we’ve reached the exact goal for the simple demo)
                pose.x, pose.y = goal_xy
                last_msg = f"Arrived @ ({pose.x:+.2f}, {pose.y:+.2f}). Holding {PICKUP_HOLD_S:.0f}s..."
                # Pickup hold
                hold_t0 = time.time()
                while time.time() - hold_t0 < PICKUP_HOLD_S:
                    screen.fill((10, 10, 12))
                    draw_map(screen)
                    draw_goal(screen, goal_xy)
                    draw_robot(screen, pose)
                    draw_status(screen, f"{status} | {last_msg}")
                    pygame.display.flip()
                    clock.tick(60)

                goal_xy = None
                busy = False
                last_msg = "Pickup hold complete."

        # Draw
        screen.fill((10, 10, 12))
        draw_map(screen)
        draw_goal(screen, goal_xy)
        draw_robot(screen, pose)
        draw_status(screen, f"{status} | {last_msg}")
        pygame.display.flip()
        clock.tick(60)

    # On exit, try to stop
    try:
        if use_robot and ppi is not None:
            ppi.set_velocity([0, 0])
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()
