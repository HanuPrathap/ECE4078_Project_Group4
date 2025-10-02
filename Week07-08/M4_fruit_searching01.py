# M4 - Autonomous fruit searching (fixed teleop)

import sys, os, cv2, numpy as np, json, argparse, time, pygame
sys.path.insert(0, "slam")
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure
import util.DatasetHandler as dh


def read_true_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list, fruit_true_pos = [], []
        aruco_true_pos = np.empty([10, 2])
        for key in gt_dict:
            x, y = np.round(gt_dict[key]['x'], 1), np.round(gt_dict[key]['y'], 1)
            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9] = [x, y]
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id] = [x, y]
            else:
                fruit_list.append(key[:-2])
                fruit_true_pos = np.vstack([fruit_true_pos, [x, y]]) if len(fruit_true_pos) else np.array([[x, y]])
        return fruit_list, fruit_true_pos, aruco_true_pos


def init_ekf(datadir, ip):
    K = np.loadtxt("calibration/param/intrinsic.txt", delimiter=',')
    D = np.loadtxt("calibration/param/distCoeffs.txt", delimiter=',')
    S = np.loadtxt("calibration/param/scale.txt", delimiter=',')
    if ip == 'localhost':
        S /= 2
    B = np.loadtxt("calibration/param/baseline.txt", delimiter=',')
    return EKF(Robot(B, S, K, D))


def load_map_to_ekf(ekf, aruco_true_pos):
    ekf.markers = aruco_true_pos.T
    ekf.taglist = list(range(1, 11))
    n = ekf.number_landmarks()
    if n > 0:
        total_size = 3 + 2 * n
        new_P = np.zeros((total_size, total_size))
        new_P[0:3, 0:3] = ekf.P[0:3, 0:3]
        new_P[3:, 3:] = 0.001 * np.eye(2 * n)
        ekf.P = new_P
    print(f"Loaded {n} known landmarks")


class LocalizationSystem:
    def __init__(self, args, aruco_true_pos):
        self.ppi = PenguinPi(args.ip, args.port)
        self.ekf = init_ekf(args.calib_dir, args.ip)
        load_map_to_ekf(self.ekf, aruco_true_pos)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)
        self.ekf_on = True
        self.last_print_time = time.time()
        self.print_interval = 2.0
        self.control_clock = time.time()
        self.command = [0, 0]   # [fwd, turn]

    def get_robot_pose(self):
        return self.ekf.robot.state.flatten()

    def print_robot_pose(self):
        x, y, th = self.get_robot_pose()
        print(f"Robot Pose: x={x:.3f}m, y={y:.3f}m, θ={np.degrees(th):.1f}°")

    def take_pic(self):
        return self.ppi.get_image()

    def update_localization(self, img):
        measurements, _ = self.aruco_det.detect_marker_positions(img)
        if self.ekf_on and measurements:
            self.ekf.update(measurements)

    def control(self, args):
        fwd, turn = self.command
        # add ticks so robot actually moves
        if fwd != 0:
            lv, rv = self.ppi.set_velocity([fwd, 0], tick=40)
        elif turn != 0:
            lv, rv = self.ppi.set_velocity([0, turn], turning_tick=40)
        else:
            lv, rv = self.ppi.set_velocity([0, 0])

        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, -rv if args.ip != 'localhost' else rv, dt)
        self.control_clock = time.time()
        return drive_meas

    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: self.command = [1, 0]
                elif event.key == pygame.K_DOWN: self.command = [-1, 0]
                elif event.key == pygame.K_LEFT: self.command = [0, 1]
                elif event.key == pygame.K_RIGHT: self.command = [0, -1]
                elif event.key == pygame.K_SPACE: self.command = [0, 0]
                elif event.key == pygame.K_ESCAPE: return True
            elif event.type == pygame.QUIT:
                return True
        return False


def main_loop():
    parser = argparse.ArgumentParser("Fruit searching with localization")
    parser.add_argument("--map", type=str, default='M4_true_map_full.txt')
    parser.add_argument("--ip", type=str, default='192.168.50.1')
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--mode", type=str, choices=['teleop', 'auto'], default='teleop')
    args, _ = parser.parse_known_args()

    pygame.init()
    pygame.display.set_mode((300, 200))
    clock = pygame.time.Clock()

    _, _, aruco_true_pos = read_true_map(args.map)
    loc_system = LocalizationSystem(args, aruco_true_pos)

    print("Ready! Use arrow keys, SPACE=stop, ESC=quit")

    running = True
    while running:
        if loc_system.update_keyboard():
            running = False
            break

        img = loc_system.take_pic()
        drive_meas = loc_system.control(args)
        if loc_system.ekf_on:
            loc_system.ekf.predict(drive_meas)
        loc_system.update_localization(img)

        if time.time() - loc_system.last_print_time >= loc_system.print_interval:
            loc_system.print_robot_pose()
            loc_system.last_print_time = time.time()

        clock.tick(30)  # 30 Hz loop

    loc_system.ppi.set_velocity([0, 0])
    pygame.quit()
    print("Shut down")


if __name__ == "__main__":
    main_loop()
