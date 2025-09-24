# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import math
import pygame
import shutil

# import SLAM components for GUI rendering only
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure
import util.DatasetHandler as dh


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for"""
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

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
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()
        for fruit in fruits:
            search_list.append(fruit.strip())
    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order"""
    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):  # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(
                    n_fruit,
                    fruit,
                    np.round(fruit_true_pos[i][0], 1),
                    np.round(fruit_true_pos[i][1], 1)
                ))
        n_fruit += 1


class AutoOperate:
    """Autonomous waypointing, GUI intact, SLAM panel replaced by truth map."""

    def __init__(self, args):
        self.args = args
        self.ip = args.ip
        self.play_data = args.play_data

        # dataset folder like operate.py
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # robot or dataset
        if self.play_data:
            self.pibot = dh.DatasetPlayer("record")
            self.data = dh.DatasetWriter('record') if args.save_data else None
        else:
            self.pibot = PenguinPi(args.ip, args.port)
            self.data = None

        # SLAM objects for GUI only
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)

        # gui state
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.yolo_vis = self.aruco_img.copy()
        self.notification = 'Autonomous mode'
        self.pred_notifier = False
        self.count_down = 300
        self.start_time = time.time()

        # control state
        self.command = {'motion': [0, 0],
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.control_clock = time.time()
        self.quit = False
        self.image_id = 0
        self.double_reset_comfirm = 0

        # auto nav params
        self.turn_scale = float(args.turn_scale)
        self.drive_scale = float(args.drive_scale)
        self.turn_tick = int(args.turn_tick)
        self.drive_tick = int(args.drive_tick)
        self.stop_standoff = float(args.stop_standoff)

        # odom pose [x, y, theta]
        self.robot_pose = np.array([0.0, 0.0, 0.0])

        # motion scheduler
        self.active_segment = None
        self.segment_queue = []

        # bounds and output
        self.bounds = tuple(map(float, args.bounds.split(',')))  # xmin,ymin,xmax,ymax
        self.output = dh.OutputWriter('lab_output')

        # always show panel
        self.ekf_on = True

        # truth map data containers
        self.fruits_list = None
        self.fruits_true_pos = None
        self.aruco_true_pos = None

    def set_truth_map(self, fruits_list, fruits_true_pos, aruco_true_pos):
        self.fruits_list = fruits_list
        self.fruits_true_pos = fruits_true_pos
        self.aruco_true_pos = aruco_true_pos

    # wheel control wrapper that also produces a Drive measurement for odom update
    def control(self):
        if self.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(self.command['motion'])
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # sim vs real polarity
        if self.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas

    def take_pic(self):
        self.img = self.pibot.get_image()
        if self.data is not None:
            self.data.write_image(self.img)
        # ArUco annotate each frame for the Detector panel
        _, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        self.yolo_vis = self.aruco_img.copy()

    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False

    def draw_truth_map(self, size=(320, 520)):
        # size matches the old SLAM panel height (480 + v_pad)
        w, h = size
        img = np.ones((h, w, 3), dtype=np.uint8) * 20  # dark background

        # world bounds
        xmin, ymin, xmax, ymax = self.bounds
        rx = xmax - xmin
        ry = ymax - ymin

        def to_px(x, y):
            u = int((x - xmin) / rx * (w - 1))
            v = int(h - 1 - (y - ymin) / ry * (h - 1))
            return u, v

        # grid every 0.3 m
        step = 0.3
        xg = np.arange(np.ceil(xmin / step) * step, xmax + 1e-6, step)
        yg = np.arange(np.ceil(ymin / step) * step, ymax + 1e-6, step)
        for x in xg:
            u0, v0 = to_px(x, ymin)
            u1, v1 = to_px(x, ymax)
            cv2.line(img, (u0, v0), (u1, v1), (50, 50, 50), 1)
        for y in yg:
            u0, v0 = to_px(xmin, y)
            u1, v1 = to_px(xmax, y)
            cv2.line(img, (u0, v0), (u1, v1), (50, 50, 50), 1)

        # border
        p0 = to_px(xmin, ymin)
        p1 = to_px(xmax, ymax)
        cv2.rectangle(img, p0, p1, (100, 100, 100), 2)

        # draw ArUco positions
        if self.aruco_true_pos is not None:
            for i in range(self.aruco_true_pos.shape[0]):
                x, y = self.aruco_true_pos[i]
                if np.isnan(x) or np.isnan(y):
                    continue
                u, v = to_px(x, y)
                cv2.circle(img, (u, v), 6, (0, 180, 255), -1)
                label = "aruco{}".format(10 if i == 9 else i + 1)
                cv2.putText(img, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        # draw fruit waypoints
        if self.fruits_list is not None and self.fruits_true_pos is not None:
            for name, (x, y) in zip(self.fruits_list, self.fruits_true_pos):
                u, v = to_px(x, y)
                cv2.circle(img, (u, v), 5, (70, 220, 70), -1)
                cv2.putText(img, name, (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

        # draw robot pose and heading
        x, y, th = self.robot_pose
        u, v = to_px(x, y)
        L = 30  # arrow length in px
        u2 = int(u + L * np.cos(th))
        v2 = int(v - L * np.sin(th))
        cv2.arrowedLine(img, (u, v), (u2, v2), (255, 200, 0), 2, tipLength=0.25)
        cv2.circle(img, (u, v), 4, (255, 255, 0), -1)

        return img

    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # Right panel shows truth map instead of EKF state
        map_img = self.draw_truth_map(size=(320, 480 + v_pad))
        self.draw_pygame_window(canvas, map_img, position=(2 * h_pad + 320, v_pad))

        # PiBot cam shows the live camera frames
        src = self.img if self.img is not None else np.zeros((240, 320, 3), dtype=np.uint8)
        robot_view = cv2.resize(src, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # Detector view shows ArUco overlays and IDs
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240 + 2 * v_pad))

        self.put_caption(canvas, caption='Truth Map', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption, False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    # keyboard for emergency only
    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [1, 0]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-1, 0]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 1]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -1]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    # ------------- Autonomous navigation helpers -------------

    def enqueue_turn_then_drive(self, waypoint):
        """Create a pair of motion segments to turn toward a waypoint then drive straight"""
        dx = waypoint[0] - self.robot_pose[0]
        dy = waypoint[1] - self.robot_pose[1]
        target_heading = math.atan2(dy, dx)
        dtheta = self.wrap_angle(target_heading - self.robot_pose[2])
        distance = math.hypot(dx, dy)

        # read calibration values
        scale = float(np.loadtxt("calibration/param/scale.txt", delimiter=','))
        baseline = float(np.loadtxt("calibration/param/baseline.txt", delimiter=','))

        # yaw rate and forward speed
        rad_per_s = scale * self.turn_tick * 2.0 / baseline
        ideal_turn_time = abs(dtheta) / max(rad_per_s, 1e-6)
        turn_time = ideal_turn_time * self.turn_scale

        m_per_s = scale * self.drive_tick
        drive_dist = max(0.0, distance - self.stop_standoff)
        ideal_drive_time = drive_dist / max(m_per_s, 1e-6)
        drive_time = ideal_drive_time * self.drive_scale

        # queue segments
        turn_dir = 1.0 if dtheta > 0 else -1.0
        self.segment_queue.append({
            'kind': 'turn',
            't_left': turn_time,
            'v_cmd': [0, 1 if turn_dir > 0 else -1],
            'tick': self.turn_tick
        })
        self.segment_queue.append({
            'kind': 'drive',
            't_left': drive_time,
            'v_cmd': [1, 0],
            'tick': self.drive_tick
        })

        self.notification = f'Nav: turn {dtheta:+.2f} rad, then drive {drive_dist:.2f} m'

    def step_motion(self, dt_max=0.05):
        """Execute queued motion in small time slices so GUI stays responsive. Returns True while moving."""
        if self.active_segment is None:
            if not self.segment_queue:
                self.command['motion'] = [0, 0]
                return False
            self.active_segment = self.segment_queue.pop(0)

        seg = self.active_segment
        if seg['t_left'] <= 0:
            self.active_segment = None
            self.command['motion'] = [0, 0]
            return len(self.segment_queue) > 0

        dt = min(dt_max, seg['t_left'])
        if seg['kind'] == 'turn':
            self.pibot.set_velocity([0, seg['v_cmd'][1]], turning_tick=seg['tick'], time=dt)
            self.update_odom_turn(dt, seg['tick'], seg['v_cmd'][1])
        else:
            self.pibot.set_velocity([1, 0], tick=seg['tick'], time=dt)
            self.update_odom_drive(dt, seg['tick'])

        seg['t_left'] -= dt
        return True

    def update_odom_drive(self, dt, tick):
        scale = float(np.loadtxt("calibration/param/scale.txt", delimiter=','))
        v = scale * tick  # m per s
        ds = v * dt
        self.robot_pose[0] += ds * math.cos(self.robot_pose[2])
        self.robot_pose[1] += ds * math.sin(self.robot_pose[2])

    def update_odom_turn(self, dt, tick, turn_sign):
        scale = float(np.loadtxt("calibration/param/scale.txt", delimiter=','))
        baseline = float(np.loadtxt("calibration/param/baseline.txt", delimiter=','))
        yaw_rate = scale * tick * 2.0 / baseline * (1.0 if turn_sign > 0 else -1.0)
        self.robot_pose[2] = self.wrap_angle(self.robot_pose[2] + yaw_rate * dt)

    @staticmethod
    def wrap_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a


# -------------- main --------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M3_prac_map_full.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    # timing knobs
    parser.add_argument("--turn_scale", type=float, default=0.80)
    parser.add_argument("--drive_scale", type=float, default=1.00)
    parser.add_argument("--turn_tick", type=int, default=30)
    parser.add_argument("--drive_tick", type=int, default=30)
    parser.add_argument("--stop_standoff", type=float, default=0.15)
    parser.add_argument("--bounds", type=str, default="-1.5,-1.5,1.5,1.5")
    args, _ = parser.parse_known_args()

    # pygame setup from operate.py
    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    # splash screen like original
    start = False
    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = AutoOperate(args)

    # read map and search list
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # store truth map for drawing
    operate.set_truth_map(fruits_list, fruits_true_pos, aruco_true_pos)

    # build waypoints in the given order
    waypoints = []
    for fruit in search_list:
        idxs = [i for i, name in enumerate(fruits_list) if name == fruit]
        if idxs:
            fx, fy = float(fruits_true_pos[idxs[0]][0]), float(fruits_true_pos[idxs[0]][1])
            waypoints.append([fx, fy])

    # enqueue motions
    for wp in waypoints:
        operate.enqueue_turn_then_drive(wp)

    # main loop
    while True:
        operate.update_keyboard()
        operate.take_pic()

        moving = operate.step_motion(dt_max=0.06)

        operate.draw(canvas)
        pygame.display.update()

        time.sleep(0.01)

        if not moving and operate.command['motion'] == [0, 0] and len(operate.segment_queue) == 0 and operate.active_segment is None:
            operate.pibot.set_velocity([0, 0])
            break
