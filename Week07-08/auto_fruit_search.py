# M4 - Autonomous fruit searching

import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import math
import pygame
import shutil

# SLAM components (we use EKF pose for navigation)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# util
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure
import util.DatasetHandler as dh


def read_true_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)
            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9] = [x, y]
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id] = [x, y]
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)
        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    search_list = []
    with open('search_list.txt', 'r') as fd:
        for fruit in fd.readlines():
            search_list.append(fruit.strip())
    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    print("Search order:")
    n = 1
    for name in search_list:
        for i in range(len(fruit_list)):
            if name == fruit_list[i]:
                x, y = np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)
                print("{}) {} at [{}, {}]".format(n, name, x, y))
        n += 1


class AutoOperate:
    """Autonomous waypointing with EKF pose. Truth map replaces SLAM panel. Detector shows ArUco IDs."""


    def seed_ekf_with_truth_markers(self):
        """Use ground-truth ArUco positions so EKF updates only robot pose."""
        if self.aruco_true_pos is None:
            return
        # taglist [1..10], aruco10 is index 9
        self.ekf.taglist = [i + 1 for i in range(self.aruco_true_pos.shape[0])]
        self.ekf.markers = self.aruco_true_pos.T.copy()  # shape (2, N)

        # Anchor landmarks with tiny covariance
        P_robot = self.ekf.P
        n_lm = self.ekf.markers.shape[1]
        Plm = np.eye(2 * n_lm) * 1e-6
        self.ekf.P = np.block([
            [P_robot, np.zeros((P_robot.shape[0], 2 * n_lm))],
            [np.zeros((2 * n_lm, P_robot.shape[0])), Plm]
        ])



    def __init__(self, args):
        self.args = args
        self.ip = args.ip
        self.play_data = args.play_data
        self.active_goal = None


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

        # EKF objects
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)

        # GUI state
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.yolo_vis = self.aruco_img.copy()
        self.notification = 'Autonomous mode (EKF driven)'
        self.count_down = 300
        self.start_time = time.time()
        self.quit = False

        # controls
        self.command = {'motion': [0, 0],
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.control_clock = time.time()
        self.image_id = 0

        # nav params
        self.turn_scale = float(args.turn_scale)
        self.drive_scale = float(args.drive_scale)
        self.turn_tick = int(args.turn_tick)
        self.drive_tick = int(args.drive_tick)
        self.stop_standoff = float(args.stop_standoff)
        self.bounds = tuple(map(float, args.bounds.split(',')))  # xmin,ymin,xmax,ymax
        self.dwell = float(args.dwell)


        # motion scheduler
        self.active_segment = None
        self.segment_queue = []
        self.waypoints = []
        self.wp_idx = 0

        # outputs
        self.output = dh.OutputWriter('lab_output')

        # truth map containers
        self.fruits_list = None
        self.fruits_true_pos = None
        self.aruco_true_pos = None

    def set_truth_map(self, fruits_list, fruits_true_pos, aruco_true_pos):
        self.fruits_list = fruits_list
        self.fruits_true_pos = fruits_true_pos
        self.aruco_true_pos = aruco_true_pos

    # EKF and calib
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

        # IO
    def take_pic(self):
        # grab frame
        self.img = self.pibot.get_image()
        if self.data is not None:
            self.data.write_image(self.img)

        # detect ArUco markers on this frame
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        # show detections in the Detector panel
        self.yolo_vis = self.aruco_img.copy()

        # only use known tag IDs (1..10) anchored from the truth map
        if self.ekf.taglist:
            lms = [lm for lm in lms if lm.tag in self.ekf.taglist]

        # update EKF pose using the detections
        self.ekf.update(lms)


    def save_image(self):
        if self.command['save_image']:
            f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
            image = cv2.cvtColor(self.pibot.get_image(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = '{} is saved'.format(f_)

    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False

    # GUI
    def draw_truth_map(self, size=(320, 520)):
        w, h = size
        img = np.ones((h, w, 3), dtype=np.uint8) * 20
        xmin, ymin, xmax, ymax = self.bounds
        rx = xmax - xmin
        ry = ymax - ymin

        def to_px(x, y):
            u = int((x - xmin) / rx * (w - 1))
            v = int(h - 1 - (y - ymin) / ry * (h - 1))
            return u, v

        # grid
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
        cv2.rectangle(img, to_px(xmin, ymin), to_px(xmax, ymax), (100, 100, 100), 2)

        # ArUco markers
        if self.aruco_true_pos is not None:
            for i in range(self.aruco_true_pos.shape[0]):
                x, y = self.aruco_true_pos[i]
                if np.isnan(x) or np.isnan(y):
                    continue
                u, v = to_px(x, y)
                cv2.circle(img, (u, v), 6, (0, 180, 255), -1)
                label = "aruco{}".format(10 if i == 9 else i + 1)
                cv2.putText(img, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        # fruits
        if self.fruits_list is not None and self.fruits_true_pos is not None:
            for name, (x, y) in zip(self.fruits_list, self.fruits_true_pos):
                u, v = to_px(x, y)
                cv2.circle(img, (u, v), 5, (70, 220, 70), -1)
                cv2.putText(img, name, (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

        # robot pose from EKF
        st = self.ekf.robot.state
        x, y = float(st[0, 0]), float(st[1, 0])
        u, v = to_px(x, y)
        cv2.circle(img, (u, v), 6, (255, 255, 0), -1)


        # # current target line
        # if 0 <= self.wp_idx < len(self.waypoints):
        #     gx, gy = self.waypoints[self.wp_idx]
        #     ug, vg = to_px(gx, gy)
        #     cv2.circle(img, (ug, vg), 6, (0, 255, 0), 2)
        #     cv2.line(img, (u, v), (ug, vg), (120, 160, 255), 1)

        return img

    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # Right panel: truth map
        #map_img = self.draw_truth_map(size=(320, 480 + v_pad))
        map_img = self.draw_truth_map(size=(320, 320))
        self.draw_pygame_window(canvas, map_img, position=(2 * h_pad + 320, v_pad))

        # PiBot cam: live camera
        src = self.img if self.img is not None else np.zeros((240, 320, 3), dtype=np.uint8)
        robot_view = cv2.resize(src, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # Detector: ArUco overlays
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240 + 2 * v_pad))

        self.put_caption(canvas, caption='Truth Map', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        note = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(note, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            txt = 'Count Down: {:03.0f}s'.format(time_remain)
        elif int(time_remain) % 2 == 0:
            txt = "Time Is Up !!!"
        else:
            txt = ""
        count_down_surface = TEXT_FONT.render(txt, False, (50, 50, 50))
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

    # ---------- Autonomous navigation (EKF pose) ----------

    def set_waypoints(self, waypoints):
        self.waypoints = list(waypoints)
        self.wp_idx = 0

    def current_pose(self):
        st = self.ekf.robot.state
        return float(st[0, 0]), float(st[1, 0]), float(st[2, 0])

    def plan_if_idle(self):
        """If not moving and waypoints remain, plan a fresh turn-then-drive from current EKF pose."""
        if self.active_segment is not None or self.segment_queue:
            return
        if self.wp_idx >= len(self.waypoints):
            return

        # if already close, skip
        x, y, _ = self.current_pose()
        gx, gy = self.waypoints[self.wp_idx]
        dist = math.hypot(gx - x, gy - y)
        if dist <= self.stop_standoff * 0.9:
            self.wp_idx += 1
            return
        
        # terminal log
        name = None
        if self.fruits_list is not None and self.fruits_true_pos is not None:
            arr = np.asarray(self.fruits_true_pos)
            idx = int(np.argmin(np.hypot(arr[:,0] - gx, arr[:,1] - gy)))
            # if exact match in the truth map, use that fruit name
            if abs(arr[idx,0] - gx) < 1e-6 and abs(arr[idx,1] - gy) < 1e-6:
                name = self.fruits_list[idx]
        label = name if name is not None else 'target'
        print(f"Going to: [{label}, {gx:.2f}, {gy:.2f}]")


        self.enqueue_turn_then_drive([gx, gy])
        self.notification = 'Target {} at [{:.2f}, {:.2f}]'.format(self.wp_idx + 1, gx, gy)

    def enqueue_turn_then_drive(self, waypoint):
        """Create two segments using current EKF pose."""
        x, y, th = self.current_pose()
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        target_heading = math.atan2(dy, dx)
        dtheta = self.wrap_angle(target_heading - th)
        distance = math.hypot(dx, dy)

        scale = float(np.loadtxt("calibration/param/scale.txt", delimiter=','))
        baseline = float(np.loadtxt("calibration/param/baseline.txt", delimiter=','))

        # yaw rate and forward speed from ticks
        rad_per_s = max(1e-6, scale * self.turn_tick * 2.0 / baseline)
        turn_time = abs(dtheta) / rad_per_s * self.turn_scale

        m_per_s = max(1e-6, scale * self.drive_tick)
        drive_dist = max(0.0, distance - self.stop_standoff)
        drive_time = drive_dist / m_per_s * self.drive_scale

        turn_sign = 1 if dtheta > 0 else -1  # +1 means CCW

        self.segment_queue.append({
            'kind': 'turn',
            't_left': turn_time,
            'turn_sign': turn_sign,
            'tick': self.turn_tick
        })
        self.segment_queue.append({
            'kind': 'drive',
            't_left': drive_time,
            'tick': self.drive_tick
        })
        # pause at the fruit
        if drive_time > 0:
            self.segment_queue.append({
                'kind': 'wait',
                't_left': self.dwell
            })

    


    def step_motion(self, dt_max=0.10):
        if self.active_segment is None:
            if not self.segment_queue:
                self.command['motion'] = [0, 0]
                return False
            self.active_segment = self.segment_queue.pop(0)

        seg = self.active_segment
        if seg['t_left'] <= 0:
            self.active_segment = None
            self.command['motion'] = [0, 0]
            # advance waypoint when the last segment for it finishes
            if not self.segment_queue:
                self.wp_idx = min(self.wp_idx + 1, len(self.waypoints))
            return len(self.segment_queue) > 0

        dt = min(dt_max, seg['t_left'])

        if seg['kind'] == 'turn':
            lv, rv = self.pibot.set_velocity(
                [0, 1 if seg['turn_sign'] > 0 else -1],
                turning_tick=seg['tick'], time=dt
            )
            drive = measure.Drive(lv, rv, dt) if self.ip == 'localhost' else measure.Drive(lv, -rv, dt)
            self.ekf.predict(drive)


        elif seg['kind'] == 'drive':
            lv, rv = self.pibot.set_velocity([1, 0], tick=seg['tick'], time=dt)
            drive = measure.Drive(lv, rv, dt) if self.ip == 'localhost' else measure.Drive(lv, -rv, dt)
            self.ekf.predict(drive)


        elif seg['kind'] == 'wait':
            # show countdown while dwelling at the target
            self.notification = f'Holding at target {self.wp_idx + 1} for {seg["t_left"]:.1f}s'
            self.pibot.set_velocity([0, 0])
            time.sleep(dt)  # no EKF predict; robot is still



        seg['t_left'] -= dt
        return True


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
    parser.add_argument("--dwell", type=float, default=3.0)  # seconds to pause at each fruit

    # faster defaults
    parser.add_argument("--turn_scale", type=float, default=0.70)
    parser.add_argument("--drive_scale", type=float, default=0.90)
    parser.add_argument("--turn_tick", type=int, default=70)
    parser.add_argument("--drive_tick", type=int, default=50)
    parser.add_argument("--stop_standoff", type=float, default=0.15)
    parser.add_argument("--bounds", type=str, default="-1.2, -1.2, 1.2, 1.2")
    args, _ = parser.parse_known_args()

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

    # splash
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

    # truth map and search list
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    operate.set_truth_map(fruits_list, fruits_true_pos, aruco_true_pos)
    operate.seed_ekf_with_truth_markers()


    # build waypoints in search order (first match for each label)
    waypoints = []
    for fruit in search_list:
        idxs = [i for i, name in enumerate(fruits_list) if name == fruit]
        if idxs:
            fx, fy = float(fruits_true_pos[idxs[0]][0]), float(fruits_true_pos[idxs[0]][1])
            waypoints.append([fx, fy])
    operate.set_waypoints(waypoints)

    # main loop
    while True:
        operate.update_keyboard()
        operate.take_pic()

        operate.plan_if_idle()
        moving = operate.step_motion(dt_max=0.10)

        operate.draw(canvas)
        pygame.display.update()

        time.sleep(0.005)

        # exit when all waypoints done and idle
        if operate.wp_idx >= len(operate.waypoints) and not moving and operate.active_segment is None and not operate.segment_queue:
            operate.pibot.set_velocity([0, 0])
            break
