#!/usr/bin/env python3

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.spatial import Delaunay
from scipy.linalg import solve
from matplotlib.patches import Circle
import numpy as np
import random
import time


class PursuitEvaderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.kinematics_integrator = 'euler'
        self.viewer = None
        high = np.array(
            [300.0, 300.0, 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0,
             300.0, 300.0, 300.0])
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.action_space = spaces.Box(low=0, high=2 * np.pi, shape=(1,))
        self.observation_space = spaces.Box(low=low, high=high)
        self.seed()
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state = self.state
        p1_x, p1_y, p1_v, p2_x, p2_y, p2_v, p3_x, p3_y, p3_v, e_x, e_y, e_v, l_1, l_2, l_3, l_4 = state
        p1 = np.array([p1_x, p1_y])
        p2 = np.array([p2_x, p2_y])
        p3 = np.array([p3_x, p3_y])
        e = np.array([e_x, e_y])

        de_x = 1.2 * math.cos(action)
        de_y = 1.2 * math.sin(action)
        # print(de_x, de_y)
        dp1_x, dp1_y, dp2_x, dp2_y, dp3_x, dp3_y = get_pursuit_strategy(e, p1, p2, p3)

        dp = []
        dp.append([dp1_x, dp1_y])
        dp.append([dp2_x, dp2_y])
        dp.append([dp3_x, dp3_y])
        dp = np.array(dp)
        dpv = np.array([0.0, 0.0, 0.0])

        for i in range(0, 3):
            if dp[i][0] > 0 and dp[i][1] >= 0:
                dpv[i] = math.atan(math.tan(dp[i][1] / dp[i][0]))
            elif dp[i][0] > 0 and dp[i][1] < 0:
                dpv[i] = math.atan(math.tan(dp[i][1] / dp[i][0])) + 2 * np.pi
            elif dp[i][0] < 0 and dp[i][1] >= 0:
                dpv[i] = math.atan(math.tan(dp[i][1] / dp[i][0])) + np.pi
            elif dp[i][0] < 0 and dp[i][1] < 0:
                dpv[i] = math.atan(math.tan(dp[i][1] / dp[i][0])) + np.pi
            elif dp[i][0] == 0:
                if dp[i][1] > 0:
                    dpv[i] = 1 / 2 * np.pi
                else:
                    dpv[i] = 3 / 2 * np.pi

        p1[0] = p1[0] + dp1_x
        p1[1] = p1[1] + dp1_y

        if p1[0] < 0:
            p1[0] = 0
        elif p1[0] > 300:
            p1[0] = 300
        if p1[1] < 0:
            p1[1] = 0
        elif p1[1] > 300:
            p1[1] = 300

        p2[0] = p2[0] + dp2_x
        p2[1] = p2[1] + dp2_y

        if p2[0] < 0:
            p2[0] = 0
        elif p2[0] > 300:
            p2[0] = 300
        if p2[1] < 0:
            p2[1] = 0
        elif p2[1] > 300:
            p2[1] = 300

        p3[0] = p3[0] + dp3_x
        p3[1] = p3[1] + dp3_y

        if p3[0] < 0:
            p3[0] = 0
        elif p3[0] > 300:
            p3[0] = 300
        if p3[1] < 0:
            p3[1] = 0
        elif p3[1] > 300:
            p3[1] = 300

        cost = 0

        if e[0] + de_x < 0:
            de_x = -e[0]
        elif e[0] + de_x > 300:
            de_x = 300 - e[0]
        if e[1] + de_y < 0:
            de_y = -e[1]
        elif e[1] + de_y > 300:
            de_y = 300 - e[1]

        e[0] = e[0] + de_x
        e[1] = e[1] + de_y
        cost = np.sqrt(de_x * de_x + de_y * de_y) - 1.2

        if de_x > 0 and de_y >= 0:
            de_v = math.atan(math.tan(de_y / de_x))
        elif de_x > 0 and de_y < 0:
            de_v = math.atan(math.tan(de_y / de_x)) + 2 * np.pi
        elif de_x < 0 and de_y >= 0:
            de_v = math.atan(math.tan(de_y / de_x)) + np.pi
        elif de_x < 0 and de_y < 0:
            de_v = math.atan(math.tan(de_y / de_x)) + np.pi
        elif de_x == 0:
            if de_y > 0:
                de_v = 1 / 2 * np.pi
            else:
                de_v = 3 / 2 * np.pi

        l_1 = e[0]
        l_2 = 300 - e[0]
        l_3 = e[1]
        l_4 = 300 - e[1]

        min_distance = min(np.sqrt(np.sum(np.square(p1 - e))), np.sqrt(np.sum(np.square(p2 - e))),
                           np.sqrt(np.sum(np.square(p3 - e))))

        next_state = np.array(
            [p1[0], p1[1], dpv[0], p2[0], p2[1], dpv[1], p3[0], p3[1], dpv[2], e[0], e[1], de_v, l_1, l_2, l_3, l_4])

        self.state = next_state

        done = np.sqrt(np.sum(np.square(p1 - e))) <= 2 or np.sqrt(np.sum(np.square(p2 - e))) <= 2 or np.sqrt(
            np.sum(np.square(p3 - e))) <= 2
        done = bool(done)

        if done == 0:
            reward = 1.0 + cost + 0.01 * min_distance
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = -10.0 + cost
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = -10.0 + cost
        # print(done)
        return next_state, reward, done, {}

    def reset(self):
        high = np.array(
            [300.0, 300.0, 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0,
             300.0, 300.0, 300.0])
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state = self.np_random.uniform(low=low, high=high)
        self.state[0] = 100
        self.state[1] = 100
        self.state[3] = 100
        self.state[4] = 200
        self.state[6] = 200
        self.state[7] = 200
        self.state[9] = 200
        self.state[10] = 100
        # for i in range(3):
        # 	self.state[i*3] = random.uniform(100.0, 200.0)
        #	self.state[i*3+1] = random.uniform(100.0, 200.0)
        self.state[12] = self.state[9]
        self.state[13] = 300 - self.state[9]
        self.state[14] = self.state[10]
        self.state[15] = 300 - self.state[10]
        self.steps_beyond_done = None

        return np.array(self.state)

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        screen_width = 310
        screen_height = 310
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.line1 = rendering.Line((5, 5), (5, 305))
            self.line2 = rendering.Line((5, 5), (305, 5))
            self.line3 = rendering.Line((305, 5), (305, 305))
            self.line4 = rendering.Line((305, 305), (5, 305))

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)

            self.pursuer1 = rendering.make_circle(1)
            self.p1trans = rendering.Transform()
            self.pursuer1.add_attr(self.p1trans)
            self.pursuer1.set_color(0, 1, 0)

            self.pursuer1_capture = rendering.make_circle(2, 30, filled=False)
            self.p1ctrans = rendering.Transform()
            self.pursuer1_capture.add_attr(self.p1ctrans)
            self.pursuer1_capture.set_color(0, 1, 0)

            self.pursuer2 = rendering.make_circle(1)
            self.p2trans = rendering.Transform()
            self.pursuer2.add_attr(self.p2trans)
            self.pursuer2.set_color(0, 1, 0)

            self.pursuer2_capture = rendering.make_circle(2, 30, filled=False)
            self.p2ctrans = rendering.Transform()
            self.pursuer2_capture.add_attr(self.p2ctrans)
            self.pursuer2_capture.set_color(0, 1, 0)

            self.pursuer3 = rendering.make_circle(1)
            self.p3trans = rendering.Transform()
            self.pursuer3.add_attr(self.p3trans)
            self.pursuer3.set_color(0, 1, 0)

            self.pursuer3_capture = rendering.make_circle(2, 30, filled=False)
            self.p3ctrans = rendering.Transform()
            self.pursuer3_capture.add_attr(self.p3ctrans)
            self.pursuer3_capture.set_color(0, 1, 0)

            self.evader = rendering.make_circle(1)
            self.etrans = rendering.Transform()
            self.evader.add_attr(self.etrans)
            self.evader.set_color(0, 0, 1)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.pursuer1)
            self.viewer.add_geom(self.pursuer1_capture)
            self.viewer.add_geom(self.pursuer2)
            self.viewer.add_geom(self.pursuer2_capture)
            self.viewer.add_geom(self.pursuer3)
            self.viewer.add_geom(self.pursuer3_capture)
            self.viewer.add_geom(self.evader)

        if self.state is None:
            return None

        self.p1trans.set_translation(self.state[0] + 5, self.state[1] + 5)
        self.p1ctrans.set_translation(self.state[0] + 5, self.state[1] + 5)
        self.p2trans.set_translation(self.state[3] + 5, self.state[4] + 5)
        self.p2ctrans.set_translation(self.state[3] + 5, self.state[4] + 5)
        self.p3trans.set_translation(self.state[6] + 5, self.state[7] + 5)
        self.p3ctrans.set_translation(self.state[6] + 5, self.state[7] + 5)
        self.etrans.set_translation(self.state[9] + 5, self.state[10] + 5)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


## ------------------------Function----------------------##
def get_outer_circle(A, B, C):
    # 顶点的坐标
    xa, ya = A[0], A[1]
    xb, yb = B[0], B[1]
    xc, yc = C[0], C[1]
    # 两条边的中点
    xab, yab = (xa + xb) / 2.0, (ya + yb) / 2.0
    xbc, ybc = (xb + xc) / 2.0, (yb + yc) / 2.0

    # 两条边的斜率
    if (xb != xa):
        kab = (yb - ya) / (xb - xa)
    else:
        kab = None

    if (xc != xb):
        kbc = (yc - yb) / (xc - xb)
    else:
        kbc = None

    if (kab != None):
        ab = np.arctan(kab)
    else:
        ab = np.pi / 2

    if (kbc != None):
        bc = np.arctan(kbc)
    else:
        bc = np.pi / 2

    # 两条边的中垂线
    if (ab == 0):
        kabm = None
        b1 = 0
        x = xab
    else:
        kabm = np.tan(ab + np.pi / 2)
        b1 = yab * 1.0 - xab * kabm * 1.0

    if (bc == 0):
        kbcm = None
        b2 = 0
        x = xbc
    else:
        kbcm = np.tan(bc + np.pi / 2)
        b2 = ybc * 1.0 - xbc * kbcm * 1.0

    if (kabm != None and kbcm != None):
        x = (b2 - b1) * 1.0 / (kabm - kbcm)

    if (kabm != None):
        y = kabm * x * 1.0 + b1 * 1.0
    else:
        y = kbcm * x * 1.0 + b2 * 1.0

    r = np.sqrt((x - xa) ** 2 + (y - ya) ** 2)

    return (x, y, r)


def get_intersect_point(a, b, c, bound):
    # 初始化
    flag = 0
    x1 = y1 = x2 = y2 = 0

    if b == 0:
        # 斜率不存在
        x1 = x2 = -c / a
        y1 = bound[2]
        y2 = bound[3]
    else:
        # 斜率存在
        if (-c - a * bound[0]) / b <= bound[3] and (-c - a * bound[0]) / b >= bound[2]:
            if flag == 0:
                x1 = bound[0]
                y1 = (-c - a * bound[0]) / b
                flag = 1
            else:
                x2 = bound[0]
                y2 = (-c - a * bound[0]) / b
                flag = 2

        if (-c - a * bound[1]) / b <= bound[3] and (-c - a * bound[1]) / b >= bound[2]:
            if flag == 0:
                x1 = bound[1]
                y1 = (-c - a * bound[1]) / b
                flag = 1
            else:
                # 找到过符合要求的交点
                x2 = bound[1]
                y2 = (-c - a * bound[1]) / b
                flag = 2

        if a == 0:
            x1 = bound[0]
            x2 = bound[1]
            y1 = -c / b
            y2 = -c / b
            flag = 2
        else:
            if (-c - b * bound[2]) / a <= bound[1] and (-c - b * bound[2]) / a >= bound[0]:
                if flag == 0:
                    y1 = bound[2]
                    x1 = (-c - b * bound[2]) / a
                    flag = 1
                else:
                    y2 = bound[2]
                    x2 = (-c - b * bound[2]) / a
                    flag = 2

            if (-c - b * bound[3]) / a <= bound[1] and (-c - b * bound[3]) / a >= bound[0]:
                if flag == 0:
                    y1 = bound[3]
                    x1 = (-c - b * bound[3]) / a
                    flag = 1
                else:
                    y2 = bound[3]
                    x2 = (-c - b * bound[3]) / a
                    flag = 2
        if flag == 1:
            # 只存在一个交点
            x2 = x1
            y2 = y1
    return flag, x1, y1, x2, y2


def intersect(A, B, bound):
    flag = 0
    C = [0, 0]
    if A[0] >= bound[0] and A[0] <= bound[1] and A[1] >= bound[2] and A[1] <= bound[3]:
        # A点在区域内
        if B[0] >= bound[0] and B[0] <= bound[1] and B[1] >= bound[2] and B[1] <= bound[3]:
            # B点在区域内
            flag = 1;
            return A[0], A[1], B[0], B[1], flag
        else:
            # B点不在区域内
            flag = 1
            if (A[0] == B[0]):
                # AB的斜率不存在
                if (B[1] > bound[3]):
                    x = A[0]
                    y = bound[3]
                else:
                    x = A[0]
                    y = bound[2]
                C[0] = x
                C[1] = y
            else:
                # AB的斜率存在
                a = B[1] - A[1]
                b = A[0] - B[0]
                c = B[0] * A[1] - A[0] * B[1]
                num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if num == 1:
                    flag = 0
                if x1 >= min(A[0], B[0]) and x1 <= max(A[0], B[0]) and y1 >= min(A[1], B[1]) and y1 <= max(A[1], B[1]):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return A[0], A[1], C[0], C[1], flag
    else:
        # A点不在区域内部
        if B[0] >= bound[0] and B[0] <= bound[1] and B[1] >= bound[2] and B[1] <= bound[3]:
            # B点在区域内
            flag = 1
            if (A[0] == B[0]):
                # AB的斜率不存在
                if (A[1] > bound[3]):
                    x = B[0]
                    y = bound[3]
                else:
                    x = B[0]
                    y = bound[2]
                C = [x, y]
            else:
                # AB的斜率存在
                a = B[1] - A[1]
                b = A[0] - B[0]
                c = B[0] * A[1] - A[0] * B[1]
                num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if num == 1:
                    flag = 0
                if x1 >= min(A[0], B[0]) and x1 <= max(A[0], B[0]) and y1 >= min(A[1], B[1]) and y1 <= max(A[1], B[1]):
                    C[0] = x1
                    C[1] = y1
                else:
                    C[0] = x2
                    C[1] = y2
            return B[0], B[1], C[0], C[1], flag
        else:
            flag = 1
            # print("B点不在区域内")
            if (A[0] == B[0]):
                # print("AB的斜率不存在")
                flag = 0
                return A[0], A[1], B[0], B[1], flag
            else:
                a = A[1] - B[1]
                b = B[0] - A[0]
                c = B[1] * A[0] - A[1] * B[0]
                num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
                if num > 0:
                    if num == 1:
                        flag = 0
                    else:
                        flag = 1
                    return x1, y1, x2, y2, flag
                else:
                    return A[0], A[1], B[0], B[1], flag


def IsIntersec(p1, p2, p3, p4):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]
    if (a * p3[0] + b * p3[1] + c) * (a * p4[0] + b * p4[1] + c) <= 0:
        return 1
    else:
        return 0


def midline(A, B, C, bound):
    a = 2 * (B[0] - A[0])
    b = 2 * (B[1] - A[1])
    c = A[0] ** 2 - B[0] ** 2 + A[1] ** 2 - B[1] ** 2
    num, x1, y1, x2, y2 = get_intersect_point(a, b, c, bound)
    D = [x1, y1]
    if IsIntersec(A, B, C, D):
        D = [x1, y1]
    else:
        D = [x2, y2]
    return D


def get_l_point(A, B, C, D):
    # A-交点1 B-交点2 C-evader D-pursuer
    a1 = A[1] - B[1]
    b1 = B[0] - A[0]
    c1 = B[1] * A[0] - B[0] * A[1]
    a2 = C[1] - D[1]
    b2 = D[0] - C[0]
    c2 = D[1] * C[0] - D[0] * C[1]
    d = a1 * b2 - a2 * b1
    if d == 0:
        print(A, B, C, D)
    x1 = (b1 * c2 - b2 * c1) / d
    y1 = (a2 * c1 - a1 * c2) / d
    v1 = [D[0] - A[0], D[1] - A[1]]
    v2 = [D[0] - C[0], D[1] - C[1]]
    if np.cross(v1, v2) > 0:
        x2 = A[0]
        y2 = A[1]
    else:
        x2 = B[0]
        y2 = B[1]
    return x1, y1, x2, y2


def is_collinear(A, B, C, D):
    if A[0] == B[0] and B[0] == C[0] and C[0] == D[0]:
        return True
    elif A[1] == B[1] and B[1] == C[1] and C[1] == D[1]:
        return True
    else:
        if A[0] == B[0]:
            return False
        else:
            a = B[1] - A[1]
            b = A[0] - B[0]
            c = B[0] * A[1] - A[0] * B[1]
            if a * C[0] + b * C[1] + c == 0 and a * D[0] + b * D[1] + c == 0:
                return True
            else:
                return False


def get_pursuit_strategy(A, B, C, D):
    bounding_box = np.array([0., 300., 0., 300.])
    distance = [np.sqrt(np.sum(np.square(A - B))), np.sqrt(np.sum(np.square(A - C))), np.sqrt(np.sum(np.square(A - D)))]
    points = np.array([A, B, C, D])
    if distance[0] == 0 or distance[1] == 0 or distance[2] == 0:
        Ve_ = [[0, 0], [0, 0], [0, 0]]
    else:
        if is_collinear(A, B, C, D):
            Ve_ = []
            for i in range(1, 4):
                Ve_.append((points[0] - points[i]) / distance[i - 1])
        else:
            tri = Delaunay(points)
            circle = []
            tri_lines = []
            for num in range(0, tri.simplices.shape[0]):
                x, y, r = get_outer_circle(points[tri.simplices[num][0]], points[tri.simplices[num][1]],
                                           points[tri.simplices[num][2]])
                circle.append([x, y])
                tri.simplices[num].sort()
                tup = (tri.simplices[num][0], tri.simplices[num][1])
                tri_lines.append(tup)
                tup = (tri.simplices[num][0], tri.simplices[num][2])
                tri_lines.append(tup)
                tup = (tri.simplices[num][1], tri.simplices[num][2])
                tri_lines.append(tup)

            i = 0
            dic = dict()
            for tri_line in tri_lines:
                if tri_lines[i] in dic.keys():
                    dic[tri_lines[i]].append(int(i) // int(3))
                    i = i + 1
                else:
                    dic[tri_lines[i]] = [int(i) // int(3)]
                    i = i + 1

            voronoi_graph = dict()

            for key, value in dic.items():
                if len(value) == 2:
                    x1, y1, x2, y2, flag = intersect(circle[value[0]], circle[value[1]], bounding_box)
                    voronoi_graph[key] = [[x1, y1], [x2, y2], flag]
                else:
                    for i in range(0, 3):
                        if (tri.simplices[value[0]][i] != key[0] and tri.simplices[value[0]][i] != key[1]):
                            peak = [points[tri.simplices[value[0]][i]][0], points[tri.simplices[value[0]][i]][1]]
                            break
                    if circle[value[0]][0] < bounding_box[0] or circle[value[0]][0] > bounding_box[1] or \
                            circle[value[0]][1] < bounding_box[2] or circle[value[0]][1] > bounding_box[3]:
                        x1, y1 = circle[value[0]][0], circle[value[0]][1]
                        x2, y2 = midline(points[key[0]], points[key[1]], peak, bounding_box)
                        flag = 0
                    else:
                        x1, y1, x2, y2, flag = intersect(circle[value[0]],
                                                         midline(points[key[0]], points[key[1]], peak, bounding_box),
                                                         bounding_box)
                    voronoi_graph[key] = [[x1, y1], [x2, y2], flag]

            neighbor = []
            unneighbor = []

            for tri_line in tri_lines:
                if (tri_line[0] == 0 or tri_line[1] == 0):
                    if tri_line[1] + tri_line[0] not in neighbor:
                        if voronoi_graph[tri_line][2] == 1:
                            if voronoi_graph[tri_line][0][0] != voronoi_graph[tri_line][1][0] or \
                                    voronoi_graph[tri_line][0][1] != voronoi_graph[tri_line][1][1]:
                                neighbor.append(tri_line[1] + tri_line[0])

            for i in range(1, 4):
                if i not in neighbor:
                    unneighbor.append(i)

            Dh = []
            Dv = []
            Len_B = []
            l = []
            u_ = []
            e = []
            un_ = []
            Ve_ = [[0, 0], [0, 0], [0, 0]]

            for i in range(0, len(neighbor)):
                square_sum = np.square(
                    voronoi_graph[(0, neighbor[i])][0][0] - voronoi_graph[(0, neighbor[i])][1][0]) + np.square(
                    voronoi_graph[(0, neighbor[i])][0][1] - voronoi_graph[(0, neighbor[i])][1][1])
                Len_B.append(np.sqrt(square_sum))
                Dh.append((-1) * Len_B[i] / 2.)
                x1, y1, x2, y2 = get_l_point(voronoi_graph[(0, neighbor[i])][0], voronoi_graph[(0, neighbor[i])][1],
                                             points[0], points[neighbor[i]])
                square_sum = np.square(x1 - x2) + np.square(y1 - y2)
                l.append(np.sqrt(square_sum))
                e.append(np.sqrt(np.sum(np.square(points[0] - points[neighbor[i]]))))
                Dv.append((np.square(l[i]) - np.square(Len_B[i] - l[i])) / (2 * e[i]))

            Dv = np.array(Dv)
            Dh = np.array(Dh)
            e = np.array(e)

            for i in range(0, len(neighbor)):
                n_h = (points[0] - points[neighbor[i]]) / e[i]
                n_h = np.array(n_h)
                n_v = [n_h[1], -n_h[0]]
                n_v = np.array(n_v)
                u_.append((-1) * (Dh[i] * n_h + Dv[i] * n_v) / (np.sqrt(np.square(Dv[i]) + np.square(Dh[i]))))
                Ve_[neighbor[i] - 1][0] = u_[i][0]
                Ve_[neighbor[i] - 1][1] = u_[i][1]

            for i in range(0, len(unneighbor)):
                # print("ssss")
                un_.append((points[0] - points[unneighbor[i]]) / (
                    np.sqrt(np.sum(np.square(points[0] - points[unneighbor[i]])))))
                Ve_[unneighbor[i] - 1][0] = un_[i][0]
                Ve_[unneighbor[i] - 1][1] = un_[i][1]
                # print(un_[i][0], un_[i][1])

            # print(Ve_[0][0], Ve_[0][1], Ve_[1][0], Ve_[1][1], Ve_[2][0], Ve_[2][1])
            # time.sleep()
    return Ve_[0][0], Ve_[0][1], Ve_[1][0], Ve_[1][1], Ve_[2][0], Ve_[2][1]
