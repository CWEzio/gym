#!/usr/bin/env python3

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class MAPursuerEvaderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.kinematics_integrator = 'euler'
        self.viewer = None
        # Evader Space
        evader_high = np.array(
            [300.0 * (2 ** 0.5), 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0, 300.0, 300.0,
             300.0, (200 ** 2 + 100 ** 2) ** 0.5, 2 * np.pi, 15, (250 ** 2 + 200 ** 2) ** 0.5, 2 * np.pi, 20])
        evader_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20])
        self.evader_action_space = spaces.Box(low=0, high=2 * np.pi, shape=(1,))
        self.evader_observation_space = spaces.Box(low=evader_low, high=evader_high)

        # Pursuers Space
        pursuer_high = np.array([300.0, 300.0, 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi,
                                  300.0, 300.0, 300.0, 300.0, (200 ** 2 + 100 ** 2) ** 0.5, 2 * np.pi, 15,
                                  (250 ** 2 + 200 ** 2) ** 0.5, 2 * np.pi, 20])
        pursuer_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20])
        self.pursuer_action_space = spaces.Box(low=0, high=2 * np.pi, shape=(1,))
        self.pursuer_observation_space = spaces.Box(low=pursuer_low, high=pursuer_high)

        self.state = None
        self.seed()
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Get state data
        # Get pursuers' and evader's params
        e_p1_r, e_p1_angle, e_p2_r, e_p2_angle, e_x, e_y, e_v, e_l1, e_l2, e_l3, e_l4, e_ob1_r, e_ob1_angle, ob1_r, e_ob2_r, e_ob2_angle, ob2_r = \
        self.state[0]
        p1_x, p1_y, p1_v, p1_p2_r, p1_p2_angle, p1_e_r, p1_e_angle, p1_l1, p1_l2, p1_l3, p1_l4, p1_ob1_r, p1_ob1_angle, ob1_r, p1_ob2_r, p1_ob2_angle, ob2_r = \
        self.state[1]
        p2_x, p2_y, p2_v, p2_p1_r, p2_p1_angle, p2_e_r, p2_e_angle, p2_l1, p2_l2, p2_l3, p2_l4, p2_ob1_r, p2_ob1_angle, ob1_r, p2_ob2_r, p2_ob2_angle, ob2_r = \
        self.state[2]

        # Get action
        evader_action = action[0]
        pursuer_action = np.array([action[1], action[2]])

        # Pursuer running
        # Pursuer No.1
        p1 = np.array([p1_x, p1_y])
        dp1_x = 2 * np.cos(pursuer_action[0])
        dp1_y = 2 * np.sin(pursuer_action[0])
        # Pursuer N0.2
        p2 = np.array([p2_x, p2_y])
        dp2_x = 2 * np.cos(pursuer_action[1])
        dp2_y = 2 * np.sin(pursuer_action[1])
        # Evader
        e = np.array([e_x, e_y])
        de_x = 3 * np.cos(evader_action)
        de_y = 3 * np.sin(evader_action)

        # Collision the wall
        # Pursuer No.1
        if p1[0] + dp1_x < 0:
            dp1_x = -p1[0]
        elif p1[0] + dp1_x > 300:
            dp1_x = 300 - p1[0]
        if p1[1] + dp1_y < 0:
            dp1_y = -p1[1]
        elif p1[1] + dp1_y > 300:
            dp1_y = 300 - p1[1]
        p1[0] = p1[0] + dp1_x
        p1[1] = p1[1] + dp1_y
        # Pursuer N0.2
        if p2[0] + dp2_x < 0:
            dp2_x = -p2[0]
        elif p2[0] + dp2_x > 300:
            dp2_x = 300 - p2[0]
        if p2[1] + dp2_y < 0:
            dp2_y = -p2[1]
        elif p2[1] + dp2_y > 300:
            dp2_y = 300 - p2[1]
        p2[0] = p2[0] + dp2_x
        p2[1] = p2[1] + dp2_y
        # Evader
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

        # Collision the obstacle
        ob1 = np.array([200, 100])
        ob2 = np.array([50, 100])
        # Pursuer No.1
        p_new = np.array([p1[0] + dp1_x, p1[1] + dp1_y])
        if np.sqrt(np.sum(np.square(p_new - ob1))) < ob1_r:
            circle = [ob1[0], ob1[1], ob1_r]
            temp = [0, 0]
            temp[0], temp[1] = LineIntersectCircle(circle, p1, p_new)
            p1[0], p1[1] = get_pose(ob1, temp, p_new)
        if np.sqrt(np.sum(np.square(p_new - ob2))) < ob2_r:
            circle = [ob2[0], ob2[1], ob2_r]
            temp = [0, 0]
            temp[0], temp[1] = LineIntersectCircle(circle, p1, p_new)
            p1[0], p1[1] = get_pose(ob2, temp, p_new)
        # Pursuer No.2
        p_new = np.array([p2[0] + dp2_x, p2[1] + dp2_y])
        if np.sqrt(np.sum(np.square(p_new - ob1))) < ob1_r:
            circle = [ob1[0], ob1[1], ob1_r]
            temp = [0, 0]
            temp[0], temp[1] = LineIntersectCircle(circle, p2, p_new)
            p2[0], p2[1] = get_pose(ob1, temp, p_new)
        if np.sqrt(np.sum(np.square(p_new - ob2))) < ob2_r:
            circle = [ob2[0], ob2[1], ob2_r]
            temp = [0, 0]
            temp[0], temp[1] = LineIntersectCircle(circle, p2, p_new)
            p2[0], p2[1] = get_pose(ob2, temp, p_new)
        # Evader
        p_new = np.array([e[0] + de_x, e[1] + de_y])
        if np.sqrt(np.sum(np.square(p_new - ob1))) < ob1_r:
            circle = [ob1[0], ob1[1], ob1_r]
            temp = [0, 0]
            temp[0], temp[1] = LineIntersectCircle(circle, e, p_new)
            e[0], e[1] = get_pose(ob1, temp, p_new)
        if np.sqrt(np.sum(np.square(p_new - ob2))) < ob2_r:
            circle = [ob2[0], ob2[1], ob2_r]
            temp = [0, 0]
            temp[0], temp[1] = LineIntersectCircle(circle, e, p_new)
            e[0], e[1] = get_pose(ob2, temp, p_new)

        # Compute the velocity of agents
        agent_v = []
        agent_v.append([dp1_x, dp1_y])
        agent_v.append([dp2_x, dp2_y])
        agent_v.append([de_x, de_y])
        agent_v = np.array(agent_v)
        agent_v_angle = np.array([0.0, 0.0, 0.0])

        for i in range(0, 3):
            if agent_v[i][0] > 0 and agent_v[i][1] >= 0:
                agent_v_angle[i] = math.atan(math.tan(agent_v[i][1] / agent_v[i][0]))
            elif agent_v[i][0] > 0 and agent_v[i][1] < 0:
                agent_v_angle[i] = math.atan(math.tan(agent_v[i][1] / agent_v[i][0])) + 2 * np.pi
            elif agent_v[i][0] < 0 and agent_v[i][1] >= 0:
                agent_v_angle[i] = math.atan(math.tan(agent_v[i][1] / agent_v[i][0])) + np.pi
            elif agent_v[i][0] < 0 and agent_v[i][1] < 0:
                agent_v_angle[i] = math.atan(math.tan(agent_v[i][1] / agent_v[i][0])) + np.pi
            elif agent_v[i][0] == 0:
                if agent_v[i][1] > 0:
                    agent_v_angle[i] = 1 / 2 * np.pi
                else:
                    agent_v_angle[i] = 3 / 2 * np.pi

        # Compute local pose
        # p1_p2
        d = p2 - p1
        p1_p2_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p1_p2_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p1_p2_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p1_p2_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p1_p2_angle = 1 / 2 * np.pi
            else:
                p1_p2_angle = 3 / 2 * np.pi
        # p1_e
        d = e - p1
        p1_e_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p1_e_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p1_e_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p1_e_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p1_e_angle = 1 / 2 * np.pi
            else:
                p1_e_angle = 3 / 2 * np.pi
        # p2_p1
        d = p1 - p2
        p2_p1_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p2_p1_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p2_p1_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p2_p1_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p2_p1_angle = 1 / 2 * np.pi
            else:
                p2_p1_angle = 3 / 2 * np.pi
        # p2_e
        d = e - p2
        p2_e_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p2_e_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p2_e_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p2_e_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p2_e_angle = 1 / 2 * np.pi
            else:
                p2_e_angle = 3 / 2 * np.pi
        # e_p1
        d = p1 - e
        e_p1_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            e_p1_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            e_p1_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            e_p1_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                e_p1_angle = 1 / 2 * np.pi
            else:
                e_p1_angle = 3 / 2 * np.pi
        # e_p2
        d = p2 - e
        e_p2_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            e_p2_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            e_p2_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            e_p2_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                e_p2_angle = 1 / 2 * np.pi
            else:
                e_p2_angle = 3 / 2 * np.pi

        # Compute the distance to obstacle
        # Evader
        d = np.array([200, 100]) - e
        e_ob1_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            e_ob1_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            e_ob1_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            e_ob1_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                e_ob1_angle = 1 / 2 * np.pi
            else:
                e_ob1_angle = 3 / 2 * np.pi
        d = np.array([50, 100]) - e
        e_ob2_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            e_ob2_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            e_ob2_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            e_ob2_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                e_ob2_angle = 1 / 2 * np.pi
            else:
                e_ob2_angle = 3 / 2 * np.pi
        # Pursuer No.1
        d = np.array([200, 100]) - p1
        p1_ob1_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p1_ob1_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p1_ob1_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p1_ob1_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p1_ob1_angle = 1 / 2 * np.pi
            else:
                p1_ob1_angle = 3 / 2 * np.pi
        d = np.array([50, 100]) - p1
        p1_ob2_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p1_ob2_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p1_ob2_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p1_ob2_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p1_ob2_angle = 1 / 2 * np.pi
            else:
                p1_ob2_angle = 3 / 2 * np.pi
        # Pursuer No.2
        d = np.array([200, 100]) - p2
        p2_ob1_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p2_ob1_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p2_ob1_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p2_ob1_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p2_ob1_angle = 1 / 2 * np.pi
            else:
                p2_ob1_angle = 3 / 2 * np.pi
        d = np.array([50, 100]) - p2
        p2_ob2_r = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            p2_ob2_angle = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            p2_ob2_angle = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            p2_ob2_angle = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                p2_ob2_angle = 1 / 2 * np.pi
            else:
                p2_ob2_angle = 3 / 2 * np.pi

        self.steps_beyond_done = None

        # Compute the distance to wall
        # Pursuer No.1
        p1_l1 = p1[0]
        p1_l2 = 300 - p1[0]
        p1_l3 = p1[1]
        p1_l4 = 300 - p1[1]
        # Pursuer No.2
        p2_l1 = p2[0]
        p2_l2 = 300 - p2[0]
        p2_l3 = p2[1]
        p2_l4 = 300 - p2[1]
        # Evader
        e_l1 = e[0]
        e_l2 = 300 - e[0]
        e_l3 = e[1]
        e_l4 = 300 - e[1]

        cost = 0
        cost = np.sqrt(de_x * de_x + de_y * de_y) - 1.2

        min_distance = min(p1_e_r, p2_e_r)

        next_state = np.array([[e_p1_r, e_p1_angle, e_p2_r, e_p2_angle, e[0], e[1], agent_v_angle[2], e_l1, e_l2, e_l3,
                                e_l4, e_ob1_r, e_ob1_angle, ob1_r, e_ob2_r, e_ob2_angle, ob2_r],
                               [p1[0], p1[1], agent_v_angle[0], p1_p2_r, p1_p2_angle, p1_e_r, p1_e_angle, p1_l1, p1_l2,
                                p1_l3, p1_l4, p1_ob1_r, p1_ob1_angle, ob1_r, p1_ob2_r, p1_ob2_angle, ob2_r],
                               [p2[0], p2[1], agent_v_angle[1], p2_p1_r, p2_p1_angle, p2_e_r, p2_e_angle, p2_l1, p2_l2,
                                p2_l3, p2_l4, p2_ob1_r, p2_ob1_angle, ob1_r, p2_ob2_r, p2_ob2_angle, ob2_r]])
        self.state = next_state

        done = np.sqrt(np.sum(np.square(p1 - e))) <= 2 or np.sqrt(np.sum(np.square(p2 - e))) <= 2
        done = bool(done)

        if done == 0:
            evader_reward = 1.0 + cost + 0.01 * min_distance
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            evader_reward = -10.0 + cost
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            evader_reward = -10.0 + cost
        return next_state, [evader_reward, 0], done, {}

    def reset(self):
        # Initial the high and low
        evader_high = np.array(
            [300.0 * (2 ** 0.5), 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi, 300.0, 300.0, 2 * np.pi, 300.0, 300.0, 300.0,
             300.0, (200 ** 2 + 100 ** 2) ** 0.5, 2 * np.pi, 15, (250 ** 2 + 200 ** 2) ** 0.5, 2 * np.pi, 20])
        evader_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20])
        pursuer_high = np.array([[300.0, 300.0, 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi,
                                  300.0, 300.0, 300.0, 300.0, (200 ** 2 + 100 ** 2) ** 0.5, 2 * np.pi, 15,
                                  (250 ** 2 + 200 ** 2) ** 0.5, 2 * np.pi, 20],
                                 [300.0, 300.0, 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi, 300.0 * (2 ** 0.5), 2 * np.pi,
                                  300.0, 300.0, 300.0, 300.0, (200 ** 2 + 100 ** 2) ** 0.5, 2 * np.pi, 15,
                                  (250 ** 2 + 200 ** 2) ** 0.5, 2 * np.pi, 20]])
        pursuer_low = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15, 0.0, 15, 20, 0.0, 20]])
        evader_state = self.np_random.uniform(low=evader_low, high=evader_high)
        pursuer_state = self.np_random.uniform(low=pursuer_low, high=pursuer_high)

        # Initial the reset pose
        evader_state[4] = 100
        evader_state[5] = 100
        pursuer_state[0][0] = 100
        pursuer_state[0][1] = 200
        pursuer_state[1][0] = 200
        pursuer_state[1][1] = 150

        # Initial the p_p_r and p_p_angle
        # p1_p2
        d = np.array([pursuer_state[1][0], pursuer_state[1][1]]) - np.array([pursuer_state[0][0], pursuer_state[0][1]])
        pursuer_state[0][3] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[0][4] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[0][4] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[0][4] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[0][4] = 1 / 2 * np.pi
            else:
                pursuer_state[0][4] = 3 / 2 * np.pi
        # p1_e
        d = np.array([evader_state[4], evader_state[5]]) - np.array([pursuer_state[0][0], pursuer_state[0][1]])
        pursuer_state[0][5] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[0][6] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[0][6] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[0][6] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[0][6] = 1 / 2 * np.pi
            else:
                pursuer_state[0][6] = 3 / 2 * np.pi
        # p2_p1
        d = np.array([pursuer_state[0][0], pursuer_state[0][1]]) - np.array([pursuer_state[1][0], pursuer_state[1][1]])
        pursuer_state[1][3] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[1][4] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[1][4] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[1][4] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[1][4] = 1 / 2 * np.pi
            else:
                pursuer_state[1][4] = 3 / 2 * np.pi
        # p2_e
        d = np.array([evader_state[4], evader_state[5]]) - np.array([pursuer_state[1][0], pursuer_state[1][1]])
        pursuer_state[1][5] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[1][6] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[1][6] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[1][6] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[1][6] = 1 / 2 * np.pi
            else:
                pursuer_state[1][6] = 3 / 2 * np.pi
        # e_p1
        d = np.array([pursuer_state[0][0], pursuer_state[0][1]]) - np.array([evader_state[4], evader_state[5]])
        evader_state[0] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            evader_state[1] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            evader_state[1] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            evader_state[1] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                evader_state[1] = 1 / 2 * np.pi
            else:
                evader_state[1] = 3 / 2 * np.pi
        # e_p2
        d = np.array([pursuer_state[1][0], pursuer_state[1][1]]) - np.array([evader_state[4], evader_state[5]])
        evader_state[2] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            evader_state[3] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            evader_state[3] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            evader_state[3] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                evader_state[3] = 1 / 2 * np.pi
            else:
                evader_state[3] = 3 / 2 * np.pi

        # Initial the distance to wall
        # Evader
        evader_state[7] = evader_state[4]
        evader_state[8] = 300 - evader_state[4]
        evader_state[9] = evader_state[5]
        evader_state[10] = 300 - evader_state[5]
        # Pursuer No.1
        pursuer_state[0][7] = pursuer_state[0][0]
        pursuer_state[0][8] = 300 - pursuer_state[0][0]
        pursuer_state[0][9] = pursuer_state[0][1]
        pursuer_state[0][10] = 300 - pursuer_state[0][1]
        # Pursuer No.2
        pursuer_state[1][7] = pursuer_state[1][0]
        pursuer_state[1][8] = 300 - pursuer_state[1][0]
        pursuer_state[1][9] = pursuer_state[1][1]
        pursuer_state[1][10] = 300 - pursuer_state[1][1]

        # Initial the distance to obstacle
        # Evader
        d = np.array([200, 100]) - np.array([evader_state[4], evader_state[5]])
        evader_state[11] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            evader_state[12] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            evader_state[12] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            evader_state[12] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                evader_state[12] = 1 / 2 * np.pi
            else:
                evader_state[12] = 3 / 2 * np.pi
        d = np.array([50, 100]) - np.array(evader_state[4], evader_state[5])
        evader_state[14] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            evader_state[15] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            evader_state[15] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            evader_state[15] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                evader_state[15] = 1 / 2 * np.pi
            else:
                evader_state[15] = 3 / 2 * np.pi
        # Pursuer No.1
        d = np.array([200, 100]) - np.array([pursuer_state[0][0], pursuer_state[0][1]])
        pursuer_state[0][11] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[0][12] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[0][12] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[0][12] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[0][12] = 1 / 2 * np.pi
            else:
                pursuer_state[0][12] = 3 / 2 * np.pi
        d = np.array([50, 100]) - np.array([pursuer_state[0][0], pursuer_state[0][1]])
        pursuer_state[0][14] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[0][15] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[0][15] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[0][15] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[0][15] = 1 / 2 * np.pi
            else:
                pursuer_state[0][15] = 3 / 2 * np.pi
        # Pursuer No.2
        d = np.array([200, 100]) - np.array([pursuer_state[1][0], pursuer_state[1][1]])
        pursuer_state[1][11] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[1][12] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[1][12] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[1][12] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[1][12] = 1 / 2 * np.pi
            else:
                pursuer_state[1][12] = 3 / 2 * np.pi
        d = np.array([50, 100]) - np.array([pursuer_state[1][0], pursuer_state[1][1]])
        pursuer_state[1][14] = np.sqrt(np.sum(np.square(d)))
        if d[0] > 0 and d[1] >= 0:
            pursuer_state[1][15] = math.atan(math.tan(d[1] / d[0]))
        elif d[0] > 0 and d[1] < 0:
            pursuer_state[1][15] = math.atan(math.tan(d[1] / d[0])) + 2 * np.pi
        elif d[0] < 0:
            pursuer_state[1][15] = math.atan(math.tan(d[1] / d[0])) + np.pi
        elif d[0] == 0:
            if d[1] > 0:
                pursuer_state[1][15] = 1 / 2 * np.pi
            else:
                pursuer_state[1][15] = 3 / 2 * np.pi

        states = [[0] * (17) for _ in range(3)]
        for i in range(17):
            states[0][i] = evader_state[i]
        for i in range(2):
            for j in range(17):
                states[i + 1][j] = pursuer_state[i][j]

        self.state = states
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

            self.obstacle1 = rendering.make_circle(15, 120)
            self.obs1trans = rendering.Transform(translation=(205, 105))
            self.obstacle1.add_attr(self.obs1trans)
            self.viewer.add_geom(self.obstacle1)

            self.obstacle2 = rendering.make_circle(20, 160)
            self.obs2trans = rendering.Transform(translation=(55, 105))
            self.obstacle2.add_attr(self.obs2trans)
            self.viewer.add_geom(self.obstacle2)

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
            self.viewer.add_geom(self.evader)

        if self.state is None:
            return None

        self.p1trans.set_translation(self.state[1][0] + 5, self.state[1][1] + 5)
        self.p1ctrans.set_translation(self.state[1][0] + 5, self.state[1][1] + 5)
        self.p2trans.set_translation(self.state[2][0] + 5, self.state[2][1] + 5)
        self.p2ctrans.set_translation(self.state[2][0] + 5, self.state[2][1] + 5)
        self.etrans.set_translation(self.state[0][4] + 5, self.state[0][5] + 5)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def LineIntersectCircle(p, lsp, lep):
    # p is the circle parameter, lsp and lep is the two end of the line
    x0, y0, r0 = p
    x1, y1 = lsp
    x2, y2 = lep
    if x1 == x2:
        p1 = [x1, y0 - (r0 ** 2 - (x1 - x0) ** 2) ** 0.5]
        p2 = [x1, y0 + (r0 ** 2 - (x1 - x0) ** 2) ** 0.5]
        # select the points lie on the line segment
        if p1[1] <= max(y1, y2) and p1[1] >= min(y1, y2):
            inp = [p1[0], p1[1]]
        else:
            inp = [p2[0], p2[1]]
    else:
        k = (y1 - y2) / (x1 - x2)
        b0 = y1 - k * x1
        a = k ** 2 + 1
        b = 2 * k * (b0 - y0) - 2 * x0
        c = (b0 - y0) ** 2 + x0 ** 2 - r0 ** 2
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            p1x = (-b - delta ** 0.5) / (2 * a)
            p2x = (-b + delta ** 0.5) / (2 * a)
            p1y = k * p1x + b0
            p2y = k * p2x + b0
            # select the points lie on the line segment
            if p1x <= max(x1, x2) and p1x >= min(x1, x2):
                inp = [p1x, p1y]
            else:
                inp = [p2x, p2y]
        else:
            inp = []
    return inp[0], inp[1]


def get_pose(P1, P2, P3):
    P4 = [0, 0]
    if P1[1] == P2[1]:
        P4[0] = P2[0]
        P4[1] = P3[1]
    elif P1[0] == P2[0]:
        P4[0] = P3[0]
        P4[1] = P2[1]
    else:
        a1 = P2[1] - P1[1]
        b1 = P1[0] - P2[0]
        k1 = -a1 / b1
        c1 = P3[1] - k1 * P3[0]
        k2 = b1 / a1
        c2 = P2[1] - k2 * P2[0]
        P4[0] = (c2 - c1) / (k1 - k2)
        P4[1] = k1 * P4[0] + c1
    return P4[0], P4[1]
