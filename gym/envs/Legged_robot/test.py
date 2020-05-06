import gym.envs.Legged_robot.gait as gait
import numpy as np
import matplotlib.pyplot as plt
import time

trotting = gait.Gait(10, [0, 5, 5, 0], [5, 5, 5, 5], "Trotting")
bounding = gait.Gait(10, [5, 5, 0, 0], [4, 4, 4, 4], "Bounding")


# t0 = time.perf_counter_ns()
# trotting.get_contact_state()
# t1 = time.perf_counter_ns()
# print(t1 - t0)
# print(trotting.get_contact_state())  # the output should be [0.6, 0, 0, 0.6]
# print(trotting.get_swing_state())  # the output should be [0, 0.6, 0.6, 0]

def plt_current_contact(current_gait):
    for i in range(50):
        contact_state = current_gait.get_contact_state() > 0
        coordinates = np.array([[1, 2], [0, 2], [1, 0], [0, 0]])
        coordinates = coordinates[contact_state, :]
        x, y = coordinates.T
        plt.scatter(x, y)
        axes = plt.gca()
        axes.set_xlim([-0.5, 1.5])
        axes.set_ylim([-0.5, 2.5])
        plt.pause(0.1)
        current_gait.step()
    plt.show()


plt_current_contact(bounding)
