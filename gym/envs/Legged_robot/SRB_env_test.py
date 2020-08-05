# TODO: Plot position with respect to time, to see how large the error will be (Stance)

import gym.envs.Legged_robot.gait as gait
import gym.envs.Legged_robot.Single_Rigid_body as SRB
import numpy as np
import matplotlib.pyplot as plt
import time

standing = gait.Gait(10, [0, 5, 5, 0], [10, 10, 10, 10], "Standing")
standing.stance = True
SRBEnv_test = SRB.SRBEnv(standing)
u = np.zeros([12, 1])

for i in range(1, 5):
    u[3*i - 1, 0] = 2.25

t0 = time.time()

# 1 step for 0.01s, 1000 step simulate 10s
for _ in range(1000):
    SRBEnv_test.step(u)

t1 = time.time()

print(t1 - t0)
