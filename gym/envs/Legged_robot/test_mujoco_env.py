import gym
import time

env = gym.make("BipedalWalker-v3")
env.reset()

t0 = time.time()
env.step(env.action_space.sample())
t1 = time.time()
env.close()
print(t1 - t0)
