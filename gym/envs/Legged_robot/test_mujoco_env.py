import gym
import time

env = gym.make("Humanoid-v2")
env.reset()


for _ in range(1000):
    data = env.sim.data
    env.render()
    env.step(env.action_space.sample())
env.close()