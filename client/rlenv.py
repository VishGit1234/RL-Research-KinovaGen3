import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from client import Robot
import random

class KinovaEnv(gym.Env):
    def __init__(self):
        self.action_space = Box(-0.75, 0.75, (3,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (3,), np.float32)
        self.r = Robot()
        self.goal = self.generate_goal()

    def get_reward(self, observations):
        reward = np.linalg.norm(observations, self.goal)
        success = reward < 0.000001
        return reward, success

    def step(self, action):
        observations = self.r.send_receive(action)
        reward = -1
        success = False
        if type(observations) != str: # skip computing reward
            reward, success = self.get_reward(observations)
        return observations, reward, success
    
    def reset(self):
        self.r.reset()
        self.goal = self.generate_goal()

    def generate_goal(self):
        MIN_RADIUS = 0.2
        MAX_RADIUS = 0.6
        goal = np.array([(random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS), 
                        (random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS), 
                        random.uniform(MIN_RADIUS, MAX_RADIUS)])
        print(f"Goal: {goal}")
        return goal