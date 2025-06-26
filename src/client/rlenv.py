import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from client import Robot
import random
import cv2 as cv
import time
import torch


class KinovaEnv(gym.Env):
    def __init__(self):
        self.action_space = Box(-0.75, 0.75, (2,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (2,), np.float32) 
        self.r = Robot()
        self.goal = self.generate_goal()

        self.x_translation = 0.32
        self.y_translation = 0.6

        self.fx = 1380.4580078125
        self.fy = 1381.84802246094
        self.cx = 970.383361816406
        self.cy = 548.931640625

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.cam = cv.VideoCapture(4)
        self.prev_pose = [0,0]

    def get_reward(self, observations):
        obj_pose = observations[:2]
        reward = np.linalg.norm(obj_pose - self.goal)
        success = reward < 0.000001
        return reward, success

    def _to_torch(self, x):
        # convert to torch and unsqueeze to give batch dimension
        return torch.from_numpy(x).to(dtype=torch.float32).unsqueeze(dim=0)

    def step(self, action):
        # convert action to numpy array
        if torch.is_tensor(action):
            action = action[0].cpu().numpy()*0.01
            action = [action[0].item(), action[1].item(), 0]
        print("stepping with action:", action)
        observations = self.get_observations(action)
        reward, success = self.get_reward(observations) 
        return self._to_torch(observations), self._to_torch(np.array([reward])), self._to_torch(np.array([success])), {}
    
    def reset(self, **kwargs):
        self.r.reset()
        self.goal = self.generate_goal()
        obs = self.get_observations(action=[0,0])
        return self._to_torch(obs)
    
    def generate_goal(self):
        MIN_RADIUS = 0.2
        MAX_RADIUS = 0.6
        goal = np.array([(random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS), 
                        (random.randint(0, 1)*2 - 1)*random.uniform(MIN_RADIUS, MAX_RADIUS)])
        print(f"Goal: {goal}")
        return goal
    
    def estimate_pose(self, img, detector: cv.aruco.ArucoDetector):
        # Detect markers
        corners, ids, _ = detector.detectMarkers(img) # define & finds vars for corners and id's of the 2 tags

        tvec = None
        success = ids is not None and (len(ids.flatten()) == 1) # makes sure there is 1 id 
        # If detected
        if success:
            corners = corners[0]
            # Define the marker's side length
            markerLength = 0.08  # in meters
            # Create a NumPy array to hold the 3D coordinates of the marker's corners
            objPoints = np.array([
                [-markerLength / 2, markerLength / 2, 0],
                [markerLength / 2, markerLength / 2, 0],
                [markerLength / 2, -markerLength / 2, 0],
                [-markerLength / 2, -markerLength / 2, 0]
            ], dtype=np.float32)
            # Detect aruco pose
            _, rvec, tvec = cv.solvePnP(objPoints, corners, self.camera_matrix, self.dist_coeffs) # 0.053 is ratio to prev aruco tag
        return tvec, success

    def get_observations(self, action):
        arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        arucoParams = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

        success = False

        while not success:
            ret, frame = self.cam.read()
            t_vec, success = self.estimate_pose(frame, detector)
        
        if success:
            pose = [-t_vec[0,0] + self.x_translation, t_vec[1,0] - self.y_translation]
        else:
            pose = self.prev_pose

        observations = np.array(self.r.send_receive(action))

        return np.concatenate((observations, self.goal, pose, np.zeros(4)),axis=0)

