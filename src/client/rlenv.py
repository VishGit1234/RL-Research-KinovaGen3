import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from client import Robot
from scipy.spatial.transform import Rotation as R
import random
import cv2 as cv
import time
import torch


class KinovaEnv(gym.Env):
    def __init__(self):
        self.action_space = Box(-0.75, 0.75, (2,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (1,11,), np.float32) 
        self.r = Robot()

        # change below 2 as needed (based on camera position)
        self.x_translation = 1.13
        self.y_translation = 1

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

        self.prev_pose = self.get_block_pose() 
        self.goal = self.prev_pose[:2]
        self.goal[1] += 0.2


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
        obs = self.get_observations(action=[0,0])
        return self._to_torch(obs)
    
    def estimate_pose(self, img, detector: cv.aruco.ArucoDetector):
        # Detect markers
        corners, ids, _ = detector.detectMarkers(img) # define & finds vars for corners and id's of the 2 tags

        tvec = None
        rvec = None
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
        return rvec, tvec, success
    
    def get_block_pose(self):
        arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        arucoParams = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

        arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        arucoParams = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

        success = False

        while not success:
            ret, frame = self.cam.read()
            rvec, t_vec, success = self.estimate_pose(frame, detector)
        
        if success:
            rotation = R.from_rotvec(rvec.T)
            quat = rotation.as_quat()[0]
            block_pose = [t_vec[0,0] + self.x_translation, t_vec[1,0] + self.y_translation, 0, quat[0], quat[1], quat[2], quat[3]]
        else:
            block_pose = self.prev_pose

        return block_pose


    def get_observations(self, action):
        robot_obs = np.array(self.r.send_receive(action))
        robot_obs[0] *= -1

        block_pose = self.get_block_pose()

        obs = np.concatenate((robot_obs, self.goal, block_pose),axis=0) # obs(2) + goal(2) + block_pose(7)
        return obs
