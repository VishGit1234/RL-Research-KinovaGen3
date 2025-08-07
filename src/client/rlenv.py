import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from client import Robot
from scipy.spatial.transform import Rotation as R
import random
import cv2 as cv
import time
import torch

# from camera_test import estimate_pose, camera_matrix, dist_coeffs

ACTION_SCALE=0.01

TERMINATION_CUBE_DISTANCE = 0.05
STATIC_MARKER_ID = 50
STATIC_MARKER_LENGTH = 0.05 # in m
BLOCK_MARKER_ID = 1
BLOCK_MARKER_LENGTH = 0.04 # in m
T_WORLD_TO_STATIC_MARKER = np.array([
    [1, 0, 0, 0.115],
    [0, 1, 0, -0.117],#-0.1097],
    [0, 0, 1, 0.010], #0.00125
    [0, 0, 0, 1]
]) # real world frame is center of arm with coordinate frame (x axis right, y axis forward, z axis up)

T_BLOCK_MARKER_TO_BLOCK_CENTER = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0.05],
    [0, 0, 1, 0.02 - 0.005],
    [0, 0, 0, 1]
])

T_SIM_TO_REAL_WORLD = np.array([
    [-1,0,0,-0.615], # robot center is actually at (-0.615, 0, 0) in sim world
    [0,-1,0,0],
    [0,0,1,0],
]) # sim world is (x axis left, y axis backward, z axis up) -> so must rotate 180 degree in z

class KinovaEnv(gym.Env):
    def __init__(self):
        self.action_space = Box(-0.75, 0.75, (3,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (1,11,), np.float32) 
        self.r = Robot()

        # These parameters are specific to the resolution
        # 640x480
        # self.cx = 324.614837646484
        # self.cy = 243.969604492188
        # self.fx = 613.536926269531
        # self.fy = 614.154663085938

        # 1920x1080 
        self.cx = 970.383361816406
        self.cy = 548.931640625
        self.fx = 1380.4580078125
        self.fy = 1381.84802246094

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.cam = cv.VideoCapture(4)
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.prev_pose = np.array([0,0,0,0,0,0,0])

        self.prev_pose = self.get_block_pose() 
        self.goal = np.copy(self.prev_pose[:3])
        self.goal[1] += 0.2

    def get_reward(self, observations):
        obj_pos = observations[6:9] # block pos
        reward = np.linalg.norm(obj_pos - self.goal)
        success = reward < TERMINATION_CUBE_DISTANCE
        return reward, success

    def _to_torch(self, x):
        # convert to torch and unsqueeze to give batch dimension
        return torch.from_numpy(x).to(dtype=torch.float32).unsqueeze(dim=0)

    def step(self, action):
        # convert action to numpy array
        if torch.is_tensor(action):
            action = action[0].cpu().numpy()*ACTION_SCALE
            action = [action[0].item(), action[1].item(), action[2].item(), 0.0]
        observations = self.get_observations(action) # np.array([-0.3142,  0.0411, -0.3000,  0.4000, -0.3000,  0.2000,  0.0200,  1.0000, 0.0000,  0.0000,  0.0000])
        reward, success = self.get_reward(observations) 
        print("action:", action) 
        print("observation:", observations)
        print("success:", success)
        return self._to_torch(observations), self._to_torch(np.array([reward])), self._to_torch(np.array([success])), self._to_torch(np.array([success])), {}
    
    def reset(self, **kwargs):
        self.r.reset()
        obs = self.get_observations(action=[0,0,0,0])
        return self._to_torch(obs)
    
    def estimate_pose(self, img, detector: cv.aruco.ArucoDetector):
        # Detect markers
        corners, ids, _ = detector.detectMarkers(img) # define & finds vars for corners and id's of the 2 tags
        
        labeled = img.copy() # creates copy of img
        # Transform from camera to static marker
        t_camera_to_static_marker = np.eye(4,4) # creates identity matrix
        # Transform from camera to block marker
        # Note: The block marker is not at the center of the block
        t_camera_to_block_marker = np.eye(4,4)
        success = ids is not None and (set(ids.flatten()) == set([STATIC_MARKER_ID, BLOCK_MARKER_ID])) # makes sure there are 2 ids

        position, quaternion = None, None

        # If detected
        if success:
            # Index of data for each marker within corners
            static_marker_ind = ids.flatten().tolist().index(STATIC_MARKER_ID)
            block_marker_ind = ids.flatten().tolist().index(BLOCK_MARKER_ID)

            static_marker_corners = corners[static_marker_ind]
            block_marker_corners = corners[block_marker_ind]

            # Create a NumPy array to hold the 3D coordinates of the marker's corners
            static_marker_points = np.array([
                [-STATIC_MARKER_LENGTH / 2, STATIC_MARKER_LENGTH / 2, 0],
                [STATIC_MARKER_LENGTH / 2, STATIC_MARKER_LENGTH / 2, 0],
                [STATIC_MARKER_LENGTH / 2, -STATIC_MARKER_LENGTH / 2, 0],
                [-STATIC_MARKER_LENGTH / 2, -STATIC_MARKER_LENGTH / 2, 0]
            ], dtype=np.float32)
            _, static_marker_rvec, static_marker_tvec = cv.solvePnP(static_marker_points, static_marker_corners, self.camera_matrix, self.dist_coeffs)

            block_marker_points = np.array([
                [-BLOCK_MARKER_LENGTH / 2, BLOCK_MARKER_LENGTH / 2, 0],
                [BLOCK_MARKER_LENGTH / 2, BLOCK_MARKER_LENGTH / 2, 0],
                [BLOCK_MARKER_LENGTH / 2, -BLOCK_MARKER_LENGTH / 2, 0],
                [-BLOCK_MARKER_LENGTH / 2, -BLOCK_MARKER_LENGTH / 2, 0]
            ], dtype=np.float32)
            _, block_marker_rvec, block_marker_tvec = cv.solvePnP(block_marker_points, block_marker_corners, self.camera_matrix, self.dist_coeffs)

            # Populate transform from camera to static marker
            # This is w.r.t to the reference frame from perspective of camera (x is right, y is down, z is far)
            static_marker_rmatrix = cv.Rodrigues(static_marker_rvec)[0]
            t_camera_to_static_marker[:3, :3] = static_marker_rmatrix
            t_camera_to_static_marker[:3, 3] = static_marker_tvec.T

            # Populate transform from camera to block marker
            block_marker_rmatrix = cv.Rodrigues(block_marker_rvec)[0]
            t_camera_to_block_marker[:3, :3] = block_marker_rmatrix
            t_camera_to_block_marker[:3, 3] = block_marker_tvec.T

            t_world_to_camera = T_WORLD_TO_STATIC_MARKER @ np.linalg.inv(t_camera_to_static_marker)
            t_world_to_block_marker = t_world_to_camera @ t_camera_to_block_marker
            t_world_to_block_center = t_world_to_block_marker @ T_BLOCK_MARKER_TO_BLOCK_CENTER
            t_sim_world_to_block_center = T_SIM_TO_REAL_WORLD @ t_world_to_block_center

            position = t_sim_world_to_block_center[:3, 3]
            rotation = R.from_matrix(t_sim_world_to_block_center[:3, :3])
            quaternion = rotation.as_quat(scalar_first=True)
            
        return position, quaternion, success, labeled
    
    def get_block_pose(self):
        arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
        arucoParams = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

        success = False

        while not success:
            ret, frame = self.cam.read()
            pos, quat, success, labeled = self.estimate_pose(frame, detector)
        
        if success:
            # 0.02 ()
            block_pose = [pos[0], pos[1], pos[2], quat[3], quat[0], quat[1], quat[2]]
        else:
            block_pose = self.prev_pose

        return block_pose


    def get_observations(self, action):
        robot_obs = np.array(self.r.send_receive(action))
        # Translate robot_obs by -0.615 in the x
        robot_obs[0] -= 0.615

        block_pose = np.array(self.get_block_pose())        

        obs = np.concatenate((robot_obs, self.goal, block_pose),axis=0) # obs(3) + goal(3) + block_pose(7)

        reference_from_sim = np.array([-0.3142, 0.0411, -0.3, 0.4, -0.3, 0.2, 0.02, 1.0, 0, 0, 0]) # outdated
        return obs
    
if __name__ == '__main__':
    env = KinovaEnv()
    for i in range(50):
        env.step([0, 0, 0, 0])
    print("done simple rlenv test")
