import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R

STATIC_MARKER_ID = 50
STATIC_MARKER_LENGTH = 0.05 # in m
BLOCK_MARKER_ID = 1
BLOCK_MARKER_LENGTH = 0.04 # in m
T_WORLD_TO_STATIC_MARKER = np.array([
    [1, 0, 0, 0.115],
    [0, 1, 0, -0.117], #-0.1097
    [0, 0, 1, 0.008], #0.00125
    [0, 0, 0, 1]
]) # in terms of real world space (x axis right, y axis forward, z axis up)

T_BLOCK_MARKER_TO_BLOCK_CENTER = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -0.05],
    [0, 0, 1, 0.02 - 0.005],
    [0, 0, 0, 1]
])

T_SIM_TO_REAL_WORLD = np.array([
    [-1,0,0,-0.615], # robot center is actually at (-0.615, 0, 0) in sim world
    [0,-1,0,0],
    [0,0,1,0],
]) # sim world is (x axis left, y axis backward, z axis up) -> so must rotate 180 degree in z


def estimate_pose(img, detector: cv.aruco.ArucoDetector, camera_matrix, dist_coeffs):
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
        print("IDs:", ids.flatten())

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
        _, static_marker_rvec, static_marker_tvec = cv.solvePnP(static_marker_points, static_marker_corners, camera_matrix, dist_coeffs)

        block_marker_points = np.array([
            [-BLOCK_MARKER_LENGTH / 2, BLOCK_MARKER_LENGTH / 2, 0],
            [BLOCK_MARKER_LENGTH / 2, BLOCK_MARKER_LENGTH / 2, 0],
            [BLOCK_MARKER_LENGTH / 2, -BLOCK_MARKER_LENGTH / 2, 0],
            [-BLOCK_MARKER_LENGTH / 2, -BLOCK_MARKER_LENGTH / 2, 0]
        ], dtype=np.float32)
        _, block_marker_rvec, block_marker_tvec = cv.solvePnP(block_marker_points, block_marker_corners, camera_matrix, dist_coeffs)

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

        # Draw axis for the aruco markers
        cv.drawFrameAxes(labeled, camera_matrix, dist_coeffs, static_marker_rvec, static_marker_tvec, 0.05)
        cv.drawFrameAxes(labeled, camera_matrix, dist_coeffs, block_marker_rvec, block_marker_tvec, 0.05)
        

    return position, quaternion, success, labeled

# These parameters are specific to the resolution
# 640x480
# cx = 324.614837646484
# cy = 243.969604492188
# fx = 613.536926269531
# fy = 614.154663085938

# 1920x1080 
cx = 970.383361816406
cy = 548.931640625
fx = 1380.4580078125
fy = 1381.84802246094

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

if __name__ == "__main__":

    cam = cv.VideoCapture(4)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cam.read()
    print(frame)
    height, width = frame.shape[:2]
    print(f"Camera resolution: {width} x {height}")

    while True:

        arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
        arucoParams = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)
        success = False
        while not success:
            ret, frame = cam.read()
            position, quaternion, success,labeled = estimate_pose(frame, detector, camera_matrix, dist_coeffs)

        rotation = R.from_quat(quaternion)
        euler_quat = rotation.as_euler('xyz', degrees=True)

        print(f"euler angles (x, y, z): {euler_quat}")
        print(f"block pos (x, y, z)", position)

        # print("Position:", t_vec)
        cv.imshow("camera", labeled)

        # Press 'q' to exit the loop
        if cv.waitKey(1) == ord('q'):
            break