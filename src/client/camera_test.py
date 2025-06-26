import numpy as np
import cv2 as cv
# from scipy.spatial.transform import Rotation as R
import json

def estimate_pose(img, detector: cv.aruco.ArucoDetector, camera_matrix, dist_coeffs):
    # Detect markers
    corners, ids, _ = detector.detectMarkers(img) # define & finds vars for corners and id's of the 2 tags

    labeled = img.copy() # creates copy of img
    t_matrix = np.eye(4,4) # creates identity matrix 
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
        _, rvec, tvec = cv.solvePnP(objPoints, corners, camera_matrix, dist_coeffs) # 0.053 is ratio to prev aruco tag

        # r_matrix = cv.Rodrigues(rvec)[0]

        # t_matrix[:3,:3] = r_matrix
        # t_matrix[:3,3] = tvec[0]
        # t_matrix = np.linalg.inv(t_matrix) # invert transforms from camera to world

        # # rotate camera to fix flipped camera issue 
        # r1 = R.from_euler('y', 180, degrees=True).as_matrix()
        # r2 = R.from_euler('z', 180, degrees=True).as_matrix()
        # t_matrix[:3,:3] = t_matrix[:3,:3] @ r1 @ r2 # multiple t matrix by the two rotations to apply them 
        
        # print(tvec[:2])

        # # Draw axis for the aruco markers
        cv.drawFrameAxes(labeled, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

    return tvec, success,labeled


fx = 1380.4580078125
fy = 1381.84802246094
cx = 970.383361816406
cy = 548.931640625

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

cam = cv.VideoCapture(4)

while True:
    ret, frame = cam.read()

    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    arucoParams = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)
    
    t_vec, success,labeled = estimate_pose(frame, detector, camera_matrix, dist_coeffs)

    print(t_vec)
    cv.imshow("camera", labeled)

    # Press 'q' to exit the loop
    if cv.waitKey(1) == ord('q'):
        break

# estimate_pose()