# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:
    def __init__(self, robot, marker_length=0.07):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters() # updated to work with newer OpenCV - maybe this is what was causing the issues yesterday
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) # updated to work with newer OpenCV
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
    
        # jas - comment: tvec is translation vector (2d position) and rvec is orienation of the marker in the camera frame
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)
        # rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params) # use this instead if you got a value error

        if ids is None:
            return [], img

        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            # jas -comment: we can also add that if an idi is greater than 10 we continue
            if idi in seen_ids or  idi > 10:
                continue
            else:
                seen_ids.append(idi)

            
            # lm_tvecs = tvecs[ids==idi].T
            # lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            # lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

            # lm_measurement = measure.Marker(lm_bff2d, idi)
            # measurements.append(lm_measurement)


            # Get the rotation and translation vectors for this marker - claude code line 52 -79
            marker_indices = np.where(ids.flatten() == idi)[0]
            marker_rvec = rvecs[marker_indices[0]]
            marker_tvec = tvecs[marker_indices[0]]
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(marker_rvec)
            
            # Calculate offset from marker face center to cube center
            # The marker face normal vector in marker coordinate system is [0, 0, 1]
            # We need to offset by half the cube size in the negative normal direction
            face_normal_marker_coords = np.array([0, 0, 1]).reshape(3, 1)
            
            # Transform the normal vector to camera coordinates
            face_normal_camera_coords = rotation_matrix @ face_normal_marker_coords
            
            # Calculate offset vector (half cube size in the direction opposite to face normal)
            offset_distance = 0.08 / 2.0
            offset_vector = -offset_distance * face_normal_camera_coords
            
            # Apply offset to get cube center position
            cube_center_tvec = marker_tvec.reshape(3, 1) + offset_vector
            
            # Transform to the coordinate system used in the original code
            lm_bff2d = np.block([[cube_center_tvec[2,:]],[-cube_center_tvec[0,:]]])

            lm_measurement = measure.Marker(lm_bff2d, idi)
            measurements.append(lm_measurement)

            
           
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked
