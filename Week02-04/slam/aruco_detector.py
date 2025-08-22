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
        self.aruco_params = cv2.aruco.DetectorParameters() # updated to work with newer OpenCV
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) # updated to work with newer OpenCV



    # updated function to offset the cube centre 

    # def detect_marker_positions(self, img, use_cube_center=True):
    #     # Perform detection
    #     corners, ids, rejected = cv2.aruco.detectMarkers(
    #         img, self.aruco_dict, parameters=self.aruco_params)
    #     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    #         corners, self.marker_length, self.camera_matrix, self.distortion_params)

    #     if ids is None:
    #         return [], img

    #     # Compute the marker positions
    #     measurements = []
    #     seen_ids = []
    #     for i in range(len(ids)):
    #         idi = ids[i,0]

    #         # Avoid handling same ID multiple times
    #         if idi in seen_ids and idi > 10:  # NOTE: your filtering condition kept
    #             continue
    #         else:
    #             seen_ids.append(idi)

    #         if use_cube_center:
    #             # --- Use cube center instead of marker face ---
    #             tvec = tvecs[i].reshape(3,)
    #             rvec = rvecs[i]
    #             R, _ = cv2.Rodrigues(rvec)

    #             # Marker normal vector (Z-axis of marker frame)
    #             normal = R[:,2]

    #             # Shift inward by half cube side length
    #             cube_center = tvec - normal * (self.cube_side_length / 2.0)

    #             # Convert to bird’s-eye 2D format
    #             lm_bff2d = np.array([[cube_center[2]], [-cube_center[0]]])

    #         else:
    #             # --- Original marker plane center method ---
    #             lm_tvecs = tvecs[ids==idi].T
    #             lm_bff2d = np.block([[lm_tvecs[2,:]], [-lm_tvecs[0,:]]])
    #             lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

    #         lm_measurement = measure.Marker(lm_bff2d, idi)
    #         measurements.append(lm_measurement)
        
    #     # Draw markers on image copy
    #     img_marked = img.copy()
    #     cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

    #     return measurements, img_marked

    

# this is uni code 

    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
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
            if idi in seen_ids or idi > 10:# added the line to on ly detetc aruco marker to 10 
                continue
            else:
                seen_ids.append(idi)


            # this is the code from uni keep 
            lm_tvecs = tvecs[ids==idi].T # this a translation vector that will give you the centre of the face we record not the centre of the cube 


            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

            lm_measurement = measure.Marker(lm_bff2d, idi)
            measurements.append(lm_measurement)
        
        # Draw markers on image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)

        return measurements, img_marked


