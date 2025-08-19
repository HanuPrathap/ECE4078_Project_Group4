import numpy as np

# hey i changed this

"""
The Robot class models a differential-drive robot for SLAM. it maintains the robots: 
1. state 
2. handels motion updates
3. landmark measurements 
4. computes derrivitives and covariances for uncertainty propogation

"""

"""
drive_meas is an object that encapsulates the measurements and uncertainties related to gthe robots drive command 

has the following attributes: 
left_speed: Speed of the left wheel (in ticks/s)
right_speed: Speed of the right wheel (in ticks/s)
dt: Time duration for the drive command (in seconds)
left_cov: Covariance (uncertainty) of the left wheel speed
right_cov: Covariance (uncertainty) of the right wheel speed

drive_meas comes from outside this class from util/measure.py


"""


class Robot:
    def __init__(self, wheels_width, wheels_scale, camera_matrix, camera_dist):
        # State is a vector of [x,y,theta]'
        self.state = np.zeros((3,1)) # (3 by 1 colum vector)
        
        # Wheel parameters
        self.wheels_width = wheels_width  # The distance between the left and right wheels (float)
        self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s (float)

        # Camera parameters
        self.camera_matrix = camera_matrix  # Matrix of the focal lengths and camera centre
        self.camera_dist = camera_dist  # Distortion coefficients
    

    """
    this method drives the robot or rather updates the state based purely of kinetic model (based on wheels speeds and time)

    """
    def drive(self, drive_meas):
        # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity
        linear_velocity, angular_velocity = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        # Apply the velocities - this is donne by uni - motion model 
        dt = drive_meas.dt
        if angular_velocity == 0:
            self.state[0] += np.cos(self.state[2]) * linear_velocity * dt
            self.state[1] += np.sin(self.state[2]) * linear_velocity * dt
        else:
            th = self.state[2]
            self.state[0] += linear_velocity / angular_velocity * (np.sin(th+dt*angular_velocity) - np.sin(th))
            self.state[1] += -linear_velocity / angular_velocity * (np.cos(th+dt*angular_velocity) - np.cos(th))
            self.state[2] += dt*angular_velocity


    """
    measure method - measures landmakrs relative to robots frame from world frame   
    1. markers is a 2*N array fo landmark positions in world frame
    2. idx_list is a list of indicies to select landmakrs 
    """
    def measure(self, markers, idx_list):
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        robot_xy = self.state[0:2,:]

        measurements = []
        for idx in idx_list:
            marker = markers[:,idx:idx+1]
            marker_bff = Rot_theta.T @ (marker - robot_xy)
            measurements.append(marker_bff)

        # Stack the measurements in a 2xm structure.
        markers_bff = np.concatenate(measurements, axis=1)
        return markers_bff
    

    """
    converts wheel speed to linear velocity and angualr velcoty of the robot 
    """
    
    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.wheels_scale
        right_speed_m = right_speed * self.wheels_scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
        return linear_velocity, angular_velocity

    # Derivatives and Covariance
    # --------------------------

    def derivative_drive(self, drive_meas):
        # Compute the differential of drive w.r.t. the robot state 
        # which is x y theta

        # se the differential as a identity matrix for now 
        DFx = np.zeros((3,3))
        DFx[0,0] = 1
        DFx[1,1] = 1
        DFx[2,2] = 1

        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        # convert angular velcoty to a small value if 0 
        ang_vel_safe = ang_vel if abs(ang_vel) > 1e-6 else 1e-6

        dt = drive_meas.dt
        th = self.state[2]
        
        # TODO: add your codes here to compute DFx using lin_vel, ang_vel, dt, and th
        # doubled checked this 
        DFx[0,2] = lin_vel/ang_vel_safe*(np.cos(th + dt * ang_vel) - np.cos(th))
        DFx[1,2] = lin_vel/ang_vel_safe*(np.sin(th + dt * ang_vel) - np.sin(th))

        # this is the jacobian matrix of the motion model with respect to the robot state vextor (x,y,theta)
        # this represents the how change in the robots displaecement and orientation affects the robot state



        return DFx

    """
    this method gets the derrivites of the landmarks measurements with respect to both robot state(x,y,theta) and landmark position(x,y)
     
    this tells us how small changes in robot pose or land mark position affect the observed landmark possiton in the robot frame
    """
    def derivative_measure(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        n = 2*len(idx_list)
        m = 3 + 2*markers.shape[1]

        DH = np.zeros((n,m))

        robot_xy = self.state[0:2,:]
        th = self.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.

            lmj_inertial = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)

            # robot xy DH
            DH[2*i:2*i+2,0:2] = - Rot_theta.T
            # robot theta DH
            DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
            # lm xy DH
            DH[2*i:2*i+2, 3+2*j:3+2*j+2] = Rot_theta.T

            # print(DH[i:i+2,:])

        return DH
    
    def covariance_drive(self, drive_meas):
        # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed

        # jacobian of the transformatino of the wheel speeds to the linear and angular velocities- 
        # tells us how the linear and angular velocities change with respect to the left and right wheel speeds
        Jac1 = np.array([[self.wheels_scale/2, self.wheels_scale/2],
                [-self.wheels_scale/self.wheels_width, self.wheels_scale/self.wheels_width]])
        
        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        th = self.state[2]
        dt = drive_meas.dt
        th2 = th + dt*ang_vel

        # Derivative of x,y,theta w.r.t. lin_vel, ang_vel
        # this is the jacobian matrix of the transformation of the linear and angular velocities to the robot state vector (x,y,theta)
        Jac2 = np.zeros((3,2))
        
        # TODO: add your codes here to compute Jac2 using lin_vel, ang_vel, dt, th, and th2
        # th = theta and th2 = theta + dt * ang_vel (new theta)

        # doubled checked this should be fine too 
        ang_vel_safe = ang_vel if abs(ang_vel) > 1e-6 else 1e-6

        Jac2[0,0] = (1/ang_vel_safe)             *(-np.sin(th) + np.sin(th2)) 
        Jac2[0,1] = (-lin_vel/(ang_vel_safe**2)) *(-np.sin(th) + np.sin(th2))

        Jac2[1,0] = (1/ang_vel_safe)             *(np.cos(th) - np.cos(th2)) 
        Jac2[1,1] = (-lin_vel/(ang_vel_safe**2)) *(np.cos(th) - np.cos(th2))

        Jac2[2,0] = 0                 
        Jac2[2,1] = dt

        # Derivative of x,y,theta w.r.t. left_speed, right_speed
        Jac = Jac2 @ Jac1

        # Compute covariance
        cov = np.diag((drive_meas.left_cov, drive_meas.right_cov))
        cov = Jac @ cov @ Jac.T
        
        return cov
