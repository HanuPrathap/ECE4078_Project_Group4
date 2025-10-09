

# functions from week 6 exercise 


import numpy as np

def get_distance_robot_to_goal(robot_state=np.zeros(3), goal=np.zeros(3)):
    robot_state = np.asarray(robot_state).reshape(-1)
    goal = np.asarray(goal).reshape(-1)
    if goal.shape[0] < 3:
        goal = np.hstack((goal[:2], 0.0))
    x_goal, y_goal, _ = goal
    x, y, _ = robot_state
    return np.hypot(x_goal - x, y_goal - y)

def get_angle_robot_to_goal(robot_state=np.zeros(3), goal=np.zeros(3)):
    robot_state = np.asarray(robot_state).reshape(-1)
    goal = np.asarray(goal).reshape(-1)
    if goal.shape[0] < 3:
        goal = np.hstack((goal[:2], 0.0))
    x_goal, y_goal, _ = goal
    x, y, theta = robot_state
    return clamp_angle(np.arctan2(y_goal - y, x_goal - x) - theta)



def clamp_angle(rad_angle=0, min_value=-np.pi, max_value=np.pi):
	"""
	Restrict angle to the range [min, max]
	:param rad_angle: angle in radians
	:param min_value: min angle value
	:param max_value: max angle value
	"""

	if min_value > 0:
		min_value *= -1

	angle = (rad_angle + max_value) % (2 * np.pi) + min_value

	return angle


