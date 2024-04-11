import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
def interpolate_euler_angles(euler_start, euler_end, n_interpolations):
    """
    Interpolates n_interpolations times between two sets of Euler angles.

    Parameters:
    euler_start: array-like, shape (3,)
        The starting Euler angles in degrees.
    euler_end: array-like, shape (3,)
        The ending Euler angles in degrees.
    n_interpolations: int
        The number of interpolations to perform between the start and end rotations.

    Returns:
    interpolated_eulers: ndarray, shape (n_interpolations, 3)
        The interpolated Euler angles in degrees.
    """
    # Convert Euler angles to quaternions
    quat_start = R.from_euler('xyz', euler_start.squeeze(), degrees=False).as_quat()
    quat_end = R.from_euler('xyz', euler_end.squeeze(), degrees=False).as_quat()

    # Create the Slerp object with start and end rotations and times
    key_rotations = R.from_quat([quat_start, quat_end])
    slerp = Slerp([0, 1], key_rotations)

    # Generate the interpolation times (excluding 0 and 1)
    interpolation_times = np.linspace(0, 1, num=n_interpolations)[1:-1]

    # Interpolate to find the rotations at intermediate times
    if interpolation_times.size is not 0:
        interpolated_rotations = slerp(interpolation_times)
        interpolated_eulers = interpolated_rotations.as_euler('xyz', degrees=False)
        interpolated_eulers = np.concatenate([euler_start, interpolated_eulers, euler_end])
        return interpolated_eulers
    # Convert the interpolated quaternions back to Euler angles in degrees
    else:
        return np.concatenate([euler_start,euler_end])


def linear_interpolate_numpy(start_vector, end_vector, n_interpolations):
    """
    Perform linear interpolation between two numpy arrays.

    :param start_vector: A numpy array representing the start vector.
    :param end_vector: A numpy array representing the end vector.
    :param n_interpolations: The number of interpolations or points to generate.
    :return: A list of numpy arrays representing the interpolated vectors.
    """
    # Generating new interpolation steps
    start_vector = start_vector.flatten()
    end_vector = end_vector.flatten()
    steps = np.linspace(0, 1, n_interpolations)

    # Interpolating for each step
    interpolated_vectors = [start_vector + (end_vector - start_vector) * step for step in steps]

    return interpolated_vectors
def slerp(init_pose,target_pose,n):
    linear_interp = linear_interpolate_numpy(init_pose[:,:3],target_pose[:,:3],n)
    rot_inerp=interpolate_euler_angles(init_pose[:,3:],target_pose[:,3:],n)
    
    return linear_interp,rot_inerp

# def update_six_d_path_numpy(original_path, start_position, end_position, current_position,tolerance):
#     """
#     Updates a six-dimensional path (position and orientation) based on the current position,
#     with all inputs being NumPy arrays.
#     """
#     # Check if the current position is within the tolerance of the end position
#     if np.linalg.norm(current_position[:3] - end_position[:3]) <= tolerance:
#         return original_path, 0

#     # Replan the path if the current position is off the path
#     pos_path = original_path[:, :3]
#     ori_path = original_path[:, 3:]
#     if not np.any(np.linalg.norm(pos_path - current_position[:,:3], axis=1) <= tolerance):
#         n = len(original_path) - 1
#         new_pos_path = np.linspace(current_position[:3], end_position[:3], n)
#         new_ori_path = interpolate_euler_angles(current_position[3:], end_position[3:], n)

#         updated_path = np.hstack((new_pos_path, new_ori_path))
#     else:
#         updated_path = original_path

#     # Remove passed path points
#     closest_point_index = np.argmin(np.linalg.norm(pos_path - current_position[:,:3], axis=1))
#     updated_path = updated_path[closest_point_index:]

#     # Determine the next target position
#     next_target_index = 0
#     next_target = updated_path[next_target_index]

#     return updated_path, next_target
# def update_path(original_path,start_position,end_position,current_position,n):
#     new_path=slerp(current_position,end_position,n)
#     if n==1:
#         next_pos=new_path[0,:]
#     else:
#         next_pos=new_path[1,:]
#     return new_path,next_pos
# # """
# original_path_6d = np.array([
#     [0, 0, 0, 0, 0, 0],
#     [2, 2, 2, 0.1, 0.1, 0.1],
#     [4, 4, 4, 0.2, 0.2, 0.2],
#     [6, 6, 6, 0.3, 0.3, 0.3],
#     [8, 8, 8, 0.4, 0.4, 0.4],
#     [10, 10, 10, 0.5, 0.5, 0.5]
# ])
# start_pos_6d = np.array([0, 0, 0, 0, 0, 0])
# end_pos_6d = np.array([10, 10, 10, 0.5, 0.5, 0.5])
# current_pos_6d = np.array([3, 3, 3, 0.15, 0.15, 0.15])
# tolerance_6d = 0.5

# updated_path_6d, next_target_6d = update_six_d_path_numpy(original_path_6d, start_pos_6d, end_pos_6d, current_pos_6d, tolerance_6d)
# print("Updated Path:\n", updated_path_6d)
# print("Next Target:", next_target_6d)
# """

