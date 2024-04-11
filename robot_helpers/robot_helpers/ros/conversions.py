import numpy as np

import geometry_msgs.msg
import sensor_msgs.msg
from shape_msgs.msg import Mesh, MeshTriangle
import std_msgs.msg

from robot_helpers.perception import CameraIntrinsic
from robot_helpers.spatial import Rotation, Transform


def from_camera_info_msg(msg):
    fx, fy, cx, cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
    return CameraIntrinsic(msg.width, msg.height, fx, fy, cx, cy)


def from_point_msg(msg):
    return np.r_[msg.x, msg.y, msg.z]


def from_quat_msg(msg):
    return Rotation.from_quat([msg.x, msg.y, msg.z, msg.w])


def from_pose_msg(msg):
    position = from_point_msg(msg.position)
    orientation = from_quat_msg(msg.orientation)
    return Transform(orientation, position)


def from_transform_msg(msg):
    translation = from_vector3_msg(msg.translation)
    rotation = from_quat_msg(msg.rotation)
    return Transform(rotation, translation)


def from_twist_msg(msg):
    linear = from_vector3_msg(msg.linear)
    angular = from_vector3_msg(msg.angular)
    return np.r_[linear, angular]


def from_vector3_msg(msg):
    return np.r_[msg.x, msg.y, msg.z]


def to_camera_info_msg(info):
    msg = sensor_msgs.msg.CameraInfo()
    msg.height = info.height
    msg.width = info.width
    msg.K = info.K.flatten().tolist()
    return msg


def to_color_msg(color):
    msg = std_msgs.msg.ColorRGBA()
    msg.r = color[0]
    msg.g = color[1]
    msg.b = color[2]
    msg.a = color[3] if len(color) == 4 else 1.0
    return msg


def to_mesh_msg(mesh):
    msg = Mesh()
    msg.triangles = [MeshTriangle(vertex_indices=triangle) for triangle in mesh.faces]
    msg.vertices = [to_point_msg(vertex) for vertex in mesh.vertices]
    return msg


def to_point_msg(point):
    msg = geometry_msgs.msg.Point()
    msg.x = point[0]
    msg.y = point[1]
    msg.z = point[2]
    return msg


def to_pose_msg(transform):
    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(transform.translation)
    msg.orientation = to_quat_msg(transform.rotation)
    return msg


def to_pose_stamped_msg(transform, frame):
    msg = geometry_msgs.msg.PoseStamped()
    msg.header.frame_id = frame
    msg.pose = to_pose_msg(transform)
    return msg


def to_quat_msg(orientation):
    quat = orientation.as_quat()
    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


def to_transform_msg(transform):
    msg = geometry_msgs.msg.Transform()
    msg.translation = to_vector3_msg(transform.translation)
    msg.rotation = to_quat_msg(transform.rotation)
    return msg


def to_transform_stamped_msg(transform, target_frame, source_frame):
    msg = geometry_msgs.msg.TransformStamped()
    msg.header.frame_id = target_frame
    msg.child_frame_id = source_frame
    msg.transform = to_transform_msg(transform)
    return msg


def to_twist_msg(dx):
    msg = geometry_msgs.msg.Twist()
    msg.linear = to_vector3_msg(dx[:3])
    msg.angular = to_vector3_msg(dx[3:])
    return msg


def to_vector3_msg(vector3):
    msg = geometry_msgs.msg.Vector3()
    msg.x = vector3[0]
    msg.y = vector3[1]
    msg.z = vector3[2]
    return msg
