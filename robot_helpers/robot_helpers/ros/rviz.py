from geometry_msgs.msg import Quaternion
import rospy
from visualization_msgs.msg import *

from .conversions import *


def create_arrow_marker(frame, start, end, scale, color, ns="", id=0):
    pose = Transform.identity()
    marker = create_marker(Marker.ARROW, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(start), to_point_msg(end)]
    return marker


def create_cube_marker(frame, pose, scale, color, ns="", id=0):
    return create_marker(Marker.CUBE, frame, pose, scale, color, ns, id)


def create_line_list_marker(frame, pose, scale, color, lines, ns="", id=0):
    marker = create_marker(Marker.LINE_LIST, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(point) for line in lines for point in line]
    return marker


def create_line_strip_marker(frame, pose, scale, color, points, ns="", id=0):
    marker = create_marker(Marker.LINE_STRIP, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(point) for point in points]
    return marker


def create_sphere_marker(frame, pose, scale, color, ns="", id=0):
    return create_marker(Marker.SPHERE, frame, pose, scale, color, ns, id)


def create_sphere_list_marker(frame, pose, scale, color, centers, ns="", id=0):
    marker = create_marker(Marker.SPHERE_LIST, frame, pose, scale, color, ns, id)
    marker.points = [to_point_msg(center) for center in centers]
    return marker


def create_mesh_marker(frame, mesh, pose, scale=None, color=None, ns="", id=0):
    marker = create_marker(Marker.MESH_RESOURCE, frame, pose, scale, color, ns, id)
    marker.mesh_resource = mesh
    return marker


def create_marker(type, frame, pose, scale=None, color=None, ns="", id=0):
    if scale is None:
        scale = [1, 1, 1]
    elif np.isscalar(scale):
        scale = [scale, scale, scale]
    if color is None:
        color = (1, 1, 1)
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.ns = ns
    msg.id = id
    msg.type = type
    msg.action = Marker.ADD
    msg.pose = to_pose_msg(pose)
    msg.scale = to_vector3_msg(scale)
    msg.color = to_color_msg(color)
    return msg


MOVE_AXIS = InteractiveMarkerControl.MOVE_AXIS
ROTATE_AXIS = InteractiveMarkerControl.ROTATE_AXIS


def create_6dof_ctrl(frame, name, pose, scale, markers):
    im = InteractiveMarker()
    im.header.frame_id = frame
    im.name = name
    im.pose = to_pose_msg(pose)
    im.scale = scale
    im.controls = [
        InteractiveMarkerControl(markers=markers, always_visible=True),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, x=1), interaction_mode=MOVE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, y=1), interaction_mode=MOVE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, z=1), interaction_mode=MOVE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, x=1), interaction_mode=ROTATE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, y=1), interaction_mode=ROTATE_AXIS
        ),
        InteractiveMarkerControl(
            orientation=Quaternion(w=1, z=1), interaction_mode=ROTATE_AXIS
        ),
    ]
    return im
