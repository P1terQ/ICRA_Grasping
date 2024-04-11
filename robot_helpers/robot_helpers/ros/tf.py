import rospy
import tf2_ros

from .conversions import *


_buffer, _listener, _broadcaster = None, None, None


def _init_listener():
    global _buffer, _listener
    _buffer = tf2_ros.Buffer()
    _listener = tf2_ros.TransformListener(_buffer)


def _init_broadcaster():
    global _broadcaster
    _broadcaster = tf2_ros.TransformBroadcaster()


def init():
    _init_listener()
    _init_broadcaster()
    rospy.sleep(1.0)  # wait for connections to be established


def lookup(target_frame, source_frame, time=rospy.Time(0), timeout=rospy.Duration(1.0)):
    if not _listener:
        _init_listener()
    msg = _buffer.lookup_transform(target_frame, source_frame, time, timeout)
    return from_transform_msg(msg.transform)


def broadcast(transform, target_frame, source_frame):
    if not _broadcaster:
        _init_broadcaster()
    msg = geometry_msgs.msg.TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = target_frame
    msg.child_frame_id = source_frame
    msg.transform = to_transform_msg(transform)
    _broadcaster.sendTransform(msg)
