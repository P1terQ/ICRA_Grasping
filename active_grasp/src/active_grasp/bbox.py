import itertools
import numpy as np

import active_grasp.msg
from robot_helpers.ros.conversions import to_point_msg, from_point_msg


class AABBox:
    def __init__(self, bbox_min, bbox_max):
        self.min = np.asarray(bbox_min)
        self.max = np.asarray(bbox_max)
        self.center = 0.5 * (self.min + self.max)
        self.size = self.max - self.min

    @property
    def corners(self):
        return list(itertools.product(*np.vstack((self.min, self.max)).T))

    def is_inside(self, p):
        return np.all(p > self.min) and np.all(p < self.max)


def from_bbox_msg(msg):
    aabb_min = from_point_msg(msg.min)
    aabb_max = from_point_msg(msg.max)
    return AABBox(aabb_min, aabb_max)


def to_bbox_msg(bbox):
    msg = active_grasp.msg.AABBox()
    msg.min = to_point_msg(bbox.min)
    msg.max = to_point_msg(bbox.max)
    return msg
