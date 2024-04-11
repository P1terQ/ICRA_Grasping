from actionlib import SimpleActionClient
import numpy as np
import rospy

import franka_msgs.msg
from franka_gripper.msg import *
from sensor_msgs.msg import JointState


class PandaArmClient:
    def __init__(self):
        rospy.Subscriber(
            "/franka_state_controller/franka_states",
            franka_msgs.msg.FrankaState,
            self._state_cb,
            queue_size=1,
        )
        self._error_recovery_client = SimpleActionClient(
            "franka_control/error_recovery",
            franka_msgs.msg.ErrorRecoveryAction,
        )
        self._error_recovery_client.wait_for_server()

        rospy.loginfo("Connected to franka_control/error_recovery.")

    @property
    def has_error(self):
        return self._state_msg.robot_mode == 4

    def get_state(self):
        q = np.asarray(self._state_msg.q)
        dq = np.asarray(self._state_msg.dq)
        return q, dq

    def recover(self):
        msg = franka_msgs.msg.ErrorRecoveryGoal()
        self._error_recovery_client.send_goal_and_wait(msg)
        rospy.loginfo("Recovered from errors.")

    def _state_cb(self, msg):
        self._state_msg = msg


class PandaGripperClient:
    def __init__(self, ns="/franka_gripper"):
        self._connect_to_action_servers(ns)
        rospy.Subscriber(f"{ns}/joint_states", JointState, self._joint_state_cb)

    def move(self, width, speed=0.1):
        msg = MoveGoal(width, speed)
        self._move_client.send_goal_and_wait(msg)

    def grasp(self, width=0.0, e_inner=0.1, e_outer=0.1, speed=0.1, force=10.0):
        msg = GraspGoal(width, GraspEpsilon(e_inner, e_outer), speed, force)
        self._grasp_client.send_goal_and_wait(msg)

    def home(self):
        msg = HomingGoal()
        self._homing_client.send_goal_and_wait(msg)

    def read(self):
        return self._joint_state_msg.position[0] + self._joint_state_msg.position[1]

    def _connect_to_action_servers(self, ns):
        self._move_client = SimpleActionClient(f"{ns}/move", MoveAction)
        self._move_client.wait_for_server()
        rospy.loginfo(f"Connected to {ns}/move action server.")

        self._grasp_client = SimpleActionClient(f"{ns}/grasp", GraspAction)
        self._grasp_client.wait_for_server()
        rospy.loginfo(f"Connected to {ns}/grasp action server.")

        self._homing_client = SimpleActionClient(f"{ns}/homing", HomingAction)
        self._homing_client.wait_for_server()
        rospy.loginfo(f"Connected to {ns}/homing action server.")

    def _joint_state_cb(self, msg):
        self._joint_state_msg = msg
