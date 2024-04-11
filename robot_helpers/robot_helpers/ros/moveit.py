import moveit_commander
from moveit_msgs.msg import CollisionObject
import numpy as np

from robot_helpers.ros.conversions import to_mesh_msg, to_pose_msg
from robot_helpers.spatial import Transform


class MoveItClient:
    def __init__(self, planning_group):
        self.planning_group = planning_group
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(self.planning_group)

    def goto(self, target, velocity_scaling=0.1, acceleration_scaling=0.1):
        _, plan = self.plan(target, velocity_scaling, acceleration_scaling)
        success = self.execute(plan)
        return success

    def plan(self, target, velocity_scaling=0.1, acceleration_scaling=0.1):
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)

        if isinstance(target, Transform):
            a=to_pose_msg(target)
            print(a)
            self.move_group.set_pose_target(to_pose_msg(target))
            print("cartesian control")
        elif isinstance(target, (list, np.ndarray)):
            print("joint control")
            self.move_group.set_joint_value_target(target)
        elif isinstance(target, str):
            self.move_group.set_named_target(target)
        elif isinstance(target,list):
            self.move_group.set_pose_target()
        else:
            raise ValueError

        success, plan, _, _ = self.move_group.plan()
        return success, plan

    def gotoL(self, target, velocity_scaling=0.1, acceleration_scaling=0.1):
        plan = self.planL(target, velocity_scaling, acceleration_scaling)
        success = self.execute(plan)
        return success

    def planL(self, target, velocity_scaling=0.1, acceleration_scaling=0.1):
        waypoints = [to_pose_msg(target)]
        plan, _ = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        state = self.robot.get_current_state()
        return self.move_group.retime_trajectory(
            state,
            plan,
            velocity_scaling_factor=velocity_scaling,
            acceleration_scaling_factor=acceleration_scaling,
            algorithm="time_optimal_trajectory_generation",
        )

    def execute(self, plan):
        success = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success


def create_collision_object_from_mesh(name, frame, pose, mesh):
    co = CollisionObject()
    co.operation = CollisionObject.ADD
    co.id = name
    co.header.frame_id = frame
    co.meshes = [to_mesh_msg(mesh)]
    co.mesh_poses = [to_pose_msg(pose)]
    return co


def create_collision_object_from_meshes(name, frame, poses, meshes):
    if poses is not list:
        poses = [poses] * len(meshes)
    co = CollisionObject()
    co.operation = CollisionObject.ADD
    co.id = name
    co.header.frame_id = frame
    co.meshes = [to_mesh_msg(mesh) for mesh in meshes]
    co.mesh_poses = [to_pose_msg(pose) for pose in poses]
    return co
