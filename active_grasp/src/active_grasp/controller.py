from controller_manager_msgs.srv import *
import copy
import cv_bridge
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from sensor_msgs.msg import Image
import trimesh
from scipy.spatial.transform import Rotation as R
from .bbox import from_bbox_msg
from .timer import Timer
from active_grasp.srv import Reset, ResetRequest
from robot_helpers.ros import tf
from robot_helpers.ros.conversions import *
from robot_helpers.ros.panda import PandaArmClient, PandaGripperClient
from robot_helpers.ros.moveit import MoveItClient, create_collision_object_from_mesh
from robot_helpers.spatial import Rotation, Transform
from robot_helpers.ros.conversions import to_mesh_msg, to_pose_msg
from vgn.utils import look_at, cartesian_to_spherical, spherical_to_cartesian   
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

from franka_control_wrappers.panda_commander import PandaCommander
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
def interpolate_euler_angles(quat_start, quat_end, n_interpolations):
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
    # quat_start = R.from_euler('xyz', euler_start.squeeze(), degrees=False).as_quat()
    # quat_end = R.from_euler('xyz', euler_end.squeeze(), degrees=False).as_quat()

    # Create the Slerp object with start and end rotations and times
    key_rotations = R.from_quat([quat_start, quat_end])
    slerp = Slerp([0, 1], key_rotations)

    # Generate the interpolation times (excluding 0 and 1)
    interpolation_times = np.linspace(0, 1, num=n_interpolations)[1:-1]
    
    # Interpolate to find the rotations at intermediate times
    if interpolation_times.size is not 0:
        interpolated_rotations = slerp(interpolation_times)
        interpolated_quat=interpolated_rotations.as_quat()
        # interpolated_quat = np.concatenate([interpolated_quat])
        interpolated_quat=interpolated_quat.tolist()
        interpolated_quat.append(quat_end)
        return interpolated_quat
    # Convert the interpolated quaternions back to Euler angles in degrees
    else:
        return np.concatenate([quat_start,quat_start])


def linear_interpolate_numpy(start_vector, end_vector, n_interpolations):
    """
    Perform linear interpolation between two numpy arrays.

    :param start_vector: A numpy array representing the start vector.
    :param end_vector: A numpy array representing the end vector.
    :param n_interpolations: The number of interpolations or points to generate.
    :return: A list of numpy arrays representing the interpolated vectors.
    """
    # Generating new interpolation steps
    start_vector = np.array(start_vector)
    end_vector = np.array(end_vector)
    steps = np.linspace(0, 1, n_interpolations)
    interpolated_vectors =[]
    # Interpolating for each step
    for step in steps:
        temp=start_vector + (end_vector - start_vector) * step  
        interpolated_vectors.append(temp.tolist())
    interpolated_vectors=interpolated_vectors[1:]
    return interpolated_vectors
def slerp(init_pose,target_pose,n):
    linear_interp = linear_interpolate_numpy(init_pose[:3],target_pose[:3],n)
    rot_inerp=interpolate_euler_angles(init_pose[3:],target_pose[3:],n)
    
    return linear_interp,rot_inerp
# def lerp(start, end, t):
#     """线性插值"""
#     return start + (end - start) * t


# def normalize(v):
#     """标准化向量或四元数"""
#     norm = np.linalg.norm(v, axis=-1, keepdims=True)
#     return v / norm

# def slerp(q1, q2, t):
#     q1 = normalize(q1)
#     q2 = normalize(q2)
    
#     dot = np.sum(q1 * q2, axis=-1)
#     dot = np.clip(dot, -1.0, 1.0)
    
#     theta = np.arccos(dot) * t
#     q2_perp = q2 - q1 * dot[..., np.newaxis]  # 保证 dot 是正确的形状
#     q2_perp_norm = normalize(q2_perp)
    
#     return np.cos(theta)[..., np.newaxis] * q1 + np.sin(theta)[..., np.newaxis] * q2_perp_norm

def quat_and_translation_to_matrix(quat, translation):
    """
    将四元数和平移向量转换为4x4的变换矩阵。
    
    参数:
    quat -- 四元数，格式为[x, y, z, w]，其中w是实部。
    translation -- 平移向量，格式为[x, y, z]。
    
    返回:
    4x4的变换矩阵。
    """
    # 从四元数创建旋转矩阵
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()  # 获得3x3的旋转矩阵
    
    # 创建4x4的变换矩阵
    transform_matrix = np.eye(4)  # 创建一个4x4的单位矩阵
    transform_matrix[:3, :3] = rotation_matrix  # 设置旋转部分
    transform_matrix[:3, 3] = translation  # 设置平移部分
    
    return transform_matrix
class GraspController:
    def __init__(self, policy):
        self.policy = policy
        self.load_parameters()
        self.init_service_proxies()
        self.init_robot_connection()
        self.init_moveit()
        self.init_camera_stream()
        self.pregrasp_pose = np.asarray([0,-0.5,0.55, 2**0.5/2, -2**0.5/2, 0, 0])
        self.pc = PandaCommander(group_name='panda_arm')
        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False

        # self.need_avoid = False
        
    
    def __recover_robot_from_error(self):
        rospy.logerr('Recovering')
        self.pc.recover()
        rospy.logerr('Done')
        self.ROBOT_ERROR_DETECTED = False

    def __robot_state_callback(self, msg):
        self.robot_state = msg
        if any(self.robot_state.cartesian_collision):
            if not self.ROBOT_ERROR_DETECTED:
                rospy.logerr('Detected Cartesian Collision')
            self.ROBOT_ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ROBOT_ERROR_DETECTED:
                    rospy.logerr('Robot Error Detected')
                self.ROBOT_ERROR_DETECTED = True

    def load_parameters(self):
        self.base_frame = rospy.get_param("~base_frame_id") # panda_link0
        self.T_grasp_ee = Transform.from_list(rospy.get_param("~ee_grasp_offset")).inv()    #! yaml中定义得offset
        self.cam_frame = rospy.get_param("~camera/frame_id") # camera_depth_optical_frame
        self.depth_topic = rospy.get_param("~camera/depth_topic")  # '/camera/depth/image_rect_raw'
        self.min_z_dist = rospy.get_param("~camera/min_z_dist") # 0.3
        self.control_rate = rospy.get_param("~control_rate") # 30
        self.linear_vel = rospy.get_param("~linear_vel")  # 0.05
        self.policy_rate = rospy.get_param("policy/rate") # 4

    def init_service_proxies(self):
        self.reset_env = rospy.ServiceProxy("reset", Reset)
        self.switch_controller = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )

    def init_robot_connection(self):
        self.arm = PandaArmClient()
        self.gripper = PandaGripperClient()
        topic = rospy.get_param("cartesian_velocity_controller/topic")  # '/cartesian_velocity_controller/set_command'
        self.cartesian_vel_pub = rospy.Publisher(topic, Twist, queue_size=10) # certesian velocity pub

    def init_moveit(self):
        self.moveit = MoveItClient("panda_arm")
        rospy.sleep(1.0)  # Wait for connections to be established.
        self.moveit.move_group.set_planner_id("RRTConnectkConfigDefault")  #! RRTConnectkConfigDefault RRTstarkConfigDefault
        self.moveit.move_group.set_planning_time(30.0)

    def switch_to_cartesian_velocity_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["cartesian_velocity_controller"]
        req.stop_controllers = ["position_joint_trajectory_controller"]
        req.strictness = 1
        self.switch_controller(req)

    def switch_to_joint_trajectory_control(self):
        req = SwitchControllerRequest()
        req.start_controllers = ["position_joint_trajectory_controller"]
        req.stop_controllers = ["cartesian_velocity_controller"]
        req.strictness = 1
        self.switch_controller(req)

    def init_camera_stream(self):
        self.cv_bridge = cv_bridge.CvBridge()
        rospy.Subscriber(self.depth_topic, Image, self.sensor_cb, queue_size=1)

    def sensor_cb(self, msg):
        self.latest_depth_msg = msg

    def run(self):
        self.pc.recover()
        

        q_place = np.asarray([1.024924061756385, 0.6902756068581029, 0.23736947749162976, -1.5500040081760835, -0.15436248639557096, 2.2270590113004047, 0.5123070080164406])   # 初始位置
        # 判断当前机械臂关节位置和q_place是否相近

        # 获取当前关节位置
        joint_positions_now = self.arm.get_state()[0]

        # 计算q_place和joint_position_now之前的差值的范数
        diff_norm = np.linalg.norm(joint_positions_now - q_place)


        if diff_norm<0.5:
        
            #! 一开始会报错can't accept new action goals. Controller is not running
            joint_mid_avoid = np.asarray([0.8081277828028325, 0.31377393442264456, 0.4676299028856712, -1.1309202249160595, -0.08720591376124719, 1.391054532332668, 0.8837474233436782])   # 初始位置
            successavoid, planavoid = self.moveit.plan(joint_mid_avoid, 0.1, 0.1) 
            self.moveit.execute(planavoid)
            # self.need_avoid = False
            # self.moveit.execute(plan1)
    # 0.8081277828028325, 0.31377393442264456, 0.4676299028856712, -1.1309202249160595, -0.08720591376124719, 1.391054532332668, 0.8837474233436782

        joint_mid = np.asarray([-0.012278128726616162, -0.7834047260627022, -0.0008402333314644925, -2.3615974901829904, -0.015725505022539034, 1.5662728869946938, 0.7658006468498042])   # 初始位置
        success, plan1 = self.moveit.plan(joint_mid, 0.1, 0.1) 
        self.moveit.execute(plan1)

        bbox = self.reset() #! request "reset" service from hw_reset_client
        self.switch_to_cartesian_velocity_control()

        with Timer("search_time"):
            grasp = self.search_grasp(bbox)

        if grasp:
            self.switch_to_joint_trajectory_control()
            with Timer("grasp_time"):
                res = self.execute_grasp(grasp)
        else:
            res = "aborted"

        print("result: ", res)

        return self.collect_info(res)

    def reset(self):
        Timer.reset()
        self.moveit.scene.clear()
        res = self.reset_env(ResetRequest())
        rospy.sleep(1.0)  # Wait for the TF tree to be updated.
        return from_bbox_msg(res.bbox)

    def search_grasp(self, bbox):
        self.view_sphere = ViewHalfSphere(bbox, self.min_z_dist)
        self.policy.activate(bbox, self.view_sphere)
        timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.send_vel_cmd)
        r = rospy.Rate(self.policy_rate)
        while not self.policy.done:
            img, pose, q = self.get_state()
            self.policy.update(img, pose, q)
            r.sleep()
        rospy.sleep(0.2)  # Wait for a zero command to be sent to the robot.
        self.policy.deactivate()
        timer.shutdown()
        return self.policy.best_grasp

    def get_state(self):
        q, _ = self.arm.get_state()
        msg = copy.deepcopy(self.latest_depth_msg)
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        pose = tf.lookup(self.base_frame, self.cam_frame, msg.header.stamp)
        return img, pose, q

    def send_vel_cmd(self, event):
        if self.policy.x_d is None or self.policy.done:
            cmd = np.zeros(6)
        else:
            x = tf.lookup(self.base_frame, self.cam_frame)
            cmd = self.compute_velocity_cmd(self.policy.x_d, x) #! track nbv

        self.cartesian_vel_pub.publish(to_twist_msg(cmd))

    def compute_velocity_cmd(self, x_d, x):
        r, theta, phi = cartesian_to_spherical(x.translation - self.view_sphere.center)
        e_t = x_d.translation - x.translation
        e_n = (x.translation - self.view_sphere.center) * (self.view_sphere.r - r) / r
        linear = 1.0 * e_t + 6.0 * (r < self.view_sphere.r) * e_n
        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, self.linear_vel) / scale
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv()
        angular = 1.0 * angular.as_rotvec()
        return np.r_[linear, angular]
 
    def execute_grasp(self, grasp):
        self.create_collision_scene()
        T_base_grasp = self.postprocess(grasp.pose) #! T_grasp相对base
        self.gripper.move(0.08) # 张开

        # T_base_prepare = Transform.t_[0, 0, 0.2] * T_base_grasp * self.T_grasp_ee  #! T_ee相对base(高0.06)
        # success, plan1 = self.moveit.plan(T_base_prepare, 0.2, 0.2) #!
        # # self.moveit.gotoL( T_base_prepare  )
        # self.moveit.execute(plan1)
        # rospy.sleep(0.5)
        # print("prepare!!!!!!")
        # joint_mid = np.asarray([0.05545767353082958, -0.006870736620928113, -0.03708130711750175, -2.4093051281201547, 0.020952787576450242, 2.455550435225169, 0.8371839164881923])
        # success, plan1 = self.moveit.plan(joint_mid, 0.1, 0.1) #!
        # # self.moveit.gotoL( T_base_prepare  )
        # self.moveit.execute(plan1)
        # # print("plan1: ", plan1)
        # rospy.sleep(0.5)
        # print("prepare!!!!!!")
        eef_link = self.moveit.move_group.get_end_effector_link()
        pose = self.moveit.move_group.get_current_pose(eef_link)
        x=pose.pose.position.x
        y=pose.pose.position.y
        z=pose.pose.position.z
        q1=pose.pose.orientation.x
        q2=pose.pose.orientation.y
        q3=pose.pose.orientation.z
        w=pose.pose.orientation.w


        # T_base_approach = T_base_grasp * Transform.t_[0.03, 0.0, 0] * self.T_grasp_ee #! T_ee相对base(高0.06)
        T_base_approach = T_base_grasp * Transform.t_[0.03, -0.01, -0.3] * self.T_grasp_ee #! T_ee相对base(高0.06)
        targe_pose=to_pose_msg(T_base_approach)
        target_x=targe_pose.position.x
        target_y=targe_pose.position.y
        target_z=targe_pose.position.z
        
        target_q1=targe_pose.orientation.x
        target_q2=targe_pose.orientation.y
        target_q3=targe_pose.orientation.z
        target_w=targe_pose.orientation.w
        
        # ori_p=quat_and_translation_to_matrix([q1,q2,q3,w],[x,y,z])
        ori_p=[x,y,z,q1,q2,q3,w]
        target_p=[target_x,target_y,target_z,target_q1,target_q2,target_q3,target_w]
        # target_p=quat_and_translation_to_matrix([ target_q1,target_q2,target_q3,target_w],[target_x,target_y,target_z])
        pose_list,ori_list=slerp(ori_p,target_p,7)
        pose_form=T_base_approach*Transform.t_[0,0,0]
        success0 = False
        # for i  in range (1):    # 5
        #      # 假设欧拉角是以度为单位的
            
        #     pose_form.translation=pose_list[i]
        #     pose_form.orientation=ori_list[i]
        #     success, plan2 = self.moveit.plan(pose_form, 0.1, 0.1) #!
        #     self.moveit.scene.clear()
        #     self.moveit.execute(plan2)
        pose_form.translation=pose_list[0]
        pose_form.orientation=ori_list[0]
        success0, plan2 = self.moveit.plan(pose_form, 0.1, 0.1) #!
        self.moveit.scene.clear()
        self.moveit.execute(plan2)
        # success, plan2 = self.moveit.plan(T_base_approach, 0.1, 0.1) #!
        pose_form.translation=pose_list[1]
        pose_form.orientation=ori_list[1]
        success1, plan2 = self.moveit.plan(pose_form, 0.1, 0.1) #!
        self.moveit.scene.clear()
        self.moveit.execute(plan2)
        # success, plan2 = self.moveit.plan(T_base_approach, 0.1, 0.1) #!
        pose_form.translation=pose_list[2]
        pose_form.orientation=ori_list[2]
        success2, plan2 = self.moveit.plan(pose_form, 0.1, 0.1) #!
        self.moveit.scene.clear()
        self.moveit.execute(plan2)
        pose_form.translation=pose_list[3]
        pose_form.orientation=ori_list[3]
        success3, plan2 = self.moveit.plan(pose_form, 0.1, 0.1) #!
        self.moveit.scene.clear()
        self.moveit.execute(plan2)
        pose_form.translation=pose_list[4]
        pose_form.orientation=ori_list[4]
        success4, plan2 = self.moveit.plan(pose_form, 0.1, 0.1) #!
        self.moveit.scene.clear()
        self.moveit.execute(plan2)
        pose_form.translation=pose_list[5]
        pose_form.orientation=ori_list[5]
        success, plan2 = self.moveit.plan(pose_form, 0.1, 0.1) #!
        self.moveit.scene.clear()
        self.moveit.execute(plan2)

        if success and not self.ROBOT_ERROR_DETECTED:
            self.moveit.scene.clear()
            # self.moveit.execute(plan2)
            rospy.sleep(0.5)  # Wait for the planning scene to be updated
            print("catch!!!!!!")
            # self.moveit.gotoL( Transform.t_[0, 0, -0.4] * T_base_grasp * self.T_grasp_ee  )   #! goto T_ee相对base(这边可以家加一个偏置)
            # T_target = T_base_grasp * self.T_grasp_ee
            # print(T_target.rotation)
            # print(T_target.translation)
            self.moveit.gotoL( T_base_grasp * Transform.t_[0.03, 0.04,  -0.18] * self.T_grasp_ee  ) #! 抓取位置
            print("T_base_grasp: ", T_base_grasp.rotation)
            rospy.sleep(0.5) 
            self.gripper.grasp()
            # T_base_retreat = Transform.t_[0, 0, 0.05] * T_base_grasp * self.T_grasp_ee  
            T_base_retreat = Transform.t_[0, 0, 0.4] * T_base_grasp * self.T_grasp_ee   #! 上升
            self.moveit.gotoL(T_base_retreat)
            rospy.sleep(1.0)  #! Wait to see whether the object slides out of the hand
            success = self.gripper.read() > 0.002

            # self.pc.goto_named_pose('grip_ready', velocity=0.25)


            placeA = np.asarray([0.024383024292818287, -0.2622615840351372, 0.03071162108569362, -2.0385405393901626, -0.014837411449187328, 1.8307929479296123, 0.7456546647372213])   # 初始位置
            successA, planA = self.moveit.plan(placeA, 0.1, 0.1) 
            self.moveit.execute(planA)

            placeB = np.asarray([0.4877395607135449, -0.051506965049722436, 0.4795486804121064, -1.6267091484917113, 0.10096644302871491, 1.591474315166473, 0.9616774624771905])   # 初始位置
            successB, placeB = self.moveit.plan(placeB, 0.1, 0.1) 
            self.moveit.execute(placeB)

            placeC = np.asarray([1.024924061756385, 0.6902756068581029, 0.23736947749162976, -1.5500040081760835, -0.15436248639557096, 2.2270590113004047, 0.5123070080164406])   # 初始位置
            successC, placeC = self.moveit.plan(placeC, 0.1, 0.1) 
            self.moveit.execute(placeC)
            #? 
            # self.pc.goto_named_pose('drop_box', velocity=0.25)
            # place= np.asarray([0.8338639966099323, 0.7813643499006305, 0.6722988716175682, -1.6374551276209672, -0.6227699804272917, 2.1757760300636293, 0.5606248298370176])   # 初始位置
            # success, plan1 = self.moveit.plan(place, 0.1, 0.1) 
            # self.moveit.execute(plan1)

            self.gripper.move(0.1)  # 开爪子

            # self.need_avoid = True

            # self.pc.goto_named_pose('grip_ready', velocity=0.25)
            return "succeeded" if success else "failed"
        else:
            self.pc.recover()
            self.pc.goto_named_pose('grip_ready', velocity=0.25)
            return "no_motion_plan_found"
        

    def create_collision_scene(self):
        # Segment support surface
        cloud = self.policy.tsdf.get_scene_cloud()
        cloud = cloud.transform(self.policy.T_base_task.as_matrix())
        _, inliers = cloud.segment_plane(0.01, 3, 1000)
        support_cloud = cloud.select_by_index(inliers)
        cloud = cloud.select_by_index(inliers, invert=True)
        # o3d.io.write_point_cloud(f"{time.time():.0f}.pcd", cloud)

        # Add collision object for the support
        self.add_collision_mesh("support", compute_convex_hull(support_cloud))

        # Cluster cloud
        labels = np.array(cloud.cluster_dbscan(eps=0.01, min_points=8))

        # Generate convex collision objects for each segment
        self.hulls = []
        for label in range(labels.max() + 1):
            segment = cloud.select_by_index(np.flatnonzero(labels == label))
            try:
                hull = compute_convex_hull(segment)
                name = f"object_{label}"
                self.add_collision_mesh(name, hull)
                self.hulls.append(hull)
            except:
                # Qhull fails in some edge cases
                pass

    def add_collision_mesh(self, name, mesh):
        frame, pose = self.base_frame, Transform.identity()
        co = create_collision_object_from_mesh(name, frame, pose, mesh)
        self.moveit.scene.add_object(co)

    def postprocess(self, T_base_grasp):
        rot = T_base_grasp.rotation
        if rot.as_matrix()[:, 0][0] < 0:  # Ensure that the camera is pointing forward
            T_base_grasp.rotation = rot * Rotation.from_euler("z", np.pi)
        T_base_grasp *= Transform.t_[0.0, 0.0, 0.01]
        return T_base_grasp

    def collect_info(self, result):
        points = [p.translation for p in self.policy.views]
        d = np.sum([np.linalg.norm(p2 - p1) for p1, p2 in zip(points, points[1:])])
        info = {
            "result": result,
            "view_count": len(points),
            "distance": d,
        }
        info.update(self.policy.info)
        info.update(Timer.timers)
        return info


def compute_convex_hull(cloud):
    hull, _ = cloud.compute_convex_hull()
    triangles, vertices = np.asarray(hull.triangles), np.asarray(hull.vertices)
    return trimesh.base.Trimesh(vertices, triangles)


class ViewHalfSphere:
    def __init__(self, bbox, min_z_dist):
        self.center = bbox.center
        self.r = 0.5 * bbox.size[2] + min_z_dist

    def get_view(self, theta, phi):
        eye = self.center + spherical_to_cartesian(self.r, theta, phi)
        up = np.r_[1.0, 0.0, 0.0]
        return look_at(eye, self.center, up)

    def sample_view(self):
        raise NotImplementedError
