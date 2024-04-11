"""Compute kinematic properties of a robot using PyKDL."""

import numpy as np
import kdl_parser_py.urdf
import PyKDL as kdl
import urdf_parser_py.urdf

from .spatial import Rotation, Transform


class KDLModel:
    def __init__(self, urdf, root_frame, tip_frame):
        self.urdf = urdf
        self.root = root_frame
        self.tip = tip_frame
        self.init_kdl_solvers()

    @classmethod
    def from_urdf_file(cls, urdf_path, root_frame, tip_frame):
        urdf = urdf_parser_py.urdf.URDF.from_xml_file(urdf_path)
        return cls(urdf, root_frame, tip_frame)

    @classmethod
    def from_parameter_server(cls, root_frame, tip_frame):
        urdf = urdf_parser_py.urdf.URDF.from_parameter_server()
        return cls(urdf, root_frame, tip_frame)

    def init_kdl_solvers(self):
        _, tree = kdl_parser_py.urdf.treeFromUrdfModel(self.urdf)
        self.chain = tree.getChain(self.root, self.tip)
        self.fk_pos_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.fk_vel_solver = kdl.ChainFkSolverVel_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

    def pose(self, frame, q):
        link_index = self._get_link_index(frame)
        jnt_array = joints_to_kdl_jnt_array(q)
        res = kdl.Frame()
        self.fk_pos_solver.JntToCart(jnt_array, res, link_index)
        return kdl_frame_to_transform(res)

    def velocities(self, frame, q, dq):
        link_index = self.link_names.index(frame)
        jnt_array_vel = kdl.JntArrayVel(
            joints_to_kdl_jnt_array(q), joints_to_kdl_jnt_array(dq)
        )
        res = kdl.FrameVel()
        self.fk_vel_solver.JntToCart(jnt_array_vel, res, link_index)
        d = res.deriv()
        linear, angular = np.r_[d[0], d[1], d[2]], np.r_[d[3], d[4], d[5]]
        return linear, angular

    def jacobian(self, q):
        jnt_array = joints_to_kdl_jnt_array(q)
        J = kdl.Jacobian(self.chain.getNrOfJoints())
        self.jac_solver.JntToJac(jnt_array, J)
        return kdl_mat_to_array(J)

    def _get_link_index(self, frame):
        return self.urdf.links.index(self.urdf.link_map[frame])


def joints_to_kdl_jnt_array(q):
    jnt_array = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        jnt_array[i] = q_i
    return jnt_array


def kdl_frame_to_transform(f):
    rotation = Rotation.from_matrix(
        np.array(
            [
                [f.M[0, 0], f.M[0, 1], f.M[0, 2]],
                [f.M[1, 0], f.M[1, 1], f.M[1, 2]],
                [f.M[2, 0], f.M[2, 1], f.M[2, 2]],
            ]
        )
    )
    translation = np.r_[f.p[0], f.p[1], f.p[2]]
    return Transform(rotation, translation)


def kdl_mat_to_array(m):
    mat = np.zeros((m.rows(), m.columns()))
    for i in range(m.rows()):
        for j in range(m.columns()):
            mat[i, j] = m[i, j]
    return mat
