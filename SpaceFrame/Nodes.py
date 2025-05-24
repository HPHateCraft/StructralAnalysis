from functools import cached_property
import bisect

from GlobalCoord import GlobalCoord

import numpy as np
from numpy.typing import NDArray 

class Nodes:
    
    R_XYZ = 1
    R_XZY = 2
    R_YXZ = 3
    R_YZX = 4
    R_ZXY = 5
    R_ZYX = 6
    
    def __init__(self) -> None:
        self._id: int = 0
        
        self.ids = []
        self.coord = []
        self.dof = []
        
        self.axes_changed = []
        self.rotation_order = []
        self.roll_angle_deg = []
        self.pitch_angle_deg = []
        self.yaw_angle_deg = []
        
    def _generate_id(self):
        id_ = self._id
        self.ids.append(id_)
        self._id += 1
        return id_

    def find_index_by_id(self, id_: int):
        index = bisect.bisect_left(self.ids, id_)
        if index < len(self.ids) and self.ids[index] == id_:
            return index  
        else:
            raise ValueError(f"Node with id: {id_} doesn't exist.")
    
    def generate_node(self, x: float, y: float, z: float, ux: bool=True, uy: bool=True, uz: bool=True, rx: bool=True, ry: bool=True, rz: bool=True): 
                
        id_ = self._generate_id()

        self.coord.append([x, y, z])
        self.dof.append([ux, uy, uz, rx, ry, rz])
        self.axes_changed.append(False)
        self.rotation_order.append(self.R_XYZ)
        self.roll_angle_deg.append(0.0)
        self.pitch_angle_deg.append(0.0)
        self.yaw_angle_deg.append(0.0)
        return id_
    
    def change_axes(self, node_id: int, rotation_order: int = R_YZX, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0):
        index = self.find_index_by_id(node_id)
        
        self.axes_changed[index] = True
        self.rotation_order[index] = rotation_order
        self.roll_angle_deg[index] = rx
        self.pitch_angle_deg[index] = ry
        self.yaw_angle_deg[index] = rz
        

class NodesManager:
    
    def __init__(self, nodes: Nodes):
        self._nodes = nodes
    
    def find_index_by_id(self, id_):
        return np.searchsorted(self.ids, id_)
    
    @cached_property
    def ids(self):
        return np.array(self._nodes.ids, dtype=np.int64)
    
    @cached_property
    def coord(self):
        return np.array(self._nodes.coord, dtype=np.float64)
    
    @cached_property
    def dof(self):
        return np.array(self._nodes.dof, dtype=np.bool_)

    @cached_property
    def axes_changed(self):
        return np.array(self._nodes.axes_changed, dtype=np.bool_)
    
    @cached_property
    def num_axes_changed(self):
        return np.count_nonzero(self.axes_changed)
    
    @cached_property
    def roll_angle_deg(self):
        return np.array(self._nodes.roll_angle_deg, dtype=np.float64)
    
    @cached_property
    def pitch_angle_deg(self):
        return np.array(self._nodes.pitch_angle_deg, dtype=np.float64)
    
    @cached_property
    def yaw_angle_deg(self):
        return np.array(self._nodes.yaw_angle_deg, dtype=np.float64)
    
    @cached_property
    def num_nodes(self):
        return self.ids.size

    @cached_property
    def total_num_dof(self):
        return self.dof.size
    
    @cached_property
    def num_free_dof(self):
        return np.count_nonzero(self.dof)
    
    @cached_property
    def rotation_order(self):
        return np.array(self._nodes.rotation_order, dtype=np.int64)
    
    @cached_property
    def _rotaion_about_x(self):
        if self.num_axes_changed != 0:
            roll_angle_deg = self.roll_angle_deg[self.axes_changed]
            cos = np.cos(np.deg2rad(roll_angle_deg))
            sin = np.sin(np.deg2rad(roll_angle_deg))
            
            rx = np.zeros((self.num_axes_changed, 3, 3), dtype=np.float64)
            rx[:, 0, 0] = 1.0
            rx[:, 1, 1] = cos
            rx[:, 1, 2] = sin
            rx[:, 2, 1] = -sin
            rx[:, 2, 2] = cos
            
            return rx
        
    @cached_property
    def _rotaion_about_y(self):
        if self.num_axes_changed != 0:
            pitch_angle_deg = self.pitch_angle_deg[self.axes_changed]
            cos = np.cos(np.deg2rad(pitch_angle_deg))
            sin = np.sin(np.deg2rad(pitch_angle_deg))
            
            ry = np.zeros((self.num_axes_changed, 3, 3), dtype=np.float64)
            ry[:, 0, 0] = cos
            ry[:, 0, 2] = -sin
            ry[:, 1, 1] = 1.0
            ry[:, 2, 0] = sin
            ry[:, 2, 2] = cos
            
            return ry
    
    @cached_property
    def _rotaion_about_z(self):
        if self.num_axes_changed != 0:
            yaw_angle_rad = np.deg2rad(self.yaw_angle_deg[self.axes_changed])
            cos = np.cos(yaw_angle_rad)
            sin = np.sin(yaw_angle_rad)
            
            rz = np.zeros((self.num_axes_changed, 3, 3), dtype=np.float64)
            rz[:, 0, 0] = cos
            rz[:, 0, 1] = sin
            rz[:, 1, 0] = -sin
            rz[:, 1, 1] = cos
            rz[:, 2, 2] = 1.0
            
            return rz
    
    @cached_property
    def direction_cosines(self):
        dc = np.zeros((self.num_nodes, 3, 3), dtype=np.float64)
        dc[:, 0, 0] = 1.0
        dc[:, 1, 1] = 1.0
        dc[:, 2, 2] = 1.0
        if self.num_axes_changed != 0:
            dc_with_axes_changed = dc[self.axes_changed]
            rx = self._rotaion_about_x
            ry = self._rotaion_about_y
            rz = self._rotaion_about_z
            
            rotation_order = self.rotation_order[self.axes_changed]
            
            is_xyz = rotation_order == self._nodes.R_XYZ
            is_xzy = rotation_order == self._nodes.R_XZY
            is_yxz = rotation_order == self._nodes.R_YXZ
            is_yzx = rotation_order == self._nodes.R_YZX
            is_zxy = rotation_order == self._nodes.R_ZXY
            is_zyx = rotation_order == self._nodes.R_ZYX
            
            dc_with_axes_changed[is_xyz] = rz@ry@rx
            dc_with_axes_changed[is_xzy] = ry@rz@rx
            dc_with_axes_changed[is_yxz] = rz@rx@ry
            dc_with_axes_changed[is_yzx] = rx@rz@ry
            dc_with_axes_changed[is_zxy] = ry@rx@rz
            dc_with_axes_changed[is_zyx] = rx@ry@rz
            
            dc[self.axes_changed] = dc_with_axes_changed
            
        return dc

    @cached_property
    def unit_vector_x(self):
        return self.direction_cosines[:, 0, :] 
    
    @cached_property
    def unit_vector_y(self):
        return self.direction_cosines[:, 1, :]
    
    @cached_property
    def unit_vector_z(self):
        return self.direction_cosines[:, 2, :]
    
    
    @cached_property
    def code_number(self):
        code_number = np.empty((self.num_nodes, 6), dtype=np.int64)
        code_number[self.dof] = np.arange(0, self.num_free_dof, 1, dtype=np.int64)
        code_number[~self.dof] = np.arange(self.num_free_dof, self.total_num_dof, 1, dtype=np.int64)
        return code_number
    
if __name__ == '__main__':
    nodes = Nodes()
    node1 = nodes.generate_node(0.0, 0.0, 0.0)
    node2 = nodes.generate_node(6.0, 0.0, 0.0)
    node3 = nodes.generate_node(6.0, 0.0, 0.0, False, False)
    node4 = nodes.generate_node(6.0, 0.0, 0.0)
    node5 = nodes.generate_node(12.0, 0.0, 0.0)
    
    nodes.change_axes(node1, rotation_order=nodes.R_YZX, rx=0, ry=45, rz=45)
    # print(rx@rz@ry)
    nodes_manager = NodesManager(nodes)
    print(nodes_manager.dof)