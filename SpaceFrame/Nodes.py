from functools import cached_property
import bisect

import numpy as np
from numpy.typing import NDArray 

class Nodes:
    
    def __init__(self) -> None:
        self._ids: list[int] = []
        self._id: int = 0

        self._coord: list[NDArray[np.float64]] = []
        self._dof: list[NDArray[np.bool_]] = []
        
        self._directions_cosines = []
        
    def _generate_id(self):
        _id = self._id
        self._ids.append(_id)
        self._id += 1
        return _id
    
    def _validate_coord(self, coord: list[float]):
        for i in coord:
            if not isinstance(i, float):
                raise TypeError(f"Expected {float}. Got {type(i)}")
        
        if coord in self._coord:
            raise ValueError(f"Attempted to assign two nodes with the same coordinates.")
    
    def _validate_dof(self, dof: list[bool]):
        for i in dof:
            if not isinstance(i, bool):
                raise TypeError(f"Expected {bool}. Got {type(i)}")
    
    def find_index_by_id(self, id_: int):
        index = bisect.bisect_left(self._ids, id_)
        if index < len(self._ids) and self._ids[index] == id_:
            return index  
        else:
            raise ValueError(f"Node with id: {id_} doesn't exist.")
    
    def generate_node(self, x: float, y: float, z: float, ux: bool=True, uy: bool=True, uz: bool=True, rx: bool=True, ry: bool=True, rz: bool=True): 
        coord = np.array([x, y, z], dtype=np.float64)
        dof = np.array([ux, uy, uz, rx, ry, rz], dtype=np.bool_)
    
        id_ = self._generate_id()
        self._coord.append(coord)
        self._dof.append(dof)
        dc = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self._directions_cosines.append(dc)
        
        return id_
    
    def change_axes(self, node_id: int, rx: float, ry: float, rz: float):
        index = self.find_index_by_id(node_id)
        dc = self._rotation_about_x(rx) @ self._rotation_about_y(ry) @ self._rotation_about_z(rz)
        self._directions_cosines[index] = dc        
    
    def _rotation_about_x(self, theta: float):
        cos = np.cos(np.deg2rad(theta))
        sin = np.sin(np.deg2rad(theta))
        rx = np.array([
            [1, 0,    0],
            [0, cos, -sin],
            [0, sin,  cos]
        ], dtype=np.float64)
        return rx
    
    def _rotation_about_y(self, theta: float):
        cos = np.cos(np.deg2rad(theta))
        sin = np.sin(np.deg2rad(theta))
        ry = np.array([
            [cos,  0, sin],
            [0,    1, 0],
            [-sin, 0, cos]
        ], dtype=np.float64)
        return ry
    
    def _rotation_about_z(self, theta: float):
        cos = np.cos(np.deg2rad(theta))
        sin = np.sin(np.deg2rad(theta))
        rz = np.array([
            [cos, -sin, 0],
            [sin,  cos, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        return rz
        
    @cached_property
    def ids(self):
        return self._ids

    @cached_property
    def coord(self):
        return np.array(self._coord, dtype=np.float64)
    
    @cached_property
    def dof(self):
        return np.array(self._dof, dtype=np.bool_)
    
    @cached_property
    def total_num_dof(self):
        return self.dof.size
    
    @cached_property
    def num_free_dof(self):
        return np.count_nonzero(self.dof)
    
    @cached_property
    def num_fixed_dof(self):
        return self.total_num_dof - self.num_free_dof
        
    @cached_property
    def num_nodes(self):
        return len(self._ids)
    
    @cached_property
    def code_number(self):
        code_number = np.empty((self.num_nodes, 6), dtype=np.int64)
        code_number[self.dof] = np.arange(0, self.num_free_dof, 1, dtype=np.int64)
        code_number[~self.dof] = np.arange(self.num_free_dof, self.total_num_dof, 1, dtype=np.int64)
        return code_number

    @cached_property
    def directions_cosines(self):
        return np.array(self._directions_cosines, dtype=np.float64)
    
if __name__ == '__main__':
    
    pass
