
from functools import cached_property
import bisect
import math 

from GlobalCoord import GlobalCoord
from Nodes import Nodes
from Materials import Materials
from Sections import Sections

import numpy as np
from numpy.typing import NDArray

class LineElements:
    
    PENALTY_NUMBER = 1e6
    
    def __init__(self, nodes: Nodes, materials: Materials, sections: Sections):
        self._ids              : list[int]       = []
        
        self._nodes_indices: list[list[float]] = []
        self._length: list[float] = []
        self._materials_indices: list[int] = []
        self._sections_indices: list[int] = []

        self._partial_fixity_vector = []
        self._partial_fixity_indices = []
        
        self._roll_angle       : list[float]     = []
       
        self._direction_cosines = []
        
        self._id: int             = 0
        
        self._nodes     = nodes
        self._materials = materials
        self._sections  = sections
        
    def _generate_id(self):
        ID = self._id
        self._ids.append(ID)
        self._id += 1
        return ID
    
    def _validate_nodei_id(self, id_: int):
        if not isinstance(id_, int):
            raise TypeError(f"Expected {int}. Got {type(id_)}")
    
    def _validate_nodej_id(self, id_: int):
        if not isinstance(id_, int):
            raise TypeError(f"Expected {int}. Got {type(id_)}")
    
    def _validate_material_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Expected {str}. Got {type(name)}")
    
    def _validate_section_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Expected {str}. Got {type(name)}")
    
    def _validate_roll_angle(self, roll_angle: float):
        if not isinstance(roll_angle, float):
            raise TypeError(f"Expected {float}. Got {type(roll_angle)}")
        
    def _compute_delta_coord(self, nodej_coord: NDArray[np.float64], nodei_coord: NDArray[np.float64]):
        return nodej_coord - nodei_coord
    
    def _compute_length(self, delta_coord: NDArray[np.float64]):
        return np.sqrt(np.sum(delta_coord**2))
    
    def _compute_unit_vector_x(self, length: np.float64, delta_coord: NDArray[np.float64]):
        return delta_coord/length
    
    def _compute_default_unit_vector_z(self, unit_vector_x: NDArray[np.float64]):
        if np.array_equal(np.abs(unit_vector_x), GlobalCoord.UNIT_VECTOR_Y):
            return GlobalCoord.UNIT_VECTOR_Z
        
        z = np.cross(unit_vector_x, GlobalCoord.UNIT_VECTOR_Y)
        z_norm = np.linalg.norm(z)
        return z/z_norm
    
    def _compute_default_unit_vector_y(self, unit_vector_x: NDArray[np.float64], default_unit_vector_z: NDArray[np.float64]):
        return np.cross(default_unit_vector_z, unit_vector_x)
    
    def _compute_roll_matrix(self, roll_angle: float):
        roll_angle_rad = math.radians(roll_angle)
        cos = math.cos(roll_angle_rad)
        sin = math.sin(roll_angle_rad)
        rx = [
            [1.0,  0.0, 0.0],
            [0.0,  cos, sin],
            [0.0, -sin, cos]
        ]
        
        return np.array(rx, dtype=np.float64)
    
    def _compute_default_direction_cosines(self, unit_vector_x: NDArray[np.float64], default_unit_vector_y: NDArray[np.float64], default_unit_vector_z: NDArray[np.float64]):
        return np.array([unit_vector_x, default_unit_vector_y, default_unit_vector_z])
    
    def _compute_direction_cosines(self, length: float, delta_coord: NDArray[np.float64], roll_angle: float):
        vec_x = self._compute_unit_vector_x(length, delta_coord)
        vec_z = self._compute_default_unit_vector_z(vec_x)
        vec_y = self._compute_default_unit_vector_y(vec_x, vec_z)
        dc    = self._compute_default_direction_cosines(vec_x, vec_y, vec_z)
        rx    = self._compute_roll_matrix(roll_angle)
        return rx@dc
    
    
    def find_index_by_id(self, id_: int):
        index = bisect.bisect_left(self._ids, id_)
        if index < len(self._ids) and self._ids[index] == id_:
            return index  
        else:
            raise ValueError(f"Node with id: {id_} doesn't exist.")
    
    def generate_element(self, nodei_id: int, nodej_id: int, material_name: str, section_name: str, roll_angle: float = 0.0):     
           
        nodei_index = self._nodes.find_index_by_id(nodei_id)
        nodej_index = self._nodes.find_index_by_id(nodej_id)
        material_index = self._materials.find_index_by_name(material_name)
        section_index = self._sections.find_index_by_name(section_name)
        
        id_ = self._generate_id()
        
        nodei_coord = self._nodes.coord[nodei_index]
        nodej_coord = self._nodes.coord[nodej_index]
        
        delta_coord = self._compute_delta_coord(nodej_coord, nodei_coord)
        L = self._compute_length(delta_coord)
        
        dc = self._compute_direction_cosines(L, delta_coord, roll_angle)

        self._materials_indices.append(material_index)
        self._sections_indices.append(section_index)
        
        self._length.append(L)
        self._nodes_indices.append(np.array([nodei_index, nodej_index], dtype=np.int64))
        self._direction_cosines.append(dc)
        return id_
    
    def add_partial_fixity(
        self,
        id_: int,
        u1 : float | None = None,
        u2 : float | None = None,
        u3 : float | None = None,
        u4 : float | None = None,
        u5 : float | None = None,
        u6 : float | None = None,
        u7 : float | None = None,
        u8 : float | None = None,
        u9 : float | None = None,
        u10: float | None = None,
        u11: float | None = None,
        u12: float | None = None
    ):
        index = self.find_index_by_id(id_)
        partial_fixity = np.array([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12], dtype=np.float64)
        # partial_fixity[np.isnan(partial_fixity)] = self.PENALTY_NUMBER
        self._partial_fixity_indices.append(index)
        self._partial_fixity_vector.append(partial_fixity)
    
    @cached_property
    def partial_fixity_vector(self):
        return np.array(self._partial_fixity_vector, dtype=np.float64)
    
    @cached_property
    def partial_fixity_indices(self):
        return np.array(self._partial_fixity_indices, dtype=np.int64)
    
    @cached_property
    def num_elements_with_partial_fixity(self):
        return self.partial_fixity_indices.shape[0]
    
    @cached_property 
    def direction_cosines(self):
        return np.array(self._direction_cosines, dtype=np.float64)
    
    @cached_property
    def num_elements(self):
        return len(self._ids)
    
    @cached_property
    def nodes_indices(self):
        return np.array(self._nodes_indices, dtype=np.int64)
    
    @cached_property
    def length(self):
        return np.array(self._length, dtype=np.float64)
   
    @cached_property
    def materials_indices(self):
        return np.array(self._materials_indices, dtype=np.int64)
    
    @cached_property
    def sections_indices(self):
        return np.array(self._sections_indices, dtype=np.int64)
    
    @cached_property    
    def code_number(self):
        return np.reshape(self._nodes.code_number[self.nodes_indices], (self.num_elements, 12))
    
    def ids(self):
        return self._ids
    
    