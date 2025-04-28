
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
    
    def __init__(self, nodes: Nodes, materials: Materials, sections: Sections):
        self._ids              : list[int]       = []
        
        self._nodes_indices: list[list[float]] = []
        self._length: list[float] = []
        self._materials_indices: list[int] = []
        self._sections_indices: list[int] = []
    
        self._roll_angle       : list[float]     = []
        self._unit_vector_x: list[NDArray[np.float64]] = []
        self._unit_vector_y: list[NDArray[np.float64]] = []
        self._unit_vector_z: list[NDArray[np.float64]] = []
        
        self._trans_matrix_3x3: list[list[list[float]]] = []
        self._trans_matrix_12x12: list[list[list[float]]] = []
        self._local_nodal_vector: list[list[float]] = []
        self._local_stiffness_matrix: list[list[list[float]]] = []
        
        self._youngs_modulus = []
        self._shear_modulus = []
        
        self._cross_section_area_yz = []
        self._moment_of_inertia_about_y = []
        self._moment_of_inertia_about_z = []
        self._torsional_constant = []
        self._shear_correction_factor = []
        self._direction_cosines = []
        
        self._id               : int             = 0
        
        self._nodes     = nodes
        self._materials = materials
        self._sections  = sections
        
        self._local_nodal_vector_cached = None
        
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
    
    def _compute_local_stiffness_matrix(
        self,
        length                   : float,
        material_index: int,
        section_index: int
    ):
        L  = length
        L2 = L**2
        L3 = L*L2
        
        E     = self._materials.youngs_modulus[material_index]
        G     = self._materials.shear_modulus[material_index]
        
        A     = self._sections.cross_section_area_yz[section_index]
        Iyy   = self._sections.moment_of_inertia_about_y[section_index]
        Izz   = self._sections.moment_of_inertia_about_z[section_index]
        J     = self._sections.torsional_constant[section_index]
        kappa = self._sections.shear_correction_factor[section_index]
        
        EA_L = E*A/L
        
        GJ_L = G*J/L
        
        beta_xz = 12*E*Iyy/(kappa*G*A*L2)
        EIyy_L3_12 = 12*E*Iyy/(L3*(1 + beta_xz))
        EIyy_L2_6 = 6*E*Iyy/(L2*(1 + beta_xz))
        EIyy_L_4 = (4 + beta_xz)*E*Iyy/(L*(1 + beta_xz))
        EIyy_L_2 = (2 - beta_xz)*E*Iyy/(L*(1 + beta_xz))
        
        beta_xy = 12*E*Izz/(kappa*G*A*L2)
        EIzz_L3_12 = 12*E*Izz/(L3*(1 + beta_xy))
        EIzz_L2_6 = 6*E*Izz/(L2*(1 + beta_xy))
        EIzz_L_4 = (4 + beta_xy)*E*Izz/(L*(1 + beta_xy))
        EIzz_L_2 = (2 - beta_xy)*E*Izz/(L*(1 + beta_xy))

        k = [
            [EA_L,  0.0,        0.0,        0.0,   0.0,       0.0,      -EA_L, 0.0,        0.0,        0.0,  0.0,       0.0],
            [0.0,   EIzz_L3_12, 0.0,        0.0,   0.0,       EIzz_L2_6, 0.0, -EIzz_L3_12, 0.0,        0.0,  0.0,       EIzz_L2_6],
            [0.0,   0.0,        EIyy_L3_12, 0.0,  -EIyy_L2_6, 0.0,       0.0,  0.0,       -EIyy_L3_12, 0.0,  EIyy_L2_6, 0.0],
            [0.0,   0.0,        0.0,        GJ_L,  0.0,       0.0,       0.0,  0.0,        0.0,       -GJ_L, 0.0,       0.0],
            [0.0,   0.0,       -EIyy_L2_6,  0.0,   EIyy_L_4,  0.0,       0.0,  0.0,        EIyy_L2_6,  0.0,  EIyy_L_2,  0.0],
            [0.0,   EIzz_L2_6,  0.0,        0.0,   0.0,       EIzz_L_4,  0.0, -EIzz_L2_6,  0.0,        0.0,  0.0,       EIzz_L_2],
            [-EA_L, 0.0,        0.0,        0.0,   0.0,       0.0,       EA_L, 0.0,        0.0,        0.0,  0.0,       0.0],
            [0.0,  -EIzz_L3_12, 0.0,        0.0,   0.0,      -EIzz_L2_6, 0.0,  EIzz_L3_12, 0.0,        0.0,  0.0,      -EIzz_L2_6],
            [0.0,   0.0,       -EIyy_L3_12, 0.0,   EIyy_L2_6, 0.0,       0.0,  0.0,        EIyy_L3_12, 0.0,  EIyy_L2_6, 0.0],
            [0.0,   0.0,        0.0,       -GJ_L,  0.0,       0.0,       0.0,  0.0,        0.0,        GJ_L, 0.0,       0.0],
            [0.0,   0.0,       -EIyy_L2_6,  0.0,   EIyy_L_2,  0.0,       0.0,  0.0,        EIyy_L2_6,  0.0,  EIyy_L_4,  0.0],
            [0.0,   EIzz_L2_6,  0.0,        0.0,   0.0,       EIzz_L_2,  0.0, -EIzz_L2_6,  0.0,        0.0,  0.0,       EIzz_L_4]
        ]
        
        return np.array(k, dtype=np.float64)
    
    def _compute_trans_matrix_3x3(self, direction_cosines: NDArray[np.float64]):
        return direction_cosines
    
    def _compute_trans_matrix_12x12(self, trans_matrix_3x3: list[list[float]]):
        t = trans_matrix_3x3
        txx = t[0][0]
        txy = t[0][1]
        txz = t[0][2]
        
        tyx = t[1][0]
        tyy = t[1][1]
        tyz = t[1][2]
        
        tzx = t[2][0]
        tzy = t[2][1]
        tzz = t[2][2]
        
        T = [
            [txx, txy, txz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [tyx, tyy, tyz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [tzx, tzy, tzz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, txx, txy, txz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, tyx, tyy, tyz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, tzx, tzy, tzz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, txx, txy, txz, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tyx, tyy, tyz, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tzx, tzy, tzz, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, txx, txy, txz],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tyx, tyy, tyz],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tzx, tzy, tzz],
        ]
        
        return np.array(T, dtype=np.float64)
    
    def _zero_vector_12(self):
        return np.zeros(12, dtype=np.float64)
    
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
    
    @cached_property 
    def direction_cosines(self):
        return np.array(self._direction_cosines, dtype=np.float64)
    
    @cached_property
    def local_stiffness_matrix(self):
        return np.array(self._local_stiffness_matrix, dtype=np.float64)
    
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
    def local_nodal_vector(self):
        if self._local_nodal_vector_cached is None:
            self._local_nodal_vector_cached = np.array(self._local_nodal_vector, dtype=np.float64)
        return self._local_nodal_vector_cached
    
    @cached_property
    def trans_matrix_3x3(self):
        return np.array(self._trans_matrix_3x3, dtype=np.float64)
    
    @cached_property
    def trans_matrix_12x12(self):
        return np.array(self._trans_matrix_12x12, dtype=np.float64)
    
    @cached_property
    def materials_indices(self):
        return np.array(self._materials_indices, dtype=np.int64)
    
    @cached_property
    def sections_indices(self):
        return np.array(self._sections_indices, dtype=np.int64)
    
    @cached_property 
    def code_number(self):
        return np.reshape(self._nodes.code_number[self.nodes_indices], (self.num_elements, 12))
    
    @cached_property
    def global_stiffness_matrix(self):
        return np.einsum('nji,njk,nkl->nil', self.trans_matrix_12x12, self.local_stiffness_matrix, self.trans_matrix_12x12)
    
    @cached_property    
    def global_nodal_vector(self):
        return np.einsum('nji,nj->ni', self.trans_matrix_12x12, self.local_nodal_vector)

    @cached_property
    def structure_stiffness_matrix(self):
        s = np.zeros((self._nodes.num_free_dof, self._nodes.num_free_dof), dtype=np.float64)
        k = self.global_stiffness_matrix
        valid_indices = self.code_number < self._nodes.num_free_dof
        
        for i in range(self.num_elements):
            s_cols = self.code_number[i][valid_indices[i]]
            s_rows = s_cols[:, None]
            s[s_rows, s_cols] += k[i][valid_indices[i]][:, valid_indices[i]]
        return s
    
    @cached_property
    def global_nodal_forces(self):
        f = np.zeros(self._nodes.num_free_dof, dtype=np.float64)
        elements_code_number_flat = self.code_number.ravel()
        elements_global_nodal_vector_flat = self.global_nodal_vector.ravel()
        valid_indices = elements_code_number_flat < self._nodes.num_free_dof
        np.add.at(f, elements_code_number_flat[valid_indices], elements_global_nodal_vector_flat[valid_indices])
        return f
    
    @cached_property
    def nodal_displacement(self):
        return np.linalg.solve(self.structure_stiffness_matrix, self.global_nodal_forces)
    
    
    def ids(self):
        return self._ids
    
    