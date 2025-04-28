from functools import cached_property

from Nodes import Nodes
from Materials import Materials
from Sections import Sections
from LineElements import LineElements
from StiffnessMatrix import StiffnessMatrix
from TransMatrix import TransMatrix

import numpy as np
from numpy.typing import NDArray

class Loads:
   
    FORCE   : int = 1
    MOMENT  : int = 2 
    
    LOCAL_X: int = 1
    LOCAL_Y: int = 2
    LOCAL_Z: int = 3
    
    GLOBAL_X: int = 4
    GLOBAL_Y: int = 5
    GLOBAL_Z: int = 6
    
    def __init__(self, elements: LineElements, materials: Materials, sections: Sections):
        self._elements = elements
        self._materials = materials
        self._sections = sections

        self._dist_force_elements_indices = []
        self._dist_force_mag_1_vec = []
        self._dist_force_mag_2_vec = []
        self._dist_force_pos_1 = []
        self._dist_force_pos_2 = []
        self._dist_force_axis = []
        
        self._dist_moment_elements_indices = []
        self._dist_moment_mag_1_vec = []
        self._dist_moment_mag_2_vec = []
        self._dist_moment_pos_1 = []
        self._dist_moment_pos_2 = []
        self._dist_moment_axis = []
        
        self._concen_force_elements_indices = []
        self._concen_force_mag_vec = []
        self._concen_force_pos = []
        self._concen_force_axis = []
        
        self._concen_moment_elements_indices = []
        self._concen_moment_mag_vec = []
        self._concen_moment_pos = []
        self._concen_moment_axis = []
        
    def _compute_mag_vector(self, mag: float, axis: int):
        pos = (axis - 1)%3
        vec = np.array([0.0, 0.0, 0.0], np.float64)
        vec[pos] = mag
        
        return vec 
    
    def _compute_pos(self, rel_pos: float, element_length: float):
        return rel_pos*element_length
   
    def add_distributed_load(self, element_id: int, type_: int, mag_1: float, mag_2: float, rel_pos_1: float, rel_pos_2: float, axis: int):
        elem_index = self._elements.find_index_by_id(element_id)
        L = self._elements.length[elem_index]
        
        mag_1_vec = self._compute_mag_vector(mag_1, axis)
        mag_2_vec = self._compute_mag_vector(mag_2, axis)

        pos_1 = self._compute_pos(rel_pos_1, L)
        pos_2 = self._compute_pos(rel_pos_2, L)
        
        if type_ == self.FORCE:
            self._dist_force_elements_indices.append(elem_index)
            self._dist_force_mag_1_vec.append(mag_1_vec)
            self._dist_force_mag_2_vec.append(mag_2_vec)
            self._dist_force_pos_1.append(pos_1)
            self._dist_force_pos_2.append(pos_2)
            self._dist_force_axis.append(axis)
        
        if type_ == self.MOMENT:
            self._dist_moment_elements_indices.append(elem_index)
            self._dist_moment_mag_1_vec.append(mag_1_vec)
            self._dist_moment_mag_2_vec.append(mag_2_vec)
            self._dist_moment_pos_1.append(pos_1)
            self._dist_moment_pos_2.append(pos_2)
            self._dist_moment_axis.append(axis)
            
    def add_concentrated_load(self, element_id: int, type_: int, mag: float, rel_pos: float, axis: int):
        elem_index = self._elements.find_index_by_id(element_id)
        L = self._elements.length[elem_index]
         
        mag_vec = self._compute_mag_vector(mag, axis)
        pos = self._compute_pos(rel_pos, L)
        
        if type_ == self.FORCE:
            self._concen_force_elements_indices.append(elem_index)
            self._concen_force_mag_vec.append(mag_vec)
            self._concen_force_pos.append(pos)
            self._concen_force_axis.append(axis)
            
        if type_ == self.MOMENT:
            self._concen_moment_elements_indices.append(elem_index)
            self._concen_moment_mag_vec.append(mag_vec)
            self._concen_moment_pos.append(pos)
            self._concen_moment_axis.append(axis)
            
    @cached_property
    def dist_force_elements_indices(self):
        return np.array(self._dist_force_elements_indices, dtype=np.int64)
    
    @cached_property
    def dist_force_mag_1_vec(self):
        return np.array(self._dist_force_mag_1_vec, dtype=np.float64)
    
    @cached_property
    def dist_force_mag_2_vec(self):
        return np.array(self._dist_force_mag_2_vec, dtype=np.float64)
    
    @cached_property
    def dist_force_pos_1(self):
        return np.array(self._dist_force_pos_1, dtype=np.float64)
    
    @cached_property
    def dist_force_pos_2(self):
        return np.array(self._dist_force_pos_2, dtype=np.float64)
    
    @cached_property
    def dist_force_axis(self):
        return np.array(self._dist_force_axis, dtype=np.int64)
    
    @cached_property
    def dist_moment_elements_indices(self):
        return np.array(self._dist_moment_elements_indices, dtype=np.int64)
    
    @cached_property
    def dist_moment_mag_1_vec(self):
        return np.array(self._dist_moment_mag_1_vec, dtype=np.float64)
    
    @cached_property
    def dist_moment_mag_2_vec(self):
        return np.array(self._dist_moment_mag_2_vec, dtype=np.float64)
    
    @cached_property
    def dist_moment_pos_1(self):
        return np.array(self._dist_moment_pos_1, dtype=np.float64)
    
    @cached_property
    def dist_moment_pos_2(self):
        return np.array(self._dist_moment_pos_2, dtype=np.float64)
    
    @cached_property
    def dist_moment_axis(self):
        return np.array(self._dist_moment_axis, dtype=np.int64)
    
    @cached_property
    def concen_force_elements_indices(self):
        return np.array(self._concen_force_elements_indices, dtype=np.int64)
     
    @cached_property
    def concen_force_mag_vec(self):
        return np.array(self._concen_force_mag_vec, dtype=np.float64)
    
    @cached_property
    def concen_force_pos(self):
        return np.array(self._concen_force_pos, dtype=np.float64)
    
    @cached_property
    def concen_force_axis(self):
        return np.array(self._concen_force_axis, dtype=np.int64)
    
    @cached_property
    def concen_moment_elements_indices(self):
        return np.array(self._concen_moment_elements_indices, dtype=np.int64)
    
    @cached_property
    def concen_moment_mag_vec(self):
        return np.array(self._concen_moment_mag_vec, dtype=np.float64)
    
    @cached_property
    def concen_moment_pos(self):
        return np.array(self._concen_moment_pos, dtype=np.float64)
    
    @cached_property
    def concen_moment_axis(self):
        return np.array(self._concen_moment_axis, dtype=np.int64)
    
    

class LoadSolver:
    
    def __init__(self, nodes: Nodes, elements: LineElements, loads: Loads, materials: Materials, sections: Sections, stiffness_matrix: StiffnessMatrix, trans_matrix: TransMatrix):
        self._nodes            = nodes
        self._elements         = elements
        self._loads            = loads
        self._materials        = materials
        self._sections         = sections
        self._stiffness_matrix = stiffness_matrix
        self._trans_matrix = trans_matrix
    
    def _global_to_local(self, elements_indices: NDArray[np.int64], mag_vec: NDArray[np.float64], axis: NDArray[np.int64]):
        is_global = axis > 3
        mag_vec[is_global] = np.einsum('nij,nj->ni', self._trans_matrix.trans_matrix_3x3[elements_indices[is_global]], mag_vec[is_global])
        
    def _solve_dist_force_local_nodal_vector(self):
        num_loads = len(self._loads.dist_force_elements_indices)
        if num_loads == 0:
            return
        
        Q = np.zeros((num_loads, 12), dtype=np.float64)
        
        L  = self._elements.length[self._loads.dist_force_elements_indices]
        L2 = L*L
        L3 = L2*L
        
        l1  = self._loads.dist_force_pos_1
        l12 = l1*l1
        l13 = l12*l1
        l14 = l12*l12
        l15 = l14*l1
        
        l2  = self._loads.dist_force_pos_2
        l22 = l2*l2
        l23 = l22*l2
        l24 = l22*l22
        l25 = l24*l2
           
        E = self._materials.youngs_modulus[self._elements.materials_indices[self._loads.dist_force_elements_indices]]
        G = self._materials.shear_modulus[self._elements.materials_indices[self._loads.dist_force_elements_indices]]
        
        A     = self._sections.cross_section_area_yz[self._elements.sections_indices[self._loads.dist_force_elements_indices]]
        Iyy   = self._sections.moment_of_inertia_about_y[self._elements.sections_indices[self._loads.dist_force_elements_indices]]
        Izz   = self._sections.moment_of_inertia_about_z[self._elements.sections_indices[self._loads.dist_force_elements_indices]]
        kappa = self._sections.shear_correction_factor[self._elements.sections_indices[self._loads.dist_force_elements_indices]]
        
        self._global_to_local(self._loads.dist_force_elements_indices, self._loads.dist_force_mag_1_vec, self._loads.dist_force_axis)
        wx1 = self._loads.dist_force_mag_1_vec[:, 0]
        wy1 = self._loads.dist_force_mag_1_vec[:, 1]
        wz1 = self._loads.dist_force_mag_1_vec[:, 2]
        
        self._global_to_local(self._loads.dist_force_elements_indices, self._loads.dist_force_mag_2_vec, self._loads.dist_force_axis)
        wx2 = self._loads.dist_force_mag_2_vec[:, 0]
        wy2 = self._loads.dist_force_mag_2_vec[:, 1]
        wz2 = self._loads.dist_force_mag_2_vec[:, 2]
        
        Q[:, 0] = -(l1 - l2)*(wx1*(3*L - 2*l1 - l2) + wx2*(3*L - l1 - 2*l2))/(6*L)
        Q[:, 6] = -Q[:, 0] + (wx1 + wx2)*(l2 - l1)/2
        
        kappaGA = kappa*G*A
        EIzz = E*Izz
        ay = (wy2 - wy1)/(l2 - l1)
        by = wy1 - ay*l1

        Q[:, 1] = -(
            (ay*(kappaGA*(10*L3*l12 - 10*L3*l22 - 15*L*l14 + 15*L*l24 + 8*l15 - 8*l25)
                + EIzz*(120*L*l12 - 120*L*l22 - 80*l13 + 80*l23)) 
            + by*(kappaGA*(20*L3*l1 - 20*L3*l2 - 20*L*l13 + 20*L*l23 + 10*l14 - 10*l24) 
                + EIzz*(240*L*l1 - 240*L*l2 - 120*l12 + 120*l22)))
            /(20*kappaGA*L3 + 240*EIzz*L)
        )
        Q[:, 7] = -Q[:, 1] + (wy1 + wy2)*(l2 - l1)/2

        Q[:, 5] = -(
            (ay*(kappaGA*(20*L3*l13 - 20*L3*l23 - 30*L2*l14 + 30*L2*l24 + 12*L*l15 - 12*L*l25)
                + EIzz*(120*L*l13 - 120*L*l23 - 90*l14 + 90*l24))
            + by*(kappaGA*(30*L3*l12 - 30*L3*l22 - 40*L2*l13 + 40*L2*l23 + 15*L*l14 - 15*L*l24)
                + EIzz*(180*L*l12 - 180*L*l22 - 120*l13 + 120*l23)))
            /(60*A*G*L3*kappa + 720*EIzz*L)
        )
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + (l2 - l1)*(wy1*(2*l1 + l2) + wy2*(l1 + 2*l2))/6
        
        
        EIyy = E*Iyy
        az = (wz2 - wz1)/(l2 - l1)
        bz = wz1 - az*l1
        
        Q[:, 2] = -(
            (az*(kappaGA*(10*L3*l12 - 10*L3*l22 - 15*L*l14 + 15*L*l24 + 8*l15 - 8*l25)
                + EIyy*(120*L*l12 - 120*L*l22 - 80*l13 + 80*l23)) 
            + bz*(kappaGA*(20*L3*l1 - 20*L3*l2 - 20*L*l13 + 20*L*l23 + 10*l14 - 10*l24) 
                + EIyy*(240*L*l1 - 240*L*l2 - 120*l12 + 120*l22)))
            /(20*kappaGA*L3 + 240*EIyy*L)
        )
        Q[:, 8] = -Q[:, 2] + (wz1 + wz2)*(l2 - l1)/2
        
        Q[:, 4] = (
            (az*(kappaGA*(20*L3*l13 - 20*L3*l23 - 30*L2*l14 + 30*L2*l24 + 12*L*l15 - 12*L*l25)
                + EIyy*(120*L*l13 - 120*L*l23 - 90*l14 + 90*l24))
            + bz*(kappaGA*(30*L3*l12 - 30*L3*l22 - 40*L2*l13 + 40*L2*l23 + 15*L*l14 - 15*L*l24)
                + EIyy*(180*L*l12 - 180*L*l22 - 120*l13 + 120*l23)))
            /(60*A*G*L3*kappa + 720*EIyy*L)
        )
        Q[:, 10] = -Q[:, 4] + Q[:, 8]*L - (l2 - l1)*(wz1*(2*l1 + l2) + wz2*(l1 + 2*l2))/6
        
        
        # Q[:, 5] *= -1
        # Q[:, 11] *= -1
        
        np.add.at(self._stiffness_matrix.elements_local_nodal_vector, self._loads.dist_force_elements_indices, Q)

    def _solve_dist_moment_local_nodal_vector(self):
        num_loads = len(self._loads.dist_moment_elements_indices)
        if num_loads == 0:
            return
        Q = np.zeros((num_loads, 12), dtype=np.float64)
        
        L  = self._elements.length[self._loads.dist_moment_elements_indices]
        L2 = L*L
        L3 = L2*L
        
        l1  = self._loads.dist_moment_pos_1
        l12 = l1*l1
        l13 = l12*l1
        l14 = l12*l12
        l15 = l14*l1
        
        l2  = self._loads.dist_moment_pos_2
        l22 = l2*l2
        l23 = l22*l2
        l24 = l22*l22
        l25 = l24*l2
        
        E = self._materials.youngs_modulus[self._elements.materials_indices[self._loads.dist_moment_elements_indices]]
        G = self._materials.shear_modulus[self._elements.materials_indices[self._loads.dist_moment_elements_indices]]
        
        A     = self._sections.cross_section_area_yz[self._elements.sections_indices[self._loads.dist_moment_elements_indices]]
        Iyy   = self._sections.moment_of_inertia_about_y[self._elements.sections_indices[self._loads.dist_moment_elements_indices]]
        Izz   = self._sections.moment_of_inertia_about_z[self._elements.sections_indices[self._loads.dist_moment_elements_indices]]
        kappa = self._sections.shear_correction_factor[self._elements.sections_indices[self._loads.dist_moment_elements_indices]]

        self._global_to_local(self._loads.dist_moment_elements_indices, self._loads.dist_moment_mag_1_vec, self._loads.dist_moment_axis)
        wx1 = self._loads.dist_moment_mag_1_vec[:, 0]
        wy1 = self._loads.dist_moment_mag_1_vec[:, 1]
        wz1 = self._loads.dist_moment_mag_1_vec[:, 2]
        
        self._global_to_local(self._loads.dist_moment_elements_indices, self._loads.dist_moment_mag_2_vec, self._loads.dist_moment_axis)
        wx2 = self._loads.dist_moment_mag_2_vec[:, 0]
        wy2 = self._loads.dist_moment_mag_2_vec[:, 1]
        wz2 = self._loads.dist_moment_mag_2_vec[:, 2]
        
        Q[:, 3] = -(l1 - l2)*(wx1*(3*L - 2*l1 - l2) + wx2*(3*L - l1 - 2*l2))/(6*L)
        Q[:, 9] = -Q[:, 3] + (wx1 + wx2)*(l2 - l1)/2
        
        kappaGA = kappa*G*A
        EIzz = E*Izz
        az = (wz2 - wz1)/(l2 - l1)
        bz = wz1 - az*l1
        
        Q[:, 1] = -(
            kappaGA*(-l1 + l2)*
            (az*(4*L*l12 + 4*L*l1*l2 + 4*L*l22 - 3*l13 - 3*l12*l2 - 3*l1*l22 - 3*l23)
            + bz*(6*L*l1 + 6*L*l2 - 4*l12 - 4*l1*l2 - 4*l22))
            /(2*L*(kappaGA*L2 + 12*EIzz))
        )
        Q[:, 7] = -Q[:, 1]
        
        Q[:, 5] = -(
            (l1 - l2)*
            (az*(kappaGA*(6*L3*l1 + 6*L3*l2 - 16*L2*l12 - 16*L2*l1*l2 - 16*L2*l22 + 9*L*l13 + 9*L*l12*l2 + 9*L*l1*l22 + 9*L*l23) 
                + EIzz*(72*L*l1 + 72*L*l2 - 48*l12 - 48*l1*l2 - 48*l22)) 
            + 12*bz*(kappaGA*(L3 - 2*L2*l1 - 2*L2*l2 + L*l12 + L*l1*l2 + L*l22) 
                    + EIzz*(12*L - 6*l1 - 6*l2)))
            /(12*L*(kappaGA*L2 + 12*EIzz))
        )
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + (wz1 + wz2)*(l2 - l1)/2
        
        EIyy = E*Iyy
        ay = (wy2 - wy1)/(l2 - l1)
        by = wy1 - ay*l1
        
        Q[:, 2] = (
            kappaGA*(-l1 + l2)*
            (ay*(4*L*l12 + 4*L*l1*l2 + 4*L*l22 - 3*l13 - 3*l12*l2 - 3*l1*l22 - 3*l23)
            + by*(6*L*l1 + 6*L*l2 - 4*l12 - 4*l1*l2 - 4*l22))
            /(2*L*(kappaGA*L2 + 12*EIyy))
        )
        Q[:, 8] = -Q[:, 2]
        
        Q[:, 4] = (
            (l1 - l2)*
            (ay*(kappaGA*(6*L3*l1 + 6*L3*l2 - 16*L2*l12 - 16*L2*l1*l2 - 16*L2*l22 + 9*L*l13 + 9*L*l12*l2 + 9*L*l1*l22 + 9*L*l23) 
                + EIyy*(72*L*l1 + 72*L*l2 - 48*l12 - 48*l1*l2 - 48*l22)) 
            + 12*by*(kappaGA*(L3 - 2*L2*l1 - 2*L2*l2 + L*l12 + L*l1*l2 + L*l22) 
                    + EIyy*(12*L - 6*l1 - 6*l2)))
            /(12*L*(kappaGA*L2 + 12*EIyy))
        )
        Q[:, 10] = -Q[:, 4] - Q[:, 8]*L - (wy1 + wy2)*(l2 - l1)/2
        
        np.add.at(self._stiffness_matrix.elements_local_nodal_vector, self._loads.dist_moment_elements_indices, Q)
        
    def _solve_concen_force_local_nodal_vector(self):
        num_loads = len(self._loads.concen_force_elements_indices)
        if num_loads == 0:
            return
        
        Q = np.zeros((num_loads, 12), dtype=np.float64)
        
        L = self._elements.length[self._loads.concen_force_elements_indices]
        l = self._loads.concen_force_pos
        
        E = self._materials.youngs_modulus[self._elements.materials_indices[self._loads.concen_force_elements_indices]]
        G = self._materials.shear_modulus[self._elements.materials_indices[self._loads.concen_force_elements_indices]]
        
        A     = self._sections.cross_section_area_yz[self._elements.sections_indices[self._loads.concen_force_elements_indices]]
        Iyy   = self._sections.moment_of_inertia_about_y[self._elements.sections_indices[self._loads.concen_force_elements_indices]]
        Izz   = self._sections.moment_of_inertia_about_z[self._elements.sections_indices[self._loads.concen_force_elements_indices]]
        kappa = self._sections.shear_correction_factor[self._elements.sections_indices[self._loads.concen_force_elements_indices]]
        
        self._global_to_local(self._loads.concen_force_elements_indices, self._loads.concen_force_mag_vec, self._loads.concen_force_axis)
        wx  = self._loads.concen_force_mag_vec[:, 0]
        wy  = self._loads.concen_force_mag_vec[:, 1]
        wz  = self._loads.concen_force_mag_vec[:, 2]
        
        Q[:, 0] = -wx*(l - L)/L
        Q[:, 6] = -Q[:, 0] + wx
        
        kappaGA = kappa*G*A
        EIzz = E*Izz
        
        Q[:, 1] = -wy*(-L + l)*(kappaGA*(L**2 + L*l - 2*l**2) + 12*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
        Q[:, 7] = -Q[:, 1] + wy
        
        Q[:, 5] = -l*wy*(-L + l)*(kappaGA*(L**2 - L*l) + 6*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + wy*l
        
        EIyy = E*Iyy

        Q[:, 2] = -wz*(-L + l)*(kappaGA*(L**2 + L*l - 2*l**2) + 12*EIyy)/(L*(kappaGA*L**2 + 12*EIyy))
        Q[:, 8] = -Q[:, 2] + wz
        
        Q[:, 4] = -l*wz*(-L + l)*(kappaGA*(L**2 - L*l) + 6*EIyy)/(L*(kappaGA*L**2 + 12*EIyy))
        Q[:, 10] = -Q[:, 4] - Q[:, 8]*L + wz*l
        
        np.add.at(self._stiffness_matrix.elements_local_nodal_vector, self._loads.concen_force_elements_indices, Q)
        
    def _solve_concen_moment_local_nodal_vector(self):
        num_loads = len(self._loads.concen_moment_elements_indices)
        if num_loads == 0:
            return
        
        Q = np.zeros((num_loads, 12), dtype=np.float64)
        
        L = self._elements.length[self._loads.concen_moment_elements_indices]
        l = self._loads.concen_moment_pos
        
        E = self._materials.youngs_modulus[self._elements.materials_indices[self._loads.concen_moment_elements_indices]]
        G = self._materials.shear_modulus[self._elements.materials_indices[self._loads.concen_moment_elements_indices]]
        
        A     = self._sections.cross_section_area_yz[self._elements.sections_indices[self._loads.concen_moment_elements_indices]]
        Iyy   = self._sections.moment_of_inertia_about_y[self._elements.sections_indices[self._loads.concen_moment_elements_indices]]
        Izz   = self._sections.moment_of_inertia_about_z[self._elements.sections_indices[self._loads.concen_moment_elements_indices]]
        kappa = self._sections.shear_correction_factor[self._elements.sections_indices[self._loads.concen_moment_elements_indices]]
        
        self._global_to_local(self._loads.concen_moment_elements_indices, self._loads.concen_moment_mag_vec, self._loads.concen_moment_axis)
        wx = self._loads.concen_moment_mag_vec[:, 0]
        wy = self._loads.concen_moment_mag_vec[:, 1]
        wz = self._loads.concen_moment_mag_vec[:, 2]
        
        Q[:, 3] = -wx*(l - L)/L
        Q[:, 9] = -Q[:, 3] + wx
        
        # calculate nodal forces in x-y plane
        kappaGA = kappa*G*A
        EIzz = E*Izz
        
        Q[:, 1] = -6*kappaGA*l*wz*(L - l)/(L*(kappaGA*L**2 + 12*EIzz))
        Q[:, 7] = -Q[:, 1]
        
        Q[:, 5] = -wz*(-L + l)*(kappaGA*(L**2 - 3*L*l) + 12*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + wz
        
        # calculate nodal forces in x-z plane
        EIyy = E*Iyy

        Q[:, 2] = 6*kappaGA*l*wy*(L - l)/(L*(kappaGA*L**2 + 12*EIyy))
        Q[:, 8] = -Q[:, 2]
        
        Q[:, 4] = wy*(-L + l)*(kappaGA*(L**2 - 3*L*l) + 12*EIyy)/(L*(kappaGA*L**2 + 12*EIyy)) 
        Q[:, 10] = -Q[:, 4] - Q[:, 8]*L - wy
        
        np.add.at(self._stiffness_matrix.elements_local_nodal_vector, self._loads.concen_moment_elements_indices, Q)

    def solve(self):
        self._solve_dist_force_local_nodal_vector()
        self._solve_dist_moment_local_nodal_vector()
        self._solve_concen_force_local_nodal_vector()
        self._solve_concen_moment_local_nodal_vector()