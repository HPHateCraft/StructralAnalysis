from functools import cached_property

from Nodes import Nodes
from LineElements import LineElements
from Materials import Materials
from Sections import Sections
from TransMatrix import TransMatrix

import numpy as np
from numpy.typing import NDArray


class StiffnessMatrix:
    
    def __init__(self, nodes: Nodes, elements: LineElements, materials: Materials, sections: Sections, trans_matrix: TransMatrix):
        self._nodes        = nodes
        self._elements     = elements
        self._materials    = materials
        self._sections     = sections
        self._trans_matrix = trans_matrix
        
    @cached_property
    def _original_stiffness_matrix_in_elements_coord(self):
        k = np.zeros((self._elements.num_elements, 12, 12), dtype=np.float64)
        
        L  = self._elements.length
        L2 = L**2
        L3 = L*L2
        
        E     = self._materials.youngs_modulus[self._elements.materials_indices]
        G     = self._materials.shear_modulus[self._elements.materials_indices]
        
        A     = self._sections.cross_section_area_yz[self._elements.sections_indices]
        Iyy   = self._sections.moment_of_inertia_about_y[self._elements.sections_indices]
        Izz   = self._sections.moment_of_inertia_about_z[self._elements.sections_indices]
        J     = self._sections.torsional_constant[self._elements.sections_indices]
        kappa = self._sections.shear_correction_factor[self._elements.sections_indices]
        
        
        k[:, 0, 0] = E*A/L
        k[:, 0, 6] = -k[:, 0, 0]
        k[:, 6, 0] = -k[:, 0, 0]
        k[:, 6, 6] = k[:, 0, 0]
        
        k[:, 3, 3] = G*J/L
        k[:, 3, 9] = -G*J/L
        k[:, 9, 3] = -G*J/L
        k[:, 9, 9] = G*J/L
        
        beta_xz = 12*E*Iyy/(kappa*G*A*L2)
        k[:, 2, 2] =  12*E*Iyy/(L3*(1 + beta_xz))
        k[:, 2, 8] = -k[:, 2, 2]
        k[:, 8, 2] = -k[:, 2, 2]
        k[:, 8, 8] =  k[:, 2, 2]
        
        k[:, 2, 4]  = -6*E*Iyy/(L2*(1 + beta_xz))
        k[:, 2, 10] =  k[:, 2, 4]
        k[:, 4, 2]  =  k[:, 2, 4]
        k[:, 4, 8]  = -k[:, 2, 4]
        k[:, 8, 4]  = -k[:, 2, 4]
        k[:, 8, 10] = -k[:, 2, 4]
        k[:, 10, 2] =  k[:, 2, 4]
        k[:, 10, 8] = -k[:, 2, 4]
        
        k[:, 4, 4] = (4 + beta_xz)*E*Iyy/(L*(1 + beta_xz))
        k[:, 10, 10] = k[:, 4, 4]
        
        k[:, 4, 10] = (2 - beta_xz)*E*Iyy/(L*(1 + beta_xz))
        k[:, 10, 4] = k[:, 4, 10]
        
        beta_xy = 12*E*Izz/(kappa*G*A*L2)
        k[:, 1, 1] =  12*E*Izz/(L3*(1 + beta_xy))
        k[:, 1, 7] = -k[:, 1, 1]
        k[:, 7, 1] = -k[:, 1, 1]
        k[:, 7, 7] =  k[:, 1, 1]
        
        k[:, 1, 5]  =  6*E*Izz/(L2*(1 + beta_xy))
        k[:, 1, 11] =  k[:, 1, 5]
        k[:, 5, 1]  =  k[:, 1, 5]
        k[:, 5, 7]  = -k[:, 1, 5]
        k[:, 7, 5]  = -k[:, 1, 5]
        k[:, 7, 11] = -k[:, 1, 5]
        k[:, 11, 1] =  k[:, 1, 5]
        k[:, 11, 7] = -k[:, 1, 5]
        
        k[:, 5, 5] = (4 + beta_xy)*E*Izz/(L*(1 + beta_xy))
        k[:, 11, 11] = k[:, 5, 5]
        
        k[:, 5, 11] = (2 - beta_xy)*E*Izz/(L*(1 + beta_xy))
        k[:, 11, 5] = k[:, 5, 11]
        
        return k
    
    @cached_property
    def partial_fixity_matrix(self):
        if self._elements.partial_fixity_indices.size == 0:
            return
        diag = np.arange(0, 12, 1, dtype=np.int64)
        k_diag = self._original_stiffness_matrix_in_elements_coord[self._elements.partial_fixity_indices][:, diag, diag]
        partial_fixity_matrix = np.zeros((self._elements.num_elements_with_partial_fixity, 12, 12), dtype=np.float64)
        is_fixed = np.isnan(self._elements.partial_fixity_vector)
        self._elements.partial_fixity_vector[is_fixed] = k_diag[is_fixed] * self._elements.PENALTY_NUMBER
        partial_fixity_matrix[:, diag, diag] = self._elements.partial_fixity_vector
        return partial_fixity_matrix
    
    @cached_property
    def partial_fixity_matrix_inv(self):
        if self._elements.partial_fixity_indices.size == 0:
            return
        return np.linalg.inv(self._original_stiffness_matrix_in_elements_coord[self._elements.partial_fixity_indices] + self.partial_fixity_matrix)
    
    @cached_property
    def stiffness_matrix_in_elements_coord(self):
        k = self._original_stiffness_matrix_in_elements_coord
        if self._elements.partial_fixity_indices.size != 0:
            k[self._elements.partial_fixity_indices] = k[self._elements.partial_fixity_indices]@self.partial_fixity_matrix_inv@self.partial_fixity_matrix
        return k
    
    @cached_property
    def stiffness_matrix_in_global_coord(self):
        return self._trans_matrix.elements_direction_cosines_12x12_T @ self.stiffness_matrix_in_elements_coord @ self._trans_matrix.elements_direction_cosines_12x12 

    @cached_property
    def stiffness_matrix_in_nodal_coord(self):
        return self._trans_matrix.elements_nodes_direction_cosines_12x12_T @ self.stiffness_matrix_in_global_coord @ self._trans_matrix.elements_nodes_direction_cosines_12x12
    
    @cached_property
    def structure_stiffness_matrix(self):
        s = np.zeros((self._nodes.num_free_dof, self._nodes.num_free_dof), dtype=np.float64)
        k = self.stiffness_matrix_in_nodal_coord
        valid_indices = self._elements.code_number < self._nodes.num_free_dof
            
        for i in range(self._elements.num_elements):
                s_cols = self._elements.code_number[i][valid_indices[i]]
                s_rows = s_cols[:, None]
                s[s_rows, s_cols] += k[i][valid_indices[i]][:, valid_indices[i]]
        
        return s
    
