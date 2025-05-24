from functools import cached_property

from Nodes import Nodes, NodesManager
from Materials import Materials
from Sections import Sections
from LineElements import ElementsManager
from TransMatrix import TransMatrix
from StiffnessMatrix import StiffnessMatrix

import numpy as np
from numpy.typing import NDArray

class NodalLoad:
    
    LOCAL_AXIS = 0
    GLOBAL_AXIS = 1
    
    def __init__(self, nodes_manager: NodesManager):
        self._nodes_manager = nodes_manager

    def add_force(self, node_id: int, axis: int = LOCAL_AXIS, fx: float = 0.0, fy: float = 0.0, fz: float = 0.0, mx: float = 0.0, my: float = 0.0, mz: float = 0.0):
        index = self._nodes_manager.find_index_by_id(node_id)
        mag_vec = np.array([fx, fy, fz, mx, my, mz], dtype=np.float64)
        self.force_axis[index] = axis
        self.force_vector[index] += mag_vec
        
    def add_displacement(self, node_id: int, axis: int = LOCAL_AXIS, ux: float = 0.0, uy: float = 0.0, uz: float = 0.0, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0):
        index = self._nodes_manager.find_index_by_id(node_id)
        displacement = np.array([ux, uy, uz, rx, ry, rz], dtype=np.float64)
        self.displacement_axis[index] = axis
        self.displacement_vector[index] += displacement 
    
    @cached_property
    def force_vector(self):
        return np.zeros((self._nodes_manager.num_nodes, 6), dtype=np.float64)
    
    @cached_property
    def force_axis(self):
        return np.full(self._nodes_manager.num_nodes, self.LOCAL_AXIS, dtype=np.int64)
    
    @cached_property
    def displacement_vector(self):
        return np.zeros((self._nodes_manager.num_nodes, 6), dtype=np.float64)
    
    @cached_property
    def displacement_axis(self):
        return np.full(self._nodes_manager.num_nodes, self.LOCAL_AXIS, dtype=np.int64)
    
class NodalLoadManager:
    
    def __init__(self, nodes_manager: NodesManager, nodal_load: NodalLoad, trans_matrix: TransMatrix):
        self._nodes_manager = nodes_manager
        self._nodal_load = nodal_load
        self._trans_matrix = trans_matrix
    
    @cached_property
    def displacement_vector_in_nodal_coord(self):
        is_global_axis = self._nodal_load.displacement_axis == self._nodal_load.GLOBAL_AXIS
        d = self._nodal_load.displacement_vector
        d[is_global_axis] = np.einsum("nij,nj->ni", self._trans_matrix.nodes_direction_cosines_6x6_T[is_global_axis], d[is_global_axis])
        
        return d
    
    @cached_property
    def force_vector_in_nodal_coord(self):
        is_global_axis = self._nodal_load.force_axis == NodalLoad.GLOBAL_AXIS
        f = self._nodal_load.force_vector
        f[is_global_axis] = np.einsum("nij,nj->ni", self._trans_matrix.nodes_direction_cosines_6x6_T[is_global_axis], f[is_global_axis])
        return f
    
    @cached_property
    def structure_force_vector_in_nodal_coord(self):
        return self.force_vector_in_nodal_coord[self._nodes_manager.dof]


class DistributedForce:
    LOCAL_X: int = 1
    LOCAL_Y: int = 2
    LOCAL_Z: int = 3
    
    GLOBAL_X: int = 4
    GLOBAL_Y: int = 5
    GLOBAL_Z: int = 6
    
    def __init__(self, elements_manager: ElementsManager):
        self._elements_manager = elements_manager
        
        self.elements_indices = []
        self.axis = []
        self.force_vector_1 = []
        self.force_vector_2 = []
        self.relative_position_1 = []
        self.relative_position_2 = []

    def _force_vector(self, axis: int, magnitude: float):
        pos = (axis - 1)%3
        vec = [0.0, 0.0, 0.0]
        vec[pos] = magnitude
        
        return vec 
        
    def add(self, element_id: int, axis: int = GLOBAL_Y, magnitude_1: float = 0.0, magnitude_2: float = 0.0, relative_position_1: float = 0.0, relative_position_2: float = 1.0):
        index = self._elements_manager.find_index_by_id(element_id)
        self.elements_indices.append(index)
        self.axis.append(axis) 
        
        force_vector_1 = self._force_vector(axis, magnitude_1)
        force_vector_2 = self._force_vector(axis, magnitude_2)
        self.force_vector_1.append(force_vector_1)
        self.force_vector_2.append(force_vector_2)
        
        self.relative_position_1.append(relative_position_1)
        self.relative_position_2.append(relative_position_2)


class DistributedMoment:
    LOCAL_X: int = 1
    LOCAL_Y: int = 2
    LOCAL_Z: int = 3
    
    GLOBAL_X: int = 4
    GLOBAL_Y: int = 5
    GLOBAL_Z: int = 6
    
    def __init__(self, elements_manager: ElementsManager):
        self._elements_manager = elements_manager
        self.elements_indices = []
        self.axis = []
        self.force_vector_1 = []
        self.force_vector_2 = []
        self.relative_position_1 = []
        self.relative_position_2 = []
   
    def _force_vector(self, axis: int, magnitude: float):
        pos = (axis - 1)%3
        vec = [0.0, 0.0, 0.0]
        vec[pos] = magnitude
        
        return vec 
        
    def add(self, element_id: int, axis: int = GLOBAL_Y, magnitude_1: float = 0.0, magnitude_2: float = 0.0, relative_position_1: float = 0.0, relative_position_2: float = 1.0):
        index = self._elements_manager.find_index_by_id(element_id)
        self.elements_indices.append(index)
        self.axis.append(axis) 
        
        force_vector_1 = self._force_vector(axis, magnitude_1)
        force_vector_2 = self._force_vector(axis, magnitude_2)
        self.force_vector_1.append(force_vector_1)
        self.force_vector_2.append(force_vector_2)
        
        self.relative_position_1.append(relative_position_1)
        self.relative_position_2.append(relative_position_2)


class ConcentratedForce:
    LOCAL_X: int = 1
    LOCAL_Y: int = 2
    LOCAL_Z: int = 3
    
    GLOBAL_X: int = 4
    GLOBAL_Y: int = 5
    GLOBAL_Z: int = 6
    
    def __init__(self, elements_manager: ElementsManager):
        self._elements_manager = elements_manager
        self.elements_indices = []
        self.axis = []
        self.force_vector = []
        self.relative_position = []
        
   
    def _force_vector(self, axis: int, magnitude: float):
        pos = (axis - 1)%3
        vec = [0.0, 0.0, 0.0]
        vec[pos] = magnitude
        
        return vec 
        
    def add(self, element_id: int, axis: int = GLOBAL_Y, magnitude: float = 0.0, relative_position: float = 0.0):
        index = self._elements_manager.find_index_by_id(element_id)
        self.elements_indices.append(index)
        self.axis.append(axis) 
        
        force_vector = self._force_vector(axis, magnitude)
        self.force_vector.append(force_vector)
        
        self.relative_position.append(relative_position)


class ConcentratedMoment:
    LOCAL_X: int = 1
    LOCAL_Y: int = 2
    LOCAL_Z: int = 3
    
    GLOBAL_X: int = 4
    GLOBAL_Y: int = 5
    GLOBAL_Z: int = 6
    
    def __init__(self, elements_manager: ElementsManager):
        self._elements_manager = elements_manager
        self.elements_indices = []
        self.axis = []
        self.force_vector = []
        self.relative_position = []
        
   
    def _force_vector(self, axis: int, magnitude: float):
        pos = (axis - 1)%3
        vec = [0.0, 0.0, 0.0]
        vec[pos] = magnitude
        
        return vec 
        
    def add(self, element_id: int, axis: int = GLOBAL_Y, magnitude: float = 0.0, relative_position: float = 0.0):
        index = self._elements_manager.find_index_by_id(element_id)
        self.elements_indices.append(index)
        self.axis.append(axis) 
        
        force_vector = self._force_vector(axis, magnitude)
        self.force_vector.append(force_vector)
        
        self.relative_position.append(relative_position)


class Temperature:
    
    def __init__(self, elements_manager: ElementsManager):
        self._elements_manager = elements_manager
       
    def add(self, element_id: int, constant: float = 0.0, linear_y: float = 0.0, linear_z: float = 0.0):
        index = self._elements_manager.find_index_by_id(element_id)
        self.temperature_applied[index] = True
        self.constant[index] = constant
        self.linear_y[index] = linear_y
        self.linear_z[index] = linear_z

    @cached_property
    def temperature_applied(self):
        return np.zeros(self._elements_manager.num_elements, dtype=np.bool_)
    
    @cached_property
    def constant(self):
        return np.zeros(self._elements_manager.num_elements, dtype=np.float64)
    
    @cached_property
    def linear_y(self):
        return np.zeros(self._elements_manager.num_elements, dtype=np.float64)
    
    @cached_property
    def linear_z(self):
        return np.zeros(self._elements_manager.num_elements, dtype=np.float64)
    
class DistributedForceManager:
    
    def __init__(
        self,
        elements_manager: ElementsManager,
        distributed_force: DistributedForce,
        trans_matrix: TransMatrix
    ):
        self._elements_manager = elements_manager
        self._distributed_force         = distributed_force
        self._trans_matrix = trans_matrix
        
    @cached_property
    def elements_indices(self):
        return np.array(self._distributed_force.elements_indices, dtype=np.int64)

    @cached_property
    def axis(self):
        return np.array(self._distributed_force.axis, dtype=np.int64)
    
    @cached_property
    def num_distributed_force(self):
        return self.elements_indices.size
    
    @cached_property
    def force_vector_1(self):
        return np.array(self._distributed_force.force_vector_1, dtype=np.float64)
    
    @cached_property
    def force_vector_2(self):
        return np.array(self._distributed_force.force_vector_2, dtype=np.float64)
    
    @cached_property
    def relative_position_1(self):
        return np.array(self._distributed_force.relative_position_1, dtype=np.float64)
    
    @cached_property
    def relative_position_2(self):
        return np.array(self._distributed_force.relative_position_2, dtype=np.float64)
    
    
    @cached_property
    def nodal_force_vector_in_elements_coord(self):
        F = np.zeros((self._elements_manager.num_elements, 12), dtype=np.float64)
        if self.num_distributed_force > 0:
            Q = np.zeros((self.num_distributed_force, 12), dtype=np.float64)

            L  = self._elements_manager.length[self.elements_indices]
            L2 = L*L
            L3 = L2*L
           
            l1  = self.relative_position_1*L
            l12 = l1*l1
            l13 = l12*l1
            l14 = l12*l12
            l15 = l14*l1

            l2  = self.relative_position_2*L
            l22 = l2*l2
            l23 = l22*l2
            l24 = l22*l22
            l25 = l24*l2

            E = self._elements_manager.youngs_modulus[self.elements_indices]
            G = self._elements_manager.shear_modulus[self.elements_indices]

            A     = self._elements_manager.cross_section_area[self.elements_indices]
            Iyy   = self._elements_manager.moment_of_inertia_about_y[self.elements_indices]
            Izz   = self._elements_manager.moment_of_inertia_about_z[self.elements_indices]
            kappa = self._elements_manager.shear_correction_factor[self.elements_indices]

            f1 = self.force_vector_1
            f2 = self.force_vector_2

            force_in_global_axis = self.axis >= DistributedForce.GLOBAL_X
            t = self._trans_matrix.elements_direction_cosines_3x3[self.elements_indices][force_in_global_axis]
            f1[force_in_global_axis] = np.einsum("nij,nj->ni", t, f1[force_in_global_axis])
            f2[force_in_global_axis] = np.einsum("nij,nj->ni", t, f2[force_in_global_axis])

            fx1 = f1[:, 0]
            fy1 = f1[:, 1]
            fz1 = f1[:, 2]

            fx2 = f2[:, 0]
            fy2 = f2[:, 1]
            fz2 = f2[:, 2]

            Q[:, 0] = -(l1 - l2)*(fx1*(3*L - 2*l1 - l2) + fx2*(3*L - l1 - 2*l2))/(6*L)
            Q[:, 6] = -Q[:, 0] + (fx1 + fx2)*(l2 - l1)/2

            kappaGA = kappa*G*A
            EIzz = E*Izz
            ay = (fy2 - fy1)/(l2 - l1)
            by = fy1 - ay*l1

            Q[:, 1] = -(
                (ay*(kappaGA*(10*L3*l12 - 10*L3*l22 - 15*L*l14 + 15*L*l24 + 8*l15 - 8*l25)
                    + EIzz*(120*L*l12 - 120*L*l22 - 80*l13 + 80*l23)) 
                + by*(kappaGA*(20*L3*l1 - 20*L3*l2 - 20*L*l13 + 20*L*l23 + 10*l14 - 10*l24) 
                    + EIzz*(240*L*l1 - 240*L*l2 - 120*l12 + 120*l22)))
                /(20*kappaGA*L3 + 240*EIzz*L)
            )
            Q[:, 7] = -Q[:, 1] + (fy1 + fy2)*(l2 - l1)/2

            Q[:, 5] = -(
                (ay*(kappaGA*(20*L3*l13 - 20*L3*l23 - 30*L2*l14 + 30*L2*l24 + 12*L*l15 - 12*L*l25)
                    + EIzz*(120*L*l13 - 120*L*l23 - 90*l14 + 90*l24))
                + by*(kappaGA*(30*L3*l12 - 30*L3*l22 - 40*L2*l13 + 40*L2*l23 + 15*L*l14 - 15*L*l24)
                    + EIzz*(180*L*l12 - 180*L*l22 - 120*l13 + 120*l23)))
                /(60*A*G*L3*kappa + 720*EIzz*L)
            )
            Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + (l2 - l1)*(fy1*(2*l1 + l2) + fy2*(l1 + 2*l2))/6


            EIyy = E*Iyy
            az = (fz2 - fz1)/(l2 - l1)
            bz = fz1 - az*l1

            Q[:, 2] = -(
                (az*(kappaGA*(10*L3*l12 - 10*L3*l22 - 15*L*l14 + 15*L*l24 + 8*l15 - 8*l25)
                    + EIyy*(120*L*l12 - 120*L*l22 - 80*l13 + 80*l23)) 
                + bz*(kappaGA*(20*L3*l1 - 20*L3*l2 - 20*L*l13 + 20*L*l23 + 10*l14 - 10*l24) 
                    + EIyy*(240*L*l1 - 240*L*l2 - 120*l12 + 120*l22)))
                /(20*kappaGA*L3 + 240*EIyy*L)
            )
            Q[:, 8] = -Q[:, 2] + (fz1 + fz2)*(l2 - l1)/2

            Q[:, 4] = (
                (az*(kappaGA*(20*L3*l13 - 20*L3*l23 - 30*L2*l14 + 30*L2*l24 + 12*L*l15 - 12*L*l25)
                    + EIyy*(120*L*l13 - 120*L*l23 - 90*l14 + 90*l24))
                + bz*(kappaGA*(30*L3*l12 - 30*L3*l22 - 40*L2*l13 + 40*L2*l23 + 15*L*l14 - 15*L*l24)
                    + EIyy*(180*L*l12 - 180*L*l22 - 120*l13 + 120*l23)))
                /(60*A*G*L3*kappa + 720*EIyy*L)
            )
            Q[:, 10] = -Q[:, 4] + Q[:, 8]*L - (l2 - l1)*(fz1*(2*l1 + l2) + fz2*(l1 + 2*l2))/6

            np.add.at(F, self.elements_indices, Q)
           
        return F


class DistributedMomentManager:
    
    def __init__(
        self,
        elements_manager: ElementsManager,
        distributed_moment: DistributedMoment,
        trans_matrix: TransMatrix
    ):
        self._elements_manager = elements_manager
        self._distributed_moment         = distributed_moment
        self._trans_matrix = trans_matrix
        
    @cached_property
    def elements_indices(self):
        return np.array(self._distributed_moment.elements_indices, dtype=np.int64)

    @cached_property
    def axis(self):
        return np.array(self._distributed_moment.axis, dtype=np.int64)
    
    @cached_property
    def num_distributed_moment(self):
        return self.elements_indices.size
    
    @cached_property
    def force_vector_1(self):
        return np.array(self._distributed_moment.force_vector_1, dtype=np.float64)
    
    @cached_property
    def force_vector_2(self):
        return np.array(self._distributed_moment.force_vector_2, dtype=np.float64)
    
    @cached_property
    def relative_position_1(self):
        return np.array(self._distributed_moment.relative_position_1, dtype=np.float64)
    
    @cached_property
    def relative_position_2(self):
        return np.array(self._distributed_moment.relative_position_2, dtype=np.float64)
    
    
    @cached_property
    def nodal_force_vector_in_elements_coord(self):
        F = np.zeros((self._elements_manager.num_elements, 12), dtype=np.float64)
        if self.num_distributed_moment > 0:
            Q = np.zeros((self.num_distributed_moment, 12), dtype=np.float64)

            L  = self._elements_manager.length[self.elements_indices]
            L2 = L*L
            L3 = L2*L

            l1  = self.relative_position_1*L
            l12 = l1*l1
            l13 = l12*l1

            l2  = self.relative_position_2*L
            l22 = l2*l2
            l23 = l22*l2

            E = self._elements_manager.youngs_modulus[self.elements_indices]
            G = self._elements_manager.shear_modulus[self.elements_indices]

            A     = self._elements_manager.cross_section_area[self.elements_indices]
            Iyy   = self._elements_manager.moment_of_inertia_about_y[self.elements_indices]
            Izz   = self._elements_manager.moment_of_inertia_about_z[self.elements_indices]
            kappa = self._elements_manager.shear_correction_factor[self.elements_indices]

            f1 = self.force_vector_1
            f2 = self.force_vector_2

            force_in_global_axis = self.axis >= DistributedForce.GLOBAL_X
            t = self._trans_matrix.elements_direction_cosines_3x3[self.elements_indices][force_in_global_axis]
            f1[force_in_global_axis] = np.einsum("nij,nj->ni", t, f1[force_in_global_axis])
            f2[force_in_global_axis] = np.einsum("nij,nj->ni", t, f2[force_in_global_axis])

            fx1 = f1[:, 0]
            fy1 = f1[:, 1]
            fz1 = f1[:, 2]

            fx2 = f2[:, 0]
            fy2 = f2[:, 1]
            fz2 = f2[:, 2]


            Q[:, 3] = -(l1 - l2)*(fx1*(3*L - 2*l1 - l2) + fx2*(3*L - l1 - 2*l2))/(6*L)
            Q[:, 9] = -Q[:, 3] + (fx1 + fx2)*(l2 - l1)/2

            kappaGA = kappa*G*A
            EIzz = E*Izz
            az = (fz2 - fz1)/(l2 - l1)
            bz = fz1 - az*l1

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
            Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + (fz1 + fz2)*(l2 - l1)/2

            EIyy = E*Iyy
            ay = (fy2 - fy1)/(l2 - l1)
            by = fy1 - ay*l1

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
            Q[:, 10] = -Q[:, 4] - Q[:, 8]*L - (fy1 + fy2)*(l2 - l1)/2

            np.add.at(F, self.elements_indices, Q)

        return F


class ConcentratedForceManager:
    
    def __init__(
        self,
        elements_manager: ElementsManager,
        concentrated_force: ConcentratedForce,
        trans_matrix: TransMatrix
    ):
        self._elements_manager = elements_manager
        self._concentrated_force         = concentrated_force
        self._trans_matrix = trans_matrix
        
    @cached_property
    def elements_indices(self):
        return np.array(self._concentrated_force.elements_indices, dtype=np.int64)

    @cached_property
    def axis(self):
        return np.array(self._concentrated_force.axis, dtype=np.int64)
    
    @cached_property
    def num_concentrated_force(self):
        return self.elements_indices.size
    
    @cached_property
    def force_vector(self):
        return np.array(self._concentrated_force.force_vector, dtype=np.float64)
    
    @cached_property
    def relative_position(self):
        return np.array(self._concentrated_force.relative_position, dtype=np.float64)
      
    @cached_property
    def nodal_force_vector_in_elements_coord(self):
        F = np.zeros((self._elements_manager.num_elements, 12), dtype=np.float64)
        
        if self.num_concentrated_force > 0:
            
            Q = np.zeros((self.num_concentrated_force, 12), dtype=np.float64)

            L  = self._elements_manager.length[self.elements_indices]

            l  = self.relative_position*L

            E = self._elements_manager.youngs_modulus[self.elements_indices]
            G = self._elements_manager.shear_modulus[self.elements_indices]

            A     = self._elements_manager.cross_section_area[self.elements_indices]
            Iyy   = self._elements_manager.moment_of_inertia_about_y[self.elements_indices]
            Izz   = self._elements_manager.moment_of_inertia_about_z[self.elements_indices]
            kappa = self._elements_manager.shear_correction_factor[self.elements_indices]

            f = self.force_vector
            
            force_in_global_axis = self.axis >= DistributedForce.GLOBAL_X
            t = self._trans_matrix.elements_direction_cosines_3x3[self.elements_indices][force_in_global_axis]
            f[force_in_global_axis] = np.einsum("nij,nj->ni", t, f[force_in_global_axis])
            
            fx = f[:, 0]
            fy = f[:, 1]
            fz = f[:, 2]

            Q[:, 0] = -fx*(l - L)/L
            Q[:, 6] = -Q[:, 0] + fx

            kappaGA = kappa*G*A
            EIzz = E*Izz

            Q[:, 1] = -fy*(-L + l)*(kappaGA*(L**2 + L*l - 2*l**2) + 12*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
            Q[:, 7] = -Q[:, 1] + fy

            Q[:, 5] = -l*fy*(-L + l)*(kappaGA*(L**2 - L*l) + 6*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
            Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + fy*l

            EIyy = E*Iyy

            Q[:, 2] = -fz*(-L + l)*(kappaGA*(L**2 + L*l - 2*l**2) + 12*EIyy)/(L*(kappaGA*L**2 + 12*EIyy))
            Q[:, 8] = -Q[:, 2] + fz

            Q[:, 4] = -l*fz*(-L + l)*(kappaGA*(L**2 - L*l) + 6*EIyy)/(L*(kappaGA*L**2 + 12*EIyy))
            Q[:, 10] = -Q[:, 4] - Q[:, 8]*L + fz*l

            np.add.at(F, self.elements_indices, Q)
        
        return F
        

class ConcentratedMomentManager:
    
    def __init__(
        self,
        elements_manager: ElementsManager,
        concentrated_moment: ConcentratedMoment,
        trans_matrix: TransMatrix
    ):
        self._elements_manager = elements_manager
        self._concentrated_moment         = concentrated_moment
        self._trans_matrix = trans_matrix
        
    @cached_property
    def elements_indices(self):
        return np.array(self._concentrated_moment.elements_indices, dtype=np.int64)

    @cached_property
    def axis(self):
        return np.array(self._concentrated_moment.axis, dtype=np.int64)
    
    @cached_property
    def num_concentrated_moment(self):
        return self.elements_indices.size
    
    @cached_property
    def force_vector(self):
        return np.array(self._concentrated_moment.force_vector, dtype=np.float64)
    
    @cached_property
    def relative_position(self):
        return np.array(self._concentrated_moment.relative_position, dtype=np.float64)
      
    @cached_property
    def nodal_force_vector_in_elements_coord(self):
        F = np.zeros((self._elements_manager.num_elements, 12), dtype=np.float64)
        
        if self.num_concentrated_moment > 0:
            
            Q = np.zeros((self.num_concentrated_moment, 12), dtype=np.float64)

            L  = self._elements_manager.length[self.elements_indices]

            l  = self.relative_position*L

            E = self._elements_manager.youngs_modulus[self.elements_indices]
            G = self._elements_manager.shear_modulus[self.elements_indices]

            A     = self._elements_manager.cross_section_area[self.elements_indices]
            Iyy   = self._elements_manager.moment_of_inertia_about_y[self.elements_indices]
            Izz   = self._elements_manager.moment_of_inertia_about_z[self.elements_indices]
            kappa = self._elements_manager.shear_correction_factor[self.elements_indices]

            f = self.force_vector
            
            force_in_global_axis = self.axis >= DistributedForce.GLOBAL_X
            t = self._trans_matrix.elements_direction_cosines_3x3[self.elements_indices][force_in_global_axis]
            f[force_in_global_axis] = np.einsum("nij,nj->ni", t, f[force_in_global_axis])
            
            fx = f[:, 0]
            fy = f[:, 1]
            fz = f[:, 2]


            Q[:, 3] = -fx*(l - L)/L
            Q[:, 9] = -Q[:, 3] + fx

            # calculate nodal forces in x-y plane
            kappaGA = kappa*G*A
            EIzz = E*Izz

            Q[:, 1] = -6*kappaGA*l*fz*(L - l)/(L*(kappaGA*L**2 + 12*EIzz))
            Q[:, 7] = -Q[:, 1]

            Q[:, 5] = -fz*(-L + l)*(kappaGA*(L**2 - 3*L*l) + 12*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
            Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + fz

            # calculate nodal forces in x-z plane
            EIyy = E*Iyy

            Q[:, 2] = 6*kappaGA*l*fy*(L - l)/(L*(kappaGA*L**2 + 12*EIyy))
            Q[:, 8] = -Q[:, 2]

            Q[:, 4] = fy*(-L + l)*(kappaGA*(L**2 - 3*L*l) + 12*EIyy)/(L*(kappaGA*L**2 + 12*EIyy)) 
            Q[:, 10] = -Q[:, 4] - Q[:, 8]*L - fy

            np.add.at(F, self._concentrated_moment.elements_indices, Q)
        
        return F


class TemperatureManager:
    
    def __init__(self, elements_manager: ElementsManager, temperature: Temperature):
        self._elements_manager = elements_manager
        self._temperature = temperature
    
    @cached_property
    def num_temperature_load(self):
        return np.count_nonzero(self._temperature.temperature_applied)    

    @cached_property    
    def nodal_force_vector_in_elements_coord(self):
        
        F = np.zeros((self._elements_manager.num_elements, 12), dtype=np.float64)
        if self.num_temperature_load > 0:
            Q = np.zeros((self.num_temperature_load, 12), dtype=np.float64)

            A = self._elements_manager.cross_section_area[self._temperature.temperature_applied]
            E = self._elements_manager.youngs_modulus[self._temperature.temperature_applied]
            Iyy = self._elements_manager.moment_of_inertia_about_y[self._temperature.temperature_applied]
            Izz = self._elements_manager.moment_of_inertia_about_z[self._temperature.temperature_applied]
            alpha = self._elements_manager.thermal_coefficient[self._temperature.temperature_applied]

            constant = self._temperature.constant[self._temperature.temperature_applied]
            linear_y = self._temperature.linear_y[self._temperature.temperature_applied]
            linear_z = self._temperature.linear_z[self._temperature.temperature_applied]

            fa = E*A*alpha*constant
            fmy = E*Iyy*alpha*linear_y
            fmz = E*Izz*alpha*linear_z

            Q[:, 0] = -fa    
            Q[:, 6] = fa    
            Q[:, 4] = -fmy    
            Q[:, 10] = fmy    
            Q[:, 5] = fmz    
            Q[:, 11] = -fmz   

            F[self._temperature.temperature_applied] = Q 

        return F


class ElementsLoadManager:
    
    def __init__(
        self,
        nodes_manager: NodesManager,
        elements_manager: ElementsManager,
        nodal_load_manager: NodalLoadManager,
        distributed_force_manager: DistributedForceManager,
        distributed_moment_manager: DistributedMomentManager,
        concentrated_force_manager: ConcentratedForceManager, 
        concentrated_moment_manager: ConcentratedMomentManager,
        temperature_manager: TemperatureManager,
        trans_matrix: TransMatrix,
        stiffness_matrix: StiffnessMatrix
    ):
        self._nodes_manager = nodes_manager
        self._elements_manager = elements_manager
        self._nodal_load_manager = nodal_load_manager
        self._distributed_force_manager = distributed_force_manager
        self._distributed_moment_manager = distributed_moment_manager
        self._concentrated_force_manager = concentrated_force_manager
        self._concentrated_moment_manager = concentrated_moment_manager
        self._temperature_manager = temperature_manager
        self._trans_matrix = trans_matrix
        self._stiffness_matrix = stiffness_matrix
        
    @cached_property
    def nodal_force_vector_in_elements_coord(self):
        f1 = self._distributed_force_manager.nodal_force_vector_in_elements_coord
        f2 = self._distributed_moment_manager.nodal_force_vector_in_elements_coord
        f3 = self._concentrated_force_manager.nodal_force_vector_in_elements_coord
        f4 = self._concentrated_moment_manager.nodal_force_vector_in_elements_coord
        f5 = self._temperature_manager.nodal_force_vector_in_elements_coord
        return f1 + f2 + f3 + f4 + f5

    @cached_property
    def nodal_force_vector_in_global_coord(self):
        f = np.einsum("nij,nj->ni", self._trans_matrix.elements_direction_cosines_12x12_T, self.nodal_force_vector_in_elements_coord)
        return f
    
    @cached_property
    def nodal_force_vector_in_nodal_coord(self):
        f = np.einsum("nij,nj->ni", self._trans_matrix.elements_nodes_direction_cosines_12x12_T, self.nodal_force_vector_in_global_coord)
        u = self._nodal_load_manager.displacement_vector_in_nodal_coord[self._elements_manager.nodes_indices].reshape(self._elements_manager.num_elements, 12)
        f -= np.einsum("nij,nj->ni", self._stiffness_matrix.stiffness_matrix_in_nodal_coord, u)
        return f
    
    @cached_property
    def structure_force_vector_in_nodal_coord(self):
        f = np.zeros(self._nodes_manager.num_free_dof, dtype=np.float64)
        elements_code_number_flat = self._elements_manager.code_number.ravel()
        elements_global_nodal_vector_flat = self.nodal_force_vector_in_nodal_coord.ravel()
        valid_indices = elements_code_number_flat < self._nodes_manager.num_free_dof
        np.add.at(f, elements_code_number_flat[valid_indices], elements_global_nodal_vector_flat[valid_indices])
        return f 

if __name__ == '__main__':
    from Sections import RectSections, Isections, SectionsManager, RectSectionsManager, ISectionsManager
    from Materials import MaterialsManager
    from LineElements import Elements
    
    nodes = Nodes()
    node1 = nodes.generate_node(0.0, 0.0, 0.0)
    node2 = nodes.generate_node(6.0, 0.0, 0.0)
    node3 = nodes.generate_node(12.0, 0.0, 0.0)
    node4 = nodes.generate_node(12.0, 3.0, 0.0)
    nodes.change_axes(node1, rx=45)
    nodes_manager = NodesManager(nodes)
    
    materials = Materials()
    mat1 = materials.generate_material("c30", 33e6, 0.49, 0.001)
    mat2 = materials.generate_material("d2", 33e6, 0.2)
    mat2 = materials.generate_material("a1", 33e6, 0.2)
    materials_manager = MaterialsManager(materials)

    sections = Sections()
    
    rect_sections = RectSections()
    rect1 = rect_sections.generate("b300x600", 0.6, 0.2)
    I_sections = Isections()
    I1 = I_sections.generate("I", 0.6, 0.05, 0.01, 0.3, 0.01, 0.3)
    
    rect_sections_manager = RectSectionsManager(rect_sections)
    I_sections_manager = ISectionsManager(I_sections)
    sections_manager = SectionsManager(rect_sections_manager, I_sections_manager)
    
    elements = Elements(nodes, materials, sections)
    elem1 = elements.generate_element(node1, node2, mat1, rect1)
    elem2 = elements.generate_element(node3, node2, mat1, I1)
    elem3 = elements.generate_element(node3, node4, mat1, I1)
    
    # elements.change_axes(elem1, 45)
    
    elements_manager = ElementsManager(nodes_manager, elements, materials_manager, sections_manager)
    trans_matrix = TransMatrix(nodes_manager, elements_manager)
    S = StiffnessMatrix(nodes_manager, elements_manager, trans_matrix)
    
    distributed_force = DistributedForce(elements_manager)
    # distributed_force.add(elem1, DistributedForce.GLOBAL_Y, 100.0, 100.0, 0.0, 1.0)
    dist_load_manager = DistributedForceManager(elements_manager, distributed_force, trans_matrix)
    
    
    dist_moment = DistributedMoment(elements_manager)
    # dist_moment.add(elem1, DisributedMoment.GLOBAL_Z, 100.0, 100.0, 0.0, 1.0)
    dist_moment_manager = DistributedMomentManager(elements_manager, dist_moment, trans_matrix)
    
    cf = ConcentratedForce(elements_manager)
    cf.add(elem1, ConcentratedForce.GLOBAL_Y, 1.0, 0.35)
    cfm = ConcentratedForceManager(elements_manager, cf, trans_matrix)
    
    cm = ConcentratedMoment(elements_manager)
    # cm.add(elem1, cm.GLOBAL_Y, 1.0, 0.35)
    cmm = ConcentratedMomentManager(elements_manager, cm, trans_matrix)
    
    t = Temperature(elements_manager)
    # t.add(elem1, 10, 10, 10)
    tm = TemperatureManager(elements_manager, t)
    
    
    nl = NodalLoad(nodes_manager)
    nl.add_force(node1, nl.GLOBAL_AXIS, 0.0, 1.0)
    # nl.add_displacement(node1, nl.LOCAL_AXIS, 0.0, 1.0)
    nlm = NodalLoadManager(nodes_manager, nl, trans_matrix)
    
    elm = ElementsLoadManager(nodes_manager, elements_manager, nlm, dist_load_manager, dist_moment_manager, cfm, cmm, tm, trans_matrix, S)
    
    print(elm.structure_force_vector_in_nodal_coord)