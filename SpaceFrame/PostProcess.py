from functools import cached_property

from Nodes import Nodes
from LineElements import LineElements
from TransMatrix import TransMatrix
from StiffnessMatrix import StiffnessMatrix
from Solver import Solver
from Loads import NodalLoadSolver, ElementsLoadSolver

import numpy as np

class PostProcess:
    
    def __init__(
        self,
        nodes: Nodes,
        elements: LineElements,
        nodal_load_solver: NodalLoadSolver,
        elements_load_solver: ElementsLoadSolver, 
        trans_matrix: TransMatrix,
        stiffness_matrix: StiffnessMatrix,
        solver: Solver
    ):
        self._nodes = nodes
        self._elements = elements
        self._trans_matrix = trans_matrix
        self._stiffness_matrix = stiffness_matrix
        self._solver = solver
        self._nodal_load_solver = nodal_load_solver
        self._elements_load_solver = elements_load_solver
    
    @cached_property
    def nodes_displacement_in_nodal_coord(self):
        v = np.zeros((self._nodes.num_nodes, 6), dtype=np.float64)
        v[self._nodes.dof] = self._solver.displacement_vector
        v += self._nodal_load_solver.displacement_vector_in_nodal_coord
        return v
    
    @cached_property    
    def nodes_displacement_in_global_coord(self):
        return np.einsum("nij,nj->ni", self._trans_matrix.nodes_direction_cosines_6x6, self.nodes_displacement_in_nodal_coord)
    
    @cached_property
    def elements_nodal_displacement_in_global_coord(self):
        return np.reshape(self.nodes_displacement_in_global_coord[self._elements.nodes_indices], (self._elements.num_elements, 12))
    
    @cached_property
    def elements_nodal_displacement_in_elements_coord(self):
        return np.einsum("nij,nj->ni", self._trans_matrix.elements_direction_cosines_12x12, self.elements_nodal_displacement_in_global_coord)
    
    @cached_property
    def elements_nodal_forces_in_elements_coord(self):
        return np.einsum('nij,nj->ni', self._stiffness_matrix.stiffness_matrix_in_elements_coord, self.elements_nodal_displacement_in_elements_coord) - self._elements_load_solver.load_vector_in_element_coord
    
    @cached_property
    def elements_nodal_forces_in_global_coord(self):
        return np.einsum("nij,nj->ni", self._trans_matrix.elements_direction_cosines_12x12_T, self.elements_nodal_forces_in_elements_coord)
    
    @cached_property    
    def elements_nodal_forces_in_nodal_coord(self):
        return np.einsum("nij,nj->ni", self._trans_matrix.elements_nodes_direction_cosines_12x12_T, self.elements_nodal_forces_in_global_coord)
    
    @cached_property
    def nodes_forces_in_nodal_coord(self):
        r = np.zeros((self._nodes.num_nodes, 6), dtype=np.float64)
        np.add.at(r, self._elements.nodes_indices[:, 0], self.elements_nodal_forces_in_nodal_coord[:, :6])
        np.add.at(r, self._elements.nodes_indices[:, 1], self.elements_nodal_forces_in_nodal_coord[:, 6:])
        r -= self._nodal_load_solver.force_vector_in_nodal_coord
        return r 
    
    @cached_property
    def nodes_forces_in_global_coord(self):
        return np.einsum("nij,nj->ni", self._trans_matrix.nodes_direction_cosines_6x6, self.nodes_forces_in_nodal_coord)