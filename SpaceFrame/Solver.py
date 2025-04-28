from functools import cached_property

from Nodes import Nodes
from Materials import Materials
from Sections import Sections
from LineElements import LineElements
from Loads import Loads, LoadSolver 
from StiffnessMatrix import StiffnessMatrix
from TransMatrix import TransMatrix

import numpy as np
from numpy.typing import NDArray

class Solver:
    
    def __init__(self, nodes: Nodes, elements: LineElements, stiffness_matrix: StiffnessMatrix, load_solver: LoadSolver, trans_matrix: TransMatrix):
        self._nodes = nodes
        self._elements = elements
        self._stiffness_matrix = stiffness_matrix
        self._load_solver = load_solver
        self._trans_matrix = trans_matrix
           
    def solve(self):
        self._load_solver.solve() 
        self._solve_structure_displacement()
        
    def _solve_structure_displacement(self):
        self._stiffness_matrix.structure_displacement_vector = np.linalg.solve(self._stiffness_matrix.structure_stiffness_matrix, self._stiffness_matrix.structure_force_vector)
    
    @cached_property
    def global_nodal_displacment(self):
        self._stiffness_matrix.global_nodal_displacment[self._nodes.dof] = self._stiffness_matrix.structure_displacement_vector
        return self._stiffness_matrix.global_nodal_displacment
    
    @cached_property
    def elements_global_nodal_displacement(self):
        return np.reshape(self.global_nodal_displacment[self._elements.nodes_indices], (self._elements.num_elements, 12))
    
    @cached_property
    def elements_local_nodal_displacement(self):
        return np.einsum('nij,nj->ni', self._trans_matrix.trans_matrix_12x12, self.elements_global_nodal_displacement)
    
    @cached_property
    def elements_local_nodal_reaction(self):
        return np.einsum('nij,nj->ni', self._stiffness_matrix.local_stiffness_matrix, self.elements_local_nodal_displacement) - self._stiffness_matrix.elements_local_nodal_vector

    @cached_property
    def elements_global_nodal_reaction(self):
        return np.einsum('nij,nj->ni', self._trans_matrix.trans_matrix_12x12_T, self.elements_local_nodal_reaction)
    
    @cached_property
    def nodal_global_reaction(self):
        r = self._stiffness_matrix.nodal_Vector()
        np.add.at(r, self._elements.nodes_indices[:, 0], self.elements_global_nodal_reaction[:, :6])
        np.add.at(r, self._elements.nodes_indices[:, 1], self.elements_global_nodal_reaction[:, 6:])
        return r
         