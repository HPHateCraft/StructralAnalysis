from functools import cached_property

from Nodes import Nodes
from Materials import Materials
from Sections import Sections
from LineElements import LineElements
from Loads import NodalLoadSolver, ElementsLoadSolver 
from StiffnessMatrix import StiffnessMatrix
from TransMatrix import TransMatrix

import numpy as np
from numpy.typing import NDArray

class Solver:
    
    def __init__(
        self,
        nodes: Nodes,
        elements: LineElements,
        nodal_load_solver: NodalLoadSolver,
        elements_load_solver: ElementsLoadSolver,
        trans_matrix: TransMatrix,
        stiffness_matrix: StiffnessMatrix
    ):
        self._nodes = nodes
        self._elements = elements
        self._stiffness_matrix = stiffness_matrix
        self._elements_load_solver = elements_load_solver
        self._trans_matrix = trans_matrix
        self._nodal_load_solver = nodal_load_solver
        
    def solve(self):
        self._elements_load_solver.solve() 
        self._solve_structure_displacement()
        
    def _solve_structure_displacement(self):
        self.displacement_vector = np.linalg.solve(self._stiffness_matrix.structure_stiffness_matrix, self._elements_load_solver.structure_load_vector + self._nodal_load_solver.structure_force_vector_in_nodal_coord)
    
    @cached_property
    def displacement_vector(self):
        return np.zeros(self._nodes.num_free_dof, dtype=np.float64)
    