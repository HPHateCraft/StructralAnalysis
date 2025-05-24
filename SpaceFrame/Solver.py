from functools import cached_property

from Nodes import NodesManager
from Loads import NodalLoadManager, ElementsLoadManager 
from StiffnessMatrix import StiffnessMatrix

import numpy as np
from numpy.typing import NDArray

class Solver:
    
    def __init__(
        self,
        nodes_manager: NodesManager,
        nodal_load_manager: NodalLoadManager,
        elements_load_manager: ElementsLoadManager,
        stiffness_matrix: StiffnessMatrix
    ):
        self._nodes_manager = nodes_manager
        self._stiffness_matrix = stiffness_matrix
        self._nodal_load_manager = nodal_load_manager
        self._elements_load_manager = elements_load_manager
        
    def solve(self):
        s = self._stiffness_matrix.structure_stiffness_matrix_in_nodal_coord
        fe = self._elements_load_manager.structure_force_vector_in_nodal_coord
        fn = self._nodal_load_manager.structure_force_vector_in_nodal_coord
        self.displacement_vector_in_nodal_coord = np.linalg.solve(s, fe + fn)
            
    @cached_property
    def displacement_vector_in_nodal_coord(self):
        return np.zeros(self._nodes_manager.num_free_dof, dtype=np.float64)
    