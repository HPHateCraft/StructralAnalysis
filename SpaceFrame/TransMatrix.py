from functools import cached_property

from Nodes import Nodes
from LineElements import LineElements

import numpy as np    
class TransMatrix:
    
    def __init__(self, nodes: Nodes, elements: LineElements):
        self._nodes = nodes
        self._elements = elements
    
    @cached_property
    def nodes_direction_cosines_3x3(self):
        return self._nodes.directions_cosines
    
    @cached_property
    def nodes_direction_cosines_6x6(self):
        r = np.zeros((self._nodes.num_nodes, 6, 6), dtype=np.float64)
        r[:, 0:3, 0:3] = self.nodes_direction_cosines_3x3
        r[:, 3:6, 3:6] = self.nodes_direction_cosines_3x3
        return r
    
    @cached_property
    def nodes_direction_cosines_6x6_T(self):
        return self.nodes_direction_cosines_6x6.transpose(0, 2, 1)
    
    @cached_property
    def elements_nodes_direction_cosines_12x12(self):
        r = np.zeros((self._elements.num_elements, 12, 12))
        r[:, 0:6, 0:6] = self.nodes_direction_cosines_6x6[self._elements.nodes_indices[:, 0]]
        r[:, 6:12, 6:12] = self.nodes_direction_cosines_6x6[self._elements.nodes_indices[:, 1]]
        return r
    
    @cached_property
    def elements_nodes_direction_cosines_12x12_T(self):
        return self.elements_nodes_direction_cosines_12x12.transpose(0, 2, 1)
    
    @cached_property
    def elements_direction_cosines_3x3(self):
        return self._elements.direction_cosines
    
    @cached_property
    def elements_direction_cosines_12x12(self):   
        t3x3 = self.elements_direction_cosines_3x3
        t12x12 = np.zeros((self._elements.num_elements, 12, 12), dtype=np.float64)
        
        t12x12[:, 0:3, 0:3]   = t3x3
        t12x12[:, 3:6, 3:6]   = t3x3
        t12x12[:, 6:9, 6:9]   = t3x3
        t12x12[:, 9:12, 9:12] = t3x3
    
        return t12x12
    
    @cached_property
    def elements_direction_cosines_12x12_T(self):
        return self.elements_direction_cosines_12x12.transpose(0, 2, 1)