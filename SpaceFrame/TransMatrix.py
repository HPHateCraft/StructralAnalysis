from functools import cached_property

from LineElements import LineElements

import numpy as np    
class TransMatrix:
    
    def __init__(self, elements: LineElements):
        self._elements = elements
    
    @cached_property
    def trans_matrix_3x3(self):
        return self._elements.direction_cosines
    
    @cached_property
    def trans_matrix_12x12(self):   
        t3x3 = self.trans_matrix_3x3
        t12x12 = np.zeros((self._elements.num_elements, 12, 12), dtype=np.float64)
        
        t12x12[:, 0:3, 0:3]   = t3x3
        t12x12[:, 3:6, 3:6]   = t3x3
        t12x12[:, 6:9, 6:9]   = t3x3
        t12x12[:, 9:12, 9:12] = t3x3
    
        return t12x12
    
    @cached_property
    def trans_matrix_12x12_T(self):
        return self.trans_matrix_12x12.transpose(0, 2, 1)