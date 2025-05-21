
from functools import cached_property
import bisect
import math 

from GlobalCoord import GlobalCoord
from Nodes import Nodes
from Materials import Materials
from Sections import Sections

import numpy as np
from numpy.typing import NDArray

class Elements:
    
    PENALTY_NUMBER = 1e6
    
    def __init__(self, nodes: Nodes, materials: Materials, sections: Sections):
        self.nodes = nodes
        self.materials = materials
        self.sections = sections

        self._id = 0
        self.ids = []
        self.nodes_ids = []
        self.materials_ids = []
        self.sections_ids = []
        
    def _generate_id(self):
        id_ = self._id
        self.ids.append(id_)
        self._id += 1
        return id_
   
    def find_index_by_id(self, id_: int):
        index = bisect.bisect_left(self._ids, id_)
        if index < len(self._ids) and self._ids[index] == id_:
            return index  
        else:
            raise ValueError(f"Node with id: {id_} doesn't exist.")
    
    def generate_element(self, nodei_id: int, nodej_id: int, material_name: str, section_name: str, roll_angle: float = 0.0):     
           
        id_ = self._generate_id()
        self.nodes_ids.append(nodei_id)
        
         
        return id_
    
    def add_partial_fixity(
        self,
        id_: int,
        u1 : float | None = None,
        u2 : float | None = None,
        u3 : float | None = None,
        u4 : float | None = None,
        u5 : float | None = None,
        u6 : float | None = None,
        u7 : float | None = None,
        u8 : float | None = None,
        u9 : float | None = None,
        u10: float | None = None,
        u11: float | None = None,
        u12: float | None = None
    ):
        index = self.find_index_by_id(id_)
        partial_fixity = np.array([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12], dtype=np.float64)
        # partial_fixity[np.isnan(partial_fixity)] = self.PENALTY_NUMBER
        self._partial_fixity_indices.append(index)
        self._partial_fixity_vector.append(partial_fixity)
    
    @cached_property
    def partial_fixity_vector(self):
        return np.array(self._partial_fixity_vector, dtype=np.float64)
    
    @cached_property
    def partial_fixity_indices(self):
        return np.array(self._partial_fixity_indices, dtype=np.int64)
    
    @cached_property
    def num_elements_with_partial_fixity(self):
        return self.partial_fixity_indices.shape[0]
    
    @cached_property 
    def direction_cosines(self):
        return np.array(self._direction_cosines, dtype=np.float64)
    
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
    def materials_indices(self):
        return np.array(self._materials_indices, dtype=np.int64)
    
    @cached_property
    def sections_indices(self):
        return np.array(self._sections_indices, dtype=np.int64)
    
    @cached_property    
    def code_number(self):
        return np.reshape(self._nodes.code_number[self.nodes_indices], (self.num_elements, 12))
    
    def ids(self):
        return self._ids
    

class ElementsManager:
    
    def __init__(self, elements: Elements):
        self._elements = elements
    
    @cached_property
    def nodes_indices(self):
        return np.array(self._elements.nodes_indices, dtype=np.int64)

    @cached_property
    def coord(self):
        return
    
    