from functools import cached_property

import numpy as np
from numpy.typing import NDArray
class Sections:

    def __init__(self):
        self._names: list[str] = []

        self._cross_section_area_yz: list[float] = []
        self._moment_of_inertia_about_y: list[float] = []
        self._moment_of_inertia_about_z: list[float] = []
        self._torsional_constant: list[float] = []
        self._shear_correction_factor: list[float] = []
        self._dim_y: list[float] = []
        self._dim_z: list[float] = []
    
    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Expected {str}. Got {type(name)}")  
    
    def _validate_dim_y(self, dim_y: float):
        if not isinstance(dim_y, float):
            raise ValueError(f"Expected {float}. Got {type(dim_y)}")      
    
    def _validate_dim_z(self, dim_z: float):
        if not isinstance(dim_z, float):
            raise ValueError(f"Expected {float}. Got {type(dim_z)}")      
    
    
    def _compute_cross_section_area_yz(self, dim_y: float, dim_z: float):
        return dim_y*dim_z
    
    def _compute_moment_of_inertia_about_y(self, dim_y: float, dim_z: float):
        return dim_y*dim_z**3/12
     
    def _compute_moment_of_inertia_about_z(self, dim_y: float, dim_z: float):
        return dim_z*dim_y**3/12
     
    def _compute_torsional_constant(self, dim_y: float, dim_z: float):
        max_dim   = max(dim_y, dim_z)
        min_dim   = min(dim_y, dim_z)
        beta = 1/3 - 0.21*min_dim*(1 - (min_dim/max_dim)**4/12)/max_dim
        return beta*max_dim*min_dim**3

          
    def find_index_by_name(self, name: str):
        self._validate_name(name)
        
        try:
            return self._names.index(name)
        
        except ValueError:
            raise ValueError(f"Section with name: {name} wasn't found.")
        
    
    def generate_section(self, name: str, dim_y: float, dim_z: float):        
        self._validate_name(name)
        self._validate_dim_y(dim_y)
        self._validate_dim_z(dim_z)
                
        self._names.append(name)
        self._cross_section_area_yz.append(self._compute_cross_section_area_yz(dim_y, dim_z))
        self._moment_of_inertia_about_y.append(self._compute_moment_of_inertia_about_y(dim_y, dim_z))
        self._moment_of_inertia_about_z.append(self._compute_moment_of_inertia_about_z(dim_y, dim_z))
        self._torsional_constant.append(self._compute_torsional_constant(dim_y, dim_z))
        self._shear_correction_factor.append(5/6)

        return name

    def delete(self, name: str):
        index = self.find_index_by_name(name)
        self._names.pop(index)

    @cached_property
    def names(self):
        return self._names
    
    @cached_property
    def cross_section_area_yz(self):
        return np.array(self._cross_section_area_yz, dtype=np.float64)
    
    @cached_property
    def moment_of_inertia_about_y(self):
        return np.array(self._moment_of_inertia_about_y, dtype=np.float64)
    
    @cached_property
    def moment_of_inertia_about_z(self):
        return np.array(self._moment_of_inertia_about_z, dtype=np.float64)
    
    @cached_property
    def torsional_constant(self):
        return np.array(self._torsional_constant, dtype=np.float64)
    
    @cached_property
    def shear_correction_factor(self):
        return np.array(self._shear_correction_factor, dtype=np.float64)
    
    
if __name__ == '__main__':
    
    pass
