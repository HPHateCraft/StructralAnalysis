from functools import cached_property

import numpy as np
from numpy.typing import NDArray
    
class Materials:

    def __init__(self) -> None:
        self._names     : list[str]         = []
        
        self._youngs_modulus: list[float] = []
        self._poisson_ratio : list[float] = []
        self._shear_modulus : list[float] = []

    def _validate_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Name must be of type {str}. Got {type(name)}")
    
    def _validate_name_in_names(self, name: str):
        if name in self._names:
            raise ValueError(f"Material with name: {name} already exist.")
        
    def _validate_youngs_modulus(self, youngs_modulus: float):
        if not isinstance(youngs_modulus, (float, int)):
            raise TypeError(f"Young's modulus must be a number. Got {type(youngs_modulus)}")
        
        if youngs_modulus <= 0:
            raise ValueError(f"Young's modulus can't be <= 0. Got {youngs_modulus}")
    
    def _validate_poisson_ratio(self, poisson_ratio: float):
        if not isinstance(poisson_ratio, (float, int)):
            raise TypeError(f"Poisson ratio must be a number. Got {type(poisson_ratio)}")
        
        if poisson_ratio < 0:
            raise ValueError(f"Poisson ratio can't be < 0. Got {poisson_ratio}")
    
    def _compute_shear_modulus(self, youngs_modulus: float, poisson_ratio: float):
        return youngs_modulus/(2 + 2*poisson_ratio)
    
    def find_index_by_name(self, name: str):
        self._validate_name(name)
        
        try:
            return self._names.index(name)
        except ValueError:
            raise ValueError(f"Material with name {name} wasn't found.")
        
    def generate_material(self, name: str, youngs_modulus: float, poisson_ratio: float): 
        self._validate_name(name)
        self._validate_name_in_names(name)
        self._validate_youngs_modulus(youngs_modulus)
        self._validate_poisson_ratio(poisson_ratio)
        
        self._names.append(name)
        self._youngs_modulus.append(youngs_modulus)
        self._poisson_ratio.append(poisson_ratio)
        shear_modulus = self._compute_shear_modulus(youngs_modulus, poisson_ratio)
        self._shear_modulus.append(shear_modulus)
                
        return name
    
    def modify_material(self, name: str, new_name: str = None, youngs_modulus: float = None, poisson_ratio: float = None):
        index = self.find_index_by_name(name)
        
        if new_name is not None:
            self._validate_name(new_name)
        
            if new_name != name:
                self._validate_name_in_names(new_name)
                self._names[index] = new_name
        
        if youngs_modulus is not None:    
            self._validate_youngs_modulus(youngs_modulus)
            self._youngs_modulus[index] = youngs_modulus
        
        if poisson_ratio is not None:
            self._validate_poisson_ratio(poisson_ratio)
            self._poisson_ratio[index] = poisson_ratio
        
        if youngs_modulus is not None or poisson_ratio is not None: 
            self._shear_modulus[index] = self._compute_shear_modulus(self._youngs_modulus[index], self._poisson_ratio[index])
     
    def delete(self, name: str):
        index = self.find_index_by_name(name)
        self._names.pop(index)
        self._youngs_modulus.pop(index)
        self._poisson_ratio.pop(index)
        self._shear_modulus.pop(index)
    
    @cached_property    
    def names(self):
        return self._names
    
    @cached_property
    def youngs_modulus(self):
        return np.array(self._youngs_modulus, dtype=np.float64)
    
    @cached_property
    def poisson_ratio(self):
        return np.array(self._poisson_ratio, dtype=np.float64)
    
    @cached_property
    def shear_modulus(self):
        return np.array(self._shear_modulus, dtype=np.float64)
    
    
if __name__ == '__main__':
    
    pass