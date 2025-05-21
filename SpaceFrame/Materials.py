from functools import cached_property

import numpy as np
from numpy.typing import NDArray
    
class Materials:

    def __init__(self) -> None:
        self.names = []
        self.youngs_moduli = []
        self.poisson_ratios = []
    
    def _compute_shear_modulus(self, youngs_modulus: float, poisson_ratio: float):
        return youngs_modulus/(2 + 2*poisson_ratio)
    
    def find_index_by_name(self, name: str):
        return self.names.index(name)
        
    def generate_material(self, name: str, youngs_modulus: float, poisson_ratio: float): 
        
        self.names.append(name)
        self.youngs_moduli.append(youngs_modulus)
        self.poisson_ratios.append(poisson_ratio)
                
        return name
    
    def modify_material(self, name: str, new_name: str = None, youngs_modulus: float = None, poisson_ratio: float = None):
        pass
     
    def delete(self, name: str):
        pass
        
    
class MaterialsManager:
    
    def __init__(self, materials: Materials):
        self._materials = materials
    
    @cached_property
    def names(self):
        return np.array(self._materials.names, dtype=np.str_)
    
    @cached_property
    def youngs_moduli(self):
        return np.array(self._materials.youngs_moduli, dtype=np.float64)
    
    @cached_property
    def poisson_ratios(self):
        return np.array(self._materials.poisson_ratios, dtype=np.float64)
    
    @cached_property
    def shear_moduli(self):
        return self.youngs_moduli/(2 + 2*self.poisson_ratios)

if __name__ == '__main__':
    
    materials = Materials()
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    
    materials_manager = MaterialsManager(materials)
    print(materials_manager.names)