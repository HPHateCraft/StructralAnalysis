from functools import cached_property

import numpy as np
from numpy.typing import NDArray
    
class Materials:

    def __init__(self) -> None:
        self.names = []
        self.youngs_modulus = []
        self.poisson_ratio = []
        self.thermal_coefficient = []
    
    def find_index_by_name(self, name: str):
        return self.names.index(name)
        
    def generate_material(self, name: str, youngs_modulus: float, poisson_ratio: float, thermal_coefficient: float = 0.00017): 

        self.names.append(name)
        self.youngs_modulus.append(youngs_modulus)
        self.poisson_ratio.append(poisson_ratio)
        self.thermal_coefficient.append(thermal_coefficient)
        
        return name
    
        
class MaterialsManager:
    
    def __init__(self, materials: Materials):
        self._materials = materials
    
    @cached_property
    def names(self):
        return np.array(self._materials.names, dtype=np.str_)
    
    @cached_property
    def youngs_modulus(self):
        return np.array(self._materials.youngs_modulus, dtype=np.float64)
    
    @cached_property
    def poisson_ratio(self):
        return np.array(self._materials.poisson_ratio, dtype=np.float64)
    
    @cached_property
    def thermal_coefficient(self):
        return np.array(self._materials.thermal_coefficient, dtype=np.float64)
    
    @cached_property
    def shear_modulus(self):
        return self.youngs_modulus/(2 + 2*self.poisson_ratio)

if __name__ == '__main__':
    
    materials = Materials()
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    
    materials_manager = MaterialsManager(materials)
    print(materials_manager.names)