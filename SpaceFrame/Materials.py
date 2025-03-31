    
    
class Materials:
    """ 
    A class to manage materials properties for line elements.
    """
    
    def __init__(self) -> None:
        self._materials = []
        
    def addMaterial(self, E : float, nu : float) -> None:
        """
        Adds a material with its mechanical properties.

        **Parameters:**
        --------------
        - **E** (*float*): Young's modulus of the material (elastic modulus).
        - **nu** (*float*): Poisson's ratio.

        **Derived Property:**
        ---------------------
        - **G** (*float*): Shear modulus, computed internally as `E / (2 * (1 + nu))`.

        **Example Usage:**
        ------------------
        ```python
        materials = Materials()
        materials.addMaterial(200e9, 0.3)  # Steel material
        ```
        """
        G = E/(2 + 2*nu)
        self._materials.append([E, G, nu])
    
    def getMaterials(self) -> list:
        return self._materials
 