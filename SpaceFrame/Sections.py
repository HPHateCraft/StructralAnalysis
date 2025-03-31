    
class Sections:
    """
    A class to manage sections properties for line elements.
    
    A Cross Section is Defined as the section in the local y-z plane
    """
    def __init__(self):
        self._rectSection = []

    def addRectSection(self, h : float, b : float, materials : list, materialsIndex : int, kappa : float = 5/6) -> None:
        """
        Add rectangular cross-section.
        
        **Parameters:**
        --------------
        - **h** (*float*): dimension along the local `y`-axis.
        - **b** (*float*): dimension along the local `z`-axis.
        - **materials** (*list*): A 2D list contain materials properties.
        - **materialsIndex** (*int*): Index of the material to be used from the `materials` list.
        - **kappa** (*float*, optional): First-order shear correction factor. Default value is `5/6`.

        **Symbols:**
        -----------------------
        - **A** (*float*): Cross-sectional area.
        - **Izz** (*float*): Moment of inertia about the local `z`-axis.
        - **Iyy** (*float*): Moment of inertia about the local `y`-axis.
        - **J** (*float*): Torsional constant.
        - **E** (*float*): Young's modulus of the material (elastic modulus).
        - **nu** (*float*): Poisson's ratio.
        - **G** (*float*): Shear modulus.

        ```
        **Coordinate System:**
        ----------------------
            The cross-section is defined within the local coordinate system:

                      y
                      |
                 _ _ _| _ _ _   
                |     |     |^
                |     |     ||
           z _ _ _ _ _|     |h
                |           ||
                |_ _ _ _ _ _|v
                <-----b----->
                
        """

        
        A  = h*b
        Izz = h**3*b/12
        Iyy = h*b**3/12

        if b > h:
            beta = 1/3 - 0.21*h*(1 - (h/b)**4/12)/b
            J = beta*h**3*b

        else:
            beta = 1/3 - 0.21*b*(1 - (b/h)**4/12)/h
            J = beta*h*b**3

        matProp = materials[materialsIndex]
        E       = matProp[0]
        G       = matProp[1]
        nu      = matProp[2]

        self._rectSection.append([h, b, A, Izz, Iyy, J, E, G, nu, kappa])

    def getRectSections(self) -> list:
        """
        Returns a 2D list. (n, m) where n is the number of rectangular sections added and m is the number of properties.
        
        **Returns:**
        -----------
        - **list**: A 2D list containing the properties of all added rectangular sections.  
        """
        return self._rectSection

