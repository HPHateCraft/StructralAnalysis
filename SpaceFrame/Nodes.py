import torch

class GlobalCoord:
    """
    The global coordinate system is a right-hand coordinate system.
      
    Vector Product:
    ---------------
        e2 x e3 = e1
        e3 x e1 = e2
        e1 x e2 = e3
        
        e1 = [1, 0, 0]
        e2 = [0, 1, 0]
        e3 = [0, 0, 1]

    Global Coordinate System:
    ------------------
        ```
               Y (e2)
               |
               |
               |
               |
               |_ _ _ _ _ _ X (e1)
               /        
              /
             /
            /   
            Z (e3)
        ```
    """
    ...


class Nodes:
    """
    A class to manage nodes and their degrees of freedom (DOF) in a 3D space.
    
    **Definition:**
    ---------------
    - A node in 3D space is defined by its global coordinates **[X, Y, Z]**.
    - Each node has **6 degrees of freedom (DOF)**:  
      - **Translations**: [UX, UY, UZ] (Displacements along X, Y, Z)  
      - **Rotations**: [RX, RY, RZ] (Rotations about X, Y, Z)  

    **References:**
    ---------------
    - The coordinate system follows the **right-hand Cartesian system** (See `GlobalCoord` for details).
    """
    
    def __init__(self) -> None:
        self._nodes = []
        self._dof   = []
        
    def addNodeCoord(self, X : float, Y : float, Z : float) -> None:
        """
        Adds a node to the global coordinate system.
        
        When adding a node all DOF corospond to the node are set to True (free) by default.
        
        **Parameters:**
        --------------
        - **X** (*float*): Coordinate along the global X-axis.
        - **Y** (*float*): Coordinate along the global Y-axis.
        - **Z** (*float*): Coordinate along the global Z-axis.

        **Raises:**
        ----------
        - `ValueError`: If the node already exists.

        **Example Usage:**
        ------------------
        ```python
        nodes = Nodes()
        nodes.addNodeCoord(1.0, 2.0, 3.0)
        ```
        """

        node = [X, Y, Z]

        if node in self._nodes:
            raise ValueError(f"Node {node} already exists.")
            
        self._nodes.append(node)
        self._dof.append([True, True, True, True, True, True])
    
    def getNodesCoord(self) -> torch.Tensor:
        """
        Returns the coordinates of all nodes in the global coordinate system.

        The coordinates are returned as a 2D tensor of shape `(n, 3)`, where `n` 
        is the number of nodes.

        **Returns:**
        -----------
        - `torch.Tensor`: A tensor of shape `(n, 3)` containing the coordinates 
        of all nodes in the system, with `dtype=torch.float64`.

        **Example Usage:**
        ------------------
        ```python
        nodes = Nodes()
        nodes.addNodeCoord(1.0, 2.0, 3.0)
        nodes.addNodeCoord(4.0, 5.0, 6.0)
        coords = nodes.getNodesCoord()
        print(coords)  # Output: tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
        ```
        """
        return torch.tensor(self._nodes, dtype=torch.float64)
    
    def getDof(self) -> torch.Tensor:
        """
        Returns the degrees of freedom (DOF) for all nodes as a boolean tensor.

        Each node has **6 degrees of freedom (DOF)**:  
        - **Translations**: [UX, UY, UZ] (Displacements along X, Y, Z)  
        - **Rotations**: [RX, RY, RZ] (Rotations about X, Y, Z)  

        The DOF tensor indicates whether each degree of freedom is active (`True`) or 
        constrained (`False`). The tensor has a shape of `(n, 6)`, where `n` is the 
        number of nodes, and each node has 6 degrees of freedom.

        **Returns:**
        -----------
        - `torch.Tensor`: A boolean tensor of shape `(n, 6)` indicating the status of 
          each degree of freedom, with `dtype=torch.bool`.

        **Example Usage:**
        ------------------
        ```python
        nodes = Nodes()
        nodes.addNodeCoord(1.0, 2.0, 3.0)  # Adds a node with all DOF set to True
        nodes.addNodeCoord(4.0, 5.0, 6.0)  # Adds a node with all DOF set to True
        dof = nodes.getDof()
        print(dof)  # Output: tensor([[True, True, True, True, True, True], [True, True, True, True, True, True]], dtype=torch.bool)
        ```
        """
        return torch.tensor(self._dof, dtype=torch.bool)
    
    def modifyDof(self, nodeIndex : int, UX : bool, UY : bool, UZ : bool, RX : bool, RY : bool, RZ : bool) -> None:
        """
        Modifies the degrees of freedom (DOF) for a specific node.

        **Parameters:**
        --------------
        - **nodeIndex** (*int*): The index of the node whose DOF are to be modified.  
        - **UX, UY, UZ** (*bool*): Status of the translation DOF along X, Y, and Z axes.  
        - **RX, RY, RZ** (*bool*): Status of the rotation DOF about X, Y, and Z axes.  

        **Example Usage:**
        ------------------
        ```python
        nodes = Nodes()
        nodes.addNodeCoord(1.0, 2.0, 3.0)  # Adds a node
        nodes.addNodeCoord(4.0, 5.0, 6.0)  # Adds another node

        # Modify DOF for the first node (index 0)
        nodes.modifyDof(0, UX=True, UY=False, UZ=True, RX=False, RY=True, RZ=False)

        # Retrieve DOF
        dof = nodes.getDof()
        print(dof)  # Output: tensor([[True, False, True, False, True, False], [True, True, True, True, True, True]], dtype=torch.bool)
        ```
        """
        self._dof[nodeIndex] = [UX, UY, UZ, RX, RY, RZ]
  
    def nodesSprings(self, nodeIndex : int, UX: float, UY: float, UZ: float, RX: float, RY: float, RZ: float):
        
        return
    
