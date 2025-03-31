
import torch

class LineElements:
    """
    A class to manage line elements in a 3D finite element model.

    **Definition:**
    ---------------
    - A **line element** connects **two nodes** in 3D space.
    - Each node has **6 degrees of freedom (DOF)**:  
      - **Translations**: [UX, UY, UZ] (Displacements along X, Y, Z)  
      - **Rotations**: [RX, RY, RZ] (Rotations about X, Y, Z)  
    - Each element has **12 total DOF** (6 per node).
    - Displacements and rotations are computed at the **nodes**.

    **Local Coordinate System:**
    ----------------------------
    - A **local coordinate system (x, y, z)** is defined for each line element.
    - The **x-axis** is oriented along the element length from **node i** to **node j**.
    - The **y- and z-axes** align with the cross-section’s principal axes of inertia.
    - If needed, the local coordinate system can be rotated about the x-axis by an **angle of roll**.

    **Vector Product:**
    -------------------------
    - The local coordinate system follows the **right-hand rule**:
        ```
        e2 × e3 = e1
        e3 × e1 = e2
        e1 × e2 = e3
        ```
    - The standard unit vectors are:
        - `e1 = [1, 0, 0]` (x-direction)
        - `e2 = [0, 1, 0]` (y-direction)
        - `e3 = [0, 0, 1]` (z-direction)

    **Visualization:**
    -----------------
    ```
           y (e2)
           |
           |
           |
           |
           |_ _ _ _ _ _ x (e1)
           /        
          /  
         /    
        /    
        z (e3)
    ```
    """
    
    def __init__(self):
        self._elementsNodesIndices   = []
        self._elementsProperties     = []
        self._deltaNodes             = []
    
    def addElement(self, nodesCoord : torch.Tensor, nodeiIndex : int, nodejIndex : int, sections : list, sectionIndex : int, angleOfRoll : float = 0.0) -> None:
        """
        Add a new line element.
            

        **Parameters:**
        --------------
        - **nodesCoord** (*torch.Tensor*):  
        A tensor of shape `(n, 3)` containing the coordinates of all nodes in the system.  
        Each row represents a node as `[X, Y, Z]`.

        - **nodeiIndex** (*int*):  
        Index of the first node.

        - **nodejIndex** (*int*):  
        Index of the second node.

        - **sections** (*list*):  
        A 2D list containing cross-sectional properties of all available sections.

        - **sectionIndex** (*int*):  
        The index corresponding to the cross-section assigned to this element.

        - **angleOfRoll** (*float*, optional):  
        The **angle of roll**, measured in **Degrees**, defining the rotation of the local coordinate system  
        about the element’s local x-axis.  
            - **Positive**: Clockwise rotation when viewed in the **negative x-direction**.  
            - Default value is `0.0` (no roll rotation).
            
        **Derived Properties:**
        -----------------------
        - **L** (*float*): Element length, computed as the Euclidean distance between `node i` and `node j`.
        - **h** (*float*): Dimension along the local **y-axis**.
        - **b** (*float*): Dimension along the local **z-axis**.
        - **A** (*float*): Cross-sectional area.
        - **Izz** (*float*): Second moment of area about the **local z-axis**.
        - **Iyy** (*float*): Second moment of area about the **local y-axis**.
        - **J** (*float*): Torsional constant.
        - **E** (*float*): Young’s modulus of the material (elastic modulus).
        - **G** (*float*): Shear modulus.
        - **ν** (*float*): Poisson’s ratio.
        - **kappa** (*float*): First-order shear correction factor.
        """
        
        self._elementsNodesIndices.append([nodeiIndex, nodejIndex])

        delta  = nodesCoord[nodejIndex] - nodesCoord[nodeiIndex]
        self._deltaNodes.append(delta.tolist())

        delta2  = delta**2
        section = sections[sectionIndex]
        L       = torch.sqrt(delta2.sum()).item()
        h       = section[0]
        b       = section[1]
        A       = section[2]
        Izz     = section[3]
        Iyy     = section[4]
        J       = section[5]
        E       = section[6]
        G       = section[7]
        nu      = section[8]
        kappa   = section[9]
        
        self._elementsProperties.append([L, h, b, A, Izz, Iyy, J, E, G, nu, kappa, angleOfRoll])

    def getElementsNodesIndices(self) -> torch.Tensor:
        """
        Returns the nodes indices attached to each line element.
        
        Return
        ------
        2D torch.Tensor of shape (n, 2).
        
        [
            [nodeiIndex, nodejIndex] -> element 0,
            [nodeiIndex, nodejIndex] -> element 1,
            [nodeiIndex, nodejIndex] -> element 2
        ] 
        """
        return torch.tensor(self._elementsNodesIndices, dtype=torch.int64)
    
    def getElementsProperties(self) -> torch.Tensor:
        return torch.tensor(self._elementsProperties, dtype=torch.float64)

    def getNumElements(self) -> int:
        return len(self._elementsNodesIndices)
    
    def getDeltaNodes(self) -> torch.Tensor:
        """
        Returns the diffrence between nodej and nodei coordinates (nodejcoord - nodeicoord) for each line element.
        
        Return
        ------
        2D torch.tensor of shape (n, 3).
        
        Tensor Shape
        -----------
        ```
        [
            [Xj - Xi, Yj - Yi, Zj - Zi] -> element 0,
            [Xj - Xi, Yj - Yi, Zj - Zi] -> element 1,
            [Xj - Xi, Yj - Yi, Zj - Zi] -> element 2
        ]
        
        """
        return torch.tensor(self._deltaNodes, dtype=torch.float64)
    
    def activeNodes(self, elementsNodesIndices : torch.Tensor) -> torch.Tensor:
        """
        Return Only Nodes That are Attached to Line Elements. 
        """
        return elementsNodesIndices.unique()
    
    def numActiveNodes(self, activeNodes: torch.Tensor):
        return activeNodes.shape[0]
    
    def activeNodesDof(self, activeNodes : torch.Tensor, dof : torch.Tensor) -> torch.Tensor:
        """
        Returns the DOF corresponding to the given active nodes.

        Inputs:
        ----------
        activeNodes : torch.Tensor
        A 1D tensor containing indices of active nodes.
        dof : torch.Tensor
        A 2D tensor of shape (N, 6), where N is the total number of nodes.
        Each row represents the six DOFs of a node.

        Returns
        -------
        torch.Tensor
        A 2D tensor of shape (len(activeNodes), 6), containing the DOFs of the specified active nodes.

        Examples
        --------
        >>> activeNodes = torch.tensor([0, 2, 3])
        >>> dof = torch.tensor([[1, 2, 3, 4, 5, 6],
        ...                     [7, 8, 9, 10, 11, 12],
        ...                     [13, 14, 15, 16, 17, 18],
        ...                     [19, 20, 21, 22, 23, 24]])
        >>> result = activeNodesDof(activeNodes, dof)
        >>> print(result)
        tensor([[ 1,  2,  3,  4,  5,  6],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24]])
        """
        return dof[activeNodes]
    
    def totalNumActiveNodesDof(self, activeNodesDof : torch.Tensor) -> int:
        """
        Rturn Total Number of DOF for Active Nodes. 
        """
        return activeNodesDof.numel()
    
    def numActiveNodesDof(self, activeNodesDof : torch.Tensor) -> int:
        """
        Return Number of Active DOF for the Active Nodes 
        """
        return int(activeNodesDof.count_nonzero())
    
    def activeNodesCodeNum(self, activeNodesDof : torch.Tensor, numActiveNodesDof : torch.Tensor, totalNumActiveNodesDof : int) -> torch.Tensor:
        """ 
        Establish the Node Code Number.
        
        Example:
        
            activeNodesDof = [
                [True, True, True, False, False, True],
                [True, False, True, False, False, True],
                [True, True, True, False, True, True]
            ]
            
            activeNodesCodeNum = [
                [0, 1, 2, 12, 13, 3],
                [4, 14, 5, 15, 16, 6],
                [7, 8, 9, 17, 10, 11]
            ]
        """
        codeNum                  = torch.empty_like(activeNodesDof, dtype=torch.int64)
        codeNum[activeNodesDof]  = torch.arange(0, numActiveNodesDof, 1, dtype=torch.int64)
        codeNum[~activeNodesDof] = torch.arange(numActiveNodesDof, totalNumActiveNodesDof, 1, dtype=torch.int64)
        return codeNum
    
    def elementsCodeNum(self, activeNodesCodeNum : torch.Tensor, elementsNodesIndices : torch.Tensor) -> torch.Tensor:
        """
        Return Line Elements Code Number To construc the Global Structre Stiffness Matrix.
        
        Example:
        
            elementsNodesIndices = [
                [0, 1],
                [2, 1],
                [0, 3]
            ]
            
            activeNodesCodeNum = [
                [0, 1, 2, 12, 13, 3],
                [4, 14, 5, 15, 16, 6],
                [7, 8, 9, 17, 10, 11]
            ]
            
            elementsCodeNum = [
                [0, 1, 2, 12, 13, 3, 4, 14, 5, 15, 16, 6],
                [7, 8, 9, 17, 10, 11, 4, 14, 5, 15, 16, 6],
                [0, 1, 2, 12, 13, 3, 7, 8, 9, 17, 10, 11]
            ]
            
        """
        return activeNodesCodeNum[elementsNodesIndices].flatten(1)
    
    
    
    def midPoints(self, nodesCoord: torch.Tensor, elementsNodesIndices: torch.Tensor):
        midPoint = nodesCoord[elementsNodesIndices]
        return (midPoint[:, 1, :] + midPoint[:, 0, :])/2