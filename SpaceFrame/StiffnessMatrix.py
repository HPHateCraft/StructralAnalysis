import torch
    
class StiffnessMatrix:
    
    def __init__(self):
        pass
    
    def localK(self, numElements : int, elementsProperties : torch.Tensor):
        K = torch.zeros((numElements, 12, 12), dtype=torch.float64)
        
        L     = elementsProperties[:, 0]
        L2    = L*L
        L3    = L2*L
        A     = elementsProperties[:, 3]
        Izz   = elementsProperties[:, 4]
        Iyy   = elementsProperties[:, 5]
        J     = elementsProperties[:, 6]
        E     = elementsProperties[:, 7]
        G     = elementsProperties[:, 8]
        kappa = elementsProperties[:, 10]
        
            # Compute axial stiffness terms
        K[:, 0, 0] =  E*A/L
        K[:, 0, 6] = -K[:, 0, 0]
        K[:, 6, 0] = -K[:, 0, 0]
        K[:, 6, 6] =  K[:, 0, 0]
        
        # Compute Torsional stiffness terms
        K[:, 3, 3] =  G*J/L
        K[:, 3, 9] = -K[:, 3, 3]
        K[:, 9, 3] = -K[:, 3, 3]
        K[:, 9, 9] =  K[:, 3, 3]
        
        # Compute bending stiffness terms in x-y plane
        betaxy = 12*E*Izz/(kappa*G*A*L2)
        
        K[:, 1, 1]  =  12*E*Izz/(L3*(1 + betaxy))
        K[:, 1, 7]  = -K[:, 1, 1]
        K[:, 7, 1]  = -K[:, 1, 1]   
        K[:, 7, 7]  =  K[:, 1, 1]
        
        K[:, 1, 5]  =  6*E*Izz/(L2*(1 + betaxy))
        K[:, 1, 11] =  K[:, 1, 5]
        K[:, 5, 1]  =  K[:, 1, 5]    
        K[:, 5, 7]  = -K[:, 1, 5]    
        K[:, 7, 5]  = -K[:, 1, 5]    
        K[:, 7, 11] = -K[:, 1, 5]    
        K[:, 11, 1] =  K[:, 1, 5]    
        K[:, 11, 7] = -K[:, 1, 5]  
        
        K[:, 5, 11] = (2 - betaxy)*E*Izz/(L*(1 + betaxy))
        K[:, 11, 5] = K[:, 5, 11]
        
        K[:, 5, 5]   = (4 + betaxy)*E*Izz/(L*(1 + betaxy))
        K[:, 11, 11] = K[:, 5, 5]
        
        # Compute bending stiffness terms in x-z plane
        betaxz = 12*E*Iyy/(kappa*G*A*L2)
        
        K[:, 2, 2]  = 12*E*Iyy/(L3*(1 + betaxz))
        K[:, 2, 8]  = -K[:, 2, 2]
        K[:, 8, 2]  = -K[:, 2, 2]
        K[:, 8, 8]  =  K[:, 2, 2]
        
        K[:, 2, 4]  = -6*E*Iyy/(L2*(1 + betaxz))
        K[:, 2, 10] =  K[:, 2, 4]
        K[:, 4, 2]  =  K[:, 2, 4]
        K[:, 4, 8]  = -K[:, 2, 4]
        K[:, 8, 4]  = -K[:, 2, 4]
        K[:, 8, 10] = -K[:, 2, 4]
        K[:, 10, 2] =  K[:, 2, 4]
        K[:, 10, 8] = -K[:, 2, 4]
        
        K[:, 4, 10] = (2 - betaxz)*E*Iyy/(L*(1 + betaxz))
        K[:, 10, 4] = K[:, 4, 10]
        
        K[:, 4, 4]   = (4 + betaxz)*E*Iyy/(L*(1 + betaxz))
        K[:, 10, 10] = K[:, 4, 4]
        
        return K
        
    def globalK(self, localK : torch.Tensor, transMatrix12x12 : torch.Tensor):
        return torch.einsum('nji,njk,nkl->nil', transMatrix12x12, localK, transMatrix12x12)
    
    def globalStructureMatrix(self, numActiveNodesDof : int, numElements : int, elementsCodeNum : torch.Tensor, globalK):
        s    = torch.zeros((numActiveNodesDof, numActiveNodesDof), dtype=torch.float64)
        kIndices = torch.arange(0, 12, 1, dtype=torch.int64)
        
        for i in range(numElements):
            validIndices = elementsCodeNum[i] < numActiveNodesDof
            s[elementsCodeNum[i][validIndices][:, None], elementsCodeNum[i][validIndices]] += globalK[i][kIndices[validIndices]][:, kIndices[validIndices]]

        return s
    
