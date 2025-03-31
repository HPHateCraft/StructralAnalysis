import torch    

class TransMatrix:
    
    def __init__(self):
        pass
    
    def transMatrix3x3(self, numElements : int, elementsProperties : torch.Tensor, deltaNodes : torch.Tensor):
        r = torch.zeros((numElements, 3, 3), dtype=torch.float64)
        
        L   = elementsProperties[:, 0]
        psi = elementsProperties[:, 11]
        
        rxX = deltaNodes[:, 0]/L
        rxY = deltaNodes[:, 1]/L
        rxZ = deltaNodes[:, 2]/L
        
        r[:, 0, 0] = rxX
        r[:, 0, 1] = rxY
        r[:, 0, 2] = rxZ
        
        # elements local x-axis is aligned with the global Y-axis
        isAligned  = (rxX == 0) & (rxZ == 0) 
        
        rxXY   = rxX[~isAligned]*rxY[~isAligned]
        rxYZ   = rxY[~isAligned]*rxZ[~isAligned]
        rxXZ_2 = torch.sqrt(rxX[~isAligned]**2 + rxZ[~isAligned]**2)
        
        psi        = torch.deg2rad(psi)
        cosPsi     = torch.cos(psi)
        sinPsi     = torch.sin(psi)

        r[~isAligned, 1, 0] = (-rxXY*cosPsi[~isAligned] - rxZ[~isAligned]*sinPsi[~isAligned])/rxXZ_2
        r[~isAligned, 1, 1] = rxXZ_2*cosPsi[~isAligned]
        r[~isAligned, 1, 2] = (-rxYZ*cosPsi[~isAligned] + rxX[~isAligned]*sinPsi[~isAligned])/rxXZ_2
        
        r[~isAligned, 2, 0] = (rxXY*sinPsi[~isAligned] - rxZ[~isAligned]*cosPsi[~isAligned])/rxXZ_2
        r[~isAligned, 2, 1] = -rxXZ_2*sinPsi[~isAligned]
        r[~isAligned, 2, 2] = (rxYZ*sinPsi[~isAligned] + rxX[~isAligned]*cosPsi[~isAligned])/rxXZ_2
        
        r[isAligned, 1, 0] = -rxY[isAligned]*cosPsi[isAligned]
        r[isAligned, 1, 2] = sinPsi[isAligned]

        r[isAligned, 2, 0] = rxY[isAligned]*sinPsi[isAligned]
        r[isAligned, 2, 2] = cosPsi[isAligned]
        
        return r

    def transMatrix12x12(self, numElements : int, transMatrix3x3 : torch.Tensor):
        T = torch.zeros((numElements, 12, 12), dtype=torch.float64)
    
        T[:, 0 : 3, 0 : 3]   = transMatrix3x3
        T[:, 3 : 6, 3 : 6]   = transMatrix3x3
        T[:, 6 : 9, 6 : 9]   = transMatrix3x3
        T[:, 9 : 12, 9 : 12] = transMatrix3x3
        
        return T
    