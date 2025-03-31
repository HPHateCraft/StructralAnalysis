import torch

class Loads:
    
    def __init__(self):
        self._distForce     = []
        self._distForceID   = []
        self._distForceAxis = []
        
        self._distMoment = []
        self._distMomentID = []
        self._distMomentAxis = []
        
        self._concenForce = []
        self._concenForceID = []
        self._concenForceAxis = []
        
        self._concenMoment = []
        self._concenMomentID = []
        self._concenMomentAxis = []
    
    def addDistForce(self, elementID : int, mag1 : float, mag2 : float, relDist1 : float, relDist2 : float, axis : int):
        self._distForce.append([mag1, mag2, relDist1, relDist2])
        self._distForceID.append(elementID)
        self._distForceAxis.append(axis)
        
    def getDistForce(self):
        return torch.tensor(self._distForce, dtype=torch.float64)

    def getDistForceID(self):
        return torch.tensor(self._distForceID, dtype=torch.int64)

    def getDistForceAxis(self):
        return torch.tensor(self._distForceAxis, dtype=torch.int64)
    
    def addDistMoment(self, elementID : int, mag1 : float, mag2 : float, relDist1 : float, relDist2 : float, axis : int):
        self._distMoment.append([mag1, mag2, relDist1, relDist2])
        self._distMomentID.append(elementID)
        self._distMomentAxis.append(axis)
        
    def getDistMoment(self):
        return torch.tensor(self._distMoment, dtype=torch.float64)

    def getDistMomentID(self):
        return torch.tensor(self._distMomentID, dtype=torch.int64)

    def getDistMomentAxis(self):
        return torch.tensor(self._distMomentAxis, dtype=torch.int64)

    def addConcenForce(self, elementID : int, mag : float, relDist : float, axis : int):
        self._concenForce.append([mag, relDist])
        self._concenForceID.append(elementID)
        self._concenForceAxis.append(axis)
        
    def getConcenForce(self):
        return torch.tensor(self._concenForce, dtype=torch.float64)

    def getConcenForceID(self):
        return torch.tensor(self._concenForceID, dtype=torch.int64)

    def getConcenForceAxis(self):
        return torch.tensor(self._concenForceAxis, dtype=torch.int64)

    def addConcenMoment(self, elementID : int, mag : float, relDist : float, axis : int):
        self._concenMoment.append([mag, relDist])
        self._concenMomentID.append(elementID)
        self._concenMomentAxis.append(axis)
        
    def getConcenMoment(self):
        return torch.tensor(self._concenMoment, dtype=torch.float64)

    def getConcenMomentID(self):
        return torch.tensor(self._concenMomentID, dtype=torch.int64)

    def getConcenMomentAxis(self):
        return torch.tensor(self._concenMomentAxis, dtype=torch.int64)
        
    def distForceLocalNodalVector(self, numElements : int, elementsProperties : torch.Tensor, distForce : torch.Tensor, distForceID : torch.Tensor, distForceAxis : torch.Tensor, rotationMatrix : torch.Tensor):
        F = torch.zeros((numElements, 12), dtype=torch.float64)
        r = rotationMatrix[distForceID]
        n  = distForce.shape[0]
        Q = torch.zeros((n, 12), dtype=torch.float64)

        w1 = torch.zeros((n, 3), dtype=torch.float64)
        w2 = torch.zeros((n, 3), dtype=torch.float64)
        
        elemProp = elementsProperties[distForceID]
        L = elemProp[:, 0]
        A = elemProp[:, 3]
        Izz = elemProp[:, 4]
        Iyy = elemProp[:, 5]
        E = elemProp[:, 7]
        G = elemProp[:, 8]
        kappa = elemProp[:, 10]
        
        L2 = L*L
        L3 = L2*L
        
        l1 = distForce[:, 2]*L
        l12 = l1*l1
        l13 = l12*l1
        l14 = l12*l12
        l15 = l14*l1
        
        l2 = distForce[:, 3]*L
        l22 = l2*l2
        l23 = l22*l2
        l24 = l22*l22
        l25 = l24*l2
        
        isGlobalAxis = distForceAxis <= 3
        
        w1[torch.arange(n), (distForceAxis - 1)%3] = distForce[:, 0]
        w2[torch.arange(n), (distForceAxis - 1)%3] = distForce[:, 1]
        
        w1[isGlobalAxis] = torch.einsum('ijk,ik->ij', r[isGlobalAxis], w1[isGlobalAxis])
        w2[isGlobalAxis] = torch.einsum('ijk,ik->ij', r[isGlobalAxis], w2[isGlobalAxis])
        
        wx1 = w1[:, 0]
        wy1 = w1[:, 1]
        wz1 = w1[:, 2]
        
        wx2 = w2[:, 0]
        wy2 = w2[:, 1]
        wz2 = w2[:, 2]
        
        # calculate nodal forces in x-y plane
        Q[:, 0] = -(l1 - l2)*(wx1*(3*L - 2*l1 - l2) + wx2*(3*L - l1 - 2*l2))/(6*L)
        Q[:, 6] = -Q[:, 0] + (wx1 + wx2)*(l2 - l1)/2
        
        # calculate nodal forces in x-y plane
        kappaGA = kappa*G*A
        EIzz = E*Izz
        ay = (wy2 - wy1)/(l2 - l1)
        by = wy1 - ay*l1
        
        Q[:, 1] = -(
            (ay*(kappaGA*(10*L3*l12 - 10*L3*l22 - 15*L*l14 + 15*L*l24 + 8*l15 - 8*l25)
                + EIzz*(120*L*l12 - 120*L*l22 - 80*l13 + 80*l23)) 
            + by*(kappaGA*(20*L3*l1 - 20*L3*l2 - 20*L*l13 + 20*L*l23 + 10*l14 - 10*l24) 
                + EIzz*(240*L*l1 - 240*L*l2 - 120*l12 + 120*l22)))
            /(20*kappaGA*L3 + 240*EIzz*L)
        )
        
        Q[:, 7] = -Q[:, 1] + (wy1 + wy2)*(l2 - l1)/2
        
        Q[:, 5] = -(
            (ay*(kappaGA*(20*L3*l13 - 20*L3*l23 - 30*L2*l14 + 30*L2*l24 + 12*L*l15 - 12*L*l25)
                + EIzz*(120*L*l13 - 120*L*l23 - 90*l14 + 90*l24))
            + by*(kappaGA*(30*L3*l12 - 30*L3*l22 - 40*L2*l13 + 40*L2*l23 + 15*L*l14 - 15*L*l24)
                + EIzz*(180*L*l12 - 180*L*l22 - 120*l13 + 120*l23)))
            /(60*A*G*L3*kappa + 720*EIzz*L)
        )
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + (l2 - l1)*(wy1*(2*l1 + l2) + wy2*(l1 + 2*l2))/6
        
        # calculate nodal forces in x-z plane
        EIyy = E*Iyy
        az = (wz2 - wz1)/(l2 - l1)
        bz = wz1 - az*l1
        
        Q[:, 2] = -(
            (az*(kappaGA*(10*L3*l12 - 10*L3*l22 - 15*L*l14 + 15*L*l24 + 8*l15 - 8*l25)
                + EIyy*(120*L*l12 - 120*L*l22 - 80*l13 + 80*l23)) 
            + bz*(kappaGA*(20*L3*l1 - 20*L3*l2 - 20*L*l13 + 20*L*l23 + 10*l14 - 10*l24) 
                + EIyy*(240*L*l1 - 240*L*l2 - 120*l12 + 120*l22)))
            /(20*kappaGA*L3 + 240*EIyy*L)
        )
        
        Q[:, 8] = -Q[:, 2] + (wz1 + wz2)*(l2 - l1)/2
        
        Q[:, 4] = (
            (az*(kappaGA*(20*L3*l13 - 20*L3*l23 - 30*L2*l14 + 30*L2*l24 + 12*L*l15 - 12*L*l25)
                + EIyy*(120*L*l13 - 120*L*l23 - 90*l14 + 90*l24))
            + bz*(kappaGA*(30*L3*l12 - 30*L3*l22 - 40*L2*l13 + 40*L2*l23 + 15*L*l14 - 15*L*l24)
                + EIyy*(180*L*l12 - 180*L*l22 - 120*l13 + 120*l23)))
            /(60*A*G*L3*kappa + 720*EIyy*L)
        )
        Q[:, 10] = -Q[:, 4] + Q[:, 8]*L - (l2 - l1)*(wz1*(2*l1 + l2) + wz2*(l1 + 2*l2))/6
        
        return F.index_add_(0, distForceID, Q)

    def distMomentLocalNodalVector(self, numElements : int, elementsProperty : torch.Tensor, distMoment : torch.Tensor, distMomentID : torch.Tensor, distMomentAxis : torch.Tensor, rotationMatrix : torch.Tensor):
        F = torch.zeros((numElements, 12), dtype=torch.float64)
        r = rotationMatrix[distMomentID]
        n  = distMoment.shape[0]
        Q = torch.zeros((n, 12), dtype=torch.float64)

        w1 = torch.zeros((n, 3), dtype=torch.float64)
        w2 = torch.zeros((n, 3), dtype=torch.float64)
        
        elemProp = elementsProperty[distMomentID]
        L = elemProp[:, 0]
        A = elemProp[:, 3]
        Izz = elemProp[:, 4]
        Iyy = elemProp[:, 5]
        E = elemProp[:, 7]
        G = elemProp[:, 8]
        kappa = elemProp[:, 10]
        
        L2 = L*L
        L3 = L2*L
        
        l1 = distMoment[:, 2]*L
        l12 = l1*l1
        l13 = l12*l1

        l2 = distMoment[:, 3]*L
        l22 = l2*l2
        l23 = l22*l2
        
        isGlobalAxis = distMomentAxis <= 3
        
        w1[torch.arange(n), (distMomentAxis - 1)%3] = distMoment[:, 0]
        w2[torch.arange(n), (distMomentAxis - 1)%3] = distMoment[:, 1]
        
        w1[isGlobalAxis] = torch.einsum('ijk,ik->ij', r[isGlobalAxis], w1[isGlobalAxis])
        w2[isGlobalAxis] = torch.einsum('ijk,ik->ij', r[isGlobalAxis], w2[isGlobalAxis])
        
        wx1 = w1[:, 0]
        wy1 = w1[:, 1]
        wz1 = w1[:, 2]
        
        wx2 = w2[:, 0]
        wy2 = w2[:, 1]
        wz2 = w2[:, 2]
        
        Q[:, 3] = -(l1 - l2)*(wx1*(3*L - 2*l1 - l2) + wx2*(3*L - l1 - 2*l2))/(6*L)
        Q[:, 9] = -Q[:, 3] + (wx1 + wx2)*(l2 - l1)/2
        
        # calculate nodal forces in x-y plane
        kappaGA = kappa*G*A
        EIzz = E*Izz
        az = (wz2 - wz1)/(l2 - l1)
        bz = wz1 - az*l1
        
        Q[:, 1] = -(
            kappaGA*(-l1 + l2)*
            (az*(4*L*l12 + 4*L*l1*l2 + 4*L*l22 - 3*l13 - 3*l12*l2 - 3*l1*l22 - 3*l23)
            + bz*(6*L*l1 + 6*L*l2 - 4*l12 - 4*l1*l2 - 4*l22))
            /(2*L*(kappaGA*L2 + 12*EIzz))
        )
        
        Q[:, 7] = -Q[:, 1]
        
        Q[:, 5] = -(
            (l1 - l2)*
            (az*(kappaGA*(6*L3*l1 + 6*L3*l2 - 16*L2*l12 - 16*L2*l1*l2 - 16*L2*l22 + 9*L*l13 + 9*L*l12*l2 + 9*L*l1*l22 + 9*L*l23) 
                + EIzz*(72*L*l1 + 72*L*l2 - 48*l12 - 48*l1*l2 - 48*l22)) 
            + 12*bz*(kappaGA*(L3 - 2*L2*l1 - 2*L2*l2 + L*l12 + L*l1*l2 + L*l22) 
                    + EIzz*(12*L - 6*l1 - 6*l2)))
            /(12*L*(kappaGA*L2 + 12*EIzz))
        )
        
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + (wz1 + wz2)*(l2 - l1)/2
        
        # # calculate nodal forces in x-z plane
        EIyy = E*Iyy
        ay = (wy2 - wy1)/(l2 - l1)
        by = wy1 - ay*l1
        
        Q[:, 2] = (
            kappaGA*(-l1 + l2)*
            (ay*(4*L*l12 + 4*L*l1*l2 + 4*L*l22 - 3*l13 - 3*l12*l2 - 3*l1*l22 - 3*l23)
            + by*(6*L*l1 + 6*L*l2 - 4*l12 - 4*l1*l2 - 4*l22))
            /(2*L*(kappaGA*L2 + 12*EIyy))
        )
        
        Q[:, 8] = -Q[:, 2]
        
        Q[:, 4] = (
            (l1 - l2)*
            (ay*(kappaGA*(6*L3*l1 + 6*L3*l2 - 16*L2*l12 - 16*L2*l1*l2 - 16*L2*l22 + 9*L*l13 + 9*L*l12*l2 + 9*L*l1*l22 + 9*L*l23) 
                + EIyy*(72*L*l1 + 72*L*l2 - 48*l12 - 48*l1*l2 - 48*l22)) 
            + 12*by*(kappaGA*(L3 - 2*L2*l1 - 2*L2*l2 + L*l12 + L*l1*l2 + L*l22) 
                    + EIyy*(12*L - 6*l1 - 6*l2)))
            /(12*L*(kappaGA*L2 + 12*EIyy))
        )
        
        Q[:, 10] = -Q[:, 4] - Q[:, 8]*L - (wy1 + wy2)*(l2 - l1)/2
        
        Q[:, 4] = -Q[:, 4]
        Q[:, 10] = -Q[:, 10]
        
        return F.index_add_(0, distMomentID, Q)

    def concenForceLocalNodalVector(self, numElements : int, elementsProperty : torch.Tensor, concenForce : torch.Tensor, concenForceID : torch.Tensor, concenForceAxis : torch.Tensor, rotationMatrix : torch.Tensor):
        F = torch.zeros((numElements, 12), dtype=torch.float64)
        r = rotationMatrix[concenForceID]
        n  = concenForce.shape[0]
        Q = torch.zeros((n, 12), dtype=torch.float64)

        w = torch.zeros((n, 3), dtype=torch.float64)
        
        elemProp = elementsProperty[concenForceID]
        L = elemProp[:, 0]
        A = elemProp[:, 3]
        Izz = elemProp[:, 4]
        Iyy = elemProp[:, 5]
        E = elemProp[:, 7]
        G = elemProp[:, 8]
        kappa = elemProp[:, 10]
        
        l = concenForce[:, 1]*L
        
        isGlobalAxis = concenForceAxis <= 3
        
        w[torch.arange(n), (concenForceAxis - 1)%3] = concenForce[:, 0]
        
        w[isGlobalAxis] = torch.einsum('ijk,ik->ij', r[isGlobalAxis], w[isGlobalAxis])
        
        wx = w[:, 0]
        wy = w[:, 1]
        wz = w[:, 2]
        
        Q[:, 0] = -wx*(l - L)/L
        Q[:, 6] = -Q[:, 0] + wx
        
        # calculate nodal forces in x-y plane
        kappaGA = kappa*G*A
        EIzz = E*Izz
        
        Q[:, 1] = -wy*(-L + l)*(kappaGA*(L**2 + L*l - 2*l**2) + 12*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
        
        Q[:, 7] = -Q[:, 1] + wy
        
        Q[:, 5] = -l*wy*(-L + l)*(kappaGA*(L**2 - L*l) + 6*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
        
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + wy*l
        
        # calculate nodal forces in x-z plane
        EIyy = E*Iyy

        Q[:, 2] = -wz*(-L + l)*(kappaGA*(L**2 + L*l - 2*l**2) + 12*EIyy)/(L*(kappaGA*L**2 + 12*EIyy))
        
        Q[:, 8] = -Q[:, 2] + wz
        
        Q[:, 4] = -l*wz*(-L + l)*(kappaGA*(L**2 - L*l) + 6*EIyy)/(L*(kappaGA*L**2 + 12*EIyy))
        
        Q[:, 10] = -Q[:, 4] - Q[:, 8]*L + wz*l
        
        return F.index_add_(0, concenForceID, Q)

    def concenMomentLocalNodalVector(self, numElements : int, elementsProperty : torch.Tensor, concenMoment : torch.Tensor, concenMomentID : torch.Tensor, concenMomentAxis : torch.Tensor, rotationMatrix : torch.Tensor):
        F = torch.zeros((numElements, 12), dtype=torch.float64)
        r = rotationMatrix[concenMomentID]
        n  = concenMoment.shape[0]
        Q = torch.zeros((n, 12), dtype=torch.float64)

        w = torch.zeros((n, 3), dtype=torch.float64)
        
        elemProp = elementsProperty[concenMomentID]
        L = elemProp[:, 0]
        A = elemProp[:, 3]
        Izz = elemProp[:, 4]
        Iyy = elemProp[:, 5]
        E = elemProp[:, 7]
        G = elemProp[:, 8]
        kappa = elemProp[:, 10]
        
        l = concenMoment[:, 1]*L
        
        isGlobalAxis = concenMomentAxis <= 3
        
        w[torch.arange(n), (concenMomentAxis - 1)%3] = concenMoment[:, 0]
        
        w[isGlobalAxis] = torch.einsum('ijk,ik->ij', r[isGlobalAxis], w[isGlobalAxis])
        
        wx = w[:, 0]
        wy = w[:, 1]
        wz = w[:, 2]
        
        Q[:, 3] = -wx*(l - L)/L
        Q[:, 9] = -Q[:, 3] + wx
        
        # calculate nodal forces in x-y plane
        kappaGA = kappa*G*A
        EIzz = E*Izz
        
        Q[:, 1] = -6*kappaGA*l*wz*(L - l)/(L*(kappaGA*L**2 + 12*EIzz))
        
        Q[:, 7] = -Q[:, 1]
        
        Q[:, 5] = -wz*(-L + l)*(kappaGA*(L**2 - 3*L*l) + 12*EIzz)/(L*(kappaGA*L**2 + 12*EIzz))
        
        Q[:, 11] = -Q[:, 5] - Q[:, 7]*L + wz
        
        # calculate nodal forces in x-z plane
        EIyy = E*Iyy

        Q[:, 2] = 6*kappaGA*l*wy*(L - l)/(L*(kappaGA*L**2 + 12*EIyy))
        
        Q[:, 8] = -Q[:, 2]
        
        Q[:, 4] = wy*(-L + l)*(kappaGA*(L**2 - 3*L*l) + 12*EIyy)/(L*(kappaGA*L**2 + 12*EIyy))
        
        Q[:, 10] = -Q[:, 4] - Q[:, 8]*L - wy
        
        return F.index_add_(0, concenMomentID, Q)

    def elementLocalNodalVector(self, distForceVec : torch.Tensor, distMomentVec : torch.Tensor, concenForceVec : torch.Tensor, concenMomentVec : torch.Tensor):
        return distForceVec + distMomentVec + concenForceVec + concenMomentVec

    def elementGlobalNodalVector(self, elementLocalNodalVec : torch.Tensor, transMatrix : torch.Tensor):
        return torch.einsum('nji,nj->ni', transMatrix, elementLocalNodalVec)

    def globalNodalVector(self, numActiveNodesDof : int, elementsCodeNum : torch.Tensor, elementGlobalNodalVec : torch.Tensor):
        F = torch.zeros(numActiveNodesDof, dtype=torch.float64)
        elementCodeNumFlat = elementsCodeNum.flatten()
        elementGlobalNodalVecFlat = elementGlobalNodalVec.flatten()
        activeIndices = elementCodeNumFlat < numActiveNodesDof
        F.index_add_(0, elementCodeNumFlat[activeIndices], elementGlobalNodalVecFlat[activeIndices])
        return F
