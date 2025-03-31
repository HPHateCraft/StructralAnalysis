import torch

class Solver:
    
    def __init__(self):
        pass

    def solveGlobalNodalDisplacement(self, globalStructureMatrix: torch.Tensor, globalNodalVector: torch.Tensor):
        return torch.linalg.solve(globalStructureMatrix, globalNodalVector)
    
    def nodesGlobalDisplacment(self, numActiveNodes: torch.Tensor, numActiveNodesDof: int, activeNodesCodeNum: torch.Tensor, globalNodalDisplacement: torch.Tensor) -> torch.Tensor:
        v = torch.zeros((numActiveNodes, 6), dtype=torch.float64)
        condition = activeNodesCodeNum < numActiveNodesDof
        v[condition] = globalNodalDisplacement[activeNodesCodeNum[condition]]
        return v
    
    def elementsNodesGlobalDisplacment(self, nodesGlobalDisplacment: torch.Tensor, elementsNodesIndices: torch.Tensor):
        return nodesGlobalDisplacment[elementsNodesIndices].flatten(1)

    def elementsNodesLocalDisplacment(self, elementsNodesGlobalDisplacment: torch.Tensor, transMatrix12x12: torch.Tensor):
        return torch.einsum('nij,nj->ni', transMatrix12x12, elementsNodesGlobalDisplacment)

    def elementsNodesLocalReaction(self, localK: torch.Tensor, elementsNodesLocalDisplacment: torch.Tensor, elementLocalNodalVector: torch.Tensor):
        return torch.einsum('nij,nj->ni', localK, elementsNodesLocalDisplacment) - elementLocalNodalVector

    def elementsNodesGlobalReaction(self, globalK: torch.Tensor, elementsNodesGlobalDisplacment: torch.Tensor, elementGlobalNodalVector: torch.Tensor):
        return torch.einsum('nij,nj->ni', globalK, elementsNodesGlobalDisplacment) - elementGlobalNodalVector
