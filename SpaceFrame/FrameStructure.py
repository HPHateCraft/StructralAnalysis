
from Nodes import Nodes
from LineElements import LineElements
from Materials import Materials
from Sections import Sections
from Loads import NodalLoad, NodalLoadSolver, ElementsLoad, ElementsLoadSolver
from StiffnessMatrix import StiffnessMatrix
from TransMatrix import TransMatrix
from Solver import Solver
from PostProcess import PostProcess

class FrameStructure:
    
    def __init__(self):
        self.nodes                 = Nodes()
        self.materials             = Materials()
        self.sections              = Sections()
        self.elements              = LineElements(self.nodes, self.materials, self.sections)
        self.nodal_load            = NodalLoad(self.nodes)
        self.elements_loads        = ElementsLoad(self.elements, self.materials, self.sections)
        self._trans_matrix         = TransMatrix(self.nodes, self.elements)
        self._stiffness_matrix     = StiffnessMatrix(self.nodes, self.elements, self.materials, self.sections, self._trans_matrix)
        self._nodal_load_solver = NodalLoadSolver(self.nodes, self.nodal_load, self._trans_matrix)
        self._elements_load_solver = ElementsLoadSolver(self.nodes, self.elements, self.elements_loads, self.materials, self.sections, self._trans_matrix, self._stiffness_matrix, self._nodal_load_solver)
        self.solver                = Solver(self.nodes, self.elements, self._nodal_load_solver, self._elements_load_solver, self._trans_matrix, self._stiffness_matrix)
        self.post_process          = PostProcess(self.nodes, self.elements, self._nodal_load_solver, self._elements_load_solver, self._trans_matrix, self._stiffness_matrix, self.solver)
        
        