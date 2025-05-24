
from Nodes import Nodes, NodesManager
from LineElements import Elements, ElementsManager
from Materials import Materials, MaterialsManager
from Sections import Sections, SectionsManager, RectSections, RectSectionsManager, Isections, ISectionsManager
from Loads import NodalLoad, NodalLoadManager, DistributedForce, DistributedForceManager, DistributedMoment, DistributedMomentManager, ConcentratedForce, ConcentratedForceManager, ConcentratedMoment, ConcentratedMomentManager, Temperature, TemperatureManager, ElementsLoadManager
from StiffnessMatrix import StiffnessMatrix
from TransMatrix import TransMatrix
from Solver import Solver
from PostProcess import PostProcess

class FrameStructure:
    
    def __init__(self):
        self.nodes          = Nodes()
        self._nodes_manager = NodesManager(self.nodes)
        
        
        self.materials          = Materials()
        self._materials_manager = MaterialsManager(self.materials)
        
        self._sections              = Sections()
        self.rect_sections          = RectSections()
        self.I_sections             = Isections()
        self._rect_sections_manager = RectSectionsManager(self.rect_sections)
        self._I_sections_manager    = ISectionsManager(self.I_sections)
        self._sections_manager      = SectionsManager(self._rect_sections_manager, self._I_sections_manager)
        
        self.elements          = Elements(self.nodes, self.materials, self._sections)
        self._elements_manager = ElementsManager(self._nodes_manager, self.elements, self._materials_manager, self._sections_manager)
        
        self._trans_matrix     = TransMatrix(self._nodes_manager, self._elements_manager)
        self._stiffness_matrix = StiffnessMatrix(self._nodes_manager, self._elements_manager, self._trans_matrix)
        
        
        self.nodal_load          = NodalLoad(self._nodes_manager)
        self._nodal_load_manager = NodalLoadManager(self._nodes_manager, self.nodal_load, self._trans_matrix)
        
        
        self.distributed_force   = DistributedForce(self._elements_manager)
        self.distributed_moment  = DistributedMoment(self._elements_manager)
        self.concentrated_force  = ConcentratedForce(self._elements_manager)
        self.concentrated_moment = ConcentratedMoment(self._elements_manager)
        self.temperature         = Temperature(self._elements_manager)
        
        self._distributed_force_manager   = DistributedForceManager(self._elements_manager, self.distributed_force, self._trans_matrix)
        self._distributed_moment_manager  = DistributedMomentManager(self._elements_manager, self.distributed_moment, self._trans_matrix)
        self._concentrated_force_manager  = ConcentratedForceManager(self._elements_manager, self.concentrated_force, self._trans_matrix)
        self._concentrated_moment_manager = ConcentratedMomentManager(self._elements_manager, self.concentrated_moment, self._trans_matrix)
        self._temperature_manager         = TemperatureManager(self._elements_manager, self.temperature)
        self._elements_load_manager       = ElementsLoadManager(self._nodes_manager, self._elements_manager, self._nodal_load_manager, self._distributed_force_manager, self._distributed_moment_manager, self._concentrated_force_manager, self._concentrated_moment_manager, self._temperature_manager, self._trans_matrix, self._stiffness_matrix)
        
        self.solver       = Solver(self._nodes_manager, self._nodal_load_manager, self._elements_load_manager, self._stiffness_matrix)
        
        self.post_process = PostProcess(self._nodes_manager, self._elements_manager, self._nodal_load_manager, self._elements_load_manager, self._trans_matrix, self._stiffness_matrix, self.solver)
        
        