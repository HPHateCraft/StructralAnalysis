
from functools import cached_property
import bisect
import math 

from GlobalCoord import GlobalCoord
from Nodes import Nodes, NodesManager
from Materials import Materials, MaterialsManager
from Sections import Sections, SectionsManager

import numpy as np
from numpy.typing import NDArray

class Elements:
    
    def __init__(self, nodes: Nodes, materials: Materials, sections: Sections):
        self._nodes = nodes
        self._materials = materials
        self._sections = sections

        self._id = 0
        self.ids = []
        self.nodes_ids = []
        self.materials_names = []
        self.sections_names = []
        self.roll_angle_deg = []
        self.axes_changed = []
        self.partial_fixity_indices = []
        self.partial_fixity_vector = []
        
    def _generate_id(self):
        id_ = self._id
        self.ids.append(id_)
        self._id += 1
        return id_
   
    def find_index_by_id(self, id_: int):
        index = bisect.bisect_left(self.ids, id_)
        if index < len(self.ids) and self.ids[index] == id_:
            return index  
        else:
            raise ValueError(f"Node with id: {id_} doesn't exist.")
    
    def generate_element(self, nodei_id: int, nodej_id: int, material_name: str, section_name: str):     
           
        id_ = self._generate_id()
        self.nodes_ids.append([nodei_id, nodej_id])
        self.materials_names.append(material_name)
        self.sections_names.append(section_name)
        self.roll_angle_deg.append(0.0)
        self.axes_changed.append(False)
        return id_
    
    def change_axes(self, element_id: int, roll_angle_deg: float):
        index = self.find_index_by_id(element_id)
        self.axes_changed[index] = True
        self.roll_angle_deg[index] = roll_angle_deg
    
    def add_partial_fixity(
        self,
        id_: int,
        uxi : float | None = None,
        uyi : float | None = None,
        uzi : float | None = None,
        rxi : float | None = None,
        ryi : float | None = None,
        rzi : float | None = None,
        uxj : float | None = None,
        uyj : float | None = None,
        uzj : float | None = None,
        rxj: float | None = None,
        ryj: float | None = None,
        rzj: float | None = None
    ):
        index = self.find_index_by_id(id_)
        partial_fixity = [uxi, uyi, uzi, rxi, ryi, rzi, uxj, uyj, uzj, rxj, ryj, rzj]
        self.partial_fixity_indices.append(index)
        self.partial_fixity_vector.append(partial_fixity)

    
class ElementsManager:
    
    PENALTY_NUMBER = 1e6
    
    def __init__(self, nodes_manager: NodesManager, elements: Elements, materials_manager: MaterialsManager, sections_manager: SectionsManager):
        self._nodes_manager = nodes_manager
        self._elements = elements
        self._materials_manager = materials_manager
        self._sections_manager = sections_manager
    
    def find_index_by_id(self, id_: int):
        return np.searchsorted(self.ids, id_)
    
    @cached_property
    def ids(self):
        return np.array(self._elements.ids, dtype=np.int64)

    @cached_property
    def num_elements(self):
        return self.ids.size

    @cached_property
    def nodes_ids(self):
        return np.array(self._elements.nodes_ids, dtype=np.int64)
    
    @cached_property
    def materials_names(self):
        return np.array(self._elements.materials_names, dtype=np.str_)
    
    @cached_property
    def sections_names(self):
        return np.array(self._elements.sections_names, dtype=np.str_)
    
    @cached_property
    def roll_angle_deg(self):
        return np.array(self._elements.roll_angle_deg, dtype=np.float64)
    
    @cached_property
    def axes_changed(self):
        return np.array(self._elements.axes_changed, dtype=np.bool_)
    
    @cached_property
    def num_axes_changed(self):
        return np.count_nonzero(self.axes_changed) 
    
    @cached_property
    def partial_fixity_indices(self):
        return np.array(self._elements.partial_fixity_indices, dtype=np.int64)
    
    @cached_property    
    def num_elements_with_partial_fixity(self):
        return self.partial_fixity_indices.size
    
    @cached_property
    def partial_fixity_vector(self):
        return np.array(self._elements.partial_fixity_vector, dtype=np.float64)
    
    @cached_property
    def nodes_indices(self):
        return np.searchsorted(self._nodes_manager.ids, self.nodes_ids)

    @cached_property
    def materials_indices(self):
        sorter = np.argsort(self._materials_manager.names)
        indices_in_sorted = np.searchsorted(self._materials_manager.names[sorter], self.materials_names)
        
        return sorter[indices_in_sorted] 
    
    @cached_property
    def sections_indices(self):
        sorter = np.argsort(self._sections_manager.names)
        indices_in_sorted = np.searchsorted(self._sections_manager.names[sorter], self.sections_names)
        
        return sorter[indices_in_sorted] 
        
    @cached_property
    def coord(self):
        return self._nodes_manager.coord[self.nodes_indices]
    
    @cached_property
    def delta_coord(self):
        return self.coord[:, 1] - self.coord[:, 0]
    
    @cached_property
    def length(self):
        return np.sqrt(np.sum(self.delta_coord**2, 1))
    
    @cached_property
    def _rotation_about_x(self):
        if self.num_axes_changed > 0:
            
            cos = np.cos(np.deg2rad(self.roll_angle_deg[self.axes_changed]))
            sin = np.sin(np.deg2rad(self.roll_angle_deg[self.axes_changed]))

            rx = np.zeros((self.num_axes_changed, 3, 3), dtype=np.float64)
            rx[:, 0, 0] = 1.0
            rx[:, 1, 1] = cos
            rx[:, 1, 2] = sin
            rx[:, 2, 1] = -sin
            rx[:, 2, 2] = cos
        
            return rx  
    
    @cached_property
    def direction_cosines(self):
        unit_vector_x = self.delta_coord/self.length[:, None]
        
        is_ux_alligned_with_uY = np.abs(unit_vector_x[:, 1]) == 1.0
        
        vector_z_ = np.zeros((self.num_elements, 3), dtype=np.float64)
        
        vector_z_[~is_ux_alligned_with_uY] = np.cross(unit_vector_x[~is_ux_alligned_with_uY], GlobalCoord.UNIT_VECTOR_Y)
        vector_z_[is_ux_alligned_with_uY] = GlobalCoord.UNIT_VECTOR_Z
        
        norm_z_ = np.linalg.norm(vector_z_, axis=1)
        unit_vector_z_ = vector_z_/norm_z_[:, None]
        unit_vector_y_ = np.cross(unit_vector_z_, unit_vector_x)
        
        dc = np.stack([unit_vector_x, unit_vector_y_, unit_vector_z_], axis=1)
        if self.num_axes_changed > 0:
            dc[self.axes_changed] = self._rotation_about_x@dc[self.axes_changed]
        
        return dc

    @cached_property
    def cross_section_area(self):
        return self._sections_manager.cross_section_area[self.sections_indices]

    @cached_property
    def moment_of_inertia_about_y(self):
        return self._sections_manager.moment_of_inertia_about_y[self.sections_indices]

    @cached_property
    def moment_of_inertia_about_z(self):
        return self._sections_manager.moment_of_inertia_about_z[self.sections_indices]

    @cached_property
    def torsional_constant(self):
        return self._sections_manager.torsional_constant[self.sections_indices]

    @cached_property
    def shear_correction_factor(self):
        return self._sections_manager.shear_correction_factor[self.sections_indices]

    @cached_property
    def youngs_modulus(self):
        return self._materials_manager.youngs_modulus[self.materials_indices]

    @cached_property
    def poisson_ratio(self):
        return self._materials_manager.poisson_ratio[self.materials_indices]

    @cached_property
    def thermal_coefficient(self):
        return self._materials_manager.thermal_coefficient[self.materials_indices]
    
    @cached_property
    def shear_modulus(self):
        return self._materials_manager.shear_modulus[self.materials_indices]

    @cached_property    
    def code_number(self):
        return np.reshape(self._nodes_manager.code_number[self.nodes_indices], (self.num_elements, 12))


if __name__ == '__main__':
    nodes = Nodes()
    node1 = nodes.generate_node(0.0, 0.0, 0.0)
    node2 = nodes.generate_node(6.0, 0.0, 0.0)
    node3 = nodes.generate_node(12.0, 0.0, 0.0)
    node4 = nodes.generate_node(12.0, 3.0, 0.0)
    
    nodes_manager = NodesManager(nodes)
    
    materials = Materials()
    mat1 = materials.generate_material("c30", 33e6, 0.2)
    mat2 = materials.generate_material("d2", 33e6, 0.2)
    mat2 = materials.generate_material("a1", 33e6, 0.2)
    materials_manager = MaterialsManager(materials)
        
    sections = Sections()
    from Sections import RectSections, Isections, RectSectionsManager, ISectionsManager
    rect_sections = RectSections()
    rect1 = rect_sections.generate("b300x600", 0.6, 0.3)
    I_sections = Isections()
    I1 = I_sections.generate("I", 0.6, 0.05, 0.01, 0.3, 0.01, 0.3)
    
    rect_sections_manager = RectSectionsManager(rect_sections)
    I_sections_manager = ISectionsManager(I_sections)
    
    sections_manager = SectionsManager(rect_sections_manager, I_sections_manager)
    
    elements = Elements(nodes, materials, sections)
    elem1 = elements.generate_element(node1, node2, mat1, rect1)
    elem2 = elements.generate_element(node3, node2, mat1, I1)
    elem3 = elements.generate_element(node3, node4, mat1, I1)
    
    elements.change_axes(elem1, 45)
    
    elements_manager = ElementsManager(nodes_manager, elements, materials_manager, sections_manager)
    
    print(elements_manager.code_number)
    