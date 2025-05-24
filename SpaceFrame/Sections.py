from functools import cached_property

import numpy as np
from numpy.typing import NDArray

class Sections:
    
    RECT_SECTION = 0
    I_SECTION = 1
    names = []
    type_ = []


class RectSections:
    
    def __init__(self):
        self.height = []
        self.width = []
        
    def generate(self, name: str, height: float, width: float):
        Sections.type_.append(Sections.RECT_SECTION)
        Sections.names.append(name)
        self.height.append(height)
        self.width.append(width)
        
        return name


class Isections:
    
    def __init__(self):
        self.height = []
        self.web_width = []
        self.top_flange_height = []
        self.top_flange_width = []
        self.bot_flange_height = []
        self.bot_flange_width = []

    def generate(self, name: str, height: float, web_width: float, top_flange_height: float, top_flange_width: float, bot_flange_height: float, bot_flange_width: float):
        Sections.type_.append(Sections.I_SECTION)
        Sections.names.append(name)
        self.height.append(height)
        self.web_width.append(web_width)
        self.top_flange_height.append(top_flange_height)
        self.top_flange_width.append(top_flange_width)
        self.bot_flange_height.append(bot_flange_height)
        self.bot_flange_width.append(bot_flange_width)
        
        return name


class RectSectionsManager:
    
    def __init__(self, rect_sections: RectSections):
        self._rect_sections = rect_sections
        
    @cached_property
    def height(self):
        return np.array(self._rect_sections.height, dtype=np.float64)

    @cached_property
    def width(self):
        return np.array(self._rect_sections.width, dtype=np.float64)
    
    @cached_property
    def num_sections(self):
        return self.height.size
    
    @cached_property
    def cross_section_area(self):
        return self.height*self.width

    @cached_property
    def moment_of_inertia_about_y(self):
        return self.height*self.width**3/12

    @cached_property
    def moment_of_inertia_about_z(self):
        return self.height**3*self.width/12

    @cached_property
    def torsional_constant(self):
        max_dim = np.maximum(self.height, self.width)
        min_dim = np.minimum(self.height, self.width)
        beta = 1/3 - 0.21*min_dim*(1 - (min_dim/max_dim)**4/12)/max_dim
      
        return beta*max_dim*min_dim**3
    
    @cached_property
    def shear_correction_factor(self):
        return np.full(self.num_sections, 5/6, dtype=np.float64)


class ISectionsManager:
    
    def __init__(self, I_sections: Isections):
        self._I_sections = I_sections
    
    @cached_property
    def height(self):
        return np.array(self._I_sections.height, dtype=np.float64)

    @cached_property
    def web_width(self):
        return np.array(self._I_sections.web_width, dtype=np.float64)

    @cached_property
    def top_flange_height(self):
        return np.array(self._I_sections.top_flange_height, dtype=np.float64)

    @cached_property
    def top_flange_width(self):
        return np.array(self._I_sections.top_flange_width, dtype=np.float64)

    @cached_property
    def bot_flange_height(self):
        return np.array(self._I_sections.bot_flange_height, dtype=np.float64)

    @cached_property
    def bot_flange_width(self):
        return np.array(self._I_sections.bot_flange_width, dtype=np.float64)

    @cached_property
    def num_sections(self):
        return self.height.size
    
    @cached_property
    def web_height(self):
        return self.height - (self.top_flange_height + self.bot_flange_height)
    
    @cached_property
    def top_flange_cross_section_area(self):
        return self.top_flange_height*self.top_flange_width

    @cached_property
    def web_cross_section_area(self):
        return self.web_height*self.web_width

    @cached_property
    def bot_flange_cross_section_area(self):
        return self.bot_flange_height*self.bot_flange_width

    @cached_property
    def cross_section_area(self):
        return self.top_flange_cross_section_area + self.web_cross_section_area + self.bot_flange_cross_section_area

    @cached_property
    def moment_of_inertia_about_y(self):
        Iyy_top_flange = self.top_flange_height*self.top_flange_width**3/12
        Iyy_web = self.web_height*self.web_width**3/12
        Iyy_bot_flange = self.bot_flange_height*self.bot_flange_width**3/12
        
        return Iyy_top_flange + Iyy_web + Iyy_bot_flange

    @cached_property
    def top_flange_center_of_gravity_y(self):
        return self.height - self.top_flange_height/2
    
    @cached_property
    def web_center_of_gravity_y(self):
        return self.height - self.top_flange_height - self.web_height/2
    
    @cached_property
    def bot_flange_center_of_gravity_y(self):
        return self.bot_flange_height/2
    
    @cached_property
    def center_of_gravity_y(self):
        return (
            (self.top_flange_cross_section_area*self.top_flange_center_of_gravity_y +
             self.web_cross_section_area*self.web_center_of_gravity_y +
             self.bot_flange_cross_section_area*self.bot_flange_center_of_gravity_y) /
             self.cross_section_area 
        )

    @cached_property
    def moment_of_inertia_about_z(self):
        Izz_top_flange = self.top_flange_height**3*self.top_flange_width/12
        Izz_web = self.web_height**3*self.web_width/12
        Izz_bot_flange = self.bot_flange_height**3*self.bot_flange_width/12
        
        Izz_top_flange += self.top_flange_cross_section_area*(self.center_of_gravity_y - self.top_flange_center_of_gravity_y)**2
        Izz_web += self.web_cross_section_area*(self.center_of_gravity_y - self.web_center_of_gravity_y)**2
        Izz_bot_flange += self.bot_flange_cross_section_area*(self.center_of_gravity_y - self.bot_flange_center_of_gravity_y)**2
        
        return Izz_top_flange + Izz_web + Izz_bot_flange
    
    def torsional_constant_of_rect(self, height: NDArray, width: NDArray):
        max_dim = np.maximum(height, width)
        min_dim = np.minimum(height, width)
        beta = 1/3 - 0.21*min_dim*(1 - (min_dim/max_dim)**4/12)/max_dim
      
        return beta*max_dim*min_dim**3
    
    @cached_property
    def torsional_constant(self):
        top_flange_torsional_constant = self.torsional_constant_of_rect(self.top_flange_height, self.top_flange_width)
        web_torsional_constant = self.torsional_constant_of_rect(self.web_height, self.web_width)
        bot_flange_torsional_constant = self.torsional_constant_of_rect(self.bot_flange_height, self.bot_flange_width)
        
        return top_flange_torsional_constant + web_torsional_constant + bot_flange_torsional_constant
    
    @cached_property
    def shear_correction_factor(self):
        return self.web_cross_section_area/self.cross_section_area

    
class SectionsManager:
    
    def __init__(self, rect_sections_manager: RectSectionsManager, I_sections_manager: ISectionsManager):
        self._rect_sections_manager = rect_sections_manager
        self._I_sections_manager = I_sections_manager
    
    @cached_property
    def names(self):
        return np.array(Sections.names, dtype=np.str_)
    
    @cached_property
    def type_(self):
        return np.array(Sections.type_, dtype=np.int64)
    
    @cached_property
    def num_sections(self):
        return self.names.size  
    
    @cached_property
    def is_rect_section(self):
        return self.type_ == Sections.RECT_SECTION
    
    @cached_property
    def is_I_section(self):
        return self.type_ == Sections.I_SECTION
    
    @cached_property
    def cross_section_area(self):
        A = np.zeros(self.num_sections, dtype=np.float64)
        A[self.is_rect_section] = self._rect_sections_manager.cross_section_area
        A[self.is_I_section] = self._I_sections_manager.cross_section_area
        return A

    @cached_property
    def moment_of_inertia_about_y(self):
        Iyy = np.zeros(self.num_sections, dtype=np.float64)
        Iyy[self.is_rect_section] = self._rect_sections_manager.moment_of_inertia_about_y
        Iyy[self.is_I_section] = self._I_sections_manager.moment_of_inertia_about_y
        return Iyy

    @cached_property
    def moment_of_inertia_about_z(self):
        Izz = np.zeros(self.num_sections, dtype=np.float64)
        Izz[self.is_rect_section] = self._rect_sections_manager.moment_of_inertia_about_z
        Izz[self.is_I_section] = self._I_sections_manager.moment_of_inertia_about_z
        return Izz

    @cached_property
    def torsional_constant(self):
        J = np.zeros(self.num_sections, dtype=np.float64)
        J[self.is_rect_section] = self._rect_sections_manager.torsional_constant
        J[self.is_I_section] = self._I_sections_manager.torsional_constant
        return J

    @cached_property
    def shear_correction_factor(self):
        kappa = np.zeros(self.num_sections, dtype=np.float64)
        kappa[self.is_rect_section] = self._rect_sections_manager.shear_correction_factor
        kappa[self.is_I_section] = self._I_sections_manager.shear_correction_factor
        return kappa
    
    
    
if __name__ == '__main__':
    
    rect_sections = RectSections()
    sec1 = rect_sections.generate("B200x600", 0.6, 0.2)
    sec2 = rect_sections.generate("B300x600", 0.6, 0.3)
    
    rect_sections_manager = RectSectionsManager(rect_sections)
    
    I_sections = Isections()
    sec3 = I_sections.generate("I1", 0.6, 0.05, 0.01, 0.3, 0.01, 0.3)
    sec4 = I_sections.generate("I2", 0.6, 0.05, 0.01, 0.3, 0.05, 0.15)
    
    I_sections_manager = ISectionsManager(I_sections)
    sec5 = rect_sections.generate("sad", 1, 1)
    
    sections_manager = SectionsManager(rect_sections_manager, I_sections_manager)
    
    print(sections_manager.torsional_constant)
    
    