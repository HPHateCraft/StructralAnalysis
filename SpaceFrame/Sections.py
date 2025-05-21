from functools import cached_property

import numpy as np
from numpy.typing import NDArray

class Sections:
    RECT_SECTION = 0
    I_SECTION = 1
    
    RECT_HEIGHT = 0
    RECT_WIDTH  = 1
    
    I_WEB_HEIGHT        = 0
    I_WEB_WIDTH         = 1
    I_TOP_FLANGE_HEIGHT = 2
    I_TOP_FLANGE_WIDTH  = 3
    I_BOT_FLANGE_HEIGHT = 4
    I_BOT_FLANGE_WIDTH  = 5
    
    def __init__(self):
        self.names     = []
        self.type_     = []
        self.rect_data = []
        self.I_data    = []
        
    def generate_rect_section(self, name: str, height: float, width: float):
        self.type_.append(self.RECT_SECTION)
        self.names.append(name)
        self.rect_data.append([height, width])
        return name

    def genereta_I_section(
        self,
        name: str,
        web_height: float,
        web_width: float,
        top_flange_height: float,
        top_flange_width: float,
        bot_flange_height: float,
        bot_flange_width: float
    ):
        self.type_.append(self.I_SECTION)
        self.names.append(name)
        self.I_data.append([web_height, web_width, top_flange_height, top_flange_width, bot_flange_height, bot_flange_width])
        return name

class SectionsManager:
    
    def __init__(self, sections: Sections):
        self._sections = sections
    
    @cached_property
    def names(self):
        return np.array(self._sections.names, dtype=np.str_)
    
    @cached_property
    def type_(self):
        return np.array(self._sections.type_, dtype=np.int64)
    
    @cached_property
    def num_sections(self):
        return self.names.size  
    
    @cached_property
    def is_rect_section(self):
        return self.type_ == Sections.RECT_SECTION
    
    @cached_property
    def num_rect_sections(self):
        return np.count_nonzero(self.is_rect_section)
    
    @cached_property
    def rect_sections_data(self):
        return np.array(self._sections.rect_data, dtype=np.float64)
    
    @cached_property
    def is_I_section(self):
        return self.type_ == Sections.I_SECTION
    
    @cached_property
    def num_I_sections(self):
        return np.count_nonzero(self.is_I_section)
    
    @cached_property
    def I_sections_data(self):
        return np.array(self._sections.I_data, dtype=np.float64)
    
    @cached_property
    def cross_section_area(self):
        return np.zeros(self.num_sections, dtype=np.float64)

    @cached_property
    def moment_of_inertia_about_y(self):
        return np.zeros(self.num_sections, dtype=np.float64)

    @cached_property
    def moment_of_inertia_about_z(self):
        return np.zeros(self.num_sections, dtype=np.float64)

    @cached_property
    def torsional_constant(self):
        return np.zeros(self.num_sections, dtype=np.float64)

    @cached_property
    def shear_correction_factor(self):
        return np.zeros(self.num_sections, dtype=np.float64)
    
    def _cross_section_area_rect(self):
        self.cross_section_area[self.is_rect_section] = self.rect_sections_data[:, Sections.RECT_HEIGHT]*self.rect_sections_data[:, Sections.RECT_WIDTH]
    
    def _moment_of_inertia_about_y_rect(self):
        self.moment_of_inertia_about_y[self.is_rect_section] = self.rect_sections_data[:, Sections.RECT_HEIGHT]*self.rect_sections_data[:, Sections.RECT_WIDTH]**3/12

    def _moment_of_inertia_about_z_rect(self):
        self.moment_of_inertia_about_z[self.is_rect_section] = self.rect_sections_data[:, Sections.RECT_HEIGHT]**3*self.rect_sections_data[:, Sections.RECT_WIDTH]/12
    
    def _torsional_constant_rect(self):
        max_dim = np.max(self.rect_sections_data, axis=1)
        min_dim = np.min(self.rect_sections_data, axis=1)
        beta = 1/3 - 0.21*min_dim*(1 - (min_dim/max_dim)**4/12)/max_dim
        
        self.torsional_constant[self.is_rect_section] = beta*max_dim*min_dim**3
        
    def _shear_correction_factor_rect(self):
        self.shear_correction_factor[self.is_rect_section] = 5/6

    def _cross_section_area_I(self):
        area_of_top_flange = self.I_sections_data[:, Sections.I_TOP_FLANGE_HEIGHT]*self.I_sections_data[:, Sections.I_TOP_FLANGE_WIDTH]
        area_of_bot_flange = self.I_sections_data[:, Sections.I_BOT_FLANGE_HEIGHT]*self.I_sections_data[:, Sections.I_BOT_FLANGE_WIDTH]
        area_of_web = self.I_sections_data[:, Sections.I_WEB_HEIGHT]*self.I_sections_data[:, Sections.I_WEB_WIDTH]
        self.cross_section_area[self.is_I_section] = area_of_top_flange + area_of_bot_flange + area_of_web        

    def _moment_of_inertia_about_y_I(self):
        moment_of_inertia_of_top_flange = self.I_sections_data[:, Sections.I_TOP_FLANGE_HEIGHT]*self.I_sections_data[:, Sections.I_TOP_FLANGE_WIDTH]**3/12
        moment_of_inertia_of_bot_flange = self.I_sections_data[:, Sections.I_BOT_FLANGE_HEIGHT]*self.I_sections_data[:, Sections.I_BOT_FLANGE_WIDTH]**3/12
        moment_of_inertia_of_web = self.I_sections_data[:, Sections.I_WEB_HEIGHT]*self.I_sections_data[:, Sections.I_WEB_WIDTH]**3/12
        self.moment_of_inertia_about_y[self.is_I_section] = moment_of_inertia_of_top_flange + moment_of_inertia_of_bot_flange + moment_of_inertia_of_web 
    
    def _moment_of_inertia_about_z_I(self):
        tfh = self.I_sections_data[:, Sections.I_TOP_FLANGE_HEIGHT]
        tfw = self.I_sections_data[:, Sections.I_TOP_FLANGE_WIDTH]
        bfh = self.I_sections_data[:, Sections.I_BOT_FLANGE_HEIGHT]
        bfw = self.I_sections_data[:, Sections.I_BOT_FLANGE_WIDTH]
        wh = self.I_sections_data[:, Sections.I_WEB_HEIGHT]
        ww = self.I_sections_data[:, Sections.I_WEB_WIDTH]
        
        moment_of_inertia_of_top_flange = tfh**3*tfw/12
        moment_of_inertia_of_bot_flange = bfh**3*bfw/12
        moment_of_inertia_of_web = wh**3*ww/12
        
        area_of_top_flange = tfh*tfw
        area_of_bot_flange = bfh*bfw
        area_of_web = wh*ww
        
        center_of_gravity_y = (area_of_top_flange*(bfh + wh + tfh/2) + area_of_web*(bfh + wh/2) + area_of_bot_flange*bfh/2)/(area_of_top_flange + area_of_bot_flange + area_of_web)
        
        moment_of_inertia_of_top_flange += area_of_top_flange*(center_of_gravity_y - (bfh + wh + tfh/2))**2 
        moment_of_inertia_of_bot_flange += area_of_bot_flange*(center_of_gravity_y - bfh/2)**2
        moment_of_inertia_of_web += area_of_web*(center_of_gravity_y - (bfh + wh/2))**2
        
        self.moment_of_inertia_about_z[self.is_I_section] = moment_of_inertia_of_top_flange + moment_of_inertia_of_bot_flange + moment_of_inertia_of_web 

    
    
if __name__ == '__main__':
    
    sections = Sections()
    rect = sections.generate_rect_section("asd", 1, 2)
    rect = sections.generate_rect_section("asd", 1, 2)
    rect = sections.generate_rect_section("asd", 1, 2)
    rect = sections.generate_rect_section("asd", 1, 2)
    
    I = sections.genereta_I_section("qqq", 0.6-0.02, 0.05, 0.01, 0.3, 0.01, 0.3)
    I = sections.genereta_I_section("qqq", 1, 1, 1, 1, 1, 1)
    I = sections.genereta_I_section("qqq", 1, 1, 1, 1, 1, 1)
    I = sections.genereta_I_section("qqq", 1, 1, 1, 1, 1, 1)
    
    sections_manager = SectionsManager(sections)
    sections_manager._moment_of_inertia_about_y_rect()
    sections_manager._moment_of_inertia_about_z_I()
    sections_manager._moment_of_inertia_about_y_I()
    sections_manager._cross_section_area_I()
    print(sections_manager.moment_of_inertia_about_y)
    