

if __name__ == '__main__':
    from FrameStructure import FrameStructure
    
    import numpy as np

    fs = FrameStructure()
    mat1 = fs.materials.generate_material("material", 33e6, 0.2)
    sec1 = fs.rect_sections.generate("B200X600", 0.6, 0.2)
    
    num_bays_x = 4
    num_bays_y = 3
    num_bays_z = 3
    
    bay_width_x = 6
    bay_width_y = 3
    bay_width_z = 6
    
    y_dir = 0.0
    for i in range(num_bays_y):
        z_dir = 0.0
        
        for j in range(num_bays_z):
            x_dir = 0.0
            
            for k in range(num_bays_x):
                fs.nodes.generate_node(x_dir, z_dir, y_dir)
                x_dir += bay_width_x
        
            z_dir -= bay_width_z
        
        y_dir += bay_width_y
    
    # COLUMNS
    
    counter = 0
    for i in range(num_bays_z-1):
        
        for j in range(num_bays_x*num_bays_z):
            fs.elements.generate_element(counter, counter+num_bays_x*num_bays_z, mat1, sec1)
            counter += 1
    
    # BEAMS X DIRICTION
    counter = num_bays_x*num_bays_z
    counter2 = 0
    
    for i in range(2):
        for i in range(num_bays_x*num_bays_z):
            if counter2 != num_bays_x - 1:
                fs.elements.generate_element(counter, counter + 1, mat1, sec1)

            else:
                counter2 = -1

            counter += 1
            counter2 += 1
        
    
    # BEAMS Y DIRICTION
    counter = num_bays_x*num_bays_z
    counter2 = 0
    range_ = num_bays_x*(num_bays_y - 1)*2
    for i in range(range_):
        fs.elements.generate_element(counter, counter + num_bays_x, mat1, sec1)     
        counter += 1
        
    
    
    print(fs._elements_manager.num_elements)
