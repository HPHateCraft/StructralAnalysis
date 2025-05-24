

if __name__ == '__main__':
    from FrameStructure import FrameStructure
    
    import numpy as np

    fs = FrameStructure()
    
    node1 = fs.nodes.generate_node(0.0, 0.0, 0.0, False, False, False, False, False, False)
    node2 = fs.nodes.generate_node(6.0, 0.0, 0.0, False, False, False, False, False, False)
    node3 = fs.nodes.generate_node(0.0, 3.0, 0.0, )
    node4 = fs.nodes.generate_node(6.0, 3.0, 0.0, )
    node5 = fs.nodes.generate_node(0.0, 6.0, 0.0, )
    node6 = fs.nodes.generate_node(6.0, 6.0, 0.0,)
    
    fs.nodes.change_axes(node1, ry=45)
    
    mat1 = fs.materials.generate_material("c30", 33e6, 0.49)
    
    sec1 = fs.rect_sections.generate("B200X600", 0.6, 0.2)
    sec2 = fs.I_sections.generate("I", 0.6, 0.015, 0.01, 0.3, 0.01, 0.3)
    
    elem1 = fs.elements.generate_element(node1, node3, mat1, sec1)
    elem2 = fs.elements.generate_element(node3, node5, mat1, sec1)
    elem3 = fs.elements.generate_element(node2, node4, mat1, sec1)
    elem4 = fs.elements.generate_element(node4, node6, mat1, sec1)
    
    elem5 = fs.elements.generate_element(node3, node4, mat1, sec1)
    elem6 = fs.elements.generate_element(node5, node6, mat1, sec1)
    
    fs.distributed_force.add(elem6, fs.distributed_force.LOCAL_Y, 100.0, 200.0, 0.25, 0.75)
    # fs.distributed_force.add(elem6, fs.distributed_force.LOCAL_Y, 100.0, 200.0, 0.25, 0.75)
    # fs.distributed_moment.add(elem1, fs.distributed_moment.LOCAL_Z, 100.0, 200.0, 0.25, 0.75)
    
    fs.solver.solve()

    np.set_printoptions(precision=5, suppress=True)
    print(fs.post_process.nodes_forces_in_nodal_coord)
    # print(fs._nodes_manager.direction_cosines)
    