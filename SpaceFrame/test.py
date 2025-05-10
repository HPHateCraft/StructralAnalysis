

if __name__ == '__main__':
    import numpy as np

    # from Nodes import Nodes
    # from LineElements import LineElements
    # from Materials import Materials
    # from Sections import Sections
    # from Loads import Loads, LoadSolver, NodalLoad
    # from StiffnessMatrix import StiffnessMatrix
    # from TransMatrix import TransMatrix
    # from Solver import Solver
    # from PostProcess import PostProcess
    from FrameStructure import FrameStructure

    fs = FrameStructure()
    node1 = fs.nodes.generate_node(0.0, 0.0, 0.0, False, False, False, False, False, False)
    node2 = fs.nodes.generate_node(6.0, 0.0, 0.0)
  
    mat1 = fs.materials.generate_material("", 33e6, 0.49)
    
    sec1 = fs.sections.generate_section("", 0.6, 0.2)
    
    elem1 = fs.elements.generate_element(node1, node2, mat1, sec1, 0.0)    
    
    fs.elements_loads.add_distributed_load(elem1, fs.elements_loads.FORCE, 1000, 1000, 0.0, 1.0, fs.elements_loads.GLOBAL_Y)
    
    np.set_printoptions(precision=7, suppress=True)
    
    fs.elements_loads.add_temperature(elem1, 1.0, 1, 1)
    fs.solver.solve()

    print(fs.materials.shear_modulus)
    print(fs.post_process.nodes_displacement_in_nodal_coord)
    # print(fs._elements_load_solver.load_vector_in_global_coord)
    
    
    