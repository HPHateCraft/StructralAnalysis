import numpy as np

class GlobalCoord:
    """
    Defines constants for a standard 3D global coordinate system.

    This class provides unit vectors for a standard right-hand Cartesian
    coordinate system commonly used in structural analysis.

    **Coordinate System Orientation**

    The axes are oriented as shown below and follow the right-hand rule:

    .. image:: _static/right_hand_coordinate_system.jpg
       :alt: Global Coordinate System Diagram
       :width: 300px
       :align: center

    Alternatively, visualized with ASCII art::

           Y (Up)
           |
           |
           |
           |
           |
           -------------------- X (Right/East)
          /
         /
        /
       /
      Z (Out of screen/North)

    The right-hand rule implies:
    - Y cross Z = X
    - Z cross X = Y
    - X cross Y = Z

    Attributes:
        UNIT_VECTOR_X (np.ndarray): Unit vector along the X-axis, [1.0, 0.0, 0.0].
        UNIT_VECTOR_Y (np.ndarray): Unit vector along the Y-axis, [0.0, 1.0, 0.0].
        UNIT_VECTOR_Z (np.ndarray): Unit vector along the Z-axis, [0.0, 0.0, 1.0].

    **Usage Examples**

    You can access the unit vectors directly as class attributes:

    .. code-block:: python

        import numpy as np
        from SpaceFrame import GlobalCoord # Assuming this is how you import

        # Get the unit vector along the X-axis
        vec_x = GlobalCoord.UNIT_VECTOR_X
        print(f"Unit vector X: {vec_x}")

        # Example: Check if a vector is aligned with the Y-axis
        my_vector = np.array([0.0, 5.0, 0.0])
        is_aligned_y = np.allclose(my_vector / np.linalg.norm(my_vector),
                                   GlobalCoord.UNIT_VECTOR_Y)
        print(f"Is my_vector aligned with Y? {is_aligned_y}")

    Further notes on coordinate transformations might go here, or link to
    another section.
    """

    UNIT_VECTOR_X = np.array([1.0, 0.0, 0.0], np.float64)
    UNIT_VECTOR_Y = np.array([0.0, 1.0, 0.0], np.float64)
    UNIT_VECTOR_Z = np.array([0.0, 0.0, 1.0], np.float64)

    @staticmethod
    def rotation_matrix_about_x_3x3(angle_deg: float):
        cos = np.cos(np.deg2rad(angle_deg))
        sin = np.sin(np.deg2rad(angle_deg))
        rx = np.array([
            [1, 0,    0],
            [0, cos, -sin],
            [0, sin,  cos]
        ], dtype=np.float64)
        return rx
    
    @staticmethod
    def rotation_matrix_about_y_3x3(angle_deg: float):
        cos = np.cos(np.deg2rad(angle_deg))
        sin = np.sin(np.deg2rad(angle_deg))
        ry = np.array([
            [cos,  0, sin],
            [0,    1, 0],
            [-sin, 0, cos]
        ], dtype=np.float64)
        return ry
    
    @staticmethod
    def rotation_matrix_about_z_3x3(angle_deg: float):
        cos = np.cos(np.deg2rad(angle_deg))
        sin = np.sin(np.deg2rad(angle_deg))
        rz = np.array([
            [cos, -sin, 0],
            [sin,  cos, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        return rz