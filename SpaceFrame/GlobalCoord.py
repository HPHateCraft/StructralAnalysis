import numpy as np

class GlobalCoord:
    """
    Defines constants for a standard 3D global coordinate system.

    This class acts as a namespace for unit vectors defining a right-hand
    Cartesian coordinate system. The axes are oriented as follows:

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

    The system adheres to the right-hand rule, meaning:
    - Y cross Z = X
    - Z cross X = Y
    - X cross Y = Z

    Attributes:
        UNIT_VECTOR_X (np.ndarray): Unit vector along the X-axis, [1.0, 0.0, 0.0].
        UNIT_VECTOR_Y (np.ndarray): Unit vector along the Y-axis, [0.0, 1.0, 0.0].
        UNIT_VECTOR_Z (np.ndarray): Unit vector along the Z-axis, [0.0, 0.0, 1.0].
    """

    UNIT_VECTOR_X = np.array([1.0, 0.0, 0.0], np.float64)
    UNIT_VECTOR_Y = np.array([0.0, 1.0, 0.0], np.float64)
    UNIT_VECTOR_Z = np.array([0.0, 0.0, 1.0], np.float64)
