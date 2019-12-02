import numpy as np
import math


class Volume:
    """
    Volume data class.

    Attributes:
        data: Numpy array with the voxel data. Its shape will be (dim_x, dim_y, dim_z).
        dim_x: Size of dimension X.
        dim_y: Size of dimension Y.
        dim_z: Size of dimension Z.
    """

    def __init__(self, array, compute_histogram=True):
        """
        Inits the volume data.
        :param array: Numpy array with shape (dim_x, dim_y, dim_z).
        """

        self.data = array
        self.histogram = np.array([])
        self.dim_x = array.shape[0]
        self.dim_y = array.shape[1]
        self.dim_z = array.shape[2]

        if compute_histogram:
            self.compute_histogram()

    def get_voxel(self, x, y, z):
        """Retrieves the voxel for the """
        return self.data[x, y, z]

    def get_minimum(self):
        return self.data.min()

    def get_maximum(self):
        return self.data.max()

    def compute_histogram(self):
        self.histogram = np.histogram(self.data, bins=np.arange(self.get_maximum() + 1))[0]


class VoxelGradient:
    def __init__(self, gx=0, gy=0, gz=0):
        self.x = gx
        self.y = gy
        self.z = gz
        self.magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)


ZERO_GRADIENT = VoxelGradient()


class GradientVolume:
    def __init__(self, volume: Volume):
        self.volume = volume
        self.data = []
        self.compute()
        self.max_magnitude = -1.0

    def get_gradient(self, x, y, z):
        print(len(self.data), self.volume.dim_x, self.volume.dim_y, self.volume.dim_z,  x + self.volume.dim_x * (y + self.volume.dim_y * z))
        return self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)]

    def set_gradient(self, x, y, z, value):
        self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)] = value

    def get_voxel(self, i):
        return self.data[i]

    def compute(self):
        """
        Computes the gradient for the current volume
        """
        # construct data with empty VoxelGradients()
        self.data = [ZERO_GRADIENT] * (self.volume.dim_x * self.volume.dim_y * self.volume.dim_z)

        # get a unique id 
        id_structure = np.unique(self.volume.data).item(1)
        structures_volume = self.volume.data.copy()
        structures_volume[structures_volume != id_structure] = 0

        for x in range(0, self.volume.dim_x):
            for y in range(0, self.volume.dim_y):
                for z in range(0, self.volume.dim_z):
                    if x == 0 or y == 0 or z == 0 or x == self.volume.dim_x - 1 or y == self.volume.dim_y - 1 or z == self.volume.dim_z - 1:
                        self.set_gradient(x, y, z, VoxelGradient())
                    else: 
                        gx = (self.volume.get_voxel(x-1, y, z) - self.volume.get_voxel(x+1, y, z)) / 2
                        gy = (self.volume.get_voxel(x, y-1, z) - self.volume.get_voxel(x, y+1, z)) / 2
                        gz = (self.volume.get_voxel(x, y, z-1) - self.volume.get_voxel(x, y, z+1)) / 2
                        self.set_gradient(x, y, z, VoxelGradient(gx, gy, gz))  

        #delta = self.get_max_gradient_magnitude() * 0.1
        #for id,x in enumerate(self.data):
        #    if abs(x.magnitude - self.get_max_gradient_magnitude()) > delta:
        #        x.magnitude = 0.0
        
    def get_max_gradient_magnitude(self):
        if self.max_magnitude < 0:
            gradient = max(self.data, key=lambda x: x.magnitude)
            self.max_magnitude = gradient.magnitude

        return self.max_magnitude