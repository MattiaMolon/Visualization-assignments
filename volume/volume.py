import numpy as np
import math
from tqdm import tqdm


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
        self.n_structures = 4           # number of biggest sections, if -1 we consider the entire brain
        self.mask = np.asarray([])      
        self.compute()
        self.max_magnitude = -1.0

    def get_gradient(self, x, y, z) -> VoxelGradient:
        return self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)]

    def set_gradient(self, x, y, z, value):
        self.data[x + self.volume.dim_x * (y + self.volume.dim_y * z)] = value

    def get_voxel(self, i):
        return self.data[i]

    def compute(self):
        """
        Computes the gradient for the current volume
        """
        # prepare data to compute gradient
        self.data = [ZERO_GRADIENT] * (self.volume.dim_x * self.volume.dim_y * self.volume.dim_z)
        volume_copy = Volume(np.copy(self.volume.data))
        unique, frequencies = np.unique(volume_copy.data, return_counts=True) 
        indexes_freq = np.flip(np.argsort(frequencies))
        
        # select only part of the ids if requested
        interested_structures = []
        if self.n_structures != -1:
            interested_structures = [unique[indexes_freq[i]] for i in range(1, self.n_structures + 1)]
        else:
            interested_structures = unique[1:]

        self.mask = np.isin(volume_copy.data, interested_structures)
        volume_copy.data[~self.mask] = 0
        volume_copy.data[self.mask] = 1000

        # compute gradient for each voxel
        for x in tqdm(range(0, volume_copy.dim_x), desc='gradient', leave=False):
            for y in range(0, volume_copy.dim_y):
                for z in range(0, volume_copy.dim_z):
                    if x == 0 or y == 0 or z == 0 or x == volume_copy.dim_x - 1 or y == volume_copy.dim_y - 1 or z == volume_copy.dim_z - 1:
                        self.set_gradient(x, y, z, VoxelGradient())
                    else: 
                        gx = 0.5 * (volume_copy.get_voxel(x-1, y, z) - volume_copy.get_voxel(x+1, y, z))
                        gy = 0.5 * (volume_copy.get_voxel(x, y-1, z) - volume_copy.get_voxel(x, y+1, z))
                        gz = 0.5 * (volume_copy.get_voxel(x, y, z-1) - volume_copy.get_voxel(x, y, z+1))
                        self.set_gradient(x, y, z, VoxelGradient(gx, gy, gz))  


    def get_max_gradient_magnitude(self):
        if self.max_magnitude < 0:
            gradient = max(self.data, key=lambda x: x.magnitude)
            self.max_magnitude = gradient.magnitude

        return self.max_magnitude