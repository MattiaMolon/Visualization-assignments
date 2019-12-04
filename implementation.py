import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor, TransferFunction
from math import floor
from volume.volume import GradientVolume, Volume, VoxelGradient
from collections.abc import ValuesView
from tqdm import tqdm
import math
import random

# TODO: fare documentazione

def get_colors_vector():
    x = []
    x.append(TFColor(255.0, 0.0, 0.0, 1.0)) # red
    x.append(TFColor(153.0, 102.0, 51.0, 1.0)) # brown
    x.append(TFColor(102.0, 0.0, 204.0, 1.0)) # dark purple
    x.append(TFColor(102.0, 255.0, 102.0, 1.0)) # light green
    x.append(TFColor(0.0, 255.0, 255.0, 1.0)) # light blue
    x.append(TFColor(255.0, 153.0, 51.0, 1.0)) # orange
    x.append(TFColor(255.0, 0.0, 255.0, 1.0)) # purple
    x.append(TFColor(0.0, 255.0, 0.0,  1.0)) # green
    x.append(TFColor(255.0, 255.0, 0.0,  1.0)) # yellow
    x.append(TFColor(0.0, 0.0, 255.0, 1.0)) # blue
    return x

def get_voxel(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x or y >= volume.dim_y or z >= volume.dim_z:
        return 0

    """
    (x0, ..., x7) nearest voxels to [x,y,z] 
       x7--------x6
      /|        /|
     / |       / |
    x3--------x2 |      volume.size = (256, 256, 163)
    |  |      |  |
    |  x4-----|--x5
    | /       | /
    |/        |/
    x0--------x1
    """

    # fix limit coordinates
    x = x - 1.0 if x > volume.dim_x - 1 else x
    y = y - 1.0 if y > volume.dim_y - 1 else y
    z = z - 1.0 if z > volume.dim_z - 1 else z

    # compute voxels
    x0 = volume.data[math.floor(x), math.floor(y), math.floor(z)]
    x1 = volume.data[math.ceil(x), math.floor(y), math.floor(z)]
    x2 = volume.data[math.ceil(x), math.ceil(y), math.floor(z)]
    x3 = volume.data[math.floor(x), math.ceil(y), math.floor(z)]
    x4 = volume.data[math.floor(x), math.floor(y), math.ceil(z)]
    x5 = volume.data[math.ceil(x), math.floor(y), math.ceil(z)]
    x6 = volume.data[math.ceil(x), math.ceil(y), math.ceil(z)]
    x7 = volume.data[math.floor(x), math.ceil(y), math.ceil(z)]

    # compute parameters
    alpha = x - math.floor(x)
    beta = y - math.floor(y)
    gamma = z - math.floor(z)

    return  (1-alpha)*(1-beta)*(1-gamma)*x0 + \
            alpha*(1-beta)*(1-gamma)*x1 + \
            (1-alpha)*(1-beta)*gamma*x4 + \
            alpha*(1-beta)*gamma*x5 + \
            (1-alpha)*beta*(1-gamma)*x3 + \
            alpha*beta*(1-gamma)*x2 + \
            (1-alpha)*beta*gamma*x7 + \
            alpha*beta*gamma*x6

# get mean value of voxel in a set of volumes
def get_mean_voxel(x: float, y: float, z: float, volumes: dict) -> float:
    value = 0.0
    n_energies = 0.0
    for key in volumes.keys():
        tmp_value = get_voxel(volumes[key], x, y, z)
        if tmp_value > 0.0:
            n_energies += 1.0
            value += get_voxel(volumes[key], x, y, z)

    if n_energies != 0.0:
        value /= n_energies
    
    return value

# get max value of voxel in a set of volumes
def get_max_voxel(x: float, y: float, z: float, volumes: dict) -> float:
    value = 0.0
    for key in volumes.keys():
        tmp_value = get_voxel(volumes[key], x, y, z)
        value = tmp_value if value < tmp_value else value
    
    return value

# get mean color value of voxel in a set of volumes
# TODO: discutere alpha value
def get_max_voxel_color(x: float, y: float, z: float, volumes: dict, tfunc: dict, treshold: float = 0.40) -> TFColor:
    value = 0.0
    key = None
    for tmp_key in volumes:
        tmp_value = round( get_voxel(volumes[tmp_key], x, y, z) )
        if tmp_value > treshold*(tfunc[tmp_key].sMax / 1.25) :
            value = tmp_value
            key = tmp_key

    color: TFColor = None
    if key != None:
        color = tfunc[key].get_color(value)        
    
    return color

class RaycastRendererImplementation(RaycastRenderer):
    """
    Class to be implemented.
    """

    def clear_image(self):
        """Clears the image data"""
        self.image.fill(0)

    def render_slicer(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # U vector. See documentation in parent's class
        u_vector = view_matrix[0:3]

        # V vector. See documentation in parent's class
        v_vector = view_matrix[4:7]

        # View vector. See documentation in parent's class
        view_vector = view_matrix[8:11]

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):
                # Get the voxel coordinate X
                voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + \
                                     volume_center[0]

                # Get the voxel coordinate Y
                voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + \
                                     volume_center[1]

                # Get the voxel coordinate Z
                voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + \
                                     volume_center[2]

                # Get voxel value
                value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def render_mip(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        # ration vectors
        u_vector = view_matrix[0:3] # X
        v_vector = view_matrix[4:7] # Y
        view_vector = view_matrix[8:11] # Z

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        volume_maximum = volume.get_maximum()
        
        # Diagonal cube
        diagonal = math.floor(math.sqrt(volume.dim_x**2 + volume.dim_y**2 + volume.dim_z**2) / 2)

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):
                
                value = 0
                for z in range(-diagonal, diagonal, 2):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) \
                                        + view_vector[0] * z + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) \
                                        + view_vector[1] * z + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) \
                                        + view_vector[2] * z + volume_center[2]

                    # Get voxel value
                    tmp = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                    value = tmp if value < tmp else value

                # Normalize value to be between 0 and 1
                red = value / volume_maximum
                green = red
                blue = red
                alpha = 1.0 if red > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    def render_compositing(self, view_matrix: np.ndarray, volume: Volume, image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()
        
        # ration vectors
        u_vector = view_matrix[0:3] # X
        v_vector = view_matrix[4:7] # Y
        view_vector = view_matrix[8:11] # Z

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [volume.dim_x / 2, volume.dim_y / 2, volume.dim_z / 2]
        diagonal = math.floor(math.sqrt(volume.dim_x**2 + volume.dim_y**2 + volume.dim_z**2) / 2)

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):

                last_color: TFColor = None
                for z in range(diagonal, -diagonal, -5):
                    # Get the voxel coordinate X
                    voxel_coordinate_x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) \
                                        + view_vector[0] * z + volume_center[0]

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) \
                                        + view_vector[1] * z + volume_center[1]

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) \
                                        + view_vector[2] * z + volume_center[2]

                    # Get voxel value
                    value = get_voxel(volume, voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z)
                    value = round(value)
                    
                    # Get voxel RGBA
                    base_color = self.tfunc.get_color(value)
                    voxel_color = TFColor(base_color.r*base_color.a, base_color.g*base_color.a, \
                                          base_color.b*base_color.a, base_color.a)
                    if last_color != None:
                        r = voxel_color.r + (1 - voxel_color.a)*last_color.r
                        g = voxel_color.g + (1 - voxel_color.a)*last_color.g
                        b = voxel_color.b + (1 - voxel_color.a)*last_color.b
                        voxel_color = TFColor(r, g, b, 1.0)
                    
                    last_color = voxel_color

                # Normalize value to be between 0 and 1
                red = last_color.r
                green = last_color.g
                blue = last_color.b
                alpha = last_color.a if red > 0 and green > 0 and blue > 0 else 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha

    # visualize annotations
    # TODO: prendere solamente intervalli di valori del gradiente
    def render_mouse_brain_annotation(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict, 
                            image_size: int, image: np.ndarray):
        # Clear the image
        self.clear_image()

        #set transfer function
        range_max = math.ceil(self.annotation_gradient_volume.get_max_gradient_magnitude() * 1.5)
        self.tfunc.init(0, range_max)
        
        # ration vectors
        u_vector = view_matrix[0:3] # X
        v_vector = view_matrix[4:7] # Y
        view_vector = view_matrix[8:11] # Z

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        gradient_volume = self.annotation_gradient_volume.volume
        volume_center = [gradient_volume.dim_x / 2, gradient_volume.dim_y / 2, gradient_volume.dim_z / 2]
        diagonal = math.floor(math.sqrt(gradient_volume.dim_x**2 + gradient_volume.dim_y**2 + gradient_volume.dim_z**2) / 2)

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):

                last_color: TFColor = None
                for z in range(diagonal, -diagonal, -1):

                    # Get the voxel coordinate X
                    voxel_coordinate_x = math.floor(u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) \
                                        + view_vector[0] * z + volume_center[0])

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = math.floor(u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) \
                                        + view_vector[1] * z + volume_center[1])

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = math.floor(u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) \
                                        + view_vector[2] * z + volume_center[2])
                    
                    # Get voxel value
                    value = self.annotation_gradient_volume.get_gradient(voxel_coordinate_x, voxel_coordinate_y, voxel_coordinate_z).magnitude
                    if value != 0:
                        value = round(value)
                        
                        # Get voxel RGBA
                        base_color = self.tfunc.get_color(value)
                        voxel_color = TFColor(base_color.r*base_color.a, base_color.g*base_color.a, \
                                            base_color.b*base_color.a, base_color.a)
                        if last_color != None:
                            r = voxel_color.r + (1 - voxel_color.a)*last_color.r
                            g = voxel_color.g + (1 - voxel_color.a)*last_color.g
                            b = voxel_color.b + (1 - voxel_color.a)*last_color.b
                            voxel_color = TFColor(r, g, b, 1.0)
                        
                        last_color = voxel_color

                # set color pixel image
                if last_color != None:
                    red = last_color.r
                    green = last_color.g
                    blue = last_color.b
                    alpha = last_color.a
                else:
                    red = 0.0
                    green = 0.0
                    blue = 0.0
                    alpha = 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha
    
    # visualize energy levels
    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict, 
                            image_size: int, image: np.ndarray):    
        # Clear the image
        self.clear_image()

        # set tfunction for each energy
        # TODO: calcolare tfuncions in funzione del massimo totale
        colors = get_colors_vector()
        tfunc_dict = {}
        for key in energy_volumes.keys():   
            # find tfcuntion color and maximum value of energy
            tmp_color = colors.pop()
            range_max = math.ceil(np.max(energy_volumes[key].data) * 1.25)
            
            # define tfunction of the energy volume
            tmp_tfunc = TransferFunction()
            tmp_tfunc.init(0, range_max, tmp_color)
            tfunc_dict[key] = tmp_tfunc

        # ration vectors
        u_vector = view_matrix[0:3] # X
        v_vector = view_matrix[4:7] # Y
        view_vector = view_matrix[8:11] # Z

        # Center of the image. Image is squared
        image_center = image_size / 2

        # Center of the volume (3-dimensional)
        volume_center = [annotation_volume.dim_x / 2, annotation_volume.dim_y / 2, annotation_volume.dim_z / 2]
        diagonal = math.floor(math.sqrt(annotation_volume.dim_x**2 + annotation_volume.dim_y**2 + annotation_volume.dim_z**2) / 2)

        # Define a step size to make the loop faster
        step = 10 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):

                last_color: TFColor = None
                max_value: float = 0.0
                for z in range(diagonal, -diagonal, -1):

                    # Get the voxel coordinate X
                    voxel_coordinate_x = math.floor(u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) \
                                        + view_vector[0] * z + volume_center[0])

                    # Get the voxel coordinate Y
                    voxel_coordinate_y = math.floor(u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) \
                                        + view_vector[1] * z + volume_center[1])

                    # Get the voxel coordinate Z
                    voxel_coordinate_z = math.floor(u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) \
                                        + view_vector[2] * z + volume_center[2])
                    
                    ###########################################
                    ######## COMPOSITING CON MAX VOXEL ########
                    # Get maximum voxel value
                    voxel_color: TFColor = get_max_voxel_color(voxel_coordinate_x, voxel_coordinate_y, \
                                                                voxel_coordinate_z, energy_volumes, tfunc_dict)

                    if voxel_color != None:
                        # Get voxel RGBA
                        voxel_color = TFColor(voxel_color.r*voxel_color.a, voxel_color.g*voxel_color.a, \
                                            voxel_color.b*voxel_color.a, voxel_color.a)
                        if last_color != None:
                            r = voxel_color.r + (1 - voxel_color.a)*last_color.r
                            g = voxel_color.g + (1 - voxel_color.a)*last_color.g
                            b = voxel_color.b + (1 - voxel_color.a)*last_color.b
                            voxel_color = TFColor(r, g, b, 1.0)
                        last_color = voxel_color
                    ###########################################
                    ################### MIP ###################
                    # tupla: tuple(TFColor, float) = get_max_voxel_color(voxel_coordinate_x, voxel_coordinate_y, \
                    #                                              voxel_coordinate_z, energy_volumes, tfunc_dict)

                    # if tupla[1] > max_value:
                    #     last_color = tupla[0]
                    #     max_value = tupla[1]


                # set color pixel image
                if last_color != None:
                    red = last_color.r
                    green = last_color.g
                    blue = last_color.b
                    alpha = 1.0
                else:
                    red = 0.0
                    green = 0.0
                    blue = 0.0
                    alpha = 0.0

                # Compute the color value (0...255)
                red = math.floor(red * 255) if red < 255 else 255
                green = math.floor(green * 255) if green < 255 else 255
                blue = math.floor(blue * 255) if blue < 255 else 255
                alpha = math.floor(alpha * 255) if alpha < 255 else 255

                # Assign color to the pixel i, j
                image[(j * image_size + i) * 4] = red
                image[(j * image_size + i) * 4 + 1] = green
                image[(j * image_size + i) * 4 + 2] = blue
                image[(j * image_size + i) * 4 + 3] = alpha


