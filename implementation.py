import numpy as np
from genevis.render import RaycastRenderer
from genevis.transfer_function import TFColor
from volume.volume import GradientVolume, Volume
from collections.abc import ValuesView
from tqdm import tqdm
import math

def get_solid_colors():
    """
    get vector containing 10 solid colors in order to color the energies
    """
    colors = []
    colors.append([1.0, 0.0, 0.0]) # red
    colors.append([0.6, 0.4, 0.2]) # brown
    colors.append([0.0, 0.0, 1.0]) # blue
    colors.append([0.4, 0.0, 0.8]) # dark purple
    colors.append([0.4, 1.0, 0.4]) # light green
    colors.append([0.0, 1.0, 0.0]) # green 
    colors.append([1.0, 1.0, 0.0]) # yellow
    colors.append([1.0, 0.0, 1.0]) # purple
    colors.append([1.0, 0.6, 0.2]) # orange
    colors.append([0.0, 1.0, 1.0]) # light blue
    return colors

def get_voxel(volume: Volume, x: float, y: float, z: float):
    """
    Retrieves the value of a voxel for the given coordinates.
    :param volume: Volume from which the voxel will be retrieved.
    :param x: X coordinate of the voxel
    :param y: Y coordinate of the voxel
    :param z: Z coordinate of the voxel
    :return: Voxel value
    """
    if x < 0 or y < 0 or z < 0 or x >= volume.dim_x - 1 or y >= volume.dim_y - 1 or z >= volume.dim_z - 1 :
        return 0

    """
    (v0, ..., v7) nearest voxels to [x,y,z] 
       v6--------v7
      /|        /|
     / |       / |
    v4--------v5 |
    |  |      |  |
    |  v2-----|--v3
    | /       | /
    |/        |/
    v0--------v1
    """
    # get voxel coordinates limits     x0-------x-------------x1
    x0 = math.floor(x)
    y0 = math.floor(y)
    z0 = math.floor(z)
    x1 = math.ceil(x)
    y1 = math.ceil(y)
    z1 = math.ceil(z)

    # compute voxels
    v0 = volume.data[x0, y0, z0] 
    v1 = volume.data[x1, y0, z0]
    v2 = volume.data[x0, y0, z1]
    v3 = volume.data[x1, y0, z1]
    v4 = volume.data[x0, y1, z0]
    v5 = volume.data[x1, y1, z0]
    v6 = volume.data[x0, y1, z1]
    v7 = volume.data[x1, y1, z1]

    # compute parameters
    alpha = x - x0
    beta = y - y0
    gamma = z - z0

    return  (1-alpha)*(1-beta)*(1-gamma)*v0 + \
            alpha*(1-beta)*(1-gamma)*v1 + \
            (1-alpha)*beta*(1-gamma)*v2 + \
            alpha*beta*(1-gamma)*v3 + \
            (1-alpha)*(1-beta)*gamma*v4 + \
            alpha*(1-beta)*gamma*v5 + \
            (1-alpha)*beta*gamma*v6 + \
            alpha*beta*gamma*v7

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
        step = 20 if self.interactive_mode else 1

        for i in range(0, image_size, step):
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
        step = 20 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):
                
                value = 0
                for k in range(-diagonal, diagonal, 2):
                    # Get the voxel coordinates
                    x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + view_vector[0] * k + volume_center[0]
                    y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + view_vector[1] * k + volume_center[1]
                    z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + view_vector[2] * k + volume_center[2]

                    # Get voxel value
                    tmp = get_voxel(volume, x, y, z)
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
        step = 20 if self.interactive_mode else 1

        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):

                last_color: TFColor = None
                for k in range(diagonal, -diagonal, -1):
                    # Get the voxel coordinates
                    x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + view_vector[0] * k + volume_center[0]
                    y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + view_vector[1] * k + volume_center[1]
                    z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + view_vector[2] * k + volume_center[2]

                    # Get voxel value
                    value = get_voxel(volume, x, y, z)
                    
                    if value > 0:
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
                if last_color != None:
                    red = last_color.r
                    green = last_color.g
                    blue = last_color.b
                    alpha = last_color.a if red > 0 and green > 0 and blue > 0 else 0.0
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

    def render_mouse_brain(self, view_matrix: np.ndarray, annotation_volume: Volume, energy_volumes: dict,
                           image_size: int, image: np.ndarray):
        """
        fucntion that implements the visualization of the mouse brain.
        select below the function that you want in order to visualize the data.
        """
        # create volumes for the gradient magnitude and mask
        magnitude_volume = np.zeros(annotation_volume.data.shape)
        for x in range(0, annotation_volume.dim_x):
            for y in range(0, annotation_volume.dim_y):
                for z in range(0, annotation_volume.dim_z):
                    magnitude_volume[x, y, z] = self.annotation_gradient_volume.get_gradient(x, y, z).magnitude
        magnitude_volume = Volume(magnitude_volume)
        mask_volume = np.copy(self.annotation_gradient_volume.mask).astype(int)
        mask_volume = Volume(mask_volume)

        #################################################
        # set of different functions
        # TODO: aggiungere posibilità di colorare le varie parti con i colori del csv!
        # self.visualize_annotations_only(view_matrix, annotation_volume, magnitude_volume, image_size, image, csv_colors = False, precision = 1)
        # self.visualize_energies_only(view_matrix, mask_volume, energy_volumes, image_size, image, annotation_aware = True, precision = 1)
        self.visualize_both_energies_annotations(view_matrix, mask_volume, magnitude_volume, energy_volumes, image_size, image, precision = 1)

    def visualize_annotations_only(self, view_matrix: np.ndarray, annotation_volume: Volume, magnitude_volume: Volume,
                           image_size: int, image: np.ndarray, csv_colors: bool = False, precision = 1):

        # Clear the image
        self.clear_image()

        # set internal transfer function for boundaries
        self.tfunc.init(0, round(self.annotation_gradient_volume.get_max_gradient_magnitude()))

        # rotation vectors
        u_vector = view_matrix[0:3]     # X
        v_vector = view_matrix[4:7]     # Y
        view_vector = view_matrix[8:11] # Z

        # image and volume in the center of the window
        image_center = image_size / 2
        volume_center = [annotation_volume.dim_x / 2, annotation_volume.dim_y / 2, annotation_volume.dim_z / 2]
        half_diagonal = math.floor(math.sqrt(annotation_volume.dim_x**2 + annotation_volume.dim_y**2 + annotation_volume.dim_z**2) / 2)

        # Define a step size to make the loop faster
        step = 20 if self.interactive_mode else 1
        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):

                last_color: TFColor = None
                for k in range(half_diagonal, -half_diagonal, -precision):

                    # Get the rotated voxel value
                    x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + view_vector[0] * k + volume_center[0]
                    y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + view_vector[1] * k + volume_center[1]
                    z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + view_vector[2] * k + volume_center[2]
                    value = get_voxel(magnitude_volume, x, y, z)

                    # compute color
                    if value != 0:
                        voxel_color: TFColor = self.tfunc.get_color(round(value)) # <- check round
                        voxel_color = TFColor(voxel_color.r*voxel_color.a, voxel_color.g*voxel_color.a, \
                                            voxel_color.b*voxel_color.a, voxel_color.a) 
                        if last_color != None:
                            voxel_color = TFColor(voxel_color.r + last_color.r*(1-voxel_color.a), \
                                                voxel_color.g + last_color.g*(1-voxel_color.a), \
                                                voxel_color.b + last_color.b*(1-voxel_color.a), \
                                                1.0)
                        last_color = voxel_color

                # Select RGB values
                if last_color == None:
                    red = 0.0
                    green = 0.0
                    blue = 0.0
                    alpha = 0.0
                else:
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

    def visualize_energies_only(self, view_matrix: np.ndarray, mask_volume: Volume, energy_volumes: dict, 
                                image_size: int, image: np.ndarray, annotation_aware = False, precision = 1):

        # Clear the image
        self.clear_image()

        # get dictionary of colors and intensity for the energies
        colors = get_solid_colors()
        energy_color = {}           # dictionary {energy_key -> [r,g,b]}
        energy_max_intensity = {}   # dictionary {energy_key -> max_intensity}
        for key in energy_volumes.keys():
            energy_color[key] = colors.pop()
            energy_max_intensity[key] = np.max(energy_volumes[key].data)

        # rotation vectors
        u_vector = view_matrix[0:3]     # X
        v_vector = view_matrix[4:7]     # Y
        view_vector = view_matrix[8:11] # Z

        # image and volume in the center of the window
        image_center = image_size / 2
        volume_center = [mask_volume.dim_x / 2, mask_volume.dim_y / 2, mask_volume.dim_z / 2]
        half_diagonal = math.floor(math.sqrt(mask_volume.dim_x**2 + mask_volume.dim_y**2 + mask_volume.dim_z**2) / 2)

        # Define a step size to make the loop faster
        step = 20 if self.interactive_mode else 1
        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):

                last_color: TFColor = None
                for k in range(half_diagonal, -half_diagonal, -precision):

                    # Get the rotated voxel value
                    x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + view_vector[0] * k + volume_center[0]
                    y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + view_vector[1] * k + volume_center[1]
                    z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + view_vector[2] * k + volume_center[2]
                    
                    # Decide if compute voxel color
                    compute_voxel = True
                    if annotation_aware:
                        compute_result = get_voxel(mask_volume, x, y, z)
                        if compute_result == 0:
                            compute_voxel = False

                    # compute voxel color
                    voxel_color: TFColor = None   # voxel color in point x, y, z
                    if compute_voxel:
                        for key in energy_volumes.keys():
                            value = get_voxel(energy_volumes[key], x, y, z)
                            if value > 0:
                                intensity = value/energy_max_intensity[key]
                                energy_voxel_color = TFColor(energy_color[key][0] * intensity, \
                                                            energy_color[key][1] * intensity, \
                                                            energy_color[key][2] * intensity, \
                                                            intensity) # color of the energy in x, y, z
                                if voxel_color != None:
                                    energy_voxel_color = TFColor(energy_voxel_color.r + voxel_color.r * (1 - intensity), \
                                                                energy_voxel_color.g + voxel_color.g * (1 - intensity), \
                                                                energy_voxel_color.b + voxel_color.b * (1 - intensity), \
                                                                max([intensity, voxel_color.a]))
                                voxel_color = energy_voxel_color

                    # compute ray color
                    if voxel_color != None:
                        voxel_color = TFColor(voxel_color.r*voxel_color.a, voxel_color.g*voxel_color.a, \
                                              voxel_color.b*voxel_color.a, voxel_color.a)
                        if last_color != None:
                            voxel_color = TFColor(voxel_color.r + last_color.r*(1-voxel_color.a), \
                                                  voxel_color.g + last_color.g*(1-voxel_color.a), \
                                                  voxel_color.b + last_color.b*(1-voxel_color.a), \
                                                  1.0)
                        last_color = voxel_color


                # Select RGB values
                if last_color == None:
                    red = 0.0
                    green = 0.0
                    blue = 0.0
                    alpha = 0.0
                else:
                    red = last_color.r
                    green = last_color.g
                    blue = last_color.b
                    alpha = last_color.a if red > 0 or green > 0 or blue > 0 else 0.0

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

    def visualize_both_energies_annotations(self, view_matrix: np.ndarray, mask_volume: Volume, magnitude_volume: Volume,  
                                            energy_volumes: dict, image_size: int, image: np.ndarray, precision = 1):

        # Clear the image
        self.clear_image()

        # get dictionary of colors and intensity for the energies
        colors = get_solid_colors()
        energy_color = {}           # dictionary {energy_key -> [r,g,b]}
        energy_max_intensity = {}   # dictionary {energy_key -> max_intensity}
        for key in energy_volumes.keys():
            energy_color[key] = colors.pop()
            energy_max_intensity[key] = np.max(energy_volumes[key].data)

        # set internal transfer function for boundaries
        self.tfunc.init(0, round(self.annotation_gradient_volume.get_max_gradient_magnitude() * 1.1))

        # rotation vectors
        u_vector = view_matrix[0:3]     # X
        v_vector = view_matrix[4:7]     # Y
        view_vector = view_matrix[8:11] # Z

        # image and volume in the center of the window
        image_center = image_size / 2
        volume_center = [mask_volume.dim_x / 2, mask_volume.dim_y / 2, mask_volume.dim_z / 2]
        half_diagonal = math.floor(math.sqrt(mask_volume.dim_x**2 + mask_volume.dim_y**2 + mask_volume.dim_z**2) / 2)

        # Define a step size to make the loop faster
        step = 20 if self.interactive_mode else 1
        for i in tqdm(range(0, image_size, step), desc='render', leave=False):
            for j in range(0, image_size, step):

                last_color: TFColor = None
                for k in range(half_diagonal, -half_diagonal, -precision):

                    # Get the rotated voxel value
                    x = u_vector[0] * (i - image_center) + v_vector[0] * (j - image_center) + view_vector[0] * k + volume_center[0]
                    y = u_vector[1] * (i - image_center) + v_vector[1] * (j - image_center) + view_vector[1] * k + volume_center[1]
                    z = u_vector[2] * (i - image_center) + v_vector[2] * (j - image_center) + view_vector[2] * k + volume_center[2]
                    
                    # Decide if compute voxel color
                    compute_voxel = True
                    compute_result = get_voxel(mask_volume, x, y, z)
                    if compute_result == 0:
                        compute_voxel = False

                    # compute voxel color
                    voxel_color: TFColor = None   # voxel color in point x, y, z
                    if compute_voxel:

                        # compute border
                        border_value = get_voxel(magnitude_volume, x, y, z)
                        if border_value > 0.0:
                            border_color = self.tfunc.get_color(round(border_value))
                            border_color = TFColor(border_color.r * border_color.a, \
                                                   border_color.g * border_color.a, \
                                                   border_color.b * border_color.a, \
                                                   border_color.a) # color border
                            if voxel_color != None:
                                border_color = TFColor(border_color.r + voxel_color.r * (1 - border_color.a), \
                                                       border_color.g + voxel_color.g * (1 - border_color.a), \
                                                       border_color.b + voxel_color.b * (1 - border_color.a), \
                                                       max([border_color.a, voxel_color.a]))
                            voxel_color = border_color

                        # compute energy
                        for key in energy_volumes.keys():
                            value = get_voxel(energy_volumes[key], x, y, z)
                            if value > 0:
                                intensity = value/energy_max_intensity[key]
                                energy_voxel_color = TFColor(energy_color[key][0] * intensity, \
                                                            energy_color[key][1] * intensity, \
                                                            energy_color[key][2] * intensity, \
                                                            intensity) # color of the energy in x, y, z
                                if voxel_color != None:
                                    energy_voxel_color = TFColor(energy_voxel_color.r + voxel_color.r * (1 - intensity), \
                                                                energy_voxel_color.g + voxel_color.g * (1 - intensity), \
                                                                energy_voxel_color.b + voxel_color.b * (1 - intensity), \
                                                                max([intensity, voxel_color.a]))
                                voxel_color = energy_voxel_color

                    # compute ray color
                    if voxel_color != None:
                        voxel_color = TFColor(voxel_color.r*voxel_color.a, voxel_color.g*voxel_color.a, \
                                              voxel_color.b*voxel_color.a, voxel_color.a)
                        if last_color != None:
                            voxel_color = TFColor(voxel_color.r + last_color.r*(1-voxel_color.a), \
                                                  voxel_color.g + last_color.g*(1-voxel_color.a), \
                                                  voxel_color.b + last_color.b*(1-voxel_color.a), \
                                                  1.0)
                        last_color = voxel_color


                # Select RGB values
                if last_color == None:
                    red = 0.0
                    green = 0.0
                    blue = 0.0
                    alpha = 0.0
                else:
                    red = last_color.r
                    green = last_color.g
                    blue = last_color.b
                    alpha = last_color.a if red > 0 or green > 0 or blue > 0 else 0.0

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
               