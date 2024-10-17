import numpy as np
import os

def crop(img: np.ndarray, crop_xy: int = 160, crop_z = 80):
    """Crop an ndarray.

    Args:
        img (np.ndarray): Original image of size 2D, 3D, or 4D. (Assuming t,x,y,z)
        crop_xy (int, optional): Cropped (xy) matrix size. Defaults to (160).
        crop_z (int, optional): Cropped (z) matrix size. Defaults to (80).
    """   
    mat_xy = img.shape[1] # initial image matrix size xy
    mat_z = img.shape[-1] # initial image matrix size z
    if mat_xy <= crop_xy:
        crop_xy = mat_xy
           
    # Calculate the cropping indices
    xy_b = int((mat_xy - crop_xy) // 2)
    xy_e = int(xy_b + crop_xy)
    z_b = int((mat_z - crop_z) // 2)
    z_e = int(z_b + crop_z)

    # Crop the image
    cropped_image = img[:, xy_b:xy_e, xy_b:xy_e, z_b:z_e]
    return cropped_image

def find_nii_files(directory, contain_string=None, extension='.nii'):
    nii_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension) and (contain_string is None or all(substring in file for substring in contain_string)):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                nii_files_list.append(relative_path)

    # Print the list of files
    for file in nii_files_list:
        print(file)
    return nii_files_list