
# %%
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from skimage.transform import resize
from functions.util import find_nii_files

# matplotlib.use('TkAgg')

single_frame = False

foldername = 'data/results/'
    
filename = find_nii_files(foldername, contain_string=['wavelet'], extension='.nii')[0] # Getting first element only!

# Create a 3D matrix of images (height, width, num_frames)
image_matrix = nib.load(foldername + filename)

image_matrix = np.array(image_matrix.get_fdata())
image_matrix = np.squeeze(image_matrix)

# Optional: omit first frame (if looking at specific/jacs vent image)
# image_matrix = image_matrix[..., 1:]
slice_min = 10
slice_max = image_matrix.shape[0] - slice_min
if len(image_matrix.shape) == 5:
    image_matrix = image_matrix[slice_min:slice_max, slice_min:slice_max, :, 0, ...]
elif len(image_matrix.shape) == 4:
    image_matrix = image_matrix[slice_min:slice_max, slice_min:slice_max, :, ...]
elif len(image_matrix.shape) == 3:
    num_frames = 6 # Repeat matrix 6 times along frame dim
    image_matrix = np.repeat(image_matrix[slice_min:slice_max, slice_min:slice_max, :, np.newaxis], 6, -1)

# Dimensions
num_frames = image_matrix.shape[-1]
resolution = image_matrix.shape[0]

# Select slice
crop = 10
circshift = 0
slice_matrix = np.roll(image_matrix[crop:-crop, crop:-crop, int(image_matrix.shape[2]*0.5), :], circshift,-1)

# Upsample
desired_res = 1080
slice_matrix_upscaled = np.zeros((desired_res, desired_res, num_frames))

for i in range(num_frames):
    tmp = slice_matrix[:, :, i]
    tmp = cv2.resize(tmp, (desired_res, desired_res), interpolation=cv2.INTER_LANCZOS4)
    slice_matrix_upscaled[:, :, i] = np.rot90(tmp, 2)

# Overwrite
slice_matrix = slice_matrix_upscaled

# Normalize and enhance contrast
min_value = np.percentile(slice_matrix, 2)
max_value = np.percentile(slice_matrix, 98)
slice_matrix[slice_matrix < min_value] = min_value
slice_matrix[slice_matrix > max_value] = max_value
slice_matrix = ((slice_matrix - min_value) / (max_value - min_value) * 255).astype(np.uint8)


# Additional contrast enhancement (optional)
# for i in range(num_frames):
    # slice_matrix[:, :, i] = cv2.equalizeHist(slice_matrix[:, :, i])

# Set the frame rate (frames per second) for the video
frame_rate = 3

# Define the output video file name and codec
output_file = foldername + filename[:-4] + '.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, frame_rate, (desired_res, desired_res))

# Loop through each frame and write it to the video
for i in range(num_frames):
    frame = slice_matrix[:, :, i]
    frame_colored = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to color format
    out.write(frame_colored)

# Release the VideoWriter and close the video file
out.release()

# Save an image
if single_frame:
    # Display the image using imshow
    plt.imshow(slice_matrix[..., 1], cmap="gray")
    plt.axis('off')  # Turn off axis
    plt.show()  # Display the plot

    # Save the displayed image
    # Change the file extension based on the desired format
    output_path = foldername + 'single_frame.png'
    plt.savefig(output_path, bbox_inches='tight',
                pad_inches=0, dpi=300)  # Save the current plot


print("Video creation complete.")

# %%
