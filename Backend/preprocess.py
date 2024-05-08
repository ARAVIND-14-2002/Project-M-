import cv2
import glob
import os
import numpy as np

def zoom_and_resize(image_path, output_dir, zoom_factor=1.2, size=(512, 512)):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the new dimensions
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Resize the image for zooming
    zoomed_image = cv2.resize(image, (new_width, new_height))

    # Calculate the crop dimensions
    start_row, start_col = int((new_height - height) / 2), int((new_width - width) / 2)
    end_row, end_col = start_row + height, start_col + width

    # Crop the zoomed image to original size
    cropped_image = zoomed_image[start_row:end_row, start_col:end_col]

    # Resize the cropped image to the desired size
    resized_image = cv2.resize(cropped_image, size)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the image to the output directory
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, resized_image)

# Get a list of all images in your dataset
image_paths = glob.glob('D:\Project\Project M#\Dataset\Images\*.jpg')

# Specify the output directory
output_dir = 'D://Project//Project M#//Dataset//Preprocess'
# Zoom in and resize each image
for image_path in image_paths:
    zoom_and_resize(image_path, output_dir)
