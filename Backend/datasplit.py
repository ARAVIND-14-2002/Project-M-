import os
import random
import shutil

# Define paths
original_dataset_dir = 'D:/Project/Project M#/Dataset/Preprocess'
base_dir = 'D:/Project/Project M#/Dataset/Preprocess3'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List image filenames
image_filenames = os.listdir(original_dataset_dir)
random.shuffle(image_filenames)

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
num_samples = len(image_filenames)

num_train_samples = int(num_samples * train_ratio)
num_val_samples = int(num_samples * val_ratio)

# Function to sanitize filenames
def sanitize_filename(filename):
    # Split filename and extension
    name, ext = os.path.splitext(filename)
    # Replace white spaces with underscores
    sanitized_name = name.replace(" ", "_")
    # Replace special characters like "." with underscores
    sanitized_name = sanitized_name.replace(".", "_")
    # Rejoin filename and extension
    sanitized_filename = sanitized_name + ext
    return sanitized_filename

# Copy images to train directory
for filename in image_filenames[:num_train_samples]:
    src = os.path.join(original_dataset_dir, filename)
    dst_filename = sanitize_filename(filename)
    dst = os.path.join(train_dir, dst_filename)
    shutil.copyfile(src, dst)

# Copy images to validation directory
for filename in image_filenames[num_train_samples:num_train_samples + num_val_samples]:
    src = os.path.join(original_dataset_dir, filename)
    dst_filename = sanitize_filename(filename)
    dst = os.path.join(val_dir, dst_filename)
    shutil.copyfile(src, dst)

# Copy remaining images to test directory
for filename in image_filenames[num_train_samples + num_val_samples:]:
    src = os.path.join(original_dataset_dir, filename)
    dst_filename = sanitize_filename(filename)
    dst = os.path.join(test_dir, dst_filename)
    shutil.copyfile(src, dst)

print("Dataset splitting completed.")
