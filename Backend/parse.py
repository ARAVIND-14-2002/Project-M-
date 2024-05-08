import os
import json

def convert_to_yolo(json_data, image_size, class_mapping):
    yolo_lines = []
    for annotation in json_data:
        label_name = annotation['labels']['labelName']
        x_min = annotation['rectMask']['xMin']
        y_min = annotation['rectMask']['yMin']
        width = annotation['rectMask']['width']
        height = annotation['rectMask']['height']

        # Calculate YOLO format coordinates
        x_center = (x_min + (width / 2)) / image_size
        y_center = (y_min + (height / 2)) / image_size
        bbox_width = abs(width) / image_size
        bbox_height = abs(height) / image_size

        # Replace label name with integer index
        label_index = class_mapping.get(label_name, -1)
        if label_index == -1:
            raise ValueError(f"Label '{label_name}' not found in class mapping.")

        yolo_line = f"{label_index} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_lines.append(yolo_line)

    return yolo_lines

def convert_json_to_yolo(json_file_path, output_dir, image_size, class_mapping):
    # Load JSON data from file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Convert JSON to YOLO format
    yolo_lines = convert_to_yolo(json_data, image_size, class_mapping)

    # Write YOLO lines to a text file
    filename = os.path.splitext(os.path.basename(json_file_path))[0]
    output_file_path = os.path.join(output_dir, f"{filename}.txt")
    with open(output_file_path, 'w') as f:
        for line in yolo_lines:
            f.write(line + '\n')

    print(f"Conversion completed. YOLOv5 text file is generated: {output_file_path}")

# Directory containing JSON files
json_dir = "D:/Project/Project M#/Dataset/Preprocess3/val"

# Directory to save YOLOv5 text files
output_dir = "D:/Project/Project M#/Dataset/Preprocess3/val"

# Image size
image_size = 512

# Class label to integer index mapping
class_mapping = {'Ceratium': 0, 'Cylindrocystis Brebissonii': 1, 'Lepocinclis': 2, 'Micrasterias': 3, 'Paramecium': 4,
                 'Peridinium': 5, 'Pinnularia': 6, 'Pleurotaenium': 7, 'Pyrocystis': 8, 'Volvox': 9,
                 'Coleps': 10, 'Collodictyon Triciliatum': 11, 'Didinium': 12, 'Dinobryon': 13, 'Frontonia': 14,
                 'Phacus': 15, 'Colsterium': 16, 'unnamed': 17}  # Add more classes if needed

# Iterate through all JSON files in the directory
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_file_path = os.path.join(json_dir, filename)
        convert_json_to_yolo(json_file_path, output_dir, image_size, class_mapping)
