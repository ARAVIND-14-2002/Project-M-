import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision.transforms import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.dataloaders import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords

# Load the YOLOv5 model with the specified weight file
def load_model(weight_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weight_file, map_location=device)
    model.eval()
    return model.to(device)

# Perform object detection on the input image
def detect_image(image_path, model):
    img_size = 640  # Input image size used during training
    conf_threshold = 0.4  # Confidence threshold for object detection
    iou_threshold = 0.5   # IoU threshold for non-maximum suppression

    # Load image and convert to RGB
    img0 = Image.open(image_path).convert('RGB')

    # Resize and pad image to fit model's expected input size
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert image to tensor and normalize
    img = F.to_tensor(img)
    img = img.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        detections = model(img)[0]  # Predictions (batch_size=1)
        detections = non_max_suppression(detections, conf_threshold, iou_threshold)[0]

    # Format results
    if detections is not None:
        detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], img0.size).round()

    return detections

# Function to handle button click event
def detect_objects():
    # Open file dialog to select image
    image_path = filedialog.askopenfilename()
    if image_path:
        # Perform object detection
        detections = detect_image(image_path, model)
        if detections is not None:
            # Display detection results
            messagebox.showinfo("Object Detection", "Objects detected!\n{}".format(detections))
        else:
            messagebox.showinfo("Object Detection", "No objects detected!")

# Create Tkinter GUI
root = tk.Tk()
root.title("YOLOv5 Object Detection")

# Load YOLOv5 model
weight_file = 'your_weight_file.pt'  # Specify the path to your weight file
model = load_model(weight_file)

# Create and pack button
button = tk.Button(root, text="Select Image", command=detect_objects)
button.pack()

# Start the GUI event loop
root.mainloop()
