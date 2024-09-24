# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:45:02 2024

@author: mkaab
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import time
from model.backbone import CBAMResNet  # Import your model architecture from the .py file

def capture():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Unable to access Cam")
        return
    ret, frame = capture.read()
    capture.release()

    if not ret:
        print("Error: Unable to capture")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

# Load the model architecture
model = CBAMResNet(num_layers=50)

# Load the model weights from .ckpt file
weights_path = "./last_net.ckpt"
checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint)
model.eval()

# Capture an image
image = capture()

# Define the transformations (resize and normalize)
preprocess_transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize the image to (112, 112) as required
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the image
])

while True:
    image = capture()
    timeStamp = time.time()

    input_data = preprocess_transform(image).unsqueeze(0)

    lread = time.time() - timeStamp  # latency in preprocessing
    tread = 1 / lread  # throughput in preprocessing

    # Run inference
    with torch.no_grad():
        output_data = model(input_data)

    # Convert the output tensor to a NumPy array
    output_data_np = output_data.numpy()
    lout = time.time() - timeStamp  # latency in output
    tout = 1 / lout  # throughput in output

    print(output_data_np)
    print("\n")

    print("Latency in preprocessing: {}".format(lread))
    print("Throughput in preprocessing: {} FPS".format(tread))
    print("\n")
    print("Latency in output: {}".format(lout))
    print("Throughput in output: {} FPS".format(tout))

    # Post-process the output data if needed
    # output_results = postprocess_output(output_data_np)  # Define your postprocessing function
