# Optimized-Face-Recognition-for-CPU-and-GPU-on-Jetson-Nano  

Optimize a PyTorch face recognition pre-trained model for inference on Jetson Nano.

## Requirements
- **Operating System:** Ubuntu >= 18.04
- **Python Version:** Python >= 3.6
- **CUDA:** Version as per your system
- **Libraries:**
  - torch 2.0.1
  - torchvision
  - onnx
  - tensorrt
  - PIL
  - numpy
  - time

## Steps

### Step 1: Convert PyTorch Model to ONNX
1. Load the model architecture from `backbone.py` (CBAMResNet).
2. Modify paths for `checkpoint_file` and `onnx_model_file` in the conversion script.
3. Execute the script to generate the ONNX model.

### Step 2: Generate TensorRT Engine from ONNX
1. Modify paths for `onnx_file` and `engine_file_path` in the TensorRT conversion script.
2. Run the script to create the TensorRT engine.

### Step 3: Run Inference on Jetson Nano
1. Update the input image path in the inference script.
2. Execute the script to perform inference using the GPU.

### Step 4: Inference on CPU  
- This solution helps to run it on **CPU** only system without requiring GPU.      
- Load the model and image in a separate script for CPU-based inference.
- Run the script to perform inference on the CPU.  

## Step 5: Live Inferencing
1. Integrate a camera or video stream into the inference script.
2. Capture frames in real-time and preprocess them for the model.
3. Use the TensorRT engine to perform inference on each frame and display results live.

## Use Cases
- **Security Surveillance:** Implement real-time facial recognition in security cameras for monitoring.
- **Access Control:** Utilize face recognition for secure entry systems in buildings.
- **Customer Insights:** Deploy in retail environments to analyze customer demographics and behavior.
- **Social Media Applications:** Enhance photo tagging features by automatically recognizing individuals in images.

