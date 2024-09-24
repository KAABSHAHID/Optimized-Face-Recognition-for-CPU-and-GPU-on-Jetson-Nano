#inference without GPU i.e by using CPU

import tensorrt as trt
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load the optimized TensorRT engine
engine_path = "/content/engine_model.trt"
with open(engine_path, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# Create an execution context to perform inference
with engine.create_execution_context() as context:
    # Allocate buffers for input and output
    inputs, outputs, bindings = [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = np.empty(size, dtype)
        device_mem = None  # For CPU inference, no need for device memory

        bindings.append(int(host_mem.ctypes.data))
        if engine.binding_is_input(binding):
            inputs.append(host_mem)
        else:
            outputs.append(host_mem)

    image_path = "/content/img.jpg"

    # Load the image using PIL
    image = Image.open(image_path)

    # Convert PIL image to a NumPy array
    image_np = np.array(image)

    # Define the transformations (resize and normalize)
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),        # Convert NumPy array back to PIL image
        transforms.Resize((112, 112)),  # Resize the image to (112, 112) as required
        transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the image
    ])

    input_data = preprocess_transform(image_np).unsqueeze(0)
    print(input_data)

    # Prepare your input data (e.g., load images, preprocess, etc.)
    # For example:
    # input_data = preprocess_input(your_input_data)  # Define your preprocessing function
    inputs[0][:] = input_data.ravel()

    # Run inference
    context.execute_v2(bindings)

# Convert the output tensor to a NumPy array
output_data_np = outputs[0]

# Post-process the output data if needed
# For example:
# output_results = postprocess_output(output_data_np)  # Define your postprocessing function

# Your output_results now contains the inference results after the optimization.
# You can use these results for further processing, analysis, or visualization.