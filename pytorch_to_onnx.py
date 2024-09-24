import torch
import torchvision.models as models
import torch.onnx
from model.backbone import CBAMResNet
# Load the PyTorch model from checkpoint
model = CBAMResNet(num_layers = 50,)  # Replace with your model architecture and checkpoint loading code

# Load the model weights from the checkpoint file
checkpoint_file = 'path_for_last_net.ckpt'
checkpoint = torch.load(checkpoint_file)
#model.load_state_dict(checkpoint)  # Assuming 'state_dict' is the key storing the model weights

# Set the model to evaluation mode
model.eval()

# Define dummy input
dummy_input = torch.randn(1, 3, 112, 112)  # Replace with appropriate input shape for your model

# Convert to ONNX format
onnx_model_file = 'model.onnx' #where you want to save your model.onnx
torch.onnx.export(model, dummy_input, onnx_model_file, verbose=True, input_names=['input'], output_names=['output'])

print("Model converted and saved to", onnx_model_file)
