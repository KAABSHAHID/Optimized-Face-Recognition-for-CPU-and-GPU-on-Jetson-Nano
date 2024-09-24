import torch
import torch.onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit




def build_engine(onnx_path, trt_logger):
    model = onnx.load(onnx_path) 

    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(model.SerializeToString()):
        raise ValueError("Failed to parse the ONNX model.")

    profile = builder.create_optimization_profile()
    profile.set_shape("input_name", (1, 3, 112, 112), (1, 3, 112, 112), (1, 3, 112, 112))
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config=config)

    return engine





def save_engine(engine, engine_file_path):
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

if __name__ == "__main__":
    onnx_file = "onnx_file_path" #pass your own path of onnx file
    trt_logger = trt.Logger(trt.Logger.WARNING)  # Set the desired logger level

    # Build the TensorRT engine from the ONNX model
    engine = build_engine(onnx_file, trt_logger)

    # Save the engine to a file
    engine_file_path = "tensorrt_file_path_to_save.trt"  #pass the path where you want to save the model
    save_engine(engine, engine_file_path)
    print(f"TensorRT engine is saved to '{engine_file_path}'")