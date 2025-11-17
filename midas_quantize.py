import torch
import os
import time
import subprocess


# Check ONNX dependencies
def ensure_package(pkg_name):
    try:
        __import__(pkg_name)
    except ImportError:
        print(f" Installing missing package: {pkg_name} ...")
        subprocess.check_call(["pip", "install", pkg_name])

ensure_package("onnx")
ensure_package("onnxruntime")

import onnx
import onnxruntime as ort

# Load MiDaS small model
print(" Loading MiDaS_small from Torch Hub...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
print(f" Model loaded on device: {device}")

# Apply INT8 Dynamic Quantization (int8)
print("⚙️ Applying dynamic quantization...")
midas_int8 = torch.quantization.quantize_dynamic(
    midas,
    {torch.nn.Linear},  # quantize only Linear layers
    dtype=torch.qint8
)
print(" Quantization complete.")

# Benchmark Inference Time
example_input = torch.randn(1, 3, 256, 256).to(device)

def benchmark(model, name="Model"):
    with torch.no_grad():
        start = time.time()
        for _ in range(5):
            _ = model(example_input)
        avg_time = (time.time() - start) / 5
        print(f"⏱ Avg inference ({name}): {avg_time*1000:.2f} ms")

print("\n Benchmarking...")
benchmark(midas, "Original FP32")
benchmark(midas_int8, "Quantized INT8")

# Export Quantized Model to ONNX (current folder)
onnx_filename = "midas_small_dynamic_quantized.onnx"
onnx_path = os.path.join(os.getcwd(), onnx_filename)
print(f"\n Exporting to ONNX → {onnx_path}")

torch.onnx.export(
    midas_int8,
    example_input.cpu(),
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['depth'],
    dynamic_axes={'input': {0: 'batch_size'}, 'depth': {0: 'batch_size'}}
)

print(f" Export complete: {onnx_path}")

# Validate ONNX Model
model_onnx = onnx.load(onnx_path)
onnx.checker.check_model(model_onnx)
print(" ONNX validation passed.")

# Test with ONNX Runtime
print("\n Testing ONNX model inference...")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

dummy_input = example_input.cpu().numpy()
start_time = time.time()
outputs = session.run(None, {input_name: dummy_input})
end_time = time.time()

depth_output = outputs[0]
print(f" Inference successful. Output shape: {depth_output.shape}")
print(f"⏱ ONNX Runtime inference time: {(end_time - start_time)*1000:.2f} ms")

# Report model size
size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f" Model size on disk: {size_mb:.2f} MB")

print("\n MiDaS_small quantized and exported successfully to current folder.")
