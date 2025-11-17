import torch
import time
import os
import numpy as np
import onnxruntime as ort
import cv2
import glob
import sys

FP32_MODEL_PATH = "midas_small.pth"  # Full precision MiDaS model (.pth)
INT8_MODEL_PATH = "midas_small_dynamic_quantized.onnx"  # Quantized ONNX model
PROCESSED_FOLDER = "processed"  # Folder with test images

def get_first_image(folder):
    """Get the first PNG image from a folder."""
    image_files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not image_files:
        print(f" No PNG images found in folder: {folder}")
        sys.exit(1)
    print(f" Using test image: {os.path.basename(image_files[0])}")
    return image_files[0]

def load_image(path, size=(256, 256)):
    """Load and preprocess an image for inference."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = np.transpose(img / 255.0, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    return img

def get_model_size_mb(path):
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)

def run_pytorch_inference(model, input_tensor, device, warmup=3, runs=10):
    """Run inference on a PyTorch model."""
    input_torch = torch.from_numpy(input_tensor).to(device)
    with torch.no_grad():
        for _ in range(warmup):  # warm-up
            _ = model(input_torch)
        start = time.time()
        for _ in range(runs):
            output = model(input_torch)
        avg_time = (time.time() - start) / runs * 1000  # ms
    return output.cpu().numpy(), avg_time

def run_onnx_inference(model_path, input_tensor, warmup=3, runs=10):
    """Run inference on an ONNX model using ONNX Runtime."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        _ = session.run(None, {input_name: input_tensor})
    start = time.time()
    for _ in range(runs):
        outputs = session.run(None, {input_name: input_tensor})
    avg_time = (time.time() - start) / runs * 1000
    return outputs[0], avg_time

def compare_outputs(depth1, depth2):
    """Compare depth predictions."""
    diff = np.abs(depth1 - depth2)
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    corr = np.corrcoef(depth1.flatten(), depth2.flatten())[0, 1]
    return mae, rmse, corr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Loading MiDaS_small FP32 model on {device}...")

# Ensure MiDaS hub repo path availability
sys.path.append(os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master"))

# Load MiDaS architecture
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

import torch.serialization
from midas.midas_net_custom import MidasNet_small
torch.serialization.add_safe_globals([MidasNet_small])

# Load models
try:
    print(" Attempting to load full MiDaS model object...")
    midas = torch.load(FP32_MODEL_PATH, map_location=device, weights_only=False)
    print(" Loaded full MiDaS model object successfully.")
except Exception as e:
    print(f"⚠️ Direct load failed ({e}), trying as state_dict...")
    state_dict = torch.load(FP32_MODEL_PATH, map_location=device, weights_only=False)
    midas.load_state_dict(state_dict)
    print(" Loaded model weights via state_dict.")

midas.eval().to(device)

# Load the first image
test_image_path = get_first_image(PROCESSED_FOLDER)
img_tensor = load_image(test_image_path)

# Run inference on both models
print("\n⚙️ Running inference on both models...")
depth_fp32, time_fp32 = run_pytorch_inference(midas, img_tensor, device)
depth_int8, time_int8 = run_onnx_inference(INT8_MODEL_PATH, img_tensor)

# Accuracy comparision
mae, rmse, corr = compare_outputs(depth_fp32, depth_int8)

# Get Model sizes
size_fp32 = get_model_size_mb(FP32_MODEL_PATH)
size_int8 = get_model_size_mb(INT8_MODEL_PATH)

# Report generation
print("\n MiDas Quantization Impact")
print("="*60)
print(f"{'Metric':<25} | {'FP32 (PyTorch)':<15} | {'INT8 (ONNX)':<15}")
print("-"*60)
print(f"{'Model Size (MB)':<25} | {size_fp32:<15.2f} | {size_int8:<15.2f}")
print(f"{'Inference Time (ms)':<25} | {time_fp32:<15.2f} | {time_int8:<15.2f}")
print("-"*60)
print(f"{'MAE (Abs Error)':<25} |      {mae:<15.4f} ")
print(f"{'RMSE (Root Mean Sq)':<25} |      {rmse:<15.4f}")
print(f"{'Correlation Coef.':<25} |      {corr:<15.4f}")
print("="*60)

# Visualization
depth_fp32 = depth_fp32.squeeze()
depth_int8 = depth_int8.squeeze()
diff_map = np.abs(depth_fp32 - depth_int8)

cv2.imshow("FP32 Depth (PyTorch)", cv2.normalize(depth_fp32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.imshow("INT8 Depth (ONNX)", cv2.normalize(depth_int8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.imshow("Difference Map", cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
