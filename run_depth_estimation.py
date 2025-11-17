import cv2
import os
import torch
import numpy as np
from PIL import Image

# Load MiDaS small model and Torch Script oad from PyHub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Use GPU if availble
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Input folder path with images to process
input_path = "processed"
# Output folder path
output_path = "depth_output"
os.makedirs(output_path, exist_ok=True)

# Iterate through the images and process (should be in .png format)
for img_name in sorted(os.listdir(input_path)):
    if not img_name.lower().endswith((".png")):
        continue

    # Read and resize
    img = cv2.imread(os.path.join(input_path, img_name))
    # Convert to RGB for Open Cascade
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Small input for MiDaS_small
    img = cv2.resize(img, (256, 256))

    # Tensor conversion and normalization
    img_input = torch.from_numpy(img / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Run inference
    with torch.no_grad():
        pred = midas(img_input)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Save the depth estimated images in the output folder
    depth = pred.cpu().numpy()
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    cv2.imwrite(os.path.join(output_path, img_name), depth_colored)
    print(f"Saved: {img_name}")

print(f"\n Save depth maps in: {output_path}/")
