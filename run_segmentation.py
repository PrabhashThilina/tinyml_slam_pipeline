import cv2
import os
import torch
import numpy as np
import segmentation_models_pytorch as smp

# Use GPU if availble
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load lightweight MobileNetV2 model for segmentation
model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", classes=1, activation="sigmoid")
model.to(device)
model.eval()

# Input folder path with images to process
input_path = "processed"
# Output folder path
output_path = "segmentation_output"
os.makedirs(output_path, exist_ok=True)

# Iterate through the images and process (should be in .png format)
for img_name in sorted(os.listdir(input_path))[:50]:  # limit for test
    img = cv2.imread(os.path.join(input_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        mask = model(tensor)
        mask = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(output_path, img_name), mask)
    print(f"Saved: {img_name}")

print("segmentation masks saves successfully.")
