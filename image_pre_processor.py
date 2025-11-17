import os
from PIL import Image

# Input and output directories
input_folder = "capture1"
output_folder = "processed"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# List all image files (you can filter by extension if needed)
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".png"))
])

# Process every 5th image
numImg = int(len(image_files)/2)
for i in range(0, numImg, 5):
    img_name = image_files[i]
    input_path = os.path.join(input_folder, img_name)
    output_path = os.path.join(output_folder, img_name)

    try:
        # Open image
        img = Image.open(input_path)

        # Reduce resolution (e.g., to 50% of original)
        new_size = (img.width // 2, img.height // 2)
        img_resized = img.resize(new_size, Image.LANCZOS)

        # Save processed image
        img_resized.save(output_path)
        print(f"Saved reduced image: {output_path}")

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(" Processing complete!")
