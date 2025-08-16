from super_image import DrlnModel, ImageLoader
from PIL import Image
import os

TEST_DIR = 'test'
UPSCALED_DIR = 'upscaled'


import traceback

print("Loading DRLN model (eugenesiow/drln-bam, scale=4)...")
model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=4)
print("Model loaded and ready.\n")

def process_image(input_path, output_path):
    print(f"\n---\nProcessing: {input_path}")
    try:
        print("Opening image...")
        image = Image.open(input_path)
        print("Image opened. Loading into model...")
        inputs = ImageLoader.load_image(image)
        print("Running model inference...")
        preds = model(inputs)
        print("Model inference complete. Saving output...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ImageLoader.save_image(preds, output_path)
        print(f"Saved upscaled image to: {output_path}\n")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}\n")
        print(traceback.format_exc())


if not os.path.exists(UPSCALED_DIR):
    print(f"Creating output directory: {UPSCALED_DIR}")
    os.makedirs(UPSCALED_DIR)

image_count = 0
print(f"Starting upscaling process...\nInput folder: {TEST_DIR}\nOutput folder: {UPSCALED_DIR}\n")
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, TEST_DIR)
            output_path = os.path.join(UPSCALED_DIR, rel_path)
            print(f"\n[{image_count+1}] Queued: {input_path} -> {output_path}")
            process_image(input_path, output_path)
            image_count += 1
print(f"\nUpscaling complete. Total images processed: {image_count}")
