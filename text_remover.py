import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm
import json
from pathlib import Path
import time

class TextRemover:
    def __init__(self):
        # Initialize the Roboflow client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="ryMpZlt8aUKbx2LVev7u"
        )
        self.model_id = "webtoon-detection-rj7lh-zqfwf/6"
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def get_image_files(self, input_folder):
        """Recursively get all image files from input folder and subfolders"""
        image_files = []
        input_path = Path(input_folder)
        
        if not input_path.exists():
            print(f"❌ Input folder '{input_folder}' does not exist!")
            return []
            
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
                
        return image_files
    
    def detect_text_regions(self, image_path):
        """Detect text regions in the image using Roboflow API"""
        try:
            result = self.client.infer(str(image_path), model_id=self.model_id)
            return result
        except Exception as e:
            print(f"⚠️  Error detecting text in {image_path.name}: {str(e)}")
            return None
    
    def create_background_patch(self, image, x, y, width, height, patch_size=20):
        """Create a background patch by sampling surrounding pixels"""
        h, w = image.shape[:2]
        
        # Define sampling area around the text region
        sample_margin = patch_size * 2
        x1 = max(0, x - sample_margin)
        y1 = max(0, y - sample_margin)
        x2 = min(w, x + width + sample_margin)
        y2 = min(h, y + height + sample_margin)
        
        # Sample background pixels
        background_samples = []
        
        # Sample from top and bottom edges
        for i in range(x1, x2, patch_size//2):
            if y1 > 0:
                background_samples.append(image[y1, i])
            if y2 < h:
                background_samples.append(image[y2-1, i])
        
        # Sample from left and right edges
        for j in range(y1, y2, patch_size//2):
            if x1 > 0:
                background_samples.append(image[j, x1])
            if x2 < w:
                background_samples.append(image[j, x2-1])
        
        if not background_samples:
            # Fallback: sample from corners
            corners = [
                (0, 0), (w-1, 0), (0, h-1), (w-1, h-1),
                (x1, y1), (x2-1, y1), (x1, y2-1), (x2-1, y2-1)
            ]
            for cx, cy in corners:
                if 0 <= cx < w and 0 <= cy < h:
                    background_samples.append(image[cy, cx])
        
        if background_samples:
            # Calculate average background color
            background_color = np.mean(background_samples, axis=0).astype(np.uint8)
        else:
            # Fallback to white
            background_color = np.array([255, 255, 255], dtype=np.uint8)
        
        return background_color
    
    def inpaint_text_region(self, image, x, y, width, height):
        """Use OpenCV inpainting to properly remove text"""
        h, w = image.shape[:2]
        
        # Create a mask for the text region
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Expand the mask slightly to ensure complete coverage
        expand = 3
        x1 = max(0, x - expand)
        y1 = max(0, y - expand)
        x2 = min(w, x + width + expand)
        y2 = min(h, y + height + expand)
        
        # Fill the mask with a softer edge for better blending
        mask[y1:y2, x1:x2] = 255
        
        # Use OpenCV inpainting with larger radius for better gradient handling
        result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
        
        return result
    
    def create_seamless_background(self, image, x, y, width, height):
        """Create a seamless background by cloning nearby areas"""
        h, w = image.shape[:2]
        
        # Find a good source region for cloning (similar background)
        search_margin = 50
        x1 = max(0, x - search_margin)
        y1 = max(0, y - search_margin)
        x2 = min(w, x + width + search_margin)
        y2 = min(h, y + height + search_margin)
        
        # Sample multiple background regions
        background_regions = []
        
        # Top region
        if y1 > 0:
            top_region = image[y1:y, x1:x2]
            if top_region.size > 0:
                background_regions.append(top_region)
        
        # Bottom region
        if y2 < h:
            bottom_region = image[y+height:y2, x1:x2]
            if bottom_region.size > 0:
                background_regions.append(bottom_region)
        
        # Left region
        if x1 > 0:
            left_region = image[y1:y2, x1:x]
            if left_region.size > 0:
                background_regions.append(left_region)
        
        # Right region
        if x2 < w:
            right_region = image[y1:y2, x+width:x2]
            if right_region.size > 0:
                background_regions.append(right_region)
        
        if not background_regions:
            return None
        
        # Find the best matching background region
        best_region = None
        best_score = float('inf')
        
        for region in background_regions:
            if region.size > 0:
                # Calculate color similarity
                avg_color = np.mean(region, axis=(0, 1))
                score = np.sum(np.abs(avg_color - np.array([128, 128, 128])))
                if score < best_score:
                    best_score = score
                    best_region = region
        
        return best_region
    
    def create_simple_background(self, image, x, y, width, height):
        """Create a simple background by sampling nearby pixels more aggressively"""
        h, w = image.shape[:2]
        
        # Sample pixels from all four sides more densely
        all_samples = []
        
        # Sample from top edge (more densely)
        if y > 0:
            for i in range(x, x + width, 2):  # Every 2 pixels instead of 5
                if 0 <= i < w:
                    all_samples.append(image[y-1, i])
        
        # Sample from bottom edge (more densely)
        if y + height < h:
            for i in range(x, x + width, 2):  # Every 2 pixels instead of 5
                if 0 <= i < w:
                    all_samples.append(image[y+height, i])
        
        # Sample from left edge (more densely)
        if x > 0:
            for j in range(y, y + height, 2):  # Every 2 pixels instead of 5
                if 0 <= j < h:
                    all_samples.append(image[j, x-1])
        
        # Sample from right edge (more densely)
        if x + width < w:
            for j in range(y, y + height, 2):  # Every 2 pixels instead of 5
                if 0 <= j < h:
                    all_samples.append(image[j, x+width])
        
        # Also sample from corners and extended areas
        if y > 0 and x > 0:
            all_samples.append(image[y-1, x-1])  # Top-left corner
        if y > 0 and x + width < w:
            all_samples.append(image[y-1, x+width])  # Top-right corner
        if y + height < h and x > 0:
            all_samples.append(image[y+height, x-1])  # Bottom-left corner
        if y + height < h and x + width < w:
            all_samples.append(image[y+height, x+width])  # Bottom-right corner
        
        if not all_samples:
            return None
        
        # Calculate average background color
        background_color = np.mean(all_samples, axis=0).astype(np.uint8)
        
        return background_color
    
    def apply_background_effect(self, image, text_regions, confidence_threshold=0.5):
        """Apply aggressive text removal techniques to completely hide text regions"""
        if not text_regions or 'predictions' not in text_regions:
            return image

        # Convert PIL image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        result_image = image.copy()

        # Always use at least 0.5 (50%) confidence threshold, even if a lower value is passed
        min_conf = max(confidence_threshold, 0.5)
        high_confidence_predictions = []
        for prediction in text_regions['predictions']:
            confidence = prediction.get('confidence', 0.0)
            if confidence >= min_conf:
                high_confidence_predictions.append(prediction)

        print(f"📊 Found {len(text_regions['predictions'])} text regions, {len(high_confidence_predictions)} with confidence ≥{min_conf*100}%")

        for prediction in high_confidence_predictions:
            try:
                # Extract bounding box coordinates
                x = int(prediction['x'] - prediction['width'] / 2)
                y = int(prediction['y'] - prediction['height'] / 2)
                width = int(prediction['width'])
                height = int(prediction['height'])

                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, image.shape[1] - x)
                height = min(height, image.shape[0] - y)

                if width <= 0 or height <= 0:
                    continue

                # Fill the detected region with solid white
                result_image[y:y+height, x:x+width] = [255, 255, 255]

            except Exception as e:
                print(f"⚠️  Error processing text region: {str(e)}")
                continue

        return result_image
    
    def process_image(self, input_path, output_path, confidence_threshold=0.5):
        """Process a single image"""
        try:
            # Load image
            image = Image.open(input_path)
            
            # Detect text regions
            text_regions = self.detect_text_regions(input_path)
            
            if text_regions and 'predictions' in text_regions and text_regions['predictions']:
                # Apply background effect with confidence filtering
                processed_image = self.apply_background_effect(image, text_regions, confidence_threshold=confidence_threshold)
                
                # Convert back to PIL if needed
                if isinstance(processed_image, np.ndarray):
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    processed_image = Image.fromarray(processed_image)
                
                # Save processed image
                output_path.parent.mkdir(parents=True, exist_ok=True)
                processed_image.save(output_path, quality=95)
                
                return True, len(text_regions['predictions'])
            else:
                # No text detected, copy original image
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, quality=95)
                return True, 0
                
        except Exception as e:
            print(f"❌ Error processing {input_path.name}: {str(e)}")
            return False, 0
    
    def process_folder(self, input_folder, output_folder, confidence_threshold=0.5):
        """Process all images in the input folder and subfolders"""
        print("🔍 Scanning for image files...")
        image_files = self.get_image_files(input_folder)
        
        if not image_files:
            print("❌ No image files found in the input folder!")
            return
        
        print(f"📁 Found {len(image_files)} image files to process")
        print(f"📂 Input folder: {input_folder}")
        print(f"📂 Output folder: {output_folder}")
        print("-" * 50)
        
        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Process images with progress bar
        successful = 0
        total_text_regions = 0
        
        with tqdm(total=len(image_files), desc="Processing images", unit="img") as pbar:
            for input_path in image_files:
                # Calculate relative path to maintain folder structure
                rel_path = input_path.relative_to(Path(input_folder))
                output_path = Path(output_folder) / rel_path
                
                # Update progress description
                pbar.set_description(f"Processing {input_path.name}")
                
                # Process the image
                success, text_count = self.process_image(input_path, output_path, confidence_threshold)
                
                if success:
                    successful += 1
                    total_text_regions += text_count
                    pbar.set_postfix({
                        'Success': f"{successful}/{len(image_files)}",
                        'Text regions': total_text_regions
                    })
                else:
                    pbar.set_postfix({'Error': input_path.name})
                
                pbar.update(1)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 Processing Complete!")
        print(f"✅ Successfully processed: {successful}/{len(image_files)} images")
        print(f"📝 Total text regions detected: {total_text_regions}")
        print(f"📂 Output saved to: {output_folder}")
        print("=" * 50)

def main():
    print("🎨 Text Remover - Webtoon Speech Bubble Processor")
    print("=" * 50)
    
    # Initialize the text remover
    remover = TextRemover()
    
    # Get input and output folders
    input_folder = input("📁 Enter input folder path (or press Enter for 'input'): ").strip()
    if not input_folder:
        input_folder = "text_input"
    
    output_folder = input("📁 Enter output folder path (or press Enter for 'output'): ").strip()
    if not output_folder:
        output_folder = "output"
    
    # Get confidence threshold
    confidence_input = input("🎯 Enter confidence threshold (0.0-1.0, or press Enter for 0.5): ").strip()
    if not confidence_input:
        confidence_threshold = 0.5
    else:
        try:
            confidence_threshold = float(confidence_input)
            if confidence_threshold < 0.0 or confidence_threshold > 1.0:
                print("⚠️  Invalid confidence threshold. Using default value of 0.5")
                confidence_threshold = 0.5
        except ValueError:
            print("⚠️  Invalid confidence threshold. Using default value of 0.5")
            confidence_threshold = 0.5
    
    print(f"\n🎯 Using confidence threshold: {confidence_threshold*100}%")
    print("\n🚀 Starting processing...")
    print("💡 This will detect text in speech bubbles and apply background-matching effects")
    print("⏳ Processing may take some time depending on the number of images...")
    
    # Process the folder
    remover.process_folder(input_folder, output_folder, confidence_threshold)

if __name__ == "__main__":
    main() 