#!/usr/bin/env python3
"""
Image Merger - Automatically merge continuous images and delete originals

This script extends the image continuity detection system to automatically
merge continuous images vertically and clean up the original files.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from image_continuity_detector import ImageContinuityDetector
import shutil

class ImageMerger:
    def __init__(self, crop_percentage: float = 0.15, similarity_threshold: float = 0.5):
        """
        Initialize the image merger.
        
        Args:
            crop_percentage: Percentage of image to crop for comparison
            similarity_threshold: Threshold for determining continuity
        """
        self.detector = ImageContinuityDetector(crop_percentage, similarity_threshold)
        self.merged_count = 0
        self.deleted_count = 0
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image and convert to RGB."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def save_image(self, image: np.ndarray, output_path: str):
        """Save an image in BGR format."""
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_image)
    
    def merge_images_vertically(self, img1: np.ndarray, img2: np.ndarray, overlap_region: int = 0) -> np.ndarray:
        """
        Merge two images vertically with optional overlap removal.
        
        Args:
            img1: First image (top)
            img2: Second image (bottom)
            overlap_region: Number of pixels to remove from overlap region
            
        Returns:
            Merged image
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use the wider width
        target_width = max(w1, w2)
        
        # Resize images to same width
        if w1 != target_width:
            img1 = cv2.resize(img1, (target_width, int(h1 * target_width / w1)))
        if w2 != target_width:
            img2 = cv2.resize(img2, (target_width, int(h2 * target_width / w2)))
        
        # Update heights after resizing
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Remove overlap region if specified
        if overlap_region > 0:
            # Remove bottom portion of img1 and top portion of img2
            img1 = img1[:-overlap_region, :]
            img2 = img2[overlap_region:, :]
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
        
        # Create merged image
        merged_height = h1 + h2
        merged_image = np.zeros((merged_height, target_width, 3), dtype=np.uint8)
        
        # Copy images
        merged_image[:h1, :] = img1
        merged_image[h1:, :] = img2
        
        return merged_image
    
    def calculate_overlap_region(self, img1_path: str, img2_path: str) -> int:
        """
        Calculate the optimal overlap region to remove based on similarity.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            Number of pixels to remove from overlap region
        """
        try:
            img1 = self.load_image(img1_path)
            img2 = self.load_image(img2_path)
            
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Calculate crop percentage in pixels
            crop_h1 = int(h1 * self.detector.crop_percentage)
            crop_h2 = int(h2 * self.detector.crop_percentage)
            
            # Return the smaller of the two crop regions
            return min(crop_h1, crop_h2)
            
        except Exception as e:
            print(f"Error calculating overlap: {e}")
            return 0
    
    def merge_continuous_pairs(self, image_folder: str, output_folder: str = None, delete_originals: bool = True) -> List[Dict]:
        """
        Find and merge continuous image pairs.
        
        Args:
            image_folder: Folder containing images to process
            output_folder: Folder to save merged images (default: same as input)
            delete_originals: Whether to delete original images after merging
            
        Returns:
            List of merge results
        """
        if output_folder is None:
            output_folder = image_folder
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in image_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)
        
        image_files = sorted(unique_files)
        
        if len(image_files) < 2:
            print(f"Found {len(image_files)} images. Need at least 2 for merging.")
            return []
        
        print(f"Found {len(image_files)} images to process...")
        
        # Analyze continuity
        results = self.detector.analyze_image_sequence(image_folder)
        
        # Track which images have been processed
        processed_images = set()
        merge_results = []
        
        # Process continuous pairs
        for i, result in enumerate(results):
            if result.get('is_continuous', False) and 'error' not in result:
                img1_path = result['img1_path']
                img2_path = result['img2_path']
                
                # Skip if either image has already been processed
                if img1_path in processed_images or img2_path in processed_images:
                    continue
                
                try:
                    print(f"\nMerging: {Path(img1_path).name} + {Path(img2_path).name}")
                    
                    # Load images
                    img1 = self.load_image(img1_path)
                    img2 = self.load_image(img2_path)
                    
                    # Calculate overlap region
                    overlap_pixels = self.calculate_overlap_region(img1_path, img2_path)
                    
                    # Merge images
                    merged_image = self.merge_images_vertically(img1, img2, overlap_pixels)
                    
                    # Generate output filename
                    img1_name = Path(img1_path).stem
                    img2_name = Path(img2_path).stem
                    output_name = f"{img1_name}_merged_{img2_name}.jpg"
                    output_path = os.path.join(output_folder, output_name)
                    
                    # Save merged image
                    self.save_image(merged_image, output_path)
                    
                    # Delete original images if requested
                    if delete_originals:
                        os.remove(img1_path)
                        os.remove(img2_path)
                        self.deleted_count += 2
                        print(f"  Deleted originals: {Path(img1_path).name}, {Path(img2_path).name}")
                    
                    # Mark as processed
                    processed_images.add(img1_path)
                    processed_images.add(img2_path)
                    
                    self.merged_count += 1
                    
                    merge_result = {
                        'merged': True,
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'output_path': output_path,
                        'similarity_score': result.get('weighted_score', 0),
                        'overlap_pixels': overlap_pixels
                    }
                    merge_results.append(merge_result)
                    
                    print(f"  ✅ Merged successfully: {output_name}")
                    print(f"  Similarity score: {result.get('weighted_score', 0):.3f}")
                    print(f"  Overlap removed: {overlap_pixels} pixels")
                    
                except Exception as e:
                    print(f"  ❌ Error merging {Path(img1_path).name} + {Path(img2_path).name}: {e}")
                    merge_result = {
                        'merged': False,
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'error': str(e)
                    }
                    merge_results.append(merge_result)
        
        return merge_results
    
    def get_summary(self) -> Dict:
        """Get summary of merge operations."""
        return {
            'merged_pairs': self.merged_count,
            'deleted_files': self.deleted_count,
            'total_operations': self.merged_count * 2
        }

def main():
    parser = argparse.ArgumentParser(description='Merge continuous images and delete originals')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing images')
    parser.add_argument('--output', '-o', help='Output folder for merged images (default: same as input)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Similarity threshold (0.0-1.0)')
    parser.add_argument('--crop-percentage', '-c', type=float, default=0.15, help='Percentage to crop from top/bottom')
    parser.add_argument('--keep-originals', action='store_true', help='Keep original images (default: delete them)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be merged without actually doing it')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist!")
        return
    
    print("Image Merger - Continuous Image Detection and Merging")
    print("=" * 60)
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output or args.input}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Crop percentage: {args.crop_percentage}")
    print(f"Delete originals: {not args.keep_originals}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 60)
    
    if args.dry_run:
        # Just analyze without merging
        detector = ImageContinuityDetector(args.crop_percentage, args.threshold)
        results = detector.analyze_image_sequence(args.input)
        
        continuous_pairs = [r for r in results if r.get('is_continuous', False)]
        
        print(f"\nDRY RUN RESULTS:")
        print(f"Total image pairs: {len(results)}")
        print(f"Continuous pairs: {len(continuous_pairs)}")
        
        if continuous_pairs:
            print("\nWould merge these pairs:")
            for i, result in enumerate(continuous_pairs, 1):
                img1_name = Path(result['img1_path']).name
                img2_name = Path(result['img2_path']).name
                score = result.get('weighted_score', 0)
                print(f"  {i}. {img1_name} + {img2_name} (Score: {score:.3f})")
        else:
            print("No continuous pairs found to merge.")
        
        return
    
    # Perform actual merging
    merger = ImageMerger(args.crop_percentage, args.threshold)
    
    try:
        results = merger.merge_continuous_pairs(
            args.input, 
            args.output, 
            delete_originals=not args.keep_originals
        )
        
        # Summary
        summary = merger.get_summary()
        
        print("\n" + "=" * 60)
        print("MERGE SUMMARY")
        print("=" * 60)
        print(f"Successfully merged pairs: {summary['merged_pairs']}")
        print(f"Original files deleted: {summary['deleted_files']}")
        print(f"Total operations: {summary['total_operations']}")
        
        if results:
            print(f"\nMerge details:")
            for i, result in enumerate(results, 1):
                if result['merged']:
                    output_name = Path(result['output_path']).name
                    score = result.get('similarity_score', 0)
                    overlap = result.get('overlap_pixels', 0)
                    print(f"  {i}. ✅ {output_name} (Score: {score:.3f}, Overlap: {overlap}px)")
                else:
                    img1_name = Path(result['img1_path']).name
                    img2_name = Path(result['img2_path']).name
                    error = result.get('error', 'Unknown error')
                    print(f"  {i}. ❌ {img1_name} + {img2_name} - {error}")
        
        print(f"\n✅ Merge operation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during merge operation: {e}")

if __name__ == "__main__":
    main() 