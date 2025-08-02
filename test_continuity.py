#!/usr/bin/env python3
"""
Test script for image continuity detection.
This script demonstrates how to use the ImageContinuityDetector class.
"""

import os
from pathlib import Path
from image_continuity_detector import ImageContinuityDetector

def test_continuity_detection():
    """Test the continuity detection with images in the input folder."""
    
    # Initialize detector with custom parameters
    detector = ImageContinuityDetector(
        crop_percentage=0.15,  # 15% of image height
        similarity_threshold=0.6  # Lower threshold for testing
    )
    
    # Get input folder path
    input_folder = "input"
    
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' not found!")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file_path in image_files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    
    image_files = sorted(unique_files)
    
    print(f"Found {len(image_files)} images in {input_folder}:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    if len(image_files) < 2:
        print("Need at least 2 images to test continuity detection!")
        return
    
    print("\n" + "="*60)
    print("TESTING IMAGE CONTINUITY DETECTION")
    print("="*60)
    
    # Test each consecutive pair
    for i in range(len(image_files) - 1):
        img1_path = str(image_files[i])
        img2_path = str(image_files[i + 1])
        
        print(f"\nPair {i+1}: {Path(img1_path).name} -> {Path(img2_path).name}")
        print("-" * 40)
        
        # Check continuity
        result = detector.check_continuity(img1_path, img2_path)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            status = "✅ CONTINUOUS" if result['is_continuous'] else "❌ NOT CONTINUOUS"
            print(f"Result: {status}")
            print(f"Weighted Score: {result['weighted_score']:.3f}")
            print(f"SSIM Score: {result['ssim_score']:.3f}")
            print(f"Histogram Score: {result['histogram_score']:.3f}")
            print(f"Template Score: {result['template_score']:.3f}")
            print(f"Threshold: {result['threshold']:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results = detector.analyze_image_sequence(input_folder)
    continuous_pairs = sum(1 for r in results if r.get('is_continuous', False))
    total_pairs = len(results)
    
    print(f"Total image pairs analyzed: {total_pairs}")
    print(f"Continuous pairs found: {continuous_pairs}")
    print(f"Non-continuous pairs: {total_pairs - continuous_pairs}")
    if total_pairs > 0:
        print(f"Continuity rate: {continuous_pairs/total_pairs*100:.1f}%")
    
    # Show continuous sequences
    if continuous_pairs > 0:
        print(f"\nContinuous sequences found:")
        for i, result in enumerate(results):
            if result.get('is_continuous', False):
                img1_name = Path(result['img1_path']).name
                img2_name = Path(result['img2_path']).name
                score = result.get('weighted_score', 0)
                print(f"  {img1_name} -> {img2_name} (Score: {score:.3f})")

def test_single_pair():
    """Test continuity detection with a specific pair of images."""
    
    detector = ImageContinuityDetector(
        crop_percentage=0.15,
        similarity_threshold=0.6
    )
    
    input_folder = "input"
    image_files = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
    image_files.sort()
    
    if len(image_files) >= 2:
        img1_path = str(image_files[0])
        img2_path = str(image_files[1])
        
        print(f"Testing single pair: {Path(img1_path).name} -> {Path(img2_path).name}")
        print("-" * 50)
        
        result = detector.check_continuity(img1_path, img2_path)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            status = "✅ CONTINUOUS" if result['is_continuous'] else "❌ NOT CONTINUOUS"
            print(f"Result: {status}")
            print(f"Weighted Score: {result['weighted_score']:.3f}")
            print(f"SSIM Score: {result['ssim_score']:.3f}")
            print(f"Histogram Score: {result['histogram_score']:.3f}")
            print(f"Template Score: {result['template_score']:.3f}")

if __name__ == "__main__":
    print("Image Continuity Detection Test")
    print("=" * 40)
    
    # Test with all images in sequence
    test_continuity_detection()
    
    print("\n" + "="*40)
    print("Single Pair Test")
    print("="*40)
    
    # Test with just the first pair
    test_single_pair() 