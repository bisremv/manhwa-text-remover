#!/usr/bin/env python3
"""
Comprehensive Demo of Image Continuity Detection System

This script demonstrates all the features and capabilities of the
image continuity detection system with various examples and use cases.
"""

import os
from pathlib import Path
from image_continuity_detector import ImageContinuityDetector

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_result(result, pair_name=""):
    """Print a formatted result."""
    if pair_name:
        print(f"\n{pair_name}")
        print("-" * 40)
    
    if 'error' in result:
        print(f"❌ ERROR: {result['error']}")
    else:
        status = "✅ CONTINUOUS" if result['is_continuous'] else "❌ NOT CONTINUOUS"
        print(f"Result: {status}")
        print(f"Weighted Score: {result['weighted_score']:.3f}")
        print(f"SSIM Score: {result['ssim_score']:.3f}")
        print(f"Histogram Score: {result['histogram_score']:.3f}")
        print(f"Template Score: {result['template_score']:.3f}")
        print(f"Threshold: {result['threshold']:.3f}")

def demo_basic_functionality():
    """Demonstrate basic functionality with different thresholds."""
    print_header("BASIC FUNCTIONALITY DEMO")
    
    input_folder = "input"
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' not found!")
        return
    
    # Get first two images
    image_files = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
    image_files.sort()
    
    if len(image_files) < 2:
        print("Need at least 2 images for demo!")
        return
    
    img1_path = str(image_files[0])
    img2_path = str(image_files[1])
    
    print(f"Testing with: {Path(img1_path).name} -> {Path(img2_path).name}")
    
    # Test with different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        detector = ImageContinuityDetector(similarity_threshold=threshold)
        result = detector.check_continuity(img1_path, img2_path)
        print_result(result, f"Threshold: {threshold}")

def demo_crop_percentage_impact():
    """Demonstrate how crop percentage affects results."""
    print_header("CROP PERCENTAGE IMPACT DEMO")
    
    input_folder = "input"
    image_files = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
    image_files.sort()
    
    if len(image_files) < 2:
        print("Need at least 2 images for demo!")
        return
    
    img1_path = str(image_files[0])
    img2_path = str(image_files[1])
    
    print(f"Testing with: {Path(img1_path).name} -> {Path(img2_path).name}")
    
    # Test with different crop percentages
    crop_percentages = [0.05, 0.10, 0.15, 0.20, 0.30]
    
    for crop_pct in crop_percentages:
        detector = ImageContinuityDetector(
            crop_percentage=crop_pct,
            similarity_threshold=0.5
        )
        result = detector.check_continuity(img1_path, img2_path)
        print_result(result, f"Crop Percentage: {crop_pct*100:.0f}%")

def demo_batch_analysis():
    """Demonstrate batch analysis of all images."""
    print_header("BATCH ANALYSIS DEMO")
    
    input_folder = "input"
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' not found!")
        return
    
    # Analyze with different thresholds
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n--- Analysis with threshold {threshold} ---")
        detector = ImageContinuityDetector(similarity_threshold=threshold)
        results = detector.analyze_image_sequence(input_folder)
        
        continuous_pairs = sum(1 for r in results if r.get('is_continuous', False))
        total_pairs = len(results)
        
        print(f"Total pairs: {total_pairs}")
        print(f"Continuous pairs: {continuous_pairs}")
        print(f"Continuity rate: {continuous_pairs/total_pairs*100:.1f}%" if total_pairs > 0 else "N/A")
        
        # Show continuous pairs
        if continuous_pairs > 0:
            print("Continuous sequences:")
            for result in results:
                if result.get('is_continuous', False):
                    img1_name = Path(result['img1_path']).name
                    img2_name = Path(result['img2_path']).name
                    score = result.get('weighted_score', 0)
                    print(f"  {img1_name} -> {img2_name} (Score: {score:.3f})")

def demo_individual_metrics():
    """Demonstrate how individual metrics perform."""
    print_header("INDIVIDUAL METRICS ANALYSIS")
    
    input_folder = "input"
    image_files = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
    image_files.sort()
    
    if len(image_files) < 2:
        print("Need at least 2 images for demo!")
        return
    
    img1_path = str(image_files[0])
    img2_path = str(image_files[1])
    
    print(f"Analyzing: {Path(img1_path).name} -> {Path(img2_path).name}")
    
    detector = ImageContinuityDetector(similarity_threshold=0.5)
    
    # Load and crop images
    img1 = detector.load_image(img1_path)
    img2 = detector.load_image(img2_path)
    bottom_region, top_region = detector.crop_image_regions(img1, img2)
    
    # Calculate individual metrics
    ssim_score = detector.calculate_ssim_similarity(bottom_region, top_region)
    hist_score = detector.calculate_histogram_similarity(bottom_region, top_region)
    template_score = detector.calculate_template_matching(bottom_region, top_region)
    
    print(f"\nIndividual Metric Scores:")
    print(f"SSIM Score: {ssim_score:.3f}")
    print(f"Histogram Score: {hist_score:.3f}")
    print(f"Template Score: {template_score:.3f}")
    
    # Show what each metric means
    print(f"\nMetric Interpretations:")
    print(f"SSIM ({ssim_score:.3f}): {'High structural similarity' if ssim_score > 0.5 else 'Low structural similarity'}")
    print(f"Histogram ({hist_score:.3f}): {'Similar color distribution' if hist_score > 0.5 else 'Different color distribution'}")
    print(f"Template ({template_score:.3f}): {'Good pattern matching' if template_score > 0.5 else 'Poor pattern matching'}")

def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print_header("ERROR HANDLING DEMO")
    
    detector = ImageContinuityDetector()
    
    # Test with non-existent files
    print("Testing with non-existent files:")
    result = detector.check_continuity("nonexistent1.jpg", "nonexistent2.jpg")
    print_result(result, "Non-existent files")
    
    # Test with invalid image files
    print("\nTesting with invalid image files:")
    # Create a temporary invalid file
    with open("temp_invalid.txt", "w") as f:
        f.write("This is not an image file")
    
    result = detector.check_continuity("temp_invalid.txt", "temp_invalid.txt")
    print_result(result, "Invalid image files")
    
    # Clean up
    if os.path.exists("temp_invalid.txt"):
        os.remove("temp_invalid.txt")

def main():
    """Run all demos."""
    print("Image Continuity Detection System - Comprehensive Demo")
    print("=" * 60)
    
    # Check if input folder exists
    if not os.path.exists("input"):
        print("❌ Input folder not found! Please ensure you have an 'input' folder with images.")
        return
    
    # Run all demos
    demo_basic_functionality()
    demo_crop_percentage_impact()
    demo_batch_analysis()
    demo_individual_metrics()
    demo_error_handling()
    
    print_header("DEMO COMPLETE")
    print("✅ All demonstrations completed successfully!")
    print("\nTo use the system:")
    print("1. python image_continuity_detector.py --input image1.jpg --second image2.jpg")
    print("2. python image_continuity_detector.py --input ./input_folder")
    print("3. python test_continuity.py")

if __name__ == "__main__":
    main() 