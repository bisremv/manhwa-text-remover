import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class ImageContinuityDetector:
    def __init__(self, crop_percentage: float = 0.15, similarity_threshold: float = 0.7, white_threshold: float = 0.9):
        self.crop_percentage = crop_percentage
        self.similarity_threshold = similarity_threshold
        self.white_threshold = white_threshold  # Threshold for considering a pixel as "white"
        
    def load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def crop_image_regions(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        crop_h1 = int(h1 * self.crop_percentage)
        crop_h2 = int(h2 * self.crop_percentage)
        
        bottom_region = img1[h1 - crop_h1:, :]
        top_region = img2[:crop_h2, :]
        
        return bottom_region, top_region
    
    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            target_width = min(w1, w2)
            
            if target_width <= 0:
                # Fallback to a reasonable size
                target_width = 100
            
            img1_resized = cv2.resize(img1, (target_width, int(h1 * target_width / w1)))
            img2_resized = cv2.resize(img2, (target_width, int(h2 * target_width / w2)))
            
            return img1_resized, img2_resized
        except Exception as e:
            print(f"Resize error: {e}")
            # Fallback: resize both to a standard size
            return cv2.resize(img1, (100, 100)), cv2.resize(img2, (100, 100))
    
    def calculate_white_background_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate similarity based on white background detection.
        Returns high similarity if both regions have predominantly white backgrounds.
        """
        try:
            # Convert to grayscale for white detection
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = img1
                
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = img2
            
            # Resize to same size for comparison
            gray1, gray2 = self.resize_to_match(gray1, gray2)
            
            # Calculate percentage of white pixels in each image
            # White pixels are those with high intensity (close to 255)
            white_threshold_value = int(255 * self.white_threshold)
            
            white_pixels_1 = np.sum(gray1 >= white_threshold_value)
            white_pixels_2 = np.sum(gray2 >= white_threshold_value)
            
            total_pixels_1 = gray1.size
            total_pixels_2 = gray2.size
            
            white_percentage_1 = white_pixels_1 / total_pixels_1
            white_percentage_2 = white_pixels_2 / total_pixels_2
            
            # If both regions are predominantly white, they are similar
            # The similarity is based on how close their white percentages are
            if white_percentage_1 >= 0.8 and white_percentage_2 >= 0.8:
                # Both are white backgrounds, calculate similarity based on how close their percentages are
                similarity = 1.0 - abs(white_percentage_1 - white_percentage_2)
                return max(0.8, similarity)  # Minimum 0.8 if both are white
            elif white_percentage_1 < 0.3 and white_percentage_2 < 0.3:
                # Both are non-white backgrounds, calculate similarity based on how close their percentages are
                similarity = 1.0 - abs(white_percentage_1 - white_percentage_2)
                return max(0.6, similarity)  # Lower base score for non-white backgrounds
            else:
                # One is white, one is not - low similarity
                return 0.2
            
        except Exception as e:
            print(f"White background similarity error: {e}")
            return 0.0
    
    def calculate_white_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate histogram similarity focusing on white/light regions.
        """
        try:
            # Resize images to same size for histogram comparison
            img1, img2 = self.resize_to_match(img1, img2)
            
            # Convert to HSV for better white detection
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            
            # Focus on saturation and value channels for white detection
            # White regions have low saturation and high value
            sat1 = hsv1[:, :, 1]
            val1 = hsv1[:, :, 2]
            sat2 = hsv2[:, :, 1]
            val2 = hsv2[:, :, 2]
            
            # Calculate histograms for saturation and value
            sat_hist1 = cv2.calcHist([sat1], [0], None, [256], [0, 256])
            val_hist1 = cv2.calcHist([val1], [0], None, [256], [0, 256])
            sat_hist2 = cv2.calcHist([sat2], [0], None, [256], [0, 256])
            val_hist2 = cv2.calcHist([val2], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(sat_hist1, sat_hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(val_hist1, val_hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(sat_hist2, sat_hist2, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(val_hist2, val_hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Calculate similarity for both channels
            sat_similarity = cv2.compareHist(sat_hist1, sat_hist2, cv2.HISTCMP_CORREL)
            val_similarity = cv2.compareHist(val_hist1, val_hist2, cv2.HISTCMP_CORREL)
            
            # Weight value channel more heavily for white detection
            weighted_similarity = 0.3 * sat_similarity + 0.7 * val_similarity
            return max(0, weighted_similarity)
            
        except Exception as e:
            print(f"White histogram comparison error: {e}")
            return 0.0
    
    def calculate_white_edge_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate similarity based on edge detection in white regions.
        """
        try:
            img1, img2 = self.resize_to_match(img1, img2)
            
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = img1
                
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = img2
            
            # Create white masks (regions that are predominantly white)
            white_threshold_value = int(255 * self.white_threshold)
            white_mask1 = gray1 >= white_threshold_value
            white_mask2 = gray2 >= white_threshold_value
            
            # Calculate edge density in white regions
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            # Count edges in white regions
            white_edges1 = np.sum(edges1 > 0 & white_mask1)
            white_edges2 = np.sum(edges2 > 0 & white_mask2)
            
            # Calculate edge density
            white_pixels1 = np.sum(white_mask1)
            white_pixels2 = np.sum(white_mask2)
            
            if white_pixels1 == 0 or white_pixels2 == 0:
                return 0.5  # Neutral score if no white regions
            
            edge_density1 = white_edges1 / white_pixels1
            edge_density2 = white_edges2 / white_pixels2
            
            # Similarity based on how close the edge densities are
            similarity = 1.0 - abs(edge_density1 - edge_density2)
            return max(0, similarity)
            
        except Exception as e:
            print(f"White edge similarity error: {e}")
            return 0.0
    
    def check_continuity(self, img1_path: str, img2_path: str) -> Dict[str, any]:
        try:
            img1 = self.load_image(img1_path)
            img2 = self.load_image(img2_path)
            
            bottom_region, top_region = self.crop_image_regions(img1, img2)
            
            white_bg_score = self.calculate_white_background_similarity(bottom_region, top_region)
            white_hist_score = self.calculate_white_histogram_similarity(bottom_region, top_region)
            white_edge_score = self.calculate_white_edge_similarity(bottom_region, top_region)
            
            weights = [0.5, 0.3, 0.2]
            scores = [white_bg_score, white_hist_score, white_edge_score]
            weighted_score = sum(w * s for w, s in zip(weights, scores))
            
            is_continuous = weighted_score >= self.similarity_threshold
            
            return {
                'is_continuous': is_continuous,
                'weighted_score': weighted_score,
                'white_bg_score': white_bg_score,
                'white_histogram_score': white_hist_score,
                'white_edge_score': white_edge_score,
                'threshold': self.similarity_threshold,
                'img1_path': img1_path,
                'img2_path': img2_path
            }
            
        except Exception as e:
            return {
                'is_continuous': False,
                'error': str(e),
                'img1_path': img1_path,
                'img2_path': img2_path
            }
    
    def analyze_image_sequence(self, image_folder: str) -> List[Dict[str, any]]:
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
            print(f"Found {len(image_files)} images. Need at least 2 for continuity analysis.")
            return []
        
        results = []
        print(f"Analyzing {len(image_files)} images for white background continuity...")
        
        for i in range(len(image_files) - 1):
            img1_path = str(image_files[i])
            img2_path = str(image_files[i + 1])
            
            print(f"Comparing {Path(img1_path).name} -> {Path(img2_path).name}")
            result = self.check_continuity(img1_path, img2_path)
            results.append(result)
            
            status = "CONTINUOUS" if result['is_continuous'] else "NOT CONTINUOUS"
            if 'weighted_score' in result:
                print(f"  Result: {status} (Score: {result['weighted_score']:.3f})")
            else:
                print(f"  Result: ERROR - {result.get('error', 'Unknown error')}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Detect image continuity using white background matching')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing images or path to first image')
    parser.add_argument('--second', '-s', help='Path to second image (if comparing just two images)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, help='Similarity threshold (0.0-1.0)')
    parser.add_argument('--crop-percentage', '-c', type=float, default=0.15, help='Percentage to crop from top/bottom')
    parser.add_argument('--white-threshold', '-w', type=float, default=0.9, help='Threshold for white pixel detection (0.0-1.0)')
    
    args = parser.parse_args()
    
    detector = ImageContinuityDetector(
        crop_percentage=args.crop_percentage,
        similarity_threshold=args.threshold,
        white_threshold=args.white_threshold
    )
    
    if args.second:
        print(f"Comparing two images for white background continuity:")
        print(f"  Image 1: {args.input}")
        print(f"  Image 2: {args.second}")
        print(f"  Threshold: {args.threshold}")
        print(f"  White threshold: {args.white_threshold}")
        print("-" * 50)
        
        result = detector.check_continuity(args.input, args.second)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            status = "CONTINUOUS" if result['is_continuous'] else "NOT CONTINUOUS"
            print(f"Result: {status}")
            print(f"Weighted Score: {result['weighted_score']:.3f}")
            print(f"White Background Score: {result['white_bg_score']:.3f}")
            print(f"White Histogram Score: {result['white_histogram_score']:.3f}")
            print(f"White Edge Score: {result['white_edge_score']:.3f}")
    
    else:
        print(f"Analyzing image sequence for white background continuity in: {args.input}")
        print(f"Threshold: {args.threshold}")
        print(f"White threshold: {args.white_threshold}")
        print("-" * 50)
        
        results = detector.analyze_image_sequence(args.input)
        
        continuous_pairs = sum(1 for r in results if r.get('is_continuous', False))
        total_pairs = len(results)
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total image pairs analyzed: {total_pairs}")
        print(f"Continuous pairs found: {continuous_pairs}")
        print(f"Non-continuous pairs: {total_pairs - continuous_pairs}")
        print(f"Continuity rate: {continuous_pairs/total_pairs*100:.1f}%" if total_pairs > 0 else "N/A")

if __name__ == "__main__":
    main() 