# Image Continuity Detection System

This system implements **lookahead with image matching** to detect if two images are continuous (part of the same larger image that was split). It compares the bottom region of one image with the top region of the next image to determine continuity.

## Features

- **Multiple Similarity Metrics**: Uses SSIM, histogram comparison, and template matching
- **Configurable Parameters**: Adjustable crop percentage and similarity threshold
- **Robust Error Handling**: Handles different image sizes and formats
- **Batch Processing**: Analyze entire folders of images
- **Detailed Results**: Provides individual scores for each similarity metric

## How It Works

1. **Region Extraction**: Crops the bottom 15% of the first image and top 15% of the second image
2. **Similarity Analysis**: Compares these regions using three different methods:
   - **SSIM (Structural Similarity Index)**: Measures structural similarity
   - **Histogram Comparison**: Compares color distributions
   - **Template Matching**: Finds how well one region matches within the other
3. **Weighted Scoring**: Combines all three scores with configurable weights
4. **Continuity Decision**: Determines if images are continuous based on a threshold

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

#### Compare Two Specific Images
```bash
python image_continuity_detector.py --input image1.jpg --second image2.jpg --threshold 0.7
```

#### Analyze All Images in a Folder
```bash
python image_continuity_detector.py --input ./input_folder --threshold 0.6
```

#### Adjust Crop Percentage
```bash
python image_continuity_detector.py --input ./input_folder --crop-percentage 0.20
```

### Python API

```python
from image_continuity_detector import ImageContinuityDetector

# Initialize detector
detector = ImageContinuityDetector(
    crop_percentage=0.15,  # 15% of image height
    similarity_threshold=0.7  # 70% similarity threshold
)

# Check continuity between two images
result = detector.check_continuity("image1.jpg", "image2.jpg")

if result['is_continuous']:
    print("Images are continuous!")
    print(f"Similarity score: {result['weighted_score']:.3f}")
else:
    print("Images are not continuous")
    print(f"Similarity score: {result['weighted_score']:.3f}")

# Analyze a sequence of images
results = detector.analyze_image_sequence("./input_folder")
```

### Test Script

Run the included test script to see the system in action:
```bash
python test_continuity.py
```

## Parameters

### Crop Percentage (`--crop-percentage`)
- **Default**: 0.15 (15%)
- **Range**: 0.05 to 0.50
- **Description**: Percentage of image height to crop from top/bottom for comparison
- **Recommendation**: 10-20% for most cases

### Similarity Threshold (`--threshold`)
- **Default**: 0.7 (70%)
- **Range**: 0.0 to 1.0
- **Description**: Minimum weighted score to consider images continuous
- **Recommendation**: 
  - 0.6-0.7 for strict matching
  - 0.4-0.5 for loose matching

## Output Format

The system returns detailed results including:

```python
{
    'is_continuous': True/False,
    'weighted_score': 0.823,  # Combined similarity score
    'ssim_score': 0.756,      # Structural similarity
    'histogram_score': 0.891, # Color histogram similarity
    'template_score': 0.723,  # Template matching score
    'threshold': 0.7,         # Used threshold
    'img1_path': 'path/to/image1.jpg',
    'img2_path': 'path/to/image2.jpg'
}
```

## Algorithm Details

### 1. SSIM (Structural Similarity Index)
- **Weight**: 50% (default)
- **Purpose**: Measures structural similarity between image regions
- **Range**: 0.0 to 1.0 (higher is better)

### 2. Histogram Comparison
- **Weight**: 30% (default)
- **Purpose**: Compares color distributions using HSV color space
- **Range**: 0.0 to 1.0 (higher is better)

### 3. Template Matching
- **Weight**: 20% (default)
- **Purpose**: Finds how well one region matches within the other
- **Range**: 0.0 to 1.0 (higher is better)

### Weighted Score Calculation
```
weighted_score = (0.5 × ssim_score) + (0.3 × histogram_score) + (0.2 × template_score)
```

## Use Cases

### 1. Document Scanning
- Detect if scanned pages are part of the same document
- Identify page breaks and continuations

### 2. Image Stitching
- Determine if images can be stitched together
- Find overlapping regions for panorama creation

### 3. Content Analysis
- Identify split images in social media content
- Detect image sequences in galleries

### 4. Quality Control
- Verify image continuity in automated processing
- Ensure proper image ordering

## Tips for Best Results

1. **Image Quality**: Higher resolution images generally give better results
2. **Overlap Amount**: 10-20% overlap typically works best
3. **Threshold Tuning**: Start with 0.6 and adjust based on your specific use case
4. **Image Formats**: Supports JPG, PNG, BMP, TIFF formats
5. **Processing Order**: Images are processed in alphabetical order

## Troubleshooting

### Common Issues

1. **"Input images must have the same dimensions"**
   - The system now handles this automatically
   - Images are resized to compatible dimensions

2. **Low similarity scores**
   - Try reducing the threshold
   - Check if images actually have overlapping content
   - Increase crop percentage if overlap is small

3. **High false positives**
   - Increase the similarity threshold
   - Reduce crop percentage
   - Check for similar backgrounds or patterns

### Performance Optimization

- For large image sets, consider processing in batches
- The system automatically handles different image sizes
- Memory usage scales with image resolution

## Example Results

```
Pair 1: 009s.jpg -> 009ss.jpg
Result: ❌ NOT CONTINUOUS
Weighted Score: 0.250
SSIM Score: 0.389
Histogram Score: 0.000
Template Score: 0.278
Threshold: 0.600

Pair 2: 02_panel_4.jpg -> 02_panel_5.jpg
Result: ❌ NOT CONTINUOUS
Weighted Score: 0.465
SSIM Score: 0.206
Histogram Score: 0.991
Template Score: 0.323
Threshold: 0.600
```

## Contributing

Feel free to contribute improvements:
- Add new similarity metrics
- Optimize performance
- Improve error handling
- Add visualization features

## License

This project is open source and available under the MIT License. 