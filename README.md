# 🎨 Text Remover - Webtoon Speech Bubble Processor

A Python script that automatically detects and removes text from speech bubbles in webtoon images using AI-powered text detection and background-matching effects.

## ✨ Features

- **AI-Powered Text Detection**: Uses Roboflow's webtoon-detection model to identify text regions
- **Background-Matching Effects**: Applies natural-looking effects that blend with the surrounding background
- **Batch Processing**: Processes entire folders and subfolders recursively
- **Progress Tracking**: Real-time progress bar with detailed statistics
- **Folder Structure Preservation**: Maintains the same folder structure in output as input
- **Multiple Image Formats**: Supports JPG, PNG, BMP, TIFF, and WebP formats

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. **Activate your virtual environment** (if using one):
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Place your images** in the `input` folder (or any folder of your choice)

2. **Run the script**:
   ```bash
   python text_remover.py
   ```

3. **Follow the prompts**:
   - Enter input folder path (or press Enter for default 'input')
   - Enter output folder path (or press Enter for default 'output')

4. **Wait for processing** - The script will show real-time progress

## 📁 Folder Structure

```
text-remove/
├── input/           # Place your images here
│   ├── image1.jpg
│   ├── subfolder/
│   │   └── image2.png
│   └── ...
├── output/          # Processed images will be saved here
│   ├── image1.jpg
│   ├── subfolder/
│   │   └── image2.png
│   └── ...
├── text_remover.py  # Main script
├── requirements.txt # Dependencies
└── README.md        # This file
```

## 🔧 How It Works

1. **Image Scanning**: Recursively scans the input folder for supported image files
2. **Text Detection**: Uses Roboflow API to detect text regions in speech bubbles
3. **Background Analysis**: Samples surrounding pixels to determine background colors
4. **Effect Application**: Applies gradient effects that blend with the background
5. **Output Generation**: Saves processed images maintaining the original folder structure

## 🎯 Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

## 📊 Progress Information

The script provides detailed progress information including:
- Total number of images found
- Current image being processed
- Number of text regions detected
- Success/error status
- Final summary with statistics

## ⚠️ Important Notes

- **API Key**: The script uses a pre-configured Roboflow API key. For production use, consider using your own API key
- **Processing Time**: Processing time depends on the number of images and API response times
- **Internet Connection**: Requires internet connection for the Roboflow API
- **Image Quality**: Higher quality images generally produce better results

## 🛠️ Troubleshooting

### Common Issues

1. **"Input folder does not exist"**: Make sure the input folder path is correct
2. **"No image files found"**: Ensure your images are in supported formats
3. **API errors**: Check your internet connection and API key validity
4. **Memory issues**: Process images in smaller batches for large folders

### Error Messages

- ⚠️ Warning messages indicate non-critical issues
- ❌ Error messages indicate processing failures
- ✅ Success messages confirm successful operations

## 📈 Performance Tips

- Process images in batches for large collections
- Use SSD storage for faster I/O operations
- Ensure stable internet connection for API calls
- Close other applications to free up memory

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License. 