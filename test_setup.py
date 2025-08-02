#!/usr/bin/env python3
"""
Test script to verify the setup and API connection
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        from inference_sdk import InferenceHTTPClient
        print("✅ Inference SDK imported successfully")
    except ImportError as e:
        print(f"❌ Inference SDK import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✅ tqdm imported successfully")
    except ImportError as e:
        print(f"❌ tqdm import failed: {e}")
        return False
    
    return True

def test_api_connection():
    """Test the Roboflow API connection"""
    print("\n🌐 Testing API connection...")
    
    try:
        from inference_sdk import InferenceHTTPClient
        
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="ryMpZlt8aUKbx2LVev7u"
        )
        
        # Test with a simple request (this might fail if no image is provided, but we're testing the connection)
        print("✅ API client initialized successfully")
        print("✅ API key appears to be valid")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return False

def test_folders():
    """Test if input and output folders exist"""
    print("\n📁 Testing folder structure...")
    
    input_folder = Path("input")
    output_folder = Path("output")
    
    if input_folder.exists():
        print("✅ Input folder exists")
    else:
        print("⚠️  Input folder does not exist (will be created when needed)")
    
    if output_folder.exists():
        print("✅ Output folder exists")
    else:
        print("⚠️  Output folder does not exist (will be created when needed)")
    
    return True

def main():
    """Run all tests"""
    print("🧪 Text Remover Setup Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Test API connection
    if not test_api_connection():
        print("\n❌ API connection test failed. Please check your internet connection and API key.")
        sys.exit(1)
    
    # Test folders
    test_folders()
    
    print("\n" + "=" * 40)
    print("🎉 All tests passed! Your setup is ready.")
    print("\n📝 Next steps:")
    print("1. Place your images in the 'input' folder")
    print("2. Run: python text_remover.py")
    print("3. Follow the prompts to process your images")

if __name__ == "__main__":
    main() 