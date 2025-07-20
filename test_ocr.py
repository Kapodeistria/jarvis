#!/usr/bin/env python3
"""
Test script for enhanced OCR functionality.
"""

import os
import sys
import time
from enhanced_ocr import EnhancedOCR, extract_text_from_image, benchmark_image_ocr


def test_basic_ocr():
    """Test basic OCR functionality."""
    print("=" * 50)
    print("Testing Basic OCR Functionality")
    print("=" * 50)
    
    test_images = ["screenshot.png", "test.png"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting basic OCR on {img_path}")
            
            # Test simple extraction
            try:
                text = extract_text_from_image(img_path, enhanced=False)
                print(f"Simple OCR result ({len(text)} chars): {text[:100]}...")
                return img_path
            except Exception as e:
                print(f"Error with simple OCR: {e}")
            
            # Test enhanced extraction
            try:
                text = extract_text_from_image(img_path, enhanced=True)
                print(f"Enhanced OCR result ({len(text)} chars): {text[:100]}...")
                return img_path
            except Exception as e:
                print(f"Error with enhanced OCR: {e}")
    
    print("No test images found or OCR failed.")
    return None


def test_preprocessing_options():
    """Test different preprocessing options."""
    print("\n" + "=" * 50)
    print("Testing Preprocessing Options")
    print("=" * 50)
    
    test_images = ["screenshot.png", "test.png"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting preprocessing on {img_path}")
            
            ocr = EnhancedOCR()
            
            preprocessing_configs = [
                {"name": "No preprocessing", "config": {}},
                {"name": "Contrast enhancement", "config": {"enhance_contrast": True, "contrast_factor": 1.5}},
                {"name": "Grayscale", "config": {"grayscale": True}},
                {"name": "Binary threshold", "config": {"binary_threshold": True, "threshold_value": 127}},
                {"name": "Combined", "config": {"enhance_contrast": True, "grayscale": True, "binary_threshold": True}}
            ]
            
            for prep in preprocessing_configs:
                try:
                    result = ocr.extract_text_with_config(
                        img_path, 
                        preprocessing_config=prep["config"],
                        save_debug=True
                    )
                    print(f"  {prep['name']}: {len(result['cleaned_text'])} chars, "
                          f"confidence: {result['avg_confidence']:.1f}%, "
                          f"time: {result['processing_time']:.2f}s")
                except Exception as e:
                    print(f"  {prep['name']}: Error - {e}")
            
            return img_path
    
    print("No test images found.")
    return None


def test_psm_modes():
    """Test different PSM modes."""
    print("\n" + "=" * 50)
    print("Testing PSM Modes")
    print("=" * 50)
    
    test_images = ["screenshot.png", "test.png"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting PSM modes on {img_path}")
            
            ocr = EnhancedOCR()
            
            # Test key PSM modes
            psm_modes = [3, 6, 7, 8, 11]
            
            for psm in psm_modes:
                try:
                    result = ocr.extract_text_with_config(img_path, psm=psm, save_debug=True)
                    print(f"  PSM {psm} ({ocr.PSM_MODES[psm][:50]}...): "
                          f"{len(result['cleaned_text'])} chars, "
                          f"confidence: {result['avg_confidence']:.1f}%, "
                          f"time: {result['processing_time']:.2f}s")
                except Exception as e:
                    print(f"  PSM {psm}: Error - {e}")
            
            return img_path
    
    print("No test images found.")
    return None


def test_benchmark():
    """Test benchmarking functionality."""
    print("\n" + "=" * 50)
    print("Testing Benchmark Functionality")
    print("=" * 50)
    
    test_images = ["screenshot.png", "test.png"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nRunning benchmark on {img_path}")
            
            try:
                start_time = time.time()
                results = benchmark_image_ocr(img_path)
                end_time = time.time()
                
                print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
                print(f"Total configurations tested: {len(results['results'])}")
                
                best_config = results.get('best_config', {})
                if best_config:
                    best = best_config.get('best_by_confidence', {})
                    if best:
                        print(f"Best configuration by confidence:")
                        print(f"  PSM: {best['psm']}")
                        print(f"  Language: {best['language']}")
                        print(f"  Confidence: {best['avg_confidence']:.1f}%")
                        print(f"  Text length: {len(best['cleaned_text'])} chars")
                        print(f"  Processing time: {best['processing_time']:.2f}s")
                        print(f"  Text preview: {best['cleaned_text'][:100]}...")
                
                return img_path
            except Exception as e:
                print(f"Benchmark error: {e}")
    
    print("No test images found.")
    return None


def create_test_image():
    """Create a simple test image with text for testing."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image with text
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        
        try:
            # Try to use a better font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw some test text
        test_text = "This is a test image\nfor OCR accuracy testing.\nNumbers: 123456\nSpecial chars: @#$%"
        d.multiline_text((10, 10), test_text, fill='black', font=font)
        
        img.save('test_ocr_image.png')
        print("Created test image: test_ocr_image.png")
        return 'test_ocr_image.png'
    except Exception as e:
        print(f"Could not create test image: {e}")
        return None


def main():
    """Run all OCR tests."""
    print("Enhanced OCR Test Suite")
    print("=====================")
    
    # Check if we have test images
    test_images = ["screenshot.png", "test.png"]
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if not available_images:
        print("No test images found. Creating a test image...")
        test_img = create_test_image()
        if test_img:
            available_images = [test_img]
    
    if not available_images:
        print("No images available for testing. Please ensure you have:")
        print("- screenshot.png")
        print("- test.png")
        print("Or ensure PIL/Pillow is available to create a test image.")
        return
    
    print(f"Found test images: {', '.join(available_images)}")
    
    # Run tests
    test_basic_ocr()
    test_preprocessing_options()
    test_psm_modes()
    test_benchmark()
    
    print("\n" + "=" * 50)
    print("OCR Tests Completed")
    print("=" * 50)
    print("Check the following directories for debug files:")
    print("- ocr_debug/: Debug images and JSON files")
    print("- ocr_benchmarks/: Benchmark results")


if __name__ == "__main__":
    main()