#!/usr/bin/env python3
"""
Test script for screenshot slicing functionality.
"""

import os
import sys
import cv2
import numpy as np
from screenshot_slicer import ScreenshotSlicer, slice_screenshot_simple, get_most_relevant_slice

def create_test_image(width=1920, height=1080, filename="test_screenshot.png"):
    """Create a synthetic test image that simulates a typical desktop screenshot."""
    # Create a base image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add a header (top bar)
    cv2.rectangle(image, (0, 0), (width, 60), (64, 64, 64), -1)
    cv2.putText(image, "Header/Menu Bar", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add a footer (bottom bar)
    cv2.rectangle(image, (0, height-40), (width, height), (64, 64, 64), -1)
    cv2.putText(image, "Footer/Status Bar", (20, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add a left sidebar
    cv2.rectangle(image, (0, 60), (200, height-40), (96, 96, 96), -1)
    cv2.putText(image, "Sidebar", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add content areas
    # Left content area
    cv2.rectangle(image, (220, 80), (960, height-60), (240, 240, 240), -1)
    cv2.putText(image, "Main Content Area - Left", (240, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(image, "Important text content here", (240, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "More content and information", (240, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Right content area
    cv2.rectangle(image, (980, 80), (width-20, height-60), (220, 220, 220), -1)
    cv2.putText(image, "Right Panel", (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(image, "Secondary content", (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Details and options", (1000, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save the test image
    cv2.imwrite(filename, image)
    print(f"Created test image: {filename}")
    return filename

def test_basic_slicing():
    """Test basic slicing functionality."""
    print("=== Testing Basic Slicing ===")
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        # Test simple slicing
        slices = slice_screenshot_simple(test_image_path, mode='horizontal')
        print(f"Horizontal slicing created {len(slices)} slices:")
        for slice_info in slices:
            print(f"  - {slice_info['name']}: {slice_info['bbox']}, area: {slice_info['area']}")
        
        # Test different modes
        for mode in ['vertical', 'quadrants', 'all']:
            slices = slice_screenshot_simple(test_image_path, mode=mode)
            print(f"{mode.title()} slicing created {len(slices)} slices")
        
        # Test with custom slicer
        slicer = ScreenshotSlicer()
        slices = slicer.slice_screenshot(test_image_path, output_dir="test_slices")
        print(f"Custom slicer created {len(slices)} slices and saved to test_slices/")
        
        # Test relevance detection
        relevant_slices = slicer.get_relevant_slices(slices, mode='auto')
        print(f"Relevant slices: {len(relevant_slices)}")
        
        best_slice = get_most_relevant_slice(slices)
        if best_slice:
            print(f"Best slice: {best_slice['name']} with priority {best_slice['priority']}")
        
    except Exception as e:
        print(f"Error in basic slicing test: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def test_ui_detection():
    """Test UI element detection."""
    print("\n=== Testing UI Detection ===")
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        # Test with UI detection enabled
        config = {
            'slice_modes': {'horizontal_halves': True},
            'ui_detection': {'enabled': True, 'header_height': 60, 'footer_height': 40, 'sidebar_width': 200},
            'performance': {'min_slice_area': 10000, 'max_slices_per_image': 6}
        }
        
        slicer = ScreenshotSlicer(config)
        slices = slicer.slice_screenshot(test_image_path)
        
        print(f"With UI detection: {len(slices)} slices created")
        for slice_info in slices:
            x1, y1, x2, y2 = slice_info['bbox']
            print(f"  - {slice_info['name']}: ({x1},{y1}) to ({x2},{y2})")
        
        # Test without UI detection
        config['ui_detection']['enabled'] = False
        slicer_no_ui = ScreenshotSlicer(config)
        slices_no_ui = slicer_no_ui.slice_screenshot(test_image_path)
        
        print(f"Without UI detection: {len(slices_no_ui)} slices created")
        
    except Exception as e:
        print(f"Error in UI detection test: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def test_performance():
    """Test performance with multiple slicing operations."""
    print("\n=== Testing Performance ===")
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        import time
        
        slicer = ScreenshotSlicer()
        
        # Time multiple slicing operations
        start_time = time.time()
        for i in range(10):
            slices = slicer.slice_screenshot(test_image_path)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Average slicing time: {avg_time:.3f} seconds")
        print(f"Estimated slices per second: {1/avg_time:.1f}")
        
        if avg_time < 0.1:  # Should be fast enough for real-time processing
            print("✓ Performance test passed - slicing is fast enough for real-time use")
        else:
            print("⚠ Performance test warning - slicing might be too slow for high-frequency capture")
            
    except Exception as e:
        print(f"Error in performance test: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def test_existing_screenshot():
    """Test slicing with an existing screenshot if available."""
    print("\n=== Testing with Existing Screenshot ===")
    
    # Look for existing screenshots
    screenshot_files = []
    for filename in ['screenshot.png', 'test.png']:
        if os.path.exists(filename):
            screenshot_files.append(filename)
    
    # Also check screenshots directory
    if os.path.exists('screenshots'):
        for filename in os.listdir('screenshots'):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                screenshot_files.append(os.path.join('screenshots', filename))
    
    if not screenshot_files:
        print("No existing screenshots found to test with")
        return
    
    for screenshot_path in screenshot_files[:2]:  # Test with first 2 screenshots
        try:
            print(f"Testing with: {screenshot_path}")
            
            slices = slice_screenshot_simple(screenshot_path, mode='horizontal')
            print(f"  Created {len(slices)} slices")
            
            slicer = ScreenshotSlicer()
            relevant_slices = slicer.get_relevant_slices(slices)
            print(f"  {len(relevant_slices)} relevant slices identified")
            
        except Exception as e:
            print(f"  Error processing {screenshot_path}: {e}")

def main():
    """Run all tests."""
    print("Screenshot Slicing Test Suite")
    print("=" * 40)
    
    try:
        test_basic_slicing()
        test_ui_detection()
        test_performance()
        test_existing_screenshot()
        
        print("\n" + "=" * 40)
        print("All tests completed!")
        
        # Cleanup test directory
        if os.path.exists("test_slices"):
            import shutil
            shutil.rmtree("test_slices")
            print("Cleaned up test files")
            
    except Exception as e:
        print(f"Test suite error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()