#!/usr/bin/env python3
"""
Demo script to showcase screenshot slicing functionality.
This script uses an existing screenshot to demonstrate the slicing process.
"""

import os
import sys
from screenshot_slicer import slice_screenshot_simple, ScreenshotSlicer, get_most_relevant_slice

def demo_slicing():
    """Demonstrate the screenshot slicing functionality."""
    print("Screenshot Slicing Demo")
    print("=" * 30)
    
    # Use existing screenshot
    demo_screenshot = "screenshot.png"
    if not os.path.exists(demo_screenshot):
        demo_screenshot = "test.png"
    
    if not os.path.exists(demo_screenshot):
        print("No existing screenshot found to demo with")
        return
    
    print(f"Using existing screenshot: {demo_screenshot}")
    
    try:
        # Demo 1: Simple horizontal slicing
        print("\n1. Simple Horizontal Slicing:")
        slices = slice_screenshot_simple(demo_screenshot, mode='horizontal')
        print(f"   Created {len(slices)} horizontal slices")
        for slice_info in slices:
            print(f"   - {slice_info['name']}: {slice_info['bbox']}")
        
        # Demo 2: Full slicing with UI detection
        print("\n2. Full Slicing with UI Detection:")
        slicer = ScreenshotSlicer()
        all_slices = slicer.slice_screenshot(demo_screenshot, output_dir="demo_slices")
        print(f"   Created {len(all_slices)} total slices")
        for slice_info in all_slices:
            print(f"   - {slice_info['name']}: priority={slice_info['priority']}, area={slice_info['area']}")
        
        # Demo 3: Relevant slice selection
        print("\n3. Relevant Slice Selection:")
        relevant_slices = slicer.get_relevant_slices(all_slices, mode='auto')
        print(f"   {len(relevant_slices)} slices selected as relevant")
        
        best_slice = get_most_relevant_slice(all_slices)
        if best_slice:
            print(f"   Best slice: {best_slice['name']} (priority: {best_slice['priority']})")
        
        # Demo 4: Performance measurement
        print("\n4. Performance Test:")
        import time
        start_time = time.time()
        for i in range(5):
            test_slices = slicer.slice_screenshot(demo_screenshot)
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        print(f"   Average slicing time: {avg_time:.3f} seconds")
        print(f"   Slicing rate: {1/avg_time:.1f} slices/second")
        
        print(f"\n✓ Demo completed successfully!")
        if os.path.exists("demo_slices"):
            slice_count = len([f for f in os.listdir("demo_slices") if f.endswith('.png')])
            print(f"   {slice_count} slice files saved in demo_slices/")
        
    except Exception as e:
        print(f"Error during demo: {e}")

if __name__ == "__main__":
    demo_slicing()

if __name__ == "__main__":
    demo_slicing()