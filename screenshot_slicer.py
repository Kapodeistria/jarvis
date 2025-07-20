"""
Screenshot Slicing Module for Efficient Processing

This module provides functionality to slice screenshots into meaningful regions
for more efficient and targeted image analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from PIL import Image


class ScreenshotSlicer:
    """
    A class to handle screenshot slicing for efficient processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ScreenshotSlicer with configuration.
        
        Args:
            config: Configuration dictionary with slicing parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for screenshot slicing."""
        return {
            'slice_modes': {
                'horizontal_halves': True,  # Left/Right halves
                'vertical_halves': True,    # Top/Bottom halves
                'quadrants': False,         # Four quadrants
                'custom_regions': []        # Custom defined regions
            },
            'ui_detection': {
                'enabled': True,
                'header_height': 60,        # Typical header height in pixels
                'footer_height': 40,        # Typical footer height in pixels
                'sidebar_width': 200        # Typical sidebar width in pixels
            },
            'performance': {
                'min_slice_area': 10000,    # Minimum area in pixels for a slice
                'max_slices_per_image': 6   # Maximum number of slices per image
            }
        }
    
    def slice_screenshot(self, image_path: str, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Slice a screenshot into predefined regions.
        
        Args:
            image_path: Path to the input screenshot
            output_dir: Directory to save sliced images (optional)
            
        Returns:
            List of dictionaries containing slice information
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            height, width = image.shape[:2]
            slices = []
            
            # Detect UI elements to avoid slicing
            ui_regions = self._detect_ui_elements(image) if self.config['ui_detection']['enabled'] else []
            
            # Generate slice regions based on configuration
            slice_regions = self._generate_slice_regions(width, height, ui_regions)
            
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Process each slice region
            for i, region in enumerate(slice_regions):
                slice_info = self._create_slice(image, region, i, image_path, output_dir)
                if slice_info:
                    slices.append(slice_info)
            
            return slices
            
        except Exception as e:
            print(f"Error slicing screenshot {image_path}: {e}")
            return []
    
    def _detect_ui_elements(self, image: np.ndarray) -> List[Dict]:
        """
        Detect common UI elements that shouldn't be sliced.
        
        Args:
            image: The input image as numpy array
            
        Returns:
            List of dictionaries describing UI regions
        """
        height, width = image.shape[:2]
        ui_regions = []
        
        # Detect potential header (top region with consistent color/pattern)
        header_height = self.config['ui_detection']['header_height']
        if header_height < height:
            header_region = image[0:header_height, :]
            if self._is_ui_element(header_region):
                ui_regions.append({
                    'type': 'header',
                    'bbox': (0, 0, width, header_height),
                    'importance': 'low'  # Headers typically contain less dynamic content
                })
        
        # Detect potential footer (bottom region)
        footer_height = self.config['ui_detection']['footer_height']
        if footer_height < height:
            footer_region = image[height-footer_height:height, :]
            if self._is_ui_element(footer_region):
                ui_regions.append({
                    'type': 'footer',
                    'bbox': (0, height-footer_height, width, height),
                    'importance': 'low'
                })
        
        # Detect potential sidebar (consistent vertical region)
        sidebar_width = self.config['ui_detection']['sidebar_width']
        if sidebar_width < width:
            # Check left sidebar
            left_sidebar = image[:, 0:sidebar_width]
            if self._is_ui_element(left_sidebar):
                ui_regions.append({
                    'type': 'left_sidebar',
                    'bbox': (0, 0, sidebar_width, height),
                    'importance': 'medium'
                })
            
            # Check right sidebar
            right_sidebar = image[:, width-sidebar_width:width]
            if self._is_ui_element(right_sidebar):
                ui_regions.append({
                    'type': 'right_sidebar',
                    'bbox': (width-sidebar_width, 0, width, height),
                    'importance': 'medium'
                })
        
        return ui_regions
    
    def _is_ui_element(self, region: np.ndarray) -> bool:
        """
        Determine if a region looks like a UI element.
        
        Args:
            region: Image region to analyze
            
        Returns:
            True if region appears to be a UI element
        """
        # Check for consistent colors (typical of UI elements)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Calculate standard deviation - UI elements often have low variance
        std_dev = np.std(gray)
        
        # Check for horizontal/vertical lines (common in UI)
        edges = cv2.Canny(gray, 50, 150)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        line_density = (np.sum(horizontal_lines) + np.sum(vertical_lines)) / (region.shape[0] * region.shape[1])
        
        # Simple heuristic: low color variance + presence of lines = likely UI element
        return std_dev < 30 and line_density > 0.01
    
    def _generate_slice_regions(self, width: int, height: int, ui_regions: List[Dict]) -> List[Dict]:
        """
        Generate slice regions based on configuration and detected UI elements.
        
        Args:
            width: Image width
            height: Image height
            ui_regions: List of detected UI regions
            
        Returns:
            List of slice region definitions
        """
        regions = []
        
        # Get available area (excluding UI elements with low importance)
        available_area = self._get_available_area(width, height, ui_regions)
        x_start, y_start, x_end, y_end = available_area
        
        slice_modes = self.config['slice_modes']
        
        # Horizontal halves (left/right)
        if slice_modes.get('horizontal_halves', False):
            mid_x = x_start + (x_end - x_start) // 2
            
            regions.append({
                'name': 'left_half',
                'bbox': (x_start, y_start, mid_x, y_end),
                'priority': 'high'
            })
            
            regions.append({
                'name': 'right_half', 
                'bbox': (mid_x, y_start, x_end, y_end),
                'priority': 'high'
            })
        
        # Vertical halves (top/bottom)
        if slice_modes.get('vertical_halves', False):
            mid_y = y_start + (y_end - y_start) // 2
            
            regions.append({
                'name': 'top_half',
                'bbox': (x_start, y_start, x_end, mid_y),
                'priority': 'medium'
            })
            
            regions.append({
                'name': 'bottom_half',
                'bbox': (x_start, mid_y, x_end, y_end),
                'priority': 'medium'
            })
        
        # Quadrants
        if slice_modes.get('quadrants', False):
            mid_x = x_start + (x_end - x_start) // 2
            mid_y = y_start + (y_end - y_start) // 2
            
            quadrants = [
                ('top_left', (x_start, y_start, mid_x, mid_y)),
                ('top_right', (mid_x, y_start, x_end, mid_y)),
                ('bottom_left', (x_start, mid_y, mid_x, y_end)),
                ('bottom_right', (mid_x, mid_y, x_end, y_end))
            ]
            
            for name, bbox in quadrants:
                regions.append({
                    'name': name,
                    'bbox': bbox,
                    'priority': 'low'
                })
        
        # Custom regions
        for custom_region in slice_modes.get('custom_regions', []):
            regions.append({
                'name': custom_region.get('name', 'custom'),
                'bbox': custom_region['bbox'],
                'priority': custom_region.get('priority', 'medium')
            })
        
        # Filter regions by minimum area
        min_area = self.config['performance']['min_slice_area']
        filtered_regions = []
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                filtered_regions.append(region)
        
        # Limit number of slices
        max_slices = self.config['performance']['max_slices_per_image']
        if len(filtered_regions) > max_slices:
            # Sort by priority and take top slices
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            filtered_regions.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
            filtered_regions = filtered_regions[:max_slices]
        
        return filtered_regions
    
    def _get_available_area(self, width: int, height: int, ui_regions: List[Dict]) -> Tuple[int, int, int, int]:
        """
        Calculate the available area for slicing, excluding low-importance UI elements.
        
        Args:
            width: Image width
            height: Image height
            ui_regions: List of detected UI regions
            
        Returns:
            Tuple of (x_start, y_start, x_end, y_end) for available area
        """
        x_start, y_start, x_end, y_end = 0, 0, width, height
        
        for ui_region in ui_regions:
            if ui_region.get('importance') == 'low':
                ui_x1, ui_y1, ui_x2, ui_y2 = ui_region['bbox']
                
                # Adjust available area based on UI element position
                if ui_region['type'] == 'header':
                    y_start = max(y_start, ui_y2)
                elif ui_region['type'] == 'footer':
                    y_end = min(y_end, ui_y1)
                elif ui_region['type'] == 'left_sidebar':
                    x_start = max(x_start, ui_x2)
                elif ui_region['type'] == 'right_sidebar':
                    x_end = min(x_end, ui_x1)
        
        return x_start, y_start, x_end, y_end
    
    def _create_slice(self, image: np.ndarray, region: Dict, index: int, 
                     original_path: str, output_dir: Optional[str]) -> Optional[Dict]:
        """
        Create a slice from the image based on the region definition.
        
        Args:
            image: Original image
            region: Region definition
            index: Slice index
            original_path: Path to original image
            output_dir: Output directory for sliced images
            
        Returns:
            Dictionary with slice information or None if failed
        """
        try:
            x1, y1, x2, y2 = region['bbox']
            
            # Extract the slice
            slice_image = image[y1:y2, x1:x2]
            
            # Create slice info
            slice_info = {
                'name': region['name'],
                'bbox': region['bbox'],
                'priority': region['priority'],
                'area': (x2 - x1) * (y2 - y1),
                'original_path': original_path,
                'slice_index': index
            }
            
            # Save slice if output directory is specified
            if output_dir:
                base_name = os.path.splitext(os.path.basename(original_path))[0]
                slice_filename = f"{base_name}_slice_{index}_{region['name']}.png"
                slice_path = os.path.join(output_dir, slice_filename)
                
                cv2.imwrite(slice_path, slice_image)
                slice_info['slice_path'] = slice_path
            else:
                # Store slice data in memory
                slice_info['slice_data'] = slice_image
            
            return slice_info
            
        except Exception as e:
            print(f"Error creating slice {index}: {e}")
            return None
    
    def get_relevant_slices(self, slices: List[Dict], mode: str = 'auto') -> List[Dict]:
        """
        Filter slices to get only the most relevant ones for processing.
        
        Args:
            slices: List of all slices
            mode: Selection mode ('auto', 'high_priority', 'all')
            
        Returns:
            Filtered list of relevant slices
        """
        if mode == 'all':
            return slices
        elif mode == 'high_priority':
            return [s for s in slices if s['priority'] == 'high']
        else:  # auto mode
            # Intelligent selection based on content and priority
            relevant_slices = []
            
            # Always include high priority slices
            high_priority = [s for s in slices if s['priority'] == 'high']
            relevant_slices.extend(high_priority)
            
            # Add medium priority slices if we have room
            max_slices = self.config['performance']['max_slices_per_image']
            remaining_slots = max_slices - len(relevant_slices)
            
            if remaining_slots > 0:
                medium_priority = [s for s in slices if s['priority'] == 'medium']
                relevant_slices.extend(medium_priority[:remaining_slots])
            
            return relevant_slices


def create_slicer_from_config(config_path: Optional[str] = None) -> ScreenshotSlicer:
    """
    Create a ScreenshotSlicer instance from a configuration file.
    
    Args:
        config_path: Path to configuration file (JSON)
        
    Returns:
        ScreenshotSlicer instance
    """
    config = None
    
    if config_path and os.path.exists(config_path):
        import json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
    
    return ScreenshotSlicer(config)


# Convenience functions for easy integration
def slice_screenshot_simple(image_path: str, mode: str = 'horizontal') -> List[Dict]:
    """
    Simple function to slice a screenshot with predefined modes.
    
    Args:
        image_path: Path to screenshot
        mode: Slicing mode ('horizontal', 'vertical', 'quadrants', 'all')
        
    Returns:
        List of slice information
    """
    config = {
        'slice_modes': {
            'horizontal_halves': mode in ['horizontal', 'all'],
            'vertical_halves': mode in ['vertical', 'all'],
            'quadrants': mode in ['quadrants', 'all'],
            'custom_regions': []
        },
        'ui_detection': {
            'enabled': True,
            'header_height': 60,
            'footer_height': 40,
            'sidebar_width': 200
        },
        'performance': {
            'min_slice_area': 10000, 
            'max_slices_per_image': 6
        }
    }
    
    slicer = ScreenshotSlicer(config)
    return slicer.slice_screenshot(image_path)


def get_most_relevant_slice(slices: List[Dict]) -> Optional[Dict]:
    """
    Get the most relevant slice from a list of slices.
    
    Args:
        slices: List of slice information
        
    Returns:
        Most relevant slice or None
    """
    if not slices:
        return None
    
    # Priority order: high > medium > low
    # Within same priority, prefer larger areas
    priority_order = {'high': 3, 'medium': 2, 'low': 1}
    
    sorted_slices = sorted(slices, 
                          key=lambda x: (priority_order.get(x['priority'], 0), x['area']), 
                          reverse=True)
    
    return sorted_slices[0]