"""
Enhanced OCR module for Jarvis assistant with improved accuracy features.

Features:
- Multiple Tesseract PSM (Page Segmentation Mode) configurations
- Multi-language support
- Image preprocessing (contrast, brightness, filters)
- Raw OCR text debugging and saving
- Benchmarking capabilities
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np


class EnhancedOCR:
    """Enhanced OCR with multiple Tesseract configurations and preprocessing options."""
    
    # Tesseract PSM modes with descriptions
    PSM_MODES = {
        0: "Orientation and script detection (OSD) only",
        1: "Automatic page segmentation with OSD",
        2: "Automatic page segmentation, but no OSD, or OCR",
        3: "Fully automatic page segmentation, but no OSD (Default)",
        4: "Assume a single column of text of variable sizes",
        5: "Assume a single uniform block of vertically aligned text",
        6: "Assume a single uniform block of text",
        7: "Treat the image as a single text line",
        8: "Treat the image as a single word",
        9: "Treat the image as a single word in a circle",
        10: "Treat the image as a single character",
        11: "Sparse text. Find as much text as possible in no particular order",
        12: "Sparse text with OSD",
        13: "Raw line. Treat the image as a single text line, bypassing hacks"
    }
    
    # Common languages for OCR
    LANGUAGES = {
        'eng': 'English',
        'deu': 'German',
        'fra': 'French',
        'spa': 'Spanish',
        'ita': 'Italian',
        'por': 'Portuguese',
        'rus': 'Russian',
        'chi_sim': 'Chinese Simplified',
        'jpn': 'Japanese',
        'kor': 'Korean'
    }
    
    def __init__(self, debug_dir: str = None, benchmark_dir: str = None, config_file: str = None):
        """
        Initialize EnhancedOCR.
        
        Args:
            debug_dir: Directory to save debug information (overrides config)
            benchmark_dir: Directory to save benchmark results (overrides config)
            config_file: Path to configuration file
        """
        # Load configuration
        try:
            from ocr_config_manager import OCRConfig
            self.config = OCRConfig(config_file) if config_file else OCRConfig()
            dirs = self.config.get_directories()
            self.debug_dir = debug_dir or dirs.get("ocr_debug", "ocr_debug")
            self.benchmark_dir = benchmark_dir or dirs.get("ocr_benchmarks", "ocr_benchmarks")
        except ImportError:
            # Fallback if config manager is not available
            self.debug_dir = debug_dir or "ocr_debug"
            self.benchmark_dir = benchmark_dir or "ocr_benchmarks"
            self.config = None
        
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure debug and benchmark directories exist."""
        for directory in [self.debug_dir, self.benchmark_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def preprocess_image(self, image_path: str, preprocessing_config: Dict) -> Image.Image:
        """
        Apply preprocessing to improve OCR accuracy.
        
        Args:
            image_path: Path to the image
            preprocessing_config: Configuration for preprocessing
            
        Returns:
            Preprocessed PIL Image
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
            
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply contrast enhancement
        if preprocessing_config.get('enhance_contrast', False):
            enhancer = ImageEnhance.Contrast(image)
            contrast_factor = preprocessing_config.get('contrast_factor', 1.5)
            image = enhancer.enhance(contrast_factor)
        
        # Apply brightness enhancement
        if preprocessing_config.get('enhance_brightness', False):
            enhancer = ImageEnhance.Brightness(image)
            brightness_factor = preprocessing_config.get('brightness_factor', 1.2)
            image = enhancer.enhance(brightness_factor)
        
        # Apply sharpness enhancement
        if preprocessing_config.get('enhance_sharpness', False):
            enhancer = ImageEnhance.Sharpness(image)
            sharpness_factor = preprocessing_config.get('sharpness_factor', 1.3)
            image = enhancer.enhance(sharpness_factor)
        
        # Convert to grayscale
        if preprocessing_config.get('grayscale', False):
            image = image.convert('L')
        
        # Apply binary threshold (black and white)
        if preprocessing_config.get('binary_threshold', False):
            # Convert to OpenCV format for binary operations
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            threshold_value = preprocessing_config.get('threshold_value', 127)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            image = Image.fromarray(binary)
        
        # Apply Gaussian blur to reduce noise
        if preprocessing_config.get('gaussian_blur', False):
            blur_radius = preprocessing_config.get('blur_radius', 1)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Apply median filter to reduce noise
        if preprocessing_config.get('median_filter', False):
            size = preprocessing_config.get('median_size', 3)
            image = image.filter(ImageFilter.MedianFilter(size=size))
        
        # Apply morphological operations
        if preprocessing_config.get('morphological_ops', False):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            if len(cv_image.shape) == 3:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            kernel_size = preprocessing_config.get('morph_kernel_size', 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if preprocessing_config.get('erosion', False):
                cv_image = cv2.erode(cv_image, kernel, iterations=1)
            if preprocessing_config.get('dilation', False):
                cv_image = cv2.dilate(cv_image, kernel, iterations=1)
            if preprocessing_config.get('opening', False):
                cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel)
            if preprocessing_config.get('closing', False):
                cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
                
            image = Image.fromarray(cv_image)
        
        return image
    
    def extract_text_with_config(
        self,
        image_path: str,
        psm: int = 3,
        language: str = 'en',
        preprocessing_config: Optional[Dict] = None,
        save_debug: bool = False
    ) -> Dict:
        """
        Extract text using specific Tesseract configuration.
        
        Args:
            image_path: Path to the image
            psm: Page Segmentation Mode (0-13)
            language: Language code for OCR
            preprocessing_config: Image preprocessing configuration
            save_debug: Whether to save debug information
            
        Returns:
            Dictionary with extracted text and metadata
        """
        start_time = time.time()
        
        # Default preprocessing config
        if preprocessing_config is None:
            preprocessing_config = {}
        
        try:
            # Apply preprocessing
            processed_image = self.preprocess_image(image_path, preprocessing_config)
            
            # Configure Tesseract
            config = f'--psm {psm} -l {language}'
            
            # Extract text
            raw_text = pytesseract.image_to_string(processed_image, config=config)
            
            # Get confidence data
            try:
                data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except:
                avg_confidence = 0
            
            processing_time = time.time() - start_time
            
            result = {
                'raw_text': raw_text,
                'cleaned_text': raw_text.strip(),
                'psm': psm,
                'language': language,
                'avg_confidence': avg_confidence,
                'processing_time': processing_time,
                'preprocessing_config': preprocessing_config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save debug information
            if save_debug:
                self.save_debug_info(image_path, processed_image, result)
            
            return result
            
        except Exception as e:
            return {
                'raw_text': '',
                'cleaned_text': '',
                'psm': psm,
                'language': language,
                'avg_confidence': 0,
                'processing_time': time.time() - start_time,
                'preprocessing_config': preprocessing_config,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_debug_info(self, original_image_path: str, processed_image: Image.Image, result: Dict):
        """Save debug information including processed image and OCR results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        
        # Save processed image
        processed_image_path = os.path.join(
            self.debug_dir, 
            f"{base_name}_processed_{timestamp}_psm{result['psm']}.png"
        )
        processed_image.save(processed_image_path)
        
        # Save OCR results
        debug_file = os.path.join(
            self.debug_dir,
            f"{base_name}_ocr_result_{timestamp}_psm{result['psm']}.json"
        )
        
        debug_data = result.copy()
        debug_data['processed_image_path'] = processed_image_path
        debug_data['original_image_path'] = original_image_path
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
    
    def benchmark_configurations(
        self,
        image_path: str,
        psm_modes: List[int] = None,
        languages: List[str] = None,
        preprocessing_configs: List[Dict] = None
    ) -> Dict:
        """
        Benchmark different OCR configurations.
        
        Args:
            image_path: Path to the image
            psm_modes: List of PSM modes to test
            languages: List of languages to test
            preprocessing_configs: List of preprocessing configurations
            
        Returns:
            Benchmark results
        """
        if psm_modes is None:
            psm_modes = [3, 6, 7, 8, 11]  # Most commonly useful modes
        
        if languages is None:
            languages = ['eng', 'deu']  # Default languages
        
        if preprocessing_configs is None:
            preprocessing_configs = [
                {},  # No preprocessing
                {'enhance_contrast': True, 'contrast_factor': 1.5},
                {'grayscale': True},
                {'binary_threshold': True, 'threshold_value': 127},
                {'enhance_contrast': True, 'grayscale': True, 'binary_threshold': True}
            ]
        
        benchmark_results = {
            'image_path': image_path,
            'start_time': datetime.now().isoformat(),
            'results': []
        }
        
        total_configs = len(psm_modes) * len(languages) * len(preprocessing_configs)
        current_config = 0
        
        for psm in psm_modes:
            for lang in languages:
                for i, preprocessing_config in enumerate(preprocessing_configs):
                    current_config += 1
                    print(f"Testing configuration {current_config}/{total_configs}: PSM={psm}, Lang={lang}, Preprocessing={i}")
                    
                    result = self.extract_text_with_config(
                        image_path,
                        psm=psm,
                        language=lang,
                        preprocessing_config=preprocessing_config,
                        save_debug=True
                    )
                    
                    result['config_id'] = f"psm{psm}_lang{lang}_prep{i}"
                    benchmark_results['results'].append(result)
        
        benchmark_results['end_time'] = datetime.now().isoformat()
        
        # Save benchmark results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_file = os.path.join(
            self.benchmark_dir,
            f"benchmark_{timestamp}.json"
        )
        
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        best_config = self.analyze_benchmark_results(benchmark_results)
        benchmark_results['best_config'] = best_config
        
        return benchmark_results
    
    def analyze_benchmark_results(self, benchmark_results: Dict) -> Dict:
        """Analyze benchmark results to find the best configuration."""
        results = benchmark_results['results']
        
        if not results:
            return {}
        
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r and r['avg_confidence'] > 0]
        
        if not valid_results:
            return {}
        
        # Find best result by confidence
        best_by_confidence = max(valid_results, key=lambda x: x['avg_confidence'])
        
        # Find best result by text length (more text might be better)
        best_by_length = max(valid_results, key=lambda x: len(x['cleaned_text']))
        
        # Find fastest result
        best_by_speed = min(valid_results, key=lambda x: x['processing_time'])
        
        return {
            'best_by_confidence': best_by_confidence,
            'best_by_text_length': best_by_length,
            'best_by_speed': best_by_speed,
            'total_configs_tested': len(results),
            'valid_configs': len(valid_results)
        }
    
    def get_optimal_text(self, image_path: str, save_debug: bool = True) -> str:
        """
        Get text using optimal configuration based on quick analysis.
        
        Args:
            image_path: Path to the image
            save_debug: Whether to save debug information
            
        Returns:
            Extracted text using best configuration
        """
        # Quick benchmark with key configurations
        quick_benchmark = self.benchmark_configurations(
            image_path,
            psm_modes=[3, 6, 11],  # Most reliable modes
            languages=['eng'],
            preprocessing_configs=[
                {},  # No preprocessing
                {'enhance_contrast': True, 'grayscale': True},
                {'binary_threshold': True, 'threshold_value': 127}
            ]
        )
        
        best_config = quick_benchmark.get('best_config', {}).get('best_by_confidence')
        
        if best_config:
            return best_config['cleaned_text']
        else:
            # Fallback to simple extraction
            return self.extract_text_with_config(image_path, save_debug=save_debug)['cleaned_text']


# Convenience functions for backward compatibility
def extract_text_from_image(image_path: str, enhanced: bool = True) -> str:
    """
    Extract text from image with optional enhanced processing.
    
    Args:
        image_path: Path to the image
        enhanced: Whether to use enhanced OCR processing
        
    Returns:
        Extracted text
    """
    if enhanced:
        ocr = EnhancedOCR()
        return ocr.get_optimal_text(image_path)
    else:
        # Original simple implementation
        from PIL import Image
        import pytesseract
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()


def benchmark_image_ocr(image_path: str) -> Dict:
    """
    Run a comprehensive benchmark on an image.
    
    Args:
        image_path: Path to the image to benchmark
        
    Returns:
        Benchmark results
    """
    ocr = EnhancedOCR()
    return ocr.benchmark_configurations(image_path)


if __name__ == "__main__":
    # Example usage
    ocr = EnhancedOCR()
    
    # Test with a sample image if available
    test_images = ["screenshot.png", "test.png"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting OCR on {img_path}")
            
            # Simple extraction
            simple_text = extract_text_from_image(img_path, enhanced=False)
            print(f"Simple OCR: {simple_text[:100]}...")
            
            # Enhanced extraction
            enhanced_text = extract_text_from_image(img_path, enhanced=True)
            print(f"Enhanced OCR: {enhanced_text[:100]}...")
            
            # Full benchmark
            print(f"Running benchmark on {img_path}...")
            benchmark_results = benchmark_image_ocr(img_path)
            best = benchmark_results.get('best_config', {}).get('best_by_confidence', {})
            if best:
                print(f"Best configuration: PSM {best['psm']}, Confidence: {best['avg_confidence']:.1f}%")
            
            break
    else:
        print("No test images found. Please ensure screenshot.png or test.png exists.")