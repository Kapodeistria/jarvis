# jarvis

A smart screenshot assistant that uses AI to analyze screenshots and provide intelligent suggestions. Now featuring **advanced screenshot slicing** for efficient processing of high-frequency captures.

## Features

- **Screenshot Analysis**: Automated screenshot capture and AI-powered analysis
- **Smart Slicing**: Divide screenshots into meaningful regions for efficient processing
- **WhatsApp Integration**: Specialized processing for WhatsApp chat screenshots
- **Real-time Processing**: Handle multiple screenshots per second with optimized slicing
- **Intelligent Suggestions**: AI-powered response recommendations

## Screenshot Slicing

The new slicing functionality enables efficient processing by dividing screenshots into targeted regions:

- **Performance**: 20+ slices per second processing rate
- **Smart Detection**: Automatic UI element detection (headers, footers, sidebars)
- **Configurable Modes**: Horizontal/vertical halves, quadrants, custom regions
- **Priority System**: Focus on most relevant content areas

### Quick Start with Slicing

1. Enable slicing in `config_enhanced.json`:
```json
{
  "screenshot_slicing": {
    "enabled": true,
    "slice_modes": {
      "horizontal_halves": true
    }
  }
}
```

2. Run the application - slicing will be applied automatically

3. Test slicing functionality:
```bash
python demo_slicing.py
python test_slicing.py
```

For detailed documentation, see [SLICING_README.md](SLICING_README.md).

## Installation

1. Install dependencies:
```bash
pip install mss pillow pytesseract torch transformers PySimpleGUI pyttsx3 opencv-python numpy
```

2. Run the application:
```bash
python main.py
```