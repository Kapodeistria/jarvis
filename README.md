# Jarvis

An AI-powered multimodal assistant that takes screenshots and provides intelligent responses using BLIP2 and LLaVA models.

## Features

### Multimodal Response Generation
- Automatic screenshot capture and OCR text extraction
- Intelligent response generation using BLIP2 model
- WhatsApp chat mode for messaging assistance
- Configurable generation parameters for optimal output

### Generation Parameters (New!)
- **Max New Tokens**: Control response length (50-500 tokens)
- **Num Beams**: Adjust beam search for better quality (1-10 beams)
- **Temperature**: Control randomness in sampling (0.1-2.0)
- **Sampling**: Enable/disable sampling for more diverse responses

### Feedback System (New!)
- Rate responses as good 👍 or bad 👎
- Track feedback statistics
- Export training data for model fine-tuning

### Data Logging (New!)
- Automatic logging of inputs, outputs, and feedback
- Export feedback data in JSON format for fine-tuning
- Configurable parameters persistence

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

### UI Controls
- **Generation Settings**: Adjust model parameters in real-time
- **Feedback Panel**: Rate responses to improve future outputs
- **Save Training Data**: Export collected data for model improvement
- **Save Config**: Persist your preferred generation settings

## Configuration

Settings are stored in `config.json`:
```json
{
  "model": "EleutherAI/gpt-neo-1.3B",
  "generation_params": {
    "max_new_tokens": 150,
    "num_beams": 3,
    "temperature": 0.7,
    "do_sample": true
  }
}
```

## Training Data

Feedback data is automatically saved as `feedback_data_TIMESTAMP.json` and includes:
- Screenshot paths and OCR text
- Generated responses
- User feedback (good/bad)
- Generation parameters used
- Timestamps for analysis

This data can be used to fine-tune the model for better performance.