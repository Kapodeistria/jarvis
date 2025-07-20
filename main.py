import os
import re
import threading
import json

def extract_three_replies(text):
    # Suche nach Nummerierung 1. 2. 3. oder - - - oder Zeilenumbrüche
    lines = text.strip().splitlines()
    replies = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Falls die Zeile mit 1., 2., 3. beginnt oder mit "-" (Liste)
        if re.match(r"^(\d\.|\-)", line):
            replies.append(line)
        else:
            replies.append(line)

        if len(replies) >= 3:
            break

    # Falls keine klare Nummerierung, nimm einfach die ersten 3 Zeilen
    if len(replies) < 3:
        replies = lines[:3]

    return "\n".join(replies)

def load_config(config_path="config_enhanced.json"):
    """Load configuration from JSON file with fallback to default config."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Fallback to original config
            with open("config.json", 'r') as f:
                config = json.load(f)
                # Add default slicing config if not present
                if 'screenshot_slicing' not in config:
                    config['screenshot_slicing'] = {
                        'enabled': False,
                        'slice_modes': {'horizontal_halves': True},
                        'ui_detection': {'enabled': True},
                        'performance': {'max_slices_per_image': 4}
                    }
                return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {'model': 'EleutherAI/gpt-neo-1.3B', 'screenshot_slicing': {'enabled': False}}

import time
import PySimpleGUI as sg
import pyttsx3
import torch
import queue
import hashlib

# Set environment variables early (also can be set in terminal)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Load configuration
config = load_config()
slicing_enabled = config.get('screenshot_slicing', {}).get('enabled', False)

# Import slicing module if enabled
if slicing_enabled:
    from screenshot_slicer import ScreenshotSlicer, get_most_relevant_slice
    screenshot_slicer = ScreenshotSlicer(config.get('screenshot_slicing'))
else:
    screenshot_slicer = None

processing_queue = queue.Queue()

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device = get_device()
print(f"Using device: {device}")

processor = None
blip_model = None

model_loaded_event = threading.Event()

from transformers import Blip2Processor, Blip2ForConditionalGeneration

def load_model():
    global processor, blip_model
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(model_name)
    blip_model.to(device)
    model_loaded_event.set()

# Screenshot & Dateinamenfunktionen
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_filename(directory="screenshots", prefix="screenshot", ext="png"):
    ensure_dir(directory)
    existing = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
    numbers = []
    for f in existing:
        start = len(prefix) + 1
        end = -len(ext) - 1
        num_part = f[start:end]
        if num_part.isdigit():
            numbers.append(int(num_part))
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(directory, f"{prefix}_{next_num}.{ext}")

def get_screenshot(path):
    import mss
    with mss.mss() as sct:
        sct.shot(output=path)
    return path

# OCR Funktion
def extract_text_from_image(image_path):
    from PIL import Image
    import pytesseract
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

# OCR-Text-Filterfunktion
def filter_ocr_text(text, min_letter_ratio=0.3, min_length=5):
    lines = text.splitlines()
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < min_length:
            continue
        letters = re.findall(r"[A-Za-z]", line)
        letter_ratio = len(letters) / len(line) if len(line) > 0 else 0
        if letter_ratio < min_letter_ratio:
            continue
        if re.match(r"^[\d\s\W_]+$", line):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)

import re

def clean_whatsapp_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Filter out UI elements, timestamps, "Chats", "File Edit", "Help", dates, etc.
        if re.search(r'(File Edit|Help|Chats|Yesterday|Today|[0-9]{2}:[0-9]{2}|TRIAL PERIOD)', line, re.I):
            continue
        cleaned.append(line.strip())
    # Return last 20 lines only to limit input size
    return "\n".join(cleaned[-20:])

def build_whatsapp_prompt(chat_text):
    prompt = (
        "Du bist mein persönlicher WhatsApp-Assistent, der mir hilft, im aktuellen Chat sinnvolle Antworten vorzuschlagen. "
        "Der folgende Text ist der sichtbare Chatverlauf:\n\n"
        f"{chat_text}\n\n"
        "Bitte gib mir 3 kurze, freundliche Antwortvorschläge, die ich schnell übernehmen kann. "
        "Berücksichtige den Ton und die vorherigen Nachrichten. Keine weiteren Erklärungen, nur die Vorschläge."
    )
    return prompt

def clear_memory():
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def generate_multimodal_response(image_path, processor, model, device, prompt_instruction="Be a sassy Jarvis. Suggest short, clever next steps or reminders.", chat_mode=False):
    from PIL import Image

    debug_logs = []
    def debug_print(*args, **kwargs):
        message = " ".join(str(a) for a in args)
        debug_logs.append(message)

    try:
        debug_print(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")

        debug_print("Extracting OCR text...")
        ocr_text_raw = extract_text_from_image(image_path)

        if chat_mode:
            ocr_text = clean_whatsapp_text(ocr_text_raw)
            debug_print(f"WhatsApp mode enabled. Filtered OCR text:\n{ocr_text}")
            prompt = build_whatsapp_prompt(ocr_text)
        else:
            ocr_text = filter_ocr_text(ocr_text_raw)
            if len(ocr_text) > 400:
                ocr_text = ocr_text[:400] + "..."
            debug_print(f"OCR text extracted (filtered & truncated): {ocr_text}")
            prompt = (
                f"INSTRUCTION: {prompt_instruction}\n\n"
                f"OCR TEXT:\n{ocr_text}\n\n"
                "ANTWORT:"
            )
            debug_print("Prompt created (content hidden for clarity).")

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        debug_print("Inputs created from processor.")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        debug_print(f"Inputs moved to device: {device}")

        inputs.pop("max_length", None)
        debug_print("Removed max_length from inputs (if present).")

        output = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        debug_print(f"Model output (tensor): {output}")

        decoded_text = processor.decode(output[0], skip_special_tokens=True)
        debug_print(f"Decoded text: {decoded_text}")

        # Cleanup
        del inputs, output
        clear_memory()

        return decoded_text, "\n".join(debug_logs)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_memory()
            warning_msg = "Error: Out of memory. Please reduce model size or token length."
            debug_print(warning_msg)
            return warning_msg, "\n".join(debug_logs)
        else:
            raise

def process_screenshot_slices(screenshot_path, processor, model, device, prompt_instruction, chat_mode=False):
    """
    Process screenshot using slicing for more efficient analysis.
    
    Args:
        screenshot_path: Path to the screenshot
        processor: BLIP2 processor
        model: BLIP2 model
        device: Processing device
        prompt_instruction: Instruction for the model
        chat_mode: Whether in chat mode
        
    Returns:
        Tuple of (combined_response, debug_logs)
    """
    debug_logs = []
    
    if not screenshot_slicer:
        # Fallback to original processing
        return generate_multimodal_response(screenshot_path, processor, model, device, prompt_instruction, chat_mode)
    
    try:
        debug_logs.append("Starting screenshot slicing process...")
        
        # Slice the screenshot
        slices = screenshot_slicer.slice_screenshot(screenshot_path)
        debug_logs.append(f"Created {len(slices)} slices from screenshot")
        
        if not slices:
            debug_logs.append("No slices created, falling back to full image processing")
            return generate_multimodal_response(screenshot_path, processor, model, device, prompt_instruction, chat_mode)
        
        # Get relevant slices based on configuration
        relevant_slices = screenshot_slicer.get_relevant_slices(slices, mode='auto')
        debug_logs.append(f"Selected {len(relevant_slices)} relevant slices for processing")
        
        responses = []
        slice_processing_mode = config.get('screenshot_slicing', {}).get('processing', {}).get('mode', 'auto')
        
        if slice_processing_mode == 'best_slice_only':
            # Process only the most relevant slice
            best_slice = get_most_relevant_slice(relevant_slices)
            if best_slice:
                if 'slice_path' in best_slice:
                    response, slice_debug = generate_multimodal_response(
                        best_slice['slice_path'], processor, model, device, prompt_instruction, chat_mode
                    )
                else:
                    # Process slice data in memory
                    import tempfile
                    import cv2
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        cv2.imwrite(temp_file.name, best_slice['slice_data'])
                        response, slice_debug = generate_multimodal_response(
                            temp_file.name, processor, model, device, prompt_instruction, chat_mode
                        )
                        os.unlink(temp_file.name)
                
                responses.append(f"[{best_slice['name']}]: {response}")
                debug_logs.append(f"Processed slice {best_slice['name']}: {slice_debug}")
        
        else:
            # Process multiple relevant slices
            for slice_info in relevant_slices:
                try:
                    if 'slice_path' in slice_info:
                        response, slice_debug = generate_multimodal_response(
                            slice_info['slice_path'], processor, model, device, prompt_instruction, chat_mode
                        )
                    else:
                        # Process slice data in memory
                        import tempfile
                        import cv2
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            cv2.imwrite(temp_file.name, slice_info['slice_data'])
                            response, slice_debug = generate_multimodal_response(
                                temp_file.name, processor, model, device, prompt_instruction, chat_mode
                            )
                            os.unlink(temp_file.name)
                    
                    responses.append(f"[{slice_info['name']}]: {response}")
                    debug_logs.append(f"Processed slice {slice_info['name']}: {slice_debug}")
                    
                except Exception as e:
                    debug_logs.append(f"Error processing slice {slice_info['name']}: {e}")
        
        # Combine responses
        if responses:
            combined_response = "\n\n".join(responses)
            debug_logs.append("Successfully combined responses from all slices")
            return combined_response, "\n".join(debug_logs)
        else:
            debug_logs.append("No successful slice responses, falling back to full image")
            return generate_multimodal_response(screenshot_path, processor, model, device, prompt_instruction, chat_mode)
            
    except Exception as e:
        debug_logs.append(f"Error in slice processing: {e}")
        debug_logs.append("Falling back to full image processing")
        return generate_multimodal_response(screenshot_path, processor, model, device, prompt_instruction, chat_mode)

def screenshot_thread():
    while True:
        filename = get_next_filename()
        screenshot_path = get_screenshot(filename)
        # Screenshot-Pfad in Queue zur Verarbeitung schicken
        processing_queue.put(screenshot_path)
        time.sleep(5)  # Screenshot-Intervall (kann angepasst werden)

def processing_thread(window, processor, model, device):
    global blip_model

    model_loaded_event.wait()

    processor = processor or globals().get('processor')
    model = model or globals().get('blip_model')

    prompt = "Be a sassy Jarvis. Suggest short, clever next steps or reminders."
    last_hash = None

    while True:
        screenshot_path = processing_queue.get()  # wartet bis neuer Screenshot da ist

        ocr_text_raw = extract_text_from_image(screenshot_path).lower()
        current_hash = hashlib.md5(ocr_text_raw.encode('utf-8')).hexdigest()

        if current_hash == last_hash:
            # Kein neuer Text, skip
            window.write_event_value('-UPDATE-', (screenshot_path, "No changes detected, skipping processing.", ""))
            continue
        last_hash = current_hash

        chat_mode = "whatsapp" in ocr_text_raw

        # Use slicing if enabled, otherwise use original processing
        if slicing_enabled and screenshot_slicer:
            description, debug_log = process_screenshot_slices(
                screenshot_path,
                processor,
                model,
                device,
                prompt_instruction=prompt,
                chat_mode=chat_mode
            )
        else:
            description, debug_log = generate_multimodal_response(
                screenshot_path,
                processor,
                model,
                device,
                prompt_instruction=prompt,
                chat_mode=chat_mode
            )
        
        prompt = description
        from response_formatter import format_replies
        replies_text = format_replies(description)
        window.write_event_value('-UPDATE-', (screenshot_path, replies_text, debug_log))

        clear_memory()

def main():
    slicing_status = "Enabled" if slicing_enabled else "Disabled"
    
    layout = [
        [sg.Text('Jarvis Screenshot Assistent', font=("Helvetica", 20), justification='center', expand_x=True)],
        [sg.Text(f'Screenshot Slicing: {slicing_status}', font=("Helvetica", 10), justification='center', expand_x=True)],
        [sg.Frame('Antwortvorschläge', [
            [sg.Multiline(size=(80, 15), key='-OUTPUT-', disabled=True, autoscroll=True, font=("Consolas", 12), expand_x=True, expand_y=True)]
        ], expand_x=True, expand_y=True)],
        [sg.Frame('Debug-Informationen', [
            [sg.Multiline(size=(80, 10), key='-DEBUG-', disabled=True, autoscroll=True, font=("Consolas", 10), text_color='gray20', background_color='#f0f0f0', expand_x=True, expand_y=True)]
        ], expand_x=True, expand_y=True)],
        [sg.Text('Modell wird geladen, bitte warten...', key='-STATUS-', size=(40, 1), justification='left')],
        [sg.Button('Beenden', size=(10,1))]
    ]

    window = sg.Window('Jarvis App', layout, finalize=True, resizable=True)

    threading.Thread(target=load_model, daemon=True).start()
    threading.Thread(target=screenshot_thread, daemon=True).start()
    threading.Thread(target=processing_thread, args=(window, None, None, device), daemon=True).start()

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED or event == 'Beenden':
            break
        elif event == '-UPDATE-':
            filename, text, debug_log = values['-UPDATE-']

            from response_formatter import format_replies
            formatted_text = format_replies(text)
            window['-OUTPUT-'].update(formatted_text, append=True)
            window['-DEBUG-'].update(debug_log + "\n" + "-"*60 + "\n", append=True)

            # Update status based on model loading
            if model_loaded_event.is_set():
                slice_info = f" | Slicing: {slicing_status}" if slicing_enabled else ""
                window['-STATUS-'].update(f"Letztes Update: {time.strftime('%H:%M:%S')}{slice_info}")
            else:
                window['-STATUS-'].update("Modell wird geladen, bitte warten...")

    window.close()

if __name__ == "__main__":
    main()
