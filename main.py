import os
import re
import threading

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

import time
import PySimpleGUI as sg
import pyttsx3
import torch
import queue
import hashlib

# Set environment variables early (also can be set in terminal)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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

# OCR-Text-Filterfunktion (improved for better UI noise filtering)
def filter_ocr_text(text, min_letter_ratio=0.3, min_length=5):
    """
    Improved OCR text filter that better recognizes and removes irrelevant UI elements.
    
    Args:
        text: Raw OCR text to filter
        min_letter_ratio: Minimum ratio of letters to total characters (default 0.3)
        min_length: Minimum line length to consider (default 5)
    
    Returns:
        Filtered text with significantly reduced UI noise
    """
    
    # Blacklist patterns for common UI elements, menus, timestamps, etc.
    ui_blacklist_patterns = [
        # Common menu items and UI elements
        r'^(File|Edit|View|Insert|Format|Tools|Add-ons|Extensions|Help|Window|Settings|Preferences)(\s+|$)',
        r'^(Menu|Settings|Options|Configuration|Properties)$',
        r'^Application Menu$',
        r'^Zoom:\s*\d+%$',
        
        # Timestamps and dates
        r'\d{1,2}:\d{2}(\s*(AM|PM))?$',
        r'^(Today|Yesterday|Tomorrow)$',
        r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'Last edit was .* ago',
        r'@ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
        
        # Trial/license notices
        r'TRIAL PERIOD',
        r'\d+ days? remaining',
        
        # Status messages and system notifications
        r'^(Status|Error|Warning|Info|DEBUG):\s*',
        r'^(Connected|Disconnected|Loading|Saving)\.?\s*$',
        r'Memory usage \d+%',
        r'Updates available',
        r'Connection (failed|lost|established)',
        r'Low battery',
        
        # UI symbols and single-word UI elements  
        r'^[☰⚙×✓□└├│▶▼◀▲►▼◄▲\s]+$',  # Lines with only UI symbols
        r'^[─├└│┌┐┘┴┬┤┼\s]+$',        # Box drawing characters
        r'^[\.\-_=\+\*#\|\s]+$',       # Punctuation/decoration lines
        r'^(&nbsp;|&amp;|&lt;|&gt;)+$', # HTML entities
        
        # Tree/hierarchy indicators and single UI words
        r'^[│├└]\s*[─\s]*\s*(Sub item|Tree item|Last sub item|Another tree item)$',
        r'^(☰ Menu|⚙ Settings|× Close|✓ Check|□ Checkbox)$',
        
        # Repetitive characters or symbols  
        r'^(.)\1{4,}$',  # Same character repeated 5+ times
        r'^[\s\.\-_=\+\*#\|]{3,}$',  # Only punctuation/symbols
    ]
    
    # Additional patterns for lines that are likely UI noise
    noise_patterns = [
        r'^\s*\d+\s*$',  # Lines with only numbers
        r'^[^\w\s]*$',   # Lines with only non-alphanumeric, non-space characters
        r'^[A-Z\s]{2,}$',  # Lines with only capital letters and spaces (likely UI labels)
    ]
    
    lines = text.splitlines()
    filtered_lines = []
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip lines that are too short
        if len(line) < min_length:
            continue
        
        # Check against UI blacklist patterns
        is_ui_noise = False
        for pattern in ui_blacklist_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_ui_noise = True
                break
        
        if is_ui_noise:
            continue
            
        # Check letter ratio (keep original logic)
        letters = re.findall(r"[A-Za-z]", line)
        letter_ratio = len(letters) / len(line) if len(line) > 0 else 0
        
        if letter_ratio < min_letter_ratio:
            continue
            
        # Skip lines that match noise patterns
        is_noise = False
        for pattern in noise_patterns:
            if re.match(pattern, line):
                is_noise = True
                break
                
        if is_noise:
            continue
            
        # Keep lines that pass all filters
        filtered_lines.append(line)
    
    return "\n".join(filtered_lines)

import re

def clean_whatsapp_text(text):
    """
    Clean WhatsApp OCR text by removing UI elements and limiting content.
    Now uses improved filtering patterns from filter_ocr_text.
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line_stripped = line.strip()
        
        # Enhanced filter patterns for WhatsApp-specific and general UI elements
        if re.search(r'(File Edit|Help|Chats|Yesterday|Today|[0-9]{1,2}:[0-9]{2}|TRIAL PERIOD)', line, re.I):
            continue
        # Additional general UI patterns from the improved filter
        if re.search(r'^(Menu|Settings|Options)$', line_stripped, re.I):
            continue
        if re.search(r'^\d+ days? remaining', line, re.I):
            continue
        if re.search(r'^(Status|Error|Warning|Info):\s*', line, re.I):
            continue
            
        cleaned.append(line_stripped)
    
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
    layout = [
        [sg.Text('Jarvis Screenshot Assistent', font=("Helvetica", 20), justification='center', expand_x=True)],
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
                window['-STATUS-'].update(f"Letztes Update: {time.strftime('%H:%M:%S')}")
            else:
                window['-STATUS-'].update("Modell wird geladen, bitte warten...")

    window.close()

if __name__ == "__main__":
    main()
