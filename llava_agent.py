from PIL import Image
from llava.mm_utils import get_model_and_processor
import mss

# Screenshot aufnehmen und speichern
def save_screenshot(path="screenshot.png"):
    with mss.mss() as sct:
        sct.shot(output=path)

# Modell und Prozessor laden
model_name = "liuhaotian/llava-v1.5-7b"
model, processor = get_model_and_processor(model_name)

def generate_jarvis_comment(image_path, style="classic"):
    """
    Generiert einen Jarvis-Kommentar mit dem neuen strukturierten Prompt-System
    
    Args:
        image_path: Pfad zum Screenshot
        style: Jarvis-Stil (classic, sassy, minimalist, friendly, technical)
    """
    from prompt_templates import get_prompt_template, JarvisStyle
    
    # Stil-Mapping
    style_mapping = {
        "classic": JarvisStyle.CLASSIC,
        "sassy": JarvisStyle.SASSY,
        "minimalist": JarvisStyle.MINIMALIST, 
        "friendly": JarvisStyle.FRIENDLY,
        "technical": JarvisStyle.TECHNICAL
    }
    
    jarvis_style = style_mapping.get(style, JarvisStyle.CLASSIC)
    
    image = Image.open(image_path).convert('RGB')
    
    # Verwende multimodales Template
    template = get_prompt_template('multimodal', jarvis_style)
    prompt = template.build_prompt(
        custom_instruction="Beschreibe den Screenshot und gib einen cleveren, hilfreichen Tipp für den Nutzer."
    )
    
    inputs = processor(prompt, images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150)
    return processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    save_screenshot("screenshot.png")
    
    # Teste verschiedene Stile
    styles = ["classic", "sassy", "minimalist", "friendly", "technical"]
    
    print("Testing different Jarvis styles:")
    for style in styles:
        print(f"\n=== {style.upper()} STYLE ===")
        result = generate_jarvis_comment("screenshot.png", style)
        print(result[:100] + "..." if len(result) > 100 else result)