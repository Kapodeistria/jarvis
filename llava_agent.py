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

def generate_jarvis_comment(image_path):
    image = Image.open(image_path).convert('RGB')
    prompt = (
        "Du bist ein smarter PC-Assistent wie Jarvis aus Iron Man. "
        "Beschreibe den Screenshot und gib einen cleveren, hilfreichen Tipp für den Nutzer. "
        "Sei freundlich, direkt und ein bisschen witzig."
    )
    inputs = processor(prompt, images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150)
    return processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    save_screenshot("screenshot.png")
    result = generate_jarvis_comment("screenshot.png")
    print(result)