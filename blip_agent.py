from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# BLIP-Modell laden
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_jarvis_comment(image_path):
    image = Image.open(image_path).convert('RGB')
    prompt = "Du bist ein smarter PC-Assistent wie Jarvis aus Iron Man. Beschreibe das Bild und gib einen cleveren, hilfreichen Tipp für den Nutzer. Sei freundlich, direkt und ein bisschen witzig."
    inputs = processor(image, prompt, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Beispiel-Aufruf
if __name__ == "__main__":
    # Pfad zum Screenshot anpassen!
    result = generate_jarvis_comment("screenshot.png")
    print(result)