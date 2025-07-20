"""
Prompt Templates für Jarvis AI Assistant
======================================

Dieses Modul stellt strukturierte Prompt-Templates mit klarer Trennung zwischen
INSTRUCTION und CONTENT bereit. Es bietet verschiedene Jarvis-Persönlichkeiten
für unterschiedliche Anwendungsfälle.
"""

from enum import Enum
from typing import Dict, Optional


class JarvisStyle(Enum):
    """Verschiedene Jarvis-Persönlichkeitsstile"""
    CLASSIC = "classic"          # Klassischer Iron Man Jarvis - höflich, professionell
    SASSY = "sassy"             # Frecher, witziger Jarvis
    MINIMALIST = "minimalist"   # Kurz, präzise, effizient
    FRIENDLY = "friendly"       # Warmherzig, unterstützend
    TECHNICAL = "technical"     # Detailliert, analytisch


class PromptTemplate:
    """Basis-Klasse für strukturierte Prompt-Templates"""
    
    def __init__(self, style: JarvisStyle = JarvisStyle.CLASSIC):
        self.style = style
        self.personalities = self._get_personalities()
    
    def _get_personalities(self) -> Dict[JarvisStyle, Dict[str, str]]:
        """Definiert die verschiedenen Jarvis-Persönlichkeiten"""
        return {
            JarvisStyle.CLASSIC: {
                "name": "Classic Jarvis",
                "description": "Höflich, professionell, wie der originale Jarvis aus Iron Man",
                "tone": "formal und respektvoll",
                "response_style": "präzise und hilfsbereit"
            },
            JarvisStyle.SASSY: {
                "name": "Sassy Jarvis", 
                "description": "Witzig, schlagfertig, aber immer noch hilfreich",
                "tone": "freundlich-frech",
                "response_style": "clever und unterhaltsam"
            },
            JarvisStyle.MINIMALIST: {
                "name": "Minimalist Jarvis",
                "description": "Kurz, präzise, keine überflüssigen Worte", 
                "tone": "direkt und effizient",
                "response_style": "kompakt und auf den Punkt"
            },
            JarvisStyle.FRIENDLY: {
                "name": "Friendly Jarvis",
                "description": "Warmherzig, unterstützend, ermutigend",
                "tone": "herzlich und empathisch", 
                "response_style": "motivierend und verständnisvoll"
            },
            JarvisStyle.TECHNICAL: {
                "name": "Technical Jarvis",
                "description": "Detailliert, analytisch, technisch versiert",
                "tone": "sachlich und präzise",
                "response_style": "detailliert und fachlich fundiert"
            }
        }
    
    def get_base_instruction(self) -> str:
        """Basis-Anweisung für den gewählten Stil"""
        personality = self.personalities[self.style]
        
        instructions = {
            JarvisStyle.CLASSIC: (
                "Du bist Jarvis, ein höflicher und professioneller KI-Assistent. "
                "Antworte stets respektvoll, präzise und hilfreich. "
                "Biete konkrete, umsetzbare Vorschläge."
            ),
            JarvisStyle.SASSY: (
                "Du bist ein frecher, aber hilfsbereiter Jarvis. "
                "Sei witzig und schlagfertig, aber immer konstruktiv. "
                "Bringe Humor in deine Antworten, ohne unhöflich zu werden."
            ),
            JarvisStyle.MINIMALIST: (
                "Du bist ein effizienter Jarvis. "
                "Antworte knapp, präzise und direkt auf den Punkt. "
                "Keine überflüssigen Worte, nur das Wesentliche."
            ),
            JarvisStyle.FRIENDLY: (
                "Du bist ein warmherziger und unterstützender Jarvis. "
                "Sei empathisch, ermutigend und verständnisvoll. "
                "Hilf mit positiver Energie und aufbauenden Worten."
            ),
            JarvisStyle.TECHNICAL: (
                "Du bist ein technisch versierter Jarvis. "
                "Analysiere detailliert und biete fachlich fundierte Lösungen. "
                "Erkläre technische Zusammenhänge verständlich."
            )
        }
        
        return instructions[self.style]


class ScreenshotPromptTemplate(PromptTemplate):
    """Template für allgemeine Screenshot-Analyse"""
    
    def build_prompt(self, ocr_text: str, custom_instruction: Optional[str] = None) -> str:
        """
        Erstellt einen strukturierten Prompt für Screenshot-Analyse
        
        Args:
            ocr_text: Extrahierter Text aus dem Screenshot
            custom_instruction: Optionale benutzerdefinierte Anweisung
            
        Returns:
            Strukturierter Prompt mit klarer INSTRUCTION/CONTENT Trennung
        """
        base_instruction = self.get_base_instruction()
        specific_instruction = custom_instruction or "Analysiere den Screenshot und schlage sinnvolle nächste Schritte vor."
        
        # Kürze OCR-Text falls zu lang
        if len(ocr_text) > 400:
            ocr_text = ocr_text[:400] + "..."
        
        prompt = f"""INSTRUCTION:
{base_instruction}

AUFGABE:
{specific_instruction}

CONTENT:
{ocr_text.strip()}

ANTWORT:"""
        
        return prompt


class WhatsAppPromptTemplate(PromptTemplate):
    """Spezialisiertes Template für WhatsApp-Chat-Analyse"""
    
    def build_prompt(self, chat_text: str) -> str:
        """
        Erstellt einen strukturierten Prompt für WhatsApp-Chat-Antworten
        
        Args:
            chat_text: Bereinigter WhatsApp-Chat-Text
            
        Returns:
            Strukturierter Prompt für Chat-Antwortvorschläge
        """
        base_instruction = self.get_base_instruction()
        
        # Style-spezifische Chat-Anweisungen
        chat_instructions = {
            JarvisStyle.CLASSIC: "Erstelle 3 höfliche, angemessene Antwortvorschläge für den Chat.",
            JarvisStyle.SASSY: "Erstelle 3 witzige, aber freundliche Antwortvorschläge für den Chat.",
            JarvisStyle.MINIMALIST: "Erstelle 3 kurze, prägnante Antwortvorschläge für den Chat.",
            JarvisStyle.FRIENDLY: "Erstelle 3 herzliche, positive Antwortvorschläge für den Chat.",
            JarvisStyle.TECHNICAL: "Erstelle 3 sachliche, hilfreiche Antwortvorschläge für den Chat."
        }
        
        chat_instruction = chat_instructions[self.style]
        
        prompt = f"""INSTRUCTION:
{base_instruction}

AUFGABE:
{chat_instruction}
Berücksichtige den Gesprächskontext und den Ton der vorherigen Nachrichten.
Gib nur die 3 Antwortvorschläge aus, keine weiteren Erklärungen.

CHAT-CONTENT:
{chat_text.strip()}

ANTWORTVORSCHLÄGE:"""
        
        return prompt


class MultimodalPromptTemplate(PromptTemplate):
    """Template für multimodale Prompts (Bild + Text)"""
    
    def build_prompt(self, context_text: str = "", custom_instruction: Optional[str] = None) -> str:
        """
        Erstellt einen strukturierten Prompt für multimodale Analyse
        
        Args:
            context_text: Zusätzlicher Kontext-Text
            custom_instruction: Optionale benutzerdefinierte Anweisung
            
        Returns:
            Strukturierter Prompt für Bild+Text-Analyse
        """
        base_instruction = self.get_base_instruction()
        specific_instruction = custom_instruction or "Beschreibe das Bild und gib hilfreiche Tipps oder Vorschläge."
        
        prompt_parts = [
            f"INSTRUCTION:\n{base_instruction}",
            f"AUFGABE:\n{specific_instruction}"
        ]
        
        if context_text.strip():
            prompt_parts.append(f"KONTEXT:\n{context_text.strip()}")
        
        prompt_parts.append("ANTWORT:")
        
        return "\n\n".join(prompt_parts)


def get_prompt_template(template_type: str, style: JarvisStyle = JarvisStyle.CLASSIC):
    """
    Factory-Funktion zum Erstellen von Prompt-Templates
    
    Args:
        template_type: Art des Templates ('screenshot', 'whatsapp', 'multimodal')
        style: Jarvis-Persönlichkeitsstil
        
    Returns:
        Entsprechendes PromptTemplate-Objekt
    """
    templates = {
        'screenshot': ScreenshotPromptTemplate,
        'whatsapp': WhatsAppPromptTemplate,
        'multimodal': MultimodalPromptTemplate
    }
    
    if template_type not in templates:
        raise ValueError(f"Unbekannter Template-Typ: {template_type}")
    
    return templates[template_type](style)


def get_available_styles() -> Dict[str, Dict[str, str]]:
    """
    Gibt eine Übersicht aller verfügbaren Jarvis-Stile zurück
    
    Returns:
        Dictionary mit Stil-Namen als Keys und Beschreibungen als Values
    """
    template = PromptTemplate()
    return {style.value: info for style, info in template.personalities.items()}


# Beispiel-Verwendung für Tests
if __name__ == "__main__":
    # Teste verschiedene Stile
    styles = [JarvisStyle.CLASSIC, JarvisStyle.SASSY, JarvisStyle.MINIMALIST]
    
    for style in styles:
        print(f"\n=== {style.value.upper()} STYLE ===")
        template = ScreenshotPromptTemplate(style)
        example_prompt = template.build_prompt("Beispiel OCR Text hier")
        print(example_prompt[:200] + "...")