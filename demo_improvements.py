#!/usr/bin/env python3
"""
Demonstration der Prompt-Optimierung für Issue #3
================================================

Zeigt die Verbesserungen der neuen strukturierten Prompt-Templates
gegenüber dem alten System.
"""

def demonstrate_improvements():
    """Demonstriert die Verbesserungen durch die neuen Prompt-Templates"""
    
    print("JARVIS PROMPT-OPTIMIERUNG - VORHER vs. NACHHER")
    print("=" * 60)
    
    # Beispiel-Situationen
    scenarios = [
        {
            "name": "Screenshot-Analyse",
            "ocr_text": "VS Code geöffnet. Python-Datei mit Syntax-Fehler in Zeile 42."
        },
        {
            "name": "WhatsApp-Chat",
            "chat_text": "Lisa: Kommst du heute Abend zur Party?\nDu: Bin noch unsicher...\nLisa: Wäre schön wenn du da wärst!"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name'].upper()}")
        print("-" * 40)
        
        if "ocr_text" in scenario:
            demonstrate_screenshot_improvement(scenario["ocr_text"])
        else:
            demonstrate_whatsapp_improvement(scenario["chat_text"])

def demonstrate_screenshot_improvement(ocr_text):
    """Zeigt Verbesserung bei Screenshot-Prompts"""
    
    # ALTE VERSION (wie es vorher war)
    old_prompt = f"""Be a sassy Jarvis. Suggest short, clever next steps or reminders.

OCR TEXT:
{ocr_text}

ANTWORT:"""
    
    # NEUE VERSION
    from prompt_templates import get_prompt_template, JarvisStyle
    template = get_prompt_template('screenshot', JarvisStyle.SASSY)
    new_prompt = template.build_prompt(ocr_text)
    
    print("VORHER (alte Struktur):")
    print(old_prompt)
    print("\nNACHHER (neue strukturierte Prompts):")
    print(new_prompt)
    
    print("\n📈 VERBESSERUNGEN:")
    print("  ✅ Klarere INSTRUCTION/CONTENT Trennung")
    print("  ✅ Spezifischere Aufgabenbeschreibung")
    print("  ✅ Konsistente Struktur")
    print("  ✅ Weniger Halluzination durch präzise Anweisungen")

def demonstrate_whatsapp_improvement(chat_text):
    """Zeigt Verbesserung bei WhatsApp-Prompts"""
    
    # ALTE VERSION
    old_prompt = f"""Du bist mein persönlicher WhatsApp-Assistent, der mir hilft, im aktuellen Chat sinnvolle Antworten vorzuschlagen. Der folgende Text ist der sichtbare Chatverlauf:

{chat_text}

Bitte gib mir 3 kurze, freundliche Antwortvorschläge, die ich schnell übernehmen kann. Berücksichtige den Ton und die vorherigen Nachrichten. Keine weiteren Erklärungen, nur die Vorschläge."""
    
    # NEUE VERSION
    from prompt_templates import get_prompt_template, JarvisStyle
    template = get_prompt_template('whatsapp', JarvisStyle.FRIENDLY)
    new_prompt = template.build_prompt(chat_text)
    
    print("VORHER (alte Struktur):")
    print(old_prompt)
    print("\nNACHHER (neue strukturierte Prompts):")
    print(new_prompt)
    
    print("\n📈 VERBESSERUNGEN:")
    print("  ✅ Stil-spezifische Anweisungen")
    print("  ✅ Strukturierte Aufgabendefinition")
    print("  ✅ Klarere Trennung von Anweisung und Chat-Content")
    print("  ✅ Personalitäts-basierte Antworten")

def show_personality_variety():
    """Zeigt die Vielfalt der neuen Jarvis-Persönlichkeiten"""
    
    print("\n" + "=" * 60)
    print("NEUE JARVIS-PERSÖNLICHKEITEN")
    print("=" * 60)
    
    from prompt_templates import get_prompt_template, JarvisStyle
    
    sample_text = "Kalender zeigt 5 Termine heute. Nächster in 20 Minuten."
    
    personalities = [
        (JarvisStyle.CLASSIC, "Professionell & höflich"),
        (JarvisStyle.SASSY, "Witzig & schlagfertig"), 
        (JarvisStyle.MINIMALIST, "Kurz & präzise"),
        (JarvisStyle.FRIENDLY, "Warmherzig & unterstützend"),
        (JarvisStyle.TECHNICAL, "Analytisch & detailliert")
    ]
    
    for style, description in personalities:
        print(f"\n{style.value.upper()} ({description}):")
        print("-" * 30)
        
        template = get_prompt_template('screenshot', style)
        prompt = template.build_prompt(sample_text)
        
        # Zeige nur die INSTRUCTION Sektion
        instruction_part = prompt.split("AUFGABE:")[0] + "AUFGABE:"
        print(instruction_part)

def show_key_improvements():
    """Zeigt die wichtigsten Verbesserungen zusammengefasst"""
    
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG DER OPTIMIERUNGEN")
    print("=" * 60)
    
    improvements = [
        "🎯 KLARERE STRUKTUR",
        "   • Strikte Trennung zwischen INSTRUCTION und CONTENT",
        "   • Konsistente Template-Struktur für alle Anwendungsfälle",
        "   • Kein Vermischen von Anweisungen und Daten",
        "",
        "🎭 MEHRERE PERSÖNLICHKEITEN",
        "   • 5 verschiedene Jarvis-Stile für unterschiedliche Situationen",
        "   • Konfigurierbar über config.json",
        "   • Einfach erweiterbar für neue Stile",
        "",
        "🔧 WENIGER HALLUZINATION",
        "   • Präzisere Anweisungen reduzieren unerwünschte Ausgaben",
        "   • Spezifische Aufgabendefinitionen",
        "   • Kontext-bewusste Prompts",
        "",
        "⚙️ EINFACHE KONFIGURATION",
        "   • JSON-basierte Stil-Auswahl",
        "   • Style Configurator Tool",
        "   • Vergleichsmöglichkeiten zwischen Stilen",
        "",
        "🔄 ERWEITERBARKEIT",
        "   • Template-System für neue Prompt-Typen",
        "   • Einfaches Hinzufügen neuer Persönlichkeiten",
        "   • Modular aufgebaut"
    ]
    
    for improvement in improvements:
        print(improvement)

if __name__ == "__main__":
    demonstrate_improvements()
    show_personality_variety()
    show_key_improvements()
    
    print(f"\n{'='*60}")
    print("✅ PROMPT-OPTIMIERUNG ERFOLGREICH IMPLEMENTIERT")
    print(f"{'='*60}")
    print("\nNächste Schritte:")
    print("• Starten Sie die Anwendung und testen Sie verschiedene Stile")
    print("• Nutzen Sie python style_configurator.py zum Stil-Wechsel") 
    print("• Beobachten Sie klarere, präzisere Antworten mit weniger Halluzination")