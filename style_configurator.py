#!/usr/bin/env python3
"""
Jarvis Style Configurator
=========================

Hilfsprogramm zum Einfachen Wechseln zwischen verschiedenen Jarvis-Persönlichkeiten.
"""

import json
import os
from prompt_templates import get_available_styles

def load_config():
    """Lädt die aktuelle Konfiguration"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "model": "EleutherAI/gpt-neo-1.3B",
            "jarvis_style": "sassy"
        }

def save_config(config):
    """Speichert die Konfiguration"""
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def show_current_style():
    """Zeigt den aktuellen Stil an"""
    config = load_config()
    current_style = config.get('jarvis_style', 'sassy')
    styles = get_available_styles()
    
    print(f"\nAktueller Jarvis-Stil: {current_style.upper()}")
    if current_style in styles:
        style_info = styles[current_style]
        print(f"Beschreibung: {style_info['description']}")
        print(f"Ton: {style_info['tone']}")
    print()

def show_all_styles():
    """Zeigt alle verfügbaren Stile an"""
    styles = get_available_styles()
    
    print("\nVerfügbare Jarvis-Stile:")
    print("=" * 40)
    
    for i, (style_name, info) in enumerate(styles.items(), 1):
        print(f"{i}. {style_name.upper()}")
        print(f"   {info['description']}")
        print(f"   Ton: {info['tone']}")
        print()

def change_style():
    """Ermöglicht das Ändern des Jarvis-Stils"""
    styles = list(get_available_styles().keys())
    
    show_all_styles()
    
    while True:
        try:
            choice = input(f"Wähle einen Stil (1-{len(styles)}) oder 'q' zum Beenden: ").strip()
            
            if choice.lower() == 'q':
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(styles):
                selected_style = styles[choice_num - 1]
                
                # Konfiguration aktualisieren
                config = load_config()
                config['jarvis_style'] = selected_style
                save_config(config)
                
                print(f"\n✅ Jarvis-Stil erfolgreich auf '{selected_style}' geändert!")
                show_current_style()
                return
            else:
                print(f"Bitte wähle eine Zahl zwischen 1 und {len(styles)}")
                
        except ValueError:
            print("Bitte gib eine gültige Zahl ein")
        except KeyboardInterrupt:
            print("\nVorgang abgebrochen")
            return

def create_style_comparison():
    """Erstellt eine Vergleichsdatei mit allen Stilen für denselben Prompt"""
    sample_ocr = "E-Mail Benachrichtigung: 'Teammeeting verschoben auf 15:30. Neue Agenda im Anhang.'"
    
    from prompt_templates import get_prompt_template, JarvisStyle
    
    comparison = []
    comparison.append("JARVIS STYLE COMPARISON")
    comparison.append("=" * 60)
    comparison.append(f"Beispiel-Situation: {sample_ocr}")
    comparison.append("=" * 60)
    
    for style in JarvisStyle:
        comparison.append(f"\n{style.value.upper()} STYLE:")
        comparison.append("-" * 30)
        
        template = get_prompt_template('screenshot', style)
        prompt = template.build_prompt(sample_ocr)
        comparison.append(prompt)
        comparison.append("\n" + "="*60)
    
    # Speichern in Datei
    with open('style_comparison.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(comparison))
    
    print("✅ Style-Vergleich wurde in 'style_comparison.txt' gespeichert")

def main():
    """Hauptfunktion"""
    print("JARVIS STYLE CONFIGURATOR")
    print("=" * 40)
    
    while True:
        show_current_style()
        
        print("Optionen:")
        print("1. Aktuellen Stil anzeigen")
        print("2. Alle verfügbaren Stile anzeigen")
        print("3. Stil ändern")
        print("4. Style-Vergleich erstellen")
        print("5. Beenden")
        
        choice = input("\nWähle eine Option (1-5): ").strip()
        
        if choice == '1':
            show_current_style()
        elif choice == '2':
            show_all_styles()
        elif choice == '3':
            change_style()
        elif choice == '4':
            create_style_comparison()
        elif choice == '5':
            print("Auf Wiedersehen!")
            break
        else:
            print("Ungültige Auswahl. Bitte wähle 1-5.")
        
        input("\nDrücke Enter um fortzufahren...")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()