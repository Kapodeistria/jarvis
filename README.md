# Jarvis AI Assistant

Ein intelligenter Screenshot-Assistent im Stil von Iron Man's Jarvis, der automatisch Screenshots analysiert und hilfreiche Vorschläge macht.

## ✨ Neue Features (Issue #3 - Prompt Optimierung)

### 🎯 Strukturierte Prompt-Templates
- **Klare Trennung**: INSTRUCTION und CONTENT sind strikt getrennt
- **Konsistente Struktur**: Einheitliche Templates für alle Anwendungsfälle
- **Weniger Halluzination**: Präzisere Anweisungen für bessere Ergebnisse

### 🎭 5 Jarvis-Persönlichkeiten
1. **Classic** - Höflich, professionell (originaler Iron Man Jarvis)
2. **Sassy** - Witzig, schlagfertig, aber konstruktiv  
3. **Minimalist** - Kurz, präzise, direkt auf den Punkt
4. **Friendly** - Warmherzig, unterstützend, ermutigend
5. **Technical** - Detailliert, analytisch, technisch versiert

### ⚙️ Einfache Konfiguration
```bash
# Stil wechseln über Konfigurator
python style_configurator.py

# Oder direkt in config.json
{
  "jarvis_style": "sassy"  # classic, sassy, minimalist, friendly, technical
}
```

### 🔧 Neue Tools
- `style_configurator.py` - Interaktiver Stil-Konfigurator
- `test_prompt_templates.py` - Template-Tests und Demonstrationen
- `demo_improvements.py` - Vorher/Nachher Vergleich

## 🚀 Verwendung

### Standard-Nutzung
```bash
python main.py
```

### Stil-Konfiguration
```bash
python style_configurator.py
```

### Template-Tests
```bash
python test_prompt_templates.py
```

## 📊 Verbesserungen

**Vorher:**
```
Be a sassy Jarvis. Suggest short, clever next steps or reminders.

OCR TEXT:
Email von Chef: Meeting um 14:00

ANTWORT:
```

**Nachher:**
```
INSTRUCTION:
Du bist ein frecher, aber hilfsbereiter Jarvis. Sei witzig und schlagfertig, aber immer konstruktiv.

AUFGABE:
Analysiere den Screenshot und schlage sinnvolle nächste Schritte vor.

CONTENT:
Email von Chef: Meeting um 14:00

ANTWORT:
```

## 🎯 Akzeptanzkriterien erfüllt
- ✅ Klare Trennung zwischen Anweisung und Content
- ✅ Verschiedene Prompt-Templates getestet
- ✅ 5 alternative Jarvis-Stile implementiert
- ✅ Output ist klarer, präziser, weniger Halluzination