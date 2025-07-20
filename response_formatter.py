def format_replies(text: str) -> str:
    paragraphs = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    if len(paragraphs) >= 3:
        formatted_text = ""
        for i, para in enumerate(paragraphs[:3], 1):
            formatted_text += f"{i}. {para}\n\n"
    else:
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        formatted_text = ""
        for i, line in enumerate(lines, 1):
            formatted_text += f"{i}. {line}\n"
    return formatted_text