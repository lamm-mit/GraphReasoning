import os
import re
 
def contains_phrase(main_string, phrase):
    return phrase in main_string

def make_dir_if_needed (dir_path):
    if not os.path.exists(dir_path):
        # Create directory
        os.makedirs(dir_path)
        return "Directory created."
    else:
        return  "Directory already exists."

def remove_markdown_symbols(text):
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    # Remove headers
    text = re.sub(r'#+\s', '', text)
    # Remove bold and italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Remove strikethrough
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove extra newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove list markers
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()