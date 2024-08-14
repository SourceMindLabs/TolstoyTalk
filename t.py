import re

def preprocess_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Remove page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any non-alphanumeric characters except periods and spaces
    text = re.sub(r'[^a-zA-Z0-9\s\.]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

# Usage
preprocess_text('war_and_peace_extracted.txt', 'war_and_peace_processed.txt')
print("Text preprocessing complete. Saved to 'war_and_peace_processed.txt'.")