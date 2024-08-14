import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    
    return text

# Usage
pdf_path = "war-and-peace.pdf"
war_and_peace_text = extract_text_from_pdf(pdf_path)

# Save the extracted text to a file
with open("war_and_peace_extracted.txt", "w", encoding="utf-8") as text_file:
    text_file.write(war_and_peace_text)

print("Text extraction complete. Saved to 'war_and_peace_extracted.txt'.")