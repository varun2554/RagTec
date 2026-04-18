import PyPDF2

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    if file_path.lower().endswith(".pdf"):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    return text.strip()
