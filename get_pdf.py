from PyPDF2 import PdfReader

def get_pdf_text(pdf_files):
    if not pdf_files:
        return ""

    raw_text = ""

    for pdf_file in pdf_files:
        pdf = PdfReader(pdf_file)
        num_pages = len(pdf.pages)

        for page_num in range(num_pages):
            page = pdf.pages[page_num]
            raw_text += page.extract_text()

    return raw_text