import pdfplumber

def pdf_to_txt(pdf_path, output_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    with open(output_path, "w") as f:
        f.write(text)

pdf_to_txt(
    "static/michael_mccallion_dissertation.pdf",
    "knowledge/dissertation.txt"
)

pdf_to_txt(
    "static/michael_mccallion_cv.pdf",
    "knowledge/cv.txt"
)

print("Documents converted successfully.")