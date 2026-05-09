import PyPDF2
import os

import os
import PyPDF2

print("🚀 pdf_to_text.py started")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def build_knowledge_base():
    os.makedirs(KB_DIR, exist_ok=True)

    pdfs = {
        "sepsis_manual.txt": "Sepsis-Manual-7th-Edition-2024-V1.0.pdf",
        "sepsis_study.txt": "A_Descriptive_Study_on_Sepsis_Causes_Outcomes_and_.pdf",
        "sepsis_research.txt": "BMRI2020-7971387.pdf",
        "sepsis_text.txt": "sepsis_data.pdf"
    }

    for txt, pdf in pdfs.items():
        content = extract_text_from_pdf(pdf)
        with open(os.path.join(KB_DIR, txt), "w", encoding="utf-8") as f:

            f.write(content)

    print("✅ Knowledge base created")

if __name__ == "__main__":
    build_knowledge_base()