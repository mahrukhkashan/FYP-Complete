import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "..", "knowledge_base")
KB_DIR = os.path.abspath(KB_DIR)

def load_documents():
    clinical_file = os.path.join(KB_DIR, "sepsis_clinical.txt")

    print("📂 LOADING FILE:", clinical_file)

    if not os.path.exists(clinical_file):
        print("❌ FILE NOT FOUND!")
        return []

    with open(clinical_file, "r", encoding="utf-8") as f:
        content = f.read()

    print("✅ DOCUMENT LOADED, LENGTH:", len(content))
    return [content]
