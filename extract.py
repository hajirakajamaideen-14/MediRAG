import os
from pypdf import PdfReader

data_folder = "data"
all_text = ""

for file in os.listdir(data_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(data_folder, file)
        print(f"Reading {file}...")
        
        reader = PdfReader(file_path)
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

# Save extracted text
with open("medical_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("Text extraction completed!")
