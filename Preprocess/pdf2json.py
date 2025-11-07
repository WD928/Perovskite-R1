import os
import json
import shutil
from glob import glob
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from natsort import natsorted  

MAX_LENGTH = 2500

def extract_pdf_content(pdf):
    rsrcmgr = PDFResourceManager()
    outfp = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr=rsrcmgr, outfp=outfp, laparams=laparams)
    try:
        with open(pdf, 'rb') as fp:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos = set()
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
                try:
                    interpreter.process_page(page)
                except Exception as e:
                    print(f"Error processing page in {pdf}: {e}", flush=True)
                    raise 
    except Exception as e:
        print(f"Error opening or processing {pdf}: {e}", flush=True)
        raise  
    finally:
        mystr = outfp.getvalue()
        device.close()
        outfp.close()
    return mystr

def split_content_by_words(content):
    words = content.split()
    content_parts = []
    step = int(MAX_LENGTH * 0.8)  
    i = 0
    while i < len(words):
        part = " ".join(words[i:i + MAX_LENGTH])
        content_parts.append(part)
        i += step
    return content_parts

def process_pdfs(input_folder, output_folder, fail_path):
    os.makedirs(output_folder, exist_ok=True)
    s = 0
    w = 0
    failed_files = []  
    
    for filename in natsorted(os.listdir(input_folder)):  
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            
            print(f"Processing {pdf_path}...")
            try:
                content = extract_pdf_content(pdf_path)
            except Exception as e:
                print(f"Skipping entire file {pdf_path} due to errors: {e}", flush=True)
                w += 1
                print(f"Number {w} file skipped.")
                failed_files.append(pdf_path)  
                continue  

            if content:
                content_parts = split_content_by_words(content)

                base_filename = os.path.splitext(filename)[0]

                for i, part in enumerate(content_parts):
                    json_filename = f"{base_filename}_part{i+1}.json"
                    json_path = os.path.join(output_folder, json_filename)
            
                    json_dir = os.path.dirname(json_path)
                    os.makedirs(json_dir, exist_ok=True)

                    try:
                        with open(json_path, 'w', encoding='utf-8') as json_file:
                            json.dump({"content": part}, json_file, ensure_ascii=False, indent=4)

                        print(f"Saved JSON part {i+1} to {json_path}", flush=True)
                    except Exception as e:
                        print(f"Error saving JSON file {json_path}: {e}", flush=True)
                        w += 1
                        print(f"Number {w} file skipped.")
                s += 1
                
    with open(os.path.join(output_folder, fail_path), "w", encoding="utf-8") as f:
        if failed_files:
            f.write("Failed files:\n")
            for file in failed_files:
                f.write(f"- {file}\n")
            f.write(f"\nTotal failed files: {len(failed_files)}")
        else:
            f.write("No files failed.\n")

    print(f"{s} files successfully processed.")
    print(f"{w} files skipped.")

def merge_json_files(json_folder, output_file):
    merged_data = []

    json_files = natsorted(glob(os.path.join(json_folder, "*.json")))
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {json_file}: {e}", flush=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

        print(f"All JSON files merged into {output_file}", flush=True)
    except Exception as e:
        print(f"Error saving merged JSON file {output_file}: {e}", flush=True)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder {folder_path} deleted successfully.", flush=True)
        except Exception as e:
            print(f"Error deleting folder {folder_path}: {e}", flush=True)

if __name__ == "__main__":

    input_folder = "user/paper_dataset_perovskite/pdf"
    output_folder = "user/paper_dataset_perovskite/json_split"
    merged_file = "user/paper_dataset_perovskite/paper_split.json"
    fail_path = "user/paper_dataset_perovskite/fail.txt"

    process_pdfs(input_folder, output_folder, fail_path)

    merge_json_files(output_folder, merged_file)
