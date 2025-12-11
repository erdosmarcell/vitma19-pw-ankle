import os
import zipfile
import urllib.request
import shutil
import requests
import pandas as pd
import json

def download_zip(url: str, output_path: str):
    print(f"[*] ZIP letöltése innen:\n{url}\n")
    try:
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)
        print("[+] Letöltés kész.")
    except Exception as e:
        print("[!] Hiba történt a letöltés során:")
        print(e)
        raise e


def extract_zip(zip_path: str, extract_dir: str):
    print(f"[*] Kicsomagolás ide: {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("[+] Kicsomagolás kész.")
    except Exception as e:
        print("[!] Hiba történt kicsomagolás közben:")
        print(e)
        raise e


def ensure_clean_directory(path: str):
    if os.path.exists(path):
        print(f"[*] Régi fájlok törlése '{path}' könyvtárban...")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"[!] Nem sikerült törölni {file_path}: {e}")
    else:
        os.makedirs(path, exist_ok=True)

EXCLUDE_DIRS = {"consensus", "sample"}

def is_valid_dir(dname, root_path):
    return dname not in EXCLUDE_DIRS and os.path.isdir(os.path.join(root_path, dname))

def process_json(json_path, base_dir):
    records = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        try:
            choices = item["annotations"][0]["result"][0]["value"]["choices"]
            if len(choices) != 1:
                continue
            label_raw = choices[0]
            label = label_raw.split("_", 1)[1]  # "Pronacio", "Neutralis", "Szupinacio"

            image_rel_path = item["file_upload"]
            image_rel_path = image_rel_path.replace("data/upload/1/", "").replace("data/upload/2/", "")
            image_rel_path = "-".join(image_rel_path.split("-")[1:])

            image_path = os.path.join(base_dir, image_rel_path)
            if os.path.exists(image_path):
                records.append({"image_path": image_path, "label": label})
            else:
                print(f"[!] Kép nem található: {image_path}")
        except Exception as e:
            print(f"[!] Hiba JSON feldolgozáskor: {json_path} -> {e}")
    return records

def prepare_dataset(root_dir, output_csv):
    all_records = []
    for student_dir in os.listdir(root_dir):
        student_path = os.path.join(root_dir, student_dir)
        if not is_valid_dir(student_dir, root_dir):
            continue
        for fname in os.listdir(student_path):
            if fname.endswith(".json"):
                json_path = os.path.join(student_path, fname)
                recs = process_json(json_path, student_path)
                all_records.extend(recs)
    print(all_records)
    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    print(f"Előkészített adatok mentve: {output_csv}")
    print(f"Összesen {len(df)} kép-címke pár készült.")

def flatten_student_dirs(root_dir):
    for student_dir in os.listdir(root_dir):
        student_path = os.path.join(root_dir, student_dir)
        if not os.path.isdir(student_path):
            continue

        for subfolder in ["normal", "pronation", "supination"]:
            subfolder_path = os.path.join(student_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            for fname in os.listdir(subfolder_path):
                src_path = os.path.join(subfolder_path, fname)
                dst_path = os.path.join(student_path, fname)
                if os.path.exists(dst_path):
                    print(f"[!] Figyelem, létezik már: {dst_path}")
                    dst_path = os.path.join(student_path, f"dup_{fname}")
                shutil.move(src_path, dst_path)
            os.rmdir(subfolder_path)



def main():
    DATA_DIR = "/data"
    ZIP_PATH = "/data/raw.zip"
    ZIP_URL = (
        "https://bmeedu-my.sharepoint.com/:u:/g/personal/"
        "gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A"
        "?e=Uu6CTj&download=1"
    )

    print("=== 01_data_processing.py ===")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    ensure_clean_directory(DATA_DIR)

    print(" ZIP letöltése...")
    download_zip(ZIP_URL, ZIP_PATH)

    print("Kicsomagolás...")
    extract_zip(ZIP_PATH, DATA_DIR)

    print("Dataset előkészítése...")
    ANKLEALIGN_DIR = os.path.join(DATA_DIR, "anklealign")
    OUTPUT_CSV = os.path.join(DATA_DIR, "prepared_dataset.csv")
    flatten_student_dirs(ANKLEALIGN_DIR)
    prepare_dataset(ANKLEALIGN_DIR, OUTPUT_CSV)

    print("\n=== Kicsomagolt fájlok listája ===")
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            print(" -", os.path.join(root, fname))

    print("\nData processing befejezve.")


if __name__ == "__main__":
    main()