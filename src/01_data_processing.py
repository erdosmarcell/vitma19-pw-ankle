import os
import zipfile
import urllib.request
import shutil

def download_zip(url: str, output_path: str):
    print(f"[*] ZIP letöltése innen:\n{url}\n")
    try:
        urllib.request.urlretrieve(url, output_path)
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
        print(f"[*] Régi '{path}' könyvtár törlése...")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    # --- Állandó elérési utak a Docker követelmények alapján ---
    DATA_DIR = "/data"
    ZIP_PATH = "/data/raw.zip"

    # A feladatban megadott SharePoint direkt-download link
    ZIP_URL = (
        "https://bmeedu-my.sharepoint.com/:u:/g/personal/"
        "gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A"
        "?e=Uu6CTj&download=1"
    )

    print("=== 01_data_processing.py ===")

    # Biztosítjuk, hogy a /data létezik
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. régi fájlok törlése
    ensure_clean_directory(DATA_DIR)

    # 2. ZIP letöltése
    print("[*] ZIP letöltése...")
    download_zip(ZIP_URL, ZIP_PATH)

    # 3. Kicsomagolás
    print("[*] Kicsomagolás...")
    extract_zip(ZIP_PATH, DATA_DIR)

    # 4. Ellenőrzés
    print("\n=== Kicsomagolt fájlok listája ===")
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            print(" -", os.path.join(root, fname))

    print("\n[✓] Data processing befejezve.")


if __name__ == "__main__":
    main()
