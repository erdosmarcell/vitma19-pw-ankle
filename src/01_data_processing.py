import os
import zipfile
import urllib.request
import shutil
import requests
import pandas as pd
import json
import random
from PIL import Image
import re
from json import JSONDecodeError
from collections import Counter, defaultdict
import config

def download_zip(url: str, output_path: str):
    print(f"Downloading ZIP from: \n{url}\n")
    try:
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)
        print("Downloading finished.")
    except Exception as e:
        print("Error while downloading:")
        print(e)
        raise e


def extract_zip(zip_path: str, extract_dir: str):
    print(f"Extract here: {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction finished.")
    except Exception as e:
        print("Error during extraction:")
        print(e)
        raise e


def ensure_clean_directory(path: str):
    if os.path.exists(path):
        print(f"Deleting old files at '{path}' ...")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Unsuccessful deletion {file_path}: {e}")
    else:
        os.makedirs(path, exist_ok=True)

EXCLUDE_DIRS = {"consensus", "sample"}

def is_valid_dir(dname, root_path):
    return dname not in EXCLUDE_DIRS and os.path.isdir(os.path.join(root_path, dname))

def process_json(json_path, base_dir):
    records = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    parent_folder_name = os.path.basename(os.path.dirname(json_path))

    for item in data:
        try:
            annotations = item.get("annotations", [])
            if not annotations:
                continue
            result_list = annotations[0].get("result", [])
            if not result_list:
                continue

            value = result_list[0].get("value", {})
            choices = value.get("choices", [])
            if len(choices) != 1:
                continue

            label_raw = choices[0]
            label = label_raw.split("_", 1)[-1]  # "Pronacio", "Neutralis", "Szupinacio"

            image_rel_path = item.get("file_upload") or item.get("data", {}).get("image", "")
            if not image_rel_path:
                continue

            image_rel_path = image_rel_path.replace("data/upload/1/", "").replace("data/upload/2/", "")
            parts = image_rel_path.split("-", 1)
            if len(parts) > 1:
                image_rel_path = parts[1]
            filename, ext = os.path.splitext(image_rel_path)
            if filename.endswith("_" + parent_folder_name):
                filename = filename[:-(len(parent_folder_name)+1)]
                image_rel_path = filename + ext

            image_path = os.path.join(base_dir, image_rel_path)
            if os.path.exists(image_path):
                records.append({"image_path": image_path, "label": label})
            else:
                print(f"Image cannot be found: {image_path}")
        except Exception as e:
            print(f"JSON processing error: {json_path} -> {e}")
    return records

def prepare_dataset(root_dir, output_csv):
    LABEL_MAP = {
        "neutral": "Neutralis",
        "pronation": "Pronacio",
        "supination": "Szupinacio"
    }

    all_records = []
    for student_dir in os.listdir(root_dir):
        student_path = os.path.join(root_dir, student_dir)
        if not is_valid_dir(student_dir, root_dir):
            continue
        for fname in os.listdir(student_path):
            if fname.endswith(".json"):
                json_path = os.path.join(student_path, fname)
                recs = process_json(json_path, student_path)
                for r in recs:
                    r["label"] = LABEL_MAP.get(r["label"].lower(), r["label"])
                print(fname, "->", len(recs), "rekord")
                all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    print(f"Prepocessed data saved at: {output_csv}")
    print(f"{len(df)} image-label pairs created.")


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
                    print(f"Already existing: {dst_path}")
                    dst_path = os.path.join(student_path, f"dup_{fname}")
                shutil.move(src_path, dst_path)
            os.rmdir(subfolder_path)

def extract_participant_segment(text):
    pattern = r"vevo(.{3})"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        extracted_part = match.group(1)
        cleaned_part = extracted_part.replace("_", "")
        return cleaned_part
    else:
        return None
    
def normalize_image_name(name: str):
    basename = os.path.basename(name)
    if "-" in basename:
        basename = basename.split("-", 1)[1]
    idx = basename.find("saj")
    if idx >= 0:
        basename = basename[idx:]
    return basename
    
def load_consensus_annotations(consensus_dir):
    consensus_data = {}

    for fname in os.listdir(consensus_dir):
        if not fname.lower().endswith(".json"):
            continue

        json_path = os.path.join(consensus_dir, fname)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except JSONDecodeError:
            print(f"JSON file excluded: {json_path}")
            continue
        except Exception as e:
            print(f"Could not read JSON: {json_path} | {e}")
            continue

        if not isinstance(data, list):
            print(f"JSON (non list-based) excluded: {json_path}")
            continue

        for item in data:
            raw_name = os.path.basename(item.get("file_upload", ""))
            image_name = normalize_image_name(raw_name)
            if not image_name:
                continue

            group_id = extract_participant_segment(image_name)
            if group_id is None:
                continue

            annotations = item.get("annotations", [])
            if not annotations:
                continue

            result = annotations[0].get("result", [])
            if not result:
                continue

            choices = result[0].get("value", {}).get("choices", [])
            if len(choices) != 1:
                continue

            label = choices[0].split("_", 1)[-1].strip().capitalize()

            consensus_data.setdefault(group_id, {})
            consensus_data[group_id].setdefault(image_name, [])
            consensus_data[group_id][image_name].append(label)

    return consensus_data

def group_json_files_by_participant(consensus_dir, base_dir):
    grouped_result = {}
    for fname in os.listdir(consensus_dir):
        if not fname.lower().endswith(".json"):
            continue
        json_name_upper = os.path.splitext(fname)[0].upper()

        json_path = os.path.join(consensus_dir, fname)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not read: {json_path} | {e}")
            continue

        file_info = []

        for item in data:
            raw_name = os.path.basename(item.get("file_upload", ""))
            image_name = normalize_image_name(raw_name)
            if not image_name:
                continue

            img_path = os.path.join(base_dir, image_name)

            annotations = item.get("annotations", [])
            label = ""
            if annotations:
                result = annotations[0].get("result", [])
                if result:
                    choices = result[0].get("value", {}).get("choices", [])
                    if len(choices) == 1:
                        label = choices[0].split("_", 1)[-1].strip().capitalize()

            file_info.append((img_path, label))

        file_info.sort(key=lambda x: os.path.basename(x[0]))

        groups = []
        current_group = []
        previous_id_numeric = None
        previous_directory = None

        for img_path, label in file_info:
            filename = os.path.basename(img_path)
            directory = os.path.basename(os.path.dirname(img_path))

            current_id_str = extract_participant_segment(filename)
            current_id_numeric = None
            try:
                if current_id_str is not None:
                    current_id_numeric = int(current_id_str)
            except ValueError:
                pass

            if previous_id_numeric is not None and current_id_numeric is not None:
                if directory != previous_directory or not (current_id_numeric == previous_id_numeric or current_id_numeric == previous_id_numeric + 1):
                    groups.append(current_group)
                    current_group = []

            current_group.append((img_path, label))
            previous_id_numeric = current_id_numeric
            previous_directory = directory

        if current_group:
            groups.append(current_group)

        file_info = []
        filtered_groups = []
        for group in groups:
            filtered_group = [(f, l) for f, l in group if ("saja" in os.path.basename(f).lower() 
                                                            or "vevo" in os.path.basename(f).lower())]
            if filtered_group:
                filtered_groups.append(filtered_group)
        grouped_result[json_name_upper] = filtered_groups

        print(f"\n=== JSON file: {os.path.basename(json_path)} ===")
        print(f"\n {len(filtered_groups)} groups found.\n")

        # for i, group in enumerate(filtered_groups):
        #     print(f"Group {i+1} (Size: {len(group)}):\n")
        #     for file_path, label in group:
        #         print(f"  - {os.path.basename(file_path)} | Label: {label}")
        #     print()

    return grouped_result

def extract_two_digit_sequence(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.findall(r'\d{2}', name)


def split_dataset_csv_only(input_csv, output_dir, DATA_DIR, ANKLEALIGN_DIR, train_ratio=0.75, val_ratio=0.2, test_ratio=0.2):
    CONSENSUS_TXT = os.path.join(ANKLEALIGN_DIR, "consensus", "anklealign-consensus.txt")
    CONSENSUS_CSV = config.CONSENSUS_TEST_CSV

    consensus_records = []
    with open(CONSENSUS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            img_path = os.path.join(DATA_DIR, line.replace("\\", "/"))
            consensus_records.append({"image_path": img_path, "label": ""})

    consensus_df = pd.DataFrame(consensus_records)

    good_indices = []
    for idx, row in consensus_df.iterrows():
        img_path = row["image_path"]
        try:
            with Image.open(img_path) as img:
                img.verify()
            good_indices.append(idx)
        except Exception as e:
            print(f"Damaged or unreadable image: {img_path} | {e}")

    consensus_df = consensus_df.loc[good_indices].reset_index(drop=True)
    os.makedirs(os.path.dirname(CONSENSUS_CSV), exist_ok=True)
    consensus_df.to_csv(CONSENSUS_CSV, index=False)

    groups = []
    current_group = []
    previous_id_numeric = None
    previous_directory = None

    for idx, row in consensus_df.iterrows():
        img_path = row["image_path"]
        filename = os.path.basename(img_path)
        directory = os.path.basename(os.path.dirname(img_path))
        
        current_id_str = extract_participant_segment(filename)
        
        current_id_numeric = None
        try:
            if current_id_str is not None:
                current_id_numeric = int(current_id_str)
        except ValueError:
            print(f"Warning: Could not convert ID '{current_id_str}' to integer for file: {filename}")

        if previous_id_numeric is not None and current_id_numeric is not None:
            if directory != previous_directory or not (current_id_numeric == previous_id_numeric or current_id_numeric == previous_id_numeric + 1):
                groups.append(current_group)
                current_group = []

        current_group.append(img_path)
        previous_id_numeric = current_id_numeric
        previous_directory = directory

    if current_group:
        groups.append(current_group)

    #print(f"Total {len(groups)} groups found in the CSV based on sequential ID logic.")

    # for i, group in enumerate(groups):
    #     print(f"\nGroup {i+1} (Size: {len(group)}):")
    #     for file_path in group:
    #         print(f"  - {os.path.basename(file_path)}")

    labeled_cons_groups = group_json_files_by_participant(os.path.join(ANKLEALIGN_DIR, "consensus"), ANKLEALIGN_DIR)
    #print(labeled_cons_groups)

    csv_group_sizes = Counter(len(group) for group in groups)

    #print("\n=== CSV consensus group sizes ===")
    #for size, count in sorted(csv_group_sizes.items()):
    #    print(f"Group size {size}: {count} db")
    
    labeled_group_sizes = Counter()

    for neptun, group_list in labeled_cons_groups.items():
        for group in group_list:
            labeled_group_sizes[len(group)] += 1

    #print("\n=== Labeled consensus group sizes ===")
    #for size, count in sorted(labeled_group_sizes.items()):
    #    print(f"Group size {size}: {count} db")

    common_sizes = set(csv_group_sizes.keys()) & set(labeled_group_sizes.keys())

    #print("\n=== Matching group sizes ===")
    total_matches = 0

    for size in sorted(common_sizes):
        csv_count = csv_group_sizes[size]
        labeled_count = labeled_group_sizes[size]
        matched = min(csv_count, labeled_count)
        total_matches += matched

        # print(
        #     f"Group size {size}: "
        #     f"CSV={csv_count}, "
        #     f"Labeled={labeled_count}, "
        #     f"Matched={matched}"
        # )

    #print(f"Total matching groups by size: {total_matches}")

    labeled_by_size = defaultdict(list)

    for neptun, group_list in labeled_cons_groups.items():
        for group in group_list:
            if len(group) > 5:
                labeled_by_size[len(group)].append(group)

        #print(f"Consensus test CSV ready: {CONSENSUS_CSV}, {len(consensus_df)} records")
    
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n=== Majority label assignment for groups ===")
    cons_labeling_records = []
    all_labels = ["Neutralis", "Pronacio", "Szupinacio"]
    for gi, group in enumerate(groups, start=1):
        group_size = len(group)

        if group_size <= 5:
            continue
        if group_size not in labeled_by_size:
            continue

        #print(f"\nGroup {gi} (size={group_size})")

        for img_path in group:
            img_seq = extract_two_digit_sequence(img_path)
            label_votes = []

            for labeled_group in labeled_by_size[group_size]:
                for labeled_img, label in labeled_group:
                    labeled_seq = extract_two_digit_sequence(labeled_img)

                    if img_seq == labeled_seq:
                        label_votes.append(label)

            if label_votes:
                vote_counter = Counter(label_votes)
                majority_label, vote_count = vote_counter.most_common(1)[0]

                print(
                    f"  {os.path.basename(img_path)} "
                    f"-> {majority_label} ({vote_count}/{len(label_votes)})"
                )
                counts = {label: vote_counter.get(label, 0) for label in all_labels}
                
                cons_labeling_records.append({
                    "image": os.path.basename(img_path),
                    "group_size": group_size,
                    "total_votes": len(label_votes),
                    "majority_label": majority_label,
                    "majority_count": vote_count,
                    **counts
                })
            else:
                print(
                    f"  {os.path.basename(img_path)} "
                    f"-> NO LABEL FOUND"
                )
    df_votes = pd.DataFrame(cons_labeling_records)
    df_votes.to_csv(config.CONSENSUS_LABELS_CSV, index=False)
    print(f"Saved consensus distribution CSV with {len(df_votes)} rows")

    print("\n=== Updating consensus_df with majority vote results ===")

    path_to_index = {
        row["image_path"]: idx for idx, row in consensus_df.iterrows()
    }

    updated_count = 0
    skipped_groups = 0

    for group in groups:
        group_size = len(group)

        if group_size <= 5 or group_size not in labeled_by_size:
            skipped_groups += 1
            continue

        for img_path in group:
            img_seq = extract_two_digit_sequence(img_path)
            label_votes = []

            for labeled_group in labeled_by_size[group_size]:
                for labeled_img, label in labeled_group:
                    if extract_two_digit_sequence(labeled_img) == img_seq:
                        label_votes.append(label)

            if len(label_votes) >= 8:
                vote_counter = Counter(label_votes)
                majority_label, vote_count = vote_counter.most_common(1)[0]

                if img_path in path_to_index:
                    consensus_df.at[path_to_index[img_path], "label"] = majority_label
                    updated_count += 1

    print(f"Number of updated records: {updated_count}")
    print(f"Number of omitted groups (not counted): {skipped_groups}")

    consensus_df.to_csv(CONSENSUS_CSV, index=False)
    print(f"Consensus CSV updated: {CONSENSUS_CSV}")

    good_indices = []
    for idx, row in df.iterrows():
        img_path = row["image_path"]
        try:
            with Image.open(img_path) as img:
                img.verify()
            good_indices.append(idx)
        except Exception as e:
            print(f"Damaged or unreadable image: {img_path} | {e}")

    df = df.loc[good_indices].reset_index(drop=True)
    print(f"{len(df)} valid image remained of {len(good_indices)} checked ones.")

    consensus_paths = set(consensus_df["image_path"])
    df = df[~df["image_path"].isin(consensus_paths)].reset_index(drop=True)
    print(f"Consensus images excluded: {len(df)} records remained before the split.")

    n = len(df)
    train_end = int(train_ratio * n)

    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:],
        #"test": df.iloc[val_end:]
    }

    os.makedirs(output_dir, exist_ok=True)

    split_paths = set(df["image_path"])
    overlap = split_paths & consensus_paths
    if overlap:
        print(f"{len(overlap)} images remained in both:")
        for p in overlap:
            print(" -", p)
    else:
        print("No overlap between split and consensus datasets.")

    for split_name, split_df in splits.items():
        split_csv_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(split_csv_path, index=False)
        print(f"{split_name}: {len(split_df)} records, CSV saved: {split_csv_path}")

def filter_valid_images(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    valid_records = []
    for idx, row in df.iterrows():
        img_path = row['image_path']
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid_records.append(row)
        except Exception as e:
            print(f"Damaged or unreadable image: {img_path} | {e}")
    filtered_df = pd.DataFrame(valid_records)
    filtered_df.to_csv(output_csv, index=False)
    print(f"{len(filtered_df)} valid pictures left of {len(df)} checked.")
    return filtered_df

def count_images_per_student(root_dir):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for student_dir in os.listdir(root_dir):
        student_path = os.path.join(root_dir, student_dir)
        if not os.path.isdir(student_path):
            continue

        count = 0
        for fname in os.listdir(student_path):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                img_path = os.path.join(student_path, fname)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    count += 1
                except Exception:
                    pass

        print(f"{student_dir}: {count} valid images")



def main():
    DATA_DIR = config.DATA_DIR
    ZIP_PATH = config.ZIP_PATH
    ZIP_URL = (
        "https://bmeedu-my.sharepoint.com/:u:/g/personal/"
        "gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A"
        "?e=Uu6CTj&download=1"
    )

    print("=== 01_data_processing.py ===")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    ensure_clean_directory(DATA_DIR)

    print("Downloading ZIP...")
    download_zip(ZIP_URL, ZIP_PATH)

    print("Extraction...")
    extract_zip(ZIP_PATH, DATA_DIR)

    print("Preparing dataset...")
    ANKLEALIGN_DIR = os.path.join(DATA_DIR, "anklealign")
    OUTPUT_CSV = os.path.join(DATA_DIR, "prepared_dataset.csv")
    flatten_student_dirs(ANKLEALIGN_DIR)
    prepare_dataset(ANKLEALIGN_DIR, OUTPUT_CSV)


    OUTPUT_DIR = os.path.join(DATA_DIR, "split")
    prepared_csv = os.path.join(DATA_DIR, "prepared_dataset.csv")
    PREPARED_CSV = config.PREPARED_CSV

    filter_valid_images(PREPARED_CSV, PREPARED_CSV)

    split_dataset_csv_only(prepared_csv, OUTPUT_DIR, DATA_DIR, ANKLEALIGN_DIR)

    count_images_per_student(ANKLEALIGN_DIR)

    print("\nData processing finished.")


if __name__ == "__main__":
    main()