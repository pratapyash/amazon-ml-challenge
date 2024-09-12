import os
import json
import shutil
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from PIL import Image
from pathlib import Path

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

def create_directories():
    directories = ['src', config['TRAIN_IMAGES_DIR'], config['TEST_IMAGES_DIR']]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_dataset():
    missing_files = []
    for filename in config['DATASET_FILES']:
        file_path = os.path.join(config['DATASET_DIR'], filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        for file in missing_files:
            pass

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return

def download_image(url, save_path, max_retries=3, delay=3):
    if not isinstance(url, str):
        return False

    filename = Path(url).name
    image_save_path = os.path.join(save_path, filename)

    if os.path.exists(image_save_path):
        return True

    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(image_save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return True
        except Exception as e:
            print(f"Error downloading {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    
    create_placeholder_image(image_save_path)  # Create a black placeholder image for invalid links/images
    return False

def download_dataset_images(csv_file, images_dir):
    csv_path = os.path.join(config['DATASET_DIR'], csv_file)
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    image_links = df['image_link'].tolist()
    
    def download_and_save(link):
        return download_image(link, images_dir)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_and_save, link) for link in image_links]
        for _ in tqdm(as_completed(futures), total=len(image_links), desc=f"Downloading images from {csv_file}"):
            pass  # Each completed future represents a downloaded and saved image

def copy_source_files():
    if not os.path.exists(config['SRC_SOURCE_DIR']):
        return

    for file in os.listdir(config['SRC_SOURCE_DIR']):
        src_path = os.path.join(config['SRC_SOURCE_DIR'], file)
        dst_path = os.path.join('src', file)
        shutil.copy2(src_path, dst_path)

def copy_sample_code():
    if not os.path.exists(config['SAMPLE_CODE_SOURCE']):
        return

    shutil.copy2(config['SAMPLE_CODE_SOURCE'], "sample_code.py")

def setup():
    create_directories()
    check_dataset()
    download_dataset_images('train.csv', config['TRAIN_IMAGES_DIR'])
    download_dataset_images('test.csv', config['TEST_IMAGES_DIR'])
    copy_source_files()
    copy_sample_code()

if __name__ == "__main__":
    setup()