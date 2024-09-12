import os
import json
import shutil
import pandas as pd
from utils import download_images
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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

def download_dataset_images(csv_file, images_dir):
    csv_path = os.path.join(config['DATASET_DIR'], csv_file)
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    image_links = df['image_link'].tolist()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(lambda link: download_images([link], images_dir), image_links), 
                  total=len(image_links), desc=f"Downloading images from {csv_file}"))

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