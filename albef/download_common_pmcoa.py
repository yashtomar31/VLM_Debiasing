#!/usr/bin/env python3
"""
Download and extract PMC-OA dataset from NCBI.

This script downloads medical paper figures from the PMC Open Access dataset,
extract them, and cleans up by removing GIF files.
"""

import pandas as pd
import os
import tarfile
import urllib.request
from urllib.parse import urljoin
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Configuration
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
DATA_FILE = "./csv/VLM_train.csv"
DOWNLOAD_DIR = "./common_pmcoa_data/"

# Ensure output directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print("Reading the common PMCOA metadata ...")
data_df = pd.read_csv(DATA_FILE)

# Optional: filter by label
# data_df = data_df[data_df['label'] == 'yes']
file_paths = data_df["Online_file_path"].tolist()
print(f"Total files to download: {len(file_paths)}")

def download_and_extract(file_path: str) -> str:
    """
    Download and extract a single PMC-OA tar.gz file.
    
    Args:
        file_path: Relative path from NCBI FTP base URL
        
    Returns:
        Status message
    """
    file_name = file_path.split('/')[-1]
    file_full_path = os.path.join(DOWNLOAD_DIR, file_name)
    url = urljoin(BASE_URL, file_path)

    # Skip if already downloaded
    if os.path.exists(file_full_path):
        return f"Skipped (already exists): {file_name}"

    try:
        # Download
        urllib.request.urlretrieve(url, file_full_path)

        # Extract
        with tarfile.open(file_full_path, "r:gz") as tar:
            tar.extractall(DOWNLOAD_DIR)

        # Delete archive
        os.remove(file_full_path)

        return f"✓ Success: {file_name}"
    except Exception as e:
        return f"✗ Failed: {file_name} - {str(e)}"


def cleanup_gifs(directory: str) -> None:
    """
    Remove all GIF files from directory tree.
    
    Args:
        directory: Root directory to clean
    """
    print("\nCleaning up .gif files...")
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".gif"):
                gif_path = os.path.join(root, file)
                try:
                    os.remove(gif_path)
                    count += 1
                except Exception as e:
                    print(f"Error deleting {gif_path}: {e}")
    print(f"Deleted {count} GIF files.")


# Run in parallel
if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(download_and_extract, file_paths), total=len(file_paths)):
            print(result)

    cleanup_gifs(DOWNLOAD_DIR)
    print(f"\nDownload complete. Data saved to: {DOWNLOAD_DIR}")

def cleanup_gifs(directory: str) -> None:
    """
    Remove all GIF files from directory tree.
    
    Args:
        directory: Root directory to clean
    """
    print("\nCleaning up .gif files...")
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".gif"):
                gif_path = os.path.join(root, file)
                try:
                    os.remove(gif_path)
                    print(f"Deleted GIF: {gif_path}")
                except Exception as e:
                    print(f"Error deleting {gif_path}: {e}")