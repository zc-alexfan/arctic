#!/bin/bash
set -e

echo "Downloading cropped images"
mkdir -p downloads/data/cropped_images_zips
python scripts_data/download_data.py --url_file ./bash/assets/urls/cropped_images.txt --out_folder downloads/data/cropped_images_zips
