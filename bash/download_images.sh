#!/bin/bash
set -e

echo "Downloading full resolution images"
mkdir -p downloads/data/images_zips
python scripts_data/download_data.py --url_file ./bash/assets/urls/images.txt --out_folder downloads/data/images_zips
