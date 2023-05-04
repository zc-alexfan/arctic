#!/bin/bash
set -e

echo "Downloading model weights"
mkdir -p downloads/
python scripts_data/download_data.py --url_file ./bash/assets/urls/models.txt --out_folder downloads
