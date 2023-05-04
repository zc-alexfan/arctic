#!/bin/bash
set -e

echo "Downloading preprocessed splits"
mkdir -p downloads/data
python scripts_data/download_data.py --url_file ./bash/assets/urls/splits.txt --out_folder downloads/data