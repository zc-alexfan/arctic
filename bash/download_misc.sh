#!/bin/bash
set -e

echo "Downloading smaller files"
mkdir -p downloads/data
python scripts_data/download_data.py --url_file ./bash/assets/urls/misc.txt --out_folder downloads/data
