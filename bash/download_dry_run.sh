#!/bin/bash
set -e

echo "Downloading smaller files"
mkdir -p downloads/data
python scripts_data/download_data.py --url_file ./bash/assets/urls/misc.txt --out_folder downloads/data --dry_run

echo "Downloading model weights"
mkdir -p downloads/
python scripts_data/download_data.py --url_file ./bash/assets/urls/models.txt --out_folder downloads --dry_run

echo "Downloading cropped images"
mkdir -p downloads/data/cropped_images_zips
python scripts_data/download_data.py --url_file ./bash/assets/urls/cropped_images.txt --out_folder downloads/data/cropped_images_zips --dry_run

echo "Downloading full resolution images"
mkdir -p downloads/data/images_zips
python scripts_data/download_data.py --url_file ./bash/assets/urls/images.txt --out_folder downloads/data/images_zips --dry_run

echo "Downloading SMPLX"
mkdir -p downloads
python scripts_data/download_data.py --url_file ./bash/assets/urls/smplx.txt --out_folder downloads
unzip downloads/models_smplx_v1_1.zip
mv models body_models

echo "Downloading MANO"
python scripts_data/download_data.py --url_file ./bash/assets/urls/mano.txt --out_folder downloads

mkdir unpack
cd downloads
unzip mano_v1_2.zip
mv mano_v1_2/models ../body_models/mano
cd ..
mv body_models unpack
