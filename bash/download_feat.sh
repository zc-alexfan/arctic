echo "Downloading features files"
mkdir -p downloads/data
python scripts_data/download_data.py --url_file ./bash/assets/urls/feat.txt --out_folder downloads/data