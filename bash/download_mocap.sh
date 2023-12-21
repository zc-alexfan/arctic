echo "Downloading features files"
mkdir -p downloads/data
python scripts_data/download_data.py --url_file ./bash/assets/urls/mocap.txt --out_folder downloads/data
