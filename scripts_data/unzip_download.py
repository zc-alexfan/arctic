import os
import os.path as op
import zipfile
from glob import glob

from tqdm import tqdm


def unzip(zip_p, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_p, "r") as zip_ref:
        zip_ref.extractall(out_dir)


def main():
    fnames = glob(op.join("downloads/data/", "**/*"), recursive=True)

    full_img_zips = []
    cropped_images_zips = []
    misc_zips = []
    models_zips = []
    for fname in fnames:
        if not (".zip" in fname or ".npy" in fname):
            continue
        if "/images_zips/" in fname:
            full_img_zips.append(fname)
        elif "/cropped_images_zips/" in fname:
            cropped_images_zips.append(fname)
        elif "raw_seqs.zip" in fname:
            misc_zips.append(fname)
        elif "splits_json.zip" in fname:
            misc_zips.append(fname)
        elif "meta.zip" in fname:
            misc_zips.append(fname)
        elif "splits.zip" in fname:
            misc_zips.append(fname)
        elif "feat.zip" in fname:
            misc_zips.append(fname)
        elif "mocap" in fname or 'smplx_corres.zip' in fname:
            misc_zips.append(fname)
        elif "models.zip" in fname:
            models_zips.append(fname)
        else:
            print(f"Unknown zip: {fname}")

    out_dir = "./unpack/arctic_data/data"
    os.makedirs(out_dir, exist_ok=True)

    # unzip misc files
    for zip_p in misc_zips:
        print(f"Unzipping {zip_p} to {out_dir}")
        unzip(zip_p, out_dir)

    # unzip models files
    for zip_p in models_zips:
        model_out = out_dir.replace("/data", "")
        print(f"Unzipping {zip_p} to {model_out}")
        unzip(zip_p, model_out)

    # unzip images
    pbar = tqdm(cropped_images_zips)
    for zip_p in pbar:
        out_p = op.join(
            out_dir,
            zip_p.replace("downloads/data/", "")
            .replace(".zip", "")
            .replace("cropped_images_zips/", "cropped_images/"),
        )
        pbar.set_description(f"Unzipping {zip_p} to {out_dir}")
        unzip(zip_p, out_p)

    # unzip images
    pbar = tqdm(full_img_zips)
    for zip_p in pbar:
        pbar.set_description(f"Unzipping {zip_p} to {out_dir}")
        out_p = op.join(
            out_dir,
            zip_p.replace("downloads/data/", "")
            .replace(".zip", "")
            .replace("images_zips/", "images/"),
        )
        unzip(zip_p, out_p)


if __name__ == "__main__":
    main()
