import json
import os.path as op
import traceback
from glob import glob
from hashlib import sha256

from tqdm import tqdm


def main():
    release_folder = "./downloads"

    print("Globing files...")
    fnames = glob(op.join(release_folder, "**/*"), recursive=True)
    print("Number of files to checksum: ", len(fnames))
    pbar = tqdm(fnames)

    with open("./bash/assets/checksum.json", "r") as f:
        gt_checksum = json.load(f)

    hash_dict = {}
    for fname in pbar:
        if op.isdir(fname):
            continue
        if ".zip" not in fname:
            continue
        if "models_smplx_v1_1.zip" in fname:
            continue
        if "mano_v1_2.zip" in fname:
            continue

        try:
            with open(fname, "rb") as f:
                pbar.set_description(f"Reading {fname}")
                data = f.read()
                hashcode = sha256(data).hexdigest()
                key = fname.replace(release_folder, "")
                hash_dict[key] = hashcode
                if hashcode != gt_checksum[key]:
                    print(f"Error: {fname} has different checksum!")
                else:
                    pbar.set_description(f"Hashcode of {fname} is correct!")
                    # print(f'Hashcode of {fname} is correct!')
        except:
            print(f"Error processing {fname}")
            traceback.print_exc()
            continue

    out_p = op.join(release_folder, "checksum.json")
    with open(out_p, "w") as f:
        json.dump(hash_dict, f, indent=4, sort_keys=True)
    print(f"Checksum file saved to {out_p}!")


if __name__ == "__main__":
    main()
