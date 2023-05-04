# Data processing & splits

## Data splits

**CVPR paper splits**

- protocol 1: allocentric split (test set GT is hidden)
- protocol 2: egocentric split (test set GT is hidden)

Note, allocentric training images in protocol 1 can be used for protocol 2 training as per our evaluation protocol in the paper. In our paper, we first pre-train on protocol 1 images and finetune on protocol 2 for the egocentric regressor. If one wants to directly train on allocentric and egocentric training images for protocol 2 evaluation, she can create a custom split.

See [`docs/data_doc.md`](../data_doc.md) for an explanation of each file in the `arctic_data` folder.

## Advanced usage

### Process raw sequences

```bash
# process a specific seq; do not save vertices for smaller storage
python scripts_data/process_seqs.py --mano_p ./unpack/arctic_data/data/raw_seqs/s01/espressomachine_use_01.mano.npy

# process all seqs; do not save vertices for smaller storage
python scripts_data/process_seqs.py

# process all seqs while exporting the vertices for visualization
python scripts_data/process_seqs.py --export_verts
```

### Create data split from processed sequences

Our baseline load the pre-processed split from `data/arctic_data/data/splits`. In case you need a custom split, you can build a data split from the example below (here we show validation set split), which generates the split files under `outputs/splits/`

Build a data split from processed sequence:

```bash
# Build validation set based on protocol p1 defined at arctic_data/data/splits_json/protocol_p1.json
python scripts_data/build_splits.py --protocol p1 --split val --process_folder ./outputs/processed/seqs

# Same as above, but build with vertices too
python scripts_data/build_splits.py --protocol p1 --split val --process_folder ./outputs/processed_verts/seqs
```

⚠️ The dataloader for our models in our CVPR paper does not require vertices in the split files. If the processed sequences are built with `--export_verts`, this script will try to aggregate the vertices as well, leading to large storage requirement.

### Crop images for faster data loading

Since our images are of high resolution, if reading speed is a limitation for your machine for training models, one can consider cropping the images around a larger region centered at the bounding boxes to reduce data loading requirement in training. We have provided data link for pre-cropped images. In case of a custom crop, one can use the script below:

```bash
# crop all images from all sequences using bbox defined in the process folder on a single machine
python scripts_data/crop_images.py --task_id -1 --process_folder ./outputs/processed/seqs

# crop all images from one sequence using bbox defined in the process folder
# this is used for cluster preprocessing where AGENT_ID is from 0 to num_nodes-1
python scripts_data/crop_images.py --task_id AGENT_ID --process_folder ./outputs/processed/seqs
```
