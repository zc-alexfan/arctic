# ARCTIC baselines

## Table of content

- [Overview](#overview)
- [Getting started](#getting-started)
- [Training examples](#training)
- [Full instructions on training](train.md)
- [Evaluation](#evaluation)
- [Visualization examples](#visualization)
- [Details on `extract_predicts.py`](extraction.md)

## Overview

**PyTorch Lightning**: To avoid boilerplate code, we use [pytorch lightning (PL) 2.0.0](https://pytorch-lightning.readthedocs.io/en/2.0.0/common/trainer.html) to handle the main logic for training and evaluation. Feel free to consult the documentation, should you have any questions.

**Comet logger**: To keep track of experiments and visualize results, our code logs experiments using [`comet.ml`](https://comet.ml). If you wish to use own logger service, you mostly modify the code in `common/comet_utils.py`. This code is only meant as a guideline; you are free to modify it to whatever extent you deem necessary.

To configure the comet logger, you need to first register an account and create a private project. An API code will be provided for you to log the experiment. Then you export the API code and the workspace ID:

```bash
export COMET_API_KEY="your_api_key_here"
export COMET_WORKSPACE="your_workspace_here"
```

It might be a good idea to add these commands to your `~/.bashrc` file, so you don't have to load the environment every time you login to your machine. Add these lines to the end of `~/.bashrc`.

We call the allocentric split and the egocentric split in our CVPR paper the `p1` split and `p2` split respectively.

Each experiment is tracked with a 9-character ID. When the training procedure starts, a random ID (e.g., `837e1e5b2`) is assigned to the experiment and a folder (e.g., `logs/837e1e5b2`) to save information on this folder.

## Getting started

Here we assume you have: 
1. Finished setting up the environment.
2. Downloaded and unzipped the data following [`docs/data/README.md`](../data/README.md).
3. Finished configuring your logger.

To use the data, you need to move them from `unpack`:

```bash
mv unpack data
```

You should have a file structure like this after running `ls data/*`:

```
data/arctic_data:
data  models

data/body_models:
mano  smplx
```

To sanity check your setup, lets run two dry-run training procedures. We call the allocentric split and the egocentric split in our CVPR paper the `p1` split and `p2` split respectively.

```bash
# train ArcticNet-SF on a very small ARCTIC dataset (allocentric split)
python scripts_method/train.py --setup p1 --method arctic_sf -f --mute --num_epoch 2
```

```bash
# train Field-SF on a very small ARCTIC dataset (egocentric split)
python scripts_method/train.py --setup p2 --method field_sf -f --mute --num_epoch 2
```

Now you enable your comet logger, by removing `--mute` to start logging to comet server. A url will be generated so that you can visualize the prediction of your model in training, and its corresponding groundtruth:

```bash
python scripts_method/train.py --setup p1 --method arctic_sf -f --num_epoch 2
```

Click on any example in `Graphics` of your comet experiment. If you can see that the hands and objects are in the left column, the groundtruth (mostly) is overlaid on the image, and the training finished 2 epochs, then your environment is in good shape. 

As a first-timer, it is normal to have some issues. See our [FAQ](../faq.md) to see if there is a solution. 

## Training

⚠️ As per our CVPR paper, our evaluation protocol requires models to be trained only on the training set. You may not train on the validation set. You may use the validation set for hyper parameter search. The test set groundtruth is hidden for online evaluation. Here we provide instructions to train on the training set and evaluate on the validation set.

Here we detail some options in the `argparse` parsers we use. There are other options in the argparser. You can check `src/parsers` for more details.

- `-f`: Run on a mini training and validation set (i.e., a dry-run). 
- `--mute`: Do not create a new experiment on the remote comet logger.
- `--setup`: name of the split to use. A split is a partition of ARCTIC data into training, validation and test sets.
- `--trainsplit`: Training split to use
- `--valsplit`: Split to use to evaluate during training

The following code trains the single-frame allocentric baselines in our paper in the allocentric setting (i.e., the `p1` split). For the complete training guide for all models, please refer to [`train.md`](train.md)

```bash
# training on ArcticNet-SF in the allocentric setting in our paper
python scripts_method/train.py --setup p1 --method arctic_sf --trainsplit train --valsplit minival
```

```bash
# training on InterField-SF in the allocentric setting in our paper
python scripts_method/train.py --setup p1 --method field_sf --trainsplit train --valsplit minival
```

Since you have the groundtruth in this split, you can view your metrics and losses under `Panels` of your comet experiment. You can also see the visualization of the prediction and groundtruth under `Graphics` in your comet experiment.

## Evaluation

Here we show evaluation steps using our pre-trained models. Copy our pre-trained models to `./logs`:

```bash
cp -r data/arctic_data/models/* logs/
```

Our evaluation process consists of two steps: 1) dumping predictions to disk (`extract_predicts.py`); 2) evaluating the prediction against the GT (`evaluate_metrics.py`). 

Here we assume you are using the `p1` or `p2` splits and we show instructions for local evaluation. 

⚠️ Instructions for `p1` and `p2` evaluation for the test set will be provided once the evaluation server is online.

### Evaluate ArcticNet

Evaluate allocentric ArcticNet-SF on val set on `p1` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/3558f1342/eval --split val --setup p1 --task pose
```

Evaluate allocentric ArcticNet-LSTM on val set on `p1` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p1 --method arctic_lstm --load_ckpt logs/423c6057b/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/423c6057b/eval --split val --setup p1 --task pose
```

Evaluate egocentric ArcticNet-SF on val set on `p2` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/28bf3642f/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/28bf3642f/eval --split val --setup p2 --task pose
```

Evaluate egocentric ArcticNet-LSTM on val set on `p2` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p2 --method arctic_lstm --load_ckpt logs/40ae50712/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose
python scripts_method/evaluate_metrics.py --eval_p logs/40ae50712/eval --split val --setup p2 --task pose
```

## Evaluate InterField

Evaluate allocentric InterField-SF on val set on `p1` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/1f9ac0b15/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/1f9ac0b15/eval --split val --setup p1 --task field
```

Evaluate allocentric InterField-LSTM on val set on `p1` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p1 --method field_lstm --load_ckpt logs/546c1e997/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/546c1e997/eval --split val --setup p1 --task field
```

Evaluate egocentric InterField-SF on val set on `p2` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p2 --method field_sf --load_ckpt logs/58e200d16/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/58e200d16/eval --split val --setup p2 --task field
```

Evaluate egocentric InterField-LSTM on val set on `p2` split:
```bash
# PLACEHOLDERS
python scripts_method/extract_predicts.py --setup p2 --method field_lstm --load_ckpt logs/701a72569/checkpoints/last.ckpt --run_on val --extraction_mode eval_field
python scripts_method/evaluate_metrics.py --eval_p logs/701a72569/eval --split val --setup p2 --task field
```

For details of `extract_predicts.py`, see [here](extraction.md).

## Visualization

We will use `scripts_method/visualizer.py` to visualize model performance and its corresponding GT. Unlike `scripts_data/visualizer.py`, here we shows the actual model input and output. Therefore, images are not full-resolution.

Options for `visualizer.py`:
- `--exp_folder`: the path to the experiment; the prediction will be saved there after the extraction.
- `--seq_name`: sequence to visualize. For example, `s03_box_grab_01_1` denotes the sequence `s03_box_grab_01` and camera id `1`. 
- `--mode`: defines what to visualize; `{gt_mesh, pred_mesh, gt_field_r, gt_field_l, pred_field_r, pred_field_l}`
- `--headless`: headless rendering. It renders the mesh2rgb overlay video. You can also render other modalities such as segmentation masks.

Examples to visualize pose estimation:

```bash
# PLACEHOLDERS
# dump predictions
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on val --extraction_mode vis_pose

# visualize gt_mesh
python scripts_method/visualizer.py --exp_folder logs/3558f1342 --seq_name s03_box_grab_01_1 --mode gt_mesh

# visualize pred_mesh
python scripts_method/visualizer.py --exp_folder logs/3558f1342 --seq_name s03_box_grab_01_1 --mode pred_mesh
```

Examples to visualize interaction field estimation:

```bash
# PLACEHOLDERS
# dump predictions
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/1f9ac0b15/checkpoints/last.ckpt --run_on val --extraction_mode vis_field

# visualize gt field for right hand
python scripts_method/visualizer.py --exp_folder logs/1f9ac0b15 --seq_name s03_box_grab_01_1 --mode gt_field_r

# visualize predicted field for left hand
python scripts_method/visualizer.py --exp_folder logs/1f9ac0b15 --seq_name s03_box_grab_01_1 --mode pred_field_l
```

For details of `extract_predicts.py`, see [here](extraction.md).

