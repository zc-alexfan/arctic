# Training our CVPR baselines

To better illustrate the training process, we give hypothetical names for each model, such as `aaaaaaaaa`.

## ArcticNet

### ArcticNet-SF: Allocentric

Model: `aaaaaaaaa`

```bash
python scripts_method/train.py --setup p1 --method arctic_sf --trainsplit train --valsplit minival
```

### ArcticNet-SF: Egocentric

Model: `bbbbbbbbb`

As per our experiment protocol, for the egocentric setting, since a model has access to both allocentric and egocentric images during training, to speed up training, we finetune pre-trained allocentric models on egocentric training images (1 camera).

To train the egocentric model, we do:

```bash
python scripts_method/train.py --setup p2 --method arctic_sf --trainsplit train --valsplit minival --load_ckpt logs/aaaaaaaaa/checkpoints/last.ckpt
```

### ArcticNet-LSTM: Allocentric

Model: `ccccccccc`

To train the LSTM model, since maintaining an image backbone is extremely costly, following [VIBE](https://github.com/mkocabas/VIBE), to train our temporal baseline, we first store image features for each training and validation images to disk and then train an LSTM to regress these features to hand and object poses directly. The following instructions explains how to 1) extract image feature to disk for the ArcticNet-SF model (`aaaaaaaaa`) backbone; 2) package the feature vectors; 3) train the LSTM model. For details of `extract_predicts.py`, see [here](extraction.md).

Extract image features from `aaaaaaaaa` backbone:

```bash
# extract image features from aaaaaaaaa on training set
# this is for training
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/aaaaaaaaa/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose

# extract image features from aaaaaaaaa on val set (or test set)
# this is for evaluation
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/aaaaaaaaa/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose
```

Packaging feature vectors for different splits:

```bash
python scripts_method/build_feat_split.py --split train --protocol p1 --eval_p logs/aaaaaaaaa/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p1 --eval_p logs/aaaaaaaaa/eval
python scripts_method/build_feat_split.py --split val --protocol p1 --eval_p logs/aaaaaaaaa/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p1 --eval_p logs/aaaaaaaaa/eval
python scripts_method/build_feat_split.py --split minival --protocol p1 --eval_p logs/aaaaaaaaa/eval
```

At the end of the packaging, there is a verification process that checks whether each image has a feature vector. If so, `Pass verification` will be printed. 

Under `src/parsers/configs/arctic_lstm.py`, update `img_feat_version`. This will decide which models' features to use to train the LSTM model. 
It will also use the single-frame model's decoder weights to initialize the LSTM model decoder. Configure image feature version to `aaaaaaaaa`:

```python
# allocentric setting
DEFAULT_ARGS_ALLO["img_feat_version"] = "aaaaaaaaa"
```

Start training:

```python
python scripts_method/train.py --setup p1 --method arctic_lstm
```

### ArcticNet-LSTM: Egocentric

Model: `ddddddddd`


Extract image features from `bbbbbbbbb` backbone:

```bash
# extract image features from bbbbbbbbb on training set
# this is for training
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/bbbbbbbbb/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose

# extract image features from bbbbbbbbb on val set (or test set)
# this is for evaluation
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/bbbbbbbbb/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose
```

Packaging feature vectors for different splits:

```bash
python scripts_method/build_feat_split.py --split train --protocol p2 --eval_p logs/bbbbbbbbb/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p2 --eval_p logs/bbbbbbbbb/eval
python scripts_method/build_feat_split.py --split val --protocol p2 --eval_p logs/bbbbbbbbb/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p2 --eval_p logs/bbbbbbbbb/eval
python scripts_method/build_feat_split.py --split minival --protocol p2 --eval_p logs/bbbbbbbbb/eval
```


Under `src/parsers/configs/arctic_lstm.py`, update `img_feat_version`. This will decide which models' features to use to train the LSTM model. 
It will also use the single-frame model's decoder weights to initialize the LSTM model decoder. Configure image feature version to `bbbbbbbbb`:

```python
# egocentric setting
DEFAULT_ARGS_EGO["img_feat_version"] = "bbbbbbbbb"
```

Start training:

```python
python scripts_method/train.py --setup p2 --method arctic_lstm
```

## InterField

### InterField-SF: Allocentric

Model: `eeeeeeeee`

```bash
# training on InterField-SF in the allocentric setting in our paper
python scripts_method/train.py --setup p1 --method field_sf --trainsplit train --valsplit minival
```

### InterField-SF: Egocentric

Model: `fffffffff`

```bash
python scripts_method/train.py --setup p2 --method field_sf --trainsplit train --valsplit minival --load_ckpt logs/eeeeeeeee/checkpoints/last.ckpt
```

### InterField-LSTM: Allocentric

Model: `ggggggggg`


```bash
# extract image features from eeeeeeeee on training set
# this is for training
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/eeeeeeeee/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose

# extract image features from eeeeeeeee on val set (or test set)
# this is for evaluation
python scripts_method/extract_predicts.py --setup p1 --method field_sf --load_ckpt logs/eeeeeeeee/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose
```

Packaging feature vectors for different splits:

```bash
python scripts_method/build_feat_split.py --split train --protocol p1 --eval_p logs/eeeeeeeee/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p1 --eval_p logs/eeeeeeeee/eval
python scripts_method/build_feat_split.py --split val --protocol p1 --eval_p logs/eeeeeeeee/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p1 --eval_p logs/eeeeeeeee/eval
python scripts_method/build_feat_split.py --split minival --protocol p1 --eval_p logs/eeeeeeeee/eval
```

At the end of the packaging, there is a verification process that checks whether each image has a feature vector. If so, `Pass verification` will be printed. 

Under `src/parsers/configs/field_lstm.py`, update `img_feat_version`. This will decide which models' features to use to train the LSTM model. 
It will also use the single-frame model's decoder weights to initialize the LSTM model decoder. Configure image feature version to `eeeeeeeee`:

```python
# allocentric setting
DEFAULT_ARGS_ALLO["img_feat_version"] = "eeeeeeeee"
```

Start training:

```python
python scripts_method/train.py --setup p1 --method field_lstm
```

### InterField-LSTM: Egocentric

Model: `hhhhhhhhh`

```bash
# extract image features from fffffffff on training set
# this is for training
python scripts_method/extract_predicts.py --setup p2 --method field_sf --load_ckpt logs/fffffffff/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose

# extract image features from fffffffff on val set (or test set)
# this is for evaluation
python scripts_method/extract_predicts.py --setup p2 --method field_sf --load_ckpt logs/fffffffff/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose
```

Packaging feature vectors for different splits:

```bash
python scripts_method/build_feat_split.py --split train --protocol p2 --eval_p logs/fffffffff/eval
python scripts_method/build_feat_split.py --split minitrain --protocol p2 --eval_p logs/fffffffff/eval
python scripts_method/build_feat_split.py --split val --protocol p2 --eval_p logs/fffffffff/eval
python scripts_method/build_feat_split.py --split tinyval --protocol p2 --eval_p logs/fffffffff/eval
python scripts_method/build_feat_split.py --split minival --protocol p2 --eval_p logs/fffffffff/eval
```

At the end of the packaging, there is a verification process that checks whether each image has a feature vector. If so, `Pass verification` will be printed. 

Under `src/parsers/configs/field_lstm.py`, update `img_feat_version`. This will decide which models' features to use to train the LSTM model. 
It will also use the single-frame model's decoder weights to initialize the LSTM model decoder. Configure image feature version to `fffffffff`:

```python
# egocentric setting
DEFAULT_ARGS_EGO["img_feat_version"] = "fffffffff"
```

Start training:

```python
python scripts_method/train.py --setup p2 --method field_lstm
```
