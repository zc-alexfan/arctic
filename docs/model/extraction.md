
# Extraction

To run our training (for LSTM models), evaluation, and visualization pipelines, we need to save certain predictions to disk in advance. Here we detail the extraction script options. 

## Script options

Options:
- `--setup`: the split to use; `{p1, p2}`
- `--method`: model name; `{arctic_sf, arctic_lstm, field_sf, field_lstm}`
- `--load_ckpt`: checkpoint path
- `--run_on`: split to extract prediction on; `{train, val, test}`
- `--extraction_mode`: this defines what predicted variables to extract

Explanation of `setup`:
- `p1`: allocentric split in our CVPR paper
- `p2`: egocentric split in our CVPR paper

Explanation of `--extraction_mode`:
- `eval_pose`: dump predicted variables that are related for evaluating pose reconstruction. The evaluation will be done locally (assume GT is provided).
- `eval_field`: dump predicted variables that are related for evaluating interaction field estimation. The evaluation will be done locally (assume GT is provided).
- `submit_pose`: dump predicted variables that are related for evaluating pose reconstruction. The evaluation will be done via a submission server for test set evaluation.
- `submit_field`: dump predicted variables that are related for evaluating interaction field estimation. The evaluation will be done via a submission serverfor test set evaluation.
- `feat_pose`: extract image feature vectors for pose estimation (e.g., these features are inputs of the LSTM model to avoid a backbone in the training process for speedup).
- `feat_field`: extract image feature vectors for interaction field estimation
- `vis_pose`: extract prediction for visualizing pose prediction in our viewer.
- `vis_field`: extract prediction for visualizing interaction field prediction in our viewer.

## Extraction examples

Here we show extraction examples using our pre-trained models. To start, copy our pre-trained models to `./logs`:

```bash
mkdir -p logs
cp -r data/arctic_data/models/* logs/
```

**Example**: Suppose that I want to:
- evaluate the *ArcticNet-SF* pose estimation model (`3558f1342`)
- run on the *val* set
- use the split `p1` to evaluate locally (therefore, `eval_pose`)
- use the checkpoint at `logs/3558f1342/checkpoints/last.ckpt`

```bash
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on val --extraction_mode eval_pose 
```

**Example**: Suppose that I want to:
- evaluate the *ArcticNet-SF* pose estimation model (`3558f1342`)
- run on the *test* set
- use the CVPR split `p1` to evaluate so that we submit to the evaluation server later (therefore, `submit_pose`)
- use the checkpoint at `logs/3558f1342/checkpoints/last.ckpt`

```bash
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on test --extraction_mode submit_pose 
```

**Example**: Suppose that I want to:
- visualize the prediction of the *ArcticNet-SF* pose estimation model (`3558f1342`); therefore, `vis_pose`
- run on the *val* set
- use the split `p1` to evaluate 
- use the checkpoint at `logs/3558f1342/checkpoints/last.ckpt`

```bash
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on val --extraction_mode vis_pose 
```

**Example**: Suppose that I want to:
- Extract images features of the *ArcticNet-LSTM* pose estimation model (`3558f1342`) on training and val sets. 
- use the split `p1`
- we need to first save the visual features of *ArcticNet-SF* model to disks; Therefore, `feat_pose`

```bash
# extract for training
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on train --extraction_mode feat_pose 

# extract for evaluation on val set
python scripts_method/extract_predicts.py --setup p1 --method arctic_sf --load_ckpt logs/3558f1342/checkpoints/last.ckpt --run_on val --extraction_mode feat_pose 
```

