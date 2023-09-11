# Submission format

## Consistent motion reconstruction

### File structure

To submit for evaluation, you need to prepare prediction files and store them in a folder (here the folder is `pose_p2_test`), and zip the folder for submission. The following shows the tree structure of a folder before zipping to `pose_p2_test.zip`.

The folder contains a single subfolder named `eval`. We refer `pose_p2_test` as the `TASK_NAME` to indicate different tasks to evaluate your submission on. The `$TASK_NAME/eval` folder then stores prediction from each sequence in a particular view. 

```
pose_p2_test
--  eval
    |-- s03_box_grab_01_0
    |   |-- meta_info
    |   |   `-- meta_info.imgname.pt
    |   `-- preds
    |       |-- pred.mano.beta.l.pt
    |       |-- pred.mano.beta.r.pt
    |       |-- pred.mano.cam_t.l.pt
    |       |-- pred.mano.cam_t.r.pt
    |       |-- pred.mano.pose.l.pt
    |       |-- pred.mano.pose.r.pt
    |       |-- pred.object.cam_t.pt
    |       |-- pred.object.radian.pt
    |       `-- pred.object.rot.pt
    |-- s03_box_use_01_0
    |   |-- meta_info
    |   |   `-- meta_info.imgname.pt
    |   `-- preds
    |       |-- pred.mano.beta.l.pt
    |       |-- pred.mano.beta.r.pt
    |       |-- pred.mano.cam_t.l.pt
    |       |-- pred.mano.cam_t.r.pt
    |       |-- pred.mano.pose.l.pt
    |       |-- pred.mano.pose.r.pt
    |       |-- pred.object.cam_t.pt
    |       |-- pred.object.radian.pt
    |       `-- pred.object.rot.pt
    ...
```

Lets take `pose_p2_test/eval/s03_box_use_01_0` as an example. The `TASK_NAME` is `pose_p2_test` and `s03_box_use_01_0` means that the folder is for predictions of the sequence `s03_box_use_01` in camera view `0`. Since this is an egocentric task, you will expect the view is always 0, but for allocentric tasks it will range from 1 to 8.

You will use one of the following `TASK_NAME`:
- `pose_p1_test`: motion reconstruction task, allocentric setting evaluation on the test set
- `pose_p2_test`: motion reconstruction task, egocentric setting evaluation on the test set
- `field_p1_test`: interaction field estimation task, allocentric setting evaluation on the test set
- `field_p2_test`: interaction field estimation task, egocentric setting evaluation on the test set

Say you want to store your prediction on the motion reconstruction task in allocentric camera setting on the test set for camera 2 and the sequence `s03_capsulemachine_use_04`. The folder to store the prediction will be `pose_p1_test/eval/s03_capsulemachine_use_04_2`. 

### File formats

Looking at the tree structure above, you can see that there are two folders `meta_info` and `preds`. The former stores information that is not prediction. In this case, it is only the image paths. The latter folder stores the predictions of the MANO model and the object model. Each `.pt` file is from `torch.save`.

- `pred.mano.beta.l.pt`: (num_frames, 10); MANO betas for left hand for each frame; FloatTensor
- `pred.mano.cam_t.l.pt`: (num_frames, 3); MANO [translation](https://github.com/zc-alexfan/arctic/blob/08c5e9396087c4529b448cdf736b65fae600866e/src/nets/hand_heads/mano_head.py#L51) for left hand; FloatTensor
- `pred.mano.pose.l.pt`: (num_frames, 16, 3, 3); MANO hand rotations for left hand; FloatTensor; assume `flat_hand_mean=False`; this includes the global orientation; rotation matrix format.
- `pred.object.cam_t.pt`: (num_frames, 3); Object [translation](https://github.com/zc-alexfan/arctic/blob/08c5e9396087c4529b448cdf736b65fae600866e/src/nets/obj_heads/obj_head.py#L60C27-L60C32); FloatTensor
- `pred.object.radian.pt`: (num_frames); Object articulation radian.
- `pred.object.rot.pt`: (num_frames, 3); Object orientation in axis-angle; FloatTensor
- `meta_info.imgname.pt`: (num_frames); A list of strings for image paths

Example of the first image path:

```
'./data/arctic_data/data/cropped_images/s03/box_use_01/0/00010.jpg'
```

You can also refer to our hand and object model classes for a reference of these variables.

