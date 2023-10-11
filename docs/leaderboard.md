# ARCTIC Leaderboard

This page contains instructions to submit your results to the ARCTIC leaderboard. **The leaderboard is currently under beta release. Should you encounter any issues feel free to contact us.** 

## Getting an account

To get started, go to our [leaderboard website](https://arctic-leaderboard.is.tuebingen.mpg.de/). Click on the `Sign up` button at the top to register a new account. You will receive an email to confirm your account. Note that this is not the same account system as in the [ARCTIC website](https://arctic.is.tue.mpg.de/), so you have to register a separate one.

> ICCV challenge participants: Use the same email you registered for the challenge to register the evaluation account. Further, your "Algorithm Name" should be your team name.

After activating your account, you may now log into the website. 

## Creating an algorithm


After logging in, click "My algorithms" at the top to manage your algorithms. To add one, click "Add algorithm" and enter your algorithm details, with only the "short name" field being mandatory. This information will appear on the leaderboard if published. Algorithm scores remain private by default unless published. Click save to create the algorithm.

**IMPORTANT: When an algorithm is created, you can submit on multiple sub-tasks. You do not need to create a separate algorithm for each sub-task**

## Submitting to leaderboard

After the step above, you'll reach a page to upload prediction results. Initially, use our provided CVPR model zip files below for a trial evaluation based on your chosen task. Post trial, you can submit your own zip files. We recommend starting with egocentric tasks due to their smaller file sizes:

- [Consistent motion reconstruction: allocentric](https://download.is.tue.mpg.de/arctic/submission/pose_p1_test.zip)
- [Consistent motion reconstruction: egocentric](https://download.is.tue.mpg.de/arctic/submission/pose_p2_test.zip)
- [Interaction field estimation: allocentric](https://download.is.tue.mpg.de/arctic/submission/field_p1_test.zip)
- [Interaction field estimation: egocentric](https://download.is.tue.mpg.de/arctic/submission/field_p2_test.zip)



Click "Upload" and select the relevant task to upload your zip file for evaluation. The evaluation time may vary based on the task and number of requests. You'll see the results in a table, which can be downloaded as a JSON file by clicking "evaluation result".

Your numbers should closely align with our CVPR models, serving as a sanity check for the file format. Results remain private unless you select "publish", allowing evaluation against the test set ground truth.

To generate zip files for evaluation, create a custom script using the provided zip files as a template. If using our original codebase, utilize the extraction scripts below to create the zip files. Find detailed data format documentation for the leaderboard [here](leaderboard_format.md).

To avoid excessive hyperparameter tuning on the test set, each account can only submit to the server for **10 successful evaluations in total every month**. 

## Preparing submission file with original codebase

We demonstrate preparing submission files using ARCTIC models as an example. First, run inference on each sequence and save the model predictions to disk. These predictions are then compiled into a zip file for submission. 

> If you're using a different codebase, and prefer to write your own script for generating the zip files, you can inspect the example zip files above. 

To submit predictions, we need to use the extraction script `scripts_method/extract_predicts.py`. Detailed documentation on the extraction script is at [here](model/extraction.md). 

To perform a trial submission, you can try to reproduce numbers on our model `28bf3642f`. It is a ArcticNet-SF model for the egocentric setting in our CVPR paper. See details on the [data documentation](data/data_doc.md) page.

If you have prepared the arctic data following our standard instructions [here](data/README.md), you can copy the pre-trained model `28bf3642f` via:

```bash
cp -r data/arctic_data/models/28bf3642f logs/
```

Then run this command to perform inference on the test set:

```bash
python scripts_method/extract_predicts.py --setup p2 --method arctic_sf --load_ckpt logs/28bf3642f/checkpoints/last.ckpt --run_on test --extraction_mode submit_pose
```

A zip file will be produced, which can be used to upload to the evaluation server.

Explanation on the options above:

- `--setup`: allocentric setting (`p1`) or egocentric setting (`p2`) to run on.
- `--method`: the model to construct
- `--load_ckpt`: path to model checkpoint
- `--run_on`: test set evaluation
- `--extraction_mode {submit_pose, submit_field}`: this specifies the extraction is for submission
