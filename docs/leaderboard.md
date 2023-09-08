# ARCTIC Leaderboard

This page contains instructions to submit your results to the ARCTIC leaderboard. **The leaderboard is currently under beta release. Should you encounter any issues feel free to contact us.** 

## Getting an account

To get started, go to our [leaderboard website](https://arctic-leaderboard.is.tuebingen.mpg.de/). Click on the `Sign up` button at the top to register a new account. You will receive an email to confirm your account. Note that this is not the same account system as in the [ARCTIC website](https://arctic.is.tue.mpg.de/), so you have to register a separate one.

> ICCV challenge participants: Use the same email you registered for the challenge to register the evaluation account. Further, your "Algorithm Name" should be your team name.

After activating your account, you may now log into the website. 

## Creating an algorithm

After the login, click on `My algorithms` at the top to manage your algorithms. To add an algorithm, click on `Add algorithm` and fill in the information on your algorithm. Note that only the `short name` field is mandatory. Information here will be used to displayed on the leaderboard if you choose to publish them. By default, your algorithm scores will be private unless you publish them. Save to create the algorithm.

## Submitting to leaderboard


Following the step above, you will be directed to a page where you can upload your prediction results. Click on `Upload`, select the corresponding task to evaluate, and upload the zip file to evaluate (see below for creating such file). The file will be uploaded and evaluated against the groundtruth on the test set. Depending on the task and the amount of requests, it may take different time to evaluate. The evaluation results will appear as a table. If you click on `evaluation result`, you can download this table as a json file. Your results will not be displayed to the public unless you click on `publish`. 


To avoid excessive hyperparameter tuning on the test set, each account can only submit to the server for **10 times in total every month**. 

## Preparing submission file

To submit predictions, we need to use the extraction script `scripts_method/extract_predicts.py`. Detailed documentation on the extraction script is at [here](model/extraction.md). 

To perform a trial submission, you can try to reproduce numbers on our model `28bf3642f`. It is a ArcticNet-SF model for the egocentric setting in our CVPR paper. See details on the data documentation page.

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
