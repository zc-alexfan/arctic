

## Getting started 

General Requirements:

- Python 3.10
- torch 1.13.0
- CUDA 11.6 (check `nvcc --version`)
- pytorch3d 0.7.3
- pytorch-lightning 2.0.0
- aitviewer 1.8.0

Install the environment:

```bash
ENV_NAME=arctic_env
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME
```

Check your CUDA `nvcc` version:

```
nvcc --version # should be 11.6
```

You can install nvcc and cuda via [runfile](https://developer.nvidia.com/cuda-11-6-0-download-archive). If `nvcc --version` is still not `11.6`, check whether you are referring the right nvcc with `which nvcc`. Assuming you have an NVIDIA driver installed, usually, you only need to run the following command to install `nvcc` (as an example):

```bash
sudo bash cuda_11.6.0_510.39.01_linux.run --toolkit --silent --override
```

After the installation, make sure the paths pointing to the current cuda toolkit location. For example:

```bash
export CUDA_HOME=/usr/local/cuda-11.6
export PATH="/usr/local/cuda-11.6/bin:$PATH"
export CPATH="/usr/local/cuda-11.6/include:$CPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64/"
```

Install packages:

```bash
pip install -r requirements.txt
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install PyTorch3D:

```bash
# pytorch3d 0.7.3
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

Install this version of numpy to avoid conflicts:

```bash
pip install numpy==1.22.4
```

Modify `smplx` package to return 21 joints for instead of 16:

```bash
vim /home/<user_name>/anaconda3/envs/<env_name>/lib/<python_version>/site-packages/smplx/body_models.py

# uncomment L1681
joints = self.vertex_joint_selector(vertices, joints)
```

If you are unsure about where `body_models.py` is, run these on a terminal:

```bash
python
>>> import smplx
>>> print(smplx.__file__)
```

