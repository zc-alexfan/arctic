# AIT Viewer with ARCTIC

Our visualization is powered by:

<a href="https://github.com/eth-ait/aitviewer"><img src="../../docs/static/aitviewer-logo.svg" alt="Image" height="30"/></a>

## Examples

```bash
# render object and MANO for a given sequence
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s01/capsulemachine_use_01.npy --object --mano

# render object and MANO for a given sequence on view 2
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s01/capsulemachine_use_01.npy --object --mano --view_idx 2

# render object and MANO for a given sequence on egocentric view
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s01/capsulemachine_use_01.npy --object --mano --view_idx 0

# render object and MANO for a given sequence on egocentric view while taking lens distortion into account
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s01/capsulemachine_use_01.npy --object --mano --view_idx 0 --distort

# render in headless mode to obtain RGB images (with meshes), depth, segmentation masks, and mp4 video of the visualization
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s01/capsulemachine_use_01.npy --object --mano --headless

# render object and SMPLX for a given sequence without images
python scripts_data/visualizer.py --seq_p ./outputs/processed_verts/seqs/s01/capsulemachine_use_01.npy --object --smplx --no_image

# render all sequences into videos, RGB images with meshes, depth maps, and segmentation masks
python scripts_data/visualizer.py --object --smplx --headless
```

## Options

- `view_idx`: camera view to visualize; `0` is for egocentric view; `{1, .., 8}` are for 3rd-person views.
- `seq_p`: path to processed sequence to visualize. When this option is not specified, the program will run on all sequences (e.g., when you want to render depth masks for all sequences).
- `headless`: when it is off, user will be have an interactive mode; when it is on, we render and save images with GT, depth maps, segmentation masks, and videos to disks.
- `mano`: include MANO in the scene
- `smplx`: include SMPLX in the scene
- `object`: include object in the scene
- `no_image`: do not show images.
- `distort`: in egocentric view, lens distortion is servere as the camera is close to the 3D objects, leading to mismatch in 3D geometry and the images. When turned on, this option makes use of the lens distortion parameters for better GT-image overlaps by simulating the distortion effect using ["vertex displacement for distortion correction"](https://stackoverflow.com/questions/44489686/camera-lens-distortion-in-opengl). It uses the distortion parameters to distort the 3D geometry so that it has better 3D overlaps with the images. However, such a method creates artifacts when the 3D geometry is close to the camera.

## Controls to interact with the viewer

[AITViewer](https://github.com/eth-ait/aitviewer) has lots of useful builtin controls. For an explanation of the frontend and control, visit [here](https://eth-ait.github.io/aitviewer/frontend.html). Here we assume you are in interactive mode (`--headless` is turned off).

- To play/pause the animation, hit `<SPACE>`.
- To center around an object, click the mesh you want to center, press `X`.
- To go between the previous and the current frame, press `<` and `>`.

More documentation can be found in [aitviewer github](https://github.com/eth-ait/aitviewer) and in [viewer docs](https://eth-ait.github.io/aitviewer/frontend.html).
