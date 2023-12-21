
import numpy as np
import argparse
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.scene.material import Material
from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocap_p", type=str, default=None)
    config = parser.parse_args()
    args = EasyDict(vars(config))
    return args
    

def main():
    args = parse_args()
    mocap_p = args.mocap_p
    data = np.load(mocap_p, allow_pickle=True).item()

    from aitviewer.viewer import Viewer
    v = Viewer(size=(2048, 1024))
    materials = {
        "subject": Material(color=(0.44, 0.56, 0.89, 1.0), ambient=0.35),
        "object": Material(color=(0.969, 0.969, 0.969, 1.0), ambient=0.35),
        "egocam": Material(color=(0.24, 0.2, 0.2, 1.0), ambient=0.35),
        "table": Material(color=(0.24, 0.2, 0.2, 1.0), ambient=0.35),
        "support": Material(color=(0.969, 0.106, 0.059, 1.0), ambient=0.35),
    }


    for key, subject in data.items():
        marker_names = subject['labels']
        print(marker_names)
        marker_pos = subject['points']/1000 # frame, marker, xyz
        rotation_flip = aa2rot_numpy(np.array([-1/2, 0, 0]) * np.pi)
        spheres = Spheres(marker_pos, rotation=rotation_flip, name=key, material=materials[key])
        v.scene.add(spheres)
        
    fps = 30
    v.playback_fps = fps
    v.scene.fps = fps
    v.run()
    
    
if __name__ == "__main__":
    main()