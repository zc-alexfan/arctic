import copy
import os

import numpy as np
import pyrender
import trimesh

# offline rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"


def flip_meshes(meshes):
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    for mesh in meshes:
        mesh.apply_transform(rot)
    return meshes


def color2material(mesh_color: list):
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        alphaMode="OPAQUE",
        baseColorFactor=(
            mesh_color[0] / 255.0,
            mesh_color[1] / 255.0,
            mesh_color[2] / 255.0,
            0.5,
        ),
    )
    return material


class Renderer:
    def __init__(self, img_res: int) -> None:
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res, viewport_height=img_res, point_size=1.0
        )

        self.img_res = img_res

    def render_meshes_pose(
        self,
        meshes,
        image=None,
        cam_transl=None,
        cam_center=None,
        K=None,
        materials=None,
        sideview_angle=None,
    ):
        # unpack
        if cam_transl is not None:
            cam_trans = np.copy(cam_transl)
            cam_trans[0] *= -1.0
        else:
            cam_trans = None
        meshes = copy.deepcopy(meshes)
        meshes = flip_meshes(meshes)

        if sideview_angle is not None:
            # center around the final mesh
            anchor_mesh = meshes[-1]
            center = anchor_mesh.vertices.mean(axis=0)

            rot = trimesh.transformations.rotation_matrix(
                np.radians(sideview_angle), [0, 1, 0]
            )
            out_meshes = []
            for mesh in copy.deepcopy(meshes):
                mesh.vertices -= center
                mesh.apply_transform(rot)
                mesh.vertices += center
                # further away to see more
                mesh.vertices += np.array([0, 0, -0.10])
                out_meshes.append(mesh)
            meshes = out_meshes

        # setting up
        self.create_scene()
        self.setup_light()
        self.position_camera(cam_trans, K)
        if materials is not None:
            meshes = [
                pyrender.Mesh.from_trimesh(mesh, material=material)
                for mesh, material in zip(meshes, materials)
            ]
        else:
            meshes = [pyrender.Mesh.from_trimesh(mesh) for mesh in meshes]

        for mesh in meshes:
            self.scene.add(mesh)

        color, valid_mask = self.render_rgb()
        if image is None:
            output_img = color[:, :, :3]
        else:
            output_img = self.overlay_image(color, valid_mask, image)
        rend_img = (output_img * 255).astype(np.uint8)
        return rend_img

    def render_rgb(self):
        color, rend_depth = self.renderer.render(
            self.scene, flags=pyrender.RenderFlags.RGBA
        )
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        return color, valid_mask

    def overlay_image(self, color, valid_mask, image):
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        return output_img

    def position_camera(self, cam_transl, K):
        camera_pose = np.eye(4)
        if cam_transl is not None:
            camera_pose[:3, 3] = cam_transl

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        self.scene.add(camera, pose=camera_pose)

    def setup_light(self):
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        self.scene.add(light, pose=light_pose)

    def create_scene(self):
        self.scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
