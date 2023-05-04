import numpy as np
import trimesh

colors = {
    "pink": [1.00, 0.75, 0.80],
    "purple": [0.63, 0.13, 0.94],
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "yellow": [1.0, 1.0, 0],
    "brown": [1.00, 0.25, 0.25],
    "blue": [0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "orange": [1.00, 0.65, 0.00],
    "grey": [0.75, 0.75, 0.75],
    "black": [0.0, 0.0, 0.0],
}


class Mesh(trimesh.Trimesh):
    def __init__(
        self,
        filename=None,
        v=None,
        f=None,
        vc=None,
        fc=None,
        process=False,
        visual=None,
        **kwargs
    ):
        if filename is not None:
            mesh = trimesh.load(filename, process=process)
            v = mesh.vertices
            f = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(
            vertices=v, faces=f, visual=visual, process=process, **kwargs
        )

        self.v = self.vertices
        self.f = self.faces
        assert self.v is self.vertices
        assert self.f is self.faces

        if vc is not None:
            self.set_vc(vc)
            self.vc = self.visual.vertex_colors
            assert self.vc is self.visual.vertex_colors
        if fc is not None:
            self.set_fc(fc)
            self.fc = self.visual.face_colors
            assert self.fc is self.visual.face_colors

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self, color, array, ids):
        color = np.array(color)

        if color.max() <= 1.0:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vc(self, vc, vertex_ids=None):
        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_fc(self, fc, face_ids=None):
        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def cat(meshes):
        return trimesh.util.concatenate(meshes)
