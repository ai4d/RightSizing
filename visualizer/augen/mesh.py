# This file is part of PyAugen
#
# Copyright (c) 2020 -- Élie Michel <elie.michel@exppad.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and non-infringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising
# from, out of or in connection with the software or the use or other dealings
# in the Software.

import numpy as np
from moderngl import TRIANGLES
from igl import read_triangle_mesh, per_vertex_normals
import torch 
from scipy.spatial.transform import Rotation as R

class Mesh:
    """Simply contains an array of triangles and an array of normals.
    Could be enhanced, for instance with an element buffer"""
    def __init__(self, P, N):
        self.P = P
        self.N = N


class ObjMesh(Mesh):
    """An example of mesh loader, using the pywavefront module.
    Only load the first mesh of the file if there are more than one."""
    def __init__(self, filepath):
        print(f"Loading mesh from {filepath} ...")
        v, f = read_triangle_mesh(filepath)
        n = per_vertex_normals(v, f)
        self.P = v[f].reshape(-1,3)
        self.N = n[f].reshape(-1,3)
        print(f"(Object has {len(self.P)//3} points)")
        self.faces = f

    def decode(self, x, model):
        num_layers = len(model.de_layers)
        num_deblocks = num_layers - 2
        for i, layer in enumerate(model.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, model.num_vert, model.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, model.edge_index[num_deblocks - i],
                            model.up_transform[num_deblocks - i])
            else:
                # last layer
                x = layer(x, model.edge_index[0])
        return x 
    
    # update mesh using PCA
    def update_mesh(self, slider_val, model, std, mean):
        torch_tensor = torch.from_numpy(np.array(slider_val)).float().cuda()
        v = self.decode(torch_tensor, model)
        v = v.reshape(-1, 3).detach().cpu().float()
        moved_vertices = ((v * std) + mean).numpy()
        r = R.from_euler('z', -45, degrees=True)
        r_matrix = r.as_matrix()
        moved_vertices = np.matmul(moved_vertices, r_matrix)
        
        self.P = moved_vertices[self.faces].reshape(-1, 3)
        
    # def update_mesh(self, slider_val, pca, std):
    #     w = np.zeros((1, pca.n_components_))
    #     for i in range(len(slider_val)):
    #         w[0, i] = std[i] * slider_val[i] * 0.1
    #     moved_vertices = pca.inverse_transform(w).reshape(-1, 3)
    #     self.P = moved_vertices[self.faces].reshape(-1, 3)
    #     # leave the normals unchanged


class RenderedMesh:
    """The equivalent of a Mesh, but stored in OpenGL buffers (on the GPU)
    ready to be rendered."""
    def __init__(self, ctx, mesh, program):
        self.mesh = mesh
        self.vboP = ctx.buffer(mesh.P.astype('f4').tobytes())
        self.vboN = ctx.buffer(mesh.N.astype('f4').tobytes())
        self.vao = ctx.vertex_array(
            program,
            [
                (self.vboP, "3f", "in_vert"),
                (self.vboN, "3f", "in_normal"),
            ]
        )

    def release(self):
        self.vboP.release()
        self.vboN.release()
        self.vao.release()

    def render(self, ctx):
        self.vao.render(TRIANGLES)
