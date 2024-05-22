# This file is part of Python 3D Viewer
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

import os.path as osp

import moderngl
import struct
import glfw
import imgui
import numpy as np
import pickle as pk
import torch

from augen import App, Camera
from augen.mesh import ObjMesh, RenderedMesh

from models import VAE_coma 
from preprocess import mesh_sampling_method

class MyApp(App):
    def init(self):
        ctx = self.ctx
        # Load the average template mesh
        # self.mesh = ObjMesh("sample-data/temp_ave_body_male_and_female_2169_20K.obj")
        self.mesh = ObjMesh("sample-data/template2.ply")
        
        # Load the glsl program
        self.program = ctx.program(
            vertex_shader=open("shaders/mesh.vert.glsl").read(),
            fragment_shader=open("shaders/mesh.frag.glsl").read(),
        )

        # load model
        device = torch.device('cuda', 0)
        
        self.mean = torch.load('sample-data/mean.pt')
        self.std = torch.load('sample-data/std.pt')
        
        edge_index_list, down_transform_list, up_transform_list = \
        mesh_sampling_method(data_fp='sample-data', template_fp='sample-data/template2.ply', ds_factors=[4, 4, 4, 4], device=device)
        
        self.model = VAE_coma(in_channels = 3,
                         out_channels = [16, 16, 16, 32],
                         latent_channels = 8,
                         edge_index = edge_index_list,
                         down_transform = down_transform_list,
                         up_transform = up_transform_list,
                         K=6).to(device)    
        
        # model_path = "pca_coma_result/20230318-100031/model.pth"
        # model_path = "pca_coma_result/20230319-195155/model.pth"
        # model_path = "pca_coma_result/20230319-211707/model.pth"
        # model_path = "pca_coma_result/20230319-224039/model.pth"
        # model_path = "pca_coma_result/20230320-001111/model.pth"
        # model_path = "pca_coma_result/20230320-010712/model.pth"
        # model_path = "pca_coma_result/20230323-195945/model.pth"

        # model_path = "deriv_result/vae_coma/20230511-230234/model.pth"
        # model_path = "deriv_result/vae_coma/20230512-002114/model.pth"
        # model_path = "deriv_result/vae_coma/20230517-092154/model.pth"
        # model_path = "deriv_result/vae_coma/20230519-002022/model.pth"
        # model_path = "deriv_result/vae_coma/20230526-104922/model.pth"
        
        # for predictor trainer2
        # model_path = "predictor/vae_coma/20230615-121204/model.pth"
        # model_path = "predictor/vae_coma/20230615-132301/model.pth"
        # model_path = "predictor/vae_coma/20230615-161501/model.pth"

        # for predictor trainer1
        # model_path = "predictor/vae_coma/20230615-204300/model.pth"
        # model_path = "predictor/vae_coma/20230615-224233/model.pth"

        # for predictor trainer4
        # model_path = "predictor/vae_coma/20230616-090456/model.pth"

        # for predictor trainer6
        # model_path = "predictor/vae_coma/20230620-002501/model.pth"

        # for feature selection trainer1
        # model_path = "feature_selection/vae_coma/20230620-160137/model.pth"

        # for feature selection trainer2
        # model_path = "feature_selection/vae_coma/20230620-215308/model.pth"

        # for vae
        # model_path = "out/vanilla_vae/20230829-104825/model.pth"
        # for beta vae
        # model_path = "out/beta_annealing/20230829-112058/model.pth"
        # for tc vae
        # model_path = "out/tc_vae/20230829-114943/model.pth"
        # for height derivative
        # model_path = "out/height_deriv/20230831-120341/model.pth"

        # model_path = "predictor/vae_coma/trainer1/20231124-185717/model.pth" # height + waist + chest (not related)
        # model_path = "predictor/vae_coma/trainer1/20231124-205554/model.pth" # height + waist + chest (related)
        
        # model_path = "predictor/vae_coma/trainer2/20231205-222033/model.pth" # height + waist
        # model_path = "predictor/vae_coma/trainer2/20231206-180205/model.pth" # height + waist
        # model_path = "predictor/vae_coma/trainer2/20231206-191601/model.pth" # height + waist
        
        # model_path = "predictor/vae_coma/trainer3/20231205-232432/model.pth" # height + waist + Lap Loss (some weried result)
        
        # model_path = "predictor/vae_coma/traienr5/20231201-215718/model.pth" 
        # model_path = "predictor/vae_coma/trainer5/20231202-000057/model.pth"
        
        # model_path = "predictor/vae_coma/trainer6/20231208-230432/model.pth"
        # model_path = "predictor/vae_coma/trainer6/20231209-211236/model.pth"
        # model_path = "predictor/vae_coma/trainer6/20231209-225502/model.pth"
        # model_path = "predictor/vae_coma/trainer6/20231214-213348/model.pth"
        # model_path = "predictor/vae_coma/trainer6/20231214-224833/model.pth"
        # model_path = "predictor/vae_coma/trainer6/20231215-142200/model.pth"
        # model_path = "predictor/vae_coma/trainer6/20231215-163433/model.pth"
        
        # model_path = "predictor/vae_coma/trainer7/20231215-175711/model.pth"
        # model_path = "predictor/vae_coma/trainer7/20231215-200003/model.pth"
        # model_path = "predictor/vae_coma/trainer7/20231215-213106/model.pth"
        # model_path = "predictor/vae_coma/trainer7/20231215-235400/model.pth"
        # model_path = "predictor/vae_coma/trainer7/20231216-104212/model.pth" # height + waist + chest
        
        # model_path = "predictor/vae_coma/trainer8/20231216-185232/model.pth"
        
        # model_path = "predictor/vae_coma/trainer9/20231217-111243/model.pth"
        
        # model_path = "predictor/vae_coma/trainer10/20231217-173912/model_500.pth"
        # model_path = "predictor/vae_coma/trainer10/20231218-122822/model.pth" # height + waist + chest + hip + arm + crotch height
        # model_path = "predictor/vae_coma/trainer10/20231219-175042/model.pth" # height + waist + chest + hip + arm + crotch (reconstruction coefficient 0.1)
        
        # model_path = "predictor/vae_coma/trainer11/20231219-001823/model.pth"
        # model_path = "predictor/vae_coma/trainer11/20231219-110852/model.pth"
        
        # model_path = "predictor/vae_coma/trainer12/20231220-123538/model.pth"
        # model_path = "predictor/vae_coma/trainer6/20240117-190657/model.pth"
        # model_path = "predictor/vae_coma/trainer13/20240119-003904/model.pth"
        # model_path = "predictor/vae_coma/trainer16/20240122-000942/model.pth"
        model_path = "predictor/vae_coma/trainer17/20240123-140521/model_900.pth"
        # model_path = "predictor/vae_coma/trainer18/20240123-190744/model_500.pth"
        self.model.load_state_dict(torch.load(model_path))
           
        # load PCA
        # pca = pk.load(open("sample-data/pca20K_on_2169.pkl", 'rb'))
        # print('pca._components_:', pca.n_components_)
        # std = np.sqrt(pca.explained_variance_)
        # print('std: ', std[0:5])
        # self.pca = pca
        # self.std = std

        # Create the rendered mesh from the mesh and the program
        self.rendered_mesh = RenderedMesh(ctx, self.mesh, self.program)

        # Setup camera
        w, h = self.size()
        self.camera = Camera(w, h)

        # Initialize some value used in the U
        self.nsliders = 8
        self.changed = [False for i in range(self.nsliders)]
        self.slider = [0.0 for i in range(self.nsliders)]

    def update(self, time, delta_time):
        # Update damping effect (and internal matrices)
        self.camera.update(time, delta_time)

    def render(self):
        ctx = self.ctx
        self.camera.set_uniforms(self.program)

        #ctx.screen.clear(1.0, 1.0, 1.0, -1.0)
        ctx.screen.clear(36.0/256, 54.0/256, 92.0/256, -1.0)

        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # scale according to the slider
        for i in range(self.nsliders):
            if self.changed[i]:
                self.mesh.update_mesh(self.slider, self.model, self.std, self.mean)
                self.rendered_mesh = RenderedMesh(ctx, self.mesh, self.program)
        self.rendered_mesh.render(ctx)

    def on_key(self, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE:
            self.should_close()

    def on_mouse_move(self, x, y):
        self.camera.update_rotation(x, y)

    def on_mouse_button(self, button, action, mods):
        if action == glfw.PRESS and button == glfw.MOUSE_BUTTON_LEFT:
            x, y = self.mouse_pos()
            self.camera.start_rotation(x, y)
        if action == glfw.RELEASE and button == glfw.MOUSE_BUTTON_LEFT:
            self.camera.stop_rotation()

    def on_resize(self, width, height):
        self.camera.resize(width, height)
        self.ctx.viewport = (0, 0, width,  height-100)

    def on_scroll(self, x, y):
        self.camera.zoom(y)

    def ui(self):
        """Use the imgui module here to draw the UI"""
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Esc', False, True
                )

                if clicked_quit:
                    self.should_close()

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Sliders", False)
        self.shape_need_update = False

        # create sliders
        for i in range(0, self.nsliders):
            if i == 0:
                self.changed[i], self.slider[i] = imgui.slider_float(
                    f"Height", self.slider[i],
                    min_value=-2, max_value=2,        # -5, 1
                    format="%.02f"
                )
            elif i == 1:
                self.changed[i], self.slider[i] = imgui.slider_float(
                    f"Waist", self.slider[i],
                    min_value=-2, max_value=2,        # -2, 6
                    format="%.02f"
                )
            elif i == 2:
                self.changed[i], self.slider[i] = imgui.slider_float(
                    f"Shoulder", self.slider[i],
                    min_value=-3, max_value=3,       # -2, 6
                    format="%.02f"
                )
            # elif i == 3:
            #     self.changed[i], self.slider[i] = imgui.slider_float(
            #         f"Hip", self.slider[i],
            #         min_value=-6, max_value=6,     # -2, 6
            #         format="%.02f"
            #     )
            # elif i == 4:
            #     self.changed[i], self.slider[i] = imgui.slider_float(
            #         f"Arm", self.slider[i],
            #         min_value=-3, max_value=4.5,    # -2, 6
            #         format="%.02f"
            #     )
            # elif i == 5:
            #     self.changed[i], self.slider[i] = imgui.slider_float(
            #         f"Crotch", self.slider[i],
            #         min_value=-5, max_value=5,    # -4, 2
            #         format="%.02f"
            #     )
            else:
                self.changed[i], self.slider[i] = imgui.slider_float(
                    f"slider_{i}", self.slider[i],
                    min_value=-4, max_value=4,
                    format="%.02f"
                )
        imgui.end()

def main():
    app = MyApp(1200, 720, "Python 3d Viewer")
    app.main_loop()

if __name__ == "__main__":
    main()

