import openmesh as om
import igl

import torch

from dataset import CoMA


class MeshData(object):
    def __init__(self, root, template_fp, transform=None, pre_transform=None):
        
        self.root = root
        self.template_fp = template_fp
        self.transform = transform
        self.pre_transform = pre_transform
        self.train_dataset = None
        self.test_dataset = None
        self.template_points = None
        self.template_face = None
        self.mean = None
        self.std = None
        self.num_nodes = None

        self.load()

    def load(self):
        # Need to change the human-body class to make it work
        self.train_dataset = CoMA(self.root, train=True, transform=self.transform, pre_transform=self.pre_transform)
        self.test_dataset = CoMA(self.root, train=False, transform=self.transform, pre_transform=self.pre_transform)

        v, f = igl.read_triangle_mesh(self.template_fp)
        self.template_points = v
        self.template_face = f
        self.num_nodes = self.train_dataset[0].num_nodes

        self.num_train_graph = len(self.train_dataset)
        self.num_test_graph = len(self.test_dataset)

        self.mean = self.train_dataset.data.x.view(self.num_train_graph, -1, 3).mean(dim=0)
        self.std = self.train_dataset.data.x.view(self.num_train_graph, -1, 3).std(dim=0)

        self.normalize()

    def normalize(self):
        print('Normalizing the training and testing dataset...')
        mesh_set = self.train_dataset.data.x.view(self.num_train_graph, -1, 3)
        test_set = self.test_dataset.data.x.view(self.num_test_graph, -1, 3)
        self.train_dataset.data.x = torch.where(torch.abs(self.std) < 1e-5, mesh_set - self.mean,
                                      torch.divide(mesh_set - self.mean, self.std)).view(-1, 3)
        self.test_dataset.data.x= torch.where(torch.abs(self.std) < 1e-5, test_set - self.mean,
                                      torch.divide(test_set - self.mean, self.std)).view(-1, 3)
        print('Normalization Done!')

    def save_mesh(self, fp, x):
        x = x * self.std + self.mean
        om.write_mesh(fp, om.TriMesh(x.numpy(), self.template_face))
        

# for testing
if __name__ == '__main__':
    MeshData = MeshData(root='data/human-body2', template_fp='template/template2.ply')

