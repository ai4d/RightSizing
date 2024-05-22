import os
import os.path as osp
from glob import glob

import pandas as pd
from tqdm import tqdm
import igl

import torch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.utils import to_undirected


def read_mesh(path):
    v, f = igl.read_triangle_mesh(path)
    face = torch.from_numpy(f).T.type(torch.long)
    x = torch.from_numpy(v.astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index)


class CoMA(InMemoryDataset):
    # In the InMemoryDataset, 
    # `self.raw_dir` is the directory: root/raw
    # `self.raw_paths` is the list of file paths: [root/raw/human-body2.zip, root/raw/metric.csv]
    
    # We may not need `self.split` and `self.text_exp` attribute
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None):

        if not osp.exists(osp.join(root, 'processed')):
            os.makedirs(osp.join(root, 'processed'))

        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['human-body2.zip', 'metric.csv']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download human-body2.zip move it to {}'.format(self.raw_dir))

    def process(self):
        print('Processing...')
        
        # read the csv file        
        df = pd.read_csv(self.raw_paths[1])
        
        # extract the data
        fps = glob(osp.join(self.raw_dir, '*/*.ply'))
        if len(fps) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            fps = glob(osp.join(self.raw_dir, '*/*.ply'))
        
        # split the data into train and test
        train_data_list, test_data_list = [], []
        for idx, fp in enumerate(tqdm(fps)):
            
            # extract the attribtues (height, arm length, crotch height...) from the csv file
            filename = os.path.basename(fp)
            row = df.loc[df['subject'] == filename]
            
            height = row['heights'].values[0]
            arm_length = row['arm_lengths'].values[0]
            crotch_height = row['crotch_heights'].values[0]
            chest_circumference = row['chest_circumferences'].values[0]
            hip_circumference = row['hip_circumferences'].values[0]
            waist_circumference = row['waist_circumferences'].values[0]         
                    
            # read the mesh data
            mesh_data = read_mesh(fp)

            if self.pre_transform is not None:
                mesh_data = self.pre_transform(mesh_data)

            # compress the data into a tuple
            data = Data(x=mesh_data.x, edge_index=mesh_data.edge_index, 
                        y=torch.Tensor([height, arm_length, crotch_height, 
                                        chest_circumference, hip_circumference, waist_circumference]))            
            if (idx % 100) < 10:
                test_data_list.append(data)
            else:
                train_data_list.append(data)

        # save the data into the processed directory
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])


# used for testing
if __name__ == "__main__":
    train_dataset = CoMA('data/human-body2',train=True, transform=None, pre_transform=None)