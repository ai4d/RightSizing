"""
To be honest, I don't think we need to customize the DataLoader.
Let us just use the default one. And take a look how it works.
"""
    
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        def dense_collate(data_list):
            batch = Batch()
            batch.batch = []
            
            # keys = ['x', 'edge_index', 'y']
            for key in data_list[0].keys:
                batch[key] = default_collate([d[key] for d in data_list])
            
            # have no idea what is used for? Do we really need this one?
            # let me comment it first and see what happens
            
            for i, data in enumerate(data_list):
                num_nodes = data.num_nodes
                if num_nodes is not None:
                    item = torch.full((num_nodes, ), i, dtype=torch.long)
                    batch.batch.append(item)
            batch.batch = torch.cat(batch.batch, dim=0)
            
            return batch

        super(DataLoader, self).__init__(dataset,
                                         batch_size,
                                         shuffle,
                                         collate_fn=dense_collate,
                                         **kwargs)

if __name__ == '__main__':
    from meshdata import MeshData

    dataset = MeshData(root='data/human-body2', template_fp='template/template2.ply')
    loader = DataLoader(dataset.test_dataset, batch_size=10, shuffle=True)
    for data in loader:
        print(data)