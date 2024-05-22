import os
import igl
import numpy as np

import torch
from pytorch3d.ops import cot_laplacian

def laplacian(vertex: torch.Tensor, faces: torch.Tensor, type: str) -> torch.Tensor:
    """
    Compute the laplacian of a mesh.
    Args:
        vertex: FloatTensor of shape (V, 3) giving vertex positions for V vertices.
        faces: LongTensor of shape (F, 3) giving faces.
        type: String giving the type of laplacian to compute. Must be either 'mean' or 'vertex'.
              mean: compute the mean laplacian value for all vertices.
              vertex: compute the laplacian value for each vertex.
    Returns:
        laplacian: FloatTensor of shape (V, 1) giving the laplacian matrix for the mesh.
                   OR, retuen the mean laplacian value for all vertices.
    """
    if type not in ['mean', 'vertex', 'vector']:
        raise ValueError('type must be one of mean, vertex or vector')
    
    with torch.no_grad():
        # Compute the cotangent weights.
        L, inv_areas = cot_laplacian(vertex, faces)

    # NOTE: The diagonal entries of the cotangent weights matrix returned by the method are 0
    #       and need to be computed with the negative sum of the weights in each row.
    L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)

    # As shown in my thesis, the laplacian is given by: L_loss = L * M^{-1} * V, 
    # However, since the diagonal is zero, we modify it like: L_loss = (L - L_sum) * M^{-1} * V, 
    # where L is the cotangent weights matrix, 
    #       M is the mass matrix, with diagonal entries being 1/3 areas of the vertices,
    #       V is the vertex positions.
    
    # NOTE: I have no idea where the 0.25 comes from.
    #       In my opinion, it should be the weight should be 1/2 (cot alpha_{ij} + cot beta_{ij}).
    #       So, the coefficient should be 0.5. 
    mass_matrix_inv = 0.25 * inv_areas

    loss = (L.mm(vertex) - L_sum * vertex) * mass_matrix_inv

    laplacian_loss = torch.norm(loss, dim=1)

    if type == 'mean':
        return torch.mean(laplacian_loss)
    elif type == 'vertex':
        return laplacian_loss
    elif type == 'vector':
        return loss

def diff_laplacian(vertex: torch.Tensor) -> torch.Tensor:
    """
    Compute the difference of Laplacian of a mesh with the template mesh. 
    Args:
        vertex (torch.Tensor): of shape (batch_size, 10002, 3) 
    """
    # TODO: Maybe we can pack the vertices together and compute the laplacian matrix at once.
    # TODO: This means for the faces, we need to add 10002 for batch1 and 10002*2 for batch2...
    
    # load the mean, std and template face info
    mean = torch.load(os.path.join("Laplacian", "mean.pt")).cuda()
    std = torch.load(os.path.join("Laplacian", "std.pt")).cuda()
    template_face = torch.load(os.path.join("Laplacian", "template_face.pt")).cuda()
    template_laplacian = torch.load(os.path.join("Laplacian", "template_laplacian_loss.pt")).cuda()
    
    # convert the vertex to the original scale
    vertex = vertex * std + mean
    
    # compute the laplacian matrix for the input vertex.
    for i in range(vertex.shape[0]):
        laplacian_single = laplacian(vertex[i], template_face, type='vertex')
        laplacian_single = laplacian_single.view(1, -1)
        if i == 0:
            laplacian_matrix = laplacian_single
        else:
            laplacian_matrix = torch.cat((laplacian_matrix, laplacian_single), dim=0)
    
    # get the batch size and broadcast the template_laplacian
    batch_size = laplacian_matrix.shape[0]
    template_laplacian = template_laplacian.repeat(batch_size, 1)
    
    # compute the difference of laplacian matrix
    diff_laplacian = torch.nn.functional.mse_loss(laplacian_matrix, template_laplacian, reduction='mean')
    
    return diff_laplacian