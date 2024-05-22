import torch
import torch.nn.functional as F


def test(model, loader, device):
    """ _description_: This is the function used during the training process.
    We are hoping to know the loss of the model during the training process.
    
    Args:
        model: the model we are training
        loader: just the test data loader
        device: the device we are using(GPU)

    Returns:
        the loss of the model 
    """
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            # out, mu, logvar, z = model(x)
            pred, _, _, _ = model(x)
            # mean of each batch * batch size = the total mean loss of the batch
            total_loss += F.l1_loss(pred, x, reduction='mean') * x.size(0)
                
    return total_loss / len(loader.dataset)


def eval_error(model, test_loader, device, meshdata, out_dir):
    """ _description_: This is the function used after finishing the training process.

    Args:
        model: the model that we have trained
        test_loader: just the test data loader
        device: the device we are using(GPU)
        meshdata: from meshdata.py
        out_dir: the output directory
    """
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            # out, mu, logvar, z = model(x)
            pred, _, _, _ = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_x *= 1000
            
            # [num_graphs, num_nodes]
            tmp_error = torch.sqrt(torch.sum((reshaped_pred - reshaped_x)**2, dim=2))  
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Mean: {:.3f} + std: {:.3f} | Median {:.3f}'.format(mean_error, std_error, median_error)

    out_error_fp = out_dir + '/log.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('The following is the eval result:\n')
        log_file.write('{:s}\n'.format(message))
    
    print(message)
