"""_description_
This a helper function use to record the loss corresponding to trainer1.py
"""

from typing import Dict
import os


def log(loss_dict: dict, output_dir: str):
    epochs = loss_dict['epochs']
    epoch = loss_dict['epoch']
    loss = loss_dict['Loss']
    time = loss_dict['time']
    
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write('Epoch: {}/{}, Loss: {:.5f}, Time: {:.1f}\n'.format(epoch, epochs, loss, time))
        print('Epoch: {}/{}, Loss: {:.5f}, Time: {:.1f}'.format(epoch, epochs, loss, time))
        
    # print all the losses in the loss_dict and write them to the log.txt
    empty_str = ""
    for key in loss_dict.keys():
        if key == 'epoch' or key == 'epochs' or key == 'Loss' or key == 'time':
            continue
        else:
            loss = loss_dict[key]
            empty_str += "{}: {:.5f}. ".format(key, loss)
    
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write(empty_str + "\n")
        print(empty_str)


# log the test loss during training process
def log_test(loss: float, output_dir: str):
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:   
        f.write('The test reconstruction loss is {:.4f}\n\n'.format(loss))
    print('The test reconstruction loss is {:.4f}\n'.format(loss))    

def eval_log(loss_dict: Dict, output_dir: str):
    loss = loss_dict['loss']
    recon_loss = loss_dict['recon_loss']
    kl_loss = loss_dict['kl_loss']

    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:  
        f.write("The following is the evaluation result on the test set: \n") 
        f.write('Loss: {:.4f}, Recon Loss: {:.4f}, KL Loss: {:.4f}\n'.format(
            loss, recon_loss, kl_loss))
    print('Loss: {:.4f}, Recon Loss: {:.4f}, KL Loss: {:.4f}'.format(
        loss, recon_loss, kl_loss))