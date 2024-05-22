import argparse
from contextlib import redirect_stdout
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg
import igl

import torch


# set arg parser
pca_parser = argparse.ArgumentParser(description='PCA parser')
pca_parser.add_argument('--n_components', type=int, default=8)
pca_parser.add_argument('--train_path', type=str, default='data/human-body2/processed/training.pt')
pca_parser.add_argument('--test_path', type=str, default='data/human-body2/processed/test.pt')
pca_parser.add_argument('--template_path', type=str, default='template/template2.ply')
pca_parser.add_argument('--log_dir', type=str, default='out/pca/')

args = pca_parser.parse_args()
print(args)


def read_data(data_path, test_path, template_path=args.template_path):
    """
    :param template_path: the path for the template
    :param data_path: the path for the data
    :return: mean_val: the mean_val for the whole mesh dataset
             std_val: the std_val for the whole mesh dataset
             normalized_mesh: the processed mesh data that can be used for PCA
    """

    # For each mesh, it contains (5023, 3) vertices and (9976, 3) faces
    v, f = igl.read_triangle_mesh(template_path)
    data, _ = torch.load(data_path)
    test_data, _ = torch.load(test_path)
    num_vertices, _ = v.shape

    # convert the data into numpy array
    # mesh_set is of shape (1935, 10002, 3) and test_set is of shape (220, 10002, 3)
    mesh_set = data.x.view(-1, num_vertices, 3).numpy()
    test_set = test_data.x.view(-1, num_vertices, 3).numpy()
    
    # data.y is of shape [11610], which is [1935, 6]
    # test.y is of shape [1320], which is [220, 6]

    # We may not need the normalization
    mean_val = mesh_set.mean(axis=0)
    
    mesh_set = mesh_set - mean_val
    test_set = test_set - mean_val

    return mesh_set, test_set, mean_val


def pca_training(normalized_mesh, args):
    """
    :param normalized_mesh: normalized data that can be directly used for pca
    :param args: regular arguments
    :return: None
    """
    num_data, _, _ = normalized_mesh.shape
    normalized_mesh = normalized_mesh.reshape(num_data, -1)
    pca = PCA(n_components=args.n_components)
    pca.fit(normalized_mesh)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    with open(args.log_dir+'log.txt', 'w+') as log_file:
        with redirect_stdout(log_file):
            print("The explained variance ratio for {:d} principal components are:".format(args.n_components))
            print(pca.explained_variance_ratio_)
            print("The sum of explained variance is %.3f\n" % (sum(pca.explained_variance_ratio_) * 100))

    with open(args.log_dir+'pca.pkl', 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)

    return


def eval_loss(data, flag, args):
    """
    :param mean_val: mean value for the mesh data
    :param std_val: std value for the mesh data
    :param normalized_data: normalize mesh data
    :param original_data: ...
    :param flag: "training" or "testing"
    :param args: regular argument
    :return: None
    """

    with open(args.log_dir+'pca.pkl', 'rb') as pickle_file:
        pca = pickle.load(pickle_file)

    # reshape the train data and find the train projection
    n_data, n_vertices, _ = data.shape
    proj = pca.inverse_transform(pca.transform(data.reshape(n_data, -1)))

    # compute the loss
    proj = proj.reshape(n_data, n_vertices, 3)
    vertices_dist = np.sqrt(np.sum(np.square(proj - data), axis=2))
    # flatten the array and convert from meter to millimeter.
    vertices_dist = vertices_dist.flatten() * 1000

    # finding the stats
    mean = np.mean(vertices_dist)
    median = np.median(vertices_dist)
    std = np.std(vertices_dist)
    less_1mm = np.sum(vertices_dist < 10) / len(vertices_dist) * 100
    less_onehalf_mm = np.sum(vertices_dist < 15) / len(vertices_dist) * 100
    less_2mm = np.sum(vertices_dist < 20) / len(vertices_dist) * 100

    # write into log file
    with open(args.log_dir+'log.txt', 'a+') as log_file:
        log_file.write("The mean distance for {:s} data is {:.3f}\n".format(flag, mean))
        log_file.write("The median distance for {:s} data is {:.3f}\n".format(flag, median))
        log_file.write("The std for {:s} data is {:.3f}\n".format(flag, std))
        log_file.write("For a 10 mm accuracy on {:s}, PCA model captures {:.2f}% vertices\n".format(flag, less_1mm))
        log_file.write("For a 15 mm accuracy on {:s}, PCA model captures {:.2f}% vertices\n".
                       format(flag, less_onehalf_mm))
        log_file.write("For a 20 mm accuracy on {:s}, PCA model captures {:.2f}% vertices\n\n\n".
                       format(flag, less_2mm))
        
def feature_analysis(data_path):
    # load the pca model
    pca_reloaded = pickle.load(open("out/pca/pca.pkl",'rb'))

    # load the data
    data, _ = torch.load(data_path)
    mesh_set = data.x.view(-1, 10002, 3).numpy()
    labels = data.y.numpy().reshape(-1, 6)

    # append 1s to the labels to make it of shape (1935, 7).
    # after transpose, it becomes (7, 1935)
    labels = np.append(labels, np.ones((labels.shape[0], 1)), axis=1).T

    # compute the weights of pca from the mesh_set
    # weights is of shape (1935, 8)
    weights = pca_reloaded.transform(mesh_set.reshape(mesh_set.shape[0], -1))

    # compute the sudo inverse of labels, which is of shape (1935, 7)
    labels_inv = linalg.pinv(labels)

    # matrix M is of shape (8, 7)
    M = np.matmul(weights.T, labels_inv)

    # save the matrix M
    np.save("out/pca/M.npy", M)
    
    


if __name__ == "__main__":
    train_set, test_set, mean_val = read_data(data_path=args.train_path, test_path=args.test_path)
    
    # save the mean value
    mean_val = np.save("out/pca/mean.npy", mean_val) 
    
    pca_training(normalized_mesh=train_set, args=args)
    
    eval_loss(data=train_set, flag="Training", args=args)

    eval_loss(data=test_set, flag="Testing", args=args)
    
    feature_analysis(args.train_path)