
# RightSizing: Disentangling Generative Models of Human Body Shapes with Metric Constraints
RightSizing is a novel method for latent space disentanglement on 3D meshes that achieves interpretability, control, and strong disentanglement. It comprises two components: a learned feature function for predicting 3D mesh features, and a generative model that predicts not only the desired meshes but also their features and feature gradients. We employ feature gradients as part of the loss function to promote disentanglement.

RightSizing is described in the paper ["RightSizing: Disentangling Generative Models of Human Body Shapes with Metric Constraints"](https://openreview.net/pdf?id=H3xJRZn1Av), by Yuhao Wu, [Chang Shu](https://nrc.canada.ca/en/corporate/contact-us/nrc-directory-science-professionals/chang-shu), [Dinesh K. Pai](https://sensorimotor.cs.ubc.ca/pai/).

The video is available here: (in processing...)


## Why RightSizing:
* **Interpretability**: specific control latent variables $z_i \in z_c$ are associated with specific quantitative features of the human body, $h_i$, such as height or waist girth
* **Controllability**: we can vary the value of $z_i$ to continuously modify the value of $h_i$;
* **Strong Disentanglement**: the latent variable $z_i$ does not affect other features $h_{j \ne i}$; the model $p(z)$ is capable of generating samples that are plausible and diverse with fixed values of some features $h_i$.
* **Diversity**: the model $p(z)$ is capable of generating samples that are plausible and diverse with fixed values of some features $h_i$. 

## Outline
* `RightSizingModel`: It contains all the source code for training, evaluation, as well as abalition study.
* `visualizer`: Due to the fact we are unable to share with our CAESAR dataset, we upload a visualizer here. It contains all the experiments we did and pretrained model. People can use the visualizer to visualize the generated mesh without retraining the model.

## Apply RightSizing to other models
`RightSizing` can be applied to many other VAE Models, or even generative model. It is not limited to CoMA, which is used in the paper. The basic recipe is like following:

```python
# we need the x to denote as the input graph
x = data.x.to(self.device)

# Feeding a batch of images into the network to obtain the output image, mu, logVar and z
out, mu, logVar, z = self.model(x)
z.retain_grad()

# construct the canonical matrix
# z is of shape (batch_size, latent_dim)

# Now, we need to compute the attributes of the mesh
# 0: height, 1: arm_length, 2: crotch_height, 
# 3: chest_circumference, 4: hip_circumference, 5: waist_circumference,
predicted_result = self.predictor(out)
height = predicted_result[:, 0]

# Compute the gradient of height w.r.t. z
height_grad = torch.autograd.grad(height, z, grad_outputs=torch.ones(height.size()).cuda(), retain_graph=True, create_graph=True)[0]

# compute the the grad loss and divide by the batch size
height_loss = torch.sum(torch.norm(height_grad[:, 1:], dim=1, p=2)) / x.size(0)
```


## Environment Setup
For setting up the environment, you can just use the following command:

```python
conda env create -f environment.yml
```

You may receive the following error:

```
Pip subprocess error:
ERROR: Could not find a version that satisfies the requirement psbody-mesh==0.4 (from versions: none)
ERROR: No matching distribution found for psbody-mesh==0.4

failed

CondaEnvException: Pip failed
```

If you are unable to install the [psbody-mesh package](https://github.com/MPI-IS/mesh), you can try the following command to see if it works:
```python
sudo apt-get install libboost-dev
pip install git+https://github.com/MPI-IS/mesh.git
```

## Acknowledgement
 This work was supported in part by the NRC AI4D program, and by an NSERC Discovery grant to Dinesh K. Pai.

This paper is based on the thesis work by Yuhao Wu during his Master's program at the University of British Columbia, under the guidance and supervision of Professor Dinesh K. Pai and Professor Chang Shu. Yuhao wishes to express his sincere gratitude to both professors for their encouragement, invaluable support, and assistance throughout his research. Without their guidance, this work would not have been possible.

Our visualizer is based on the Élie Michel's [Python 3D Viewer](https://github.com/eliemichel/Python3dViewer). 

## Referencing our work
Here are the Bibtex snippets for citing RightSizing in your work.
```
@inproceedings{
wu2024rightsizing,
title={RightSizing: Disentangling Generative Models of Human Body Shapes with Metric Constraints},
author={Yuhao Wu and Chang Shu and Dinesh K. Pai},
booktitle={Graphics Interface 2024 Second Deadline},
year={2024},
url={https://openreview.net/forum?id=H3xJRZn1Av}
}
```


