
# RightSizing: Disentangling Generative Models of Human Body Shapes with Metric Constraints
RightSizing is a novel method for latent space disentanglement on 3D meshes that achieves interpretability, control, and strong disentanglement. It comprises two components: a learned feature function for predicting 3D mesh features, and a generative model that predicts not only the desired meshes but also their features and feature gradients. We employ feature gradients as part of the loss function to promote disentanglement.

RightSizing is described in the paper ["RightSizing: Disentangling Generative Models of Human Body Shapes with Metric Constraints"](LinkInProcessing...), by Yuhao Wu, [Chang Shu](https://nrc.canada.ca/en/corporate/contact-us/nrc-directory-science-professionals/chang-shu), [Dinesh K. Pai](https://sensorimotor.cs.ubc.ca/pai/).

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
In processing... 
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
In processing... 


## Referencing our work
Here are the Bibtex snippets for citing RightSizing in your work.
```
In processing... 
```


