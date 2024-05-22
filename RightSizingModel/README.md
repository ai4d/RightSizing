## Disentanglement Loss
This is the most important part of our method. 

## label_process

Our CAESAR data comes from National Research Council. Due to the it is not an open source dataset, we are unable to release it here. 

### label_process/label-process.ipynb

This is a Jupyter Notebook used to process our `metric.xls`. It does two things.

* We select from the `metric.xls` to pick the attributes we need. In the `select_columns(data_frame, type='standardization')` function, there are two types. `standardization` means we do standardization to the features we are hoping to disentangle. `original` means we keep the original data. What's more, excpet that, we need to remove entries with NA value and correponding meshes.
* We make a scatter plot between these features. 


## predictor.ipynb

This is a Jupyter Notebook used to train the predictor network for features, like height, chest circumference, arm length... In the process, we designed two networks as a comparison. One is MLP structure, the other is GNN structure.

* In this version, there is no MLP structure. As in previous version, MLP loss is much larger than the GNN model. Intuitively, it makes sense. As mesh is a structure containing not only point cloud information but also topology information. It is unable for the MLP structure to capture the topology information.
* For the GNN model, we do training for all the features all together by one GNN model. We also try to design three GNN models to train them separately. From the result, there is no much difference.
  * NOTE: for the final loss of the features, we mutiple the corresponding `std` to match their original scale.

## AutomaticWeightedLoss

This is a folder containing the code for the automatic weighted loss. The main idea is to use the gradient of the loss to automatically adjust the weight of each term in loss function. The idea comes from the following two papers:

* Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
* Auxiliary Tasks in Multi-task Learning

The implementation is in `models/AutomaticWeightedLoss.py`. The usage is as follows:

```python
loss1=1
loss2=2
awl = AutomaticWeightedLoss(2)
loss_sum = awl([loss1, loss2])

# NOTE: You also need to include the parameters of AWL in the optimizer.
```