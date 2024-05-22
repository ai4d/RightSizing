# Visualizer

Due to the fact we are unable to provide our mesh data, we provide a visualizer of our trained model here, so that everyone can see how the result is without retraining the model.

## How to use it?
The architecture of neural network is fixed. We just introduce different ways of training. So, we only need to change the parameter of the model, then we can generate different result. 

To switch between different parameter sets of the model, we only need to change the path of pretrained model. The path has already been written in the `main.py`. 

Switch the pretrained model you want, then run the `main.py`, the visualizer should come out. 

## What is the meaning of each different trainer?

We list the loss function of each different trainers here:

Actually, all the Trainers have similar structure. The only difference is the loss function. 

* `Trainer1.py`: The Loss is height, chest, waist.
* `Trainer2.py`: The Loss is height, waist
* `Trainer3.py`: The Loss is height, waist. We also introduce the Laplacian Loss to make the generated mesh smoother. The implementation detail is in `Laplacian/Laplacian.py`.
* `Trainer4.py`: Trainer4 includes the ratio. Since I have changed the predictor network. It is not working now. (In fact, even before the change, loss with ratios is not working well.)
* `Trainer5.py`: The loss is height, chest, wasit and Laplacian.
* `Trainer6.py`: Height + waist + Laplacian loss. We introduce AWL to balance the loss.
* `Trainer7.py`: Height + waist + chest + Laplacian loss. We introduce AWL to balance the loss.
* `Trainer8.py`: Height + waist + chest + arm + Laplacian loss. We introduce AWL to balance the loss.
* `Trainer9.py`: Height + waist + chest + hip + Laplacian loss. We introduce AWL to balance the loss.
* `Trainer10.py`: Height + waist + chest + hip + arm + crotch + Laplacian loss. We introduce AWL to balance the loss. The loss list is like the following: 
```python
loss_list = [0.1 * (recon_loss + kl_divergence), 
             1000 * height_loss, 1000 * waist_loss, 1000 * chest_loss, 1000 * hip_loss,
             1000 * arm_loss, 1000 * crotch_loss,
             laplacian_loss]
```
* `Trainer11.py`: Different from the above one, we delete the arm loss. We introduce AWL to balance the loss. The loss list is like the following: 
```python
loss_list = [0.1 * (recon_loss + kl_divergence), 
               1000 * height_loss, 1000 * waist_loss, 1000 * chest_loss, 1000 * hip_loss, 1000 * crotch_loss,
               laplacian_loss]
```
* `Trainer12.py`: Different from the above one, we use AWL to automatically balance the the coefficient of recon_loss and KL_div. The loss list is like the following: 
```python
loss_list = [0.1 * recon_loss, 0.1 * kl_divergence, 
               1000 * height_loss, 1000 * waist_loss, 1000 * chest_loss, 1000 * hip_loss,
               1000 * arm_loss, 1000 * crotch_loss,
               laplacian_loss]
```
* The following is basically ablation studies: 

  1.  `Trainer13.py`: Different from the above one, we delete the Laplacian Loss. The loss list is like the following: 
  ```python
  loss_list = [0.1 * recon_loss, 0.1 * kl_divergence, 
                1000 * height_loss, 1000 * waist_loss, 1000 * chest_loss, 1000 * hip_loss,
                1000 * arm_loss, 1000 * crotch_loss]
  ```
  1. `Trainer14.py`: There is no AWL now. 
    ```python
    loss = alpha * recon_loss + beta * (kl_divergence + 100 * height_loss + 100 * waist_loss 
                                        + 100 * chest_loss + 0.1 * laplacian_loss
                                        + 100 * hip_loss + 100 * arm_loss + 100 * crotch_loss)   
    ```
  1. `Trainer15.py`: (reconstruction) + (KL) + (laplacian). There is no disentanglement. 
  2.  `Trainer16.py`: Basically, it includes everything: 
    ```python
                loss_list = [0.1 * recon_loss, 0.1 * kl_divergence, 
                        1000 * height_loss, 1000 * waist_loss, 1000 * chest_loss, 1000 * hip_loss,
                        1000 * arm_loss, 1000 * crotch_loss,
                        laplacian_loss]
    ```

* `Trainer17.py`: In this trainer, we reduce the number of features we are hoping to disentangle. This is the one we show in our video. The loss list is like the following:
```python
loss_list = [0.1 * recon_loss, 0.1 * kl_divergence, 
            1000 * height_loss, 1000 * waist_loss, 1000 * shoulder_loss]
```
* `Trainer18.py`: In this trainer, we introduce the most features we are hoping to disentangle. The loss list is like the following:
```python
loss_list = [0.1 * recon_loss, 0.1 * kl_divergence, 
               1000 * height_loss, 1000 * waist_loss, 1000 * chest_loss, 1000 * hip_loss,
               1000 * arm_loss, 1000 * crotch_loss, 1000 * head_loss,
               laplacian_loss]
```