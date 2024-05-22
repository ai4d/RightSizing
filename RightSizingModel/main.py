import os
import os.path as osp
import time

import yaml
from ast import literal_eval
import numpy as np
import torch
import wandb

from dataset import MeshData, DataLoader
from preprocess import mesh_sampling_method
from models import VAE_coma, GraphPredictor, AutomaticWeightedLoss
from trainer import Trainer1, Trainer2, Trainer3, Trainer4, Trainer5, Trainer6, Trainer7, Trainer8, Trainer9, Trainer10, Trainer11, Trainer12, Trainer13, Trainer14, Trainer15, Trainer16
from eval import eval_error

# add deterministic behavior
seed = 3407 # Torch.manual_seed(3407) is all you need
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# read into the config file
config_path = 'config/config_temp1.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# initialize wandb
wandb.init(project="CAESAR_with_label", config=config)


# set the device, we can just assume we are using single GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create some file paths
dataset = config["dataset"]
template = config["template"]
work_dir = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(work_dir, 'data', dataset)
template_fp = osp.join(work_dir, 'template', template)


# create the save file path
model_name = config["model_name"]
current_time = str(time.strftime('%Y%m%d-%H%M%S', time.localtime()))
out_dir = osp.join(work_dir, 'out', model_name, config["trainer"], current_time)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
# load the ouput directory into the config file
config['out_dir'] = out_dir

if config["model_name"] in ["vae_coma"]:
    ds_factor = config["ds_factor"]
    edge_index_list, down_transform_list, up_transform_list = mesh_sampling_method(data_fp=data_dir,
                                                                                   template_fp=template_fp,
                                                                                   ds_factors=ds_factor,
                                                                                   device=device)

# create the model
if config["model_name"] in  ["vae_coma"]:
    in_channels = config["model"]["in_channels"]
    out_channels = config["model"]["out_channels"]
    latent_channels = config["model"]["latent_channels"]
    K = config["model"]["K"]

if config["model_name"] == "vae_coma":
    model = VAE_coma(in_channels = in_channels,
                    out_channels = out_channels,
                    latent_channels = latent_channels,
                    edge_index = edge_index_list,
                    down_transform = down_transform_list,
                    up_transform = up_transform_list,
                    K=K).to(device)
    wandb.watch(model, log="all")
    
print("Start Loading original CAESAR data...")
batch_size = config["batch_size"]
CAESAR_meshdata = MeshData(root=data_dir, template_fp=template_fp)
CAESAR_train_loader = DataLoader(CAESAR_meshdata.train_dataset, batch_size=batch_size)
CAESAR_test_loader = DataLoader(CAESAR_meshdata.test_dataset, batch_size=batch_size)
print("CAESAR data loading finishes!\n")

# save the mean and std into Laplacian folder
# save the Template Face info into the Laplacian folder as well
if not os.path.exists(os.path.join("Laplacian", "mean.pt")) or not os.path.exists(os.path.join("Laplacian", "std.pt")) or not os.path.exists(os.path.join("Laplacian", "template_face.pt")):
    print("Start saving the mean, std and template face info into Laplacian folder...")
    torch.save(CAESAR_meshdata.mean, os.path.join("Laplacian", "mean.pt"))
    torch.save(CAESAR_meshdata.std, os.path.join("Laplacian", "std.pt"))
    template_face = torch.from_numpy(CAESAR_meshdata.template_face)
    torch.save(template_face, os.path.join("Laplacian", "template_face.pt"))
    print("Saving finishes!\n")

# create the CAESAR optimizer.
lr = config["lr"]
weight_decay = config["weight_decay"]
betas = literal_eval(config["betas"])
CAESAR_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

print("Start Loading the Predictor Network...")
predictor = GraphPredictor(in_channels = in_channels, out_channels = out_channels,
                        edge_index = edge_index_list, down_transform = down_transform_list, K=K).to(device)
predictor.load_state_dict(torch.load("predictor_network/predictor.pth"))
print("Predictor Network loading finishes!\n")

print("Start CAESAR data training...")
reduce = config["reduce"]
# this contains height, waist circumference, chest circumference
if config["trainer"] == "trainer1":
    temp_trainer1 = Trainer1(model, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer1.train(reduce=reduce)

# this contains height, waist circumference
if config["trainer"] == "trainer2":
    temp_trainer2 = Trainer2(model, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer2.train(reduce=reduce)    

# this contains height, waist circumference, Laplacian Loss
if config["trainer"] == "trainer3":
    temp_trainer3 = Trainer3(model, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer3.train(reduce=reduce)

# this contains height, waist hip ratio
if config["trainer"] == "trainer4":
    temp_trainer4 = Trainer4(model, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer4.train(reduce=reduce)
    
# this contains height, waist circumference, chest circumference, Laplacian Loss
if config["trainer"] == "trainer5":
    temp_trainer5 = Trainer5(model, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer5.train(reduce=reduce)
    
# this contains height, waist circumference. We use the AutomaticWeightedLoss
if config["trainer"] == "trainer6":
    AWL = AutomaticWeightedLoss(num=5) # (reconstruction + KL) + (height + waist) + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer6 = Trainer6(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer6.train(reduce="sum")
    
# this contains height, waist circumference, chest circumference. We use the AutomaticWeightedLoss
if config["trainer"] == "trainer7":
    AWL = AutomaticWeightedLoss(num=5) # (reconstruction + KL) + height + waist + chest + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer7 = Trainer7(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer7.train(reduce="sum")

# this contains height, waist circumference, chest circumference. We use the AutomaticWeightedLoss
if config["trainer"] == "trainer8":
    AWL = AutomaticWeightedLoss(num=6) # (reconstruction + KL) + height + waist + chest + arm length + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer8 = Trainer8(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer8.train(reduce="sum")

if config["trainer"] == "trainer9":
    AWL = AutomaticWeightedLoss(num=6) # (reconstruction + KL) + height + waist + chest + hip + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer9 = Trainer9(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer9.train(reduce="sum")

if config["trainer"] == "trainer10":
    AWL = AutomaticWeightedLoss(num=8) # (reconstruction + KL) + height + waist + chest + hip + arm + crotch height + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer10 = Trainer10(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer10.train(reduce="sum")
    
if config["trainer"] == "trainer11":
    AWL = AutomaticWeightedLoss(num=7) # (reconstruction + KL) + height + waist + chest + hip + crotch height + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer11 = Trainer11(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer11.train(reduce="sum")
    
if config["trainer"] == "trainer12":
    AWL = AutomaticWeightedLoss(num=9) # (reconstruction) + (KL) + height + waist + chest + hip + arm + crotch height + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer12 = Trainer12(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer12.train(reduce="sum")
    
if config["trainer"] == "trainer13":
    AWL = AutomaticWeightedLoss(num=8) # (reconstruction) + (KL) + height + waist + chest + hip + arm + crotch height
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer13 = Trainer13(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer13.train(reduce="sum")
    
if config["trainer"] == "trainer14":
     # (reconstruction) + (KL) + height + waist + chest + hip + arm + crotch height + Laplacian without AWL
    temp_trainer14 = Trainer14(model, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer14.train(reduce="mean")
    
if config["trainer"] == "trainer15":
    AWL = AutomaticWeightedLoss(num=3) # (reconstruction) + (KL) + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer15 = Trainer15(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer15.train(reduce="sum")
 
if config["trainer"] == "trainer16":
    AWL = AutomaticWeightedLoss(num=9) # (reconstruction) + (KL) + height + waist + chest + hip + arm + crotch height + (laplacian)
    # declare a new optimizer: model.parameters() + AWL.parameters()
    CAESAR_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': AWL.parameters()}], lr=lr, weight_decay=weight_decay, betas=betas)
    temp_trainer16 = Trainer16(model, AWL, predictor, CAESAR_optimizer, CAESAR_train_loader, CAESAR_test_loader, out_dir, config)
    temp_trainer16.train(reduce="sum")   
    
print("CAESAR data training finishes!")

# evaluate the model
eval_error(model=model,
           test_loader=CAESAR_test_loader,
           device=device,
           meshdata=CAESAR_meshdata,
           out_dir=out_dir)

# write the config file to the output directory
with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f, sort_keys=False)
