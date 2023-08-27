import sys
sys.path.append('/Users/jwahn/Desktop/temp_folder/RebuilderAI_AHN/ULIP')
print(sys.path)
# dataset, loss
from dataset import T_2D_3D_Dataset
from loss import CLIPLoss
from modules import ImgEncoder, ProjectionHead
from ULIP.models.pointbert.point_encoder import PointTransformer
# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
# etc.
import os
import datetime
import wandb
from tqdm import tqdm

from utils import eval_phase, save_checkpoints, save_ckpt, train_phase

class CLIP_2D_3D():
    def __init__(self, temp):
        pass
#======================================= DEVICE ===========================================#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#===================================== Dataloader =======================================#
# Split dataset into train/val sets
dataset = T_2D_3D_Dataset(root_dir='/Users/jwahn/Downloads/Dataset')
train_ratio = 0.85  # 
num_samples = len(dataset)
train_size = int(train_ratio * num_samples)
valid_size = num_samples - train_size
train_set, valid_set = random_split(dataset, [train_size, valid_size])
# Create DataLoader instances for train/val sets
batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
#================================= Model, Loss, Optimizer =====================================#
# Init img encoding model
img_encoder = ImgEncoder() # load checkpoint
# Init obj encoding model
obj_encoder = PointTransformer.to(device)
obj_encoder.load_model_from_ckpt('ULIP/models/pointbert/pretrained_models-ckpt_zero-sho_classification-pointbert_ULIP-2.pt') # load checkpoint
# Init projection model
img_projector = ProjectionHead(projection_dim=512, embedding_dim=256).to(device)
obj_projector = ProjectionHead(projection_dim=8192, embedding_dim=256).to(device)

clip_loss = CLIPLoss().to(device)

parameters = list(img_encoder.parameters()) + list(obj_encoder.parameters()) + list(clip_loss.parameters())
optimizer = torch.optim.AdamW(
    parameters,  # Model parameters to optimize
    lr=0.001,  # Learning rate
    betas=(0.9, 0.999),  # Exponential decay rates for the first and second moments estimates
    eps=1e-08,  # Term added to the denominator to improve numerical stability
    weight_decay=0.01,  # L2 regularization strength
    amsgrad=False,  # Whether to use the AMSGrad variant of the optimizer
    maximize=False,  # Whether to maximize the objective (not used in AdamW)
    foreach=None,
    capturable=False,
    differentiable=False,
    fused=None
)
#========================================================================================#
result_dir = 'result' # exp1, exp2, ... 
if not os.path.exists(result_dir):
    os.makedirs(f'{result_dir}/checkpoints')
    print(f'New directory ({result_dir})  made')
#===============================TRAIN, VALIDATE==========================================#

wandb.init(project="2d_3d_clip_retrieval", name="training-run-name")
for epoch in range(500):
    train_phase(img_encoder, img_projector, obj_encoder, obj_projector, clip_loss)
    train_loss_sum = 0.0
    for idx, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        
        img = batch['2d'].to(device)
        obj = batch['3d'].to(device)
        img_encoded = img_encoder(img)
        obj_encoded = obj_encoder(obj)
        img_projected = img_projector(img_encoded)
        obj_projected = obj_projector(obj_encoded)
        
        loss = clip_loss(img_projected, obj_projected)
        loss.backward()
        
        optimizer.step()
        
        train_loss_sum += loss.item()
        
        wandb.log({
        'train_loss': loss.item()
        })
        
    train_loss = train_loss_sum / len(train_loader)
    wandb.log({
        'epoch': epoch,
        'train_loss_per_epoch': train_loss,
        'clip_temp': clip_loss.temp
    })    
    save_checkpoints(epoch, result_dir, img_encoder, img_projector, obj_encoder, obj_projector, clip_loss)
    
    # validate
    eval_phase(img_encoder, img_projector, obj_encoder, obj_projector, clip_loss)
    with torch.no_grad():
        valid_loss_sum = 0.0  
        for idx, batch in tqdm(enumerate(valid_loader)):
            img = batch['2d'].to(device)
            obj = batch['3d'].to(device)
            img_encoded = img_encoder(img)
            obj_encoded = obj_encoder(obj)
            img_projected = img_projector(img_encoded)
            obj_projected = obj_projector(obj_encoded)
        
            loss = clip_loss(img_projected, obj_projected)
            valid_loss_sum += loss.item()
            
        valid_loss = valid_loss_sum / len(valid_loader)
        wandb.log({
            'valid_loss': valid_loss 
        })
    print("--------------------------------------------------------------------------")
    