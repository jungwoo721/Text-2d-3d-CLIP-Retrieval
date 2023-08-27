import os
import torch

def save_checkpoints(epoch, result_dir, img_encoder, img_projector, obj_encoder, obj_projector, clip_loss):
    fname = os.path.join(result_dir, 'checkpoints', "checkpoint_{:04d}.pth".format(epoch))
    print(f"> Saving model to {fname}...")
    model = {"img_encoder": img_projector.state_dict(), 
             "img_projector": img_encoder.state_dict(), 
             "obj_encoder": obj_encoder.state_dict(), 
             "obj_projector": obj_projector.state_dict(), 
             "obj_encoder": obj_encoder.state_dict(), 
             "clip_loss_temp": clip_loss.state_dict()}
    torch.save(model, fname)
    
def eval_phase(img_encoder, img_projector, obj_encoder, obj_projector, clip_loss):
    img_encoder.eval()
    img_projector.eval()
    obj_encoder.eval()
    obj_projector.eval()
    clip_loss.eval()
    
def train_phase(img_encoder, img_projector, obj_encoder, obj_projector, clip_loss):
    img_encoder.train()
    img_projector.train()
    obj_encoder.train()
    obj_projector.train()
    clip_loss.train()