# 2d img encoder -> ImgEncoder()
# 3d img encoder -> ObjEncoder()
# projection head -> ProjectionHead()
import torch
from torch import nn
from ULIP.models.resnet import resnet50
from torchvision.models import ResNet50_Weights

class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        resnet = resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        return self.encoder(x)
    
class ObjEncoder(nn.Module):
    pass

class ProjectionHead(nn.Module):
    def __init__(self, projection_dim=8192, embedding_dim=512):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        #self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        #x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x