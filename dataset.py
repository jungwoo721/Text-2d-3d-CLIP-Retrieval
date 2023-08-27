import os
import trimesh
from PIL import Image
import pandas as pd
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm




class T_2D_3D_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata', 'sample_metadata.csv'))
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomCrop((224, 224)),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Model ID
        model_id = self.metadata.iloc[idx, 0]
        model_folder = os.path.join(self.root_dir, 'models', model_id)

        # Class Label
        class_label = self.metadata.iloc[idx, 1]
        
        # 3D Pointcloud
        obj_path = os.path.join(model_folder, '3d_model', f'{model_id}.obj')
        mesh = trimesh.load(obj_path)
        num_points = 8192  # Number of points in the point cloud
        sampled_points = trimesh.sample.sample_surface(mesh, num_points)  # Returns a numpy array
        sampled_points, _ = sampled_points
        points = torch.tensor(sampled_points, dtype=torch.float32) # Convert the numpy array to a PyTorch tensor

        # 2D Image
        render_image_list = []
        random_i = random.randint(0, 4)
        image_path = os.path.join(model_folder, 'render_images', f'0000{random_i}.png')
        render_image = Image.open(image_path)
        render_image = self.transform_img(render_image)

        # Return Dict
        sample = {
            'id' : model_id,
            'class': class_label,
            '3d': points,
            '2d': render_image,
        }

        return sample

# Example usage
#root_directory = '/Users/jwahn/Downloads/Dataset'
#dataset = T_2D_3D_Dataset(root_directory)
#sample_data = dataset[0]
#print(sample_data)
#print(sample_data['3d'].shape, sample_data['2d_list'][0].shape)

if __name__ == "__main__":
    dataset = T_2D_3D_Dataset(root_dir='/Users/jwahn/Downloads/Dataset')
    train_loader = DataLoader(dataset, batch_size=3, shuffle=True)
    for epoch in range(10):
        for idx, batch in tqdm(enumerate(train_loader)):
            print(f'idx : {idx}')
            print(f'batch: \n {batch}')
            shape_2d = batch['2d'].shape
            shape_3d = batch['3d'].shape
            print(f'dim_img_batch : {shape_2d}')
            print(f'dim_pcl_batch : {shape_3d}')
            break
        break
