import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tfs
import ast
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF

class ResideOutdoorDataset(Dataset):
    def __init__(self, root_dir, crop_size=(256, 256)):
        self.root_dir = root_dir
        self.hazy_dir = os.path.join(root_dir, 'hazy')
        self.gt_dir = os.path.join(root_dir, 'GT')
        
        self.hazy_files = [f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transforms = tfs.Compose([
            # tfs.CenterCrop(crop_size),
            tfs.ToTensor()  
        ])

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy_name = self.hazy_files[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        
        img_id = hazy_name.split('_')[0] 
        gt_name = f"{img_id}.png"
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        hazy_img = Image.open(hazy_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        hazy_tensor = self.transforms(hazy_img)
        gt_tensor = self.transforms(gt_img)
        
        return {
            'source': hazy_tensor,
            'target': gt_tensor,
            'filename': hazy_name
        }


class DehazingCSVDataset(Dataset):
    def __init__(self, csv_file, mode='train', crop_size=(256, 256)):
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clear_path = row['clear_image_path']
        
        hazy_paths_list = ast.literal_eval(row['hazy_image_paths'])
        
        if len(hazy_paths_list) > 1:
            weights = [i + 1 for i in range(len(hazy_paths_list))]
            selected_hazy_name = random.choices(hazy_paths_list, weights=weights, k=1)[0]
        else:
            selected_hazy_name = hazy_paths_list[0]
            
        hazy_path = selected_hazy_name

        clear_img = Image.open(clear_path).convert('RGB')
        hazy_img = Image.open(hazy_path).convert('RGB')
        file_name = hazy_path.split('/')[-1]

        clear_img, hazy_img = self._transform(clear_img, hazy_img)

        return {
            'source': hazy_img,
            'target': clear_img,
            'filename': file_name
        }

    def _transform(self, clear, hazy):
        if self.mode == 'val':
            clear = TF.center_crop(clear, self.crop_size)
            hazy = TF.center_crop(hazy, self.crop_size)
        
        elif self.mode == 'train':
            
            w_img, h_img = clear.size
            th, tw = self.crop_size
            
            if w_img > tw and h_img > th:
                i = random.randint(0, h_img - th)
                j = random.randint(0, w_img - tw)
                
                clear = TF.crop(clear, i, j, th, tw)
                hazy = TF.crop(hazy, i, j, th, tw)
            else:
                clear = TF.center_crop(clear, self.crop_size)
                hazy = TF.center_crop(hazy, self.crop_size)

            if random.random() > 0.5:
                clear = TF.hflip(clear)
                hazy = TF.hflip(hazy)

            if random.random() > 0.5:
                clear = TF.vflip(clear)
                hazy = TF.vflip(hazy)

        clear = TF.to_tensor(clear)
        hazy = TF.to_tensor(hazy)
        
        return clear, hazy