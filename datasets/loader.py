import os
import random
import numpy as np
import pandas as pd
import cv2
import ast
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tfs

class TestingData(Dataset):
    def __init__(self, root_dir, crop_size=(256, 256)):
        self.root_dir = root_dir
        self.hazy_dir = os.path.join(root_dir, 'hazy')
        self.gt_dir = os.path.join(root_dir, 'GT')
        
        self.hazy_files = [f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transforms = tfs.Compose([
            tfs.CenterCrop(crop_size),
            # tfs.ToTensor()  # Converts to [C, H, W] and scales to [0, 1]
        ])

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy_name = self.hazy_files[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        
        img_id = hazy_name.split('_')[0] 
        gt_name = f"{img_id}.png"
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        source_img = read_img(hazy_path) * 2 - 1
        target_img = read_img(gt_path) * 2 - 1

        source_tensor = torch.from_numpy(source_img.copy()).float().permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_img.copy()).float().permute(2, 0, 1)
        
        hazy_tensor = self.transforms(source_tensor)
        gt_tensor = self.transforms(target_tensor)
        
        return {
            'source': hazy_tensor,
            'target': gt_tensor,
            'filename': hazy_name
        }
    
def augment_all(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]
    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    return imgs

def augment1(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H-Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W-Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
            
    return imgs

def augment(imgs, size, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]   # Assuming size is (h, w)

    # --- FIX START: Pad if image is smaller than crop size ---
    pad_h = max(0, Hc - H)
    pad_w = max(0, Wc - W)

    if pad_h > 0 or pad_w > 0:
        imgs = [cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101) for img in imgs]
        
        # Update H and W after padding
        H, W, _ = imgs[0].shape
    # --- FIX END ---

    # Original cropping logic
    Hs = random.randint(0, H - Hc)
    Ws = random.randint(0, W - Wc)

    # Apply crop
    imgs = [img[Hs:(Hs + Hc), Ws:(Ws + Wc), :] for img in imgs]

    # ... rest of your augmentation code (flipping, etc.) ...
    if random.random() < 0.5:
        # Horizontal Flip
        imgs = [img[:, ::-1, :] for img in imgs]

    if not only_h_flip:
        if random.random() < 0.5:
            # Vertical Flip
            imgs = [img[::-1, :, :] for img in imgs]
            
    return imgs


def align1(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    return imgs

def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # --- FIX: Pad if image is smaller than target size ---
    pad_h = max(0, Hc - H)
    pad_w = max(0, Wc - W)

    if pad_h > 0 or pad_w > 0:
        # Pad equally on both sides to keep the original content in the center
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        imgs = [cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101) for img in imgs]
        
        # Update dimensions after padding
        H, W, _ = imgs[0].shape
    # -----------------------------------------------------

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

    return imgs

class TripletLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, depth_dir, meta_data_path=None, size=512, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        self.depth_dir = depth_dir
        
        # Path setup
        if mode == 'test':
            self.root_dir = data_dir
            self.gt_dir = os.path.join(self.root_dir, 'GT')
            self.hazy_dir = os.path.join(self.root_dir, 'hazy')
        else:
            self.root_dir = os.path.join(data_dir, sub_dir)
            self.gt_dir = os.path.join(self.root_dir, 'GT')
            self.hazy_dir = os.path.join(self.root_dir, 'hazy')

        self.img_names = sorted(os.listdir(self.gt_dir))
        
        # Pre-calculate mapping to avoid O(N) lookup in __getitem__
        self.data_mapping = self._build_mapping(meta_data_path, mode)

    def _build_mapping(self, meta_data_path, mode):
        """
        Creates a dictionary mapping: GT_Filename -> List of Hazy Filenames
        """
        mapping = {}
        
        if mode != 'test':
            if meta_data_path is None or not os.path.exists(meta_data_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_data_path}")
                
            df = pd.read_csv(meta_data_path)
            # Create a dict for O(1) access
            # Assumes 'clear_image_path' format is "clear/image.png"
            for _, row in df.iterrows():
                gt_name = os.path.basename(row['clear_image_path'])
                if gt_name in self.img_names:
                    # Safely parse the list string
                    hazy_paths = ast.literal_eval(row['hazy_image_paths'])
                    # Extract just the filename from the paths in the csv
                    hazy_filenames = [os.path.basename(p) for p in hazy_paths]
                    mapping[gt_name] = hazy_filenames
        else:
            # Test mode logic: map based on prefix matching
            hazy_files = sorted(os.listdir(self.hazy_dir))
            temp_map = defaultdict(list)
            for h_file in hazy_files:
                # Assumes hazy file structure matches GT prefix (e.g., "01_hazy.png" -> "01.png")
                prefix = h_file.split("_")[0] + ".png"
                if prefix in self.img_names:
                    temp_map[prefix].append(h_file)
            mapping = dict(temp_map)
            
        return mapping

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        gt_name = self.img_names[idx]
        
        # 1. Load Ground Truth
        gt_path = os.path.join(self.gt_dir, gt_name)
        target_img = read_img(gt_path) * 2 - 1

        # 2. Load Hazy Image (Source)
        if gt_name in self.data_mapping:
            hazy_list = self.data_mapping[gt_name]
            if self.mode == 'train':
                hazy_name = random.choice(hazy_list)
            else:
                hazy_name = hazy_list[0]
        else:
            # Fallback if mapping fails
            raise IndexError(f"Image {gt_name} not found in metadata mapping.")

        source_path = os.path.join(self.hazy_dir, hazy_name)
        source_img = read_img(source_path) * 2 - 1

        # 3. Load Depth (if applicable)
        depth_img = None
        if self.mode != 'test' and self.depth_dir:
            depth_name = gt_name.replace('.png', '.tiff') # Handle extension change
            depth_path = os.path.join(self.depth_dir, depth_name)
            
            if os.path.exists(depth_path):
                # Load Tiff, keep as numpy array
                d_img = np.array(Image.open(depth_path))
                # Resize to match source (W, H)
                d_img = cv2.resize(d_img, (source_img.shape[1], source_img.shape[0]), interpolation=cv2.INTER_AREA)
                # Ensure channel dimension exists (H, W, 1)
                depth_img = np.expand_dims(d_img, axis=-1)
            else:
                 # Create dummy depth if missing to prevent crash, or raise error
                 depth_img = np.zeros((source_img.shape[0], source_img.shape[1], 1), dtype=np.float32)

        # 4. Processing
        # if self.mode == 'train' and depth_img is not None:
        #     [source_img, target_img, depth_img] = self._augment([source_img, target_img, depth_img])
        
        # elif self.mode == 'valid':
        #     if depth_img is not None:
        #         [source_img, target_img, depth_img] = self._align([source_img, target_img, depth_img])
        #     else:
        #         [source_img, target_img] = self._align([source_img, target_img])

        # 5. Return
        data = {
            'source': hwc_to_chw(source_img),
            'target': hwc_to_chw(target_img),
            'filename': gt_name
        }

        if depth_img is not None:
            data['depth'] = hwc_to_chw(depth_img)

        return data

    def _augment(self, imgs):
        
        H, W, _ = imgs[0].shape
        Hc, Wc = [self.size, self.size]

        # Edge Decay Crop Logic
        if random.random() < Hc / H * self.edge_decay:
            Hs = 0 if random.randint(0, 1) == 0 else H - Hc
        else:
            Hs = random.randint(0, H - Hc)

        if random.random() < Wc / W * self.edge_decay:
            Ws = 0 if random.randint(0, 1) == 0 else W - Wc
        else:
            Ws = random.randint(0, W - Wc)

        # Apply Crop
        for i in range(len(imgs)):
            imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

        # Horizontal Flip
        if random.randint(0, 1) == 1:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1).copy() # .copy() avoids negative stride errors in torch

        # Rotation (Outdoor logic skipped based on only_h_flip flag)
        if not self.only_h_flip:
            rot_deg = random.randint(0, 3)
            for i in range(len(imgs)):
                imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1)).copy()

        return imgs

    def _align(self, imgs):
        """Center Crop"""
        H, W, _ = imgs[0].shape
        Hc, Wc = [self.size, self.size]

        Hs = (H - Hc) // 2
        Ws = (W - Wc) // 2
        
        for i in range(len(imgs)):
            imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

        return imgs

class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, meta_data_path=None, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        
        if mode != 'test':
            self.root_dir = os.path.join(data_dir, sub_dir)
        else:
            self.root_dir = data_dir
            
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
        self.img_num = len(self.img_names)
        
        if mode != 'test':
            self.metadata = pd.read_csv(meta_data_path)
        else:
            image_names = sorted(os.listdir(os.path.join(self.root_dir, 'hazy')))
            mapping = defaultdict(list)
            for aug in image_names:
                prefix = aug.split("_")[0] + ".png"  
                if prefix in self.img_names:
                    mapping[prefix].append(aug)
            self.mapping = dict(mapping)

    def __len__(self):
        return self.img_num

    def center_crop(self, img):
        """Helper to crop the center of the image to self.size"""
        h, w, _ = img.shape
        
        y_start = (h - self.size) // 2
        x_start = (w - self.size) // 2
        
        y_start = max(0, y_start)
        x_start = max(0, x_start)

        return img[y_start:y_start+self.size, x_start:x_start+self.size, :]

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        
        if self.mode != 'test':
            img_name = self.img_names[idx]
            hazy_img_name = self.metadata[self.metadata['clear_image_path'] == f"clear/{img_name}"]
            hazy_list = eval(hazy_img_name['hazy_image_paths'].values[0])
            random_hazy_img = random.choice(hazy_list)
            source_img = read_img(os.path.join(self.root_dir, 'hazy', random_hazy_img.split('/')[-1])) * 2 - 1
            target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
        else:
            img_name = self.img_names[idx]
            random_hazy_img = self.mapping[img_name][0]
            source_img = read_img(os.path.join(self.root_dir, 'hazy', random_hazy_img)) * 2 - 1
            target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
            
            source_img = self.center_crop(source_img)
            target_img = self.center_crop(target_img)

        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)
        
        source_tensor = torch.from_numpy(source_img.copy()).float().permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_img.copy()).float().permute(2, 0, 1)
        
        return {'source': source_tensor, 'target': target_tensor, 'filename': img_name}
    

class HazyDataLoader(Dataset):
    def __init__(self, mode, meta_data_path, size=256, edge_decay=0, only_h_flip=False):
            assert mode in ['train', 'valid', 'test']
            self.medadata = pd.read_csv(meta_data_path)
            self.size = size
            self.edge_decay = edge_decay
            self.only_h_flip = only_h_flip
            self.mode = mode
        
    def __len__(self):
            return self.medadata.shape[0]
            
    
    def center_crop(self, img):
        """Helper to crop the center of the image to self.size"""
        h, w, _ = img.shape
        
        y_start = (h - self.size) // 2
        x_start = (w - self.size) // 2
        
        y_start = max(0, y_start)
        x_start = max(0, x_start)

        return img[y_start:y_start+self.size, x_start:x_start+self.size, :]

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        data_item = self.medadata.iloc[idx].to_dict()
        clear_img = data_item['clear_image_path']
        source_img = data_item['hazy_image_paths']
        source_img = read_img(source_img) * 2 - 1
        target_img = read_img(clear_img) * 2 - 1
        source_img = self.center_crop(source_img)
        target_img = self.center_crop(target_img)

        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)
        
        source_tensor = torch.from_numpy(source_img.copy()).float().permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_img.copy()).float().permute(2, 0, 1)
        print(source_tensor.shape)
        
        return {'source': source_tensor, 'target': target_tensor, 'filename': data_item['image_id']}

class AllDataLoader(Dataset):
    def __init__(self, mode, meta_data_path, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']
        self.metadata = pd.read_csv(meta_data_path)
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        self.mode= mode

    def center_crop(self, img):
        h, w, _ = img.shape
        
        if h < self.size or w < self.size:
            return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        
        y_start = (h - self.size) // 2
        x_start = (w - self.size) // 2
        
        return img[y_start:y_start+self.size, x_start:x_start+self.size, :]
    
    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
         cv2.setNumThreads(0)
         cv2.ocl.setUseOpenCL(False)
         data_item = self.metadata.iloc[idx].to_dict()
         hazy_list = eval(data_item['hazy_image_paths'])
         weights = [i + 1 for i in range(len(hazy_list))]
         sampled_item = random.choices(hazy_list, weights=weights, k=1)
         random_hazy_img = sampled_item[0]
         clear_img = data_item['clear_image_path']
         source_img = read_img(random_hazy_img) * 2 - 1
         target_img = read_img(clear_img) * 2 - 1
         if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)
         if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)
         source_tensor = torch.from_numpy(source_img.copy()).float().permute(2, 0, 1)
         target_tensor = torch.from_numpy(target_img.copy()).float().permute(2, 0, 1)
        
         return {'source': source_tensor, 'target': target_tensor}

class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'img': hwc_to_chw(img), 'filename': img_name}