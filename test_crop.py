import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_ssim import ssim
from utils import AverageMeter
from datasets.better_loaders import DehazingCSVDataset, ResideOutdoorDataset
from models import *
from utils.metrics import psnr
import time
import numpy as np
from PIL import Image
from trainer_modify import LossUtils

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', type=str, help='model name')
parser.add_argument('--model_path', default='', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--data_path', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--dehaze_result_dir', default='./results/dehaze_result/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--out_dir', default='img', type=str, help='GPUs used for training')
parser.add_argument('--window_size', default=256, type=int, help='Sliding window size')
parser.add_argument('--stride', default=128, type=int, help='Sliding window stride (overlap)')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_as_uint8(tensor, save_path):
    """
    Helper to save: converts [0, 1] float tensor to [0, 255] uint8 image
    """
    img_np = tensor.cpu().detach().permute(1, 2, 0).numpy()
    
    img_np = np.clip(img_np, 0, 1)
    
    img_np = (img_np * 255.0).astype(np.uint8)
    
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path)
    
def make_gaussian_2d(sigma, window_size):
    """
    Creates a 2D Gaussian window to weigh the center of the patch higher than the edges.
    """
    x = torch.arange(window_size).float()
    gauss_1d = torch.exp(-(x - window_size // 2)**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.max() # Normalize
    gauss_2d = torch.ger(gauss_1d, gauss_1d) # Outer product
    return gauss_2d.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, H, W)

def sliding_window_inference(model, image, window_size=256, stride=128):
    """
    Performs sliding window inference with Gaussian blending.
    """
    b, c, h, w = image.shape
    
    # 1. Pad the image so it is divisible by window size/stride requirements
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    # Add a bit extra padding if needed to ensure we cover edges with stride
    if (h + pad_h - window_size) % stride != 0:
        pad_h += stride - ((h + pad_h - window_size) % stride)
    if (w + pad_w - window_size) % stride != 0:
        pad_w += stride - ((w + pad_w - window_size) % stride)

    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, H, W = image_padded.shape

    output_canvas = torch.zeros((b, 3, H, W)).to(image.device)
    weight_mask = torch.zeros((b, 1, H, W)).to(image.device)
    
    patch_weight = make_gaussian_2d(sigma=window_size//4, window_size=window_size).to(image.device)

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            crop = image_padded[:, :, y:y+window_size, x:x+window_size]
            
            with torch.no_grad():
                output_crop = model(crop)[0].clamp_(-1, 1)
            
            output_canvas[:, :, y:y+window_size, x:x+window_size] += output_crop * patch_weight
            weight_mask[:, :, y:y+window_size, x:x+window_size] += patch_weight

    output_final = output_canvas / (weight_mask + 1e-8)
    
    return output_final[:, :, :h, :w]


def test_dehaze(test_loader, network, dehaze_result_dir, out_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    device = DEVICE

    network.eval()

    os.makedirs(os.path.join(dehaze_result_dir, out_dir), exist_ok=True)
    f_result = open(os.path.join(dehaze_result_dir, 'dehaze_results.csv'), 'w')

    print(f"Starting inference with Sliding Window (Size: {args.window_size}, Stride: {args.stride})...")

    for idx, batch in enumerate(test_loader):
        _input = batch['source'].to(device)
        target = batch['target'].to(device)
        
        filename = batch['filename'][0]

        with torch.no_grad():
            output = sliding_window_inference(
                network, 
                _input, 
                window_size=args.window_size, 
                stride=args.stride
            )
            
            newoutput = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = psnr(newoutput, target)

            _, _, H, W = newoutput.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(newoutput, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                             size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test_dehaze: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

        save_as_uint8(output.squeeze(0), os.path.join(dehaze_result_dir, out_dir, filename))

    f_result.close()

    os.rename(os.path.join(dehaze_result_dir, 'dehaze_results.csv'),
              os.path.join(dehaze_result_dir, '%s | %.02f | %.04f.csv'%(out_dir, PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
    network = DIACMPN_dehaze_Indoor_merged(n_channels=3, freeze_clip=False, use_fadhc=True) 
    network.to(DEVICE)

    saved_model_dir = args.model_path
    dataset_path = args.data_path

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        dehaze_state_dict = torch.load(saved_model_dir)
        network.load_state_dict(dehaze_state_dict['dehaze_net'])
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = ResideOutdoorDataset(dataset_dir)
    # test_dataset = DehazingCSVDataset(dataset_path, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    dehaze_result_dir = os.path.join(args.dehaze_result_dir, args.dataset, args.model)
    test_dehaze(test_loader, network, dehaze_result_dir, args.out_dir)