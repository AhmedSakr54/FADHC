import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_ssim import ssim
from utils import AverageMeter
from datasets.better_loaders import ResideOutdoorDataset, DehazingCSVDataset
from models import *
from utils.metrics import psnr
from tqdm import tqdm
from PIL import Image
import time
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', type=str, help='model name')
parser.add_argument('--model_path', default='', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--dataset_path', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--dehaze_result_dir', default='./results/dehaze_result/', type=str, help='path to results saving')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='1', type=str, help='GPUs used for training')
parser.add_argument('--out_dir', default='img', type=str, help='GPUs used for training')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_as_uint8(tensor, save_path):
    """
    Helper to save: converts [0, 1] float tensor to [0, 255] uint8 image
    """
    # 1. Move to CPU and transform to (H, W, C)
    img_np = tensor.cpu().detach().permute(1, 2, 0).numpy()
    
    # 2. Clamp values to [0, 1] to avoid artifacts
    img_np = np.clip(img_np, 0, 1)
    
    # 3. Scale to [0, 255] and convert to uint8
    img_np = (img_np * 255.0).astype(np.uint8)
    
    # 4. Convert to PIL Image and save
    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path)

def test_dehaze(test_loader, network, dehaze_result_dir, out_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()
	device = DEVICE

	network.eval()

	os.makedirs(os.path.join(dehaze_result_dir, out_dir), exist_ok=True)
	f_result = open(os.path.join(dehaze_result_dir, 'dehaze_results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		start_time = time.time()
		_input = batch['source'].to(device)
		target = batch['target'].to(device)
		# print(input.shape)
		filename = batch['filename'][0]

		with torch.no_grad():
			output = network(_input)[0]

			new_output=output.clamp_(-1,1)

			new_output = new_output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = psnr(new_output, target)

			_, _, H, W = new_output.size()
			down_ratio = max(1, round(min(H, W) / 256))
			ssim_val = ssim(F.adaptive_avg_pool2d(new_output, (int(H / down_ratio), int(W / down_ratio))),
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
	network = FADHC_dehaze_Indoor_merged(n_channels=3)
	network.to(DEVICE)

	saved_model_dir = args.model_path
	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		dehaze_state_dict = torch.load(saved_model_dir)
		network.load_state_dict(dehaze_state_dict['dehaze_net'])
	else:
		print('==> No existing trained model!')
		exit(0)

	test_dataset = DehazingCSVDataset(args.dataset_path, 'val')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	dehaze_result_dir = os.path.join(args.dehaze_result_dir, args.dataset, args.model)
	test_dehaze(test_loader, network, dehaze_result_dir, args.out_dir)