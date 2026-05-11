import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth.networks import * 
from models import DepthNet, FADHC_dehaze_Indoor_merged, DIACMPN_dehaze_Indoor
from pytorch_ssim import ssim
from utils import AverageMeter
from datasets.better_loaders import DehazingCSVDataset
    
from loss.CR_loss import ContrastLoss as crloss
from loss.losses import LossUtils, ColorConsistencyLoss

class Trainer:
    def __init__(self, args, current_index=0):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if current_index:
            self.current_index=current_index
        else:
            self.current_index = ""
        with open(args.config_path, 'r') as f:
            self.setting = json.load(f)

        self.save_dir = os.path.join(args.save_dir, args.exp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.init_wandb()
        self.init_models()
        self.init_loss()
        self.init_optimizers()
        self.init_dataloaders()
        
        self.scaler = GradScaler(enabled=not args.no_autocast)
        self.best_psnr = 0.0
        self.start_epoch = 0

        self.load_checkpoint()

        if args.load_checkpoint_depth and args.load_checkpoint_dehaze:
            self.load_checkpoint_pretrained(args.load_checkpoint_dehaze, args.load_checkpoint_depth)

    def init_wandb(self):
        wandb.init(
            project="FADHC", 
            name=f"{self.args.exp}_{self.args.model}",
            config={**vars(self.args), **self.setting}
        )

    def init_models(self):
        with torch.no_grad():
            model_path = os.path.join("./depth/models", 'RA-Depth')
            assert os.path.isdir(model_path), \
                "Cannot find a folder at {}".format(model_path)
            print("-> Loading weights from {}".format(model_path))

            encoder_path = os.path.join(model_path, "encoder.pth")
            decoder_path = os.path.join(model_path, "depth.pth")
            encoder_dict = torch.load(encoder_path)
            self.encoder = hrnet18(False)
            self.depth_decoder = DepthDecoder_MSF(self.encoder.num_ch_enc, [0], num_output_channels=1)
            model_dict = self.encoder.state_dict()
            self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            self.depth_decoder.load_state_dict(torch.load(decoder_path))
            self.encoder.cuda()
            self.encoder.eval()
            self.depth_decoder.cuda()
            self.depth_decoder.eval()
        if self.args.use_baseline:
            print("Using BaseLine...")
            self.net_dehaze = DIACMPN_dehaze_Indoor(n_channels=3).to(self.device)
        else:
            self.net_dehaze = FADHC_dehaze_Indoor_merged(n_channels=3, freeze_clip=self.args.freeze_clip, use_fadhc=self.args.use_fadhc).to(self.device)
        self.net_depth = DepthNet.DN().to(self.device)

    def init_loss(self):
        self.criterion_l1 = nn.L1Loss()
        self.criterion_cr = crloss()
        self.cc_loss = ColorConsistencyLoss()

    def init_optimizers(self):
        params_dehaze = self.net_dehaze.parameters()
        params_depth = self.net_depth.parameters()

        lr_dehaze = self.setting['lr_dehaze']
        lr_depth = self.setting['lr_depth']

        if self.setting['optimizer'] == 'adam':
            self.opt_dehaze = torch.optim.Adam(params_dehaze, lr=lr_dehaze)
            self.opt_depth = torch.optim.Adam(params_depth, lr=lr_depth)
        elif self.setting['optimizer'] == 'adamw':
            self.opt_dehaze = torch.optim.AdamW(params_dehaze, lr=lr_dehaze)
            self.opt_depth = torch.optim.AdamW(params_depth, lr=lr_depth)
        else:
            raise ValueError("Unsupported optimizer")

        self.sched_dehaze = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_dehaze, T_max=self.setting['epochs'], eta_min=lr_dehaze * 1e-2
        )
        self.sched_depth = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_depth, T_max=self.setting['epochs'], eta_min=lr_depth * 1e-2
        )

    def init_dataloaders(self):
        train_dataset = DehazingCSVDataset(
            self.setting['metadata_path_train'],
            'train',
            (self.setting['patch_size'], self.setting['patch_size'])
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.setting['batch_size'], shuffle=True,
            num_workers=self.args.num_workers,
        )

        val_dataset = DehazingCSVDataset(
            self.setting['metadata_path_val'],
            'val',
            (self.setting['patch_size'], self.setting['patch_size'])
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.setting['batch_size'],
            num_workers=self.args.num_workers,
        )

    def load_checkpoint_pretrained(self, path_dehaze, path_depth):
        if os.path.exists(path_dehaze) and os.path.exists(path_depth):
            print(f"Loading checkpoint from {path_dehaze}")
            ckpt_dehaze = torch.load(path_dehaze)
            ckpt_depth = torch.load(path_depth)

            self.net_dehaze.load_state_dict(ckpt_dehaze['dehaze_net'])
            self.net_depth.load_state_dict(ckpt_depth['depth_net'])
        else:
            print("No checkpoint found. Starting fresh.")

    def load_checkpoint(self):
        path_dehaze = os.path.join(self.save_dir, f"best_{self.args.model}" + '.pth')
        path_depth = os.path.join(self.save_dir, f"best_{self.args.model_depth}" + '.pth')
        if os.path.exists(path_dehaze) and os.path.exists(path_depth):
            print(f"Loading checkpoint from {path_dehaze}")
            ckpt_dehaze = torch.load(path_dehaze)
            ckpt_depth = torch.load(path_depth)

            self.net_dehaze.load_state_dict(ckpt_dehaze['dehaze_net'])
            self.net_depth.load_state_dict(ckpt_depth['depth_net'])
            self.opt_dehaze.load_state_dict(ckpt_dehaze['dehaze_optimizer'])
            self.opt_depth.load_state_dict(ckpt_depth['depth_optimizer'])
            
            self.best_psnr = ckpt_dehaze['dehaze_net_best_psnr']
            self.start_epoch = ckpt_dehaze['epoch_dehaze'] + 1
        else:
            print("No checkpoint found. Starting fresh.")

    def train_step(self, batch):
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)

        with autocast(enabled=not self.args.no_autocast, device_type=self.device):

            model_input = source

            dehaze_out, _, _, _ = self.net_dehaze(model_input)
            pred_depth = self.net_depth(dehaze_out)
            real_img_2_depth_img = self.depth_decoder(self.encoder(target))
            real_depth = real_img_2_depth_img[("disp", 0)]
            
            diff_dehaze = dehaze_out - target
            B, C, H, W = diff_dehaze.shape
            diff_flat = diff_dehaze.permute(0, 2, 3, 1).reshape(-1, C * H * W)
            weights = F.softmax(diff_flat, dim=-1) + 1e-7
            weights = weights.reshape(B, H, W, C).permute(0, 3, 1, 2)
            weight_sum = torch.sum(weights, dim=1, keepdim=True)

            weighted_pred_depth = pred_depth * weight_sum
            weighted_real_depth = real_depth * weight_sum

            loss_depth_consis = self.criterion_l1(weighted_pred_depth, weighted_real_depth)
            loss_depth_consis_w = self.criterion_l1(pred_depth, real_depth)

            loss_total_depth = loss_depth_consis + loss_depth_consis_w

            loss_dehaze_basic = self.criterion_l1(dehaze_out, target)
            loss_dehaze_depth_w = self.criterion_l1(pred_depth, real_depth)
            loss_dehaze_cr = self.criterion_cr(dehaze_out, target, source)

            loss_dehaze_total = (loss_dehaze_basic + 
                                 0.1 * loss_dehaze_depth_w + 
                                 loss_dehaze_cr)

            loss_freq_val = torch.tensor(0.).to(self.device)
            loss_cc_val = torch.tensor(0.).to(self.device)

            if self.args.use_freq_loss:
                loss_freq_val = LossUtils.freq_loss(dehaze_out, target)
                loss_dehaze_total += (self.args.lambda_freq * loss_freq_val)
            if self.args.use_cc_loss:
                loss_cc_val = self.cc_loss(dehaze_out, target)
                loss_dehaze_total += (self.args.lambda_cc * loss_cc_val)
            

        self.opt_dehaze.zero_grad()
        self.opt_depth.zero_grad()

        self.scaler.scale(loss_dehaze_total + loss_total_depth).backward()
        self.scaler.unscale_(self.opt_dehaze)
        self.scaler.unscale_(self.opt_depth)

        torch.nn.utils.clip_grad_norm_(self.net_dehaze.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.net_depth.parameters(), max_norm=1.0)
        
        self.scaler.step(self.opt_dehaze)
        self.scaler.step(self.opt_depth)
        self.scaler.update()

        return {
            'dehaze_total': loss_dehaze_total.item(),
            'depth_total': loss_total_depth.item(),
            'freq_loss': loss_freq_val.item(),
            'cc_loss': loss_cc_val.item(),
        }
    def train_epoch(self, epoch):
        self.net_dehaze.train()
        self.net_depth.train()

        loss_meter = {}

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        i = 0
        for batch in pbar:
            loss_dict = self.train_step(batch)
            if i == 0:
                loss_meter = {}
                for k, v in loss_dict.items():
                    loss_meter[k] = AverageMeter()
            i+=1
            for k, v in loss_dict.items():
                loss_meter[k].update(v)
            
            pbar.set_postfix(
                dehaze_loss=loss_dict['dehaze_total'], 
                depth_loss=loss_dict['depth_total'],
            )

        log_data = {"epoch": epoch}
        for k, v in loss_meter.items():
            log_data[f"train/{k}"] = v.avg
        wandb.log(log_data)

    def eval_step(self, batch):
        source = batch['source'].to(self.device)
        target = batch['target'].to(self.device)

        with torch.no_grad():
            output = self.net_dehaze(source)[0].clamp_(-1, 1)

        mse = F.mse_loss(output * 0.5 + 0.5, target * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse).mean()
        
        ssim_val = ssim(output, target, size_average=True)

        return psnr.item(), ssim_val.item(), source.size(0)

    def evaluate(self, epoch):
        self.net_dehaze.eval()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        for batch in tqdm(self.val_loader, desc="Validation"):
            psnr, val_ssim, batch_size = self.eval_step(batch)
            psnr_meter.update(psnr, batch_size)
            ssim_meter.update(val_ssim, batch_size)

        avg_psnr, avg_ssim = psnr_meter.avg, ssim_meter.avg
        
        wandb.log({
            "val/psnr": avg_psnr,
            "val/ssim": avg_ssim,
            "epoch": epoch
        })
        
        print(f'Epoch: {epoch} | Val PSNR: {avg_psnr:.4f} | Val SSIM: {avg_ssim:.4f}')
        return avg_psnr

    def save_models(self, epoch, psnr, is_best=False):
        dehaze_state = {
            'dehaze_net': self.net_dehaze.state_dict(),
            'dehaze_optimizer': self.opt_dehaze.state_dict(),
            'epoch_dehaze': epoch,
            'dehaze_net_best_psnr': self.best_psnr
        }
        depth_state = {
            'depth_net': self.net_depth.state_dict(),
            'depth_optimizer': self.opt_depth.state_dict(),
            'epoch_depth': epoch
        }

        prefix = "best" if is_best else "normal"
        torch.save(dehaze_state, os.path.join(self.save_dir, f"{prefix}_{self.args.model}.pth"))
        torch.save(depth_state, os.path.join(self.save_dir, f"{prefix}_{self.args.model_depth}.pth"))
        
    def fit(self):
        for epoch in range(self.start_epoch, self.setting['epochs'] + 1):
            self.train_epoch(epoch)
            
            self.sched_dehaze.step()
            self.sched_depth.step()

            if epoch % self.setting['eval_freq'] == 0:
                current_psnr = self.evaluate(epoch)
                
                is_best = current_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = current_psnr
                    print(f"New Best PSNR: {self.best_psnr}")
                    wandb.log({"best_psnr": self.best_psnr, "epoch": epoch})
                    self.save_models(epoch, current_psnr, is_best)

class defaultdict(dict):
    def __missing__(self, key):
        self[key] = AverageMeter()
        return self[key]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DIACMPN-dehaze-Indoor', type=str)
    parser.add_argument('--model_depth', default='DIACMPN-depth-Indoor', type=str)
    parser.add_argument('--config_path', default='./configs/indoor/default.json', type=str, help='Training Config Path')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--no_autocast', action='store_false', default=True)
    parser.add_argument('--save_dir', default='./saved_models/', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--dataset', default='RESIDE-IN', type=str)
    parser.add_argument('--exp', default='indoor', type=str)
    parser.add_argument('--freeze_clip', action='store_true')
    parser.add_argument('--use_baseline', action='store_true')
    parser.add_argument('--load_checkpoint_dehaze', type=str)
    parser.add_argument('--load_checkpoint_depth', type=str)
    parser.add_argument('--use_freq_loss', action='store_true', help='Enable Frequency Domain loss')
    parser.add_argument('--lambda_freq', default=0.1, type=float, help='Weight for Frequency loss')
    parser.add_argument('--use_cc_loss', action='store_true', help='Enable Color Consistency loss')
    parser.add_argument('--use_fadhc', action='store_true', help='Enable Color Consistency loss')
    parser.add_argument('--lambda_cc', default=0.1, type=float, help='Weight for  Color Consistency loss')

    args = parser.parse_args()
    trainer = Trainer(args, None)
    trainer.fit()