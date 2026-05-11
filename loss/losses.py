import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthDistillationLoss(nn.Module):
    def __init__(self, alpha=1.0, lambda_distill=0.1):
        """
        alpha: Weight for gradient (edge) matching loss.
        lambda_distill: Weight of this entire loss in the global optimization.
        """
        super(DepthDistillationLoss, self).__init__()
        self.alpha = alpha
        self.lambda_distill = lambda_distill
        self.l1_loss = nn.L1Loss()

    def gradient_xy(self, img):
        """Calculates gradients (edges) in x and y directions."""
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return grad_x, grad_y

    def forward(self, student_depth, teacher_depth):
        """
        student_depth: Output from your lightweight DE network.
        teacher_depth: Output from frozen Depth Anything V2.
        """
        # 1. Pixel-wise L1 Loss (The formula from your image)
        loss_pixel = self.l1_loss(student_depth, teacher_depth)

        # 2. Gradient Loss (For sharper edges - Research Grade addition)
        student_dx, student_dy = self.gradient_xy(student_depth)
        teacher_dx, teacher_dy = self.gradient_xy(teacher_depth)
        
        loss_grad = self.l1_loss(student_dx, teacher_dx) + \
                    self.l1_loss(student_dy, teacher_dy)

        # Total Distillation Loss
        total_distill = loss_pixel + (self.alpha * loss_grad)
        
        return self.lambda_distill * total_distill

        
class LossUtils:
    @staticmethod
    def get_dcp(img, patch_size=15):
        min_rgb, _ = torch.min(img, dim=1, keepdim=True)
        padding = patch_size // 2
        dcp = -F.max_pool2d(
            -min_rgb, 
            kernel_size=patch_size, 
            stride=1, 
            padding=padding
        )
        return dcp

    @staticmethod
    def freq_loss(pred, target):
        fft_pred = torch.fft.rfft2(pred, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')
        
        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)
        
        return F.l1_loss(mag_pred, mag_target)
    
class SpectralConsensusLoss(nn.Module):
    def __init__(self, lambda_amp=1.0, lambda_phase=1.0):
        super(SpectralConsensusLoss, self).__init__()
        self.lambda_amp = lambda_amp
        self.lambda_phase = lambda_phase
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        fft_pred = torch.fft.rfft2(pred, norm='backward')
        fft_target = torch.fft.rfft2(target, norm='backward')

        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        loss_amp = self.l1_loss(amp_pred, amp_target)

        eps = 1e-8
        phase_vec_pred = fft_pred / (amp_pred + eps)
        phase_vec_target = fft_target / (amp_target + eps)
        
        loss_phase = self.l1_loss(phase_vec_pred, phase_vec_target)

        return (self.lambda_amp * loss_amp) + (self.lambda_phase * loss_phase)
    
class DarkChannelLoss(nn.Module):
    def __init__(self, window_size=15):
        super(DarkChannelLoss, self).__init__()
        self.window_size = window_size

    def get_dark_channel(self, x):
        min_c, _ = torch.min(x, dim=1, keepdim=True)
        

        pad = self.window_size // 2
        dark_channel = -F.max_pool2d(
            -min_c, 
            kernel_size=self.window_size, 
            stride=1, 
            padding=pad
        )
        return dark_channel

    def forward(self, pred, target):
        dc_pred = self.get_dark_channel(pred)
        dc_target = self.get_dark_channel(target)
        
        return F.l1_loss(dc_pred, dc_target)
    

class ColorConsistencyLoss(nn.Module):
    def __init__(self, kernel_size=31, sigma=5.0):
        super(ColorConsistencyLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = self.get_gaussian_kernel(kernel_size, sigma)
        self.kernel = kernel.expand(3, 1, kernel_size, kernel_size).contiguous()
        
    def get_gaussian_kernel(self, kernel_size, sigma):
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        base_1d = torch.exp(-(coords.pow(2)) / (2 * sigma ** 2))
        base_1d = base_1d / base_1d.sum()
        kernel_2d = base_1d.unsqueeze(1) @ base_1d.unsqueeze(0)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        if self.kernel.device != pred.device:
            self.kernel = self.kernel.to(pred.device)
        pad = self.kernel_size // 2
        pred_blur = F.conv2d(pred, self.kernel, padding=pad, groups=3)
        target_blur = F.conv2d(target, self.kernel, padding=pad, groups=3)
        return F.l1_loss(pred_blur, target_blur)