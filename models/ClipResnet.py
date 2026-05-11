import torch
import torch.nn as nn
import torch.nn.functional as F
import clip 

class CLIPResNetEncoder(nn.Module):
    def __init__(self, device='cuda', freeze=True):
        super().__init__()
        clip_model, _ = clip.load("RN50", device=device, jit=False)
        self.visual = clip_model.visual.float()
        
        for param in self.visual.parameters():
            param.requires_grad = not freeze
            
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std

        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x)

        features = []
        x = self.visual.layer1(x)
        features.append(x)
        x = self.visual.layer2(x)
        features.append(x)
        x = self.visual.layer3(x)
        features.append(x)
        
        return features

class CLIPInjectionNeck(nn.Module):
    """
    Projects CLIP features to match DIACMPN embed_dims.
    """
    def __init__(self, embed_dims=[24, 48, 96]):
        super().__init__()
        # CLIP RN50 Channels: Layer1=256, Layer2=512, Layer3=1024
        self.proj_d3 = nn.Conv2d(256, embed_dims[0], kernel_size=1)  # Shallowest (1/4)
        self.proj_d2 = nn.Conv2d(512, embed_dims[1], kernel_size=1)  # Middle (1/8)
        self.proj_d1 = nn.Conv2d(1024, embed_dims[2], kernel_size=1) # Deepest (1/16)

    def forward(self, features):
        c1, c2, c3 = features
        return self.proj_d3(c1), self.proj_d2(c2), self.proj_d1(c3)