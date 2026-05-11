import torch
import torch.nn as nn
import torch.nn.functional as F
from .ClipResnet import CLIPInjectionNeck, CLIPResNetEncoder
from models import DepthNet
from models.UNet import UNet
import torch.fft
from .DIACMPN import LayNormal, Down, UNet, UP, LEGM, DRDB, MFM

def FADHC_dehaze_Indoor_modified(n_channels=3):
    return FADHC_modifed(
        in_chans=n_channels,
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def FADHC_dehaze_Indoor_merged(n_channels=3, freeze_clip=True, use_fadhc=True):
    return FADHC_CLIP(
        in_chans=n_channels,
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        freeze_clip=freeze_clip, use_fadhc=use_fadhc)

def FADHC_dehaze_Outdoor_modified(n_channels=3):
    return FADHC_merged(
        in_chans=n_channels,
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[4., 8., 8., 4., 4.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[4, 6, 8, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        unet_out_channels=[128, 64, 3],
        depthnet_out_channels=[96, 48]
    )

class FADHC_CLIP(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[LayNormal, LayNormal, LayNormal, LayNormal, LayNormal],
                 depthnet_out_channels=[96, 48],
                 device='cuda',
                 freeze_clip=True,
                 use_fadhc=True,
                 **kwargs):
        super(FADHC_CLIP, self).__init__()

        self.patch_size = 4
        self.in_chans = in_chans
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.kwargs = kwargs
        self.use_fadhc = use_fadhc
        self.clip_encoder = CLIPResNetEncoder(device=device, freeze=freeze_clip)
        self.clip_neck = CLIPInjectionNeck(embed_dims=embed_dims[:3])

        self.patch_embed = Down(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.legm1 = LEGM(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                          num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], norm_layer=norm_layer[0], 
                          window_size=window_size, attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])
        self.patch_merge1 = Down(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.legm2 = LEGM(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                          num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], norm_layer=norm_layer[1], 
                          window_size=window_size, attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])
        self.patch_merge2 = Down(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.legm3 = LEGM(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                          num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], norm_layer=norm_layer[2], 
                          window_size=window_size, attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = UP(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        self.mfm1 = MFM(embed_dims[3])
        self.fadhc1 = FADHC(channels=embed_dims[3])
        self.proj_depth_1 = nn.Conv2d(depthnet_out_channels[0], embed_dims[3], kernel_size=1)
        self.legm4 = LEGM(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                          num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], norm_layer=norm_layer[3], 
                          window_size=window_size, attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = UP(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        self.mfm2 = MFM(embed_dims[4])
        self.fadhc2 = FADHC(channels=embed_dims[4])
        self.proj_depth_2 = nn.Conv2d(depthnet_out_channels[1], embed_dims[4], kernel_size=1)
        self.legm5 = LEGM(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                          num_heads=num_heads[4], mlp_ratio=mlp_ratios[4], norm_layer=norm_layer[4], 
                          window_size=window_size, attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed = UP(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        self.conv1 = nn.Conv2d(embed_dims[0] * 2 + 1, embed_dims[0], 1)
        
        self.conv2 = nn.Conv2d(embed_dims[1] * 2, embed_dims[1], 1)
        
        self.conv3 = nn.Conv2d(embed_dims[2] * 2, embed_dims[2], 1)

        total_encoder_channels = embed_dims[0] + embed_dims[1] + embed_dims[2]
        self.ca = nn.Sequential(
            nn.Conv2d(total_encoder_channels, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, padding=0, bias=True),
        )

        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(total_encoder_channels, embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dims[0]), nn.ReLU(True))
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(total_encoder_channels, embed_dims[1], kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(embed_dims[1]), nn.ReLU(True))
        self.fuse_conv3 = nn.Sequential(
            nn.Conv2d(total_encoder_channels, embed_dims[2], kernel_size=5, stride=4, padding=1),
            nn.InstanceNorm2d(embed_dims[2]), nn.ReLU(True))

        self.conv_d1 = nn.Conv2d(embed_dims[2] * 2, embed_dims[2], 1)
        self.conv4 = nn.Conv2d(embed_dims[3] + embed_dims[1], embed_dims[3], 1)
        self.conv_d2 = nn.Conv2d(embed_dims[1] + embed_dims[3], embed_dims[3], 1)
        self.conv5 = nn.Conv2d(embed_dims[4] + embed_dims[0], embed_dims[4], 1)

        self.dpn = DepthNet.DN_modified(in_channels=in_chans) 
        self.drdb = DRDB()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        target_mod = 32 
        mod_pad_h = (target_mod - h % target_mod) % target_mod
        mod_pad_w = (target_mod - w % target_mod) % target_mod
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        H, W = x.shape[2:]
        
        raw_clip_feats = self.clip_encoder(x)
        d3, d2, d1 = self.clip_neck(raw_clip_feats)
        
        if d3.shape[2:] != x.shape[2:]: d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear')
        
        dp_map, feat_s8, feat_s4 = self.dpn(x)
        dp_crop = dp_map[:, :, :H, :W]
        d = self.drdb(dp_crop) 

        x_inp = self.patch_embed(x)
        
        if d3.shape[2:] != x_inp.shape[2:]:
            d3 = F.interpolate(d3, size=x_inp.shape[2:], mode='bilinear')
            
        x_dn1 = self.legm1(self.conv1(torch.cat([x_inp, d3, d], dim=1)))
        skip1 = x_dn1
        
        x = self.patch_merge1(x_dn1)
        if d2.shape[2:] != x.shape[2:]:
            d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear')
        x_dn2 = self.legm2(self.conv2(torch.cat([x, d2], dim=1)))
        skip2 = x_dn2

        x = self.patch_merge2(x_dn2)
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear')
        x_dn3 = self.legm3(self.conv3(torch.cat([x, d1], dim=1)))

        x_avg1 = self.avg_pool(x_dn1)
        x_avg2 = self.avg_pool(x_dn2)
        x_avg3 = self.avg_pool(x_dn3)
        
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3], dim=1)
        attention_score = self.ca(fea_avg)
        w1, w2, w3 = torch.chunk(attention_score, 3, dim=1)
        
        x_down1_reweight = x_dn1 * w1
        x_down2_reweight = x_dn2 * w2
        x_down3_reweight = x_dn3 * w3
        
        fuse1 = x_down1_reweight
        fuse2 = F.interpolate(x_down2_reweight, scale_factor=2)
        fuse3 = F.interpolate(x_down3_reweight, scale_factor=4)
        fuse_feature = torch.cat((fuse1, fuse2, fuse3), dim=1)
        
        fuse_1 = self.fuse_conv1(fuse_feature)
        fuse_2 = self.fuse_conv2(fuse_feature)
        fuse_3 = self.fuse_conv3(fuse_feature)

        x = self.conv_d1(torch.cat([fuse_3, x_dn3], dim=1))
        x = self.patch_split1(x)

        x = self.mfm1([x, self.skip2(skip2)]) + x 
        
        depth_feat_1 = self.proj_depth_1(feat_s8)
        depth_feat_1 = F.interpolate(depth_feat_1, size=x.shape[2:], mode='bilinear', align_corners=False)
        if self.use_fadhc:
            x = self.fadhc1(x, x_depth=depth_feat_1)
        
        x = self.legm4(self.conv4(torch.cat([x, d2], dim=1)))

        x = self.conv_d2(torch.cat([fuse_2, x], dim=1))
        x = self.patch_split2(x)

        x = self.mfm2([x, self.skip1(skip1)]) + x
        
        depth_feat_2 = self.proj_depth_2(feat_s4)
        depth_feat_2 = F.interpolate(depth_feat_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        if self.use_fadhc:
            x = self.fadhc2(x, x_depth=depth_feat_2)
        
        x = self.legm5(self.conv5(torch.cat([x, d3], dim=1)))
        x = self.patch_unembed(x)
        return x, d1, d2, d3

    def forward(self, x):
        H, W = x.shape[2:]
        x1 = self.check_image_size(x)
        feat, d11, d22, d33 = self.forward_features(x1)
        x = feat + x1[:, :3, :, :]
        x = x[:, :, :H, :W]

        return x, d11, d22, d33
    

class FADHC_merged(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[LayNormal, LayNormal, LayNormal, LayNormal, LayNormal],
                 # Default output channels of the external UNet [Deepest, Middle, Shallowest]
                 unet_out_channels=[128, 64, 3], 
                 # Default output channels of DepthNet [Scale 1/8, Scale 1/4]
                 depthnet_out_channels=[96, 48],
                 **kwargs):
        super(FADHC_merged, self).__init__()

        self.patch_size = 4
        self.in_chans = in_chans
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.kwargs = kwargs

        # --- 1. Patch Embedding ---
        self.patch_embed = Down(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # --- 2. Encoder Stage 1 ---
        self.legm1 = LEGM(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                          num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                          norm_layer=norm_layer[0], window_size=window_size,
                          attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = Down(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        # --- 3. Encoder Stage 2 ---
        self.legm2 = LEGM(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                          num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                          norm_layer=norm_layer[1], window_size=window_size,
                          attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = Down(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        # --- 4. Bottleneck ---
        self.legm3 = LEGM(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                          num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                          norm_layer=norm_layer[2], window_size=window_size,
                          attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        # --- 5. Decoder Stage 1 (Up 1) ---
        self.patch_split1 = UP(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        
        # MFM & FADHC Init
        self.mfm1 = MFM(embed_dims[3])
        self.fadhc1 = FADHC(channels=embed_dims[3])
        
        # Dynamic Projection: Map DepthNet channels (e.g. 96) to Decoder channels (e.g. 48 or 96)
        self.proj_depth_1 = nn.Conv2d(depthnet_out_channels[0], embed_dims[3], kernel_size=1)

        self.legm4 = LEGM(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                          num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                          norm_layer=norm_layer[3], window_size=window_size,
                          attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        # --- 6. Decoder Stage 2 (Up 2) ---
        self.patch_split2 = UP(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        # MFM & FADHC Init
        self.mfm2 = MFM(embed_dims[4])
        self.fadhc2 = FADHC(channels=embed_dims[4])
        
        # Dynamic Projection: Map DepthNet channels (e.g. 48) to Decoder channels (e.g. 24 or 48)
        self.proj_depth_2 = nn.Conv2d(depthnet_out_channels[1], embed_dims[4], kernel_size=1)

        self.legm5 = LEGM(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                          num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                          norm_layer=norm_layer[4], window_size=window_size,
                          attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed = UP(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        # -----------------------------------------------------------------------
        # DYNAMIC FUSION LAYER DEFINITIONS
        # -----------------------------------------------------------------------
        
        # UNet Adapters: Map UNet fixed outputs to current Embed Dims
        self.d3_conv = nn.Conv2d(unet_out_channels[2], embed_dims[0], 1) # Shallowest
        self.d2_conv = nn.Conv2d(unet_out_channels[1], embed_dims[1], 1) # Middle
        self.d1_conv = nn.Conv2d(unet_out_channels[0], embed_dims[2], 1) # Deepest

        # Encoder Fusion Convolutions
        # conv1: Input = embed_dims[0] + embed_dims[0] (UNet) + 1 (Depth DRDB)
        self.conv1 = nn.Conv2d(embed_dims[0] * 2 + 1, embed_dims[0], 1)
        
        # conv2: Input = embed_dims[1] + embed_dims[1] (UNet)
        self.conv2 = nn.Conv2d(embed_dims[1] * 2, embed_dims[1], 1)
        
        # conv3: Input = embed_dims[2] + embed_dims[2] (UNet)
        self.conv3 = nn.Conv2d(embed_dims[2] * 2, embed_dims[2], 1)

        # Global Fusion / Attention
        total_encoder_channels = embed_dims[0] + embed_dims[1] + embed_dims[2]
        
        self.ca = nn.Sequential(
            nn.Conv2d(total_encoder_channels, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, padding=0, bias=True),
        )

        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(total_encoder_channels, embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(embed_dims[0]),
            nn.ReLU(True)
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(total_encoder_channels, embed_dims[1], kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(embed_dims[1]),
            nn.ReLU(True)
        )
        self.fuse_conv3 = nn.Sequential(
            nn.Conv2d(total_encoder_channels, embed_dims[2], kernel_size=5, stride=4, padding=1),
            nn.InstanceNorm2d(embed_dims[2]),
            nn.ReLU(True)
        )

        # Decoder Fusion Convolutions
        # conv_d1: Fuses GlobalFuse3 (embed_dims[2]) + Current Feature (embed_dims[2])
        self.conv_d1 = nn.Conv2d(embed_dims[2] * 2, embed_dims[2], 1)
        
        # conv4: Fuses Current (embed_dims[3]) + UNet d2 (embed_dims[1])
        # Note: In standard architecture embed_dims[3] == embed_dims[1] usually
        self.conv4 = nn.Conv2d(embed_dims[3] + embed_dims[1], embed_dims[3], 1)

        # conv_d2: Fuses GlobalFuse2 (embed_dims[1]) + Current (embed_dims[3])
        self.conv_d2 = nn.Conv2d(embed_dims[1] + embed_dims[3], embed_dims[3], 1)

        # conv5: Fuses Current (embed_dims[4]) + UNet d3 (embed_dims[0])
        self.conv5 = nn.Conv2d(embed_dims[4] + embed_dims[0], embed_dims[4], 1)

        # -----------------------------------------------------------------------

        # External Models
        self.Unet1 = UNet(n_channels=in_chans)
        self.dpn = DepthNet.DN_modified(in_channels=in_chans) 
        self.drdb = DRDB()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        target_mod = 32 
        mod_pad_h = (target_mod - h % target_mod) % target_mod
        mod_pad_w = (target_mod - w % target_mod) % target_mod
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        H, W = x.shape[2:]
        
        # 1. Run UNet1 (Features)
        d1, d2, d3 = self.Unet1(x)
        # Dynamic Adaption
        d3 = self.d3_conv(d3)
        d2 = self.d2_conv(d2)
        d1 = self.d1_conv(d1)
        
        # 2. Run DepthNet
        dp_map, feat_s8, feat_s4 = self.dpn(x)
        
        dp_crop = dp_map[:, :, :H, :W]
        d = self.drdb(dp_crop) 

        # 3. Main Encoder Path
        x = self.patch_embed(x)
        
        # Dynamic Conv1 handles the concatenation size automatically
        x_dn1 = self.legm1(self.conv1(torch.cat([x, d3, d], dim=1)))
        skip1 = x_dn1
        
        x = self.patch_merge1(x_dn1)
        x_dn2 = self.legm2(self.conv2(torch.cat([x, d2], dim=1)))
        skip2 = x_dn2

        x = self.patch_merge2(x_dn2)
        x_dn3 = self.legm3(self.conv3(torch.cat([x, d1], dim=1)))

        # 4. Global Fusion
        x_avg1 = self.avg_pool(x_dn1)
        x_avg2 = self.avg_pool(x_dn2)
        x_avg3 = self.avg_pool(x_dn3)
        
        # This concatenation size is now handled dynamically by self.ca and self.fuse_convX
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3], dim=1)
        attention_score = self.ca(fea_avg)
        w1, w2, w3 = torch.chunk(attention_score, 3, dim=1)
        
        x_down1_reweight = x_dn1 * w1
        x_down2_reweight = x_dn2 * w2
        x_down3_reweight = x_dn3 * w3
        
        fuse1 = x_down1_reweight
        fuse2 = F.interpolate(x_down2_reweight, scale_factor=2)
        fuse3 = F.interpolate(x_down3_reweight, scale_factor=4)
        fuse_feature = torch.cat((fuse1, fuse2, fuse3), dim=1)
        
        fuse_1 = self.fuse_conv1(fuse_feature)
        fuse_2 = self.fuse_conv2(fuse_feature)
        fuse_3 = self.fuse_conv3(fuse_feature)

        # 5. Decoder Path
        x = self.conv_d1(torch.cat([fuse_3, x], dim=1))
        x = self.patch_split1(x)

        # --- Stage 1 ---
        # MFM with Encoder Skip
        x = self.mfm1([x, self.skip2(skip2)]) + x 
        
        # FADHC with Depth
        depth_feat_1 = self.proj_depth_1(feat_s8) # Dynamically projected
        depth_feat_1 = F.interpolate(depth_feat_1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.fadhc1(x, x_depth=depth_feat_1)
        
        # Concatenate with UNet d2
        x = self.legm4(self.conv4(torch.cat([x, d2], dim=1)))

        x = self.conv_d2(torch.cat([fuse_2, x], dim=1))
        x = self.patch_split2(x)

        # --- Stage 2 ---
        # MFM with Encoder Skip
        x = self.mfm2([x, self.skip1(skip1)]) + x
        
        # FADHC with Depth
        depth_feat_2 = self.proj_depth_2(feat_s4) # Dynamically projected
        depth_feat_2 = F.interpolate(depth_feat_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.fadhc2(x, x_depth=depth_feat_2)
        
        # Concatenate with UNet d3
        x = self.legm5(self.conv5(torch.cat([x, d3], dim=1)))
        
        x = self.patch_unembed(x)

        return x, d1, d2, d3
    def forward(self, x):
        H, W = x.shape[2:]
        x1 = self.check_image_size(x)
        feat, d11, d22, d33 = self.forward_features(x1)
        x = feat + x1[:, :3, :, :]
        x = x[:, :, :H, :W]

        return x, d11, d22, d33


class FADHC_modifed(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[LayNormal, LayNormal, LayNormal, LayNormal, LayNormal], **kwargs):
        super(FADHC_modifed, self).__init__()

        self.patch_size = 4
        self.in_chans = in_chans
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.kwargs = kwargs

        self.patch_embed = Down(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # Backbone Encoder
        self.legm1 = LEGM(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = Down(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.legm2 = LEGM(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = Down(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        # Bottleneck
        self.legm3 = LEGM(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        # Backbone Decoder Stage 1
        self.patch_split1 = UP(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3] # 48 channels
        
        # --- FADHC 1 (Scale 1/8) ---
        self.fadhc1 = FADHC(channels=embed_dims[3]) 
        # Projection: DepthNet sends 96ch -> FADHC needs 48ch
        self.proj_depth_1 = nn.Conv2d(96, 48, kernel_size=1) 

        self.legm4 = LEGM(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                        num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                        norm_layer=norm_layer[3], window_size=window_size,
                        attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        # Backbone Decoder Stage 2
        self.patch_split2 = UP(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4] # 24 channels
        
        # --- FADHC 2 (Scale 1/4) ---
        self.fadhc2 = FADHC(channels=embed_dims[4])
        # Projection: DepthNet sends 48ch -> FADHC needs 24ch
        self.proj_depth_2 = nn.Conv2d(48, 24, kernel_size=1)

        self.legm5 = LEGM(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed = UP(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        # Convolutional layers for skip connection fusion
        self.conv1 = nn.Conv2d(49, 24, 1)
        self.conv2 = nn.Conv2d(96, 48, 1)
        self.conv3 = nn.Conv2d(192, 96, 1)
        self.conv4 = nn.Conv2d(96, 48, 1)
        self.conv5 = nn.Conv2d(48, 24, 1)

        # UNet1 & Fusion blocks
        self.Unet1 = UNet(n_channels=in_chans)
        self.d3_conv= nn.Conv2d(3, 24, 1)
        self.d2_conv = nn.Conv2d(64, 48, 1)
        self.d1_conv = nn.Conv2d(128, 96, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(168, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, padding=0, bias=True),
        )
        self.fuse_conv1 = nn.Sequential(nn.Conv2d(168, 24, kernel_size=3, stride=1, padding=1),
                                      nn.InstanceNorm2d(24),
                                      nn.ReLU(True))
        self.fuse_conv2 = nn.Sequential(nn.Conv2d(168, 48, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(24),
                                        nn.ReLU(True))
        self.fuse_conv3 = nn.Sequential(nn.Conv2d(168, 96,  kernel_size=5, stride=4, padding=1),
                                        nn.InstanceNorm2d(24),
                                        nn.ReLU(True))
        self.conv_d1 = nn.Conv2d(192, 96, 1)
        self.conv_d2 = nn.Conv2d(96, 48, 1)

        # Depth Modules
        # Ensure we use the NEW DN class defined above
        self.dpn = DepthNet.DN_modified(in_channels=in_chans) 
        self.drdb = DRDB()

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # FIX: Align to 32 (LCM of window_size=8 and max_stride=16/32)
        # This prevents 1-pixel rounding errors that destroy FFT phase
        target_mod = 32 
        mod_pad_h = (target_mod - h % target_mod) % target_mod
        mod_pad_w = (target_mod - w % target_mod) % target_mod
        
        # Use reflection padding to avoid hard edge boundaries
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        H, W = x.shape[2:]
        
        # 1. Run UNet1 (Dehazing prior/features)
        d1, d2, d3 = self.Unet1(x)
        d3 = self.d3_conv(d3)
        d2 = self.d2_conv(d2)
        d1 = self.d1_conv(d1)
        
        # 2. Run DepthNet (Now returns tuple)
        # dp_map: final depth, feat_s8: Scale 1/8 (96ch), feat_s4: Scale 1/4 (48ch)
        dp_map, feat_s8, feat_s4 = self.dpn(x)
        
        dp_crop = dp_map[:, :, :H, :W]
        d = self.drdb(dp_crop) # Refined depth map

        # 3. Main Encoder Path
        x = self.patch_embed(x)
        x_dn1 = self.legm1(self.conv1(torch.cat([x, d3, d], dim=1)))
        skip1 = x_dn1
        
        x = self.patch_merge1(x_dn1)
        x_dn2 = self.legm2(self.conv2(torch.cat([x, d2], dim=1)))
        skip2 = x_dn2

        x = self.patch_merge2(x_dn2)
        x_dn3 = self.legm3(self.conv3(torch.cat([x, d1], dim=1)))

        # 4. Global Fusion / Attention Mechanism (unchanged)
        x_avg1 = self.avg_pool(x_dn1)
        x_avg2 = self.avg_pool(x_dn2)
        x_avg3 = self.avg_pool(x_dn3)
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3], dim=1)
        attention_score = self.ca(fea_avg)
        w1, w2, w3 = torch.chunk(attention_score, 3, dim=1)
        x_down1_reweight = x_dn1 * w1
        x_down2_reweight = x_dn2 * w2
        x_down3_reweight = x_dn3 * w3
        fuse1 = x_down1_reweight
        fuse2 = F.interpolate(x_down2_reweight, scale_factor=2)
        fuse3 = F.interpolate(x_down3_reweight, scale_factor=4)
        fuse_feature = torch.cat((fuse1, fuse2, fuse3), dim=1)
        fuse_1 = self.fuse_conv1(fuse_feature)
        fuse_2 = self.fuse_conv2(fuse_feature)
        fuse_3 = self.fuse_conv3(fuse_feature)

        # 5. Decoder Path with FADHC
        x = self.conv_d1(torch.cat([fuse_3, x], dim=1))
        x = self.patch_split1(x)

        # --- FADHC Integration 1 (Scale 1/8) ---
        # Fuse Encoder Skip first
        x_dehaze_fused_1 = x + self.skip2(skip2)
        
        # Prepare Depth features: Project 96ch -> 48ch
        depth_feat_1 = self.proj_depth_1(feat_s8)
        # Ensure spatial size match (just in case of slight rounding diffs)
        depth_feat_1 = F.interpolate(depth_feat_1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.fadhc1(x_dehaze_fused_1, x_depth=depth_feat_1)
        
        # Continue Decoder
        x = self.legm4(self.conv4(torch.cat([x, d2], dim=1)))

        x = self.conv_d2(torch.cat([fuse_2, x], dim=1))
        x = self.patch_split2(x)

        # --- FADHC Integration 2 (Scale 1/4) ---
        # Fuse Encoder Skip first
        x_dehaze_fused_2 = x + self.skip1(skip1)
        
        # Prepare Depth features: Project 48ch -> 24ch
        depth_feat_2 = self.proj_depth_2(feat_s4)
        depth_feat_2 = F.interpolate(depth_feat_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.fadhc2(x_dehaze_fused_2, x_depth=depth_feat_2)
        
        # Finalize
        x = self.legm5(self.conv5(torch.cat([x, d3], dim=1)))
        x = self.patch_unembed(x)

        return x, d1, d2, d3

    def forward(self, x):
        H, W = x.shape[2:]
        x1 = self.check_image_size(x)
        feat, d11, d22, d33 = self.forward_features(x1)
        x = feat + x1[:, :3, :, :]
        x = x[:, :, :H, :W]

        return x, d11, d22, d33

class FADHC(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(FADHC, self).__init__()
        
        self.phase_conv = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.amp_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.lambda_param = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_dehaze, x_depth):
        B, C, H, W = x_dehaze.shape

        # FIX: Change norm to 'ortho'. 
        # 'backward' (default) makes amplitude dependent on image size (H*W).
        # 'ortho' keeps amplitude scale consistent between 256x256 crops and 620x460 test images.
        fft_dehaze = torch.fft.rfft2(x_dehaze, norm='ortho')
        fft_depth = torch.fft.rfft2(x_depth, norm='ortho')

        amp_dehaze = torch.abs(fft_dehaze)
        phase_dehaze = torch.angle(fft_dehaze)
        
        amp_depth = torch.abs(fft_depth)
        phase_depth = torch.angle(fft_depth)

        sin_dehaze = torch.sin(phase_dehaze)
        cos_dehaze = torch.cos(phase_dehaze)
        sin_depth = torch.sin(phase_depth)
        cos_depth = torch.cos(phase_depth)
        
        phase_cat = torch.cat([sin_dehaze, cos_dehaze, sin_depth, cos_depth], dim=1)
        
        m_spectral = self.phase_conv(phase_cat)

        depth_response = self.amp_conv(amp_depth)
        
        rectification = 1.0 + self.lambda_param * depth_response
        
        a_refined = amp_dehaze * rectification

        a_final = a_refined * m_spectral

        fft_refined = torch.polar(a_final, phase_dehaze)
        
        # FIX: Must match the norm used in rfft2
        f_refined = torch.fft.irfft2(fft_refined, s=(H, W), norm='ortho')
        
        out = x_dehaze + f_refined
        
        return out