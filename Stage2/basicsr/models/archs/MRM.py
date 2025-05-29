## Modulation_refinement: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers

from einops import rearrange

from basicsr.models.src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x,info=None):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x,info=None):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self,input):
        q_in,kv_in = input.chunk(2, dim=1)
        b, c, h, w = q_in.shape
        q = self.q_dwconv(self.q(q_in))
        kv = self.kv_dwconv(self.kv(kv_in))
        k,v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = out + q_in
        return out
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.attn = Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x1,x2 = x.chunk(2,dim=1)
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x = torch.cat([x1, x2],dim=1)
        x = x1 + self.attn(x)
        x = x + self.ffn(self.norm3(x))
        x = torch.cat([x,x2],dim=1)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class OverlapPatchEmbed1(nn.Module):
    def __init__(self, in_c=3, embed_dim=192, bias=False):
        super(OverlapPatchEmbed1, self).__init__()

        self.conv1 = nn.Conv2d(in_c*2, embed_dim//4, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class OverlapPatchEmbed2(nn.Module):
    def __init__(self, in_c=3, embed_dim=96, bias=False):
        super(OverlapPatchEmbed2, self).__init__()

        self.conv1 = nn.Conv2d(in_c*2, embed_dim//2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class OverlapPatchEmbed3(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed3, self).__init__()

        self.conv1 = nn.Conv2d(in_c*2, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Lap_Pyramid_Bicubic(nn.Module):
    """

    """
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Bicubic, self).__init__()

        self.interpolate_mode = 'bicubic'
        self.num_high = num_high

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            down = nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode=self.interpolate_mode, align_corners=True)
            up = nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode, align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
        return image

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks):
        super(Trans_low, self).__init__()

        model = [nn.Conv2d(3, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out

class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high

        self.init_layer = nn.Sequential(*[nn.Conv2d(9, 64, 3, padding=1),
                                          nn.LeakyReLU()])
        self.resblock1 =  ResidualBlock(64)
        self.resblock2 =  ResidualBlock(64)
        self.resblock3 =  ResidualBlock(64)

        self.out = nn.Conv2d(64, 3, 3, padding=1)

        self.pos_embdding=nn.Conv2d(3, 64, 3, padding=1)
        #########################################################################
        dim = 48
        heads = [1,2,4,8]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        num_blocks = [1,6,6,8]

        self.patch_embed1 = OverlapPatchEmbed1(3, 192*2)
        self.trans_mask_block1 =nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.proj1 = nn.Conv2d(192, 3, kernel_size=3, stride=1, padding=1, bias=bias)

        self.patch_embed2 = OverlapPatchEmbed2(3, 96*2)
        self.trans_mask_block2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.proj2 = nn.Conv2d(96, 3, kernel_size=3, stride=1, padding=1, bias=bias)

        self.patch_embed3 = OverlapPatchEmbed3(3, 48*2)
        self.trans_mask_block3 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.proj3 = nn.Conv2d(48, 3, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x, pyr_original, fake_low,realA_low):

        pos = self.pos_embdding(realA_low)

        mask = self.init_layer(x)
        mask = mask + pos
        mask = self.resblock1(mask)
        mask = mask + pos
        mask = self.resblock2(mask)
        mask = mask + pos
        mask = self.resblock3(mask)
        mask = mask + pos
        mask = self.out(mask)

        input1 = torch.concat((pyr_original[-2],mask),dim=1)
        input1 = self.patch_embed1(input1)
        hf1 = self.trans_mask_block1(input1)
        hf1,_ = hf1.chunk(2, dim=1)
        hf1 = self.proj1(hf1)
        hf1 = hf1+mask

        mask = nn.functional.interpolate(mask, size=(pyr_original[-2 - 1].shape[2], pyr_original[-2 - 1].shape[3]))
        input2 = torch.concat((pyr_original[-2-1],mask),dim=1)
        input2 = self.patch_embed2(input2)
        hf2 = self.trans_mask_block2(input2)
        hf2,_ = hf2.chunk(2, dim=1)
        hf2 = self.proj2(hf2)
        hf2 = hf2 + mask

        mask = nn.functional.interpolate(mask, size=(pyr_original[-2 - 2].shape[2], pyr_original[-2 - 2].shape[3]))
        input3 = torch.concat((pyr_original[-2-2],mask),dim=1)
        input3 = self.patch_embed3(input3)
        hf3 = self.trans_mask_block3(input3)
        hf3,_ = hf3.chunk(2, dim=1)
        hf3 = self.proj3(hf3)
        hf3 = hf3 + mask

        pyr_result = [hf3,hf2,hf1,fake_low]

        return pyr_result

model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        share_encoder=0,
        condition=True,
        input_condition=False)

diffusion = ResidualDiffusion(
        model,
        image_size=64,
        timesteps=1000,           # number of steps
        sampling_timesteps=5,
        objective='pred_res_noise',
        loss_type='l1',            # L1 or L2
        condition=True,
        sum_scale = 1,
        input_condition=False,
        input_condition_mask=False
    )

folder = [""] # Stage 1 file path
trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=1,
    num_samples=1,
    train_lr=8e-5,
    train_num_steps=80000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=True,
    save_and_sample_every=1000,
    equalizeHist=False,
    crop_patch=False,
    generation = False
)

class Modulation_refinement(nn.Module):
    def __init__(self, ):

        super(Modulation_refinement, self).__init__()

        self.lap_pyramid = Lap_Pyramid_Conv(num_high=3)
        self.ddpm = trainer
        self.ddpm.load(r"Weight of the first stage model")
        trans_high = Trans_high(3, num_high=3)
        self.trans_high = trans_high.cuda()
    def forward(self, inp_img):

        pyr_A = self.lap_pyramid.pyramid_decom(img=inp_img)
        fake_B_low = self.ddpm.test(x_input_sample=pyr_A[-1], last=True)
        real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low,real_A_up)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)
        return fake_B_full
