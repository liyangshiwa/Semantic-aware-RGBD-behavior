
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv

__all__ = ['SPDConv', 'FGM', 'OmniKernel', 'CSPOmniKernel']


class SPDConv(nn.Module):
    def __init__(self, inc, ouc, dimension=1):
        """Initialize SPDConv with input/output channels."""
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([
            x[..., ::2, ::2],   # 左上
            x[..., 1::2, ::2],  # 右上
            x[..., ::2, 1::2],  # 左下
            x[..., 1::2, 1::2]  # 右下
        ], 1)
        x = self.conv(x)
        return x


class FGM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)

    
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False)


        self.bn = nn.BatchNorm2d(dim * 2)

  
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):

        hidden = self.project_in(x)
        hidden = self.bn(self.dwconv(hidden))

     
        x1, x2 = hidden.chunk(2, dim=1)

      
        x_gated = x1 * F.silu(x2)


        return self.project_out(x_gated)


class OmniKernel(nn.Module):
      def __init__(self, dim, ker=11):
        super().__init__()
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )


        self.dw_1k = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), groups=dim, bias=False)
        self.dw_k1 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), groups=dim, bias=False)
        self.dw_kk = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, groups=dim, bias=False)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=dim, bias=False)

  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sca_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )

             self.fgm = FGM(dim)

             self.fusion_bn = nn.BatchNorm2d(dim)          self.fusion_act = nn.SiLU(inplace=True)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
    
        out = self.in_conv(x)
        attention_weight = self.sca_conv(self.pool(out))
        x_sca = attention_weight * out

        x_refined = self.fgm(x_sca)

        fused = x + self.dw_1k(out) + self.dw_k1(out) + self.dw_kk(out) + self.dw_11(out) + x_refined

        fused = self.fusion_act(self.fusion_bn(fused))
        return self.out_conv(fused)


class CSPOmniKernel(nn.Module):
  
    def __init__(self, dim, e=0.25, ker=11): 
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e), ker=ker)

    def forward(self, x):
        ok_branch, identity = torch.split(
            self.cv1(x),
            [int(self.cv1.conv.out_channels * self.e), int(self.cv1.conv.out_channels * (1 - self.e))],
            dim=1
        )
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))



