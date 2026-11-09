import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ci, co, k=7, d=1, p=0.1):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=(k - 1) * d // 2, dilation=d),
            nn.BatchNorm1d(co), nn.GELU(), nn.Dropout(p))

    def forward(self, x):
        return self.b(x)


class DenoiseNet(nn.Module):
    """1D U-Net: 3 strided-conv downsamplings, dilated bottleneck, skip connections."""

    def __init__(self, in_ch=4, bc=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, bc, 7);      self.p1 = nn.Conv1d(bc, bc, 4, 2, 1)
        self.e2 = ConvBlock(bc, bc * 2, 7);     self.p2 = nn.Conv1d(bc * 2, bc * 2, 4, 2, 1)
        self.e3 = ConvBlock(bc * 2, bc * 4, 5); self.p3 = nn.Conv1d(bc * 4, bc * 4, 4, 2, 1)
        self.bo = nn.Sequential(ConvBlock(bc * 4, bc * 4, 5, 2),
                                ConvBlock(bc * 4, bc * 4, 5, 4),
                                ConvBlock(bc * 4, bc * 4, 5, 8))
        self.u3 = nn.ConvTranspose1d(bc * 4, bc * 2, 4, 2, 1); self.d3 = ConvBlock(bc * 2 + bc * 4, bc * 2, 5)
        self.u2 = nn.ConvTranspose1d(bc * 2, bc, 4, 2, 1);     self.d2 = ConvBlock(bc + bc * 2, bc, 7)
        self.u1 = nn.ConvTranspose1d(bc, bc, 4, 2, 1);         self.d1 = ConvBlock(bc + bc, bc, 7)
        self.head = nn.Conv1d(bc, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b = self.bo(self.p3(e3))
        d3 = self.d3(self._c(self.u3(b), e3))
        d2 = self.d2(self._c(self.u2(d3), e2))
        d1 = self.d1(self._c(self.u1(d2), e1))
        return self.head(d1)

    @staticmethod
    def _c(x, skip):
        d = skip.shape[2] - x.shape[2]
        if d > 0:
            x = nn.functional.pad(x, (0, d))
        elif d < 0:
            x = x[:, :, :skip.shape[2]]
        return torch.cat([x, skip], 1)
