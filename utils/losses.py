import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps * self.eps))
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .05], [.25, -1.2, .25], [.05, .25, .05]])
        self.kernel = k.unsqueeze(0).repeat(1, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kw//2, kh//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def forward(self, x, y):
        x_out = self.conv_gauss(x)
        y_out = self.conv_gauss(y)
        loss = self.loss(x_out, y_out)
        return loss
