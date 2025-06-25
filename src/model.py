import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim=512, heads=4, ff_dim=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: [seq_len, batch, dim]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class HybridConvTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(3, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DepthwiseSeparableConv(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DepthwiseSeparableConv(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 28 * 28, 512)
        self.transformer = TransformerBlock(dim=512, heads=4, ff_dim=1024)
        self.linear2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.pool1(self.conv1(x))  # [B, 16, 112, 112]
        x = self.pool2(self.conv2(x))  # [B, 32, 56, 56]
        x = self.pool3(self.conv3(x))  # [B, 64, 28, 28]
        x = self.flatten(x)            # [B, 64*28*28]
        x = self.linear1(x)            # [B, 512]
        x = x.unsqueeze(1)             # [1, B, 512] for transformer
        x = self.transformer(x)        # [1, B, 512]
        x = x.squeeze(1)               # [B, 512]
        x = self.linear2(x)            # [B, 1]
        return x