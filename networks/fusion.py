import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention_1 = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            activation="gelu",
            batch_first=True,
            bias=True,
        )
        self.cross_attention_2 = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            activation="gelu",
            batch_first=True,
            bias=True,
        )
        self.cross_attention_3 = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            activation="gelu",
            batch_first=True,
            bias=True,
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            LayerNorm2d(256),
        )

        self.neck1 = nn.Sequential(
            nn.Conv2d(256 * 3,256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256,256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        self.neck2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, q, k1, k2, k3):
        B, C, H, W = q.shape

        identity = q

        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k1 = k1.reshape(B, C, H * W).permute(0, 2, 1)
        k2 = k2.reshape(B, C, H * W).permute(0, 2, 1)
        k3 = k3.reshape(B, C, H * W).permute(0, 2, 1)

        k1 = self.cross_attention_1(q, k1)
        k2 = self.cross_attention_2(q, k2)
        k3 = self.cross_attention_3(q, k3)

        k1 = k1.permute(0, 2, 1).reshape(B, C, H, W)
        k2 = k2.permute(0, 2, 1).reshape(B, C, H, W)
        k3 = k3.permute(0, 2, 1).reshape(B, C, H, W)

        q_123 = torch.cat((k1, k2, k3), dim=1)
        q_123 = self.neck1(q_123)

        identity = self.shortcut(identity)

        output = self.neck2(q_123 + identity)

        return output


class CatFuse(nn.Module):
    def __init__(self, channels=256):
        super(CatFuse, self).__init__()

        self.neck = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, a, b, c, d):
        xo = torch.cat((a, b, c, d), dim=1)
        xo = self.neck(xo)
        return xo


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.drop = nn.Dropout(drop)

        self.fc3 = nn.Linear(hidden_features, out_features)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)

        x = self.act2(x) * 1024
        return x


class DAF(nn.Module):
    # DirectAddFuse
    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    def __init__(self, channels=256, r=4, modal_num=3):
        super(iAFF, self).__init__()
        self.modal_num = modal_num
        self.cam1 = MS_CAM(channels, r)
        self.cam2 = MS_CAM(channels, r)
        self.multi_proj1 = nn.Conv2d(channels, self.modal_num * channels, kernel_size=1, stride=1, padding=0)
        self.multi_proj2 = nn.Conv2d(channels, self.modal_num * channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        xa = torch.sum(inputs, dim=0)

        xa = self.cam1(xa)
        xa = self.multi_proj1(xa)
        wei1 = torch.chunk(xa, self.modal_num, dim=1)

        xi = torch.zeros_like(inputs[0], device=inputs.device)
        for i in range(self.modal_num):
            xi = xi + inputs[i] * self.sigmoid(wei1[i])

        xi = self.cam2(xi)
        xi = self.multi_proj2(xi)
        wei2 = torch.chunk(xi, self.modal_num, dim=1)

        xo = torch.zeros_like(inputs[0], device=inputs.device)
        for i in range(self.modal_num):
            xo = xo + inputs[i] * self.sigmoid(wei2[i])

        return xo


class AFF(nn.Module):
    def __init__(self, channels=256, r=4):
        super(AFF, self).__init__()
        self.cam = MS_CAM(channels, r)
        self.multi_proj = nn.Conv2d(channels, 4 * channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.neck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, a, b, c, d):
        xa = a + b + c + d
        xa = self.cam(xa)
        xa = self.multi_proj(xa)
        wei1, wei2, wei3, wei4 = torch.chunk(xa, 4, dim=1)
        xo = a * self.sigmoid(wei1) + b * self.sigmoid(wei2) + c * self.sigmoid(wei3) + d * self.sigmoid(wei4)

        xo = self.neck(xo)
        return xo


class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        if channels > 64:
            inter_channels = int(channels // r)
        else:
            inter_channels = 64

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        # wei = self.sigmoid(xlg)
        return xlg


# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#     device = torch.device("cuda:0")
#
#     x, residual= torch.ones(8,64, 32, 32).to(device),torch.ones(8,64, 32, 32).to(device)
#     channels=x.shape[1]
#
#     model=AFF(channels=channels)
#     model=model.to(device).train()
#     output = model(x, residual)
#     print(output.shape)
