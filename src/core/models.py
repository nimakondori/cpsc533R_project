import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, log
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Off-the-shelf ViT implementation from timm
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed


class UMT(nn.Module):
    def __init__(self, 
                 backbone, 
                 image_size,
                 num_classes_1, 
                 num_classes2, 
                 drop_rate, 
                 drop_path_rate,
                 pretrained):

        super(UMT, self).__init__()
        # Create the vision transformer transformer
        self.transformer = create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes_1,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=None,
            img_size=image_size
        )
        self.num_classes2 = num_classes2
        self.embed_dim = self.transformer.embed_dim

        # Add a new cls_token for the classification task
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.head2 = nn.Linear(self.embed_dim, self.num_classes2)

    def forward(self, x, return_attention=False):
        # concatenate the cls token to the sequence
        x = self.transformer.patch_embed(x.expand(-1, 3, -1, -1))

        x = x + self.transformer.pos_embed[:, :x.shape[1], :]
        # Add a cls tokens  to the beginning of the sequence
        x = torch.cat([self.transformer.cls_token.expand(x.shape[0], -1, -1),
                       self.cls_token2.expand(x.shape[0], -1, -1),
                       x], dim=1)
        attention_maps = []
        for i, layer in enumerate(self.transformer.blocks):
            x = layer(x)
        # x = self.transformer.blocks(x)
        x = self.transformer.norm(x)
        # Head 1 from the cls token 1
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1 = self.transformer.head(x1)
        x2 = self.head2(x2)
        return [x1.view(-1, 4, 2), x2] if not return_attention else [x1.view(-1, 4, 2), x2, attention_maps]

class CNN_Basic(nn.Module):
    def __init__(self):
        super(CNN_Basic, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 8) # 4 output points                
        
    def forward(self, x):        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))        
        x = x.view(-1, 256 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                
        return x.view(-1, 4, 2)
    


# Simple ViT
################################################################

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout=0.):
        super().__init__()
        inner_dim = d_head * n_heads
        project_out = not (n_heads == 1 and d_head == d_model)

        self.n_heads = n_heads
        self.scale = d_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(d_model, inner_dim*3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attention=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(self.dropout(attn), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        if return_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_head, d_mlp, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, Attention(d_model, n_heads, d_head, dropout=dropout)),
                PreNorm(d_model, FeedForward(d_model, d_mlp, dropout=dropout))
            ]))

    def forward(self, x, return_attention=False):
        attention_maps = []

        for attn, ff in self.layers:
            x_n, attention = attn(x, return_attention=True)
            x = x_n + x
            x = ff(x) + x
            
            if return_attention:
                attention_maps.append(attention.cpu())
        
        if return_attention:
            return x, attention_maps
        else:
            return x

class ViT(nn.Module):
    def __init__(self, *, image_size, n_channels, patch_size, n_classes, d_model, n_layers, n_heads, d_mlp, d_head=64, pool='cls', dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size        

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        n_patches = (image_height // patch_height) * (image_width // patch_width)
        d_patch = n_channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),            
            nn.LayerNorm(d_patch),
            nn.Linear(d_patch, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(d_model, n_layers, n_heads, d_head, d_mlp, dropout)
        self.pool = pool        

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x, return_attention=False):        
        x = self.to_patch_embedding(x)        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        if return_attention:
            x, attention_maps = self.transformer(x, return_attention=return_attention)
        else:
            x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]        
        x = self.mlp_head(x)

        if return_attention:
            return x.view(-1, 4, 2), attention_maps
        else:
            return x.view(-1, 4, 2)
    