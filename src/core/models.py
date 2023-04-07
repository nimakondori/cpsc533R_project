import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, log
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from transformers import ViTFeatureExtractor, ViTForImageClassification


class UMMT(nn.Module):
    def __init__(self, 
                 image_size=224, 
                 patch_size=16,
                 num_landmarks=2,
                 hidden_dim=768, 
                 num_heads=12, 
                 num_layers=12, 
                 dropout=0.1):
        
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2
        self.hidden_dim = hidden_dim
        self.num_landmarks = num_landmarks

        self.patch_embedding = nn.Conv2d(in_channels=1, out_channels=hidden_dim,
                                         kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers)
        self.lvot_mlp = MLP(input_dim=hidden_dim, output_dim=2*num_landmarks, dropout_p=dropout) # 2*num_landmarks since 2 landmarks each with x,y coordinates
        self.lvid_mlp = MLP(input_dim=hidden_dim, output_dim=4*num_landmarks, dropout_p=dropout) # LVID has 4 landmarks so 4x num_landmarks 

    def forward(self, input_frames, data_type=None):
        x = self.patch_embedding(input_frames) #(b, hidden_dim, patch_size, patch_size)
        x = x.flatten(2).transpose(1, 2) # (b, num_patch, hidden_dim)
        b, n, c = x.size()
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (b, num_patch + 1, hidden_dim) 
        x += self.pos_embedding[:, :(n + 1)] # positional encoding added to all but cls token
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_decoder(x, self.pos_embedding[:, :self.num_patches + 1])
        # Extracting out the cls token 
        x = x[:, 1:, :]
        # Apply mean pooling to get the representation of the image
        x = x.mean(dim=1)   
        if data_type == 'lvid':
            x = self.lvid_mlp(x)
            x = x.view(-1, 2*self.num_landmarks, 2)
        return x

# class UMMT(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
#         super(UMMT).__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.num_heads = num_heads
        
#         # Encoder
#         self.enc_embedding = nn.Linear(input_dim, hidden_dim)
#         self.enc_positional_encoding = PositionalEncoding(hidden_dim)
#         self.enc_layers = nn.ModuleList([
#             EncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
#         ])

#         # Decoder
#         self.dec_embedding = nn.Linear(output_dim, hidden_dim)
#         self.dec_positional_encoding = PositionalEncoding(hidden_dim)
#         self.dec_layers = nn.ModuleList([
#             DecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
#         ])
#         self.dec_fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, y):
#         # Encoder
#         x = self.enc_embedding(x)
#         x = self.enc_positional_encoding(x)
#         for layer in self.enc_layers:
#             x = layer(x)

#         # Decoder
#         y = self.dec_embedding(y)
#         y = self.dec_positional_encoding(y)
#         for layer in self.dec_layers:
#             y = layer(y, x)
#         y = self.dec_fc(y)

#         return y


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return x


# class EncoderLayer(nn.Module):
#     def __init__(self, hidden_dim, num_heads):
#         super().__init__()

#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)

#     def forward(self, x):
#         x = x.permute(1, 0, 2)
#         x, _ = self.self_attn(x, x, x)
#         x = x.permute(1, 0, 2)
#         x = self.norm1(x) + x
#         x = x.permute(1, 0, 2)
#         x = self.feed_forward(x)
#         x = x.permute(1, 0, 2)
#         x = self.norm2(x) + x
#         return x


# class DecoderLayer(nn.Module):
#     def __init__(self, hidden_dim, num_heads):
#         super().__init__()

#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
#         self.enc_dec_attn = nn.MultiheadAttention(hidden_dim)


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
    
class PreTrainedViT(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, num_channels=1, do_normalize=False)
        self.vit = ViTForImageClassification.from_pretrained(model_name, num_labels=n_classes, ignore_mismatched_sizes=True, num_channels=1)

    def forward(self, x):
        x = self.feature_extractor(x.cuda(), return_tensors='pt')

        # get the logits (raw scores) for each class
        x['pixel_values'] = x['pixel_values'].cuda()
        logits = self.vit(**x).logits
        
        return logits.view(-1, 4, 2)

    
