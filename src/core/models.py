# Define you transfomer model here
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, log


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
        elif data_type == 'lvot':
            x = self.lvot_mlp(x)
            x = x.view(-1, self.num_landmarks, 2)
        else:
            raise NotImplementedError(f"data_type {data_type} must be 'lvid' or 'lvot'")
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 4) # 4 output points

        print(1/0)
        print('Yo Yo Yo !!!')
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
