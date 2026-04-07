import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
# sys.path.append('../torch-mnf')
# from torch_mnf.layers import MNFLinear

class BasicBlock(nn.Module):
    """Basic ResNet block for the custom ResNet model"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetFeatureExtractor(nn.Module):
    
    def __init__(self, block, num_blocks):
        super().__init__()
        
        self.in_planes = 64
        
        # Standard ResNet layers (deterministic)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        return out

class DeterministicResNetRegression(nn.Module):
    
    def __init__(self, block, num_blocks, num_params=1):
        super().__init__()
        
        self.in_planes = 64
        
        # Standard ResNet layers (deterministic)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512 * block.expansion, 256 * block.expansion)
        self.drop1 = nn.Dropout(0.5)
        self.out = nn.Linear(256 * block.expansion, num_params)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.drop1(out)
        out = self.out(out)
        return out
    
class ResNetRegression(nn.Module):
    
    def __init__(self, block, num_blocks, num_params=1, 
                 dropout_rate=0.5):
        super().__init__()
        
        self.feature_extractor = ResNetFeatureExtractor(
            block, num_blocks)
        
        self.regressor = nn.Sequential(
            nn.Linear(512 * block.expansion, 256 * block.expansion),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256 * block.expansion, 128 * block.expansion),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128 * block.expansion, num_params)
        )
        
    def forward(self, x):
        out = self.feature_extractor(x)
        out = torch.flatten(out, 1)
        out = self.regressor(out)
        return out

class BayesianResNetRegression(nn.Module):
    """
    Bayesian ResNet for epistemic uncertainty
    and heteroscedastic aleatoric uncertainty (predicts mean and log variance)
    """
    
    def __init__(self, block, num_blocks, num_params=1,
                 num_fc_layers=2, bayesian_type='mc_dropout', dropout_rate=0.5, 
                 fix_feature_extractor=False, feature_extractor=None):
        super().__init__()
        
        if feature_extractor is None:
            print('Creating new feature extractor')
            self.feature_extractor = ResNetFeatureExtractor(block, num_blocks)
        else:
            print('Using feature extractor from base model')
            self.feature_extractor = feature_extractor
        
        if fix_feature_extractor:
            print('Fixing weights of feature extractor')
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.fc_layers = nn.ModuleList()
        in_features = 512 * block.expansion
        
        if bayesian_type == 'mc_dropout':
            print('Using MC Dropout for Bayesian ResNet')
            for _ in range(num_fc_layers):
                self.fc_layers.append(nn.Linear(in_features, in_features // 2))
                self.fc_layers.append(nn.GELU())
                self.fc_layers.append(nn.Dropout(dropout_rate))
                in_features = in_features // 2
                
        elif bayesian_type == 'mnf':
            print('Using MNF for Bayesian ResNet')
            for _ in range(num_fc_layers):
                self.fc_layers.append(MNFLinear(in_features, in_features // 2))
                self.fc_layers.append(nn.GELU())
                self.fc_layers.append(nn.BatchNorm1d(in_features // 2))
                in_features = in_features // 2
                
        else:
            
            raise ValueError(f"Invalid Bayesian type: {bayesian_type}")
        
        # Output heads: one for mean, one for log variance (aleatoric uncertainty)
        self.mean_head = nn.Linear(in_features, num_params)
        self.logvar_head = nn.Linear(in_features, num_params)
        
    
    def forward(self, x):
        out = self.feature_extractor(x)
        out = torch.flatten(out, 1)
        
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
        
        # Predict mean and log variance
        mean = self.mean_head(out)
        logvar = self.logvar_head(out)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mean, logvar
    
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding for rectangular images"""
    def __init__(self, img_size=(40, 480), patch_size=(4, 16), in_chans=2, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerRegression(nn.Module):
    """Vision Transformer for regression on rectangular spectral images"""
    def __init__(self, img_size=(40, 480), patch_size=(4, 16), in_chans=2, 
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., num_params=1):
        super().__init__()
        
        self.num_params = num_params
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                     in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (2D aware)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_params)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, num_patches, embed_dim
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Use class token for prediction
        x = x[:, 0]
        x = self.head(x)
        
        return x


class BayesianVisionTransformerRegression(nn.Module):
    """
    Bayesian Vision Transformer with MC Dropout for epistemic uncertainty
    and heteroscedastic aleatoric uncertainty (predicts mean and log variance)
    """
    def __init__(self, img_size=(40, 480), patch_size=(4, 16), in_chans=2, 
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0.5, attn_drop_rate=0., num_params=1):
        super().__init__()
        
        self.num_params = num_params
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                     in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (2D aware)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate)  # Dropout stays active for MC sampling
        )
        
        # Separate heads for mean and log variance
        self.mean_head = nn.Linear(256, num_params)
        self.logvar_head = nn.Linear(256, num_params)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, num_patches, embed_dim
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Use class token for prediction
        x = x[:, 0]
        x = self.feature_extractor(x)
        
        # Predict mean and log variance
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mean, logvar


if __name__ == "__main__":
    
    # # Test ResNet
    # print("=" * 50)
    # print("ResNet Model")
    # print("=" * 50)
    # model_resnet = DeterministicResNetRegression(BasicBlock, [2, 2, 2, 2], num_params=1)
    # summary(model_resnet, (2, 40, 480), device='cpu')
    
    # Test ViT
    print("\n" + "=" * 50)
    print("Vision Transformer Model")
    print("=" * 50)
    model_vit = VisionTransformerRegression(
        img_size=(40, 480),
        patch_size=(4, 16),  # Results in 10x30 = 300 patches
        in_chans=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        num_params=1
    )
    summary(model_vit, (2, 40, 480), device='cpu')