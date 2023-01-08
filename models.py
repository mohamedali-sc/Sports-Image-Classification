#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#VGG architecture from scratch
VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])
        
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)


# In[ ]:


#Resnet architecture from scratch
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


# In[ ]:


#GoogleNet architecture from scratch

class googlenet(nn.Module):
    def __init__(self,in_channels=3,classes=6):
        super(googlenet,self).__init__()
        self.conv1=con_block(in_channels=in_channels,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=con_block(64,192,kernel_size=3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.inc1=Inception_block(192,64,96,128,16,32,32)
        self.inc2=Inception_block(256,128,128,192,32,96,64)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.inc3=Inception_block(480,192,96,208,16,48,64)
        self.inc4=Inception_block(512,160,112,224,24,64,64)
        self.inc5=Inception_block(512,128,128,256,24,64,64)
        self.inc6=Inception_block(512,112,144,288,32,64,64)
        self.inc7=Inception_block(528,256,160,320,32,128,128)
        
        self.maxpool4=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inc8=Inception_block(832,256,160,320,32,128,128)
        self.inc9=Inception_block(832,384,192,384,48,128,128)
        
        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout=nn.Dropout(p=0.4)
        self.fc1=nn.Linear(1024,classes)
        
        
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.inc1(x)
        x=self.inc2(x)
        x=self.maxpool3(x)
        x=self.inc3(x)
        x=self.inc4(x)
        x=self.inc5(x)
        x=self.inc6(x)
        x=self.inc7(x)

        x=self.maxpool4(x)
        x=self.inc8(x)
        x=self.inc9(x)
        x=self.avgpool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.dropout(x)
        x=self.fc1(x)
        
        return x
        
class Inception_block(nn.Module):
    def __init__(self,in_channels,out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,out_1x1pool):
        super(Inception_block,self).__init__()
        self.branch1=con_block(in_channels,out_1x1,kernel_size=1)
        self.branch2=nn.Sequential(
            con_block(in_channels,red_3x3,kernel_size=1),
            con_block(red_3x3,out_3x3,kernel_size=3,padding=1)
        )
        self.branch3=nn.Sequential(
            con_block(in_channels,red_5x5,kernel_size=1),
            con_block(red_5x5,out_5x5,kernel_size=5,padding=2)
        )
    
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            con_block(in_channels,out_1x1pool,kernel_size=1)
            
        )
        
    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)
    
class con_block(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(con_block,self).__init__()
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels,out_channels,**kwargs)
        self.batchnorm=nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))



# In[ ]:


#EfficientNet architecture from scratch

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


# In[ ]:


#transformer for image classification from scratch

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2


        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )

    def forward(self, x):
        
        x = self.proj(
                x
            )  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  

        return x


class Attention(nn.Module):
    
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  
        dp = (
           q @ k_t
        ) * self.scale 
        attn = dp.softmax(dim=-1)  
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  
        weighted_avg = weighted_avg.flatten(2)  

        x = self.proj(weighted_avg) 
        x = self.proj_drop(x)  

        return x


class MLP(nn.Module):
   
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
      
        x = self.fc1(
                x
        ) 
        x = self.act(x)  
        x = self.drop(x)  
        x = self.fc2(x)  
        x = self.drop(x)  

        return x


class Block(nn.Module):
    
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
       
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            n_classes=6,
            embed_dim=50,
            depth=10,
            n_heads=10,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
       
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x


# In[ ]:


#modified GoogleNet with pixel_normaliztion and weight_scale


class googlenet(nn.Module):
    def __init__(self,in_channels=3,classes=6):
        super(googlenet,self).__init__()
        self.conv1=con_block(in_channels=in_channels,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=con_block(64,192,kernel_size=3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.inc1=Inception_block(192,64,96,128,16,32,32)
        self.inc2=Inception_block(256,128,128,192,32,96,64)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.inc3=Inception_block(480,192,96,208,16,48,64)
        self.inc4=Inception_block(512,160,112,224,24,64,64)
        self.inc5=Inception_block(512,128,128,256,24,64,64)
        self.inc6=Inception_block(512,112,144,288,32,64,64)
        self.inc7=Inception_block(528,256,160,320,32,128,128)
        
        self.maxpool4=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inc8=Inception_block(832,256,160,320,32,128,128)
        self.inc9=Inception_block(832,384,192,384,48,128,128)
        
        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout=nn.Dropout(p=0.4)
        self.fc1=nn.Linear(1024,classes)
        
        
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.inc1(x)
        x=self.inc2(x)
        x=self.maxpool3(x)
        x=self.inc3(x)
        x=self.inc4(x)
        x=self.inc5(x)
        x=self.inc6(x)
        x=self.inc7(x)

        x=self.maxpool4(x)
        x=self.inc8(x)
        x=self.inc9(x)
        x=self.avgpool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.dropout(x)
        x=self.fc1(x)
        
        return x
                

class Inception_block(nn.Module):
    def __init__(self,in_channels,out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,out_1x1pool):
        super(Inception_block,self).__init__()
        self.branch1=con_block(in_channels,out_1x1,kernel_size=1)
        self.branch2=nn.Sequential(
            con_block(in_channels,red_3x3,kernel_size=1),
            con_block(red_3x3,out_3x3,kernel_size=3,padding=1)
        )
        self.branch3=nn.Sequential(
            con_block(in_channels,red_5x5,kernel_size=1),
            con_block(red_5x5,out_5x5,kernel_size=5,padding=2)
        )
    
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            con_block(in_channels,out_1x1pool,kernel_size=1)
            
        )
        
    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)
        
        
        

class con_block(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(con_block,self).__init__()
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels,out_channels,**kwargs)
        
        self.scale = (2/ (in_channels * (kwargs["kernel_size"] ** 2))) ** 0.5 ##weight_scale
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
        
        #self.batchnorm=nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
       # g= x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        #x=self.relu(self.batchnorm(self.conv(x))) 
        x=self.relu(self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1))
        x=x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8) ##pixel_normaliztion
        return x
        


# In[ ]:


#CNN with transformer
class googlenet(nn.Module):
    def __init__(self,in_channels=3,classes=6):
        super(googlenet,self).__init__()
        self.conv1=con_block(in_channels=in_channels,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=con_block(64,192,kernel_size=3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.inc1=Inception_block(192,64,96,128,16,32,32)
        self.inc2=Inception_block(256,128,128,192,32,96,64)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.inc3=Inception_block(480,192,96,208,16,48,64)
        self.inc4=Inception_block(512,160,112,224,24,64,64)
        self.inc5=Inception_block(512,128,128,256,24,64,64)
        self.inc6=Inception_block(512,112,144,288,32,64,64)
        self.inc7=Inception_block(528,256,160,320,32,128,128)
        
        self.maxpool4=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inc8=Inception_block(832,256,160,320,32,128,128)
        self.branch0=nn.Conv2d(832,3,kernel_size=1)

     

        ##after that 1*1 to 3
        ##after that upsampling
        ##transformervisoion





        # self.inc9=Inception_block(832,384,192,384,48,128,128)
        
        # self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
        # self.dropout=nn.Dropout(p=0.4)
        # self.fc1=nn.Linear(1024,classes)
        self.batchnorm=nn.BatchNorm2d(3)
        self.relu=nn.ReLU()
        self.t=VisionTransformer()


        
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.inc1(x)
        x=self.inc2(x)
        x=self.maxpool3(x)
        x=self.inc3(x)
        x=self.inc4(x)
        x=self.inc5(x)
        x=self.inc6(x)
        x=self.inc7(x)

        x=self.maxpool4(x)
        x=self.inc8(x)
        x=self.branch0(x)
        x=f.interpolate(x,scale_factor=32,mode="nearest")
        x=self.relu(self.batchnorm(x))

        x=self.t(x)
    
        # x=self.inc9(x)
        # x=self.avgpool(x)
        # x=x.reshape(x.shape[0],-1)
        # x=self.dropout(x)
        # x=self.fc1(x)
        
        return x
        

        
        
        
       
        
        

class Inception_block(nn.Module):
    def __init__(self,in_channels,out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,out_1x1pool):
        super(Inception_block,self).__init__()
        self.branch1=con_block(in_channels,out_1x1,kernel_size=1)
        self.branch2=nn.Sequential(
            con_block(in_channels,red_3x3,kernel_size=1),
            con_block(red_3x3,out_3x3,kernel_size=3,padding=1)
        )
        self.branch3=nn.Sequential(
            con_block(in_channels,red_5x5,kernel_size=1),
            con_block(red_5x5,out_5x5,kernel_size=5,padding=2)
        )
    
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            con_block(in_channels,out_1x1pool,kernel_size=1)
            
        )
        
    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)
        
        
        

class con_block(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(con_block,self).__init__()
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels,out_channels,**kwargs)
        
       # self.scale = (2/ (in_channels * (kwargs["kernel_size"] ** 2))) ** 0.5
       # self.bias = self.conv.bias
       # self.conv.bias = None

        # initialize conv layer
       # nn.init.normal_(self.conv.weight)
       # nn.init.zeros_(self.bias)
        
        self.batchnorm=nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
       # g= x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        x=self.relu(self.batchnorm(self.conv(x))) 
       # x=x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return x
        

