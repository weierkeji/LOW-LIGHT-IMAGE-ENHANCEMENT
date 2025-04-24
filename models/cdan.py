import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels, growth_rate))
            in_channels += growth_rate

        self.transition_layer = self._make_transition_layer(in_channels, out_channels)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)

        out = self.transition_layer(torch.cat(features, dim=1))
        return out

    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
        )

    def _make_transition_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

# 添加一个新的递归残差模块，用于细节增强
class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels, num_recurrent=2):
        super(RecurrentResidualBlock, self).__init__()
        self.num_recurrent = num_recurrent
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = x
        for _ in range(self.num_recurrent):
            out = self.conv(out)
            out = torch.add(out, residual)
            out = self.relu(out)
        return out

# 添加多尺度融合模块
class MultiScaleFusion(nn.Module):
    def __init__(self, channels):
        super(MultiScaleFusion, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch1, branch2, branch3], dim=1)
        out = self.fusion(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)

        self.dense1 = DenseBlock(64, 64, 16, 4)
        self.dense2 = DenseBlock(128, 128, 16, 4)
        self.dense3 = DenseBlock(256, 256, 16, 4)   

        # 添加多尺度融合模块
        self.msf1 = MultiScaleFusion(64)
        self.msf2 = MultiScaleFusion(128)
        self.msf3 = MultiScaleFusion(256)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp = nn.Dropout(0.2)
    
    def forward(self, x):
        skip_connections = []
        denses = []
        multi_scale_features = []
        
        out = self.conv1(x)
        out = self.maxpool(out)
        dense1 = self.dense1(out)
        denses.append(dense1)
        msf1 = self.msf1(out)  # 多尺度特征
        multi_scale_features.append(msf1)
        out = self.dp(out)
        skip_connections.append(out)

        out = self.conv2(out)
        out = self.maxpool(out)
        dense2 = self.dense2(out)
        denses.append(dense2)
        msf2 = self.msf2(out)  # 多尺度特征
        multi_scale_features.append(msf2)
        out = self.dp(out)
        skip_connections.append(out)

        out = self.conv3(out)
        out = self.maxpool(out)
        dense3 = self.dense3(out)
        denses.append(dense3)
        msf3 = self.msf3(out)  # 多尺度特征
        multi_scale_features.append(msf3)
        out = self.dp(out)
        skip_connections.append(out)

        out = self.conv4(out)
        out = self.dp(out)

        return out, skip_connections, denses, multi_scale_features
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(256)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(128)
        self.bn2 = nn.BatchNorm2d(128)
    
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.cbam3 = CBAM(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(3)
        
        # 添加递归残差块进行细节增强
        self.recurrent1 = RecurrentResidualBlock(256)
        self.recurrent2 = RecurrentResidualBlock(128)
        self.recurrent3 = RecurrentResidualBlock(64)
        
        # 添加嵌套跳跃连接的处理
        self.nested_fusion1 = nn.Conv2d(256 * 2, 256, kernel_size=1)
        self.nested_fusion2 = nn.Conv2d(128 * 2, 128, kernel_size=1)
        self.nested_fusion3 = nn.Conv2d(64 * 2, 64, kernel_size=1)
        
        self.final_dense = DenseBlock(3, 3, 16, 4)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, out, skip_connections, denses, multi_scale_features=None):
        # 第一解码层
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.add(out, skip_connections[2])
        if multi_scale_features:  # 如果有多尺度特征
            # 嵌套跳跃连接：融合多尺度特征
            nested_features = torch.cat([out, multi_scale_features[2]], dim=1)
            out = self.nested_fusion1(nested_features)
        out = self.cbam1(out)
        out = self.recurrent1(out)  # 应用递归处理改善细节

        # 应用密集特征
        out = torch.mul(out, denses[2])
        
        # 第二解码层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, skip_connections[1])
        if multi_scale_features:  # 如果有多尺度特征
            # 嵌套跳跃连接：融合多尺度特征
            nested_features = torch.cat([out, multi_scale_features[1]], dim=1)
            out = self.nested_fusion2(nested_features)
        out = self.cbam2(out)
        out = self.recurrent2(out)  # 应用递归处理改善细节

        # 应用密集特征
        out = torch.mul(out, denses[1])
        
        # 第三解码层
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, skip_connections[0])
        if multi_scale_features:  # 如果有多尺度特征
            # 嵌套跳跃连接：融合多尺度特征
            nested_features = torch.cat([out, multi_scale_features[0]], dim=1)
            out = self.nested_fusion3(nested_features)
        out = self.cbam3(out)
        out = self.recurrent3(out)  # 应用递归处理改善细节

        # 应用密集特征
        out = torch.mul(out, denses[0])
        
        # 最终输出层
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, x)

        dense_out = self.final_dense(out)
        out = self.sigmoid(dense_out)

        return out


# 原始CDAN模型
class CDAN(nn.Module):
    def __init__(self):
        super(CDAN, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = CBAM(512)
        self.decoder = Decoder()
    
    def forward(self, x):
        out, skip_connections, denses, _ = self.encoder(x)
        out = self.bottleneck(out)
        out = self.decoder(x, out, skip_connections, denses)

        return out

# 增强版CDAN模型 - 集成了多尺度特征和嵌套跳跃连接
class EnhancedCDAN(nn.Module):
    def __init__(self):
        super(EnhancedCDAN, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = CBAM(512)
        self.decoder = Decoder()
    
    def forward(self, x):
        out, skip_connections, denses, multi_scale_features = self.encoder(x)
        out = self.bottleneck(out)
        out = self.decoder(x, out, skip_connections, denses, multi_scale_features)

        return out