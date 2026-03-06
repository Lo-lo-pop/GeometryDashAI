import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    """SE-Block: перекалибровка каналов"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DenseBlock(nn.Module):
    """DenseNet блок: каждый слой видит все предыдущие"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False),
                nn.Dropout2d(0.1)
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class GeometryDashNet(nn.Module):
    def __init__(self, action_space: int = 2):
        super().__init__()
        
        # Начальные свертки
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Dense блоки
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=3)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(64 + 3*32),
            nn.ReLU(),
            nn.Conv2d(64 + 3*32, 128, 1, bias=False),  # bias=False
            nn.AvgPool2d(2)
        )
        
        self.dense2 = DenseBlock(128, growth_rate=32, num_layers=4)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(128 + 4*32),
            nn.ReLU(),
            nn.Conv2d(128 + 4*32, 256, 1, bias=False),  # bias=False
            nn.AvgPool2d(2)
        )
        
        self.dense3 = DenseBlock(256, growth_rate=32, num_layers=5)
        
        # SE-внимание
        self.se = SqueezeExcitation(256 + 5*32)
        
        self.feature_size = 256 + 5*32
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, action_space)
        self.distance_pred = nn.Linear(256, 1)
        
        # Убрали ручную инициализацию — PyTorch сам справится

    def _init_weights(self):
        # Опционально: только для Linear слоев с bias
        for m in [self.value, self.advantage, self.distance_pred]:
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_auxiliary=False):
        x = self.stem(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        
        x = self.dense2(x)
        x = self.trans2(x)
        
        x = self.dense3(x)
        x = self.se(x)
        
        # Global pooling
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        
        features = self.classifier(x)
        
        value = self.value(features)
        adv = self.advantage(features)
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        
        if return_auxiliary:
            dist = torch.sigmoid(self.distance_pred(features)) * 500  # 0-500px
            return q, dist
        
        return q


class SimpleDQN(nn.Module):
    def __init__(self, action_space: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, action_space)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)