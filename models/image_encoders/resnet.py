from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from models.image_encoders.od_resnet import od_resnet50


from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder

''' 负责从输入图像中提取特征，直到ResNet18网络的第四层 '''
class ResNet18Layer4Lower(AbstractBaseImageLowerEncoder):
    # 初始化ResNet18模型，可以选择是否加载预训练权重
    def __init__(self, pretrained=True):
        super().__init__()
        self._model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 定义了数据如何通过网络流动，从第一层到第四层，并返回第四层的输出以及前三层的输出。
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]: # 这里的“->”符号表示函数返回的数据类型
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    # 返回每一层输出的特征维度
    def layer_shapes(self):
        return {'layer4': 512, 'layer3': 256, 'layer2': 128, 'layer1': 64}

''' 这个类接收ResNet18Layer4Lower输出的第四层特征，进行进一步的处理和转换，以生成最终的特征表示 '''
class ResNet18Layer4Upper(AbstractBaseImageUpperEncoder):
    # 初始化包括平均池化层和全连接层，以及接收来自ResNet18Layer4Lower的特征形状和目标特征大小
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.lower_feature_shape, self.feature_size)
        self.norm_scale = kwargs['norm_scale']

    # 定义了如何将第四层的输出通过平均池化和全连接层处理，最后进行归一化和缩放
    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(self.fc(x)) * self.norm_scale

        return x


class GAPResNet18Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(x) * self.norm_scale

        return x


class ResNet50Layer4Lower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True, stride=False):
        super().__init__()
        self._model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # self._model = od_resnet50()
        # avg pooling to global pooling
        if stride == True:
            self._model.layer4[0].downsample[0].stride = (1,1)
            self._model.layer4[0].conv2.stride = (1,1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    def layer_shapes(self):
        return {'layer4': 2048, 'layer3': 1024, 'layer2': 512, 'layer1': 256}


class ResNet50Layer4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.lower_feature_shape, self.feature_size)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(self.fc(x)) * self.norm_scale

        return x

''' ResNet50x4 '''
class ResNet50x4Lower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True, stride=False):
        super().__init__()
        self._model = resnet50(pretrained=pretrained)
        self._model.conv1 = nn.Conv2d(3, 64 * 4, kernel_size=7, stride=2, padding=3, bias=False)
        self._model.bn1 = nn.BatchNorm2d(64 * 4)
        # Multiply the number of filters by 4 for each layer
        self._model.layer1 = self._make_layer(self._model.layer1, 256 * 4)
        self._model.layer2 = self._make_layer(self._model.layer2, 512 * 4)
        self._model.layer3 = self._make_layer(self._model.layer3, 1024 * 4)
        self._model.layer4 = self._make_layer(self._model.layer4, 2048 * 4)

        if stride:
            self._model.layer4[0].downsample[0].stride = (1, 1)
            self._model.layer4[0].conv2.stride = (1, 1)

    def _make_layer(self, original_layer, out_channels):
        for block in original_layer:
            in_channels = block.conv1.in_channels * 4  # Adjust in_channels for the first conv layer
            block.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, bias=False)
            block.bn1 = nn.BatchNorm2d(out_channels // 4)
            block.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False)
            block.bn2 = nn.BatchNorm2d(out_channels // 4)
            block.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, bias=False)
            block.bn3 = nn.BatchNorm2d(out_channels)
            if block.downsample is not None:
                block.downsample[0] = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
                block.downsample[1] = nn.BatchNorm2d(out_channels)
        return original_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    def layer_shapes(self):
        return {'layer4': 2048 * 4, 'layer3': 1024 * 4, 'layer2': 512 * 4, 'layer1': 256 * 4}


class ResNet50x4Upper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.lower_feature_shape, self.feature_size)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(layer4_out)
        x = torch.flatten(x, 1)
        x = F.normalize(self.fc(x)) * self.norm_scale

        return x
