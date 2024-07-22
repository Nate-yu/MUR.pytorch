from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from models.attention_modules.cot_attention import CoTAttention


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
        if stride:
            self._model.layer4[0].downsample[0].stride = (1, 1)
            self._model.layer4[0].conv2.stride = (1, 1)
        self.cot_attention = CoTAttention(2048)  # 添加 CoTAttention 模块，输入通道数为 2048

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        layer4_out = self.cot_attention(layer4_out)  # 通过 CoTAttention 模块

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
