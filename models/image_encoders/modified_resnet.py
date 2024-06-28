import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Union, Any
import torch
from torch import nn
from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder
''' 一个残差块，用于构建ResNet的一部分 '''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

''' 一个基于注意力的池化层，用于聚合特征 '''
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn((spacial_dim[0] * spacial_dim[1]) + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

''' 修改版的ResNet '''
class ModifiedResNet(nn.Module):
    """
    一个类似于torchvision的ResNet类，但包含以下更改:
        - 有3个“stem枝干”卷积，而不是1个，有一个平均池而不是最大池。
        - 执行抗混叠跨步卷积，其中avgpool被添加到跨步> 1的卷积中
        - 最终池化层是QKV注意力机制而不是平均池
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(width // 2)
        # self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(width // 2)
        # self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        spacial_dim = (input_resolution[0] // 32, input_resolution[1] // 32)
        self.attnpool = AttentionPool2d(spacial_dim, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

''' 创建一个新的 ModifiedResNetLower 类，类似于 ResNet18Layer4Lower 和 ResNet50Layer4Lower，用于提取特征 '''
class ModifiedResNetLower(AbstractBaseImageLowerEncoder):
    def __init__(self, pretrained=True, layers=[3, 4, 6, 3], output_dim=512, heads=8, input_resolution=(224, 224), width=64):
        super().__init__()
        self._model = ModifiedResNet(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.avgpool(x)

        layer1_out = self._model.layer1(x)
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)
        layer4_out = self._model.layer4(layer3_out)

        return layer4_out, (layer3_out, layer2_out, layer1_out)

    def layer_shapes(self):
        return {'layer4': 2048, 'layer3': 1024, 'layer2': 512, 'layer1': 256}

''' 创建一个新的 ModifiedResNetUpper 类，类似于 ResNet18Layer4Upper 和 ResNet50Layer4Upper，用于进一步处理特征 '''
class ModifiedResNetUpper(AbstractBaseImageUpperEncoder):
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
