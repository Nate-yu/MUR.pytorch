from collections import OrderedDict
from typing import Any, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder

''' torch.nn.LayerNorm的一个子类：Torch的层归一化来处理半精度浮点数（fp16） '''
class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        # 保存输入张量x的数据类型
        orig_type = x.dtype
        # 将输入张量x的数据类型转换为torch.float32（单精度浮点数），这是为了在执行层归一化时保持数值的稳定性
        # 调用父类torch.nn.LayerNorm的forward方法处理转换后的张量
        ret = super().forward(x.type(torch.float32))
        # 将结果张量的数据类型转换回原始输入张量x的数据类型（orig_type），然后返回这个结果张量
        return ret.type(orig_type)

''' 激活函数 '''
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

''' 残差注意力模块 '''
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

''' Transformer模块，用于ViT '''
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

''' ViT '''
class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int, width: int, layers: int,
                 heads: int, output_dim: int):
        """构造ViT模型

        Parameters
        ----------
        input_resolution : Tuple[int, int]
            输入图像的分辨率（高度, 宽度）

        patch_size: int
            图像块的大小

        stride_size: int
            图像块的步幅

        width: int
            Transformer的隐藏维度

        layers: int
            Transformer中的残差块数量

        heads: int
            多头注意力机制中的头数量

        output_dim: int
            最终输出的维度
        """
        super().__init__()
        self.input_resolution = input_resolution  # (384, 128)/(224,224)

        # 计算图像块的数量：num_x 和 num_y 分别表示在宽度和高度方向上可以提取的图像块数量
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        num_patches = self.num_x * self.num_y

        self.output_dim = output_dim
        # 初始化卷积层：conv1 是一个卷积层，用于将输入图像分割成多个图像块，并将每个块映射到 width 维的特征向量
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size,
                               bias=False)

        # 初始化嵌入和位置编码：class_embedding 是一个可训练的参数，用于表示类别标识（class token）；positional_embedding 是位置编码，用于表示每个图像块的位置
        scale = width ** -0.5  # 1/sqrt(768)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        # 初始化层归一化层与Transformer模块
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        # 初始化投影矩阵
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # 卷积操作：输入图像通过卷积层，生成特征图
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        # 特征图重塑：将特征图重塑为二维的形式，并调整维度顺序
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # 添加类别标识：将类别标识（class token）添加到特征图前面
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)
        # 预归一化
        x = self.ln_pre(x)
        # 变换器处理
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 后归一化
        x = self.ln_post(x)

        # 投影到输出维度
        if self.proj is not None:
            x = x @ self.proj

        return x

''' 创建VisionTransformerLower类 '''
class VisionTransformerLower(AbstractBaseImageLowerEncoder):
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self._model = VisionTransformer(input_resolution, patch_size, stride_size, width, layers, heads, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        x = self._model(x)
        # 将输出拆分为layer4_out和其他层输出（这里假设x包含这些信息，可能需要进一步处理）
        layer4_out = x[:, 0, :]  # 假设第一个token是class token
        other_layers_out = x[:, 1:, :]  # 假设其余tokens是图像块
        return layer4_out, other_layers_out

    def layer_shapes(self):
        return {'layer4': self._model.output_dim, 'other_layers': self._model.output_dim}

''' 创建VisionTransformerUpper类 '''
class VisionTransformerUpper(AbstractBaseImageUpperEncoder):
    def __init__(self, lower_feature_shape, feature_size, pretrained=True, *args, **kwargs):
        super().__init__(lower_feature_shape, feature_size, pretrained=pretrained, *args, **kwargs)
        self.fc = nn.Linear(lower_feature_shape, feature_size)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, layer4_out: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.fc(layer4_out)) * self.norm_scale
        return x
