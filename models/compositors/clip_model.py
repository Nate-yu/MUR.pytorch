from collections import OrderedDict
import logging
import math
import os
from typing import List, Tuple, Union
import hashlib
import urllib
from tqdm import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.image_encoders.modified_resnet import ModifiedResNet
from models.image_encoders.vit import VisionTransformer,Transformer,LayerNorm


logger = logging.getLogger("IRRA.model")

# 包含了不同CLIP模型的URL，用于下载预训练的模型
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

'''返回可用的模型名称列表'''
def available_models() -> List[str]:
    return list(_MODELS.keys())

''' 负责从指定URL下载模型，并验证文件的SHA256校验和 '''
def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


class CLIP(nn.Module):
    def __init__(self,
                 # 图像和文本特征向量的嵌入维度
                 embed_dim: int,
                 # vision 输入图像的分辨率，可以是一个整数或一个整数元组（高度和宽度）
                 image_resolution: Union[int, Tuple[int, int]],
                 # 表示视觉模型的层数，可以是一个整数或一个四元组（用于 ResNet）
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 # 视觉模型的宽度，即每个层的通道数
                 vision_width: int,
                 # 视觉模型中图像块的大小
                 vision_patch_size: int,
                 # 图像块的步幅
                 stride_size: int,
                 # text 文本模型的上下文长度
                 context_length: int,
                 # 文本模型的词汇表大小
                 vocab_size: int,
                 # Transformer 模型的隐藏层维度
                 transformer_width: int,
                 # Transformer 模型的多头注意力头数
                 transformer_heads: int,
                 # Transformer 模型的层数
                 transformer_layers: int
                 ):
        # 初始化 nn.Module 并设置 context_length
        super().__init__()
        self.context_length = context_length

        # 视觉模型的初始化
        # 根据 vision_layers 的类型，初始化视觉模型。可以是 ModifiedResNet 或 VisionTransformer
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        # 文本模型的初始化
        # 初始化文本模型，包括 Transformer、词嵌入、位置嵌入、LayerNorm 和投影矩阵
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # 参数初始化：调用 initialize_parameters 方法对模型参数进行初始化
        self.initialize_parameters()

    ''' 参数初始化方法 '''
    def initialize_parameters(self):
        """ 对模型的各个参数进行正态初始化，包括词嵌入、位置嵌入、视觉模型和 Transformer 的参数 """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    ''' 构建注意力掩码 '''
    def build_attention_mask(self):
        """ 构建用于 Transformer 的注意力掩码，以实现自回归模型 """
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    ''' 返回模型的权重数据类型 '''
    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    ''' 图像编码：对输入图像进行编码，生成图像特征 '''
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        return image_features, text_features
    
    ''' 用于加载预训练的参数到模型 '''
    def load_param(self, state_dict):
        # 将pretrained_dict里不属于model_dict的键剔除掉
        """ 首先过滤掉那些不属于模型的参数，然后根据需要调整某些参数的形状，最后将参数复制到模型的状态字典中 """
        # 过滤无关参数
        param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}

        # 处理嵌套字典：如果参数字典包含嵌套的字典（例如包含键 'model' 或 'state_dict'），则进入相应的嵌套字典
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        # 调整特定参数的形状：
        # 对于 visual.positional_embedding 和 positional_embedding，如果它们的形状与模型中对应的嵌入矩阵形状不匹配，
        # 则调用 resize_pos_embed 或 resize_text_pos_embed 函数调整形状。
        for k, v in param_dict.items():
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_text_pos_embed(v, self.context_length)
            # 复制参数到模型：尝试将参数复制到模型中。如果在复制过程中发生错误（例如形状不匹配），则捕获异常并打印错误信息
            try:
                self.state_dict()[k].copy_(v)
            except:
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))

''' 
    该函数调整位置嵌入矩阵的大小，以适应不同的输入图像分辨率
    它首先将位置嵌入矩阵的形状调整为合适的形状，然后使用双线性插值进行重采样，最后将其转换回原始的嵌入矩阵形式 
'''
def resize_pos_embed(posemb, posemb_new, hight, width):
    # 从state_dict加载时重新缩放位置嵌入的网格
    # 添加批次维度：将输入的 posemb 和 posemb_new 添加一个批次维度，以便后续的处理
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)

    # 分离分类标记嵌入和位置嵌入网格：将 posemb 分成两个部分：分类标记嵌入（第一个嵌入）和位置嵌入网格（其余嵌入）
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    # 计算原始位置嵌入网格的大小gs_old，并打印调整前后的尺寸信息
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))

    # 调整位置嵌入网格的形状并进行重采样：
    # 将 posemb_grid 重新调整形状为 [1, channels, height, width]，
    # 然后使用双线性插值将其调整为新的高度和宽度，最后将其恢复为 [1, new_height * new_width, channels] 的形状
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)

    # 重新组合分类标记嵌入和调整后的位置嵌入网格：将分类标记嵌入和调整后的位置嵌入网格拼接在一起，最后移除批次维度，返回结果
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def convert_weights(model: nn.Module):
    """ 将模型中的某些参数转换为 16 位浮点数（fp16），以减少模型的内存占用和加快计算速度 """

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "mcq_proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_CLIP_from_openai_pretrained(name: str, image_size: Union[int, Tuple[int, int]], stride_size: int, jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        由' clip.available_models() '列出的模型名，或者包含state_dict的模型checkpoint的路径
    
    image_size: Union[int, Tuple[int, int]]
        输入图像大小，224x224

    jit : bool
        是加载优化的JIT模型还是更容易被破解的非JIT模型(默认)。

    download_root: str
        模型文件下载路径；默认情况下，它使用“~/.cache/clip”

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # 加载JIT存档
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        # 加载已保存的状态字典
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    state_dict = state_dict or model.state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model_cfg = {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers, 
        'vision_width': vision_width, 
        'vision_patch_size': vision_patch_size,
        'context_length': context_length, 
        'vocab_size': vocab_size, 
        'transformer_width': transformer_width, 
        'transformer_heads': transformer_heads, 
        'transformer_layers': transformer_layers
    }


    # modify image resolution to adapt Re-ID task
    model_cfg['image_resolution'] = image_size
    model_cfg['stride_size'] = stride_size
    logger.info(f"Load pretrained {name} CLIP model with model config: {model_cfg}")
    model = CLIP(**model_cfg)

    model.load_param(state_dict)
    return model, model_cfg


