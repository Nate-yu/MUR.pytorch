import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
import torch.nn.functional as F
from models.compositors.clip_model import build_CLIP_from_openai_pretrained


class CLIPTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_name = 'ViT-B/32'  # 使用ViT-B/32的CLIP模型
        self.config = config
        self.clip_model, _ = build_CLIP_from_openai_pretrained(
            name=self.model_name,
            image_size=(224, 224),
            stride_size=32
        )
        self.text_projection = self.clip_model.text_projection
        self.feature_size = self.text_projection.shape[1]
        self.context_length = self.clip_model.context_length  # 添加 context_length 属性
        self.padding_idx = self.clip_model.token_embedding.padding_idx  # 获取 padding_idx

    def forward(self, x, attn_mask=None, len_modifiers=None):
        # 确保输入张量的大小与位置嵌入的大小一致
        x = F.pad(x, (0, self.context_length - x.size(1)), value=self.padding_idx)
        # 如果存在 attn_mask 和 len_modifiers，则忽略它们，只处理 x
        text_features = self.clip_model.encode_text(x)
        text_features = text_features[:, 0, :]  # 只保留第一个时间步的特征
        return text_features

    @classmethod
    def code(cls) -> str:
        return 'clip'