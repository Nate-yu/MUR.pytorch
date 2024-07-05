import torch
import torch.nn as nn

from models.attention_modules.self_attention import AttentionModule

''' 一个去纠缠的 transformer 模型，用于对图像特征进行处理，并通过 self-attention 机制融合图像和文本特征 '''
class DisentangledTransformer(nn.Module):
    def __init__(self, feature_size, text_feature_size, num_heads, global_styler=None, *args, **kwargs):
        super().__init__()
        self.n_heads = num_heads
        self.c_per_head = feature_size // num_heads
        assert feature_size == self.n_heads * self.c_per_head

        # 两个自注意力模块，对图像特征进行处理
        self.att_module = AttentionModule(feature_size, text_feature_size, num_heads, *args, **kwargs)
        self.att_module2 = AttentionModule(feature_size, text_feature_size, num_heads, *args, **kwargs)
        # 全局样式转换器，用于融合图像和文本特征
        self.global_styler = global_styler

        # 两个可训练参数，用于调节注意力输出的权重
        self.weights = nn.Parameter(torch.tensor([1., 1.]))
        # 用于对输入进行实例归一化
        self.instance_norm = nn.InstanceNorm2d(feature_size)

    def forward(self, x, t, *args, **kwargs):
        # 对输入 x 进行实例归一化，得到 normed_x
        normed_x = self.instance_norm(x)
        # 使用第一个注意力模块 att_module 对 normed_x 和文本特征 t 进行处理，得到注意力输出 att_out 和注意力图 att_map
        att_out, att_map = self.att_module(normed_x, t, return_map=True)
        # 将 normed_x 和 att_out 加权相加，得到中间输出 out
        out = normed_x + self.weights[0] * att_out

        # 使用第二个注意力模块 att_module2 对 out 和文本特征 t 进行处理，得到第二次注意力输出 att_out2 和第二次注意力图 att_map2
        att_out2, att_map2 = self.att_module2(out, t, return_map=True)
        # 将第一次输出 out 和 att_out2 加权相加，得到最终的融合特征 out
        out = out + self.weights[1] * att_out2

        # 使用全局样式转换器 global_styler 对融合特征 out 和文本特征 t 进行进一步处理，得到最终输出
        out = self.global_styler(out, t, x=x)

        return out, att_map
