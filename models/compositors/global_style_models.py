from torch import nn

from models.utils import calculate_mean_std, EqualLinear
from trainers.abc import AbstractGlobalStyleTransformer

''' 用于对文本特征进行全局样式转换 '''
class GlobalStyleTransformer2(AbstractGlobalStyleTransformer):
    def __init__(self, feature_size, text_feature_size, *args, **kwargs):
        super().__init__()
        # 两个线性变换层，分别用于计算全局样式参数和门控参数
        self.global_transform = EqualLinear(text_feature_size, feature_size * 2)
        self.gate = EqualLinear(text_feature_size, feature_size * 2)
        # 激活函数用于计算门控参数的值
        self.sigmoid = nn.Sigmoid()
        # 初始化样式权重
        self.init_style_weights(feature_size)

    def forward(self, normed_x, t, *args, **kwargs):
        # 从输入参数 kwargs['x'] 中提取图像特征图，计算输入特征的均值和标准差
        x_mu, x_std = calculate_mean_std(kwargs['x'])
        # 对文本特征 t 进行线性变换和 sigmoid 激活，得到 gate，并将其分割为 std_gate 和 mu_gate
        gate = self.sigmoid(self.gate(t)).unsqueeze(-1).unsqueeze(-1)
        std_gate, mu_gate = gate.chunk(2, 1)

        # 对文本特征 t 进行全局转换，得到 gamma 和 beta
        global_style = self.global_transform(t).unsqueeze(2).unsqueeze(3)
        gamma, beta = global_style.chunk(2, 1)

        # 将 std_gate 和 x_std 相乘，将 gamma 加上结果；将 mu_gate 和 x_mu 相加，将 beta 加上结果
        gamma = std_gate * x_std + gamma
        beta = mu_gate * x_mu + beta
        # 最后输出 gamma * normed_x + beta，即融合后的特征
        out = gamma * normed_x + beta
        return out

    def init_style_weights(self, feature_size):
        self.global_transform.linear.bias.data[:feature_size] = 1
        self.global_transform.linear.bias.data[feature_size:] = 0

    @classmethod
    def code(cls) -> str:
        return 'global2'
