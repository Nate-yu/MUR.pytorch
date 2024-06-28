import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

''' 封装RoBERTa模型，可以用于特征提取 '''
class RobertaEncoder(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        model_name = 'roberta-base'
        self.feature_size = feature_size
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # roberta的分词器
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ''' 
        x: 模型输入
        attn_mask: 注意力掩码
    '''
    def forward(self, x, attn_mask):
        # 根据模型的训练状态（self.training），选择是训练模式还是评估模式
        # 在训练模式下，模型会进行梯度计算；在评估模式下，模型不进行梯度计算
        # 返回的outputs是模型的最后隐藏状态的第一个时间步的输出
        if self.training == True:
            self.model.train()
            outputs = self.model(x, attn_mask).last_hidden_state[:, 0]
        else:    
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x, attn_mask).last_hidden_state[:, 0]
        
        return outputs
    
    @classmethod
    def code(cls) -> str:
        return 'roberta'

class BertFc(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hidden_size = 768
        # 包含一个批量归一化层和一个线性层的序列模型
        self.model = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, feature_size),
        )

    # 输入x（模型输入）
    # 输出通过全连接层处理后的结果
    def forward(self, x):
        outputs = self.model(x)
        return outputs

    @classmethod
    def code(cls) -> str:
        return 'bertfc'