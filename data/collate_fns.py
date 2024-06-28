from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

'''
这段代码定义了几个用于数据批处理的类，主要用于机器学习模型的数据预处理。
这些类的目的是将数据批次（batch）中的多个元素整合成模型可以接受的格式。
具体来说，这些类处理图像和文本数据，适用于图像和自然语言处理任务
'''

''' 这个类用于将一批数据（包括图像和文本）整合并进行填充处理，以便所有数据具有相同的长度，方便模型处理 '''
class PaddingCollateFunction(object):
    # 接收一个 padding_idx 参数，用于填充文本序列的索引
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    # 处理批数据
    def __call__(self, batch: List[tuple]):
        # 将batch数据解压缩
        reference_images, target_images, modifiers, lengths, ref_id, targ_id = zip(*batch)

        # 将图像数据堆叠成一个张量
        reference_images = torch.stack(reference_images, dim=0)
        target_images = torch.stack(target_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()

        # 将文本数据（modifiers）使用 pad_sequence 方法进行填充
        modifiers = pad_sequence(modifiers, padding_value=self.padding_idx, batch_first=True)

        # 返回处理后的图像数据、填充后的文本数据和文本长度
        return reference_images, target_images, modifiers, seq_lengths, None

'''  PaddingCollateFunction 的一个变体，用于测试数据集 '''
class PaddingCollateFunctionTest(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    # _collate_test_dataset 静态方法处理只包含图像和 ID 的简单数据批次
    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    # _collate_test_query_dataset 方法处理更复杂的数据批次，包括图像、文本和其他属性
    def _collate_test_query_dataset(self, batch):
        reference_images, ref_attrs, modifiers, target_attrs, lengths = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()
        modifiers = pad_sequence(modifiers, padding_value=self.padding_idx, batch_first=True)
        return reference_images, ref_attrs, modifiers, target_attrs, seq_lengths, None

    # __call__ 方法根据批次中元素的数量决定使用哪个处理方法
    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2:
            return self._collate_test_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)

''' 这个类扩展了 PaddingCollateFunction，使用了 BERT 模型的 tokenizer 来处理文本数据 '''
class BertPaddingCollateFunction(object):
    # 除了接收 padding_idx，还初始化了一个 BERT tokenizer
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
        # 初始化一个AutoTokenizer，使用预训练的"roberta-base"模型
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # 使用 tokenizer 对文本数据进行编码，并处理图像数据，类似于 PaddingCollateFunction，但是返回的是编码后的文本数据和注意力掩码（attention mask）
    def __call__(self, batch: List[tuple]):
        # 从batch中解压各个组件
        reference_images, target_images, modifiers, lengths, ref_id, targ_id = zip(*batch)

        # 将reference_images堆叠成一个新的tensor
        reference_images = torch.stack(reference_images, dim=0)
        # 将target_images堆叠成一个新的tensor
        target_images = torch.stack(target_images, dim=0)
        # 将lengths转换为tensor，并转换为长整型
        seq_lengths = torch.tensor(lengths).long()

        # 将modifiers列表转换为列表
        modifiers = list(modifiers)
        # 使用tokenizer批量编码modifiers，设置padding为最长，返回tensor
        token = self.tokenizer.batch_encode_plus(modifiers, padding='longest', return_tensors='pt')

        # 从token中获取attention_mask
        attn_mask = token['attention_mask']
        # 从token中获取input_ids，即编码后的modifiers
        modifiers = token['input_ids']

        # 返回处理后的各个组件
        return reference_images, target_images, modifiers, seq_lengths, attn_mask


''' 是 BertPaddingCollateFunction 的测试版本，处理逻辑与 BertPaddingCollateFunction 类似，但适用于不同类型的数据批次。 '''
class BertPaddingCollateFunctionTest(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    def _collate_test_query_dataset(self, batch):
        reference_images, ref_attrs, modifiers, target_attrs, lengths = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()

        modifiers = list(modifiers)
        token = self.tokenizer.batch_encode_plus(modifiers, padding='longest', return_tensors='pt')

        attn_mask = token['attention_mask']
        modifiers = token['input_ids']

        return reference_images, ref_attrs, modifiers, target_attrs, seq_lengths, attn_mask

    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2:
            return self._collate_test_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)
