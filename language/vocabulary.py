from collections import defaultdict
from typing import List

import abc
from tqdm import tqdm

from language.abc import _UNK_TOKEN, _BOS_TOKEN, _EOS_TOKEN, _PAD_TOKEN, _DEFAULT_TOKEN2ID, AbstractBaseVocabulary

""" 
这是一个抽象基类，用于定义文本的分词（tokenize）和还原（detokenize）的基本框架 
"""
class AbstractBaseTokenizer(abc.ABC):
    # 输入文本 text，返回一个字符串列表
    def tokenize(self, text: str) -> List[str]:
        # 分词过程首先添加一个开始标记 _BOS_TOKEN，然后调用 _tokenize 方法进行实际的分词处理，最后添加一个结束标记 _EOS_TOKEN
        return [_BOS_TOKEN] + self._tokenize(text) + [_EOS_TOKEN]

    # 输入一个字符串列表 tokens，返回一个字符串
    def detokenize(self, tokens: List[str]) -> str:
        # 还原过程首先找到开始和结束标记的索引，然后只取这两个标记之间的部分，并过滤掉填充标记 _PAD_TOKEN，最后调用 _detokenize 方法将这些标记还原为文本。
        start_idx = tokens.index(_BOS_TOKEN)
        end_idx = tokens.index(_EOS_TOKEN)
        tokens = tokens[start_idx + 1: end_idx]
        tokens = list(filter(_PAD_TOKEN.__ne__, tokens))
        return self._detokenize(tokens)

    @abc.abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def _detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError

# 这个类继承自 AbstractBaseVocabulary，用于管理词汇表和提供文本与ID之间的转换
class SimpleVocabulary(AbstractBaseVocabulary):
    def __init__(self, tokenizer: AbstractBaseTokenizer):
        # 初始化词汇表，使用一个分词器 tokenizer
        self.tokenizer = tokenizer
        # _token2id 和 _id2token 分别存储从标记到ID和从ID到标记的映射
        self._token2id = _DEFAULT_TOKEN2ID
        self._id2token = {i: token for token, i in _DEFAULT_TOKEN2ID.items()}
        #  用于记录每个标记的出现次数
        self._token_count = defaultdict(int)
        self._token_count[_UNK_TOKEN] = int(9e9)
        self._token_count[_PAD_TOKEN] = int(9e9)
        self._token_count[_BOS_TOKEN] = int(9e9)
        self._token_count[_EOS_TOKEN] = int(9e9)

    # 将输入的文本 text 添加到词汇表中
    def add_text_to_vocab(self, text):
        # 文本首先被分词，然后更新 _token2id、_id2token 和 _token_count
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            if token not in self._token2id:
                idx = len(self._token2id)
                self._token2id[token] = idx
                self._id2token[idx] = token
            self._token_count[token] += 1

    ''' 过滤罕见词汇 '''
    def threshold_rare_words(self, wordcount_threshold=5):
        # 方法遍历词汇表中的每个词汇，如果某个词汇的出现次数低于 wordcount_threshold（默认为5），则将其 ID 设置为未知词汇的 ID
        for w in self._token2id:
            if self._token_count[w] < wordcount_threshold:
                self._token2id[w] = _DEFAULT_TOKEN2ID[_UNK_TOKEN]

    # 这两个方法提供从文本到ID列表和从ID列表到文本的转换
    def convert_text_to_ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        encoded_text = [self._token2id.get(t, _DEFAULT_TOKEN2ID[_UNK_TOKEN]) for t in tokens]
        return encoded_text

    def convert_ids_to_text(self, ids):
        tokens = [self._id2token.get(token_id, _UNK_TOKEN) for token_id in ids]
        return self.tokenizer.detokenize(tokens)

    # 返回词汇表中不同词汇的数量
    def __len__(self):
        return len(self._token2id)

    # 这些方法用于从不同的数据源创建词汇表，并使用提供的 write_func 函数将词汇表存储起来
    @staticmethod
    def create_and_store_vocabulary_from_txt_files(txt_file_paths, tokenizer, write_func, txt_reader_func):
        vocab = SimpleVocabulary(tokenizer)
        for txt_path in txt_file_paths:
            texts = txt_reader_func(txt_path)
            for t in tqdm(texts):
                vocab.add_text_to_vocab(t)
        write_func(vocab)
        return vocab

    @staticmethod
    def create_and_store_vocabulary_from_list(list_data, tokenizer, write_func):
        vocab = SimpleVocabulary(tokenizer)
        for l in tqdm(list_data):
            vocab.add_text_to_vocab(l)
        write_func(vocab)
        return vocab

    @staticmethod
    def create_and_store_vocabulary_from_datasets(datasets, tokenizer, write_func, caption_pos=(2, 1)):
        vocab = SimpleVocabulary(tokenizer)
        for pos, dataset in zip(caption_pos, datasets):
            for record in tqdm(dataset):
                vocab.add_text_to_vocab(record[pos])
        write_func(vocab)
        return vocab

    @staticmethod
    def create_vocabulary_from_storage(read_func):
        return read_func()
