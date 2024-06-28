from language.abc import AbstractBaseVocabulary
from language.tokenizers import BasicTokenizer
from language.utils import create_read_func
from language.vocabulary import SimpleVocabulary


# 实现了一个词汇表的创建和处理过程(词汇表工厂函数)
def vocabulary_factory(config):
    vocab_path = config['vocab_path']
    vocab_threshold = config['vocab_threshold'] # vocab_threshold：用于过滤词汇的阈值

    # 使用 create_read_func 函数创建一个读取函数 read_func
    read_func = create_read_func(vocab_path)

    # 调用 SimpleVocabulary 类的静态方法 create_vocabulary_from_storage 来创建词汇表
    vocab = SimpleVocabulary.create_vocabulary_from_storage(read_func)
    # 使用 threshold_rare_words 方法过滤掉出现次数低于阈值的词汇
    vocab.threshold_rare_words(vocab_threshold)
    return vocab
