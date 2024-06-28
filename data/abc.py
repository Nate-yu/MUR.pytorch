import abc

from PIL import Image
from torch.utils.data import Dataset


class AbstractBaseDataset(Dataset, abc.ABC):
    """数据集基类"""

    def __init__(self, root_path, split='train', img_transform=None, text_transform=None):
        pass

    @classmethod
    @abc.abstractmethod
    def code(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def vocab_path(cls):
        raise NotImplementedError
