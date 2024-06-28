import torch
import torch.nn.functional as F
import numpy as np

from trainers.abc import AbstractBaseMetricLoss
from losses.batch_based_classification_loss import BatchBasedClassificationLoss

class AleatoricLoss(AbstractBaseMetricLoss):
    def __init__(self):
        super().__init__()

    def forward(self, ref_features, tar_features, augmented_features, epoch=0):
        # 基于批处理的增强特征分类损失
        se = BatchBasedClassificationLoss.cal_loss(ref_features, augmented_features)

        # aleatoric loss
        std = torch.std(augmented_features)
        inv_std = torch.exp(-std)
        mse = torch.mean(inv_std * se)
        reg = torch.mean(std)
        return 0.5 * (mse + reg)

    @classmethod
    def cal_loss(cls, augmented_features, features):
        se = BatchBasedClassificationLoss.cal_loss(augmented_features, features)
        std = torch.std(augmented_features)
        inv_std = torch.exp(-std)
        mse = torch.mean(inv_std * se)
        reg = torch.mean(std)
        return 0.5 * (mse + reg)

    @classmethod
    def code(cls):
        return 'aleatoric_loss'

class BatchBasedAleatoricLoss(AbstractBaseMetricLoss):
    def __init__(self, total_epoch=50, gamma_scale=1):
        super().__init__()
        self.total_epoch = total_epoch
        self.gamma_scale = gamma_scale

    def forward(self, ref_features, tar_features, augmented_features, epoch):
        # 基于批处理的增强特征分类损失
        se = BatchBasedClassificationLoss.cal_loss(ref_features, augmented_features) # L_info
        
        # aleatoric loss
        # std = torch.std(augmented_features) # 标准差
        # inv_std = torch.exp(-std)
        # mse = torch.mean(inv_std * se)
        # reg = torch.mean(std)
        # L_u = (mse + reg) / 2

        # paper aleatoric loss
        std = torch.std(augmented_features)
        std_sq = std ** 2
        se = BatchBasedClassificationLoss.cal_loss(ref_features, augmented_features)
        mse = torch.mean(se / (2 * std_sq))
        reg = torch.mean(0.5 * torch.log(std_sq))
        L_u = mse + reg

        # 基于批处理的参考特征分类损失
        L_info = BatchBasedClassificationLoss.cal_loss(ref_features, tar_features)

        gamma = np.exp(- self.gamma_scale * epoch / self.total_epoch)
        return gamma * L_u + (1 - gamma) * L_info

    @classmethod
    def code(cls):
        return 'batch_based_aleatoric_loss'
