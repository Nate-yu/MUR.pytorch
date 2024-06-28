import random

import torch
import numpy as np
import wandb

from utils.metrics import AverageMeterSet

''' 用于计算和验证机器学习模型中的相似性矩阵和召回率 '''
class ValidationMetricsCalculator:
    # 接收查询特征、测试特征、属性匹配矩阵等作为输入
    def __init__(self, original_query_features: torch.tensor, composed_query_features: torch.tensor,
                 test_features: torch.tensor, attribute_matching_matrix: np.array,
                 ref_attribute_matching_matrix: np.array, top_k: tuple):
        # 初始化各种内部变量，如相似性矩阵、最高分数、最相似的索引等
        self.original_query_features = original_query_features
        self.composed_query_features = composed_query_features
        self.test_features = test_features
        self.top_k = top_k
        self.attribute_matching_matrix = attribute_matching_matrix
        self.ref_attribute_matching_matrix = ref_attribute_matching_matrix
        self.num_query_features = composed_query_features.size(0)
        self.num_test_features = test_features.size(0)
        self.similarity_matrix = torch.zeros(self.num_query_features, self.num_test_features)
        self.top_scores = torch.zeros(self.num_query_features, max(top_k))
        self.most_similar_idx = torch.zeros(self.num_query_features, max(top_k))
        self.recall_results = {}
        self.recall_positive_queries_idxs = {k: [] for k in top_k}
        self.similarity_matrix_calculated = False
        self.top_scores_calculated = False

    def __call__(self):
        # 计算相似性矩阵
        self._calculate_similarity_matrix()
        # Filter query_feat == target_feat
        assert self.similarity_matrix.shape == self.ref_attribute_matching_matrix.shape
        # 根据参考属性匹配矩阵调整相似性矩阵，将匹配的查询特征和目标特征的相似度设置为最小值
        self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min()
        # 计算并返回各个 top_k 值的召回率
        return self._calculate_recall_at_k()

    def _calculate_similarity_matrix(self) -> torch.tensor:
        """
        query_features = torch.tensor. Size = (N_test_query, Embed_size)
        test_features = torch.tensor. Size = (N_test_dataset, Embed_size)
        output = torch.tensor, similarity matrix. Size = (N_test_query, N_test_dataset)
        """
        # 如果相似性矩阵尚未计算，使用查询特征和测试特征的矩阵乘法来计算它
        if not self.similarity_matrix_calculated:
            self.similarity_matrix = self.composed_query_features.mm(self.test_features.t())
            self.similarity_matrix_calculated = True

    def _calculate_recall_at_k(self):
        # 计算每个查询的前 k 个最相似的测试特征
        average_meter_set = AverageMeterSet()
        self.top_scores, self.most_similar_idx = self.similarity_matrix.topk(max(self.top_k))
        self.top_scores_calculated = True
        # 根据属性匹配矩阵确定每个查询的匹配情况
        topk_attribute_matching = np.take_along_axis(self.attribute_matching_matrix, self.most_similar_idx.numpy(),
                                                     axis=1)

        # 更新并计算召回率的平均值
        for k in self.top_k:
            query_matched_vector = topk_attribute_matching[:, :k].sum(axis=1).astype(bool)
            self.recall_positive_queries_idxs[k] = list(np.where(query_matched_vector > 0)[0])
            num_correct = query_matched_vector.sum()
            num_samples = len(query_matched_vector)
            average_meter_set.update('recall_@{}'.format(k), num_correct, n=num_samples)
        recall_results = average_meter_set.averages()
        return recall_results

    # 从属性列表中根据给定的索引提取属性
    @staticmethod
    def _multiple_index_from_attribute_list(attribute_list, indices):
        attributes = []
        for idx in indices:
            attributes.append(attribute_list[idx.item()])
        return attributes
