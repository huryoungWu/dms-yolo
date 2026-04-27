"""训练脚本 split_dataset 函数的单元测试"""

import numpy as np
import pytest

from training.train_cnn import split_dataset


class TestSplitDataset:
    """split_dataset 函数测试"""

    def test_basic_split_proportions(self):
        """测试基本的 80/10/10 划分比例"""
        N = 100
        X = np.arange(N)
        y = np.array([0] * 50 + [1] * 50)

        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        assert len(X_train) == 80
        assert len(X_val) == 10
        assert len(X_test) == 10
        assert len(X_train) + len(X_val) + len(X_test) == N

    def test_no_overlap(self):
        """测试三个子集互不重叠"""
        N = 100
        X = np.arange(N)
        y = np.array([0] * 50 + [1] * 50)

        X_train, X_val, X_test, _, _, _ = split_dataset(X, y)

        train_set = set(X_train.tolist())
        val_set = set(X_val.tolist())
        test_set = set(X_test.tolist())

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_union_equals_original(self):
        """测试三个子集的并集等于原始数据集"""
        N = 100
        X = np.arange(N)
        y = np.array([0] * 50 + [1] * 50)

        X_train, X_val, X_test, _, _, _ = split_dataset(X, y)

        union = set(X_train.tolist()) | set(X_val.tolist()) | set(X_test.tolist())
        assert union == set(range(N))

    def test_custom_ratios(self):
        """测试自定义划分比例"""
        N = 100
        X = np.arange(N)
        y = np.array([0] * 50 + [1] * 50)

        X_train, X_val, X_test, _, _, _ = split_dataset(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        assert len(X_train) == 70
        assert len(X_val) + len(X_test) == 30

    def test_invalid_ratios_sum(self):
        """测试比例之和不为 1 时抛出异常"""
        X = np.arange(100)
        y = np.array([0] * 50 + [1] * 50)

        with pytest.raises(ValueError, match="比例之和必须为 1.0"):
            split_dataset(X, y, train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    def test_too_small_dataset(self):
        """测试数据集太小时抛出异常"""
        X = np.array([1, 2])
        y = np.array([0, 1])

        with pytest.raises(ValueError, match="至少需要 3 个样本"):
            split_dataset(X, y)

    def test_labels_preserved(self):
        """测试标签与特征对应关系保持一致"""
        N = 100
        X = np.arange(N)
        y = np.array([0] * 50 + [1] * 50)

        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)

    def test_small_dataset(self):
        """测试较小数据集的划分"""
        N = 20
        X = np.arange(N)
        y = np.array([0] * 10 + [1] * 10)

        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == N
        assert len(X_train) >= 1
