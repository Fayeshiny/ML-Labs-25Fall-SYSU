import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class Node:
    """决策树节点类"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 划分特征
        self.threshold = threshold  # 划分阈值
        self.left = left            # 左子树
        self.right = right          # 右子树
        self.value = value          # 叶节点的预测值

class MyDecisionTree:
    """自定义决策树分类器"""
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def _gini(self, y):
        """计算基尼指数"""
        counter = Counter(y)
        gini = 1.0
        for count in counter.values():
            p = count / len(y)
            gini -= p ** 2
        return gini
    
    def _information_gain(self, X, y, feature, threshold):
        """计算信息增益（基尼指数减少）"""
        parent_gini = self._gini(y)
        
        # 根据阈值分割数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        
        # 计算子节点的基尼指数
        left_gini = self._gini(y[left_mask])
        right_gini = self._gini(y[right_mask])
        
        # 计算加权平均基尼指数
        n = len(y)
        n_left, n_right = len(y[left_mask]), len(y[right_mask])
        child_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        
        # 信息增益 = 父节点基尼指数 - 子节点加权基尼指数
        return parent_gini - child_gini
    
    def _best_split(self, X, y):
        """寻找最佳分割特征和阈值"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            # 获取该特征的所有唯一值作为候选阈值
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # 寻找最佳分割
        feature, threshold, gain = self._best_split(X, y)
        
        # 如果信息增益为0，则创建叶节点
        if gain == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # 递归构建左右子树
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=feature, threshold=threshold, 
                   left=left_subtree, right=right_subtree)
    
    def _most_common_label(self, y):
        """返回出现次数最多的标签"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def fit(self, X, y):
        """训练决策树"""
        self.root = self._build_tree(X, y)
    
    def _traverse_tree(self, x, node):
        """遍历决策树进行预测"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        """预测"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

# 数据预处理和测试
def preprocess_adult_data():
    """加载并预处理adult数据集"""
    # 加载数据
    data = fetch_openml(name='adult', version=2, as_frame=True)
    X = data.data
    y = data.target
    
    # 处理缺失值
    X = X.fillna('Unknown')
    
    # 对分类特征进行编码
    label_encoders = {}
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # 对标签进行编码
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    
    return X, y, label_encoders, label_encoder_y

# 测试自定义决策树
def test_my_decision_tree():
    """测试自定义决策树"""
    print("加载和预处理数据...")
    X, y, _, _ = preprocess_adult_data()
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 训练自定义决策树
    print("训练自定义决策树...")
    my_tree = MyDecisionTree(max_depth=8, min_samples_split=20)
    my_tree.fit(X_train.values, y_train)
    
    # 预测
    y_pred = my_tree.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"自定义决策树准确率: {accuracy:.4f}")
    
    return accuracy

# 运行测试
if __name__ == "__main__":
    accuracy = test_my_decision_tree()