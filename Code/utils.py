import re
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import backend as K

# 加载配置文件
def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# F1-score 评估指标
def f1_m(y_true, y_pred):
    # 将 y_true 转换为 float32 类型
    y_true = tf.cast(y_true, tf.float32)
    # 创建掩码，忽略填充部分
    mask = K.cast(K.not_equal(y_true, -1), 'float32')
    y_true = y_true * mask
    y_pred = y_pred * mask
    y_pred = K.cast(K.greater_equal(y_pred, 0.5), 'float32')
    # 计算 TP、FP、FN
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    # 计算 Precision 和 Recall
    epsilon = K.epsilon()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    # 计算 F1-score
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return f1

# 准确率（Accuracy）评估指标
def accuracy_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mask = K.cast(K.not_equal(y_true, -1), 'float32')
    y_true = y_true * mask
    y_pred = y_pred * mask
    y_pred = K.cast(K.greater_equal(y_pred, 0.5), 'float32')
    correct = K.equal(y_true, y_pred)
    correct = K.cast(correct, 'float32') * mask
    accuracy = K.sum(correct) / (K.sum(mask) + K.epsilon())
    return accuracy

# 正样本位点的准确率
def positive_accuracy_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mask = K.cast(K.not_equal(y_true, -1), 'float32')
    y_true = y_true * mask

    y_pred_binary = K.cast(K.greater_equal(y_pred, 0.5), 'float32')
    y_pred_binary = y_pred_binary * mask

    positive_mask = K.cast(K.equal(y_true, 1.0), 'float32')
    correct_predictions = K.sum(positive_mask * y_pred_binary)
    total_positive = K.sum(positive_mask)
    accuracy = correct_predictions / (total_positive + K.epsilon())
    return accuracy

def negative_accuracy_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mask = K.cast(K.not_equal(y_true, -1), 'float32')
    y_true = y_true * mask

    y_pred_binary = K.cast(K.less(y_pred, 0.5), 'float32')
    y_pred_binary = y_pred_binary * mask

    negative_mask = K.cast(K.equal(y_true, 0.0), 'float32')
    correct_predictions = K.sum(negative_mask * y_pred_binary)
    total_negative = K.sum(negative_mask)
    accuracy = correct_predictions / (total_negative + K.epsilon())
    return accuracy

# 自定义的精确率（Precision）评估指标
def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mask = K.cast(K.not_equal(y_true, -1), 'float32')
    y_true = y_true * mask
    y_pred = y_pred * mask
    y_pred = K.cast(K.greater_equal(y_pred, 0.5), 'float32')
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    epsilon = K.epsilon()
    precision = tp / (tp + fp + epsilon)
    return precision

# 召回率（Recall）评估指标
def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mask = K.cast(K.not_equal(y_true, -1), 'float32')
    y_true = y_true * mask
    y_pred = y_pred * mask
    y_pred = K.cast(K.greater_equal(y_pred, 0.5), 'float32')
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    epsilon = K.epsilon()
    recall = tp / (tp + fn + epsilon)
    return recall

# 加权的二元交叉熵损失函数
def weighted_binary_crossentropy(y_true, y_pred, class_weights):
    y_true = tf.cast(y_true, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    bce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

    weights = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
    weighted_bce = bce * weights * mask
    loss = K.sum(weighted_bce) / (K.sum(mask) + epsilon)
    return loss

# 文件名清理，避免特殊符号导致临时文件生成失败
def sanitize_filename(sequence_id):
    return re.sub(r'[^\w\-_\. ]', '_', sequence_id)

# 构建邻接矩阵
def create_adjacency_matrix(seq_len):
    adjacency_matrix = np.eye(seq_len)
    for i in range(seq_len - 1):
        adjacency_matrix[i, i + 1] = 1.0
        adjacency_matrix[i + 1, i] = 1.0
    return adjacency_matrix

# 定义氨基酸的理化属性字典
amino_acid_properties = {
    'A': [1.8, 89.1], 'C': [2.5, 121.2], 'D': [-3.5, 133.1], 'E': [-3.5, 147.1],
    'F': [2.8, 165.2], 'G': [-0.4, 75.1], 'H': [-3.2, 155.2], 'I': [4.5, 131.2],
    'K': [-3.9, 146.2], 'L': [3.8, 131.2], 'M': [1.9, 149.2], 'N': [-3.5, 132.1],
    'P': [-1.6, 115.1], 'Q': [-3.5, 146.2], 'R': [-4.5, 174.2], 'S': [-0.8, 105.1],
    'T': [-0.7, 119.1], 'V': [4.2, 117.1], 'W': [-0.9, 204.2], 'Y': [-1.3, 181.2],
    'X': [0.0, 0.0]
}