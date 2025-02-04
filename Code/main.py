import tensorflow as tf
from sklearn.utils import class_weight
import os
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, accuracy_score
)
import logging
from utils import load_config
from feature_extraction import extract_and_save_features, init_model
from models import build_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import sanitize_filename, create_adjacency_matrix
from data_preprocessing import extract_entries_with_methylation_info, generate_samples
import h5py
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from scipy.stats import ttest_rel
import time
import gc

# 随机数种子，复现实验用的
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 日志相关的
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量，存储各消融实验的结果
global_experiment_results = {}


# 一次消融要做的内容
def run_experiment(experiment_config, experiment_name=""):
    # 从config中，读取实验的一些配置
    paths = experiment_config['paths']
    model_params = experiment_config['model_params']

    # 读取消融实验配置，CNN、GAT，以及用到哪些特征等
    ablation_config = experiment_config.get('ablation', {})
    use_cnn = ablation_config.get('use_cnn', True)
    use_gat = ablation_config.get('use_gat', True)
    use_attention = ablation_config.get('use_attention', True)
    use_lstm = ablation_config.get('use_lstm', True)
    feature_list = ablation_config.get('features', ['protbert', 'pssm', 'properties', 'onehot'])

    # 创建目录保存当前消融实验的数据
    experiment_name = ablation_config.get('name', experiment_name)
    output_dir = paths['output_dir']
    this_experiment_features_dir = os.path.join(output_dir, "features", experiment_name)
    os.makedirs(this_experiment_features_dir, exist_ok=True)
    logger.info(f"本次实验 ({experiment_name}) 的特征文件将保存在: {this_experiment_features_dir}")

    # 创建result目录保存实验的结果，模型之类的
    this_experiment_results_dir = os.path.join(this_experiment_features_dir, "results")
    os.makedirs(this_experiment_results_dir, exist_ok=True)

    # 导入特征提取需要的一些文件路径
    fasta_file = paths['fasta_file']
    tsv_file = paths['tsv_file']
    # output_file = paths['output_file']
    blast_db = paths['blast_db']

    # 提取甲基化的信息
    entries_info = extract_entries_with_methylation_info(tsv_file)
    # 生成序列和标签儿，就是哪些是甲基位点哪些不是非甲基位点
    samples = generate_samples(fasta_file, entries_info)
    if samples is None:
        return
    else:
        sequences, sequence_ids, labels = samples
    logger.info(f"包含甲基化位点的蛋白质序列数量：{len(sequences)}")

    # 计算序列需要的长度，去除极端长度
    sequence_lengths = [len(seq) for seq in sequences]
    min_len = max(10, int(np.percentile(sequence_lengths, 5)))
    max_len = min(3027, int(np.percentile(sequence_lengths, 95)))
    logger.info(f"使用的 min_len 为：{min_len}")
    logger.info(f"使用的 max_len 为：{max_len}")

    # 加载预训练模型和分词器，给protbert预训练语言模型用
    tokenizer, model_bert = init_model()

    # 根据名字判断特征是否存在
    all_features_exist = True
    for seq_id, lbl in zip(sequence_ids, labels):
        sanitized_id = sanitize_filename(seq_id)
        feature_file = os.path.join(this_experiment_features_dir, f"{sanitized_id}_features.h5")
        if not os.path.exists(feature_file):
            all_features_exist = False
            break

    # 如果特征都存在
    if all_features_exist:
        logger.info(f"实验 {experiment_name}: {this_experiment_features_dir} 下所有特征文件已存在，跳过特征提取。")
    else:
        logger.info(f"实验 {experiment_name}: 检测到部分特征文件缺失，开始提取特征...")
        with tf.device('/GPU:0'):
            extract_and_save_features(
                sequence_ids, sequences, labels,
                features_dir=this_experiment_features_dir,
                blast_db=blast_db,
                num_iterations=model_params['num_iterations'],
                batch_size_protbert=model_params['batch_size_protbert'],
                min_len=min_len,
                max_len=max_len,
                tokenizer=tokenizer,
                model=model_bert,
                feature_set=feature_list
            )
        logger.info(f"实验 {experiment_name}: 特征提取完成。")

    # 收集成功提取特征的序列ID，构建数据集
    available_sequence_ids = []
    available_sequences = []
    available_labels = []
    for seq_id, seq, label in zip(sequence_ids, sequences, labels):
        sanitized_id = sanitize_filename(seq_id)
        feature_file = os.path.join(this_experiment_features_dir, f"{sanitized_id}_features.h5")
        if os.path.exists(feature_file):
            available_sequence_ids.append(seq_id)
            available_sequences.append(seq)
            available_labels.append(label)
        else:
            logger.warning(f"特征文件 {feature_file} 不存在，跳过该序列 {seq_id}。")

    logger.info(f"成功提取特征的蛋白质序列数量：{len(available_sequence_ids)}")

    # 标签转换为0/1
    available_labels = [1 if np.any(lbl) else 0 for lbl in available_labels]

    # 组装 (seq_id, label)
    data = list(zip(available_sequence_ids, available_labels))

    # 划分训练验证集、测试集
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    logger.info(f"训练验证集大小：{len(train_val_data)}，测试集大小：{len(test_data)}")

    # 5 折交叉验证
    n_splits = 5
    X_train_val = [d[0] for d in train_val_data]
    y_train_val = [d[1] for d in train_val_data]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []

    # 获取 in_channels
    in_channels = None
    for seq_id in available_sequence_ids:
        sanitized_id = sanitize_filename(seq_id)
        feature_file = os.path.join(this_experiment_features_dir, f"{sanitized_id}_features.h5")
        if os.path.exists(feature_file):
            with h5py.File(feature_file, 'r') as hf:
                feats = hf['features'][:]
                in_channels = feats.shape[1]
            break
    else:
        logger.error("没有找到任何特征文件，无法确定输入特征维度。")
        return

    # 计算特征文件总大小，如果小于内存，则一次性导入运行内存
    total_feature_size = 0
    for seq_id in available_sequence_ids:
        sanitized_id = sanitize_filename(seq_id)
        feature_file = os.path.join(this_experiment_features_dir, f"{sanitized_id}_features.h5")
        if os.path.exists(feature_file):
            total_feature_size += os.path.getsize(feature_file)
        else:
            logger.warning(f"特征文件 {feature_file} 不存在，跳过该序列 {seq_id}。")

    mem = psutil.virtual_memory()
    available_memory = mem.available
    logger.info(f"可用内存大小为：{available_memory / (1024 ** 3):.2f} GB")
    logger.info(f"本实验特征文件总大小为：{total_feature_size / (1024 ** 3):.2f} GB")

    # 和本机内存相对比，看能否全部加载进去
    if total_feature_size * 2 < available_memory:
        logger.info("当前实验特征数据可以全部加载到内存中，开始加载...")
        feature_data = {}
        for sid in available_sequence_ids:
            sanitized_id = sanitize_filename(sid)
            feature_file = os.path.join(this_experiment_features_dir, f"{sanitized_id}_features.h5")
            with h5py.File(feature_file, 'r') as hf:
                fts = hf['features'][:]
                lbls = hf['labels'][:]
            feature_data[sid] = {'features': fts, 'labels': lbls}
        logger.info("特征数据已全部加载到内存中。")
    else:
        logger.info("特征数据过大，无法全部加载到内存，将分批加载。")
        feature_data = None

    # 收集验证集预测结果，用于计算最优阈值
    all_val_y_true = []
    all_val_y_pred_probs = []

    start_time = time.time()

    # 五折交叉验证
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
        logger.info(f"正在处理第 {fold + 1} 折...")

        # 根据当前训练和验证的索引来提取数据
        train_data_split = [train_val_data[i] for i in train_index]
        val_data_split = [train_val_data[i] for i in val_index]
        # 解压数据，获取ID和标签儿
        train_sequence_ids_, train_labels_ = zip(*train_data_split)
        val_sequence_ids_, val_labels_ = zip(*val_data_split)

        if feature_data is not None:
            train_feature_data = {
                sid: feature_data[sid] for sid in train_sequence_ids_ if sid in feature_data
            }
            val_feature_data = {
                sid: feature_data[sid] for sid in val_sequence_ids_ if sid in feature_data
            }
        else:
            train_feature_data = None
            val_feature_data = None

        batch_size = model_params.get('batch_size', 32)

        # 从文件中读取特征以及标签和一些其他信息，然候对序列扩充截断，然候输出一个数据集
        def create_dataset(sequence_ids, batch_size, max_len, features_dir, in_channels, feature_data, shuffle=True):
            def generator():
                for sid in sequence_ids:
                    # 文件名儿规范和特征的路径
                    sanitized_id = sanitize_filename(sid)
                    feature_file_ = os.path.join(features_dir, f"{sanitized_id}_features.h5")

                    # 特征在内存中直接读取
                    if feature_data is not None and sid in feature_data:
                        data_ = feature_data[sid]
                        features_ = data_['features']
                        labels_ = data_['labels']
                    else:
                        if not os.path.exists(feature_file_):
                            continue
                        with h5py.File(feature_file_, 'r') as hf_:
                            features_ = hf_['features'][:]
                            labels_ = hf_['labels'][:]

                    # 对当前序列的长度阶段填充操作
                    seq_len = features_.shape[0]
                    if seq_len > max_len:
                        features_ = features_[:max_len, :]
                        labels_ = labels_[:max_len]
                        seq_len = max_len
                    else:
                        pad_len = max_len - seq_len
                        features_ = np.pad(features_, ((0, pad_len), (0, 0)), mode='constant')
                        labels_ = np.pad(labels_, (0, pad_len), mode='constant', constant_values=-1)

                    # 通过掩码标记有效的，无效为0
                    mask_ = np.where(labels_ != -1, True, False)
                    labels_ = np.where(labels_ != -1, labels_, 0)

                    # 创建GNN用的邻接矩，然候填充
                    adj_matrix_ = create_adjacency_matrix(seq_len)
                    adj_matrix_ = np.pad(
                        adj_matrix_,
                        ((0, max_len - seq_len), (0, max_len - seq_len)),
                        mode='constant'
                    )

                    # 用生成器返回样本的标签和字典
                    yield {
                        'input_layer': features_,
                        'adjacency_input': adj_matrix_,
                        'mask_input': mask_,
                        'sequence_id': sid.encode('utf-8')
                    }, labels_.reshape(-1, 1)

            # 定义输出的签名儿，形状和数据类型
            output_signature = (
                {
                    'input_layer': tf.TensorSpec(shape=(None, in_channels), dtype=tf.float32),
                    'adjacency_input': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                    'mask_input': tf.TensorSpec(shape=(None,), dtype=tf.bool),
                    'sequence_id': tf.TensorSpec(shape=(), dtype=tf.string)
                },
                tf.TensorSpec(shape=(None, 1), dtype=tf.int32)
            )
            # 形状
            padded_shapes = (
                {
                    'input_layer': [None, in_channels],
                    'adjacency_input': [None, None],
                    'mask_input': [None],
                    'sequence_id': []
                },
                [None, 1]
            )
            # 填充的默认值0
            padding_values = (
                {
                    'input_layer': tf.constant(0, dtype=tf.float32),
                    'adjacency_input': tf.constant(0, dtype=tf.float32),
                    'mask_input': tf.constant(False, dtype=tf.bool),
                    'sequence_id': tf.constant('', dtype=tf.string)
                },
                tf.constant(0, dtype=tf.int32)
            )
            # 从生成器创建数据集
            dataset_ = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            # 打乱顺序
            if shuffle:
                dataset_ = dataset_.shuffle(buffer_size=len(sequence_ids))
            dataset_ = dataset_.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
            dataset_ = dataset_.prefetch(tf.data.AUTOTUNE)
            return dataset_

        # 创建训练集
        train_dataset = create_dataset(
            train_sequence_ids_, batch_size, max_len,
            this_experiment_features_dir, in_channels, train_feature_data, shuffle=True
        )
        # 创建验证集
        val_dataset = create_dataset(
            val_sequence_ids_, batch_size, max_len,
            this_experiment_features_dir, in_channels, val_feature_data, shuffle=False
        )

        # 从训练集上计算类别权重
        all_y_train_local = []
        for _, y_batch_ in train_dataset:
            y_batch_np_ = y_batch_.numpy().flatten()
            valid_indices_ = y_batch_np_ != -1
            y_batch_flat_ = y_batch_np_[valid_indices_]
            all_y_train_local.extend(y_batch_flat_)
        all_y_train_local = np.array(all_y_train_local)
        classes_ = np.unique(all_y_train_local)
        class_weights_ = class_weight.compute_class_weight(
            class_weight='balanced', classes=classes_, y=all_y_train_local
        )
        max_weight = 5.0
        min_weight = 4.0
        class_weights_ = np.log1p(class_weights_)
        class_weights_ = np.clip(class_weights_, min_weight, max_weight)
        class_weight_dict = dict(zip(classes_, class_weights_))
        logger.info(f"调整后的类别权重：{class_weight_dict}")

        model_save_path = os.path.join(
            this_experiment_results_dir,
            f"my_trained_model_fold_{fold + 1}.h5"
        )

        # 将消融实验开关写回 model_params
        model_params['use_cnn'] = use_cnn
        model_params['use_gat'] = use_gat
        model_params['use_attention'] = use_attention
        model_params['use_lstm'] = use_lstm

        # 如果已有该折的模型，就加载；否则重新训练
        if os.path.exists(model_save_path):
            logger.info(f"发现已保存的模型，加载它以跳过模型训练。")
            model_fold = build_model(None, in_channels, None, class_weight_dict, model_params=model_params)
            model_fold.load_weights(model_save_path)
            logger.info(f"模型权重已从 '{model_save_path}' 加载")
        else:
            logger.info(f"未找到保存的模型，开始模型训练。")
            model_fold = build_model(None, in_channels, None, class_weight_dict, model_params=model_params)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            model_fold.fit(
                train_dataset,
                epochs=model_params.get('epochs', 200),
                validation_data=val_dataset,
                callbacks=[early_stopping, reduce_lr]
            )
            model_fold.save_weights(model_save_path)
            logger.info(f"模型已保存为 '{model_save_path}'")

        # 验证集上的评估
        y_true_ = []
        y_pred_probs_ = []
        y_pred_ = []
        sequence_ids_list_ = []
        positions_list_ = []

        for x_batch, y_batch_ in val_dataset:
            y_pred_batch_ = model_fold.predict(x_batch)
            y_batch_np_ = y_batch_.numpy()
            mask_ = x_batch['mask_input'].numpy()
            batch_sequence_ids_ = x_batch['sequence_id'].numpy()
            b_size_ = mask_.shape[0]
            seq_length_ = mask_.shape[1]

            for i_ in range(b_size_):
                seq_id_bytes_ = batch_sequence_ids_[i_]
                seq_id_ = seq_id_bytes_.decode('utf-8')
                mask_i_ = mask_[i_]
                positions_ = np.arange(seq_length_)[mask_i_]

                y_true_i_ = y_batch_np_[i_].flatten()[mask_i_]
                y_pred_probs_i_ = y_pred_batch_[i_].flatten()[mask_i_]
                y_pred_i_ = (y_pred_probs_i_ >= 0.5).astype(int)

                sequence_ids_list_.extend([seq_id_] * len(positions_))
                positions_list_.extend(positions_.tolist())
                y_true_.extend(y_true_i_.tolist())
                y_pred_probs_.extend(y_pred_probs_i_.tolist())
                y_pred_.extend(y_pred_i_.tolist())

        accuracy_ = accuracy_score(y_true_, y_pred_)

        # 保存预测结果
        predictions_file = os.path.join(
            this_experiment_results_dir,
            f"predictions_fold_{fold + 1}.csv"
        )
        df_predictions = pd.DataFrame({
            'Sequence_ID': sequence_ids_list_,
            'Position': positions_list_,
            'True_Label': y_true_,
            'Predicted_Probability': [f"{prob_ * 100}%" for prob_ in y_pred_probs_],
            'Predicted_Label': y_pred_
        })
        df_predictions.to_csv(predictions_file, index=False)
        logger.info(f"Fold {fold + 1} 的预测结果已保存到 {predictions_file}")

        # 计算评估指标
        precision_ = precision_score(y_true_, y_pred_, zero_division=0)
        recall_ = recall_score(y_true_, y_pred_, zero_division=0)
        f1_ = f1_score(y_true_, y_pred_, zero_division=0)
        roc_auc_ = roc_auc_score(y_true_, y_pred_probs_)
        mcc_ = matthews_corrcoef(y_true_, y_pred_)

        conf_matrix_ = confusion_matrix(y_true_, y_pred_)
        logger.info(f"Fold {fold + 1} - Accuracy:  {accuracy_ * 100:.3f}%")
        logger.info(f"Fold {fold + 1} - MCC:       {mcc_ * 100:.5f}%")
        logger.info(f"Fold {fold + 1} - 混淆矩阵：\n{conf_matrix_}")
        logger.info(f"Fold {fold + 1} - Precision: {precision_ * 100:.3f}%")
        logger.info(f"Fold {fold + 1} - Recall:    {recall_ * 100:.3f}%")
        logger.info(f"Fold {fold + 1} - F1-score:  {f1_ * 100:.3f}%")
        logger.info(f"Fold {fold + 1} - AUC-ROC:   {roc_auc_ * 100:.3f}%")

        metrics_file = os.path.join(
            this_experiment_results_dir,
            f"metrics_fold_{fold + 1}.txt"
        )
        with open(metrics_file, 'w') as f__:
            f__.write(f"Fold {fold + 1} - Accuracy:  {accuracy_ * 100:.3f}%\n")
            f__.write(f"Fold {fold + 1} - Precision: {precision_ * 100:.3f}%\n")
            f__.write(f"Fold {fold + 1} - Recall:    {recall_ * 100:.3f}%\n")
            f__.write(f"Fold {fold + 1} - F1-score:  {f1_ * 100:.3f}%\n")
            f__.write(f"Fold {fold + 1} - AUC-ROC:   {roc_auc_ * 100:.3f}%\n")
            f__.write(f"Fold {fold + 1} - MCC:       {mcc_ * 100:.5f}%\n")
            f__.write(f"Fold {fold + 1} - Confusion Matrix:\n{conf_matrix_}\n")

        # 画ROC图
        fpr_, tpr_, thresholds_ = roc_curve(y_true_, y_pred_probs_)
        roc_auc_score_value = auc(fpr_, tpr_)
        plt.figure()
        plt.plot(fpr_, tpr_, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score_value:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Fold {fold + 1} Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(
            this_experiment_results_dir,
            f"roc_curve_fold_{fold + 1}.png"
        ))
        plt.close()

        # 画Precision-Recall图
        precision_vals_, recall_vals_, thresholds_pr_ = precision_recall_curve(y_true_, y_pred_probs_)
        average_precision_ = average_precision_score(y_true_, y_pred_probs_)
        plt.figure()
        plt.step(recall_vals_, precision_vals_, where='post', label=f'Average precision = {average_precision_:0.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Fold {fold + 1} Precision-Recall curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(
            this_experiment_results_dir,
            f"pr_curve_fold_{fold + 1}.png"
        ))
        plt.close()

        # 画混淆矩阵
        disp_ = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_)
        disp_.plot()
        plt.title(f'Fold {fold + 1} Confusion Matrix')
        plt.savefig(os.path.join(
            this_experiment_results_dir,
            f"confusion_matrix_fold_{fold + 1}.png"
        ))
        plt.close()

        # 保存验证集标签与预测概率，用来计算最右的阈值
        all_val_y_true.extend(y_true_)
        all_val_y_pred_probs.extend(y_pred_probs_)

        # 存储当前折的指标到all_metrics文件中
        all_metrics.append({
            'fold': fold + 1,
            'accuracy': accuracy_,
            'precision': precision_,
            'recall': recall_,
            'f1_score': f1_,
            'roc_auc': roc_auc_,
            'mcc': mcc_
        })

    # 一次交叉验证技术了，输出指标的平均值
    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
    avg_roc_auc = np.mean([m['roc_auc'] for m in all_metrics])
    avg_mcc = np.mean([m['mcc'] for m in all_metrics])

    logger.info(f"平均 Accuracy:  {avg_accuracy * 100:.3f}%")
    logger.info(f"平均 MCC:       {avg_mcc * 100:.5f}%")
    logger.info(f"平均 Precision: {avg_precision * 100:.3f}%")
    logger.info(f"平均 Recall:    {avg_recall * 100:.3f}%")
    logger.info(f"平均 F1-score:  {avg_f1 * 100:.3f}%")
    logger.info(f"平均 AUC-ROC:   {avg_roc_auc * 100:.3f}%")

    overall_metrics_file = os.path.join(this_experiment_results_dir, "overall_metrics.txt")
    with open(overall_metrics_file, 'w') as f__:
        f__.write(f"平均 Accuracy:  {avg_accuracy * 100:.3f}%\n")
        f__.write(f"平均 Precision: {avg_precision * 100:.3f}%\n")
        f__.write(f"平均 Recall:    {avg_recall * 100:.3f}%\n")
        f__.write(f"平均 F1-score:  {avg_f1 * 100:.3f}%\n")
        f__.write(f"平均 AUC-ROC:   {avg_roc_auc * 100:.3f}%\n")
        f__.write(f"平均 MCC:       {avg_mcc * 100:.5f}%\n")

    # 根据条件，找最好的阈值
    logger.info("在验证集上计算最佳阈值...")
    all_val_y_true_np = np.array(all_val_y_true)
    all_val_y_pred_probs_np = np.array(all_val_y_pred_probs)

    precision_vals_, recall_vals_, thresholds_ = precision_recall_curve(
        all_val_y_true_np, all_val_y_pred_probs_np
    )
    f1_scores_ = 2 * (precision_vals_ * recall_vals_) / (precision_vals_ + recall_vals_ + 1e-6)

    target_precision = 0.90
    target_recall = 0.90
    optimal_threshold = None
    for p_, r_, t_ in zip(precision_vals_, recall_vals_, thresholds_):
        if p_ >= target_precision and r_ >= target_recall:
            optimal_threshold = t_
            logger.info(f"找到 Precision 和 Recall 都超过 {target_precision * 100:.0f}% 的阈值：{optimal_threshold:.4f}")
            break

    # 如果没有满足上面条件的阈值，那就用f1_scores最大的对应的阈值来作为最佳阈值
    if optimal_threshold is None:
        opt_idx_ = np.argmax(f1_scores_)
        optimal_threshold = thresholds_[opt_idx_]
        logger.info(
            f"未找到 Precision 和 Recall 都超过 {target_precision * 100:.0f}%，"
            f"选择 F1-score 最大的阈值：{optimal_threshold:.4f}"
        )
    logger.info(f"最终选择的最佳阈值为：{optimal_threshold:.4f}")

    #  在完整的训练验证集上训练最终模型，然候在测试集上评估
    logger.info("在整个训练验证集上训练最终模型...")

    train_val_sequence_ids, train_val_labels = zip(*train_val_data)
    if feature_data is not None:
        train_val_feature_data = {
            sid: feature_data[sid] for sid in train_val_sequence_ids if sid in feature_data
        }
    else:
        train_val_feature_data = None

    train_val_dataset = create_dataset(
        train_val_sequence_ids,
        model_params.get('batch_size', 32),
        max_len,
        this_experiment_features_dir,
        in_channels,
        train_val_feature_data,
        shuffle=True
    )

    test_sequence_ids, test_labels = zip(*test_data)
    if feature_data is not None:
        test_feature_data = {
            sid: feature_data[sid] for sid in test_sequence_ids if sid in feature_data
        }
    else:
        test_feature_data = None

    test_dataset = create_dataset(
        test_sequence_ids,
        model_params.get('batch_size', 32),
        max_len,
        this_experiment_features_dir,
        in_channels,
        test_feature_data,
        shuffle=False
    )

    # 对训练验证集再次计算类别权重，本次和之前一次计算不同
    all_y_train_val = []
    for _, y_batch_ in train_val_dataset:
        y_batch_np_ = y_batch_.numpy().flatten()
        valid_indices_ = y_batch_np_ != -1
        y_batch_flat_ = y_batch_np_[valid_indices_]
        all_y_train_val.extend(y_batch_flat_)
    all_y_train_val = np.array(all_y_train_val)

    classes__ = np.unique(all_y_train_val)
    class_weights__ = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes__,
        y=all_y_train_val
    )
    # 甲基化与非甲基化的占比权重调整
    max_weight = 5.0
    min_weight = 4.0
    class_weights__ = np.log1p(class_weights__)
    class_weights__ = np.clip(class_weights__, min_weight, max_weight)
    class_weight_dict_ = dict(zip(classes__, class_weights__))
    logger.info(f"调整后的类别权重：{class_weight_dict_}")

    # 训练集+验证集统一进行完整训练得到的最终模型
    final_model_save_path = os.path.join(this_experiment_results_dir, "my_trained_model_final.h5")
    # 读取文件是否存在，存在就加载
    if os.path.exists(final_model_save_path):
        logger.info(f"发现已保存的最终模型，加载它以跳过模型训练。")
        model_final = build_model(None, in_channels, None, class_weight_dict_, model_params=model_params)
        model_final.load_weights(final_model_save_path)
        logger.info(f"模型权重已从 '{final_model_save_path}' 加载")
    else:
        # 否则就训练，得到模型
        model_final = build_model(None, in_channels, None, class_weight_dict_, model_params=model_params)
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
        model_final.fit(
            train_val_dataset,
            epochs=model_params.get('epochs', 200),
            callbacks=[early_stopping, reduce_lr]
        )
        model_final.save_weights(final_model_save_path)
        logger.info(f"模型已保存为 '{final_model_save_path}'")

    logger.info("在测试集上评估模型...")

    y_true_test = []
    y_pred_probs_test = []
    y_pred_test = []
    sequence_ids_test = []
    positions_test = []

    for x_batch, y_batch in test_dataset:
        y_pred_batch = model_final.predict(x_batch)
        y_batch_np = y_batch.numpy()
        mask_ = x_batch['mask_input'].numpy()
        batch_sequence_ids_ = x_batch['sequence_id'].numpy()
        b_size_ = mask_.shape[0]
        seq_length_ = mask_.shape[1]

        for i_ in range(b_size_):
            seq_id_bytes_ = batch_sequence_ids_[i_]
            seq_id_ = seq_id_bytes_.decode('utf-8')
            mask_i_ = mask_[i_]
            positions_ = np.arange(seq_length_)[mask_i_]

            y_true_i_ = y_batch_np[i_].flatten()[mask_i_]
            y_pred_probs_i_ = y_pred_batch[i_].flatten()[mask_i_]
            y_pred_i_ = (y_pred_probs_i_ >= optimal_threshold).astype(int)

            sequence_ids_test.extend([seq_id_] * len(positions_))
            positions_test.extend(positions_.tolist())
            y_true_test.extend(y_true_i_.tolist())
            y_pred_probs_test.extend(y_pred_probs_i_.tolist())
            y_pred_test.extend(y_pred_i_.tolist())

    # 保存测试集的结果到文件价中
    test_predictions_file = os.path.join(this_experiment_results_dir, "test_predictions.csv")
    df_test_predictions = pd.DataFrame({
        'Sequence_ID': sequence_ids_test,
        'Position': positions_test,
        'True_Label': y_true_test,
        'Predicted_Probability': [f"{prob_ * 100}%" for prob_ in y_pred_probs_test],
        'Predicted_Label': y_pred_test
    })
    df_test_predictions.to_csv(test_predictions_file, index=False)
    logger.info(f"测试集的预测结果已保存到 {test_predictions_file}")

    # 计算评估指标
    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    precision_test = precision_score(y_true_test, y_pred_test, zero_division=0)
    recall_test = recall_score(y_true_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_true_test, y_pred_test, zero_division=0)
    roc_auc_test = roc_auc_score(y_true_test, y_pred_probs_test)
    mcc_test = matthews_corrcoef(y_true_test, y_pred_test)

    conf_matrix_test = confusion_matrix(y_true_test, y_pred_test)
    logger.info(f"测试集 - Accuracy:  {accuracy_test * 100:.3f}%")
    logger.info(f"测试集 - MCC:       {mcc_test * 100:.5f}%")
    logger.info(f"测试集 - 混淆矩阵：\n{conf_matrix_test}")
    logger.info(f"测试集 - Precision: {precision_test * 100:.3f}%")
    logger.info(f"测试集 - Recall:    {recall_test * 100:.3f}%")
    logger.info(f"测试集 - F1-score:  {f1_test * 100:.3f}%")
    logger.info(f"测试集 - AUC-ROC:   {roc_auc_test * 100:.3f}%")

    test_metrics_file = os.path.join(this_experiment_results_dir, "test_metrics.txt")
    with open(test_metrics_file, 'w') as f__:
        f__.write(f"测试集 - Accuracy:  {accuracy_test * 100:.3f}%\n")
        f__.write(f"测试集 - Precision: {precision_test * 100:.3f}%\n")
        f__.write(f"测试集 - Recall:    {recall_test * 100:.3f}%\n")
        f__.write(f"测试集 - F1-score:  {f1_test * 100:.3f}%\n")
        f__.write(f"测试集 - AUC-ROC:   {roc_auc_test * 100:.3f}%\n")
        f__.write(f"测试集 - MCC:       {mcc_test * 100:.5f}%\n")
        f__.write(f"测试集 - Confusion Matrix:\n{conf_matrix_test}\n")

    # 绘制测试集 ROC、PR、混淆矩阵
    fpr_test, tpr_test, thresholds_test = roc_curve(y_true_test, y_pred_probs_test)
    roc_auc_score_test = auc(fpr_test, tpr_test)
    plt.figure()
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc_score_test:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(this_experiment_results_dir, "roc_curve_test.png"))
    plt.close()

    precision_test_vals, recall_test_vals, thresholds_test_pr = precision_recall_curve(
        y_true_test, y_pred_probs_test
    )
    average_precision_test = average_precision_score(y_true_test, y_pred_probs_test)
    plt.figure()
    plt.step(recall_test_vals, precision_test_vals, where='post',
             label=f'Average precision = {average_precision_test:0.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Set Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(this_experiment_results_dir, "pr_curve_test.png"))
    plt.close()

    disp_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test)
    disp_test.plot()
    plt.title('Test Set Confusion Matrix')
    plt.savefig(os.path.join(this_experiment_results_dir, "confusion_matrix_test.png"))
    plt.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"本次实验总训练时间: {elapsed_time:.2f} 秒")

    # 将本次消融实验的结果存起来
    global_experiment_results[experiment_name] = {
        "fold_accuracy": [m['accuracy'] for m in all_metrics],
        "fold_precision": [m['precision'] for m in all_metrics],
        "fold_f1": [m['f1_score'] for m in all_metrics],
        "fold_auc": [m['roc_auc'] for m in all_metrics],
        "fold_mcc": [m['mcc'] for m in all_metrics],
        "fold_recall": [m['recall'] for m in all_metrics],

        "avg_accuracy": avg_accuracy,
        "avg_precision": avg_precision,
        "avg_f1": avg_f1,
        "avg_roc": avg_roc_auc,
        "avg_mcc": avg_mcc,
        "avg_recall": avg_recall
    }
    # 结束后清理内存，但是不会一下子清空内存
    if feature_data is not None:
        # 清空特征数据
        feature_data.clear()
        del feature_data
    # 清理运行环境中变了
    tf.keras.backend.clear_session()
    gc.collect()
    logger.info("已在 run_experiment 内清理本次实验的特征数据与运行内存。")


def main():
    # 加载配置
    config_file = 'config.yaml'
    base_config = load_config(config_file)

    # 11个消融实验，第一个为基准实验，其余分别从模块和特征层面进行
    ablation_configs = [
        {
            'name': 'baseline_all_features_all_modules',
            'use_cnn': True,
            'use_gat': True,
            'use_attention': True,
            'use_lstm': True,
            'features': ['protbert', 'pssm', 'properties', 'onehot']
        },
        {
            'name': 'no_cnn',
            'use_cnn': False,
            'use_gat': True,
            'use_attention': True,
            'use_lstm': True,
            'features': ['protbert', 'pssm', 'properties', 'onehot']
        },
        {
            'name': 'no_gat',
            'use_cnn': True,
            'use_gat': False,
            'use_attention': True,
            'use_lstm': True,
            'features': ['protbert', 'pssm', 'properties', 'onehot']
        },
        {
            'name': 'no_attention',
            'use_cnn': True,
            'use_gat': True,
            'use_attention': False,
            'use_lstm': True,
            'features': ['protbert', 'pssm', 'properties', 'onehot']
        },
        {
            'name': 'no_lstm',
            'use_cnn': True,
            'use_gat': True,
            'use_attention': True,
            'use_lstm': False,
            'features': ['protbert', 'pssm', 'properties', 'onehot']
        },
        {
            'name': 'no_cnn_no_gat',
            'use_cnn': False,
            'use_gat': False,
            'use_attention': True,
            'use_lstm': True,
            'features': ['protbert', 'pssm', 'properties', 'onehot']
        },
        {
            'name': 'no_cnn_no_attention',
            'use_cnn': False,
            'use_gat': True,
            'use_attention': False,
            'use_lstm': True,
            'features': ['protbert', 'pssm', 'properties', 'onehot']
        },
        {
            'name': 'features_protbert_pssm',
            'use_cnn': True,
            'use_gat': True,
            'use_attention': True,
            'use_lstm': True,
            'features': ['protbert', 'pssm']
        },
        {
            'name': 'features_onehot_properties',
            'use_cnn': True,
            'use_gat': True,
            'use_attention': True,
            'use_lstm': True,
            'features': ['onehot', 'properties']
        },
        {
            'name': 'no_cnn_no_gat_features_protbert_pssm',
            'use_cnn': False,
            'use_gat': False,
            'use_attention': True,
            'use_lstm': True,
            'features': ['protbert', 'pssm']
        },
        {
            'name': 'no_attention_no_lstm_features_onehot_properties',
            'use_cnn': True,
            'use_gat': True,
            'use_attention': False,
            'use_lstm': False,
            'features': ['onehot', 'properties']
        },
    ]

    # 通过for循环来循环运行消融实验
    for i, ab_conf in enumerate(ablation_configs, start=1):
        logger.info(f"\n===== 开始消融实验配置 {i} / {len(ablation_configs)} =====")
        logger.info(f"配置详情: {ab_conf}")
        experiment_config = dict(base_config)
        experiment_config['ablation'] = ab_conf
        run_experiment(experiment_config, experiment_name=ab_conf['name'])
        logger.info(f"===== 完成消融实验配置 {i} / {len(ablation_configs)} =====\n")
        # 内存清理
        # import gc
        # tf.keras.backend.clear_session()
        # gc.collect()  # 进行一次垃圾回收
        # logger.info("已清理本次消融实验的运行内存。")

    # 与一开始的基准实验做对比
    baseline_name = 'baseline_all_features_all_modules'
    if baseline_name in global_experiment_results:
        baseline_results = global_experiment_results[baseline_name]

        # 读取基准实验中每折的Accuracy、F1、AUC、MCC
        baseline_acc_folds = baseline_results["fold_accuracy"]
        baseline_prec_folds = baseline_results["fold_precision"]
        baseline_f1_folds = baseline_results["fold_f1"]
        baseline_auc_folds = baseline_results["fold_auc"]
        baseline_mcc_folds = baseline_results["fold_mcc"]
        baseline_recall_folds = baseline_results["fold_recall"]

        # 读取基准实验评估的平均
        baseline_avg_acc = baseline_results["avg_accuracy"]
        baseline_avg_prec = baseline_results["avg_precision"]
        baseline_avg_f1 = baseline_results["avg_f1"]
        baseline_avg_auc = baseline_results["avg_roc"]
        baseline_avg_mcc = baseline_results["avg_mcc"]
        baseline_avg_recall = baseline_results["avg_recall"]

        # logger.info("======= 开始对比各消融实验与baseline的性能下降幅度，并进行t检验=======")
        #
        # baseline_name = 'baseline_all_features_all_modules'
        # if baseline_name in global_experiment_results:
        #     baseline_results = global_experiment_results[baseline_name]
        #
        #     # 从 baseline 中提取每折指标
        #     baseline_acc_folds = baseline_results["fold_accuracy"]
        #     baseline_f1_folds = baseline_results["fold_f1"]
        #     baseline_auc_folds = baseline_results["fold_auc"]
        #     baseline_mcc_folds = baseline_results["fold_mcc"]
        #     baseline_precision_folds = baseline_results["fold_precision"]
        #
        #     # 读取 baseline 实验的平均分
        #     baseline_avg_acc = baseline_results["avg_accuracy"]
        #     baseline_avg_f1 = baseline_results["avg_f1"]
        #     baseline_avg_auc = baseline_results["avg_roc"]
        #     baseline_avg_mcc = baseline_results["avg_mcc"]
        #     # 同样也要有 baseline 平均 precision
        #     baseline_avg_precision = baseline_results["avg_precision"]
        #
        #     for name, result_dict in global_experiment_results.items():
        #         if name == baseline_name:
        #             continue
        #
        #         # 获取当前实验的每折结果
        #         ablation_acc_folds = result_dict["fold_accuracy"]
        #         ablation_f1_folds = result_dict["fold_f1"]
        #         ablation_auc_folds = result_dict["fold_auc"]
        #         ablation_mcc_folds = result_dict["fold_mcc"]
        #         ablation_precision_folds = result_dict["fold_precision"]
        #
        #         # 计算减少量(= baseline_avg - ablation_avg)
        #         delta_acc = baseline_avg_acc - result_dict["avg_accuracy"]
        #         delta_f1 = baseline_avg_f1 - result_dict["avg_f1"]
        #         delta_auc = baseline_avg_auc - result_dict["avg_roc"]
        #         delta_mcc = baseline_avg_mcc - result_dict["avg_mcc"]
        #         delta_precision = baseline_avg_precision - result_dict["avg_precision"]
        #
        #         # t 检验 (配对)
        #         t_stat_acc, p_val_acc = ttest_rel(baseline_acc_folds, ablation_acc_folds)
        #         t_stat_f1, p_val_f1 = ttest_rel(baseline_f1_folds, ablation_f1_folds)
        #         t_stat_auc, p_val_auc = ttest_rel(baseline_auc_folds, ablation_auc_folds)
        #         t_stat_mcc, p_val_mcc = ttest_rel(baseline_mcc_folds, ablation_mcc_folds)
        #         t_stat_prec, p_val_prec = ttest_rel(baseline_precision_folds, ablation_precision_folds)
        #
        #         logger.info(f"实验: {name}")
        #         logger.info(f"  Accuracy 减少量:   {delta_acc:.4f}, t检验 p-value={p_val_acc:.4e}")
        #         logger.info(f"  F1-score 减少量:   {delta_f1:.4f}, t检验 p-value={p_val_f1:.4e}")
        #         logger.info(f"  AUC 减少量:        {delta_auc:.4f}, t检验 p-value={p_val_auc:.4e}")
        #         logger.info(f"  MCC 减少量:        {delta_mcc:.4f}, t检验 p-value={p_val_mcc:.4e}")
        #         logger.info(f"  Precision 减少量: {delta_precision:.4f}, t检验 p-value={p_val_prec:.4e}")
        #
        #     logger.info("======= 对比完成 =======")
        #
        # # 绘制简单柱状图，展示和基准实验相比的下降量
        # try:
        #     import numpy as np
        #     import matplotlib.pyplot as plt
        #
        #     experiment_names = []
        #     deltas_f1 = []
        #     deltas_auc = []
        #     deltas_mcc = []
        #     deltas_precision = []
        #
        #     for name, result_dict in global_experiment_results.items():
        #         if name == baseline_name:
        #             continue
        #
        #         # 计算与 baseline 的差值
        #         d_f1 = baseline_avg_f1 - result_dict["avg_f1"]
        #         d_auc = baseline_avg_auc - result_dict["avg_roc"]
        #         d_mcc = baseline_avg_mcc - result_dict["avg_mcc"]
        #         d_prec = baseline_avg_precision - result_dict["avg_precision"]
        #
        #         experiment_names.append(name)
        #         deltas_f1.append(d_f1)
        #         deltas_auc.append(d_auc)
        #         deltas_mcc.append(d_mcc)
        #         deltas_precision.append(d_prec)
        #
        #     x_idx = np.arange(len(experiment_names))
        #     width = 0.2
        #
        #     plt.figure(figsize=(10, 6))
        #     # 这里顺序可以随意，示例：F1、AUC、MCC、Precision
        #     plt.bar(x_idx - 1.5 * width, deltas_f1, width, label='Delta_F1')
        #     plt.bar(x_idx - 0.5 * width, deltas_auc, width, label='Delta_AUC')
        #     plt.bar(x_idx + 0.5 * width, deltas_mcc, width, label='Delta_MCC')
        #     plt.bar(x_idx + 1.5 * width, deltas_precision, width, label='Delta_Precision')
        #
        #     plt.xticks(x_idx, experiment_names, rotation=45, ha='right')
        #     plt.ylabel('Decrease from Baseline')
        #     # 标题可相应修改
        #     plt.title('Performance Drop from Baseline (F1/AUC/MCC/Precision)')
        #     plt.legend()
        #     plt.tight_layout()
        #
        #     # 保存图片
        #     plt.savefig(os.path.join(base_config['paths']['output_dir'], 'ablation_performance_drop.png'))
        #     plt.close()
        #     logger.info("已保存 ablation_performance_drop.png 柱状图 (改为 Precision).")
        # except ImportError:
        #     logger.warning("未安装 matplotlib 或其他库，无法绘制图表。仅记录数值。")
        #
        # else:
        #     logger.warning("未找到 baseline 实验结果，无法执行与baseline的统计对比。")
        logger.info(
            "======= 开始对比各消融实验与 baseline 的性能下降幅度，并进行 t 检验=======")

        # 遍历全部实验
        import numpy as np
        from scipy.stats import ttest_rel

        experiment_names = []
        deltas_acc = []
        deltas_rec = []
        deltas_f1 = []
        deltas_auc = []
        deltas_mcc = []
        deltas_prec = []

        for name, result_dict in global_experiment_results.items():
            if name == baseline_name:
                continue

            # 提取每折
            ab_f1_folds = result_dict["fold_f1"]
            ab_auc_folds = result_dict["fold_auc"]
            ab_mcc_folds = result_dict["fold_mcc"]
            ab_prec_folds = result_dict["fold_precision"]
            ab_acc_folds = result_dict["fold_accuracy"]
            ab_rec_folds = result_dict["fold_recall"]

            # 平均
            ab_avg_f1 = result_dict["avg_f1"]
            ab_avg_auc = result_dict["avg_roc"]
            ab_avg_mcc = result_dict["avg_mcc"]
            ab_avg_prec = result_dict["avg_precision"]
            ab_avg_acc = result_dict["avg_accuracy"]
            ab_avg_rec = result_dict["avg_recall"]

            # 计算差值 (Baseline - Ablation)
            delta_f1_ = baseline_avg_f1 - ab_avg_f1
            delta_auc_ = baseline_avg_auc - ab_avg_auc
            delta_mcc_ = baseline_avg_mcc - ab_avg_mcc
            delta_prec_ = baseline_avg_prec - ab_avg_prec
            delta_rec_ = baseline_avg_recall - ab_avg_rec
            delta_acc_ = baseline_avg_acc - ab_avg_acc

            # t 检验
            t_stat_acc, p_val_acc = ttest_rel(baseline_acc_folds, ab_acc_folds)
            t_stat_f1, p_val_f1 = ttest_rel(baseline_f1_folds, ab_f1_folds)
            t_stat_auc, p_val_auc = ttest_rel(baseline_auc_folds, ab_auc_folds)
            t_stat_mcc, p_val_mcc = ttest_rel(baseline_mcc_folds, ab_mcc_folds)
            t_stat_prec, p_val_prec = ttest_rel(baseline_prec_folds, ab_prec_folds)
            t_stat_rec, p_val_rec = ttest_rel(baseline_recall_folds, ab_rec_folds)

            logger.info(f"实验: {name}")
            logger.info(f"  F1减少量:         {delta_f1_:.4f}, p-value={p_val_f1:.4e}")
            logger.info(f"  AUC减少量:        {delta_auc_:.4f}, p-value={p_val_auc:.4e}")
            logger.info(f"  MCC减少量:        {delta_mcc_:.4f}, p-value={p_val_mcc:.4e}")
            logger.info(f"  Precision减少量:  {delta_prec_:.4f}, p-value={p_val_prec:.4e}")
            logger.info(f"  Recall减少量:    {delta_rec_:.4f}, p-value={p_val_rec:.4e}")
            logger.info(f"  Accuracy减少量:  {delta_acc_:.4f}, p-value={p_val_acc:.4e}")

            experiment_names.append(name)
            deltas_f1.append(delta_f1_)
            deltas_auc.append(delta_auc_)
            deltas_mcc.append(delta_mcc_)
            deltas_prec.append(delta_prec_)
            deltas_rec.append(delta_rec_)
            deltas_acc.append(delta_acc_)

        logger.info("======= 对比完成 =======")

        # 画图(用 F1/AUC/MCC/Precision)
        try:
            import matplotlib.pyplot as plt
            x_idx = np.arange(len(experiment_names))
            width = 0.12

            plt.figure(figsize=(12, 6))
            plt.bar(x_idx - 2.5 * width, deltas_acc, width, label='Delta_Acc')
            plt.bar(x_idx - 1.5 * width, deltas_prec, width, label='Delta_Precision')
            plt.bar(x_idx - 0.5 * width, deltas_f1, width, label='Delta_F1')
            plt.bar(x_idx + 0.5 * width, deltas_auc, width, label='Delta_AUC')
            plt.bar(x_idx + 1.5 * width, deltas_mcc, width, label='Delta_MCC')
            plt.bar(x_idx + 2.5 * width, deltas_rec, width, label='Delta_Recall')

            plt.xticks(x_idx, experiment_names, rotation=45, ha='right')
            plt.ylabel('Decrease from Baseline')
            plt.title('Performance Drop from Baseline (Acc/Prec/F1/AUC/MCC/Recall)')
            plt.legend()
            plt.tight_layout()

            fig_path = os.path.join(base_config['paths']['output_dir'], 'ablation_performance_drop_extended.png')
            plt.savefig(fig_path)
            plt.close()
            logger.info(f"已保存 {fig_path} (包含所有评估指标).")
        except ImportError:
            logger.warning("未安装 matplotlib 或其他库，无法绘制图表，仅记录数值。")


if __name__ == '__main__':
    # GPU显示增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("已启用动态显存分配")
        except RuntimeError as e:
            print(f"启用动态显存分配失败: {e}")

    main()