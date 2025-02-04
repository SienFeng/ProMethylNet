import os
import uuid
import h5py
import numpy as np
import subprocess
import logging
import tensorflow as tf
import random
from transformers import AutoTokenizer, TFAutoModel, AutoConfig
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import logging as transformers_logging
from data_preprocessing import clean_sequence
from utils import sanitize_filename, amino_acid_properties
import gc
import warnings

# 判断pssm是什么类型的任务，然候根据类型用多线程
task_type = "compute"
warnings.filterwarnings('ignore', category=FutureWarning, message='.*resume_download.*')

# 不显示提示信息，只显示TensorFlow的错误的信息
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义输出目录（如果需要，可以在主程序中覆盖此路径）
output_dir = r"F:/project_output/"
os.makedirs(output_dir, exist_ok=True)

# One-Hot编码的全局变量
amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
categories = [list(amino_acids)]
encoder = OneHotEncoder(categories=categories, sparse=False, handle_unknown='ignore')
encoder.fit(np.array(categories[0]).reshape(-1, 1))

# One-Hot 编码
def one_hot_encode(sequence):
    one_hot = encoder.transform(np.array(list(sequence)).reshape(-1, 1))
    return one_hot

# 理化特征提取
def extract_properties(sequence):
    properties = [amino_acid_properties.get(aa, [0.0, 0.0]) for aa in sequence]
    return np.array(properties)

# 获取 PSSM 的函数，使用多进程并行处理
def get_pssm(args):
    sequence_id, sequence, blast_db, num_iterations = args
    sanitized_sequence_id = sanitize_filename(sequence_id)

    # 添加唯一标识符
    unique_id = uuid.uuid4().hex
    seq_file = os.path.join(output_dir, f"{sanitized_sequence_id}_{unique_id}.fasta")
    pssm_temp_file = os.path.join(output_dir, f"{sanitized_sequence_id}_{unique_id}.pssm.txt")

    # 转换 blast_db 为绝对路径
    blast_db = os.path.abspath(blast_db)

    try:
        # 将蛋白质序列的ID和序列写到一个临时的fasta的文件中
        with open(seq_file, 'w') as output_handle:
            output_handle.write(f">{sanitized_sequence_id}\n")
            output_handle.write(sequence)

        # 创建需要生成pssm文件的命令
        psiblast_exe = 'psiblast'
        blast_cmd = [
            psiblast_exe,
            '-query', seq_file,
            '-db', blast_db,
            '-num_iterations', str(num_iterations),
            '-out_ascii_pssm', pssm_temp_file,
            '-num_threads', '24',
        ]

        # 开始执行命令，并且抓取错误
        process = subprocess.Popen(blast_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout, stderr = process.communicate()
        return_code = process.returncode

        # 调试使用，返回出现的错误
        if return_code != 0 or not os.path.exists(pssm_temp_file):
            print(f"PSI-BLAST 命令执行失败，序列 ID：{sequence_id}")
            print(f"错误信息：{stderr.decode(errors='ignore')}")
            return sequence_id, None

        # 提取生成的pssm文件的数据
        pssm_data = []
        start_reading = False
        skip_next_line = False
        with open(pssm_temp_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                if "Last position-specific scoring matrix computed" in line:
                    start_reading = True
                    skip_next_line = True
                    continue
                if skip_next_line:
                    skip_next_line = False
                    continue
                if start_reading and line.strip():
                    if 'Lambda' in line:
                        break
                    if line.strip()[0].isdigit():
                        columns = line.strip().split()
                        pssm_row = [float(x) for x in columns[2:22]]
                        pssm_data.append(pssm_row)
                    else:
                        continue
        if not pssm_data:
            print(f"警告！无法从 {pssm_temp_file} 读取 PSSM 数据，序列 ID：{sequence_id}")
            return sequence_id, None
        pssm_matrix = np.array(pssm_data)
        pssm_matrix = (pssm_matrix - pssm_matrix.min()) / (pssm_matrix.max() - pssm_matrix.min())

        return sequence_id, pssm_matrix
    except Exception as e:
        print(f"处理序列 {sequence_id} 时发生异常：{str(e)}")
        return sequence_id, None
    # 最后删除临时文件
    finally:
        if os.path.exists(seq_file):
            os.remove(seq_file)
        if os.path.exists(pssm_temp_file):
            os.remove(pssm_temp_file)

# 用CPU的多线程来批量提取，加快速度
def extract_pssm_features(sequence_ids, sequences, blast_db, num_iterations):
    pssm_dict = {}
    args_list = [(seq_id, seq, blast_db, num_iterations) for seq_id, seq in zip(sequence_ids, sequences)]
    # 密集型，调用全部的CPU线程，加速
    if task_type == "io":
        max_workers = os.cpu_count()
    else:
        # 否则就用一般的线程
        max_workers = max(1, os.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_pssm, args) for args in args_list]
        for future in as_completed(futures):
            seq_id, pssm = future.result()
            if pssm is not None:
                pssm_dict[seq_id] = pssm
    return pssm_dict

# 初始化模型的函数，在主进程中调用
# def init_model():
#     config = AutoConfig.from_pretrained('Rostlab/prot_bert', output_hidden_states=False)
#     tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
#     model = TFAutoModel.from_pretrained('Rostlab/prot_bert', from_pt=True)
#     return tokenizer, model

# 加载预训练语言模型
def init_model():
    config = AutoConfig.from_pretrained('E:/Methylation/prot_bert', output_hidden_states=False)
    tokenizer = AutoTokenizer.from_pretrained('E:/Methylation/prot_bert', do_lower_case=False)
    model = TFAutoModel.from_pretrained('E:/Methylation/prot_bert', from_pt=True)
    return tokenizer, model

# 获取ProtBERT嵌入的函数，批量处理，利用 GPU
def get_protbert_embeddings(sequences, tokenizer, model):
    max_length = 1024
    batch_size = 32
    embeddings_list = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        sequences_spaced = [' '.join(list(seq)) for seq in batch_seqs]
        inputs = tokenizer(
            sequences_spaced,
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        embeddings = embeddings[:, 1:-1, :]
        embeddings = embeddings.numpy()
        embeddings_list.extend(embeddings)
        del inputs
        del outputs
        gc.collect()
        tf.keras.backend.clear_session()
    return embeddings_list

# 只拼接有效的特征
def concatenate_features(seq, features_dict):
    # 收集非 None 特征
    arrs = [ft for ft in features_dict.values() if ft is not None]
    if len(arrs) == 0:
        return None
    # 先找最短长度
    min_length = min(arr.shape[0] for arr in arrs)
    # 截断每个特征到 min_length
    arrs = [arr[:min_length, :] for arr in arrs]
    combined_features = np.concatenate(arrs, axis=1)

    if np.isnan(combined_features).any() or np.isinf(combined_features).any():
        print("特征包含 NaN 或 Inf 值")
        return None
    return combined_features

# 特征提取并保存的函数，批量处理序列，根据feature_set来看那些特征需要拼接，消融实验用
def extract_and_save_features(sequence_ids, sequences, labels, features_dir, blast_db, num_iterations,
                              batch_size_protbert, min_len, max_len, tokenizer, model,
                              feature_set=None):
    logger.info("开始提取序列的特征。。。")
    os.makedirs(features_dir, exist_ok=True)

    # 筛选需要提取特征的序列
    sequences_to_process = []
    sequence_ids_to_process = []
    labels_to_process = []
    for idx, (seq_id, seq, label) in enumerate(zip(sequence_ids, sequences, labels)):
        sanitized_id = sanitize_filename(seq_id)
        feature_file = os.path.join(features_dir, f"{sanitized_id}_features.h5")
        if not os.path.exists(feature_file):
            # 清理序列
            cleaned_seq = clean_sequence(seq, seq_id, min_len, max_len)
            if cleaned_seq is not None:
                sequences_to_process.append(cleaned_seq)
                sequence_ids_to_process.append(seq_id)
                labels_to_process.append(label)
        # else: 特征文件已存在则跳过

    if not sequences_to_process:
        logger.info("所有序列的特征文件已存在，无需提取。")
        return

    # 设置随机数种子
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # 分批次处理，避免占用过多内存
    batch_size_ = 1000
    for i in range(0, len(sequences_to_process), batch_size_):
        batch_sequences = sequences_to_process[i:i + batch_size_]
        batch_sequence_ids = sequence_ids_to_process[i:i + batch_size_]
        batch_labels = labels_to_process[i:i + batch_size_]

        # 提取 PSSM 特征
        if 'pssm' in feature_set:
            pssm_dict = extract_pssm_features(batch_sequence_ids, batch_sequences, blast_db, num_iterations)
        else:
            pssm_dict = {}

        # 提取ProtBERT嵌入
        if 'protbert' in feature_set:
            protbert_embeddings = get_protbert_embeddings(batch_sequences, tokenizer, model)
            protbert_dict = dict(zip(batch_sequence_ids, protbert_embeddings))
        else:
            protbert_dict = {}

        # 融合特征并保存
        for sid, seq_, lbl_ in zip(batch_sequence_ids, batch_sequences, batch_labels):
            # 分别获取四种特征，看在不在feats_dict中
            feats_dict = {}
            if 'onehot' in feature_set:
                feats_dict['onehot'] = one_hot_encode(seq_)
            else:
                feats_dict['onehot'] = None

            if 'properties' in feature_set:
                feats_dict['properties'] = extract_properties(seq_)
            else:
                feats_dict['properties'] = None

            if 'protbert' in feature_set:
                feats_dict['protbert'] = protbert_dict.get(sid, None)
            else:
                feats_dict['protbert'] = None

            if 'pssm' in feature_set:
                feats_dict['pssm'] = pssm_dict.get(sid, None)
            else:
                feats_dict['pssm'] = None

            # 开始融合
            combined_features = concatenate_features(seq_, feats_dict)
            if combined_features is not None:
                sanitized_id = sanitize_filename(sid)
                feature_file_ = os.path.join(features_dir, f"{sanitized_id}_features.h5")
                with h5py.File(feature_file_, 'w') as hf:
                    hf.create_dataset('features', data=combined_features, compression='gzip')
                    hf.create_dataset('labels', data=lbl_, compression='gzip')
            else:
                logger.warning(f"序列 {sid} 的特征融合失败。")
                continue

        # 清理内存
        del pssm_dict
        del protbert_dict
        if 'protbert' in feature_set:
            del protbert_embeddings
        gc.collect()
        tf.keras.backend.clear_session()

    logger.info("所有序列特征提取完毕！")