import re
import pandas as pd
from Bio import SeqIO
import logging
from utils import amino_acid_properties

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 清理序列，统一长度
def clean_sequence(sequence, seq_id, min_len, max_len):
    sequence = sequence.upper()
    if not min_len <= len(sequence) <= max_len:
        # 如果序列长度不在指定范围内，返回空
        return None
    valid_amino_acids = set(amino_acid_properties.keys())
    # 无效的氨基酸换成X
    sequence = ''.join([aa if aa in valid_amino_acids else 'X' for aa in sequence])
    if not sequence:
        return None
    return sequence

# 从TSV文件中提取所有的Entry，以及是否有甲基化位点信息
def extract_entries_with_methylation_info(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    entries_info = {}
    for _, row in df.iterrows():
        entry = row['Entry']
        modified_residues = row['Modified residue']
        if isinstance(modified_residues, str) and 'methyl' in modified_residues.lower():
            # 有甲基化位点
            sites = []
            # 用正则表达式找甲基信息
            mod_res_matches = re.finditer(r'MOD_RES\s+(\d+);', modified_residues)
            for match in mod_res_matches:
                pos = int(match.group(1))
                sites.append(pos)
            if not sites:
                # 如果未匹配到位点信息，尝试另一种匹配方式
                pos_matches = re.finditer(r'(\d+);', modified_residues)
                for match in pos_matches:
                    pos = int(match.group(1))
                    sites.append(pos)
            if not sites:
                # 要是还没有则给个提示
                logger.warning(f"在 Entry {entry} 中，无法提取位点信息：'{modified_residues}'")
            entries_info[entry] = {'has_methylation': True, 'sites': sites}
        else:
            # 没有甲基化位点，就跳过这个entry
            pass
    return entries_info

# 生成样本，只使用包含甲基化位点的序列，根据上面得到的甲基信息，到fasta文件中去找与之匹配的蛋白质序列
def generate_samples(fasta_file, entries_info):
    sequences = []
    sequence_ids = []
    labels = []

    # 将 fasta 文件中的序列读取为字典，键为序列 ID
    from Bio import SeqIO
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'), key_function=lambda rec: rec.id.split('|')[1])

    sample_count = 0

    for entry, info in entries_info.items():
        if entry in fasta_sequences:
            # 只处理包含甲基化位点的序列
            record = fasta_sequences[entry]
            seq = str(record.seq)
            seq_id = entry
            # 清理序列
            cleaned_seq = clean_sequence(seq, seq_id, 20, 1024)
            if cleaned_seq is None:
                continue
            sequences.append(cleaned_seq)
            sequence_ids.append(seq_id)
            # 标签中标记甲基化位点
            label = [0] * len(cleaned_seq)
            methylation_positions = info['sites']
            for pos in methylation_positions:
                if pos - 1 < len(cleaned_seq):
                    label[pos - 1] = 1
            labels.append(label)
            sample_count += 1
        else:
            logger.warning(f"Entry {entry} 在 fasta 文件中未找到。")

    if len(sequences) == 0:
        logger.error("没有生成任何序列，终止模型！")
        return None

    return sequences, sequence_ids, labels