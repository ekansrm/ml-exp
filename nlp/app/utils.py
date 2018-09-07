# 句子抽取
# 我们认为, 只有相邻的句子才有相关性
# 句子的取法为:
#   给定一个窗口长度
#   从中抽取句子对
#
from typing import List, Dict, Tuple, TextIO, BinaryIO

from collections import defaultdict
from functools import reduce
import random
import time
import itertools

random.seed(time.time())

dep_token = defaultdict(lambda: 0)

dep_token.update(

    {
        'SBV': 1,  # 主谓关系	subject-verb	我送她一束花 (我 <-- 送)
        'VOB': 2,  # 动宾关系	直接宾语，verb-object	我送她一束花 (送 --> 花)
        'IOB': 3,  # 间宾关系	间接宾语，indirect-object	我送她一束花 (送 --> 她)
        'FOB': 4,  # 前置宾语	前置宾语，fronting-object	他什么书都读 (书 <-- 读)
        'DBL': 5,  # 兼语	double	他请我吃饭 (请 --> 我)
        'ATT': 6,  # 定中关系	attribute	红苹果 (红 <-- 苹果)
        'ADV': 7,  # 状中结构	adverbial	非常美丽 (非常 <-- 美丽)
        'CMP': 8,  # 动补结构	complement	做完了作业 (做 --> 完)
        'COO': 9,  # 并列关系	coordinate	大山和大海 (大山 --> 大海)
        'POB': 10,  # 介宾关系	preposition-object	在贸易区内 (在 --> 内)
        'LAD': 11,  # 左附加关系	left adjunct	大山和大海 (和 <-- 大海)
        'RAD': 12,  # 右附加关系	right adjunct	孩子们 (孩子 --> 们)
        'IS': 13,  # 独立结构	independent structure	两个单句在结构上彼此独立
        'WP': 14,  # 标点符号	punctuation	标点符号
        'HED': 15,  # 核心关系	head	指整个句子的核心
    }
)


def dep_tokenizer(dep):
    return dep_token[dep]


def random_sample_key(m, w, n, k, unique=True):
    """
    随机从m个样本中取出k个排列, 每个排列大小是n, 要求不重复
    :param m:
    :param n:
    :return:
    """
    # 首先, 全排列的个数为 m*(m-1)...*(m-n+1) 如果超过这个数, 是不可能不重复的
    if w < n:
        raise RuntimeError("样本序列长度不能超过窗口长度")

    # 首先, 每个窗口的全排列的总数是
    full_permutation_num_per_window = reduce(lambda x, y: x * y, range(w - n + 1, w + 1))

    # TODO 实际上真实的样本空间比这个大, 但先用小的进行计算
    total_sampling_space_size = (m // w) * full_permutation_num_per_window

    if k > total_sampling_space_size:
        return []

    # 先随机生产取样窗口,

    windows_b = list(range(0, m - w))
    random.shuffle(windows_b)

    sample_set = set()
    # 随机地去窗口, 生成全排列, 加入到样本序列集里
    i = 0
    while i < len(windows_b):
        w_b = windows_b[i]
        w_e = w_b + w
        for s in itertools.permutations(range(w_b, w_e), n):
            sample_set.add(s)
            if len(sample_set) >= k:
                return list(sample_set)
        i += 1

    return list(sample_set)


sample_window_len = 5


# 二元打分
def score_binary(pos1, pos2):
    return 1 if pos1 < pos2 else 0


def corpus_to_sample(corpus, batch_size, score_func):
    return [
        (corpus[a], corpus[b], score_func(a, b))
        for a, b in random_sample_key(len(corpus), sample_window_len, 2, batch_size)
    ]


def slice_sentence(x: str):
    items = x.split("|")
    dep = items[0::3]
    op1 = items[1::3]
    op2 = items[2::3]
    # tokenizer
    # embedding

    return dep, op1, op2


max_len = 50


def process_sample(batch):
    """
    [(sentence1, sentence2, y)]
    :param batch:
    :return:
    """
    dep_a = []
    op1_a = []
    op2_a = []
    dep_b = []
    op1_b = []
    op2_b = []
    y = []
    for (sentence1, sentence2, _y) in batch:
        _dep_a, _op1_a, _op2_a = slice_sentence(sentence1)
        _dep_b, _op1_b, _op2_b = slice_sentence(sentence2)

        dep_a.append(_dep_a)
        op1_a.append(_op1_a)
        op2_a.append(_op2_a)

        dep_b.append(_dep_b)
        op1_b.append(_op1_b)
        op2_b.append(_op2_b)
        y.append(_y)

    return dep_a, op1_a, op2_a, dep_b, op1_b, op2_b, y


def format_corpus():
    pass


def read_corpus(path: str) -> List[List[str]]:
    batches = []
    with open(path, 'r', encoding='utf-8') as fp:
        batch = []
        for line in fp.readlines():
            line = line.strip()
            batch.append(line)
            if '' == line:
                batches.append(batch)
                batch = []
                continue
        if 0 != len(batch):
            batches.append(batch)
    return batches


def read_vocab(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as fp:
        vocab = fp.readlines()
        vocab = list([word.strip() for word in vocab])
    return vocab


def to_file_for_str_list(file, str_list: List[str], delimiter: str = '\n'):
    with open(file, 'w', encoding='utf-8') as fp:
        fp.write(delimiter.join(str_list))
        fp.flush()
