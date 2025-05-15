# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:55
# @Author  : Changzai Pan
# @File    : wiki_topwords_seg.py


# from RDA.xiehe.xhw_0_config import *
from os.path import join
from utils_seg import load_json, output_json, is_chinese_character, read_txt_encoding
from collections import defaultdict
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import scipy.sparse as sp
from statsmodels.stats.multitest import fdrcorrection as fdr
import scipy.stats
import argparse


def get_corpus(feature_data_seg, disease_list):
    max_index = len(feature_data_seg)
    corpus = [[] for i in range(max_index)]
    for ix, data in tqdm(enumerate(feature_data_seg)):
        for k, v in data.items():
            if k == '诊断名称':
                corpus[ix].append(disease_list[v])
            else:
                corpus[ix].extend(v)
    corpus = [list(set(c)) for c in corpus]
    return corpus


def get_correlarion_matrix(corpus):
    word_list = list(set([w for c in corpus for w in c]))
    word_ix = {word: ix for ix, word in enumerate(word_list)}
    data = []
    indices = []
    indptr = [0]
    for ix, c in tqdm(enumerate(corpus)):
        ix_list = [word_ix[ci] for ci in c]
        ix_list.sort()
        data.extend([1] * len(ix_list))
        indices.extend(ix_list)
        indptr.append(indptr[-1] + len(ix_list))
    freq_matrix_tmp = sp.csc_matrix((data, indices, indptr), shape=(len(word_list), len(corpus)))
    sample_n = freq_matrix_tmp.shape[1]
    cov_matrix = freq_matrix_tmp.dot(freq_matrix_tmp.T)
    freq_sum = freq_matrix_tmp.sum(1).T.getA().tolist()[0]  # 求合后再转化成list
    return cov_matrix.toarray(), word_list, freq_sum, sample_n


# 生成pvalue矩阵 共现为0或比例小于平均的都设置为0.5
def get_pvalue_matrix(cov_matrix, freq_sum, sample_n):
    correlation_matrix = sp.csc_matrix(cov_matrix)
    row_ix = correlation_matrix.nonzero()[0]
    col_ix = correlation_matrix.nonzero()[1]
    data = correlation_matrix.data
    pvalue_matrix = np.ones(cov_matrix.shape) / 2
    for t in tqdm(range(len(data))):
        i = row_ix[t]
        j = col_ix[t]
        if i != j:
            co_num = data[t]
            f1 = freq_sum[i]
            f2 = freq_sum[j]
            if co_num * sample_n > f1 * f2:
                cont_table = [[co_num, f1 - co_num], [f2 - co_num, sample_n - f1 - f2 + co_num]]
                p_value = scipy.stats.chi2_contingency(cont_table)[1] / 2
                pvalue_matrix[i, j] = p_value
    return pvalue_matrix


def flatten_matrix(pvalue_matrix):  # 把对称的方阵的上三角压缩成1维向量
    pvalue_matrix2 = []
    for ix, line in enumerate(pvalue_matrix):
        pvalue_matrix2.extend(line[ix + 1:])
    return np.array(pvalue_matrix2)


def reshape_matrix(reject, L):
    binary_matrix = []
    reject = list(reject)
    for i in range(L):
        l1 = int((2 * L - i - 1) * i / 2)
        l2 = int((2 * L - i - 2) * (i + 1) / 2)
        l3 = int((L + i) * (L - i - 1) / 2)
        l4 = int((L + i - 1) * (L - i) / 2)
        # print(l1, l2, l3, l4)
        binary_matrix.append([0] * (l4 - l3) + [0] + reject[l1:l2])
    return np.array(binary_matrix) + np.array(binary_matrix).T


# 由生成binary矩阵
# 1. Banjamini FDR_control
# 2. Bonferroni correction
def get_binary_matrix(pvalue_matrix, method=1, thr=0.05):
    if pvalue_matrix.shape[0] != pvalue_matrix.shape[1]:
        print('pvalue_matrix not square')
    else:
        L = pvalue_matrix.shape[0]
    if method == 0:  # Bonferroni correction
        pvalue_thr = thr
        binary_matrix = pvalue_matrix < pvalue_thr
    elif method == 1:  # FDR_control
        pvalue_matrix2 = flatten_matrix(pvalue_matrix)
        reject, pvals_corrected = fdr(pvalue_matrix2, alpha=thr)
        binary_matrix = reshape_matrix(reject, L)
    elif method == 2:  # Bonferroni correction
        pvalue_thr = thr / (L * (L - 1) / 2)
        binary_matrix = pvalue_matrix < pvalue_thr
    return binary_matrix


def get_high_cor_word(binary_matrix, word_list, freq_sum, cov_matrix, pvalue_matrix, technical_word_dict):
    df_matrix = []
    for index, element in np.ndenumerate(binary_matrix):
        if element:
            i1 = index[0]
            if i1 == word_list.index('子宫肌瘤'):
                print(i2, [word_list[i1], technical_word_dict.get(word_list[i1], '疾病'), freq_sum[i1],
                              word_list[i2], technical_word_dict.get(word_list[i2], '疾病'), freq_sum[i2],
                              cov_matrix[i1, i2], corr, pvalue_matrix[i1, i2]])
            i2 = index[1]
            corr = cov_matrix[i1, i2] / np.sqrt(freq_sum[i1] * freq_sum[i2])
            df_matrix.append([word_list[i1], technical_word_dict.get(word_list[i1], '疾病'), freq_sum[i1],
                              word_list[i2], technical_word_dict.get(word_list[i2], '疾病'), freq_sum[i2],
                              cov_matrix[i1, i2], corr, pvalue_matrix[i1, i2]])
    df_high_cor = pd.DataFrame(df_matrix, columns=['word1', 'cate1', 'freq1', 'word2', 'cate2', 'freq2', 'co_num', 'corr', 'p_value'])
    return df_high_cor


def main(feature_table_dir, technical_word_dict_dir, output_dir, thr=0.05):

    feature_data_seg = load_json(join(feature_table_dir, 'feature_data_seg_standard.json'))
    disease_list = load_json(join(feature_table_dir, 'disease_list.json'))
    technical_word_dict = load_json(technical_word_dict_dir)

    # 分词模式：
    # 同一行的各个栏目为同一个文本，且文本中加入诊断名称,并且把变种名称转化为标准名称
    corpus = get_corpus(feature_data_seg, disease_list)
    cov_matrix, word_list, freq_sum, sample_n = get_correlarion_matrix(corpus)
    pvalue_matrix = get_pvalue_matrix(cov_matrix, freq_sum, sample_n)

    binary_matrix = get_binary_matrix(pvalue_matrix, method=1, thr=thr)
    num_nodes = len(word_list)
    num_edges = sum(sum(binary_matrix)) / 2
    num_density = num_edges / (num_nodes * (num_nodes - 1) * 2)
    print(num_nodes, num_edges, num_density)

    df_high_cor = get_high_cor_word(binary_matrix, word_list, freq_sum, cov_matrix, pvalue_matrix, technical_word_dict)
    df_high_cor = df_high_cor[(df_high_cor.co_num>2)&(df_high_cor['corr']>0.01)]
    print(len(df_high_cor))
    df_high_cor.to_excel(output_dir)


if __name__ == '__main__':

    # 输入： FeatureTable
    # 依赖: statsmodels, scipy, tqdm

    # python knowledgegraph_construction.py --feature_table_path data\feature_table --technical_word_dict_dir data\dictionary\technical_term_dict_final.json --output_dir data\knowledge_graph\knowledge_graph.xlsx

    parser = argparse.ArgumentParser(
        description="Generate a knowledge graph based on feature table and technical word dictionary.")

    # 添加命令行参数
    parser.add_argument('--feature_table_path', type=str, required=True, help='Path to the feature table file.')
    parser.add_argument('--technical_word_dict_dir', type=str, required=True,
                        help='Path to the technical word dictionary file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the knowledge graph.')
    parser.add_argument('--thr', type=float, default=0.05, help='Threshold value for the algorithm (default: 0.05).')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(
        feature_table_path=args.feature_table_path,
        technical_word_dict_dir=args.technical_word_dict_dir,
        output_dir=args.output_dir,
        thr=args.thr
    )
    # feature_table_path = r'D:\PycharmProjects\SmartGyne\data\feature_table'
    # technical_word_dict_dir = r'D:\PycharmProjects\SmartGyne\data\dictionary\technical_term_dict_final.json'
    # output_dir = r'D:\PycharmProjects\SmartGyne\data\knowledge_graph\knowledge_graph.xlsx'
    # main(feature_table_path, technical_word_dict_dir, output_dir, thr=0.05)
