# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 21:10
# @Author  : Changzai Pan
# @File    : other_method.py


import os
import numpy as np
from os.path import join
from utils_seg import print2, sent_to_seg, sent_to_seg_parrel, combine_seg_01_result, seg_to_01_reverse,\
    get_input_text_file, check_result_dir, summary_text, integrate_result, load_json, output_json,\
    combine_dict_result
from typing import Generator, Set
from tqdm import tqdm
from utils import Char


def segment_other_method(inputfile, filedir, methods=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'],
                         user_dict_dir=None, num_of_processes=4, log_file='log.txt',
                         modeldir='D:/pycharm/PycharmProjects/TopWORDS-Seg/models/stanford-corenlp-full-2018-10-05'):

    print2(f'Segmentation by Other Method:{methods}', file=log_file)
    print2(f'inputfile: {inputfile}', file=log_file)
    print2(f'filedir: {filedir}', file=log_file)

    print2(f'load text file', file=log_file)
    sent_list = get_input_text_file(inputfile)  # 读取文本数据
    summary_text(sent_list, log_file)

    print2(f'check result dir', file=log_file)
    seg_result_dir, seg_01_result_dir, dict_result_dir = check_result_dir(filedir)  # 检查各个目录

    # 运行 5 种方法
    # methods = ['jieba', 'snownlp', 'pkuseg', 'thulac', 'pyltp', 'foolnltk']
    print2(f'start segmentation', file=log_file)
    for method in methods:

        print2(f'method: {method}', file=log_file)
        if num_of_processes > 1:
            seg_list = sent_to_seg_parrel(sent_list, method=method, user_dict_dir=user_dict_dir,
                                          modeldir=modeldir, num_of_processes=num_of_processes)
        else:
            seg_list = sent_to_seg(sent_list, method=method, user_dict_dir=user_dict_dir,
                                   modeldir=modeldir)

        # 如果用到了词典，方法名后缀加'_userdict'
        if user_dict_dir:
            method = method + '_userdict'

        print2(f'integrate result of method {method}', file=log_file)
        integrate_result(method, seg_list, sent_list, seg_result_dir, seg_01_result_dir, dict_result_dir)
    print2('finish', file=log_file)


def segment_combine_method(inputfile, filedir, methods=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'],
                    log_file='log.txt'):

    print2(f'Segmentation by Combine Method:{methods}', file=log_file)
    print2(f'inputfile: {inputfile}', file=log_file)
    print2(f'filedir: {filedir}', file=log_file)

    print2(f'load text file', file=log_file)
    sent_list = get_input_text_file(inputfile)  # 读取文本数据
    summary_text(sent_list, log_file)

    print2(f'check result dir', file=log_file)
    seg_result_dir, seg_01_result_dir, dict_result_dir = check_result_dir(filedir)  # 检查各个目录

    # 通过之前的分词文件得到combine方法的seg_list和seg_01_list
    method = 'combine'
    print2(f'start segmentation', file=log_file)
    print2(f'method: {method}', file=log_file)
    seg_01_count_list = combine_seg_01_result(methods, seg_01_result_dir)
    print(seg_01_count_list[0])
    seg_01_count_list = [seg_01 + np.random.random(len(seg_01)) - 0.5 for seg_01 in seg_01_count_list]  # 加随机扰动，避免偶数时情况
    print(seg_01_count_list[0])
    seg_01_list = [[int(a) for a in seg_01_count > len(methods) / 2] for seg_01_count in seg_01_count_list]
    print(seg_01_list[0])
    seg_list = [seg_to_01_reverse(seg_01, sent) for seg_01, sent in zip(seg_01_list, sent_list)]

    print2(f'integrate result of method {method}', file=log_file)
    integrate_result(f'{method}', seg_list, sent_list, seg_result_dir, seg_01_result_dir, dict_result_dir)

    # 词典添加其他分词的词典
    # combine_dict = load_json(join(dict_result_dir, 'combine.json'))
    # combine_dict_set = combine_dict_result(methods, dict_result_dir)
    # for w in set(combine_dict_set) - set(combine_dict.keys()):
    #     combine_dict[w] = 0
    # output_json(combine_dict, join(dict_result_dir, 'combine.json'))

    print2('finish', file=log_file)


def get_sent_string_list_from_text_string(text_string, punctuations):
    sent_string = []
    for char in text_string:
        if char in punctuations:
            if sent_string:
                yield ''.join(sent_string)
                sent_string = []
            yield char
        else:
            sent_string.append(char)
    if sent_string:
        yield ''.join(sent_string)


def yield_char_from_text_file(text_file: str) -> Generator[str, None, None]:
    with open(text_file, encoding='utf-8') as f:
        while True:
            char = f.read(1)
            if char:
                yield char
            else:
                break

# 多了一步去除标点
def segment_multi_method(inputfile, filedir, punctuations,
                         methods=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'],
                         user_dict_dir=None, num_of_processes=4, log_file='log.txt',
                         modeldir='D:/pycharm/PycharmProjects/Allsegment/stanford-corenlp-full-2018-10-05'):

    print2(f'Segmentation by Other Method:{methods}', file=log_file)
    print2(f'inputfile: {inputfile}', file=log_file)
    print2(f'filedir: {filedir}', file=log_file)
    print2(f'load and modify text file', file=log_file)
    sent_list = []
    for sent_string in get_sent_string_list_from_text_string(yield_char_from_text_file(inputfile), punctuations):
        if sent_string not in punctuations:
            sent_list.append(sent_string)

    summary_text(sent_list, log_file)

    print2(f'check result dir', file=log_file)
    seg_result_dir, seg_01_result_dir, dict_result_dir = check_result_dir(filedir)  # 检查各个目录

    # 运行 5 种方法
    # methods = ['jieba', 'snownlp', 'pkuseg', 'thulac', 'pyltp', 'foolnltk']
    print2(f'start segmentation', file=log_file)
    for method in methods:
        print2(f'method: {method}', file=log_file)
        if num_of_processes > 1:
            seg_list = sent_to_seg_parrel(sent_list, method=method, user_dict_dir=user_dict_dir,
                                          modeldir=modeldir, num_of_processes=num_of_processes)
        else:
            seg_list = sent_to_seg(sent_list, method=method, user_dict_dir=user_dict_dir,
                                   modeldir=modeldir)

        # 如果用到了词典，方法名后缀加'_userdict'
        if user_dict_dir:
            method = method + '_userdict'

        print2(f'integrate result of method {method}', file=log_file)
        integrate_result(method, seg_list, sent_list, seg_result_dir, seg_01_result_dir, dict_result_dir)
    print2('finish', file=log_file)



if __name__ == '__main__':
    inputfile = os.path.join('RDA', 'test', 'data', 'modify_text.txt')
    filedir = os.path.join('RDA', 'test', 'output')
    segment_other_method(inputfile, filedir, methods=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'],
                         user_dict_dir=None, num_of_processes=2, log_file=join(filedir, 'log.txt'),
                         modeldir='D:/pycharm/PycharmProjects/Allsegment/stanford-corenlp-full-2018-10-05')
    segment_combine_method(inputfile, filedir, methods=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'],
                           log_file=join(filedir, 'log.txt'))

