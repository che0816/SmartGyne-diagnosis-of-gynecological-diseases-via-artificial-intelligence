# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 0:55
# @Author  : Changzai Pan
# @File    : wiki_topwords_seg.py

# import os
# import sys
# print(os.getcwd())
# sys.path.append(os.getcwd())
# from RDA.xiehe.xhw_0_config_server import *
import os
from os.path import join
from TopwordsSeg import topwords_seg, get_sent_prior_from_methods
from utils_seg import load_json
from other_method import segment_multi_method
import argparse


def kappa_transfer(kappa):
    if kappa == 1:
        return float('inf')
    else:
        return kappa / (1 - kappa)

def main(inputfile_dir, punctuations_dir, prior_words_dir, output_dir, kappa_d, kappa_s, tau_l, tau_f):
    punctuations = load_json(punctuations_dir)
    # 先分词
    segment_multi_method(inputfile_dir, output_dir, punctuations,
                         methods=['jieba'],  # , 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'],  # 'ltp',
                         user_dict_dir=None, num_of_processes=4, log_file='log.txt')

    get_sent_prior_from_methods(
        seg_01_result_dir=join(output_dir, 'seg_01_result'),
        output_dir=join(output_dir, 'sent_prior_whole_jieba.pkl'),
        method_list=['jieba'])


    # 再算法
    topwords_seg(inputfile=inputfile_dir, filedir=output_dir,
                 output_dir=None, method_name=None,
                 model_path=None,
                 tau_l=tau_l, tau_f=tau_f,
                 seg_thr=0.5,
                 kappa_d=kappa_transfer(kappa_d),  # parameter estimation
                 kappa_s=kappa_transfer(kappa_s),  # segmetation
                 prior_words_dir=prior_words_dir,
                 protected_prior_words_dir=None,
                 sent_prior_dir=join(output_dir, 'sent_prior_whole_jieba.pkl'),
                 punctuations=punctuations,
                 num_of_processes=10)

if __name__ == '__main__':

    # requirements jieba, argparse

    # inputfile_dir = r'D:\PycharmProjects\SmartGyne\data\text\text_whole.txt'
    # punctuations_dir = r'D:\PycharmProjects\SmartGyne\data\text\punctuations.json'
    # prior_words_dir = r'D:\PycharmProjects\SmartGyne\data\text\prior_word.txt'
    # output_dir = r'D:\PycharmProjects\SmartGyne\data\seg_result'
    # kappa_d = 0.9
    # kappa_s = 0.001
    # tau_l = 15
    # tau_f = 5

    # python

    parser = argparse.ArgumentParser(description="Process text data and generate segmentation results.")
    parser.add_argument('--inputfile_dir', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--punctuations_dir', type=str, required=True, help='Path to the punctuations JSON file.')
    parser.add_argument('--prior_words_dir', type=str, required=True, help='Path to the prior words text file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the segmentation results.')
    parser.add_argument('--kappa_d', type=float, default=0.5,
                        help='Kappa value for parameter estimation (default: 0.5).')
    parser.add_argument('--kappa_s', type=float, default=0.001, help='Kappa value for segmentation (default: 0.001).')
    parser.add_argument('--tau_l', type=int, default=15, help='Tau value for length threshold (default: 15).')
    parser.add_argument('--tau_f', type=int, default=5, help='Tau value for frequency threshold (default: 5).')

    args = parser.parse_args()

    main(
        inputfile_dir=args.inputfile_dir,
        punctuations_dir=args.punctuations_dir,
        prior_words_dir=args.prior_words_dir,
        output_dir=args.output_dir,
        kappa_d=args.kappa_d,
        kappa_s=args.kappa_s,
        tau_l=args.tau_l,
        tau_f=args.tau_f
    )

