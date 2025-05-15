import pickle
from utils import print2
from TopWORDS import TopWORDS
from os.path import join
import os
from utils_seg import get_input_text_file, summary_text, check_result_dir, integrate_result
import json
import numpy as np
from utils_seg import read_txt_encoding

def save_model(T, model_path):
    with open(model_path, 'wb') as f:
        f.write(pickle.dumps(T))


def load_model(model_path):
    with open(model_path, 'rb') as f:
        T = pickle.loads(f.read())
    return T


def get_sent_prior_from_methods(seg_01_result_dir, output_dir,
                                method_list=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp']):
    sent_prior_dict = {}
    for method in method_list:
        with open(join(seg_01_result_dir, f'{method}.json')) as jf:
            seg_01_list = [row for row in json.load(jf)]
        seg_01_list = [seg_01 + [1.0] for seg_01 in seg_01_list]  # 长度和句子长度保持一致
        sent_prior_dict[method] = seg_01_list
    with open(output_dir, 'wb') as f:
        f.write(pickle.dumps(sent_prior_dict))


def get_sent_prior_list(sent_prior_dict, kappa, rho=0.5):

    method_list = list(sent_prior_dict.keys())
    sent_size_list = [len(seg_01) for seg_01 in sent_prior_dict[method_list[0]]]
    sent_count_list = [np.zeros(size) for size in sent_size_list]
    for method in method_list:
        sent_prior_list_tmp = sent_prior_dict[method]
        tmp = [len(seg_01) for seg_01 in sent_prior_list_tmp]
        assert tmp == sent_size_list, "sent_prior_list not consistent"
        sent_count_list = [sent_count + np.array(sent_prior)
                           for sent_count, sent_prior, in zip(sent_count_list, sent_prior_list_tmp)]
    if kappa == float('inf'):
        sent_prior_list = [np.ones(len(seg_count)) * rho for seg_count in sent_count_list]
    else:
        sent_prior_list = [(seg_count + kappa * rho) / (len(method_list) + kappa) for seg_count in sent_count_list]
    for sent_prior in sent_prior_list:
        sent_prior[-1] = 1.0
    return sent_prior_list


def get_prior_word_freq(prior_words_dir):
    dictionary = read_txt_encoding(prior_words_dir)
    word_freq = {}
    for line in dictionary:
        line2 = line.strip().split(' ')
        # print(line2)
        assert len(line2) in [1, 2], f'read prior dict format wrong: {line2} length: {len(line2)}'
        word = line2[0]
        freq = 0
        if len(line2) == 2:
            try:
                freq = float(line2[1])
            except:
                print(f'{line2} not standard format!')
        word_freq[word] = freq
    return word_freq


def topwords_seg(inputfile, filedir,
                 output_dir=None, method_name=None,
                 model_path=None,
                 tau_l=8, tau_f=2,
                 seg_thr=0.5,
                 kappa_d=0.01,  # parameter estimation
                 kappa_s=0.01,  # segmetation
                 prior_words_dir=None,
                 protected_prior_words_dir=None,
                 delete_words_dir=None,
                 sent_prior_dir=None,
                 is_output_dictionary=True,
                 punctuations={'，', '。', '！', '？', '“', '：', '\n'},
                 num_of_processes=1):
    if not method_name:  # 模拟起名需要
        tmp1 = '%.2g' % (kappa_d)
        tmp2 = '%.2g' % (kappa_s)
        method_name = f'topwords_seg_{tmp1}_{tmp2}'
    if not output_dir:
        output_dir = join(filedir, method_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not model_path:
        model_path = join(filedir, 'model')
    origin_dir = os.getcwd()

    # 日志文件
    log_file = join(output_dir, 'log.txt')
    print2('-' * 100, mode='w', add_time=False, file=log_file)
    print2('Welcome to use TopWORDS-Seg program developed by Changzai Pan (panpanaqm@126.com)', file=log_file)
    print2('-' * 100, add_time=False)
    print2(f'inputfile: {inputfile}', file=log_file, add_time=False)
    print2(f'filedir: {filedir}', file=log_file, add_time=False)
    print2(f'method_name: {method_name}', file=log_file, add_time=False)
    print2(f'output_dir: {output_dir}', file=log_file, add_time=False)
    print2(f'tau_l: {tau_l}', file=log_file, add_time=False)
    print2(f'tau_f: {tau_f}', file=log_file, add_time=False)
    print2(f'seg_thr: {seg_thr}', file=log_file, add_time=False)
    print2(f'kappa_d: {kappa_d}', file=log_file, add_time=False)
    print2(f'kappa_s: {kappa_s}', file=log_file, add_time=False)
    print2(f'prior_words_dir: {prior_words_dir}', file=log_file, add_time=False)
    print2(f'protected_prior_words_dir: {protected_prior_words_dir}', file=log_file, add_time=False)
    print2(f'delete_words_dir: {delete_words_dir}', file=log_file, add_time=False)
    print2(f'sent_prior_dir: {sent_prior_dir}', file=log_file, add_time=False)
    print2(f'num_of_processes: {num_of_processes}', file=log_file, add_time=False)

    # 读取数据
    print2(f'load text file', file=log_file)
    sent_list = get_input_text_file(inputfile)  # 读取文本数据
    summary_text(sent_list, log_file)
    prior_word_freq = get_prior_word_freq(prior_words_dir)
    # prior_words = set(get_input_text_file(prior_words_dir) if prior_words_dir else set())
    protected_prior_words = set(get_input_text_file(protected_prior_words_dir) if protected_prior_words_dir else set())
    delete_words = set(get_input_text_file(delete_words_dir) if delete_words_dir else set())

    # 检查输出目录
    print2(f'check result dir', file=log_file)
    seg_result_dir, seg_01_result_dir, dict_result_dir = check_result_dir(filedir)  # 检查各个目录

    # 先验分词信息处理
    if sent_prior_dir:
        with open(sent_prior_dir, 'rb') as f:
            sent_prior_dict = pickle.loads(f.read())
        sent_dict_prior_list = get_sent_prior_list(sent_prior_dict, kappa_d)
    else:
        sent_prior_dict = None
        sent_dict_prior_list = None

    # TopWORDS_Seg
    model_name = f'tw_{kappa_d}_{tau_l}_{tau_f}.pkl'
    model_dir = join(model_path, model_name)
    if model_path and os.path.exists(model_dir):
        print2(f'load model {model_dir}', file=log_file)
        T = load_model(model_dir)
        os.chdir(output_dir)
    else:
        os.chdir(output_dir)
        T = TopWORDS(text_file=inputfile,
                     prior_word_freq=prior_word_freq, protected_prior_words=protected_prior_words,
                     delete_words=delete_words,
                     punctuations=punctuations,
                     sent_prior_list=sent_dict_prior_list,
                     word_max_len_for_screening=tau_l, word_min_freq_for_screening=tau_f,
                     num_of_processes=num_of_processes)
        T.em_update(prune_by_score_iteration_num = 1, em_para_threshold=1e-6)
        print2(f'save model {model_dir}', file=log_file)
        save_model(T, model_dir)

    if is_output_dictionary:
        T.output_dictionary_result(is_compute_score=True, is_compute_posterior=True)
    if sent_prior_dict:
        sent_seg_prior_list = get_sent_prior_list(sent_prior_dict, kappa_s)
        print(sent_seg_prior_list[0])
        T.set_sent_prior_in_sentences(sent_seg_prior_list)
    modify_sent_list = T.output_sent('modify_text.txt')
    seg_list = T.output_decoded_result(seg_thr=seg_thr)

    print2(f'integrate result of method {method_name}', file=log_file)
    os.chdir(origin_dir)
    integrate_result(method_name, seg_list, modify_sent_list, seg_result_dir, seg_01_result_dir, dict_result_dir)

    print2('finish', file=log_file)


if __name__ == '__main__':

    data_dir = os.path.join('D:\pycharm\PycharmProjects\TopWORDS-Seg', 'RDA', 'test', 'data')
    inputfile = os.path.join(data_dir, 'modify_text.txt')
    filedir = os.path.join('D:\pycharm\PycharmProjects\TopWORDS-Seg', 'RDA', 'test', 'output')
    sent_prior_dir = os.path.join(filedir, 'sent_prior_dict.pkl')

    get_sent_prior_from_methods(
        seg_01_result_dir=r'D:\pycharm\PycharmProjects\TopWORDS-Seg\RDA\test\output\seg_01_result',
        output_dir=r'D:\pycharm\PycharmProjects\TopWORDS-Seg\RDA\test\output\sent_prior_dict.pkl',
        method_list=['stanfordnlp', 'jieba', 'thulac', 'pkuseg', 'snownlp', 'ltp'])

    topwords_seg(inputfile=inputfile, filedir=filedir,
                 output_dir=None, method_name=None,
                 model_path=None,
                 tau_l=8, tau_f=2,
                 seg_thr=0.5,
                 kappa_d=0.01,  # parameter estimation
                 kappa_s=0.01,  # segmetation
                 prior_words_dir=None,
                 protected_prior_words_dir=None,
                 delete_words_dir=None,
                 sent_prior_dir=sent_prior_dir,
                 num_of_processes=1)
