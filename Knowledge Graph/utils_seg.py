from typing import Tuple, List
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
from os.path import join
import json
from time import strftime, localtime
from nltk import FreqDist
from functools import partial

##################################################
# Part One: common utils

def print2(*args, file: str = 'log.txt', mode: str = 'a', add_time: bool = True,
           sep='\t', end='\n') -> None:
    if add_time:
        args = (strftime(f'%Y-%m-%d{sep}%H:%M:%S', localtime()), *args)
    print(*args, sep=sep, end=end)
    with open(file, mode=mode, encoding='utf-8') as f:
        print(*args, sep=sep, end=end, file=f)


def check_dir(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        print(f'creat file {file_dir}')
        return True
    else:
        return False


# 读取txt文件，自动识别编码
def read_txt_encoding(filename):
    encoding_list = ['utf8', 'gbk', 'gb18030']
    data = []
    for enc in encoding_list:
        try:
            with open(filename, 'r', encoding=enc) as f:
                data = f.readlines()
            print(filename, enc)
            break
        except:
            pass
    return data


def load_json(input_dir):
    with open(input_dir, encoding='utf8', errors='ignore') as jf:
        input_data = json.load(jf, strict=False)
    return input_data


def output_json(output_data, output_dir):
    with open(output_dir, 'w', encoding='utf8') as jf:
        json.dump(output_data, jf, ensure_ascii=False)


def is_chinese_character(char: str) -> bool:
    # import zhon.hanzi
    # zhon.hanzi.character
    # ord_char = ord(char)
    if len(char) != 1:
        raise ValueError

    if char == '\u3007':  # Ideographic number zero, see issue #17
        return True
    if '\u4E00' <= char <= '\u9FFF':  # CJK Unified Ideographs
        return True
    if '\u3400' <= char <= '\u4DBF':  # CJK Unified Ideographs Extension A
        return True
    if '\uF900' <= char <= '\uFAFF':  # CJK Compatibility Ideographs
        return True
    if '\U00020000' <= char <= '\U0002A6DF':  # CJK Unified Ideographs Extension B
        return True
    if '\U0002A700' <= char <= '\U0002B73F':  # CJK Unified Ideographs Extension C
        return True
    if '\U0002B740' <= char <= '\U0002B81F':  # CJK Unified Ideographs Extension D
        return True
    if '\U0002B820' <= char <= '\U0002CEAF':  # CJK Unified Ideographs Extension E
        return True
    if '\U0002CEB0' <= char <= '\U0002EBEF':  # CJK Unified Ideographs Extension F
        return True
    if '\U0002F800' <= char <= '\U0002FA1F':  # CJK Compatibility Ideographs Supplement
        return True

    return False


def text_distance(str1, str2):
    m, n = len(str1), len(str2)

    # 构建二维数组来存储子问题的答案
    dp = [[0 for _ in range(n + 1)] for x in range(m + 1)]

    # 利用动态规划算法，填充数组
    for i in range(m + 1):
        for j in range(n + 1):

            # 假设第一个字符串为空，则转换的代价为j(j次的插入)
            if i == 0:
                dp[i][j] = j

            # 同样，假设第二个字符串为空，则转换的代价为i(i次的插入)
            elif j == 0:
                dp[i][j] = i

            # 如果最后一个字符相等，就不会产生代价
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # 如果最后一个字符不一样，则考虑多种可能性，并且选择其中的最小值
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace
    return dp[m][n]


##################################################
# Part Two: utils in text analysis

def para_to_modify_sent(text_file: str = None,
                        text_string: str = None,
                        modify_json_file: str = None,
                        modify_text_file: str = None,
                        modify_punct_file: str = None):
    if text_file is None and text_string is None:
        raise ValueError('Both two input variables are None!')
    if text_file is not None and text_string is not None:
        raise ValueError('Both two input variables are not None!')

    if text_file:
        with open(text_file, encoding='utf8') as f:
            line_list = f.readlines()
    if text_string:
        line_list = text_string
    # 生成一个短句和标点一一对应的list：out；punct_list
    out = []
    punct_list = []
    punct = []
    label = []
    for line_ix, line in enumerate(line_list):
        sent = []
        # punct = []
        for char in line:
            if is_chinese_character(char):
                sent.append(char)
                if punct and out != []:
                    punct_list.append(''.join(punct))
                    punct = []
            else:
                punct.append(char)
                if sent:
                    out.append(''.join(sent))
                    label.append(line_ix)
                    sent = []
            # print(sent, punct)
    if punct != []:
        punct_list.append(''.join(punct))
        # punct_list.append(''.join(punct))

    # output modified json text
    if modify_json_file:
        import json
        with open(modify_json_file, 'w', encoding='utf8') as jf:
            json.dump(out, jf, ensure_ascii=False, indent=2)

    # output modified text
    if modify_text_file:
        with open(modify_text_file, 'w', encoding='utf8') as f:
            for row in out:
                _ = f.write(row + '\n')

    if modify_punct_file:
        import json
        with open(modify_punct_file, 'w', encoding='utf8') as jf2:
            json.dump(punct_list, jf2, ensure_ascii=False, indent=2)

    char_removed = set(''.join(punct_list))
    return out, punct_list, label, char_removed


def text_to_modify_text(text_file: str = None,
                        text_string: str = None,
                        modify_json_file: str = None,
                        modify_text_file: str = None) -> Tuple[List[str], set]:
    """
    Parameters
    ----------
    text_file: input text file
    text_string: input text string
    modify_json_file: output modified json file
    modify_text_file: output modified text file

    Returns
    -------
    out: list of sentences
    char_removed: set of removed characters

    """

    if text_file is None and text_string is None:
        raise ValueError('Both two input variables are None!')
    if text_file is not None and text_string is not None:
        raise ValueError('Both two input variables are not None!')

    if text_file:
        with open(text_file, encoding='utf8') as f:
            text_string = f.read()

    out = []
    punct = []
    out_temp = []
    char_removed = set()
    for char in text_string:
        if is_chinese_character(char):
            out_temp.append(char)
        else:
            char_removed |= {char}
            if out_temp:
                out.append(''.join(out_temp))
                out_temp = []
    if out_temp:
        out.append(''.join(out_temp))

    # output modified json text
    if modify_json_file:
        import json
        with open(modify_json_file, 'w') as jf:
            json.dump(out, jf)

    # output modified text
    if modify_text_file:
        with open(modify_text_file, 'w', encoding='utf8') as f:
            for row in out:
                _ = f.write(row + '\n')

    return out, char_removed


def seg_to_01(word_list):
    out = []
    for word in word_list:
        out.extend([0] * (len(word) - 1) + [1])
    return out[:-1]


def seg_to_01_reverse(seg_01, sent):
    # seg_01 to seg_sent
    out = []
    seg = np.where(seg_01)[0]  # array([ 1,  3,  5], dtype=int64)
    seg = np.append(seg, len(sent) - 1)
    start = 0
    for pos in seg:
        out.append(sent[start:(pos + 1)])
        start = pos + 1
    return out


def chunk_seg(seg, dep, label='ATT'):
    cluster_list = [[i] for i in range(1, len(seg) + 1)]
    wordx2clusterix = {i:i - 1 for i in range(1, len(seg) + 1)}
    bool_list = [True for _ in range(1, len(seg) + 1)]
    for item in dep:
        if item[2] == label and item[0] in wordx2clusterix.keys() and item[1] in wordx2clusterix.keys():
            ix1 = wordx2clusterix[item[0]]
            ix2 = wordx2clusterix[item[1]]
            cluster_list[ix1].extend(cluster_list[ix2])
            for wordix in cluster_list[ix2]:
                wordx2clusterix[wordix] = ix1
            bool_list[ix2] = False
    seg_chunk = []
    for wordix in range(1, len(seg) + 1):
        if wordix == 1:
            seg_chunk.append(seg[wordix-1])
        elif wordx2clusterix[wordix] == wordx2clusterix[wordix-1]:
            seg_chunk[-1] += seg[wordix-1]
        else:
            seg_chunk.append(seg[wordix - 1])
    return seg_chunk


def sent_to_seg(sent_list: List[str], method: str, user_dict_dir=None,
                modeldir='D:/pycharm/PycharmProjects/TopWORDS-Seg/models/stanford-corenlp-full-2018-10-05') -> List[List[str]]:
    """

    Parameters
    ----------
    sent_list
    method

    Returns
    -------

    Examples
    --------
    sent_list = ['我爱北京天安门', '南京市长江大桥']
    sent_to_seg(sent_list, method='jieba')
    [['我', '爱', '北京', '天安门'], ['南京市', '长江大桥']]
    """
    from tqdm import tqdm
    if method == 'jieba':
        # https://github.com/fxsjy/jieba
        import jieba
        if user_dict_dir:
            jieba.load_userdict(user_dict_dir)
        return [list(jieba.cut(sent)) for sent in tqdm(sent_list)]

    if method == 'pkuseg':
        # https://github.com/lancopku/pkuseg-python
        import pkuseg
        if user_dict_dir:
            seg = pkuseg.pkuseg(user_dict=user_dict_dir)
        else:
            seg = pkuseg.pkuseg()
        return [seg.cut(sent) for sent in tqdm(sent_list)]

    if method == 'thulac':
        # https://github.com/thunlp/THULAC-Python
        import thulac
        if user_dict_dir:
            thu1 = thulac.thulac(user_dict=user_dict_dir)
        else:
            thu1 = thulac.thulac(seg_only=True)
        return [[word for word, _ in thu1.cut(sent)]
                for sent in tqdm(sent_list)]

    if method == 'snownlp':
        # https://github.com/isnowfy/snownlp
        from snownlp import SnowNLP
        return [SnowNLP(sent).words for sent in tqdm(sent_list)]

    if method == 'stanfordnlp':
        from stanfordcorenlp import StanfordCoreNLP
        nlp = StanfordCoreNLP(modeldir, lang='zh')
        # nlp.__dict__
        seg_list = []
        from tqdm import tqdm
        for sent in tqdm(sent_list):
            try:
                seg_list.append(nlp.word_tokenize(sent))
            except:
                print(sent)
                seg_list.append([sent])
        return seg_list

    if method == 'stanfordnlp_chunk':
        from stanfordcorenlp import StanfordCoreNLP
        from tqdm import tqdm
        nlp = StanfordCoreNLP(modeldir, lang='zh')
        seg_list = []
        for sent in tqdm(sent_list):
            try:
                seg = nlp.word_tokenize(sent)
                dep = [(n1, n2, r) for r, n1, n2 in nlp.dependency_parse(sent)]
                seg_list.append(chunk_seg(seg, dep, 'compound:nn'))
            except:
                print(sent)
                seg_list.append([sent])
        return seg_list

    if method == 'ltp':
        from ltp import LTP
        from tqdm import tqdm
        # https://github.com/HIT-SCIR/ltp
        ltp = LTP()  # 默认加载 Small 模型， 从官网上下载（注意版本v3）
        # 模型下载地址 https://github.com/HIT-SCIR/ltp/blob/master/MODELS.md
        if user_dict_dir:
            word_list = read_txt_encoding(user_dict_dir)
            word_list = [word.strip() for word in word_list]
            ltp.add_words(words=word_list, max_window=15)
            # ltp.init_dict(path=user_dict_dir)
        seg_list = []
        max_size = 100
        for batch in tqdm(range(0, len(sent_list), max_size)):
            seg_tmp, _ = ltp.seg(sent_list[batch:min(batch + max_size, len(sent_list))])
            seg_list.extend(seg_tmp)
        return seg_list

    if method == 'ltp_chunk':
        from ltp import LTP
        from tqdm import tqdm
        ltp = LTP()
        if user_dict_dir:
            word_list = read_txt_encoding(user_dict_dir)
            word_list = [word.strip() for word in word_list]
            ltp.add_words(words=word_list, max_window=15)
        seg_list = []
        max_size = 100
        for batch in tqdm(range(0, len(sent_list), max_size)):
            seg_tmp, hidden = ltp.seg(sent_list[batch:min(batch + max_size, len(sent_list))])
            dep_tmp = ltp.dep(hidden)
            seg_chunk = [chunk_seg(seg, dep) for seg, dep in zip(seg_tmp, dep_tmp)]
            seg_list.extend(seg_chunk)
        return seg_list

    if method == 'stanza':
        import stanza
        stanza.download('zh', processors='tokenize')
        zh_nlp = stanza.Pipeline('zh', processors='tokenize')
        # 注：这里每个sent必须是一句话，不然stanza会有自动分句功能
        return [[token.text for sent in zh_nlp(sent).sentences for token in sent.tokens] for sent in tqdm(sent_list)]

    if method =='stanza_chunk':
        import stanza
        stanza.download('zh', processors='lemma, tokenize, pos, depparse')
        zh_nlp = stanza.Pipeline(lang='zh', processors='lemma, tokenize, pos, depparse')
        seg_list = []
        for sent in tqdm(sent_list):
            doc = zh_nlp(sent)  # 只有一句话
            seg = [[word.text for word in sent.words] for sent in doc.sentences][0]
            dep = [[(word.id, word.head, word.deprel) for word in sent.words] for sent in doc.sentences][0]
            seg_chunk = chunk_seg(seg, dep, 'compound')
            seg_list.append(seg_chunk)
        return seg_list

def divide_n(a, n):
    len_list = [int(len(a) * i / n) for i in range(n + 1)]
    return [a[len_list[i]:len_list[i + 1]] for i in range(n)]


def sent_to_seg_parrel(sent_list, method, user_dict_dir=None,
                       modeldir='D:/pycharm/PycharmProjects/TopWORDS-Seg/models/stanford-corenlp-full-2018-10-05',
                       num_of_processes=4):
    sent_to_seg_partial = partial(sent_to_seg, method=method, user_dict_dir=user_dict_dir, modeldir=modeldir)
    sent_list_cut = divide_n(sent_list, num_of_processes)
    with Pool(num_of_processes) as p:
        seg_list = p.map(sent_to_seg_partial, (sent_list for sent_list in sent_list_cut))
    seg_list2 = [s for seg in seg_list for s in seg]
    return seg_list2


def check_seg_list(seg_list: List[List[str]], sent_list: List[str]) -> bool:
    return all(
        len(''.join(seg)) == len(sent)
        for seg, sent in zip(seg_list, sent_list)
    )


def combine_seg_01_result(methods: List[str],
                          seg_01_result_dir: str) -> List:
    import json
    seg_01_count_list = None
    for method in methods:
        with open(join(seg_01_result_dir, f'{method}.json')) as jf:
            seg_01_list = [row for row in json.load(jf)]
        if seg_01_count_list == None:
            seg_01_count_list = [np.array(seg_01) for seg_01 in seg_01_list]
        else:
            seg_01_count_list = [np.array(seg_01_list[i]) + np.array(seg_01_count_list[i])
                                 for i in range(len(seg_01_list))]
    return seg_01_count_list


def combine_dict_result(methods: List[str],
                          dict_result_dir: str) -> List:
    combine_dict = set()
    for method in methods:
        dict_result = load_json(join(dict_result_dir, f'{method}.json'))
        combine_dict = combine_dict | set(dict_result.keys())
    return combine_dict


def get_input_text_file(inputfile):
    if len(inputfile) < 5:
        raise NameError('inputfile length too short')

    elif inputfile[-4:] == '.txt':
        with open(inputfile, encoding='utf-8') as f:
            data = f.readlines()
            sent_list = [d.strip() for d in data]
    elif inputfile[-5:] == '.json':
        with open(inputfile, encoding='utf-8') as jf:
            sent_list = json.load(jf)
    else:
        raise NameError('inputfile is not a txt or json file')
    return sent_list


def summary_text(sent_list, log_file):
    sent_num = len(sent_list)
    char_num = len(''.join(sent_list))
    print2(f'sent num: {sent_num}', file=log_file)
    print2(f'char num: {char_num}', file=log_file)


def check_result_dir(filedir):
    seg_result_dir = join(filedir, 'seg_result')
    seg_01_result_dir = join(filedir, 'seg_01_result')
    dict_result_dir = join(filedir, 'dict_result')
    model_result_dir = join(filedir, 'model')
    if not os.path.exists(seg_result_dir):
        os.makedirs(seg_result_dir)
    if not os.path.exists(seg_01_result_dir):
        os.makedirs(seg_01_result_dir)
    if not os.path.exists(dict_result_dir):
        os.makedirs(dict_result_dir)
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)
    return seg_result_dir, seg_01_result_dir, dict_result_dir


def get_result_dir(output_dir):
    seg_result_dir = join(output_dir, 'seg_result')
    seg_01_result_dir = join(output_dir, 'seg_01_result')
    dict_result_dir = join(output_dir, 'dict_result')
    eval_dir = join(output_dir, 'eval_result')
    for tmp_dir in [seg_result_dir, seg_01_result_dir, dict_result_dir, eval_dir]:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
    return seg_result_dir, seg_01_result_dir, dict_result_dir, eval_dir


def integrate_result(method, seg_list, sent_list, seg_result_dir, seg_01_result_dir, dict_result_dir):

    if check_seg_list(seg_list, sent_list):
        print('check seg list: pass; method: {}'.format(method))

        # seg result
        with open(join(seg_result_dir, f'{method}.txt'), 'w', encoding='utf-8') as f:
            for word_list in seg_list:
                _ = f.write(' '.join(word_list) + '\n')

        # seg 01 result
        seg_01_list = [seg_to_01(word_list) for word_list in seg_list]
        with open(join(seg_01_result_dir, f'{method}.json'), 'w', encoding='utf-8') as jf:
            json.dump(seg_01_list, jf)

        # word dict result
        word_dict = FreqDist(word
                             for word_list in seg_list
                             for word in word_list)
        with open(join(dict_result_dir, f'{method}.json'), 'w', encoding='utf-8') as jf:
            json.dump(word_dict, jf, ensure_ascii=False, indent=2)

    else:
        print('check seg list: NOT pass!!! ; method: {}'.format(method))


def get_uint_dtype(num: int) -> type:
    if num < 2 ** 8:
        return np.uint8
    if num < 2 ** 16:
        return np.uint16
    if num < 2 ** 32:
        return np.uint32
    if num < 2 ** 64:
        return np.uint64
    return np.uint64


def compare_seg_01_result(methods: List[str],
                          seg_01_result_dir: str,
                          output_xlsx_file: str = None):
    import json
    seg_01_result = dict()
    for method in methods:
        with open(seg_01_result_dir + method + '.json') as jf:
            seg_01_result[method] = [i
                                     for row in json.load(jf)
                                     for i in row]

    seg_01_result_matrix = np.zeros(
        (len(methods), len(seg_01_result[methods[0]])),
        dtype=get_uint_dtype(len(seg_01_result[methods[0]]))
    )

    for i, method in enumerate(methods):
        seg_01_result_matrix[i, :] = seg_01_result[method]
    seg_01_count = sum(seg_01_result_matrix)

    seg_01_result_compare = np.dot(seg_01_result_matrix, seg_01_result_matrix.transpose())
    if output_xlsx_file:
        seg_01_result_compare = pd.DataFrame(seg_01_result_compare, index=methods, columns=methods)
        seg_01_result_compare.to_excel(output_xlsx_file)
    return seg_01_result_compare, seg_01_count


def summary_dict_result(methods: List[str],
                        dict_result_dir: str,
                        output_csv_file: str = None) -> pd.DataFrame:
    import json
    dict_result = dict()
    for method in methods:
        with open(dict_result_dir + method + '.json') as jf:
            dict_result[method] = json.load(jf)

    word_set = {word for method in methods for word in dict_result[method].keys()}
    word_list = list(word_set)
    word_list.sort()
    word2ix = {word: ix for ix, word in enumerate(word_list)}
    max_freq = max(freq for method in methods for freq in dict_result[method].values())

    dict_result_summary = np.zeros(
        (len(methods), len(word_set)),
        dtype=get_uint_dtype(max_freq)
    )

    for i, method in enumerate(methods):
        for word, freq in dict_result[method].items():
            dict_result_summary[i, word2ix[word]] = freq

    dict_result_summary = pd.DataFrame(dict_result_summary.transpose(), index=word_list, columns=methods)
    if output_csv_file:
        dict_result_summary.to_csv(output_csv_file, sep='\t', header=True, index=True)
    return dict_result_summary


def compare_dict_result(dict_result_summary: pd.DataFrame,
                        output_xlsx_file: str = None) -> pd.DataFrame:
    methods = dict_result_summary.columns.tolist()

    dict_result_summary = dict_result_summary.values
    num_of_word_tokens = dict_result_summary.sum(axis=0).reshape(-1, 1)

    dict_result_summary_bool = dict_result_summary.astype(np.bool_)
    num_of_unique_words = dict_result_summary_bool.sum(axis=0).reshape(-1, 1)

    dict_result_summary_bool = dict_result_summary_bool.astype(
        get_uint_dtype(dict_result_summary_bool.shape[0])
    )
    dict_overlap = np.dot(dict_result_summary_bool.transpose(), dict_result_summary_bool)

    out = np.concatenate((num_of_word_tokens,
                          num_of_unique_words,
                          np.zeros(len(methods), dtype=np.uint8).reshape(-1, 1),
                          dict_overlap), axis=1)

    out = pd.DataFrame(out,
                       index=methods,
                       columns=['num_of_word_tokens',
                                'num_of_unique_words',
                                'blank'] + methods)
    if output_xlsx_file:
        out.to_excel(output_xlsx_file)
    return out


'''
def score(seg_01_list, gold_list):
    l1 = len(seg_01_list)
    l2 = len(gold_list)
    if l1 != l2:
        print('length not equal')
    else:
        count = np.zeros([2, 2])
        for i in range(l1):
            k1 = len(seg_01_list[i])
            k2 = len(gold_list[i])
            if k1 != k2:
                print('inner length not equal')
            else:
                for k in range(k1):
                    a = gold_list[i][k]
                    b = seg_01_list[i][k]
                    count[a,b] += 1

    recall = count[1, 1] / (count[1, 1] + count[1, 0])  # 正确的被判断为正确的比例
    precision = count[1, 1] / (count[1, 1] + count[0, 1])  # 预测为正且为正的比例
    total_rate = (count[1, 1] + count[0, 0]) / (count[1, 1] + count[0, 0] + count[1, 0] + count[0, 1])  # 总的正确的
    return recall, precision, total_rate
'''

from tqdm import tqdm
def get_sent_string_list_from_text_string(text_string, punctuations):
    sent_string = []
    for char in tqdm(text_string, desc='screen texts'):
        if char in punctuations:
            if sent_string:
                yield ''.join(sent_string)
                sent_string = []
            yield char
        else:
            sent_string.append(char)
    if sent_string:
        yield ''.join(sent_string)


def yield_char_from_text_file(text_file):
    with open(text_file, encoding='utf-8') as f:
        while True:
            char = f.read(1)
            if char:
                yield char
            else:
                break

# precision, recall
def transfer_full_list(seg_01_list):
    seg_full_list = []
    for seg_01 in seg_01_list:
        seg_full_list.extend(seg_01 + [1])
    return seg_full_list


def ngrams(sequence, n: int):
    # from nltk import ngrams
    sequence_len = len(sequence)
    for i in range(sequence_len - n + 1):
        yield sequence[i:(i+n)]


def transfer_word_pos_list(seg_01):
    tmp = [tuple(i) for i in ngrams(np.where([1] + seg_01)[0], 2)]
    return tmp


def seg_score(seg_01_list, gold_list):

    seg_pos_list = transfer_word_pos_list(transfer_full_list(seg_01_list))
    seg_gold_list = transfer_word_pos_list(transfer_full_list(gold_list))

    precision = len(set(seg_gold_list) & set(seg_pos_list))/len(set(seg_pos_list))
    recall = len(set(seg_gold_list) & set(seg_pos_list))/len(set(seg_gold_list))
    f1_score = 2 * precision * recall/(precision + recall)
    return precision, recall, f1_score


def seg_score_mask(seg_01_list, gold_list, mask_pos_list, mask_01_list):

    seg_pos_list = transfer_word_pos_list(transfer_full_list(seg_01_list))
    seg_gold_list = transfer_word_pos_list(transfer_full_list(gold_list))

    seg_pos_list_trun = [(pos_s, pos_e) for pos_s, pos_e in seg_pos_list
                         if not all([mask_01_list[ix] for ix in range(pos_s, pos_e)])]
    seg_gold_list_trun = set(seg_gold_list) - set(mask_pos_list)

    precision = len(set(seg_gold_list_trun) & set(seg_pos_list_trun)) / len(set(seg_pos_list_trun))
    recall = len(set(seg_gold_list_trun) & set(seg_pos_list_trun)) / len(set(seg_gold_list_trun))
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def get_mask_list(seg_01_full_list, gold_01_full_list):

    seg_pos_list = transfer_word_pos_list(seg_01_full_list)
    gold_pos_list = transfer_word_pos_list(gold_01_full_list)

    mask_pos_tw_list = []
    for pos_s, pos_e in seg_pos_list:
        if pos_s == 0:
            word_01 = [1] + gold_01_full_list[pos_s: pos_e]
        else:
            word_01 = gold_01_full_list[pos_s-1: pos_e]
        if word_01[0] == 1 and word_01[-1] == 1 and sum(word_01[1:-1]) > 0:
            mask_pos_tw_list.append((pos_s, pos_e))

    mask_01_list = [False for _ in range(len(gold_01_full_list))]  # 代表的是字的index
    for pos_s, pos_e in mask_pos_tw_list:
        for ix in range(pos_s, pos_e):
            mask_01_list[ix] = True

    mask_pos_list = [(pos_s, pos_e) for pos_s, pos_e in gold_pos_list
                     if all([mask_01_list[ix] for ix in range(pos_s, pos_e)])]

    # mask 百分比
    char_rate = sum(mask_01_list)/(len(gold_01_full_list))
    word_rate = len(mask_pos_list)/len(gold_pos_list)
    print(f'char_rate:{char_rate}')
    print(f'word_rate:{word_rate}')

    return mask_pos_list, mask_01_list, char_rate, word_rate


def modify_mask_seg_score(seg_01_list, gold_list):
    seg_01_full_list = transfer_full_list(seg_01_list)  # 最终长度是文本总字数
    gold_01_full_list = transfer_full_list(gold_list)

    mask_pos_list, mask_01_list, char_rate, word_rate = get_mask_list(seg_01_full_list, gold_01_full_list)
    precision, recall, f1_score = seg_score_mask(seg_01_list, gold_list, mask_pos_list, mask_01_list)

    return precision, recall, f1_score, mask_pos_list, mask_01_list, char_rate, word_rate