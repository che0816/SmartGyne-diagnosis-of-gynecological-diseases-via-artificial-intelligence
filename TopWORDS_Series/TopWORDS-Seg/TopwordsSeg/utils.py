from time import strftime, localtime
from typing import List

import numpy as np
from tqdm import tqdm

Char = str
Word = str


def get_int_dtype(num: int):
    if - 2 ** 8 <= num < 2 ** 8:
        return np.int8
    if - 2 ** 16 <= num < 2 ** 16:
        return np.int16
    if - 2 ** 32 <= num < 2 ** 32:
        return np.int32
    # if - 2 ** 64 <= num < 2 ** 64:
    #     return np.int64
    return np.int64


def get_uint_dtype(num: int):
    if num < 2 ** 8:
        return np.uint8
    if num < 2 ** 16:
        return np.uint16
    if num < 2 ** 32:
        return np.uint32
    # if num < 2 ** 64:
    #     return np.uint64
    return np.uint64


def generate_sentence(word_list: List[Word], word_num: int, theta_value: np.ndarray, word_count: np.ndarray) -> str:
    out = []
    while True:
        word_ix = np.random.choice(word_num, p=theta_value)
        word_count[word_ix] += 1
        out.append(word_list[word_ix])
        if word_ix == word_num - 1:
            break
    return ''.join(out)


def generate_file(word_list: List[str], word_num: int, theta_value: np.ndarray,
                  file_name: str, sentence_num: int) -> np.ndarray:
    word_count = np.zeros(word_num, np.uint64)
    with open(file_name, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(sentence_num), desc='generate sentences'):
            sentence = generate_sentence(word_list, word_num, theta_value, word_count)
            print(sentence, file=f)
    return word_count


def print2(*args, file: str = 'log.txt', mode: str = 'a', add_time: bool = True,
           sep='\t', end='\n') -> None:
    if add_time:
        args = (strftime(f'%Y-%m-%d{sep}%H:%M:%S', localtime()), *args)
    print(*args, sep=sep, end=end)
    with open(file, mode=mode, encoding='utf-8') as f:
        print(*args, sep=sep, end=end, file=f)


def print_error(*args, file: str = 'error.txt', mode: str = 'a',
                sep='\t', end='\n') -> None:
    args = (strftime(f'%Y-%m-%d{sep}%H:%M:%S', localtime()), *args)
    print(*args, sep=sep, end=end)
    with open(file, mode=mode, encoding='utf-8') as f:
        print(*args, sep=sep, end=end, file=f)


def is_chinese_character(char: Char) -> bool:
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


# def list_to_ndarray(sequences):
#     out = np.array(sequences)
#     dtype = get_uint_dtype(max(out[:, 0].max() + 1, out[:, 1:].max()))
#     return out.astype(dtype)


def ngrams(sequence, n: int):
    # from nltk import ngrams
    sequence_len = len(sequence)
    for i in range(sequence_len - n + 1):
        yield sequence[i:(i+n)]


def chinese_convert(string: str, conversion: str = 't2s') -> str:
    """
    https://pypi.org/project/opencc-python-reimplemented/
    https://github.com/yichen0831/opencc-python
    pip install opencc-python-reimplemented

    t2s - 繁体转简体（Traditional Chinese to Simplified Chinese）
    s2t - 简体转繁体（Simplified Chinese to Traditional Chinese）
    mix2t - 混合转繁体（Mixed to Traditional Chinese）
    mix2s - 混合转简体（Mixed to Simplified Chinese）

    Parameters
    ----------
    string
    conversion

    Returns
    -------

    Examples
    --------
    >>> chinese_convert('眾議長與李克強會談', conversion='t2s')
    '众议长与李克强会谈'
    >>> chinese_convert('开放中文转换', conversion='s2t')
    '開放中文轉換'
    """

    from opencc import OpenCC
    cc = OpenCC(conversion=conversion)
    return cc.convert(string)


def sent_to_seg(sent_list: List[str], method: str) -> List[List[str]]:
    """
    Examples
    --------
    >>> sent_list = ['我爱北京天安门', '南京市长江大桥']
    >>> sent_to_seg(sent_list, method='jieba')
    [['我', '爱', '北京', '天安门'], ['南京市', '长江大桥']]
    """

    if method == 'jieba':
        # https://github.com/fxsjy/jieba
        import jieba
        return [list(jieba.cut(sent)) for sent in tqdm(sent_list)]

    if method == 'snownlp':
        # https://github.com/isnowfy/snownlp
        from snownlp import SnowNLP
        return [SnowNLP(sent).words for sent in tqdm(sent_list)]

    if method == 'pkuseg':
        # https://github.com/lancopku/pkuseg-python
        import pkuseg
        seg = pkuseg.pkuseg()
        return [seg.cut(sent) for sent in tqdm(sent_list)]

    if method == 'thulac':
        # https://github.com/thunlp/THULAC-Python
        import thulac
        thu1 = thulac.thulac(seg_only=True)
        return [[word for word, _ in thu1.cut(sent)]
                for sent in tqdm(sent_list)]

    if method == 'hanlp':
        # https://github.com/hankcs/pyhanlp
        from pyhanlp import HanLP
        return [[term.word for term in HanLP.segment(sent)]
                for sent in tqdm(sent_list)]

    if method == 'pyltp':
        # https://github.com/HIT-SCIR/pyltp
        # xcode-select --install
        # https://github.com/HIT-SCIR/pyltp/issues/191
        # https://github.com/HIT-SCIR/pyltp/pull/193
        from pyltp import Segmentor
        segmentor = Segmentor()
        segmentor.load("/Users/jiazexu/PycharmProjects/AllSegmentation/ltp_data_v3.4.0/cws.model")
        out = [[word for word in segmentor.segment(sent)]
               for sent in tqdm(sent_list)]
        segmentor.release()
        return out

    if method == 'foolnltk':
        # https://github.com/rockyzhengwu/FoolNLTK
        import fool
        return fool.cut(sent_list)

