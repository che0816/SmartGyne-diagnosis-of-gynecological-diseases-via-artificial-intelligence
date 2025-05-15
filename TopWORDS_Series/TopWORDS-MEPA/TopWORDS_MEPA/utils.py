import os
from time import strftime, localtime
from typing import Tuple, List, Dict

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

Char = str
Category = str
Word = str
Collocation = Tuple[int, ...]


def mkdir_chdir(directory: str):
    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)


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


def generate_sentence(collocation_list: List[Collocation], collocation_num: int, rho_value: np.ndarray,
                      category_num: int,
                      word_list: List[Word], word_num: int, theta_value: np.ndarray,
                      collocation_count: np.ndarray, word_count: np.ndarray) -> str:
    out = []

    while True:
        collocation_ix = np.random.choice(collocation_num, p=rho_value)
        collocation_count[collocation_ix] += 1
        if collocation_ix == collocation_num - 1:
            break

        for tag_ix in collocation_list[collocation_ix]:

            if tag_ix < category_num:
                word_ix = np.random.choice(word_num, p=theta_value[tag_ix])
                word_count[tag_ix, word_ix] += 1
                out.append(word_list[word_ix])
            elif tag_ix > category_num:
                out.append(word_list[tag_ix - category_num - 1])
            else:
                raise ValueError

    return ''.join(out)


def generate_file(collocation_list: List[Collocation], collocation_num: int, rho_value: np.ndarray,
                  category_num: int,
                  word_list: List[str], word_num: int, theta_value: np.ndarray,
                  file_name: str, sentence_num: int) -> Tuple[np.ndarray, np.ndarray]:

    collocation_count = np.zeros(collocation_num, dtype=np.uint64)
    word_count = np.zeros((category_num, word_num), np.uint64)
    with open(file_name, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(sentence_num), desc='generate sentences'):
            sentence = generate_sentence(collocation_list, collocation_num, rho_value,
                                         category_num,
                                         word_list, word_num, theta_value,
                                         collocation_count, word_count)
            _ = f.write(sentence)
            _ = f.write('\n')

    return collocation_count, word_count


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
    assert isinstance(char, str)
    assert len(char) == 1

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
#     sequences = np.array(sequences)
#     dtype = get_uint_dtype(sequences.max())
#     return sequences.astype(dtype)


def pad_sequences(sequences):
    """
    from keras.preprocessing.sequence import pad_sequences
    """
    if sequences == []:
        return []
    row_len_s = [len(row) for row in sequences]
    max_len = max(row_len_s)
    max_value = max(max(row) for row in sequences)
    dtype = get_uint_dtype(max_value)

    out = np.zeros((len(sequences), max_len), dtype=dtype)
    for ix, (row, row_len) in enumerate(zip(sequences, row_len_s)):
        out[ix, :row_len] = row

    return out


def collocation_to_string(collocation: Collocation,
                          category_num: int, category_list: List[Category],
                          word_list: List[Word],
                          sep: str = '+') -> str:
    out = []
    for tag_ix in collocation:
        if tag_ix < category_num:
            out.append(category_list[tag_ix])
        elif tag_ix > category_num:
            out.append(word_list[tag_ix - category_num - 1])
        elif tag_ix == category_num:
            out.append('end')
        else:
            raise ValueError
    return sep.join(out)


def string_to_collocation(string: str,
                          category_num: int, category2ix: Dict[Category, int],
                          word2ix: Dict[Word, int],
                          sep: str = '+') -> Collocation:
    """
    Examples
    --------
    string = 'background+你+好+name'
    category_num = 2
    category2ix = {'background': 0, 'name': 1}
    word2ix = {'你': 0, '好': 1}
    string_to_collocation(string, category_num, category2ix, word2ix)
    """
    assert len(string) > 0
    out = []
    # for tag_str in string.split(sep):
    #     if tag_str in category2ix:
    #         out.append(category2ix[tag_str])
    #     elif tag_str in word2ix:
    #         out.append(word2ix[tag_str] + category_num + 1)
    #     elif tag_str == 'end':
    #         out.append(category_num)
    #     else:
    #         return tuple()
    #         # raise ValueError
    if string == 'end':
        out.append(category_num)
    else:
        for tag_str in string.split(sep):
            if len(tag_str) == 1:
                assert tag_str in word2ix
                out.append(word2ix[tag_str] + category_num + 1)
            else:
                assert tag_str in category2ix
                out.append(category2ix[tag_str])

    return tuple(out)


def ngrams(sequence, n: int):
    """
    from nltk import ngrams
    Examples
    --------
    >>> seq = 'abcdefg'
    >>> list(ngrams(seq, n=2))
    ['ab', 'bc', 'cd', 'de', 'ef', 'fg']
    """
    sequence_len = len(sequence)
    for i in range(sequence_len - n + 1):
        yield sequence[i:(i+n)]


def get_entities(seq: List[str]) -> List[Tuple[str, int, int]]:
    """
    from seqeval.metrics.sequence_labeling import get_entities
    
    Examples
    --------
    >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
    >>> get_entities(seq)
    [('PER', 0, 1), ('LOC', 3, 3)]
    """

    start = 0
    end = 0
    pre_bio = 'O'
    pre_category = ''
    chunks = []

    for i, tag in enumerate(seq + ['O']):
        if tag == 'O':
            bio = tag
            if pre_bio != 'O':
                chunks.append((pre_category, start, end))
        else:
            bio, category = tag.split('-')
            if bio == 'B':
                if pre_bio != 'O':
                    chunks.append((pre_category, start, end))
                pre_category = category
                start = i
            elif bio == 'I':
                if category != pre_category:
                    raise ValueError('Continued two tags are inconsistent!')
            else:
                raise ValueError('Invalid tag!')
            end = i
        pre_bio = bio

    return chunks


def classification_report(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    """
    from seqeval.metrics import classification_report

    Examples
    --------
    >>> y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'O']
    >>> y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'O']
    >>> print(classification_report(y_true, y_pred))
           precision  recall  f1-score  support
    MISC         0.0     0.0       0.0        1
    PER          1.0     1.0       1.0        1
    Total        0.5     0.5       0.5        2
    """
    # from seqeval.metrics import classification_report
    true_entities = defaultdict(set)
    for category, start, end in get_entities(y_true):
        true_entities[category].add((start, end))

    categories = list(true_entities.keys())
    categories.sort()

    pred_entities = defaultdict(set)
    for category, start, end in get_entities(y_pred):
        pred_entities[category].add((start, end))

    ps, rs, f1s, s = [], [], [], []
    for i, category in enumerate(categories):
        nb_correct = len(true_entities[category] & pred_entities[category])
        nb_pred = len(pred_entities[category])
        nb_true = len(true_entities[category])

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    ps.append(np.average(ps, weights=s))
    rs.append(np.average(rs, weights=s))
    f1s.append(np.average(f1s, weights=s))
    s.append(np.sum(s))

    return pd.DataFrame(
        {"precision": ps, "recall": rs, "f1-score": f1s, "support": s},
        index=categories + ['Total']
    )


