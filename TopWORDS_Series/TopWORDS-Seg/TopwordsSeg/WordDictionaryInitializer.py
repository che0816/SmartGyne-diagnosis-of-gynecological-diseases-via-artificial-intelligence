import csv
from collections import deque, Counter
from typing import Set
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Word, get_uint_dtype, ngrams, print2


class WordDictionaryInitializer:
    @staticmethod
    def get_complete_word_dict(sentences: deque,
                               prior_words: Set[Word],
                               delete_words: Set[Word],
                               word_max_len_for_screening: int,
                               word_min_freq_for_screening: int) -> pd.DataFrame:
        print2('---', add_time=False)
        char_count = Counter(char
                             for sent in tqdm(sentences, desc=f'screen sentences (1/{word_max_len_for_screening})')
                             if sent.bool
                             for char in sent.sent_string)
        print2(f'num of single char: {len(char_count)}', add_time=False)
        print2('---', add_time=False)
        print2('word_len\tword_num\tsubtotal', add_time=False)
        word_count = {char: count
                      for char, count in char_count.items()
                      if count >= word_min_freq_for_screening}
        print2(f'1\t{len(word_count)}\t{len(word_count)}', add_time=False)

        for word_len in range(2, word_max_len_for_screening + 1):
            word_count_temp = Counter(
                word
                for sent in tqdm(sentences, desc=f'screen sentences ({word_len}/{word_max_len_for_screening})')
                if sent.bool
                for word in ngrams(sent.sent_string, word_len)
                if word[1:] in word_count and word[:-1] in word_count)
            word_count_temp = {word: count
                               for word, count in word_count_temp.items()
                               if count >= word_min_freq_for_screening}
            word_count.update(word_count_temp)
            print2(f'{word_len}\t{len(word_count_temp)}\t{len(word_count)}', add_time=False)
            if len(word_count_temp) == 0:
                break
        print2('---', add_time=False)
        word_count.update(char_count)
        additional_prior_words = prior_words - word_count.keys()
        word_count_temp = Counter()
        if additional_prior_words:
            word_max_len_in_prior = max(len(word) for word in additional_prior_words)
            word_count_temp = Counter(
                word
                for sent in tqdm(sentences, desc='screen sentences (for prior words)')
                if sent.bool
                for word_len in range(1, word_max_len_in_prior + 1)
                for word in ngrams(sent.sent_string, word_len)
                if word in additional_prior_words)
            word_count.update(word_count_temp)
        print2(f'num of additional prior words: {len(word_count_temp)}', add_time=False)
        print2('---', add_time=False)

        delete_words = {word for word in delete_words if word not in prior_words and len(word) > 1}
        word_count = {word: count for word, count in word_count.items() if word not in delete_words}
        print2(f'num of delete words: {len(delete_words)}', add_time=False)
        print2('---', add_time=False)
        print2(f'num of total words: {len(word_count)}', add_time=False)
        print2('---', add_time=False)

        complete_word_dict = pd.DataFrame.from_dict(word_count, orient='index', columns=['count'])
        complete_word_dict.sort_values(by=['count'], ascending=False, inplace=True)
        complete_word_dict.index.rename('word', inplace=True)
        complete_word_dict = complete_word_dict.reset_index()

        dtype = get_uint_dtype(complete_word_dict['count'].max())
        complete_word_dict['count'] = complete_word_dict['count'].astype(dtype)

        complete_word_dict['word_len'] = np.fromiter((len(word) for word in complete_word_dict['word']),
                                                     dtype=np.uint8, count=len(complete_word_dict))

        complete_word_dict['is_single_char'] = (complete_word_dict['word_len'] <= 1)

        complete_word_dict['is_screened_word'] = (
                (complete_word_dict['count'] >= word_min_freq_for_screening)
                & (complete_word_dict['word_len'] <= word_max_len_for_screening)
        )

        complete_word_dict['is_prior_word'] = complete_word_dict['word'].isin(prior_words)

        with open('word_count.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(complete_word_dict.columns)
            for row in tqdm(complete_word_dict.values, desc='screen words'):
                writer.writerow(row)

        return complete_word_dict

    def __init__(self, sentences: deque, prior_words: Set[Word], delete_words,
                 word_max_len_for_screening: int = 3, word_min_freq_for_screening: int = 100):

        self.complete_word_dict = self.get_complete_word_dict(
            sentences, prior_words, delete_words, word_max_len_for_screening, word_min_freq_for_screening)
