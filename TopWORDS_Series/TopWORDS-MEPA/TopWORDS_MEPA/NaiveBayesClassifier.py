
import csv
import typing
from itertools import chain
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

from TopWORDS_MEPA.utils import Char, Category, Word


class NaiveBayesClassifier:

    @staticmethod
    def get_word_len_dist_and_char_freq_dist(
            technical_terms: Dict[Category, List[Word]],
            category_list: List[Category], category_num: int, category2ix: Dict[Category, int],
            word_max_len: int,
            char_count: typing.Counter[Char], char_list: List[Char], char_num: int, char2ix: Dict[Char, int],
            flat_prior_for_word_len_dist: float = 0.01, flat_prior_for_char_freq_dist: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:

        # count
        word_len_count = np.zeros((category_num, word_max_len), dtype=np.uint64)
        char_freq_count = np.zeros((category_num, char_num), dtype=np.uint64)

        for category, word_list in technical_terms.items():
            category_ix = category2ix[category]
            for word in tqdm(word_list, desc='screen words in ' + category + ' category'):
                word_len = len(word)
                if word_len <= word_max_len:
                    word_len_count[category_ix, word_len - 1] += 1
                for char in word:
                    if char in char2ix:
                        char_ix = char2ix[char]
                        char_freq_count[category_ix, char_ix] += 1

        # add char count into char_freq_dist[0]
        if char_freq_count[0].sum() == 0:
            for char, count in char_count.items():
                char_ix = char2ix[char]
                char_freq_count[0, char_ix] = count

        # modify non-updated row in word_len_dist
        for category_ix, row in enumerate(word_len_count):
            if row.sum() == 0:
                for word_len in range(1, word_max_len + 1):
                    word_len_count[category_ix, word_len - 1] = 2 ** (word_max_len - word_len)

        # distribution
        word_len_dist = word_len_count.astype(np.float64)
        char_freq_dist = char_freq_count.astype(np.float64)

        # normalization
        for category_ix, row in enumerate(word_len_dist):
            row_sum = row.sum()
            if row_sum == 0:
                row_sum = 1
            row *= (1 - flat_prior_for_word_len_dist) / row_sum
            row += flat_prior_for_word_len_dist / word_max_len

        for category_ix, row in enumerate(char_freq_dist):
            row_sum = row.sum()
            if row_sum == 0:
                row_sum = 1
            row *= (1 - flat_prior_for_char_freq_dist) / row_sum
            row += flat_prior_for_char_freq_dist / char_num

        # output
        with open('word_len_dist.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = chain(['word_len'],
                        (f'{category}_count' for category in category_list),
                        (f'{category}_distribution' for category in category_list))
            writer.writerow(row)
            for word_len in tqdm(range(1, word_max_len + 1), desc='screen word lengths'):
                row = chain([word_len], word_len_count[:, word_len - 1], word_len_dist[:, word_len - 1])
                writer.writerow(row)

        with open('char_freq_dist.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = chain(['char'],
                        (f'{category}_count' for category in category_list),
                        (f'{category}_distribution' for category in category_list))
            writer.writerow(row)
            for char_ix, char in enumerate(tqdm(char_list, desc='screen characters')):
                row = chain([char], char_freq_count[:, char_ix], char_freq_dist[:, char_ix])
                writer.writerow(row)

        return word_len_dist, char_freq_dist

    def __init__(self,
                 technical_terms: Dict[Category, List[Word]],
                 word_max_len: int,
                 category_list: List[Category], category_num: int, category2ix: Dict[Category, int],
                 char_count: typing.Counter[Char], char_list: List[Char], char_num: int, char2ix: Dict[Char, int],
                 min_prob_in_nb_smooth: float,
                 category2prior: Dict[Category, int] = None):
        self.min_prob_in_nb_smooth = min_prob_in_nb_smooth

        if category2prior is None:
            self.prior = np.ones(category_num, dtype=np.float64) / category_num
        else:
            self.prior = np.array([category2prior[category] for category in category_list], dtype=np.float64)
            self.prior /= self.prior.sum()

        self.word_len_dist, self.char_freq_dist = self.get_word_len_dist_and_char_freq_dist(
            technical_terms,
            category_list, category_num, category2ix,
            word_max_len,
            char_count, char_list, char_num, char2ix,
            min_prob_in_nb_smooth, min_prob_in_nb_smooth
        )

        self.category_num = len(category_list)
        self.category2ix = category2ix
        self.char2ix = char2ix

        # self.thresholds = np.ones(category_num - 1, dtype=np.float64) * 50

    def get_post(self, word: Word) -> np.ndarray:
        word_len = len(word)

        posterior = self.prior * self.word_len_dist[:, word_len - 1]

        char_ixs = np.fromiter((self.char2ix[char] for char in word), dtype=np.uint64)
        posterior *= (self.char_freq_dist[:, char_ixs].prod(axis=1)) ** (1 / word_len)

        # 使最小值为 self.min_prob_in_nb_smooth
        posterior *= (1 - self.min_prob_in_nb_smooth * self.category_num) / posterior.sum()
        posterior += self.min_prob_in_nb_smooth
        return posterior

    @staticmethod
    def get_hard_pred_from_post(posterior: np.ndarray, screen_tt_threshold: float = 0.5) -> np.ndarray:

        predict = (posterior >= screen_tt_threshold)
        # predict = (posterior == posterior.max())
        predict[0] = False
        return predict

    def get_soft_pred_from_post(self, posterior: np.ndarray, num_of_open_categories_of_a_word: int) -> np.ndarray:

        if num_of_open_categories_of_a_word >= self.category_num:
            return np.ones(self.category_num, dtype=np.bool_)

        ix = np.argpartition(posterior, -num_of_open_categories_of_a_word)[-num_of_open_categories_of_a_word:]
        out = np.zeros(len(posterior), dtype=np.bool_)
        out[ix] = True
        out[0] = True
        return out
        # predict = posterior > 0.2
        # predict[0] = True
        # if predict.sum() == 1:
        #     predict[np.argsort(predict)[-2]] = True
        #
        # return predict
        # return np.ones(len(posterior), dtype=np.bool_)
