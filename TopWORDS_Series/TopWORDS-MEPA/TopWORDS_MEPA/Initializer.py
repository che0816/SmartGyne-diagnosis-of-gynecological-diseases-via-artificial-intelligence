
import csv
import typing
from collections import deque, Counter
from itertools import chain
from typing import List, Tuple, Dict, Set, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from TopWORDS_MEPA.utils import Char, Category, Word, Collocation
from TopWORDS_MEPA.utils import get_uint_dtype
from TopWORDS_MEPA.utils import collocation_to_string
from TopWORDS_MEPA.utils import ngrams
from TopWORDS_MEPA.utils import print2, print_error
from TopWORDS_MEPA.NaiveBayesClassifier import NaiveBayesClassifier


class WordDictionaryInitializer:
    @staticmethod
    def get_char_info(sentences: deque) -> Tuple[typing.Counter[Char], List[Char], int, Dict[Char, int]]:
        char_count = Counter(char
                             for sent in tqdm(sentences, desc='screen sentences')
                             if sent.bool
                             for char in sent.sent_string)
        char_list = list(char_count.keys())
        char_list.sort()
        char_num = len(char_list)
        char2ix = {char: ix for ix, char in enumerate(char_list)}

        return char_count, char_list, char_num, char2ix

    @staticmethod
    def get_category_info(
            technical_terms: Dict[Category, List[Word]]) -> Tuple[List[Category], int, Dict[Category, int]]:

        category_set = set(technical_terms.keys())
        category_set -= {'background'}
        category_list = ['background'] + sorted(category_set)
        category_num = len(category_list)

        category2ix = {category: ix for ix, category in enumerate(category_list)}

        return category_list, category_num, category2ix

    def get_complete_word_dict_part_1(self, sentences: deque, prior_word_info: Dict[Word, Union[List[Category], str]],
                                      word_max_len_for_screening: int,
                                      word_min_freq_for_screening: int) -> int:

        print2('---', add_time=False)
        print2(f'num of single char: {len(self.char_count)}', add_time=False)
        print2('---', add_time=False)
        print2('word_len\tword_num\tsubtotal', add_time=False)
        word_count = {char: count
                      for char, count in self.char_count.items()
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
        word_count.update(self.char_count)
        additional_prior_words = {word
                                  for word, category_list in prior_word_info.items()
                                  if isinstance(category_list, list)} - word_count.keys()
        if additional_prior_words:
            word_max_len_in_prior = max(len(word) for word in additional_prior_words)
            word_count_temp = Counter(
                word
                for sent in tqdm(sentences, desc='screen sentences (for prior words)')
                if sent.bool
                for word_len in range(1, word_max_len_in_prior + 1)
                for word in ngrams(sent.sent_string, word_len)
                if word in additional_prior_words)
            print2(f'num of additional prior words: {len(word_count_temp)}', add_time=False)
            word_count.update(word_count_temp)

        print2('---', add_time=False)
        self.word_count = word_count
        print2(f'num of total words: {len(self.word_count)}', add_time=False)
        print2('---', add_time=False)

        return max(len(word) for word in word_count)

    def get_complete_word_dict_part_2(self, sentences: deque,
                                      word_max_len_for_screening: int,
                                      word_min_freq_for_screening: int,
                                      word_max_len_for_screening_tt: int,
                                      word_min_freq_for_screening_tt: int,
                                      screen_tt_threshold: float) -> Set[Word]:
        if word_max_len_for_screening_tt is None or word_min_freq_for_screening_tt is None:
            return set()
        if (word_max_len_for_screening_tt <= word_max_len_for_screening
                and word_min_freq_for_screening_tt >= word_min_freq_for_screening):
            return set()

        print2('---', add_time=False)
        print2('word_len\tword_num\tsubtotal', add_time=False)
        word_count = {char: count
                      for char, count in self.char_count.items()
                      if count >= word_min_freq_for_screening_tt}
        print2(f'1\t{len(word_count)}\t{len(word_count)}', add_time=False)

        for word_len in range(2, word_max_len_for_screening_tt + 1):
            word_count_temp = Counter(
                word
                for sent in tqdm(sentences, desc=f'screen sentences ({word_len}/{word_max_len_for_screening_tt})')
                if sent.bool
                for word in ngrams(sent.sent_string, word_len)
                if word[1:] in word_count and word[:-1] in word_count)
            word_count_temp = {word: count
                               for word, count in word_count_temp.items()
                               if count >= word_min_freq_for_screening_tt}
            word_count.update(word_count_temp)
            print2(f'{word_len}\t{len(word_count_temp)}\t{len(word_count)}', add_time=False)
            if len(word_count_temp) == 0:
                break
        print2('---', add_time=False)

        word_count = {word: count
                      for word, count in word_count.items()
                      if self.word_classifier.get_hard_pred_from_post(self.word_classifier.get_post(word),
                                                                      screen_tt_threshold).any()}
        screened_tt_set = set(word_count.keys())
        print2(f'num of screened technical terms: {len(word_count)}', add_time=False)

        print2('---', add_time=False)
        self.word_count.update(word_count)
        print2(f'num of total words: {len(self.word_count)}', add_time=False)
        print2('---', add_time=False)
        return screened_tt_set

    def get_complete_word_dict_part_3(self, prior_word_info: Dict[Word, Union[List[Category], str]]):
        out = set()
        for word, value in prior_word_info.items():
            if isinstance(value, str):
                assert value == 'delete'
                if word in self.word_count:
                    out.add(word)
        print2('---', add_time=False)
        print2(f'num of deleted words: {len(out)}', add_time=False)
        for word in out:
            self.word_count.pop(word)
        print2(f'num of total words: {len(self.word_count)}', add_time=False)
        print2('---', add_time=False)

    def output_complete_word_dict(self, prior_word_info,
                                  word_max_len_for_screening, word_min_freq_for_screening,
                                  screened_tt_set):
        complete_word_dict = pd.DataFrame.from_dict(self.word_count, orient='index', columns=['count'])
        complete_word_dict.index.rename('word', inplace=True)
        complete_word_dict = complete_word_dict.reset_index()
        complete_word_dict.sort_values(by=['count', 'word'], ascending=False, inplace=True)

        dtype = get_uint_dtype(complete_word_dict['count'].max())
        complete_word_dict['count'] = complete_word_dict['count'].astype(dtype)

        complete_word_dict['word_len'] = np.fromiter((len(word) for word in complete_word_dict['word']),
                                                     dtype=np.uint64, count=len(complete_word_dict))
        dtype = get_uint_dtype(complete_word_dict['word_len'].max())
        complete_word_dict['word_len'] = complete_word_dict['word_len'].astype(dtype)

        complete_word_dict['is_single_char'] = (complete_word_dict['word_len'] <= 1)

        complete_word_dict['is_screened_word'] = (
                (complete_word_dict['count'] >= word_min_freq_for_screening)
                & (complete_word_dict['word_len'] <= word_max_len_for_screening)
        )

        complete_word_dict['is_screened_technical_term'] = complete_word_dict['word'].isin(screened_tt_set)

        complete_word_dict['is_prior_word'] = complete_word_dict['word'].isin(prior_word_info)

        return complete_word_dict

    def word_classification(self,
                            prior_word_info: Dict[Word, List[Category]],
                            prior_word_category_setting: str,
                            screen_tt_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        word2ix = {word: ix for ix, word in enumerate(self.complete_word_dict['word'])}
        posterior_matrix = np.empty((len(self.complete_word_dict.index), self.category_num), dtype=np.float64)
        for word_ix, word in enumerate(tqdm(self.complete_word_dict['word'], desc='screen words')):
            posterior_matrix[word_ix, :] = self.word_classifier.get_post(word)

        hard_predict_matrix = np.empty_like(posterior_matrix, dtype=np.bool_)
        for word_ix, posterior in enumerate(tqdm(posterior_matrix, desc='screen posterior distributions')):
            hard_predict_matrix[word_ix, :] = self.word_classifier.get_hard_pred_from_post(posterior,
                                                                                           screen_tt_threshold)

        soft_predict_matrix = np.empty_like(posterior_matrix, dtype=np.bool_)
        for word_ix, posterior in enumerate(tqdm(posterior_matrix, desc='screen posterior distributions')):
            soft_predict_matrix[word_ix, :] = self.word_classifier.get_soft_pred_from_post(
                posterior, self.num_of_open_categories_of_a_word)

        soft_predict_matrix |= hard_predict_matrix

        if prior_word_category_setting == 'add':
            for word, categories in prior_word_info.items():
                if word in word2ix:
                    for category in categories:
                        hard_predict_matrix[word2ix[word], self.category2ix[category]] = True
                        soft_predict_matrix[word2ix[word], self.category2ix[category]] = True
        elif prior_word_category_setting == 'dominate':
            for word, categories in prior_word_info.items():
                if word in word2ix:
                    word_ix = word2ix[word]
                    if categories:
                        for category_ix, category in enumerate(self.category_list):
                            if category in categories:
                                hard_predict_matrix[word_ix, category_ix] = True
                                soft_predict_matrix[word_ix, category_ix] = True
                            else:
                                hard_predict_matrix[word_ix, category_ix] = False
                                soft_predict_matrix[word_ix, category_ix] = False
        else:
            print_error("wrong prior_word_category_setting value")
        sparse_posterior = posterior_matrix * soft_predict_matrix
        for row in sparse_posterior:
            row /= row.sum()

        with open('word_classification.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = chain(self.complete_word_dict.columns,
                        [f'posterior {category}' for category in self.category_list],
                        [f'hard_predict {category}' for category in self.category_list],
                        [f'soft_predict {category}' for category in self.category_list],
                        [f'sparse_posterior {category}' for category in self.category_list])
            writer.writerow(row)
            for ix, row in enumerate(tqdm(self.complete_word_dict.values, desc='screen words')):
                row_2 = chain(row,
                              posterior_matrix[ix], hard_predict_matrix[ix],
                              soft_predict_matrix[ix], sparse_posterior[ix])
                writer.writerow(row_2)

        return hard_predict_matrix.transpose(), soft_predict_matrix.transpose(), sparse_posterior.transpose()

    def __init__(self, technical_terms: Dict[Category, List[Word]], sentences: deque,
                 prior_info: dict,
                 word_max_len_for_screening: int = 3, word_min_freq_for_screening: int = 100,
                 word_max_len_for_screening_tt: int = None,
                 word_min_freq_for_screening_tt: int = None,
                 screen_tt_threshold: float = 0.5,
                 screen_collo_tt_threshold: float = 0.5,
                 num_of_open_categories_of_a_word: int = None,
                 min_prob_in_nb_smooth: float = None):

        # char
        print2("get char info")
        self.char_count, self.char_list, self.char_num, self.char2ix = self.get_char_info(sentences)
        print2("get char info, DONE!")

        # category
        print2("get category info")
        self.category_list, self.category_num, self.category2ix = self.get_category_info(technical_terms)
        print2("get category info, DONE!")

        if num_of_open_categories_of_a_word is None:
            self.num_of_open_categories_of_a_word = self.category_num
        else:
            self.num_of_open_categories_of_a_word = num_of_open_categories_of_a_word

        # complete word dict
        self.word_count = Counter()
        # part 1
        print2("get complete word dict, part 1/3")
        word_max_len_1 = self.get_complete_word_dict_part_1(
            sentences, prior_info['prior_word_info'], word_max_len_for_screening, word_min_freq_for_screening)
        print2("get complete word dict, part 1/3, DONE!")

        # word classifier
        print2("get word classifier")
        self.word_classifier = NaiveBayesClassifier(technical_terms,
                                                    max(word_max_len_1, word_max_len_for_screening_tt),
                                                    self.category_list, self.category_num, self.category2ix,
                                                    self.char_count, self.char_list, self.char_num, self.char2ix,
                                                    min_prob_in_nb_smooth,
                                                    prior_info.get('category2prior', None))
        print2("get word classifier, DONE!")

        # complete word dict part 2
        print2("get complete word dict, part 2/3")
        screened_tt_set = self.get_complete_word_dict_part_2(
            sentences,
            word_max_len_for_screening, word_min_freq_for_screening,
            word_max_len_for_screening_tt, word_min_freq_for_screening_tt,
            screen_tt_threshold)
        print2("get complete word dict, part 2/3, DONE!")

        # complete word dict part 3
        print2("get complete word dict, part 3/3")
        self.get_complete_word_dict_part_3(prior_info['prior_word_info'])
        print2("get complete word dict, part 3/3, DONE!")

        # output complete word dict
        print2("output complete word dict")

        self.complete_word_dict = self.output_complete_word_dict(
            prior_info['prior_word_info'],
            word_max_len_for_screening, word_min_freq_for_screening,
            screened_tt_set)
        print2("output complete word dict, DONE!")

        # word classification
        print2("word classification")
        self.category_word_hard_bool, self.category_word_soft_bool, self.sparse_posterior = self.word_classification(
            prior_info['prior_word_info'], prior_info['prior_word_category_setting'], screen_collo_tt_threshold)
        print2("word classification, DONE!")

        # add category_prior_in_em
        category_prior_in_em = prior_info.get('category_prior_in_em', None)
        if category_prior_in_em is not None:
            category_prior = np.array([category_prior_in_em[category]
                                       for category in self.category_list],
                                      dtype=np.float64)
            self.sparse_posterior *= category_prior[:, np.newaxis]
            temp = self.sparse_posterior.sum(axis=0)
            temp[temp == 0] = 1
            self.sparse_posterior /= temp


class CollocationDictionaryInitializer:

    def __init__(self, category_list: List[Category], category_num: int, word_list: List[Word],
                 category_ixs_of_word: List[List[int]],
                 sentences: deque,
                 prior_collocation_set: Set[Collocation],
                 collocation_max_len_for_screening: int = 3, collocation_min_freq_for_screening: int = 100,
                 is_prune_by_prior_entity: bool = False
                 ):
        collocation_count = Counter(
            collocation
            for sent in tqdm(sentences, desc='screen sentences')
            if sent.bool
            for collocation in sent.generate_collocation(collocation_max_len_for_screening,
                                                         category_num,
                                                         category_ixs_of_word,
                                                         is_prune_by_prior_entity))
        for collocation in prior_collocation_set:
            if collocation not in collocation_count:
                collocation_count[collocation] = 1

        self.collocation_list = []
        self.collocation_raw_hard_count = []

        with open('collocation_count.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = ['collocation', 'collocation_string', 'collocation_len', 'hard_count', 'is_prior']
            writer.writerow(row)
            for collocation, count in tqdm(collocation_count.most_common(), desc='screen collocations'):
                row = (collocation,
                       collocation_to_string(collocation, category_num, category_list, word_list),
                       len(collocation), count, collocation in prior_collocation_set)
                writer.writerow(row)

                if (count >= collocation_min_freq_for_screening or len(collocation) == 1
                        or collocation in prior_collocation_set):
                    self.collocation_list.append(collocation)
                    self.collocation_raw_hard_count.append(count)

        del collocation_count
