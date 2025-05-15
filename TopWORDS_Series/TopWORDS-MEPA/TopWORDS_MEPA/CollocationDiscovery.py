"""

input:
(1) text
(2) technical_terms

procedure:
- get category_list, word_len_dist, char_freq_dist from technical_terms, output as json file
- split text into sentences by punctuations
- screen text and get char_list
- get word_len_dist, char_freq_dist, compatible with char_list in text
- construct naive bayes classifier model, get complete word dictionary
    - word has (freq >= word_min_freq) and (len <= word_max_len)
    - word classified as technical term remain
- generate word_list in sentence
- get (complete) collocation info
    - generate collocation in each sentence
    - filter with high freq
    - get collocation_list,
- generate collocation_list in sentence

- em

"""

import csv
import time
from collections import deque
from typing import Tuple, Dict, Generator, Set, List

from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import chi2
from collections import defaultdict

from TopWORDS_MEPA.utils import Char, Category
from TopWORDS_MEPA.utils import get_uint_dtype
from TopWORDS_MEPA.utils import collocation_to_string, string_to_collocation
from TopWORDS_MEPA.utils import ngrams
from TopWORDS_MEPA.utils import print2, print_error
from TopWORDS_MEPA.Initializer import WordDictionaryInitializer, CollocationDictionaryInitializer
from TopWORDS_MEPA.Dictionary import Dictionary
from TopWORDS_MEPA.Sentence import Sentence, e_step_in_one_sent, compute_score_in_one_sent


class CollocationDiscovery:

    @staticmethod
    def get_sent_string_list_from_text_string(text_string: Generator or str,
                                              punctuations: Set[Char]) -> Generator[str, None, None]:
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

    @staticmethod
    def yield_char_from_text_file(text_file: str) -> Generator[str, None, None]:
        with open(text_file, encoding='utf-8') as f:
            while True:
                char = f.read(1)
                if char:
                    yield char
                else:
                    break

    @staticmethod
    def get_sent_string_and_ner_list_from_text_string(text_file, punctuations, text_ner_pos_list):
        sent_string_list = []
        sent_ner_pos_list = []
        with open(text_file, encoding='utf-8') as f:
            text_data = f.readlines()
        if text_ner_pos_list is None:
            text_ner_pos_list = [[]] * len(text_data)
        for text_string, text_ner_pos in tqdm(zip(text_data, text_ner_pos_list), desc='screen texts'):
            sent_string = []
            for ix, char in enumerate(text_string):
                if char in punctuations:
                    if sent_string:
                        sent_string_list.append(''.join(sent_string))
                        start_ix = ix - len(sent_string)
                        end_ix = ix
                        sent_ner_pos_list.append(
                            [[entity[0], entity[1], entity[2] - start_ix, entity[3] - start_ix]
                             for entity in text_ner_pos if entity[2] >= start_ix and entity[3] <= end_ix])
                        sent_string = []
                    sent_string_list.append(char)
                    sent_ner_pos_list.append([])
                else:
                    sent_string.append(char)
            if sent_string:
                sent_string_list.append(''.join(sent_string))

        return zip(sent_string_list, sent_ner_pos_list)

    def initialize_word_dictionary(self, technical_terms,
                                   prior_info,
                                   word_max_len_for_screening, word_min_freq_for_screening,
                                   word_max_len_for_screening_tt,
                                   word_min_freq_for_screening_tt,
                                   screen_tt_threshold,
                                   screen_collo_tt_threshold,
                                   num_of_open_categories_of_a_word,
                                   min_prob_in_nb_smooth):
        print2("initialize word dictionary (2/6)")
        word_dictionary_initializer = WordDictionaryInitializer(
            technical_terms, self.sentences,
            prior_info,
            word_max_len_for_screening, word_min_freq_for_screening,
            word_max_len_for_screening_tt, word_min_freq_for_screening_tt,
            screen_tt_threshold, screen_collo_tt_threshold,
            num_of_open_categories_of_a_word,
            min_prob_in_nb_smooth)

        self.dictionary.initialize_word_dictionary(
            word_dictionary_initializer.category_list,
            word_dictionary_initializer.complete_word_dict['word'].to_list(),
            word_dictionary_initializer.complete_word_dict['count'].to_list(),
            word_dictionary_initializer.category_word_hard_bool,
            word_dictionary_initializer.category_word_soft_bool,
            word_dictionary_initializer.sparse_posterior)
        del word_dictionary_initializer
        print2(f'word_num: {self.dictionary.word_bool.sum()}', add_time=False)
        print2(f'category_word_pair_num_hard: {self.dictionary.category_word_hard_bool.sum()}', add_time=False)
        print2(f'category_word_pair_num_soft: {self.dictionary.category_word_soft_bool.sum()}', add_time=False)
        print2("initialize word dictionary (2/6), DONE!")

    def set_word_list_in_sentences(self):
        print2("set word list in sentences (3/6)")
        word2ix = {word: ix for ix, word in enumerate(self.dictionary.word_list)}
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.set_word_list(self.dictionary.word_max_len, word2ix)
        print2("set word list in sentences (3/6), DONE!")

    def set_word_list_in_sentences_ner(self, category2ix):
        print2("set word list in sentences (3/6)")
        word2ix = {word: ix for ix, word in enumerate(self.dictionary.word_list)}
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.set_word_list_ner(self.dictionary.word_max_len, word2ix, category2ix)
        print2("set word list in sentences (3/6), DONE!")

    def initialize_collocation_dictionary(self,
                                          prior_collocation_info: Set[str],
                                          collocation_max_len_for_screening: int = 3,
                                          collocation_min_freq_for_screening: int = 100,
                                          is_constrain_by_prior_entity: bool = False):
        print2("initialize collocation dictionary (4/6)")
        word2ix = {word: ix for ix, word in enumerate(self.dictionary.word_list)}
        if is_constrain_by_prior_entity:
            # category2ix = {category: ix for ix, category in enumerate(self.dictionary.category_list)}
            # entity_ix_category = defaultdict(list)
            # for line in text_ner_pos_list:
            #     for entity in line:
            #         entity_ix_category[word2ix.get(entity[0], 0)].append(category2ix[entity[1]])
            # category_ixs_of_word = [
            #     list(set(entity_ix_category.get(word_ix, [])))
            #     for word_ix, word in enumerate(self.dictionary.word_list)
            # ]
            category_ixs_of_word = [
                []
                for word_ix, word in enumerate(self.dictionary.word_list)
            ]
        else:
            category_ixs_of_word = [
                np.nonzero(self.dictionary.category_word_hard_bool[:, word_ix])[0].tolist()
                for word_ix, word in enumerate(self.dictionary.word_list)
            ]
        prior_collocation_set = {string_to_collocation(string, self.dictionary.category_num,
                                                       self.dictionary.category2ix, word2ix)
                                 for string in prior_collocation_info}

        collocation_dictionary_initializer = CollocationDictionaryInitializer(
            self.dictionary.category_list, self.dictionary.category_num, self.dictionary.word_list,
            category_ixs_of_word,
            self.sentences,
            prior_collocation_set,
            collocation_max_len_for_screening, collocation_min_freq_for_screening,
            is_constrain_by_prior_entity)
        self.dictionary.initialize_collocation_dictionary(
            collocation_dictionary_initializer.collocation_list,
            collocation_dictionary_initializer.collocation_raw_hard_count)
        print2(f'collocation_num: {self.dictionary.collocation_bool.sum()}', add_time=False)
        print2("initialize collocation dictionary (4/6), DONE!")

    def set_collocation_list_in_sentences(self, mode='soft', prior_word_info=None):
        print2("set collocation list in sentences (5/6)")
        collocation2ix = {collocation: ix for ix, collocation in enumerate(self.dictionary.collocation_list)}
        if mode == 'soft':
            category_ixs_of_word = [
                np.nonzero(self.dictionary.category_word_soft_bool[:, word_ix])[0].tolist()
                for word_ix, word in enumerate(self.dictionary.word_list)
            ]
        else:  # 只允许词典中词汇
            category_ixs_of_word = [
                np.nonzero(self.dictionary.category_word_hard_bool[:, word_ix])[
                    0].tolist() if word in prior_word_info else []
                for word_ix, word in enumerate(self.dictionary.word_list)
            ]
            category_ixs_of_word_dict = {
                word: np.nonzero(self.dictionary.category_word_hard_bool[:, word_ix])[0].tolist() if sum(
                    self.dictionary.category_word_hard_bool[:, word_ix]) == 1 else []
                for word_ix, word in enumerate(self.dictionary.word_list)
            }
            print2(category_ixs_of_word_dict)
        collocation_fragment_set = {fragment
                                    for collocation in self.dictionary.collocation_list
                                    for collocation_len in range(1, len(collocation) + 1)
                                    for fragment in ngrams(collocation, collocation_len)}
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.set_collocation_list(self.dictionary.category_num,
                                          self.dictionary.collocation_max_len,
                                          collocation2ix, category_ixs_of_word, collocation_fragment_set)
        print2("set collocation list in sentences (5/6), DONE!")

    def prune_collocation_by_examples(self, technical_terms, collocation_example_thr):
        print2("prune collocation dictionary by examples (5/6)")
        word_list_technical = list(
            set([word for category, words in technical_terms.items() if category != 'background' for word in
                 words]) & set(self.dictionary.word_list))
        word2ix = {word: ix for ix, word in enumerate(self.dictionary.word_list)}
        word_ix_list_technical = [word2ix[word] for word in word_list_technical]
        collocation_example_count = np.zeros(len(self.dictionary.collocation_list))
        collocation_examples = defaultdict(list)
        for sent in tqdm(self.sentences):
            if sent.bool:
                for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in sent.collocation_list:
                    if any([word_ix in word_ix_list_technical for word_ix in word_list_in_collocation]):
                        collocation_example_count[collocation_ix] += 1
                        collocation_examples[collocation_ix].append(sent.sent_string[c_s_p:c_e_p])

        collocation_example_count_dict = {self.dictionary.collocation_list[collocation_ix]: count
                                          for collocation_ix, count in enumerate(collocation_example_count)}

        df = pd.DataFrame({'collocation': self.dictionary.collocation_list,
                           'collocation_string': [collocation_to_string(collocation, self.dictionary.category_num,
                                                                        self.dictionary.category_list,
                                                                        self.dictionary.word_list) for collocation in
                                                  self.dictionary.collocation_list],
                           'collocation_len': [len(collocation) for collocation in self.dictionary.collocation_list],
                           'hard_count': self.dictionary.collocation_raw_hard_count,
                           'example_count': collocation_example_count,
                           'examples': ['; '.join(collocation_examples[collocation_ix])
                                        for collocation_ix in range(len(self.dictionary.collocation_list))]})
        df.to_excel('collocation_example_count.xlsx', index=False)

        collocation_list = [collocation for collocation in self.dictionary.collocation_list
                            if collocation_example_count_dict[collocation] > collocation_example_thr]
        collocation_raw_hard_count = [count for collocation, count in
                                      zip(self.dictionary.collocation_list, self.dictionary.collocation_raw_hard_count)
                                      if collocation_example_count_dict[collocation] > collocation_example_thr]
        self.dictionary.initialize_collocation_dictionary(
            collocation_list,
            collocation_raw_hard_count)

        print2(f'collocation_num: {self.dictionary.collocation_bool.sum()}', add_time=False)
        print2("prune collocation dictionary by examples (5/6), DONE!")

    def initialize_parameters(self):
        print2("initialize parameters (6/6)")
        collocation_raw_soft_count = np.zeros(self.dictionary.collocation_num, dtype=np.float64)
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in sent.collocation_list:
                    collocation_raw_soft_count[collocation_ix] += np.mean(
                        [self.dictionary.sparse_posterior[category_ix, word_ix]
                         for category_ix, word_ix in
                         zip(self.dictionary.category_ixs_in_collocation[collocation_ix], word_list_in_collocation)
                         ])

        # collo_count = Counter(collocation_ix
        #                       for sent in tqdm(self.sentences, desc='screen sentences')
        #                       if sent.bool
        #                       for collocation_ix in sent.collocation_list[:, 2])
        # for ix, count in collo_count.items():
        #     collocation_raw_soft_count[ix] = count

        # collocation_raw_soft_count = self.dictionary.collocation_raw_hard_count.copy()
        collocation_raw_soft_count[-1] = self.active_sent_num
        self.dictionary.initialize_parameters(collocation_raw_soft_count)
        print2("initialize parameters (6/6), DONE!")

    def __init__(self,
                 text_file: str, technical_terms: dict, prior_info: dict,
                 punctuations: set,
                 word_max_len_for_screening: int = 3, word_min_freq_for_screening: int = 100,
                 word_max_len_for_screening_tt: int = 1, word_min_freq_for_screening_tt: int = 1,
                 screen_tt_threshold: float = 0.5, screen_collo_tt_threshold: float = 0.5,
                 collocation_example_thr: int = 0,
                 collocation_max_len_for_screening: int = 3, collocation_min_freq_for_screening: int = 100,
                 num_of_open_categories_of_a_word: int = None,
                 min_prob_in_nb_smooth: float = 0.01,
                 num_of_processes: int = 1,
                 mode: str = 'Bayes', alpha: float = 10,
                 active_sent_num: int = None,
                 text_ner_pos_list: List = None,
                 ):
        print2('-' * 100, mode='w', add_time=False)
        print2("Welcome to use Collocation Discovery program developed by Jiaze Xu (xujiaze13@126.com)")
        print2('-' * 100, add_time=False)
        print2(f'text_file: {text_file}', add_time=False)
        print2(f'word_max_len_for_screening: {word_max_len_for_screening}', add_time=False)
        print2(f'word_min_freq_for_screening: {word_min_freq_for_screening}', add_time=False)
        print2(f'word_max_len_for_screening_tt: {word_max_len_for_screening_tt}', add_time=False)
        print2(f'word_min_freq_for_screening_tt: {word_min_freq_for_screening_tt}', add_time=False)
        print2(f'screen_tt_threshold: {screen_tt_threshold}', add_time=False)
        print2(f'screen_collo_tt_threshold: {screen_collo_tt_threshold}', add_time=False)
        print2(f'collocation_example_thr: {collocation_example_thr}', add_time=False)
        print2(f'collocation_max_len_for_screening: {collocation_max_len_for_screening}', add_time=False)
        print2(f'collocation_min_freq_for_screening: {collocation_min_freq_for_screening}', add_time=False)
        print2(f'num_of_open_categories_of_a_word: {num_of_open_categories_of_a_word}', add_time=False)
        print2(f'min_prob_in_nb_smooth: {min_prob_in_nb_smooth}', add_time=False)
        print2(f'num_of_processes: {num_of_processes}', add_time=False)
        print2(f'mode: {mode}', add_time=False)
        print2(f'alpha: {alpha}', add_time=False)
        print2(f'active_sent_num: {active_sent_num}', add_time=False)
        print2('-' * 100, add_time=False)
        print2(f'prior_word_category_setting: {prior_info["prior_word_category_setting"]}', add_time=False)
        print2('-' * 100, add_time=False)

        self.num_of_processes = num_of_processes
        self.mode = mode  # 'Bayes', 'None'
        self.alpha = alpha  # pseudo count
        self.num_of_open_categories_of_a_word = num_of_open_categories_of_a_word

        print2("split text into sentences (1/6)")
        self.sentences = deque(
            Sentence(sent_string, punctuations, ner_pos_list)
            for sent_string, ner_pos_list in
            self.get_sent_string_and_ner_list_from_text_string(text_file, punctuations, text_ner_pos_list))
        # self.sentences = deque(
        #     Sentence(sent_string, punctuations)
        #     for sent_string in self.get_sent_string_list_from_text_string(
        #         self.yield_char_from_text_file(text_file), punctuations))
        # self.sent_ner_pos_list = self.get_sent_ner_pos_list(ner_pos_list, text_file, punctuations)
        self.bool_true_sent_num = sum(sent.bool for sent in self.sentences)
        if active_sent_num is None:
            self.active_sent_num = self.bool_true_sent_num
        else:
            self.active_sent_num = active_sent_num

        print2('text_len: {}'.format(sum(sent.len for sent in self.sentences)), add_time=False)
        print2(f'sentence_num: {len(self.sentences)}', add_time=False)
        print2(f'bool_true_sent_num: {self.bool_true_sent_num}', add_time=False)
        print2('text_len (in_bool_true_sent): {}'.format(sum(sent.len for sent in self.sentences if sent.bool)),
               add_time=False)
        print2("split text into sentences (1/6), DONE!")

        self.dictionary = Dictionary()
        self.initialize_word_dictionary(technical_terms,
                                        prior_info,
                                        word_max_len_for_screening, word_min_freq_for_screening,
                                        word_max_len_for_screening_tt, word_min_freq_for_screening_tt,
                                        screen_tt_threshold,
                                        screen_collo_tt_threshold,
                                        num_of_open_categories_of_a_word,
                                        min_prob_in_nb_smooth)
        # import sys
        # sys.exit()
        if text_ner_pos_list is not None:
            self.set_word_list_in_sentences_ner(self.dictionary.category2ix)
            self.initialize_collocation_dictionary(prior_info['prior_collocation_info'],
                                                   collocation_max_len_for_screening,
                                                   collocation_min_freq_for_screening,
                                                   is_constrain_by_prior_entity=True)
            self.set_word_list_in_sentences()
        else:
            self.set_word_list_in_sentences()
            self.initialize_collocation_dictionary(prior_info['prior_collocation_info'],
                                                   collocation_max_len_for_screening,
                                                   collocation_min_freq_for_screening)


        if collocation_example_thr == 0:
            self.set_collocation_list_in_sentences()

        else:
            self.set_collocation_list_in_sentences('hard', prior_info['prior_word_info'])
            self.prune_collocation_by_examples(technical_terms, collocation_example_thr)
            self.set_collocation_list_in_sentences('soft')

        self.initialize_parameters()
        print2('-' * 100, add_time=False)

    def _e_step_without_multiprocessing(self):
        start = time.time()
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.update_para(self.dictionary)
                if isinstance(sent.likelihood, float):
                    self.dictionary.log_likelihood += np.log(sent.likelihood)
                else:
                    self.dictionary.log_likelihood += sent.likelihood.ln().__float__()
        self.dictionary.log_likelihood += \
            (self.active_sent_num - self.bool_true_sent_num) * np.log(self.dictionary.rho_value[-1])
        self.dictionary.rho_new[-1] += self.active_sent_num
        e_time = time.time() - start
        return e_time

    # @memory_profiler.profile
    def _e_step_with_multiprocessing(self, num_of_processes: int):
        start = time.time()
        e_step_in_one_sent_partial = partial(
            e_step_in_one_sent,
            rho_value_=self.dictionary.rho_value,
            theta_value_=self.dictionary.theta_value,
            category_ixs_in_collocation_=self.dictionary.category_ixs_in_collocation)

        with Pool(num_of_processes) as p:
            for out in p.map(e_step_in_one_sent_partial,
                             (sent.collocation_list for sent in self.sentences if sent.bool)):

                likelihood, rho_new, theta_new = out
                self.dictionary.log_likelihood += likelihood.ln().__float__()
                for collocation_ix, value in rho_new.items():
                    self.dictionary.rho_new[collocation_ix] += value
                for category_word_ix, value in theta_new.items():
                    self.dictionary.theta_new[category_word_ix] += value
        self.dictionary.log_likelihood += \
            (self.active_sent_num - self.bool_true_sent_num) * np.log(self.dictionary.rho_value[-1])
        self.dictionary.rho_new[-1] += self.active_sent_num
        e_time = time.time() - start
        return e_time

    def _e_step(self) -> float:
        self.dictionary.rho_new = np.zeros(self.dictionary.collocation_num, dtype=np.float64)
        self.dictionary.theta_new = np.zeros((self.dictionary.category_num, self.dictionary.word_num), dtype=np.float64)
        self.dictionary.log_likelihood = 0.0

        if self.num_of_processes == 1:
            e_time = self._e_step_without_multiprocessing()
        elif self.num_of_processes > 1:
            e_time = self._e_step_with_multiprocessing(self.num_of_processes)
        else:
            raise ValueError

        return e_time

    def _prune_by_count(self, prune_by_count_threshold_collocation: float, prune_by_count_threshold_word: float):
        is_prune_by_count = False
        if prune_by_count_threshold_collocation > 0.0 or prune_by_count_threshold_word > 0.0:
            active_collocation_num = self.dictionary.collocation_bool.sum()
            mask_1 = (self.dictionary.rho_new <= prune_by_count_threshold_collocation)
            mask_2 = np.array([(len(collocation) > 1) for collocation in self.dictionary.collocation_list])
            mask = mask_1 & mask_2
            self.dictionary.collocation_bool[mask] = False
            is_prune_collocation = True if self.dictionary.collocation_bool.sum() < active_collocation_num else False

            active_word_num = self.dictionary.word_bool.sum()
            mask_1 = (self.dictionary.theta_new.sum(axis=0) <= prune_by_count_threshold_word)
            mask_2 = np.array([(len(word) > 1) for word in self.dictionary.word_list])
            mask = mask_1 & mask_2
            self.dictionary.word_bool[mask] = False
            self.dictionary.category_word_soft_bool[:, mask] = False
            is_prune_word = True if self.dictionary.word_bool.sum() < active_word_num else False

            is_prune_by_count |= is_prune_collocation or is_prune_word
        return is_prune_by_count

    def _add_pseudo_count(self):
        # add pseudo count
        mask = self.dictionary.theta_new.astype(np.bool_)
        theta_new_temp = self.dictionary.sparse_posterior * mask * self.alpha

        self.dictionary.theta_new += theta_new_temp
        theta_c_count = theta_new_temp.sum(axis=1)
        self.dictionary.rho_new[1:(self.dictionary.category_num + 1)] += theta_c_count
        self.dictionary.rho_new[-1] += theta_c_count.sum()

        theta_w = self.dictionary.theta_new.sum(axis=0)
        theta_w_sum = theta_w.sum()
        if theta_w_sum == 0:
            print_error('sum of theta_w is 0')
            theta_w_sum = 1
        self.dictionary.theta_w = theta_w / theta_w_sum

        theta_w[theta_w == 0] = 1
        self.dictionary.theta_c_g_w = self.dictionary.theta_new / theta_w

    def _m_step(self) -> Tuple[float, float]:

        self.dictionary.rho_new /= self.dictionary.rho_new.sum()
        for row in self.dictionary.theta_new:
            row /= row.sum()
        dis_rho = (self.dictionary.rho_value - self.dictionary.rho_new).__abs__().max()
        dis_theta = (self.dictionary.theta_value - self.dictionary.theta_new).__abs__().max()

        self.dictionary.rho_value[:] = self.dictionary.rho_new
        self.dictionary.theta_value[:] = self.dictionary.theta_new

        return dis_rho, dis_theta

    def _prune_by_para(self, prune_by_para_threshold_collocation: float, prune_by_para_threshold_word: float):
        # prune by parameters
        is_prune_by_para = False
        if prune_by_para_threshold_collocation > 0.0 or prune_by_para_threshold_word > 0.0:
            is_prune_by_para = self.dictionary.prune_by_para(prune_by_para_threshold_collocation,
                                                             prune_by_para_threshold_word)
            # self.dictionary.rho_value[~self.dictionary.collocation_bool] = 0.0
            # self.dictionary.theta_value[~self.dictionary.category_word_soft_bool] = 0.0
        return is_prune_by_para

    def _prune_sent(self) -> float:
        start = time.time()
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.prune(self.dictionary)

        prune_time = time.time() - start
        return prune_time

    def _em_one_step(self, em_iteration_ix: int,
                     prune_by_count_threshold_collocation: float, prune_by_count_threshold_word: float,
                     prune_by_para_threshold_collocation: float, prune_by_para_threshold_word: float):

        e_time = self._e_step()
        is_prune_by_count = self._prune_by_count(prune_by_count_threshold_collocation, prune_by_count_threshold_word)
        self._add_pseudo_count()
        dis_rho, dis_theta = self._m_step()
        is_prune_by_para = self._prune_by_para(prune_by_para_threshold_collocation, prune_by_para_threshold_word)
        is_prune = is_prune_by_count or is_prune_by_para
        if max(prune_by_count_threshold_collocation, prune_by_count_threshold_word,
               prune_by_para_threshold_collocation, prune_by_para_threshold_word) > 0:
            prune_time = self._prune_sent()
        else:
            prune_time = 0

        # print
        print2('\t'.join([
            '%3d' % em_iteration_ix, '%8.2e' % e_time, '%8.2e' % prune_time,
            '%14.7e' % self.dictionary.log_likelihood,
            '%8.2e' % dis_rho, '%8.2e' % dis_theta,
            '%7d' % self.dictionary.collocation_bool.sum(),
            '%6d' % self.dictionary.word_bool.sum(),
            '%16d' % self.dictionary.category_word_soft_bool.sum()])
        )
        return is_prune, dis_rho, dis_theta

    # @memory_profiler.profile
    def _compute_score(self):
        self.dictionary.collocation_score = np.zeros(self.dictionary.collocation_num, dtype=np.float64)
        self.dictionary.word_score = np.zeros(self.dictionary.word_num, dtype=np.float64)
        self.dictionary.category_word_score = np.zeros((self.dictionary.category_num, self.dictionary.word_num),
                                                       dtype=np.float64)

        if self.num_of_processes == 1:
            # with np.errstate(divide='ignore', over='ignore'):
            for sent in tqdm(self.sentences, desc='screen sentences'):
                if sent.bool:
                    sent.compute_score(self.dictionary)
        elif self.num_of_processes > 1:
            compute_score_in_one_sent_partial = partial(
                compute_score_in_one_sent,
                rho_value_=self.dictionary.rho_value,
                theta_value_=self.dictionary.theta_value,
                category_ixs_in_collocation_=self.dictionary.category_ixs_in_collocation)
            with Pool(self.num_of_processes) as p:
                for collocation_score, word_score, category_word_score in p.map(
                        compute_score_in_one_sent_partial,
                        (sent.collocation_list for sent in self.sentences if sent.bool)):
                    for collocation_ix, value in collocation_score.items():
                        self.dictionary.collocation_score[collocation_ix] += value
                    for word_ix, value in word_score.items():
                        self.dictionary.word_score[word_ix] += value
                    for category_word_ix, value in category_word_score.items():
                        self.dictionary.category_word_score[category_word_ix] += value
        else:
            raise ValueError

    def _compute_prune_by_score_threshold(self, prune_by_score_significance_level: float):
        non_trivial_collocation_num = sum(1
                                          for collocation_ix, collocation in enumerate(self.dictionary.collocation_list)
                                          if self.dictionary.collocation_bool[collocation_ix] and len(collocation) > 1)
        non_trivial_word_num = sum(1
                                   for word_ix, word in enumerate(self.dictionary.word_list)
                                   if self.dictionary.word_bool[word_ix] and len(word) > 1)
        print2(f'non_trivial_collocation_num: {non_trivial_collocation_num}', add_time=False)
        print2(f'non_trivial_word_num: {non_trivial_word_num}', add_time=False)
        statistical_hypothesis_testing_num = non_trivial_collocation_num + non_trivial_word_num
        print2(f'statistical_hypothesis_testing_num: {statistical_hypothesis_testing_num}', add_time=False)

        q = 1 - prune_by_score_significance_level / statistical_hypothesis_testing_num
        prune_by_score_threshold_collocation = chi2.ppf(q=q, df=1) / 2

        df = self.dictionary.category_num
        if self.num_of_open_categories_of_a_word is not None:
            df = min(df, self.num_of_open_categories_of_a_word)
        prune_by_score_threshold_word = chi2.ppf(q=q, df=df) / 2

        print2(f'prune_by_score_threshold_collocation: {prune_by_score_threshold_collocation}', add_time=False)
        print2(f'prune_by_score_df_word: {df}', add_time=False)
        print2(f'prune_by_score_threshold_word: {prune_by_score_threshold_word}', add_time=False)

        return prune_by_score_threshold_collocation, prune_by_score_threshold_word

    def _prune_by_score(self, prune_by_word_score_significance_level: float,
                        prune_by_collocation_score_significance_level: float):
        is_prune_by_score = False
        prune_by_score_threshold_collocation, _ = \
            self._compute_prune_by_score_threshold(prune_by_collocation_score_significance_level)
        _, prune_by_score_threshold_word = \
            self._compute_prune_by_score_threshold(prune_by_word_score_significance_level)
        if prune_by_score_threshold_collocation > 0.0 or prune_by_score_threshold_word > 0.0:
            is_prune_by_score = self.dictionary.prune_by_score(prune_by_score_threshold_collocation,
                                                               prune_by_score_threshold_word)
            # self.dictionary.rho_value[~self.dictionary.collocation_bool] = 0.0
            # self.dictionary.theta_value[~self.dictionary.category_word_soft_bool] = 0.0
        return is_prune_by_score

    def _compute_approx_score(self):
        self.dictionary.collocation_approx_score = np.zeros(self.dictionary.collocation_num, dtype=np.float64)
        self.dictionary.category_word_approx_score = np.zeros((self.dictionary.category_num,
                                                               self.dictionary.word_num), dtype=np.float64)
        # with np.errstate(divide='ignore', over='ignore'):
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.compute_approx_score(self.dictionary)

    def em_update(self,
                  em_iteration_num: int,
                  prune_by_count_threshold_collocation: float = 0.1, prune_by_count_threshold_word: float = 0.1,
                  prune_by_para_threshold_collocation: float = 0.0, prune_by_para_threshold_word: float = 0.0,
                  prune_by_score_iteration_num: int = 0,
                  em_iteration_num_in_prune_by_score: int = 0,
                  prune_by_word_score_significance_level: float = 0.05,
                  prune_by_collocation_score_significance_level: float = 0.05,
                  is_first_time: bool = True):
        if is_first_time:
            print2('-' * 100, add_time=False)
            print2(f'em_iteration_num: {em_iteration_num}', add_time=False)
            print2(f'prune_by_count_threshold_collocation: {prune_by_count_threshold_collocation}', add_time=False)
            print2(f'prune_by_count_threshold_word: {prune_by_count_threshold_word}', add_time=False)
            print2(f'prune_by_para_threshold_collocation: {prune_by_para_threshold_collocation}', add_time=False)
            print2(f'prune_by_para_threshold_word: {prune_by_para_threshold_word}', add_time=False)
            print2(f'prune_by_score_iteration_num: {prune_by_score_iteration_num}', add_time=False)
            print2(f'em_iteration_num_in_prune_by_score: {em_iteration_num_in_prune_by_score}', add_time=False)
            print2(f'prune_by_word_score_significance_level: {prune_by_word_score_significance_level}', add_time=False)
            print2(f'prune_by_collocation_score_significance_level: {prune_by_collocation_score_significance_level}', add_time=False)
            print2('-' * 100, add_time=False)
            print2(f'active_collocation_num: {self.dictionary.collocation_bool.sum()}', add_time=False)
            print2(f'active_word_num: {self.dictionary.word_bool.sum()}', add_time=False)
            print2(f'active_category_word_pair_num: {self.dictionary.category_word_soft_bool.sum()}', add_time=False)
            print2('-' * 100, add_time=False)
        print2('\t'.join(['%-10s' % 'date', '%-8s' % 'time',
                          '%-3s' % 'ix', '%-8s' % 'e_time', '%-8s' % 'p_time',
                          '%-14s' % 'log_likelihood',
                          '%-8s' % 'd_rho', '%-8s' % 'dis_theta',
                          '%-7s' % 'collo_n', '%-6s' % 'word_n', '%-16s' % 'cate_word_pair_n']), add_time=False)
        print2('\t'.join(['-' * l for l in [10, 8, 3, 8, 8, 14, 8, 8, 7, 6, 16]]), add_time=False)

        for em_iteration_ix in range(1, em_iteration_num + 1):
            is_prune, dis_rho, dis_theta = self._em_one_step(
                em_iteration_ix,
                prune_by_count_threshold_collocation, prune_by_count_threshold_word,
                prune_by_para_threshold_collocation, prune_by_para_threshold_word)
            if not is_prune and \
                    dis_rho <= 10 * prune_by_para_threshold_collocation and \
                    dis_theta <= 10 * prune_by_para_threshold_word:
                break

        print2('-' * 100, add_time=False)
        # prune by score
        for prune_by_score_iteration_ix in range(1, prune_by_score_iteration_num + 1):
            print2("compute significance score ({}/{})".format(prune_by_score_iteration_ix,
                                                               prune_by_score_iteration_num))
            self._compute_score()
            print2("compute significance score ({}/{}), DONE!".format(prune_by_score_iteration_ix,
                                                                      prune_by_score_iteration_num))
            print2('-' * 100, add_time=False)
            is_prune = self._prune_by_score(prune_by_word_score_significance_level,
                                            prune_by_collocation_score_significance_level)
            print2(f'active_collocation_num: {self.dictionary.collocation_bool.sum()}', add_time=False)
            print2(f'active_word_num: {self.dictionary.word_bool.sum()}', add_time=False)
            print2(f'active_category_word_pair_num: {self.dictionary.category_word_soft_bool.sum()}', add_time=False)
            prune_time = self._prune_sent()
            print2(f'prune_time: {prune_time}', add_time=False)
            print2('-' * 100, add_time=False)
            if not is_prune:
                break
            self.em_update(em_iteration_num=em_iteration_num_in_prune_by_score,
                           prune_by_count_threshold_collocation=prune_by_count_threshold_collocation,
                           prune_by_count_threshold_word=prune_by_count_threshold_word,
                           prune_by_para_threshold_collocation=prune_by_para_threshold_collocation,
                           prune_by_para_threshold_word=prune_by_para_threshold_word,
                           prune_by_word_score_significance_level=prune_by_word_score_significance_level,
                           prune_by_collocation_score_significance_level=prune_by_collocation_score_significance_level,
                           is_first_time=False)

    def output_decoded_result(self, mode='mle+pc'):
        print2('-' * 100, add_time=False)
        print2("output decode result")

        self.dictionary.collocation_seg_count = np.zeros(self.dictionary.collocation_num, dtype=np.uint64)
        self.dictionary.word_seg_count = np.zeros((self.dictionary.category_num,
                                                   self.dictionary.word_num), dtype=np.uint64)
        with np.errstate(divide='ignore'):
            with open('segmented_text.txt', 'w', encoding='utf-8') as f:
                for sent in tqdm(self.sentences, desc='screen sentences'):
                    if sent.bool:
                        _ = f.write(sent.decode_to_string(self.dictionary, mode))
                    else:
                        _ = f.write(sent.sent_string)
                    _ = f.write(' ')

        print2("output decode result, DONE!")
        print2('-' * 100, add_time=False)

    def _compute_posterior(self):
        _ = self._e_step()
        self._add_pseudo_count()

    def output_dictionary_result(self,
                                 is_compute_posterior: bool = False,
                                 is_compute_score: bool = False,
                                 is_compute_approx_score: bool = False):
        print2('-' * 100, add_time=False)
        print2(f'is_compute_posterior: {is_compute_posterior}', add_time=False)
        print2(f'is_compute_score: {is_compute_score}', add_time=False)
        print2('-' * 100, add_time=False)
        # compute posterior
        if is_compute_posterior:
            print2("compute posterior")
            self._compute_posterior()
            print2("compute posterior, DONE!")
        # compute score
        if is_compute_score:
            print2("compute significance score")
            self._compute_score()
            print2("compute significance score, DONE!")
        # compute approx score
        if is_compute_approx_score:
            print2("compute approximate significance score")
            self._compute_approx_score()
            print2("compute approximate significance score, DONE!")
        print2('-' * 100, add_time=False)

        print2("output collocation dictionary")
        collocation_dictionary = pd.DataFrame({
            'collocation_ix': np.arange(self.dictionary.collocation_num,
                                        dtype=get_uint_dtype(self.dictionary.collocation_num)),
            'collocation': [str(collocation) for collocation in self.dictionary.collocation_list],
            'collocation_string': [collocation_to_string(collocation,
                                                         self.dictionary.category_num,
                                                         self.dictionary.category_list,
                                                         self.dictionary.word_list)
                                   for collocation in self.dictionary.collocation_list],
            'collocation_len': [len(collocation) for collocation in self.dictionary.collocation_list],
            'collocation_raw_hard_count': self.dictionary.collocation_raw_hard_count,
            'collocation_raw_soft_count': self.dictionary.collocation_raw_soft_count,
            'collocation_raw_modified_soft_count': self.dictionary.collocation_raw_modified_soft_count,
            'rho_value': self.dictionary.rho_value
        })
        if is_compute_posterior:
            collocation_dictionary['post_seg_soft_count'] = self.dictionary.rho_new
        if is_compute_score:
            collocation_dictionary['collocation_score'] = self.dictionary.collocation_score
        if is_compute_approx_score:
            collocation_dictionary['collocation_approx_score'] = self.dictionary.collocation_approx_score

        if self.dictionary.collocation_seg_count is not None:
            collocation_dictionary['mle_seg_soft_count'] = self.dictionary.collocation_seg_count
        collocation_dictionary = collocation_dictionary[self.dictionary.collocation_bool]

        with open('collocation_dictionary.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(collocation_dictionary.columns)
            for row in tqdm(collocation_dictionary.values, desc='screen collocations'):
                writer.writerow(row)

        print2("output collocation dictionary, DONE!")

        print2("output word dictionary")

        columns_1 = ['word_ix', 'word', 'word_len', 'word_raw_count']
        columns_2 = ['word_ix', 'word', 'word_len', 'word_raw_count']
        word_dictionary = pd.DataFrame({
            'word_ix': np.arange(self.dictionary.word_num, dtype=get_uint_dtype(self.dictionary.word_num)),
            'word': self.dictionary.word_list,
            'word_len': [len(word) for word in self.dictionary.word_list],
            'word_raw_count': self.dictionary.word_raw_count
        })
        # post_by_NB
        for category_ix, category in enumerate(self.dictionary.category_list):
            word_dictionary[f'post_by_NB_{category}'] = self.dictionary.sparse_posterior[category_ix]
            columns_1.append('post_by_NB')
            columns_2.append(category)
        # theta
        for category_ix, category in enumerate(self.dictionary.category_list):
            word_dictionary[f'theta_{category}'] = self.dictionary.theta_value[category_ix]
            columns_1.append('theta')
            columns_2.append(category)

        col_name = 'theta_w'
        while col_name in set(word_dictionary.columns):
            col_name += 'w'
        word_dictionary[col_name] = self.dictionary.theta_w
        columns_1.append('theta')
        columns_2.append('w')
        # post_seg_count and post_classify
        if is_compute_posterior:

            for category_ix, category in enumerate(self.dictionary.category_list):
                word_dictionary[f'post_seg_count_{category}'] = self.dictionary.theta_new[category_ix]
                columns_1.append('post_seg_count')
                columns_2.append(category)

            word_dictionary['post_seg_count'] = self.dictionary.theta_new.sum(axis=0)
            columns_1.append('post_seg_count')
            columns_2.append('sum')

            for category_ix, category in enumerate(self.dictionary.category_list):
                word_dictionary[f'post_classify_{category}'] = self.dictionary.theta_c_g_w[category_ix]
                columns_1.append('post_classify')
                columns_2.append(category)
        # score
        if is_compute_score:
            for category_ix, category in enumerate(self.dictionary.category_list):
                word_dictionary[f'score_{category}'] = self.dictionary.category_word_score[category_ix]
                columns_1.append('score')
                columns_2.append(category)
            word_dictionary['score'] = self.dictionary.word_score
            columns_1.append('score')
            columns_2.append('word_score')
        # approx score
        if is_compute_approx_score:
            for category_ix, category in enumerate(self.dictionary.category_list):
                word_dictionary[f'approx_score_{category}'] = self.dictionary.category_word_approx_score[category_ix]
                columns_1.append('approx_score')
                columns_2.append(category)
        # mle_seg_count
        if self.dictionary.word_seg_count is not None:
            for category_ix, category in enumerate(self.dictionary.category_list):
                word_dictionary[f'mle_seg_count_{category}'] = self.dictionary.word_seg_count[category_ix]
                columns_1.append('mle_seg_count')
                columns_2.append(category)
            word_dictionary['mle_seg_count'] = self.dictionary.word_seg_count.sum(axis=0)
            columns_1.append('mle_seg_count')
            columns_2.append('sum')
        # summary
        word_dictionary = word_dictionary[self.dictionary.word_bool]

        with open('word_dictionary.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns_1)
            writer.writerow(columns_2)
            for row in tqdm(word_dictionary.values, desc='screen words'):
                writer.writerow(row)
        print2("output word dictionary, DONE!")
        print2('-' * 100, add_time=False)


def main():
    pass


if __name__ == "__main__":
    main()
