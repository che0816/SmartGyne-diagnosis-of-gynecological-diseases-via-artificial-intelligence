import csv
import time
from collections import deque
from typing import Generator, Set, Dict
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import chi2

from utils import Char, Word, get_uint_dtype, print2

from WordDictionaryInitializer import WordDictionaryInitializer
from Dictionary import Dictionary
from Sentence import Sentence, update_para_job, compute_score_job


class TopWORDS:

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

    def initialize_word_dictionary(self, prior_word_freq: Dict[Word, int], protected_prior_words: Set[Word],
                                   delete_words: Set[Word],
                                   word_max_len_for_screening: int, word_min_freq_for_screening: int,
                                   active_sent_num: int):
        print2("initialize word dictionary (2/4)")
        prior_words = set(prior_word_freq.keys())
        prior_words = prior_words | protected_prior_words
        word_dictionary_initializer = WordDictionaryInitializer(self.sentences, prior_words, delete_words,
                                                                word_max_len_for_screening, word_min_freq_for_screening)

        self.dictionary.initialize_word_dictionary(
            word_dictionary_initializer.complete_word_dict['word'].to_list(),
            word_dictionary_initializer.complete_word_dict['count'].to_list(),
            active_sent_num,
            prior_word_freq,
            protected_prior_words)

        print2(f'word_num: {self.dictionary.word_num}', add_time=False)
        print2("initialize word dictionary (2/4), DONE!")

    def output_sent(self, text_dir):
        modify_sent_list = []
        with open(text_dir, 'w', encoding='utf8') as f:
            for sent in tqdm(self.sentences, desc='screen sentences'):
                if sent.bool:
                    modify_sent_list.append(sent.sent_string)
                    f.write(sent.sent_string + '\n')
        return modify_sent_list

    def set_sent_prior_in_sentences(self, sent_prior_list):
        print2("set prior in sentences (3/4)")
        if sent_prior_list is not None:
            assert len(sent_prior_list) == self.active_sent_num, \
                f"length of sent prior {len(sent_prior_list)} and sentences num {self.active_sent_num} are not equal"
            ix = 0
            for sent in tqdm(self.sentences, desc='screen sentences'):
                if sent.bool:
                    sent_prior = sent_prior_list[ix]
                    sent.set_sent_prior(sent_prior)
                    ix += 1
        print2("set prior in sentences (3/4), DONE!")

    def set_word_list_in_sentences(self):
        print2("set word list in sentences (4/4)")
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.set_word_list(self.dictionary.word_max_len, self.dictionary.word2ix)
        print2("set word list in sentences (4/4), DONE!")

    def _e_step_without_multiprocessing(self):
        for sent in tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                # print(sent.sent_string)
                sent.update_para(self.dictionary.theta_value, self.dictionary.theta_new)
                if isinstance(sent.likelihood, float):
                    self.dictionary.log_likelihood += np.log(sent.likelihood)
                else:
                    self.dictionary.log_likelihood += float(sent.likelihood.ln())
        for word_ix, freq in self.dictionary.word_prior_count.items():
            self.dictionary.theta_new[word_ix] += freq
        self.dictionary.log_likelihood += \
            (self.active_sent_num - self.bool_true_sent_num) * np.log(self.dictionary.theta_value[-1])
        self.dictionary.theta_new[-1] = self.active_sent_num

    def _e_step_with_multiprocessing(self, num_of_processes: int):
        update_para_partial = partial(update_para_job, theta_value=self.dictionary.theta_value)

        with Pool(num_of_processes) as p:
            for log_likelihood, theta_new in p.map(update_para_partial,
                                                   ((sent.word_list, sent.sent_prior) for sent in self.sentences if
                                                    sent.bool)):
                self.dictionary.log_likelihood += log_likelihood
                for word_ix, value in theta_new.items():
                    self.dictionary.theta_new[word_ix] += value
        for word_ix, freq in self.dictionary.word_prior_count.items():
            self.dictionary.theta_new[word_ix] += freq
        self.dictionary.log_likelihood += \
            (self.active_sent_num - self.bool_true_sent_num) * np.log(self.dictionary.theta_value[-1])
        self.dictionary.theta_new[-1] += self.active_sent_num

    def _e_step(self) -> float:
        start = time.time()
        self.dictionary.theta_new = np.zeros(self.dictionary.word_num, dtype=np.float64)
        self.dictionary.log_likelihood = 0.0

        if self.num_of_processes == 1:
            self._e_step_without_multiprocessing()
        elif self.num_of_processes > 1:
            self._e_step_with_multiprocessing(self.num_of_processes)
        else:
            raise ValueError
        e_time = time.time() - start
        return e_time

    def _m_step(self) -> float:
        self.dictionary.theta_new /= self.dictionary.theta_new.sum()
        dis_theta = (self.dictionary.theta_value - self.dictionary.theta_new).__abs__().max()
        self.dictionary.theta_value[:] = self.dictionary.theta_new
        return dis_theta

    def _prune_sent(self) -> float:
        start = time.time()
        for sent in self.sentences:  # tqdm(self.sentences, desc='screen sentences'):
            if sent.bool:
                sent.prune(self.dictionary.word_bool)
        prune_time = time.time() - start
        return prune_time

    def _em_one_step(self, em_iteration_ix: int,
                     prune_by_count_threshold: float, prune_by_para_threshold: float):
        e_time = self._e_step()
        is_prune_by_count = self.dictionary.prune_by_count(prune_by_count_threshold)
        dis_theta = self._m_step()
        is_prune_by_para = self.dictionary.prune_by_para(prune_by_para_threshold)
        is_prune = is_prune_by_count or is_prune_by_para
        prune_time = self._prune_sent() if is_prune else 0.0

        print2('\t'.join([
            '%3d' % em_iteration_ix, '%8.2e' % e_time, '%8.2e' % prune_time,
            '%52.45e' % self.dictionary.log_likelihood,
            '%9.2e' % dis_theta,
            '%7d' % self.dictionary.word_bool.sum()])
        )
        return is_prune, dis_theta

    def _compute_score(self):
        start = time.time()
        self.dictionary.word_score = np.zeros(self.dictionary.word_num, dtype=np.float64)
        if self.num_of_processes == 1:
            # with np.errstate(divide='ignore', over='ignore'):
            for sent in tqdm(self.sentences, desc='screen sentences'):
                if sent.bool:
                    sent.compute_score(self.dictionary.theta_value, self.dictionary.word_score)
        elif self.num_of_processes > 1:
            compute_score_partial = partial(compute_score_job, theta_value=self.dictionary.theta_value)

            with Pool(self.num_of_processes) as p:
                for word_score in p.map(compute_score_partial,
                                        ((sent.word_list, sent.sent_prior) for sent in self.sentences if sent.bool)):
                    for word_ix, value in word_score.items():
                        self.dictionary.word_score[word_ix] += value
        else:
            raise ValueError
        s_time = time.time() - start
        return s_time

    def _compute_prune_by_score_threshold(self, prune_by_score_significance_level: float):
        statistical_hypothesis_testing_num = non_trivial_word_num = \
            ((self.dictionary.word_len > 1) & self.dictionary.word_bool).sum()

        print2(f'statistical_hypothesis_testing_num = non_trivial_word_num: {non_trivial_word_num}', add_time=False)
        q = 1 - prune_by_score_significance_level / statistical_hypothesis_testing_num
        prune_by_score_threshold = chi2.ppf(q=q, df=1) / 2
        print2(f'prune_by_score_threshold: {prune_by_score_threshold}', add_time=False)

        return prune_by_score_threshold

    def _score_one_step(self, prune_by_score_iteration_ix, prune_by_score_iteration_num,
                        prune_by_score_significance_level):

        s_time = self._compute_score()
        prune_by_score_threshold = self._compute_prune_by_score_threshold(prune_by_score_significance_level)
        is_prune = self.dictionary.prune_by_score(prune_by_score_threshold)
        prune_time = self._prune_sent() if is_prune else 0.0
        print2('-' * 100, add_time=False)
        print2('%-10s' % 'date', '%-8s' % 'time',
               '%-3s' % 'ix', '%-3s' % 'ixs',
               '%-8s' % 's_time', '%-8s' % 'p_time',
               '%-7s' % 'word_n', add_time=False)
        print2(*['-' * l for l in [10, 8, 3, 3, 8, 8, 7]], add_time=False)
        print2('%3d' % prune_by_score_iteration_ix, '%3d' % prune_by_score_iteration_num,
               '%8.2e' % s_time, '%8.2e' % prune_time,
               '%7d' % self.dictionary.word_bool.sum())

        return is_prune

    def em_update(self,
                  em_iteration_num: int = 100,
                  prune_by_count_threshold: float = 0.0,
                  prune_by_para_threshold: float = 1e-8,
                  prune_by_score_iteration_num: int = 0,
                  em_iteration_num_in_prune_by_score: int = 100,
                  em_para_threshold = 1e-6,
                  prune_by_score_significance_level: float = 0.05,
                  is_first_time: bool = True):
        if is_first_time:
            print2('-' * 100, add_time=False)
            print2(f'em_iteration_num: {em_iteration_num}', add_time=False)
            print2(f'prune_by_count_threshold: {prune_by_count_threshold}', add_time=False)
            print2(f'prune_by_para_threshold: {prune_by_para_threshold}', add_time=False)
            print2(f'prune_by_score_iteration_num: {prune_by_score_iteration_num}', add_time=False)
            print2(f'em_iteration_num_in_prune_by_score: {em_iteration_num_in_prune_by_score}', add_time=False)
            print2(f'prune_by_score_significance_level: {prune_by_score_significance_level}', add_time=False)
            print2('-' * 100, add_time=False)
            print2(f'active_word_num: {self.dictionary.word_bool.sum()}', add_time=False)
            print2('-' * 100, add_time=False)
        print2('\t'.join(['%-10s' % 'date', '%-8s' % 'time',
                          '%-3s' % 'ix', '%-8s' % 'e_time', '%-8s' % 'p_time',
                          '%-52s' % 'log_likelihood',
                          '%-9s' % 'dis_theta',
                          '%-7s' % 'word_n']), add_time=False)
        print2('\t'.join(['-' * l for l in [10, 8, 3, 8, 8, 52, 9, 7]]), add_time=False)
        for em_iteration_ix in range(1, em_iteration_num + 1):
            is_prune, dis_theta = self._em_one_step(em_iteration_ix, prune_by_count_threshold, prune_by_para_threshold)
            if not is_prune and dis_theta <= em_para_threshold:
                break
        print2('-' * 100, add_time=False)

        # prune by score
        for prune_by_score_iteration_ix in range(1, prune_by_score_iteration_num + 1):
            print(prune_by_score_iteration_ix)
            is_prune = self._score_one_step(prune_by_score_iteration_ix, prune_by_score_iteration_num,
                                            prune_by_score_significance_level)
            print2('-' * 100, add_time=False)
            if not is_prune:
                break
            self.em_update(em_iteration_num=em_iteration_num_in_prune_by_score,
                           prune_by_count_threshold=prune_by_count_threshold,
                           prune_by_para_threshold=prune_by_para_threshold,
                           is_first_time=False)

    def output_decoded_result(self, seg_thr=0.5, seg_mode='post_mean'):
        print2('-' * 100, add_time=False)
        seg_list = []
        # post_mean_list = []
        with np.errstate(divide='ignore'):
            if seg_mode == 'post_mean':
                print2("output posterior mean decode result")
                self.dictionary.word_post_mean_seg_count = np.zeros(self.dictionary.word_num, dtype=np.uint64)
                with open(f'segmented_text.txt', 'w', encoding='utf-8') as f:
                    for sent in tqdm(self.sentences, desc='screen sentences'):
                        if sent.bool:
                            out_word_01_list, out_word_list, post_mean = sent.decode_by_post_mean(
                                self.dictionary.theta_value,
                                self.dictionary.word_post_mean_seg_count,
                                self.dictionary.word2ix,
                                threshold = seg_thr)
                            print(' '.join(out_word_list),
                                  end='', file=f)
                            seg_list.append(out_word_list)
                            # post_mean_list.append(post_mean)
                        else:
                            print(sent.sent_string, end='', file=f)
                print2("output posterior mean decode result, DONE!")
            elif seg_mode == 'post_mode':
                print2("output posterior mode decode result")
                self.dictionary.word_post_mode_seg_count = np.zeros(self.dictionary.word_num, dtype=np.uint64)
                with open('segmented_text.txt', 'w', encoding='utf-8') as f:
                    for sent in tqdm(self.sentences, desc='screen sentences'):
                        if sent.bool:
                            out_word_01_list, out_word_list, _ = sent.decode_by_post_mode(self.dictionary.theta_value,
                                                                                          self.dictionary.word_post_mode_seg_count)
                            print(' '.join(out_word_list),
                                  end='', file=f)
                            seg_list.append(out_word_list)
                        else:
                            print(sent.sent_string, end='', file=f)
                print2("output posterior mode decode result, DONE!")
        print2('-' * 100, add_time=False)
        return seg_list

    def _compute_posterior(self):
        _ = self._e_step()
        self.dictionary.word_post_count = self.dictionary.theta_new.copy()

    def output_dictionary_result(self,
                                 is_compute_posterior: bool = False,
                                 is_compute_score: bool = False):
        print2('-' * 100, add_time=False)
        print2("output word dictionary")
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

        word_dictionary = pd.DataFrame({
            'word_ix': np.arange(self.dictionary.word_num, dtype=get_uint_dtype(self.dictionary.word_num)),
            'word': self.dictionary.word_list,
            'word_len': self.dictionary.word_len,
            'word_raw_count': self.dictionary.word_raw_count,
            'theta_value': self.dictionary.theta_value,
        })
        if is_compute_score:
            word_dictionary['word_score'] = self.dictionary.word_score
        if is_compute_posterior:
            word_dictionary['post_seg_count'] = self.dictionary.word_post_count
        if self.dictionary.word_post_mean_seg_count is not None:
            word_dictionary['post_mean_seg_count'] = self.dictionary.word_post_mean_seg_count
        if self.dictionary.word_post_mode_seg_count is not None:
            word_dictionary['post_mode_seg_count'] = self.dictionary.word_post_mode_seg_count
        word_dictionary = word_dictionary[self.dictionary.word_bool]

        with open('word_dictionary.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(word_dictionary.columns)
            for row in tqdm(word_dictionary.values, desc='screen words'):
                writer.writerow(row)

        print2("output word dictionary, DONE!")
        print2('-' * 100, add_time=False)

    def __init__(self,
                 text_file: str,
                 prior_word_freq: Dict[Word, int], protected_prior_words: Set[Word],
                 delete_words: Set[Word],
                 punctuations: set,
                 sent_prior_list: np.ndarray = None,
                 word_max_len_for_screening: int = 3, word_min_freq_for_screening: int = 100,
                 num_of_processes: int = 1):
        print2('-' * 100, mode='w', add_time=False)
        print2("Welcome to use TopWORDS_Seg program developed by Changzai Pan(panpanaqm@126.com), "
               "which modified from TopWORDS program developed by Jiaze Xu(xujiaze13@126.com)")
        print2('-' * 100, add_time=False)
        print2(f'text_file: {text_file}', add_time=False)
        print2(f'word_max_len_for_screening: {word_max_len_for_screening}', add_time=False)
        print2(f'word_min_freq_for_screening: {word_min_freq_for_screening}', add_time=False)
        print2(f'num_of_processes: {num_of_processes}', add_time=False)
        print2('-' * 100, add_time=False)

        self.num_of_processes = num_of_processes

        print2("split text into sentences (1/4)")
        self.sentences = deque(
            Sentence(sent_string, punctuations)
            for sent_string in self.get_sent_string_list_from_text_string(
                self.yield_char_from_text_file(text_file), punctuations)
        )
        self.active_sent_num = sum(sent.bool for sent in self.sentences)
        self.bool_true_sent_num = sum(sent.bool for sent in self.sentences)

        print2(f'sentence_num: {len(self.sentences)}', add_time=False)
        print2('text_len: {}'.format(sum(sent.len for sent in self.sentences)), add_time=False)
        print2(f'bool_true_sent_num: {self.bool_true_sent_num}', add_time=False)
        print2('text_len (in_bool_true_sent): {}'.format(sum(sent.len
                                                             for sent in self.sentences
                                                             if sent.bool)), add_time=False)
        print2("split text into sentences (1/4), DONE!")

        self.dictionary = Dictionary()

        self.initialize_word_dictionary(prior_word_freq, protected_prior_words, delete_words,
                                        word_max_len_for_screening, word_min_freq_for_screening,
                                        self.active_sent_num)

        # self.output_sent('modify_text.txt')
        self.set_sent_prior_in_sentences(sent_prior_list)

        self.set_word_list_in_sentences()
        print2('-' * 100, add_time=False)
